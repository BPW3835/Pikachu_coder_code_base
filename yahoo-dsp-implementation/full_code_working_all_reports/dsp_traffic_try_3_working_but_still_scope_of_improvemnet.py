"""
dsp_traffic.py
──────────────
Yahoo DSP Traffic API module — production-ready (resilient).

Key fixes added:
  1) Tenacity-based retry for transient failures (429/5xx/timeouts)
  2) Per-endpoint + per-parent retry queue (re-run only failed parent IDs)
  3) Sleep between paginated page pulls to reduce rate-limit spikes
  4) Do not break whole pipeline because of one endpoint’s instability
"""

import io
import csv
import os
import time
import random
import traceback
from dataclasses import dataclass
from typing import Optional, Any

import pendulum
import polars as pl
import requests
from loguru import logger
from tqdm import tqdm

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)

from helpers import YahooHelpers, sanitize_bq_column_name, flatten_json
from metrics import TRAFFIC_API_ENDPOINT_CONFIGS

API_TYPE = "traffic_api"
TZ = "America/New_York"


# ─────────────────────────────────────────────────────────────
# Exceptions to drive tenacity retry
# ─────────────────────────────────────────────────────────────

@dataclass
class RetryableHttpError(Exception):
    status_code: int
    url: str
    message: str
    retry_after_seconds: Optional[int] = None


@dataclass
class NonRetryableHttpError(Exception):
    status_code: int
    url: str
    message: str


def _is_retryable_exception(exc: BaseException) -> bool:
    if isinstance(exc, RetryableHttpError):
        return True
    if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
        return True
    return False


# ─────────────────────────────────────────────────────────────
# Endpoint Config Base + Dynamic Class Factory
# ─────────────────────────────────────────────────────────────

class TrafficEndpointConfig:
    ENDPOINT_NAME: str = "base"
    ENDPOINT_PATH: str = "/traffic/base"
    SCOPE: str = "seat"          # seat | advertiser | campaign | line
    PARAM_NAME: str = ""         # accountId | orderId | lineId
    ID_IN_PATH: bool = False     # if True, parent id is placed into {id} in path
    SUPPORTS_DATE_FILTER: bool = False  # if you keep startTime/endTime support


def _build_endpoint_class(name: str, config: dict) -> type:
    class_name = "".join(w.capitalize() for w in name.split("_")) + "Endpoint"
    attrs = {
        "ENDPOINT_NAME": name,
        "ENDPOINT_PATH": config["path"],
        "SCOPE": config.get("scope", "seat"),
        "PARAM_NAME": config.get("param_name", ""),
        "ID_IN_PATH": config.get("id_in_path", False),
        "SUPPORTS_DATE_FILTER": config.get("supports_date_filter", False),
    }
    return type(class_name, (TrafficEndpointConfig,), attrs)


ENDPOINT_REGISTRY: dict[str, type[TrafficEndpointConfig]] = {}
for _name, _config in TRAFFIC_API_ENDPOINT_CONFIGS.items():
    ENDPOINT_REGISTRY[_name] = _build_endpoint_class(_name, _config)

ALL_TRAFFIC_ENDPOINTS: list[type[TrafficEndpointConfig]] = list(ENDPOINT_REGISTRY.values())


# ─────────────────────────────────────────────────────────────
# YahooTraffic
# ─────────────────────────────────────────────────────────────

class YahooTraffic:
    """
    Yahoo DSP Traffic API extractor with resilient retry semantics.

    Rate limiting:
      - Soft limiter at RATE_LIMIT_PER_MIN calls/min
      - Extra sleep between pages to reduce 429

    Retry:
      - Tenacity retry for retryable errors
      - Per-parent retry queue at endpoint level
    """

    BASE_URL = "https://dspapi.admanagerplus.yahoo.com"

    RATE_LIMIT_PER_MIN = 55
    SLEEP_BETWEEN_CALLS = 1.2
    SLEEP_BETWEEN_PAGES = 0.8  # NEW: reduce 429 bursts during pagination
    PAGE_LIMIT = 100

    # How many times to retry a single request (tenacity)
    REQ_RETRY_ATTEMPTS = 6

    # How many rounds to re-run failed parentIds per endpoint
    ENDPOINT_RETRY_ROUNDS = 2

    def __init__(self):
        self.helper = YahooHelpers()
        self._call_timestamps: list[float] = []
        self._session = requests.Session()
        logger.success("YahooTraffic initialised.")

    # ─────────────────────────────────────────────
    # Auth + Rate limit
    # ─────────────────────────────────────────────

    def _auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    def _rate_limit_wait(self):
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        if len(self._call_timestamps) >= self.RATE_LIMIT_PER_MIN:
            oldest = self._call_timestamps[0]
            wait = 60 - (now - oldest) + 1
            if wait > 0:
                logger.info(f"⏱️  Rate limit: waiting {wait:.0f}s ...")
                time.sleep(wait)
        self._call_timestamps.append(time.time())

    # ─────────────────────────────────────────────
    # Tenacity-wrapped request
    # ─────────────────────────────────────────────

    # @retry(
    #     retry=retry_if_exception(_is_retryable_exception),
    #     stop=stop_after_attempt(REQ_RETRY_ATTEMPTS),
    #     wait=wait_random_exponential(multiplier=2, min=2, max=90),
    #     before_sleep=before_sleep_log(logger, "WARNING"),
    #     reraise=True,
    # )
    # def _request_json(self, url: str, params: dict | None = None) -> dict:
    #     """
    #     Request wrapper that raises:
    #       - RetryableHttpError for 429/5xx/timeouts
    #       - NonRetryableHttpError for 400/403/404 etc
    #     Tenacity retries RetryableHttpError automatically.
    #     """
    #     self._rate_limit_wait()
    #     headers = self._auth_headers()

    #     try:
    #         resp = self._session.get(url, headers=headers, params=params, timeout=60)
    #     except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
    #         raise e

    #     status = resp.status_code
    #     text_snip = (resp.text or "")[:400]

    #     # 429: respect Retry-After if present, else random 45-60s (your log shows 49s,51s)
    #     if status == 429:
    #         ra = resp.headers.get("Retry-After")
    #         if ra and ra.isdigit():
    #             wait_s = int(ra)
    #         else:
    #             wait_s = random.randint(45, 60)

    #         # Tenacity will backoff; we ALSO force the required wait here
    #         time.sleep(wait_s)
    #         raise RetryableHttpError(status, url, f"429 Rate limited: {text_snip}", retry_after_seconds=wait_s)

    #     # Retryable 5xx / gateway issues
    #     if status in (500, 502, 503, 504):
    #         raise RetryableHttpError(status, url, f"{status}: {text_snip}")

    #     # Non-retryable client errors (bad request / forbidden)
    #     if status in (400, 401, 403, 404):
    #         raise NonRetryableHttpError(status, url, f"{status}: {text_snip}")

    #     if status != 200:
    #         # default: retry for unknown 5xx-like, else fail
    #         if 500 <= status < 600:
    #             raise RetryableHttpError(status, url, f"{status}: {text_snip}")
    #         raise NonRetryableHttpError(status, url, f"{status}: {text_snip}")

    #     return resp.json()

#     @retry(
#     retry=retry_if_exception(_is_retryable_exception),
#     stop=stop_after_attempt(REQ_RETRY_ATTEMPTS),
#     wait=wait_random_exponential(multiplier=2, min=2, max=90),
#     before_sleep=before_sleep_log(logger, "WARNING"),
#     reraise=True,
# )
#     def _request_json(self, url: str, params: dict | None = None) -> dict:
#         """
#         GET request wrapper with:
#         - Rate limiting
#         - Auth header injection (with safe catch for helper KeyErrors)
#         - Retryable/Non-retryable exception classification
#         - JSON response validation
#         """
#         self._rate_limit_wait()

#         # ── Auth — protect against KeyError / any helper failure ──
#         try:
#             headers = self._auth_headers()
#         except Exception as e:
#             logger.error(
#                 f"Auth header build failed | url={url} | error={repr(e)}\n"
#                 f"{traceback.format_exc()}"
#             )
#             # Treat auth failure as retryable (token refresh might recover)
#             raise RetryableHttpError(
#                 status_code=0,
#                 url=url,
#                 message=f"Auth error: {repr(e)}",
#             )

#         # ── HTTP call ──
#         try:
#             resp = self._session.get(url, headers=headers, params=params, timeout=60)
#         except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
#             raise e

#         status = resp.status_code
#         text_snip = (resp.text or "")[:800]

#         # 429
#         if status == 429:
#             ra = resp.headers.get("Retry-After")
#             wait_s = int(ra) if (ra and ra.isdigit()) else random.randint(45, 60)
#             logger.warning(f"429 Rate limited | sleep={wait_s}s | url={url}")
#             time.sleep(wait_s)
#             raise RetryableHttpError(
#                 status_code=status, url=url,
#                 message=f"429 Rate limited: {text_snip}",
#                 retry_after_seconds=wait_s,
#             )

#         # Retryable 5xx
#         if status in (500, 502, 503, 504):
#             raise RetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

#         # Non-retryable client errors
#         if status in (400, 401, 403, 404):
#             raise NonRetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

#         # Any other non-200
#         if status != 200:
#             if 500 <= status < 600:
#                 raise RetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")
#             raise NonRetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

#         # ── Parse JSON safely ──
#         try:
#             body = resp.json()
#         except Exception as e:
#             raise NonRetryableHttpError(
#                 status_code=status, url=url,
#                 message=f"JSON parse failed | url={url} | params={params} | error={repr(e)} | body={text_snip}",
#             )

#         if not isinstance(body, dict):
#             logger.warning(
#                 f"Unexpected response type={type(body).__name__} | url={url} "
#                 f"| params={params} | body={str(body)[:500]}"
#             )

#         return body
    
    @retry(
    retry=retry_if_exception(_is_retryable_exception),
    stop=stop_after_attempt(REQ_RETRY_ATTEMPTS),
    wait=wait_random_exponential(multiplier=2, min=2, max=90),
    before_sleep=before_sleep_log(logger, "WARNING"),
    reraise=True,
)
    def _request_json(self, url: str, params: dict | None = None) -> dict:
        self._rate_limit_wait()

        try:
            headers = self._auth_headers()
        except Exception as e:
            logger.error(f"Auth failed | url={url} | error={repr(e)}\n{traceback.format_exc()}")
            raise RetryableHttpError(status_code=0, url=url, message=f"Auth error: {repr(e)}")

        try:
            resp = self._session.get(url, headers=headers, params=params, timeout=60)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise e

        status = resp.status_code
        text_snip = (resp.text or "")[:800]

        if status == 429:
            ra = resp.headers.get("Retry-After")
            wait_s = int(ra) if (ra and ra.isdigit()) else random.randint(45, 60)
            logger.warning(f"429 Rate limited | sleep={wait_s}s | url={url}")
            time.sleep(wait_s)
            self._call_timestamps.clear()  # ← THIS IS THE FIX — reset rate limiter after 429 wait
            raise RetryableHttpError(
                status_code=status, url=url,
                message=f"429 Rate limited: {text_snip}",
                retry_after_seconds=wait_s,
            )

        if status in (500, 502, 503, 504):
            raise RetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

        if status in (400, 401, 403, 404):
            raise NonRetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

        if status != 200:
            if 500 <= status < 600:
                raise RetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")
            raise NonRetryableHttpError(status_code=status, url=url, message=f"{status}: {text_snip}")

        try:
            body = resp.json()
        except Exception as e:
            raise NonRetryableHttpError(
                status_code=status, url=url,
                message=f"JSON parse failed | error={repr(e)} | body={text_snip}",
            )

        return body
    
    # ─────────────────────────────────────────────
    # Response extraction
    # ─────────────────────────────────────────────

    # def _extract_records(self, body: dict | None) -> list[dict]:
    #     if not body:
    #         return []
    #     resp = body.get("response")
    #     if resp is None:
    #         return []
    #     if isinstance(resp, list):
    #         return [r for r in resp if isinstance(r, dict)]
    #     if isinstance(resp, dict):
    #         list_values = {k: v for k, v in resp.items() if isinstance(v, list)}
    #         if list_values:
    #             records = []
    #             for key, items in list_values.items():
    #                 for item in items:
    #                     if isinstance(item, dict):
    #                         item["_source_key"] = key
    #                         records.append(item)
    #             return records if records else [resp]
    #         return [resp]
    #     return []

    
    def _extract_records(self, body: dict | None) -> list[dict]:
        if not body:
            return []

        if not isinstance(body, dict):
            logger.warning(f"_extract_records: body is {type(body).__name__}, not dict | body={str(body)[:500]}")
            return []

        if "response" not in body:
            logger.warning(
                f"_extract_records: 'response' key missing | "
                f"keys={list(body.keys())} | body={str(body)[:500]}"
            )
            return []

        resp = body.get("response")
        if resp is None:
            return []

        if isinstance(resp, list):
            return [r for r in resp if isinstance(r, dict)]

        if isinstance(resp, dict):
            list_values = {k: v for k, v in resp.items() if isinstance(v, list)}
            if list_values:
                records = []
                for key, items in list_values.items():
                    for item in items:
                        if isinstance(item, dict):
                            item["_source_key"] = key
                            records.append(item)
                return records if records else [resp]
            return [resp]

        return []
    
    # ─────────────────────────────────────────────
    # Pagination with page-sleep + tqdm
    # ─────────────────────────────────────────────

    # def fetch_all_pages(self, url: str, params: dict | None = None,
    #                     desc: str = "Paginating") -> list[dict]:
    #     params = dict(params) if params else {}
    #     params["limit"] = self.PAGE_LIMIT
    #     page = 1
    #     all_records = []

    #     with tqdm(desc=desc, unit="rec", dynamic_ncols=True,
    #               bar_format="{l_bar}{bar}| {n_fmt} rec [{elapsed}]") as pbar:
    #         while True:
    #             params["page"] = page
    #             try:
    #                 body = self._request_json(url, params=params)
    #             except NonRetryableHttpError as e:
    #                 logger.warning(f"Non-retryable: {e.message}")
    #                 break
    #             except RetryableHttpError as e:
    #                 # Tenacity already retried; if still failing, break safely
    #                 logger.error(f"Retryable exhausted: {e.message}")
    #                 break
    #             except Exception as e:
    #                 logger.error(f"Unhandled request error: {e}")
    #                 break

    #             records = self._extract_records(body)
    #             if not records:
    #                 break

    #             all_records.extend(records)
    #             pbar.update(len(records))
    #             pbar.set_postfix_str(f"page={page}")

    #             if len(records) < self.PAGE_LIMIT:
    #                 break

    #             page += 1
    #             time.sleep(self.SLEEP_BETWEEN_PAGES)

    #     return all_records

    def fetch_all_pages(self, url: str, params: dict | None = None,
                    desc: str = "Paginating") -> list[dict]:
        params = dict(params) if params else {}
        params["limit"] = self.PAGE_LIMIT
        page = 1
        all_records = []

        with tqdm(desc=desc, unit="rec", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt} rec [{elapsed}]") as pbar:
            while True:
                params["page"] = page
                try:
                    body = self._request_json(url, params=params)
                except NonRetryableHttpError as e:
                    logger.warning(f"Non-retryable | url={url} | page={page} | {e.message}")
                    break
                except RetryableHttpError as e:
                    logger.error(f"Retryable exhausted | url={url} | page={page} | {e.message}")
                    break
                except Exception as e:
                    # Full traceback so we can see EXACTLY where the error is
                    logger.error(
                        f"Unhandled request error | url={url} | page={page} "
                        f"| params={params} | error={repr(e)}\n{traceback.format_exc()}"
                    )
                    break

                records = self._extract_records(body)
                if not records:
                    break

                all_records.extend(records)
                pbar.update(len(records))
                pbar.set_postfix_str(f"page={page}")

                if len(records) < self.PAGE_LIMIT:
                    break

                page += 1
                time.sleep(self.SLEEP_BETWEEN_PAGES)

        return all_records
    
    # ─────────────────────────────────────────────
    # Transform (CSV buffer -> Polars)
    # ─────────────────────────────────────────────

    def transform_to_polars(self, records: list[dict], report_type_value: str) -> pl.DataFrame:
        if not records:
            return pl.DataFrame()

        flat_records = [flatten_json(r) for r in records]

        all_cols = []
        seen = set()
        for rec in flat_records:
            for k in rec.keys():
                if k not in seen:
                    all_cols.append(k)
                    seen.add(k)

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=all_cols,
                                extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(flat_records)

        df = pl.read_csv(
            io.StringIO(csv_buffer.getvalue()),
            infer_schema_length=0,
            null_values=["", "null", "NULL", "None", "N/A"],
        )
        logger.info(f"  → {df.shape[0]} rows × {df.shape[1]} cols")

        ingestion_ts = pendulum.now("UTC").to_iso8601_string()
        ingestion_date = pendulum.now(TZ).format("YYYY-MM-DD")
        df = df.with_columns([
            pl.lit(ingestion_date).alias("ingestion_date"),
            pl.lit(ingestion_ts).alias("ingestion_timestamp"),
            pl.lit(report_type_value).alias("report_type"),
        ])

        rename_map = {}
        for col in df.columns:
            safe = sanitize_bq_column_name(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            df = df.rename(rename_map)

        return df

    # ─────────────────────────────────────────────
    # Endpoint fetch with per-parent retry queue
    # ─────────────────────────────────────────────

    def fetch_endpoint_records(self, endpoint: type[TrafficEndpointConfig],
                               parent_ids: list[int] | None = None,
                               start_time: str | None = None,
                               end_time: str | None = None) -> list[dict]:
        """
        Fetch records for endpoint. If failures happen for specific parents,
        collect them and re-run only those parents in ENDPOINT_RETRY_ROUNDS.
        """
        name = endpoint.ENDPOINT_NAME
        path = endpoint.ENDPOINT_PATH
        scope = endpoint.SCOPE
        param_name = endpoint.PARAM_NAME
        id_in_path = getattr(endpoint, "ID_IN_PATH", False)
        supports_date = getattr(endpoint, "SUPPORTS_DATE_FILTER", False)

        base_params = {}
        if supports_date and start_time and end_time:
            base_params.update({"startTime": start_time, "endTime": end_time})

        all_records: list[dict] = []

        # Seat-level endpoints
        # if scope == "seat":
        #     url = f"{self.BASE_URL}{path}"
        #     logger.info(f"[{name}] Fetching (seat-level): {path}")
        #     recs = self.fetch_all_pages(url, params=base_params or None, desc=f"🔄 {name}")
        #     logger.info(f"[{name}] Got {len(recs)} record(s)")
        #     return recs
        
        if scope == "seat":
            url = f"{self.BASE_URL}{path}"
            logger.info(f"[{name}] Fetching (seat-level): {path}")

            # These seat endpoints do NOT support page/limit pagination.
            # They return all records in a single GET call.
            NON_PAGINATED_SEATS = {"account_groups", "seats", "data_dictionary"}

            if name in NON_PAGINATED_SEATS:
                try:
                    body = self._request_json(url, params=base_params or None)
                except NonRetryableHttpError as e:
                    logger.warning(f"[{name}] Non-retryable: {e.message}")
                    return []
                except RetryableHttpError as e:
                    logger.error(f"[{name}] Retryable exhausted: {e.message}")
                    return []
                except Exception as e:
                    logger.error(f"[{name}] Unhandled: {repr(e)}\n{traceback.format_exc()}")
                    return []

                recs = self._extract_records(body)
                logger.info(f"[{name}] Got {len(recs)} record(s)")
                return recs
            else:
                recs = self.fetch_all_pages(url, params=base_params or None, desc=f"🔄 {name}")
                logger.info(f"[{name}] Got {len(recs)} record(s)")
                return recs


        # Child-level endpoints need parent IDs
        if not parent_ids:
            logger.warning(f"[{name}] No parent IDs for scope={scope} — skipping")
            return []

        def _fetch_for_parent(pid: int) -> list[dict]:
            if id_in_path:
                url = f"{self.BASE_URL}{path.replace('{id}', str(pid))}"
                params = dict(base_params)
                return self.fetch_all_pages(url, params=params, desc=f"{name}/{pid}")
            else:
                url = f"{self.BASE_URL}{path}"
                params = {param_name: pid}
                params.update(base_params)
                return self.fetch_all_pages(url, params=params, desc=f"{name}/{pid}")

        # Initial pass + retry queue
        failed_parents: list[int] = []
        logger.info(f"[{name}] Fetching ({scope}-level): {path} × {len(parent_ids)} parent(s) ({param_name})")

        with tqdm(total=len(parent_ids), desc=f"🔄 {name}", unit=param_name, dynamic_ncols=True) as pbar:
            for pid in parent_ids:
                try:
                    recs = _fetch_for_parent(pid)
                    for r in recs:
                        r[f"_parent_{param_name}"] = str(pid)
                    all_records.extend(recs)
                except Exception as e:
                    failed_parents.append(pid)
                    logger.error(f"[{name}] parent={pid} failed: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix_str(f"total={len(all_records)}")
                    time.sleep(self.SLEEP_BETWEEN_CALLS)

        # Retry rounds ONLY for failed parents
        for round_i in range(1, self.ENDPOINT_RETRY_ROUNDS + 1):
            if not failed_parents:
                break

            logger.warning(f"[{name}] Retry round {round_i}/{self.ENDPOINT_RETRY_ROUNDS} for {len(failed_parents)} parent(s)")
            still_failed: list[int] = []

            with tqdm(total=len(failed_parents), desc=f"♻️ retry {name}", unit=param_name, dynamic_ncols=True) as pbar:
                for pid in failed_parents:
                    try:
                        recs = _fetch_for_parent(pid)
                        for r in recs:
                            r[f"_parent_{param_name}"] = str(pid)
                        all_records.extend(recs)
                    except Exception as e:
                        still_failed.append(pid)
                        logger.error(f"[{name}] retry parent={pid} failed: {e}")
                    finally:
                        pbar.update(1)
                        pbar.set_postfix_str(f"total={len(all_records)}")
                        # bigger sleep on retry round to reduce 429
                        time.sleep(self.SLEEP_BETWEEN_CALLS + 1.5)

            failed_parents = still_failed

        if failed_parents:
            logger.error(f"[{name}] Permanently failed parents ({len(failed_parents)}): {failed_parents}")

        logger.info(f"[{name}] {len(all_records)} record(s) across {len(parent_ids)} parent(s)")
        return all_records

    # ─────────────────────────────────────────────
    # Run one endpoint
    # ─────────────────────────────────────────────

    def run_endpoint(self, endpoint: type[TrafficEndpointConfig],
                     output_dir: str,
                     api_type: str = API_TYPE,
                     parent_ids: list[int] | None = None,
                     start_time: str | None = None,
                     end_time: str | None = None) -> Optional[dict]:
        t0 = time.time()
        name = endpoint.ENDPOINT_NAME.upper()

        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name}] Starting traffic endpoint")
            logger.info("=" * 60)

            records = self.fetch_endpoint_records(endpoint, parent_ids, start_time, end_time)
            report_type_value = f"{api_type}_{endpoint.ENDPOINT_NAME}"
            df = self.transform_to_polars(records, report_type_value)

            if df.is_empty():
                logger.warning(f"[{name}] 0 records — nothing to save.")
                return {"df": df, "filepath": None, "day_str": None}

            day_str = pendulum.now(TZ).format("YYYY-MM-DD")
            epoch = int(time.time())
            group_dir = os.path.join(output_dir, endpoint.ENDPOINT_NAME, day_str)
            os.makedirs(group_dir, exist_ok=True)
            filename = f"{endpoint.ENDPOINT_NAME}_{day_str}_{epoch}.csv"
            filepath = os.path.join(group_dir, filename)
            df.write_csv(filepath)

            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {time.time() - t0:.1f}s → {os.path.abspath(filepath)}"
            )
            return {"df": df, "filepath": filepath, "day_str": day_str}

        except Exception as e:
            logger.error(f"❌ [{name}] FAILED: {e}\n{traceback.format_exc()}")
            return None

    # ─────────────────────────────────────────────
    # Run all endpoints (keep your old interface name)
    # ─────────────────────────────────────────────

    def run_all_endpoints(self, endpoints: list[type[TrafficEndpointConfig]],
                          output_dir: str,
                          api_type: str = API_TYPE,
                          start_time: str | None = None,
                          end_time: str | None = None) -> dict[str, Optional[dict]]:
        """
        Keeps backward compatibility with your existing main.py call signature.
        """
        results: dict[str, Optional[dict]] = {}

        seat_eps = [e for e in endpoints if e.SCOPE == "seat"]
        adv_eps = [e for e in endpoints if e.SCOPE == "advertiser"]
        camp_eps = [e for e in endpoints if e.SCOPE == "campaign"]
        line_eps = [e for e in endpoints if e.SCOPE == "line"]

        # PRE-STEP: collect parent IDs from raw API (no Polars)
        advertiser_ids: list[int] = []
        campaign_ids: list[int] = []
        line_ids: list[int] = []

        # advertiser IDs
        # if adv_eps or camp_eps or line_eps:
        #     adv_url = f"{self.BASE_URL}/traffic/advertisers"
        #     adv_recs = self.fetch_all_pages(adv_url, desc="📋 advertisers")
        #     advertiser_ids = [int(r["id"]) for r in adv_recs if isinstance(r, dict) and r.get("id") is not None]
        #     logger.success(f"Collected {len(advertiser_ids)} advertiser id(s)")

        if adv_eps or camp_eps or line_eps:
            adv_url = f"{self.BASE_URL}/traffic/advertisers"
            logger.info("=" * 60)
            logger.info("📋 PRE-STEP: Collecting advertiser IDs")
            logger.info("=" * 60)

            try:
                adv_recs = self.fetch_all_pages(adv_url, desc="📋 advertisers")
            except Exception as e:
                logger.error(
                    f"CRITICAL: /traffic/advertisers fetch failed | error={repr(e)}\n"
                    f"{traceback.format_exc()}"
                )
                adv_recs = []

            advertiser_ids = [
                int(r["id"])
                for r in adv_recs
                if isinstance(r, dict) and r.get("id") is not None
            ]

            if not advertiser_ids:
                logger.error(
                    "CRITICAL: 0 advertiser IDs collected from /traffic/advertisers. "
                    "All advertiser/campaign/line scoped endpoints will be skipped. "
                    "Check auth token, API access, and response format."
                )
            else:
                logger.success(f"Collected {len(advertiser_ids)} advertiser id(s): {advertiser_ids}")

        # campaign IDs
        if (camp_eps or line_eps) and advertiser_ids:
            camp_url = f"{self.BASE_URL}/traffic/campaigns"
            for aid in tqdm(advertiser_ids, desc="📋 campaigns ids", unit="accountId", dynamic_ncols=True):
                params = {"accountId": aid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    params["page"] = page
                    try:
                        body = self._request_json(camp_url, params=params)
                    except Exception:
                        break
                    recs = self._extract_records(body)
                    if not recs:
                        break
                    campaign_ids.extend([int(r["id"]) for r in recs if r.get("id") is not None])
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP_BETWEEN_PAGES)
                time.sleep(self.SLEEP_BETWEEN_CALLS)
            logger.success(f"Collected {len(campaign_ids)} campaign id(s)")

        # line IDs (needed for ads)
        if line_eps and campaign_ids:
            lines_url = f"{self.BASE_URL}/traffic/lines"
            for oid in tqdm(campaign_ids, desc="📋 line ids", unit="orderId", dynamic_ncols=True):
                params = {"orderId": oid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    params["page"] = page
                    try:
                        body = self._request_json(lines_url, params=params)
                    except Exception:
                        break
                    recs = self._extract_records(body)
                    if not recs:
                        break
                    line_ids.extend([int(r["id"]) for r in recs if r.get("id") is not None])
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP_BETWEEN_PAGES)
                time.sleep(self.SLEEP_BETWEEN_CALLS)
            logger.success(f"Collected {len(line_ids)} line id(s)")

        total_eps = len(seat_eps) + len(adv_eps) + len(camp_eps) + len(line_eps)

        with tqdm(total=total_eps, desc="🚀 Traffic Pipeline", unit="ep", dynamic_ncols=True) as master:
            # seat
            for ep in seat_eps:
                results[ep.ENDPOINT_NAME] = self.run_endpoint(ep, output_dir, api_type, None, start_time, end_time)
                master.update(1)
                master.set_postfix_str(ep.ENDPOINT_NAME)

            # advertiser
            for ep in adv_eps:
                results[ep.ENDPOINT_NAME] = self.run_endpoint(ep, output_dir, api_type, advertiser_ids, start_time, end_time)
                master.update(1)
                master.set_postfix_str(ep.ENDPOINT_NAME)

            # campaign
            for ep in camp_eps:
                results[ep.ENDPOINT_NAME] = self.run_endpoint(ep, output_dir, api_type, campaign_ids, start_time, end_time)
                master.update(1)
                master.set_postfix_str(ep.ENDPOINT_NAME)

            # line
            for ep in line_eps:
                results[ep.ENDPOINT_NAME] = self.run_endpoint(ep, output_dir, api_type, line_ids, start_time, end_time)
                master.update(1)
                master.set_postfix_str(ep.ENDPOINT_NAME)

        return results