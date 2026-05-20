"""
dsp_traffic.py
──────────────
Yahoo DSP Traffic API module — production-ready (resilient).

Hierarchy:
    Seat (no params) → Advertiser (accountId) → Campaign (orderId) → Line (lineId)

Key design decisions:
    • NO date filters — Traffic API is metadata/snapshots, NOT time-series.
      POC confirmed: without date filters → 17 advertisers (ALL).
      With date filters → only 9 (recently modified). We want ALL.
    • IDs collected from raw JSON BEFORE any Polars transforms.
    • Transform uses csv.DictWriter → pl.read_csv(infer_schema_length=0)
      which guarantees ALL columns are Utf8 (no type clashes).

Resilience:
    1) Tenacity-based retry for transient failures (429/5xx/timeouts)
    2) Rate-limiter timestamp reset after 429 sleep
    3) Per-endpoint consecutive-failure bail-out (skip → retry later)
    4) Pipeline-level retry queue (deferred endpoints run at the end)
    5) Sleep between every paginated page pull to reduce 429 spikes
    6) Non-paginated seat endpoints handled with single GET (no page/limit)

Returns same dict format as reporting.py:
    {"df": pl.DataFrame, "filepath": str, "day_str": str, "failed_parents": list}

GCS / BQ uploads are handled by main.py via helpers.py.
"""

import io
import csv
import os
import time
import random
import traceback
from dataclasses import dataclass
from typing import Optional

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


# ═══════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
# ENDPOINT CONFIG — dynamic from metrics.py
# ═══════════════════════════════════════════════════════════════

class TrafficEndpointConfig:
    """Base class for a Traffic API endpoint descriptor."""
    ENDPOINT_NAME: str = "base"
    ENDPOINT_PATH: str = "/traffic/base"
    SCOPE: str = "seat"          # seat | advertiser | campaign | line
    PARAM_NAME: str = ""         # accountId | orderId | lineId
    ID_IN_PATH: bool = False     # if True, parent id is placed into {id} in path


def _build_endpoint_class(name: str, config: dict) -> type:
    class_name = "".join(w.capitalize() for w in name.split("_")) + "Endpoint"
    attrs = {
        "ENDPOINT_NAME": name,
        "ENDPOINT_PATH": config["path"],
        "SCOPE": config.get("scope", "seat"),
        "PARAM_NAME": config.get("param_name", ""),
        "ID_IN_PATH": config.get("id_in_path", False),
    }
    return type(class_name, (TrafficEndpointConfig,), attrs)


ENDPOINT_REGISTRY: dict[str, type[TrafficEndpointConfig]] = {}
for _name, _config in TRAFFIC_API_ENDPOINT_CONFIGS.items():
    ENDPOINT_REGISTRY[_name] = _build_endpoint_class(_name, _config)

ALL_TRAFFIC_ENDPOINTS: list[type[TrafficEndpointConfig]] = list(ENDPOINT_REGISTRY.values())


# ═══════════════════════════════════════════════════════════════
# YAHOO TRAFFIC CLASS
# ═══════════════════════════════════════════════════════════════

class YahooTraffic:
    """
    Yahoo DSP Traffic API extractor with resilient retry semantics.

    Phases:
        PRE-STEP — Collect all parent IDs (raw JSON, no Polars)
        PHASE 1  — Seat-level endpoints (no params)
        PHASE 2  — Advertiser-level endpoints (per accountId)
        PHASE 3  — Campaign-level endpoints (per orderId)
        PHASE 4  — Line-level endpoints (per lineId)
        RETRY    — Deferred endpoints from first pass
    """

    BASE_URL = "https://dspapi.admanagerplus.yahoo.com"

    RATE_LIMIT_PER_MIN = 55
    SLEEP_BETWEEN_CALLS = 3.0
    SLEEP_BETWEEN_PAGES = 3.0
    PAGE_LIMIT = 100

    REQ_RETRY_ATTEMPTS = 6          # tenacity retries per single request
    ENDPOINT_RETRY_ROUNDS = 2       # internal retry rounds per endpoint
    MAX_CONSECUTIVE_PARENT_FAILURES = 5  # bail-out threshold

    # Seat endpoints that do NOT support page/limit pagination
    NON_PAGINATED_SEATS = {"account_groups", "seats", "data_dictionary"}

    def __init__(self):
        self.helper = YahooHelpers()
        self._call_timestamps: list[float] = []
        self._session = requests.Session()
        logger.success("YahooTraffic initialised.")

    # ───────────────────────────────────────────────────────
    # AUTH HEADERS
    # ───────────────────────────────────────────────────────

    def auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    # ───────────────────────────────────────────────────────
    # RATE LIMITER
    # ───────────────────────────────────────────────────────

    def rate_limit_wait(self):
        now = time.time()
        self._call_timestamps = [t for t in self._call_timestamps if now - t < 60]
        if len(self._call_timestamps) >= self.RATE_LIMIT_PER_MIN:
            oldest = self._call_timestamps[0]
            wait = 60 - (now - oldest) + 1
            if wait > 0:
                logger.info(f"⏱️  Rate limit: waiting {wait:.0f}s ...")
                time.sleep(wait)
        self._call_timestamps.append(time.time())

    # ───────────────────────────────────────────────────────
    # TENACITY-WRAPPED REQUEST
    # ───────────────────────────────────────────────────────

    @retry(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(REQ_RETRY_ATTEMPTS),
        wait=wait_random_exponential(multiplier=2, min=2, max=90),
        before_sleep=before_sleep_log(logger, "WARNING"),
        reraise=True,
    )
    def request_json(self, url: str, params: dict | None = None) -> dict:
        """
        GET request wrapper with:
          - Rate limiting
          - Auth header injection (safe catch for helper errors)
          - Retryable/Non-retryable exception classification
          - JSON response validation
          - Rate-limiter reset after 429 sleep
        """
        self.rate_limit_wait()

        # ── Auth ──
        try:
            headers = self.auth_headers()
        except Exception as e:
            logger.error(
                f"Auth header build failed | url={url} | error={repr(e)}\n"
                f"{traceback.format_exc()}"
            )
            raise RetryableHttpError(
                status_code=0, url=url,
                message=f"Auth error: {repr(e)}",
            )

        # ── HTTP call ──
        try:
            resp = self._session.get(url, headers=headers, params=params, timeout=60)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            raise e

        status = resp.status_code
        text_snip = (resp.text or "")[:800]

        # 429
        if status == 429:
            ra = resp.headers.get("Retry-After")
            wait_s = int(ra) if (ra and ra.isdigit()) else random.randint(45, 60)
            logger.warning(f"429 Rate limited | sleep={wait_s}s | url={url}")
            time.sleep(wait_s)
            self._call_timestamps.clear()
            raise RetryableHttpError(
                status_code=status, url=url,
                message=f"429 Rate limited: {text_snip}",
                retry_after_seconds=wait_s,
            )

        # Retryable 5xx
        if status in (500, 502, 503, 504):
            raise RetryableHttpError(
                status_code=status, url=url,
                message=f"{status}: {text_snip}",
            )

        # Non-retryable client errors
        if status in (400, 401, 403, 404):
            raise NonRetryableHttpError(
                status_code=status, url=url,
                message=f"{status}: {text_snip}",
            )

        # Any other non-200
        if status != 200:
            if 500 <= status < 600:
                raise RetryableHttpError(
                    status_code=status, url=url,
                    message=f"{status}: {text_snip}",
                )
            raise NonRetryableHttpError(
                status_code=status, url=url,
                message=f"{status}: {text_snip}",
            )

        # ── Parse JSON safely ──
        try:
            body = resp.json()
        except Exception as e:
            raise NonRetryableHttpError(
                status_code=status, url=url,
                message=(
                    f"JSON parse failed | url={url} | params={params} "
                    f"| error={repr(e)} | body={text_snip}"
                ),
            )

        if not isinstance(body, dict):
            logger.warning(
                f"Unexpected response type={type(body).__name__} | url={url} "
                f"| params={params} | body={str(body)[:500]}"
            )

        return body

    # ───────────────────────────────────────────────────────
    # EXTRACT RECORDS FROM RESPONSE JSON
    # ───────────────────────────────────────────────────────

    def extract_records(self, body: dict | None) -> list[dict]:
        """
        Extract list of records from Traffic API JSON response.
        Handles:
          {"response": [...]}
          {"response": {...}}
          {"response": {"advertisers": [...]}}
        """
        if not body:
            return []

        if not isinstance(body, dict):
            logger.warning(
                f"extract_records: body is {type(body).__name__}, not dict "
                f"| body={str(body)[:500]}"
            )
            return []

        if "response" not in body:
            logger.warning(
                f"extract_records: 'response' key missing | "
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

        logger.warning(
            f"extract_records: unsupported response type={type(resp).__name__} "
            f"| body={str(body)[:500]}"
        )
        return []

    # ───────────────────────────────────────────────────────
    # PAGINATED FETCH — with page-sleep + tqdm
    # ───────────────────────────────────────────────────────
    
    def fetch_all_pages(self,url: str,params: dict | None = None,desc: str = "Paginating",raise_on_error: bool = False,) -> list[dict]:
        """
        Paginate through a list endpoint. Returns all records.

        Args:
            raise_on_error: If True, re-raise the exception instead of
                            silently returning partial/empty results.
                            Use True for child-level fetches (per parent ID)
                            so the caller knows the parent failed.
                            Use False (default) for top-level collection
                            (advertiser IDs etc.) for backward compatibility.
        """
        params = dict(params) if params else {}
        params["limit"] = self.PAGE_LIMIT
        page = 1
        all_records: list[dict] = []

        with tqdm(
            desc=desc, unit="rec", dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt} rec [{elapsed}]",
        ) as pbar:
            while True:
                params["page"] = page
                try:
                    body = self.request_json(url, params=params)
                except NonRetryableHttpError as e:
                    logger.warning(
                        f"Non-retryable | url={url} | page={page} | {e.message}"
                    )
                    if raise_on_error:
                        raise
                    break
                except RetryableHttpError as e:
                    logger.error(
                        f"Retryable exhausted | url={url} | page={page} | {e.message}"
                    )
                    if raise_on_error:
                        raise
                    break
                except Exception as e:
                    logger.error(
                        f"Unhandled request error | url={url} | page={page} "
                        f"| params={params} | error={repr(e)}\n{traceback.format_exc()}"
                    )
                    if raise_on_error:
                        raise
                    break

                records = self.extract_records(body)
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

    # ───────────────────────────────────────────────────────
    # TRANSFORM → POLARS (via CSV buffer, guaranteed Utf8)
    # ───────────────────────────────────────────────────────

    def transform_to_polars(self, records: list[dict], report_type_value: str,) -> pl.DataFrame:
        """
        Flatten JSON records → CSV in memory → Polars DataFrame.
        ALL columns are Utf8 (STRING in BQ). No type clashes.
        """
        if not records:
            return pl.DataFrame()

        flat_records = [flatten_json(r) for r in records]

        all_cols: list[str] = []
        seen: set[str] = set()
        for rec in flat_records:
            for k in rec.keys():
                if k not in seen:
                    all_cols.append(k)
                    seen.add(k)

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer, fieldnames=all_cols,
            extrasaction="ignore", restval="",
        )
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

        rename_map: dict[str, str] = {}
        for col in df.columns:
            safe = sanitize_bq_column_name(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            df = df.rename(rename_map)

        return df

    # ═══════════════════════════════════════════════════════
    # FETCH ENDPOINT RECORDS — with bail-out + retry queue
    # ═══════════════════════════════════════════════════════

    def fetch_endpoint_records(
        self,
        endpoint: type[TrafficEndpointConfig],
        parent_ids: list[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> tuple[list[dict], list[int]]:
        """
        Fetch records for one endpoint.
        Returns: (records, failed_parent_ids)

        Bail-out: if MAX_CONSECUTIVE_PARENT_FAILURES consecutive parents fail,
        stop early — remaining parents go into failed list for pipeline-level retry.
        """
        name = endpoint.ENDPOINT_NAME
        path = endpoint.ENDPOINT_PATH
        scope = endpoint.SCOPE
        param_name = endpoint.PARAM_NAME
        id_in_path = getattr(endpoint, "ID_IN_PATH", False)

        all_records: list[dict] = []

        # ── SEAT-LEVEL ──
        if scope == "seat":
            url = f"{self.BASE_URL}{path}"
            logger.info(f"[{name}] Fetching (seat-level): {path}")

            if name in self.NON_PAGINATED_SEATS:
                try:
                    body = self.request_json(url)
                except NonRetryableHttpError as e:
                    logger.warning(f"[{name}] Non-retryable: {e.message}")
                    return [], []
                except RetryableHttpError as e:
                    logger.error(f"[{name}] Retryable exhausted: {e.message}")
                    return [], []
                except Exception as e:
                    logger.error(
                        f"[{name}] Unhandled: {repr(e)}\n{traceback.format_exc()}"
                    )
                    return [], []

                recs = self.extract_records(body)
                logger.info(f"[{name}] Got {len(recs)} record(s)")
                time.sleep(self.SLEEP_BETWEEN_CALLS)
                return recs, []
            else:
                recs = self.fetch_all_pages(url, desc=f"🔄 {name}")
                logger.info(f"[{name}] Got {len(recs)} record(s)")
                return recs, []

        # ── CHILD-LEVEL ──
        if not parent_ids:
            logger.warning(f"[{name}] No parent IDs for scope={scope} — skipping")
            return [], []

        # def fetch_for_parent(pid: int) -> list[dict]:
        #     if id_in_path:
        #         _url = f"{self.BASE_URL}{path.replace('{id}', str(pid))}"
        #         return self.fetch_all_pages(_url, desc=f"{name}/{pid}")
        #     else:
        #         _url = f"{self.BASE_URL}{path}"
        #         _params = {param_name: pid}
        #         return self.fetch_all_pages(_url, params=_params, desc=f"{name}/{pid}")
        
        
        def fetch_for_parent(pid: int) -> list[dict]:
            if id_in_path:
                _url = f"{self.BASE_URL}{path.replace('{id}', str(pid))}"
                return self.fetch_all_pages(_url, desc=f"{name}/{pid}", raise_on_error=True)
            else:
                _url = f"{self.BASE_URL}{path}"
                _params = {param_name: pid}
                return self.fetch_all_pages(_url, params=_params, desc=f"{name}/{pid}", raise_on_error=True)


        # ── First pass with consecutive-failure bail-out ──
        failed_parents: list[int] = []
        consecutive_failures = 0
        bailed_out = False

        logger.info(
            f"[{name}] Fetching ({scope}-level): {path} "
            f"× {len(parent_ids)} parent(s) ({param_name})"
        )

        with tqdm(
            total=len(parent_ids), desc=f"🔄 {name}",
            unit=param_name, dynamic_ncols=True,
        ) as pbar:
            for i, pid in enumerate(parent_ids):
                try:
                    recs = fetch_for_parent(pid)
                    for r in recs:
                        r[f"_parent_{param_name}"] = str(pid)
                    all_records.extend(recs)
                    consecutive_failures = 0
                except Exception as e:
                    consecutive_failures += 1
                    failed_parents.append(pid)
                    logger.error(f"[{name}] parent={pid} failed: {e}")

                    if consecutive_failures >= self.MAX_CONSECUTIVE_PARENT_FAILURES:
                        remaining = parent_ids[i + 1:]
                        failed_parents.extend(remaining)
                        logger.error(
                            f"[{name}] ⚠️ {self.MAX_CONSECUTIVE_PARENT_FAILURES} "
                            f"consecutive failures — bailing out. "
                            f"{len(remaining)} parent(s) deferred to retry queue."
                        )
                        bailed_out = True
                        pbar.update(len(remaining) + 1)
                        break
                finally:
                    if not bailed_out:
                        pbar.update(1)
                        pbar.set_postfix_str(
                            f"ok={len(all_records)} fail={len(failed_parents)}"
                        )
                        time.sleep(self.SLEEP_BETWEEN_CALLS)

        # ── Internal retry rounds ONLY if we did NOT bail out ──
        if not bailed_out:
            for round_i in range(1, self.ENDPOINT_RETRY_ROUNDS + 1):
                if not failed_parents:
                    break

                logger.warning(
                    f"[{name}] Retry round {round_i}/{self.ENDPOINT_RETRY_ROUNDS} "
                    f"for {len(failed_parents)} parent(s)"
                )
                still_failed: list[int] = []

                with tqdm(
                    total=len(failed_parents), desc=f"♻️ retry {name}",
                    unit=param_name, dynamic_ncols=True,
                ) as pbar:
                    for pid in failed_parents:
                        try:
                            recs = fetch_for_parent(pid)
                            for r in recs:
                                r[f"_parent_{param_name}"] = str(pid)
                            all_records.extend(recs)
                        except Exception as e:
                            still_failed.append(pid)
                            logger.error(f"[{name}] retry parent={pid} failed: {e}")
                        finally:
                            pbar.update(1)
                            pbar.set_postfix_str(f"total={len(all_records)}")
                            time.sleep(self.SLEEP_BETWEEN_CALLS + 1.5)

                failed_parents = still_failed
        else:
            logger.warning(
                f"[{name}] Skipping internal retry rounds (bailed out). "
                f"{len(failed_parents)} parent(s) queued for pipeline-level retry."
            )

        if failed_parents:
            logger.error(
                f"[{name}] {len(failed_parents)} parent(s) still failed: "
                f"{failed_parents[:20]}..."
            )

        logger.info(
            f"[{name}] {len(all_records)} record(s) fetched | "
            f"{len(failed_parents)} parent(s) pending retry"
        )
        return all_records, failed_parents

    # ═══════════════════════════════════════════════════════
    # RUN SINGLE ENDPOINT (fetch + transform + save CSV)
    # ═══════════════════════════════════════════════════════

    def run_endpoint(
        self,
        endpoint: type[TrafficEndpointConfig],
        output_dir: str,
        api_type: str = API_TYPE,
        parent_ids: list[int] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> Optional[dict]:
        t0 = time.time()
        name = endpoint.ENDPOINT_NAME.upper()

        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name}] Starting traffic endpoint")
            logger.info("=" * 60)

            records, failed_parents = self.fetch_endpoint_records(
                endpoint, parent_ids, start_time, end_time,
            )
            report_type_value = f"{api_type}_{endpoint.ENDPOINT_NAME}"
            df = self.transform_to_polars(records, report_type_value)

            if df.is_empty():
                logger.warning(f"[{name}] 0 records — nothing to save.")
                return {
                    "df": df, "filepath": None, "day_str": None,
                    "failed_parents": failed_parents,
                }

            day_str = pendulum.now(TZ).format("YYYY-MM-DD")
            epoch = int(time.time())
            group_dir = os.path.join(output_dir, endpoint.ENDPOINT_NAME, day_str)
            os.makedirs(group_dir, exist_ok=True)
            filename = f"{endpoint.ENDPOINT_NAME}_{day_str}_{epoch}.csv"
            filepath = os.path.join(group_dir, filename)
            df.write_csv(filepath)

            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| failed_parents={len(failed_parents)} "
                f"| {time.time() - t0:.1f}s → {os.path.abspath(filepath)}"
            )
            return {
                "df": df, "filepath": filepath, "day_str": day_str,
                "failed_parents": failed_parents,
            }

        except Exception as e:
            logger.error(f"❌ [{name}] FAILED: {e}\n{traceback.format_exc()}")
            return None

    # ═══════════════════════════════════════════════════════
    # RUN ALL — hierarchical pipeline + retry queue
    # ═══════════════════════════════════════════════════════

    def run_all_endpoints(
        self,
        endpoints: list[type[TrafficEndpointConfig]],
        output_dir: str,
        api_type: str = API_TYPE,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Optional[dict]]:
        """
        Run all traffic endpoints respecting the hierarchy.

        Traffic API = metadata snapshots. NO date filters.
        Always fetches the full current state of all entities.

        Any endpoint that fails or partially fails is queued for a
        pipeline-level retry at the very end (after a 90s cooldown).
        """
        results: dict[str, Optional[dict]] = {}
        retry_queue: list[tuple[type[TrafficEndpointConfig], list[int] | None]] = []

        seat_eps = [e for e in endpoints if e.SCOPE == "seat"]
        adv_eps = [e for e in endpoints if e.SCOPE == "advertiser"]
        camp_eps = [e for e in endpoints if e.SCOPE == "campaign"]
        line_eps = [e for e in endpoints if e.SCOPE == "line"]

        # ══════════════════════════════════════════════════
        # PRE-STEP: Collect all parent IDs (raw, no Polars)
        # ══════════════════════════════════════════════════
        advertiser_ids: list[int] = []
        campaign_ids: list[int] = []
        line_ids: list[int] = []

        if adv_eps or camp_eps or line_eps:
            logger.info("=" * 60)
            logger.info("📋 PRE-STEP: Collecting advertiser IDs")
            logger.info("=" * 60)

            try:
                adv_url = f"{self.BASE_URL}/traffic/advertisers"
                adv_recs = self.fetch_all_pages(adv_url, desc="📋 advertisers")
            except Exception as e:
                logger.error(
                    f"CRITICAL: /traffic/advertisers fetch failed | "
                    f"error={repr(e)}\n{traceback.format_exc()}"
                )
                adv_recs = []

            advertiser_ids = [
                int(r["id"]) for r in adv_recs
                if isinstance(r, dict) and r.get("id") is not None
            ]

            if not advertiser_ids:
                logger.error(
                    "CRITICAL: 0 advertiser IDs collected. "
                    "All advertiser/campaign/line endpoints will be skipped. "
                    "Check auth token, API access, and response format."
                )
            else:
                logger.success(
                    f"Collected {len(advertiser_ids)} advertiser id(s): "
                    f"{advertiser_ids}"
                )

        if (camp_eps or line_eps) and advertiser_ids:
            logger.info("=" * 60)
            logger.info(
                f"📋 PRE-STEP: Collecting campaign IDs "
                f"({len(advertiser_ids)} advertisers)"
            )
            logger.info("=" * 60)

            camp_url = f"{self.BASE_URL}/traffic/campaigns"
            for aid in tqdm(
                advertiser_ids, desc="📋 campaign ids",
                unit="accountId", dynamic_ncols=True,
            ):
                params = {"accountId": aid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    params["page"] = page
                    try:
                        body = self.request_json(camp_url, params=params)
                    except Exception:
                        break
                    recs = self.extract_records(body)
                    if not recs:
                        break
                    campaign_ids.extend(
                        [int(r["id"]) for r in recs if r.get("id") is not None]
                    )
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP_BETWEEN_PAGES)
                time.sleep(self.SLEEP_BETWEEN_CALLS)
            logger.success(f"Collected {len(campaign_ids)} campaign id(s)")

        if line_eps and campaign_ids:
            logger.info("=" * 60)
            logger.info(
                f"📋 PRE-STEP: Collecting line IDs "
                f"({len(campaign_ids)} campaigns)"
            )
            logger.info("=" * 60)

            lines_url = f"{self.BASE_URL}/traffic/lines"
            for oid in tqdm(
                campaign_ids, desc="📋 line ids",
                unit="orderId", dynamic_ncols=True,
            ):
                params = {"orderId": oid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    params["page"] = page
                    try:
                        body = self.request_json(lines_url, params=params)
                    except Exception:
                        break
                    recs = self.extract_records(body)
                    if not recs:
                        break
                    line_ids.extend(
                        [int(r["id"]) for r in recs if r.get("id") is not None]
                    )
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP_BETWEEN_PAGES)
                time.sleep(self.SLEEP_BETWEEN_CALLS)
            logger.success(f"Collected {len(line_ids)} line id(s)")

        # ── Map scope → parent IDs ──
        scope_to_pids = {
            "seat": None,
            "advertiser": advertiser_ids,
            "campaign": campaign_ids,
            "line": line_ids,
        }

        total_eps = len(seat_eps) + len(adv_eps) + len(camp_eps) + len(line_eps)

        # ══════════════════════════════════════════════════
        # FIRST PASS — run all endpoints
        # ══════════════════════════════════════════════════
        phases = [
            ("SEAT", seat_eps),
            ("ADVERTISER", adv_eps),
            ("CAMPAIGN", camp_eps),
            ("LINE", line_eps),
        ]

        with tqdm(
            total=total_eps, desc="🚀 Traffic Pipeline",
            unit="ep", dynamic_ncols=True,
        ) as master:
            for phase_name, eps in phases:
                if not eps:
                    continue

                pids = scope_to_pids.get(eps[0].SCOPE)

                if phase_name != "SEAT" and not pids:
                    logger.warning(f"⚠️ No parent IDs — skipping {phase_name}")
                    for ep in eps:
                        results[ep.ENDPOINT_NAME] = None
                        master.update(1)
                    continue

                for ep in eps:
                    result = self.run_endpoint(
                        ep, output_dir, api_type, pids, start_time, end_time,
                    )
                    results[ep.ENDPOINT_NAME] = result

                    if result is None:
                        retry_queue.append((ep, pids))
                        logger.warning(
                            f"⏳ [{ep.ENDPOINT_NAME}] queued for retry (crashed)"
                        )
                    elif result.get("failed_parents"):
                        retry_queue.append((ep, result["failed_parents"]))
                        logger.warning(
                            f"⏳ [{ep.ENDPOINT_NAME}] queued for retry "
                            f"({len(result['failed_parents'])} failed parent(s))"
                        )

                    master.update(1)
                    master.set_postfix_str(ep.ENDPOINT_NAME)
                    time.sleep(self.SLEEP_BETWEEN_CALLS)

        # ══════════════════════════════════════════════════
        # RETRY QUEUE — deferred endpoints
        # ══════════════════════════════════════════════════
        if retry_queue:
            cooldown = 90
            logger.info("")
            logger.info("=" * 60)
            logger.info(
                f"♻️ RETRY QUEUE: {len(retry_queue)} endpoint(s) to retry "
                f"| cooldown {cooldown}s before starting"
            )
            logger.info("=" * 60)

            self._call_timestamps.clear()
            time.sleep(cooldown)

            with tqdm(
                total=len(retry_queue), desc="♻️ Retry Queue",
                unit="ep", dynamic_ncols=True,
            ) as rpbar:
                for ep, retry_pids in retry_queue:
                    logger.info(
                        f"♻️ Retrying [{ep.ENDPOINT_NAME}] "
                        f"| {len(retry_pids) if retry_pids else 0} parent(s)"
                    )

                    retry_result = self.run_endpoint(
                        ep, output_dir, api_type, retry_pids, start_time, end_time,
                    )

                    if retry_result is not None:
                        still_failed = retry_result.get("failed_parents", [])
                        if still_failed:
                            logger.error(
                                f"[{ep.ENDPOINT_NAME}] Still "
                                f"{len(still_failed)} parent(s) failed after retry: "
                                f"{still_failed[:20]}..."
                            )
                        else:
                            logger.success(
                                f"✅ [{ep.ENDPOINT_NAME}] retry fully succeeded"
                            )
                    else:
                        logger.error(
                            f"❌ [{ep.ENDPOINT_NAME}] retry also crashed"
                        )

                    rpbar.update(1)
                    rpbar.set_postfix_str(ep.ENDPOINT_NAME)
                    time.sleep(self.SLEEP_BETWEEN_CALLS * 2)

        # ══════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════
        ok = sum(
            1 for v in results.values()
            if v and v.get("df") is not None and not v["df"].is_empty()
        )
        fail = len(results) - ok

        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"📋 TRAFFIC SUMMARY: {ok} OK | {fail} fail/empty "
            f"| {len(results)} total"
        )
        logger.info(f"📋 RETRY QUEUE HAD: {len(retry_queue)} endpoint(s)")

        for ep_name, res in results.items():
            if res and res.get("df") is not None and not res["df"].is_empty():
                fp_count = len(res.get("failed_parents", []))
                info = (
                    f"{res['df'].shape[0]:>8,} rows "
                    f"× {res['df'].shape[1]} cols | pending={fp_count}"
                )
            elif res and res.get("df") is not None:
                info = "0 rows (empty)"
            else:
                info = "ERROR/SKIPPED"
            logger.info(f"    {ep_name:30s} → {info}")

        logger.info("=" * 60)

        return results