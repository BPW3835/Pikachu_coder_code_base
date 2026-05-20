"""
dsp_traffic.py
──────────────
Yahoo DSP Traffic API module — production-ready.

Hierarchy:
    Seat → Advertiser (accountId) → Campaign (orderId for packages/lines)

Features:
    • Date range filtering (startTime/endTime) on all list endpoints
    • Paginated GET with tqdm progress bars
    • Rate limiting (55 GET/min, under 60 quota)
    • Nested JSON flattening → CSV → Polars (guaranteed Utf8)
    • Column sanitization for BQ
    • Same output dict as reporting.py

GCS / BQ load steps are handled by main.py via helpers.py.
"""

import io
import csv
import os
import json
import time
import random
import traceback
from typing import Optional

import pendulum
import polars as pl
import requests
from loguru import logger
from tqdm import tqdm

from helpers import YahooHelpers, sanitize_bq_column_name, flatten_json
from metrics import TRAFFIC_API_ENDPOINT_CONFIGS

API_TYPE = "traffic_api"


# ═══════════════════════════════════════════════════════════════
# TRAFFIC ENDPOINT CONFIG (base class)
# ═══════════════════════════════════════════════════════════════

class TrafficEndpointConfig:
    ENDPOINT_NAME: str = "base"
    ENDPOINT_PATH: str = "/traffic/base"
    SCOPE: str = "seat"
    PARAM_NAME: str = ""
    ID_IN_PATH: bool = False
    SUPPORTS_DATE_FILTER: bool = False


# ═══════════════════════════════════════════════════════════════
# DYNAMIC ENDPOINT CLASS GENERATION from metrics.py
# ═══════════════════════════════════════════════════════════════

def _build_endpoint_class(name: str, config: dict) -> type:
    class_name = "".join(w.capitalize() for w in name.split("_")) + "Endpoint"
    attrs = {
        "ENDPOINT_NAME":        name,
        "ENDPOINT_PATH":        config["path"],
        "SCOPE":                config.get("scope", "seat"),
        "PARAM_NAME":           config.get("param_name", ""),
        "ID_IN_PATH":           config.get("id_in_path", False),
        "SUPPORTS_DATE_FILTER": config.get("supports_date_filter", False),
    }
    return type(class_name, (TrafficEndpointConfig,), attrs)


ENDPOINT_REGISTRY: dict[str, type[TrafficEndpointConfig]] = {}
for _name, _config in TRAFFIC_API_ENDPOINT_CONFIGS.items():
    _cls = _build_endpoint_class(_name, _config)
    ENDPOINT_REGISTRY[_name] = _cls

ALL_TRAFFIC_ENDPOINTS: list[type[TrafficEndpointConfig]] = list(ENDPOINT_REGISTRY.values())


# ═══════════════════════════════════════════════════════════════
# YAHOO TRAFFIC CLASS
# ═══════════════════════════════════════════════════════════════

class YahooTraffic:

    BASE_URL = "https://dspapi.admanagerplus.yahoo.com"
    RATE_LIMIT_PER_MIN = 55
    SLEEP_BETWEEN_CALLS = 1.2
    PAGE_LIMIT = 100

    def __init__(self):
        try:
            self.helper = YahooHelpers()
            self._call_timestamps: list[float] = []
            logger.success("YahooTraffic initialised.")
        except Exception as e:
            logger.error(f"Init failed: {e}\n{traceback.format_exc()}")
            raise

    # ───────────────────────────────────────────────────────
    # AUTH
    # ───────────────────────────────────────────────────────

    def _auth_headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    # ───────────────────────────────────────────────────────
    # RATE LIMITER
    # ───────────────────────────────────────────────────────

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

    # ───────────────────────────────────────────────────────
    # BUILD DATE RANGE PARAMS
    # ───────────────────────────────────────────────────────

    @staticmethod
    def build_date_params(start_time: str | None = None,
                          end_time: str | None = None) -> dict:
        """
        Build startTime/endTime query params.
        If not provided, defaults to last 13 months → today.
        """
        tz = pendulum.timezone("America/New_York")
        if not end_time:
            end_time = pendulum.now(tz).format("YYYY-MM-DD[T]HH:mm:ss[Z]")
        if not start_time:
            start_time = pendulum.now(tz).subtract(months=13).format("YYYY-MM-DD[T]HH:mm:ss[Z]")
        return {"startTime": start_time, "endTime": end_time}

    # ───────────────────────────────────────────────────────
    # SINGLE GET WITH RETRY
    # ───────────────────────────────────────────────────────

    def fetch_get(self, url: str, params: dict | None = None,
                  max_retries: int = 3) -> dict | None:
        for attempt in range(1, max_retries + 1):
            try:
                self._rate_limit_wait()
                headers = self._auth_headers()
                resp = requests.get(url, headers=headers,
                                    params=params, timeout=60)

                if resp.status_code == 429:
                    wait = random.randint(30, 60)
                    logger.warning(f"429 Rate limited — sleeping {wait}s")
                    time.sleep(wait)
                    continue

                if resp.status_code == 400:
                    logger.warning(f"400 Bad Request: {resp.text[:300]}")
                    return None

                if resp.status_code != 200:
                    logger.error(f"GET {url} → {resp.status_code}: {resp.text[:300]}")
                    if attempt < max_retries:
                        time.sleep(5 * attempt)
                        continue
                    return None

                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(10)
                    continue
                return None

            except Exception as e:
                logger.error(f"GET error: {e}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    continue
                return None

        return None

    # ───────────────────────────────────────────────────────
    # EXTRACT RECORDS FROM RESPONSE
    # ───────────────────────────────────────────────────────

    def _extract_records(self, body: dict | None) -> list[dict]:
        if not body:
            return []
        resp = body.get("response")
        if resp is None:
            return []
        if isinstance(resp, list):
            return [r for r in resp if isinstance(r, dict)]
        if isinstance(resp, dict):
            list_values = {
                k: v for k, v in resp.items() if isinstance(v, list)
            }
            if list_values:
                records = []
                for key, items in list_values.items():
                    for item in items:
                        if isinstance(item, dict):
                            item["_source_key"] = key
                            records.append(item)
                return records if records else [resp]
            else:
                return [resp]
        return []

    # ───────────────────────────────────────────────────────
    # PAGINATED GET — fetches ALL pages with tqdm
    # ───────────────────────────────────────────────────────

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
                body = self.fetch_get(url, params)

                if body is None:
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
                time.sleep(self.SLEEP_BETWEEN_CALLS)

        return all_records

    # ───────────────────────────────────────────────────────
    # RAW ID COLLECTION (no Polars, no transform)
    # ───────────────────────────────────────────────────────

    def collect_advertiser_ids(self, date_params: dict | None = None) -> list[int]:
        """Fetch all advertiser IDs from /traffic/advertisers (raw)."""
        url = f"{self.BASE_URL}/traffic/advertisers"
        params = dict(date_params) if date_params else {}
        raw = self.fetch_all_pages(url, params, desc="📋 Fetching advertiser IDs")
        ids = [int(r["id"]) for r in raw if "id" in r]
        logger.success(f"Collected {len(ids)} advertiser ID(s)")
        return ids

    def collect_campaign_ids(self, advertiser_ids: list[int],
                             date_params: dict | None = None) -> list[int]:
        """Fetch all campaign IDs across all advertisers (raw)."""
        all_cids = []
        url = f"{self.BASE_URL}/traffic/campaigns"

        with tqdm(total=len(advertiser_ids),
                  desc="📋 Fetching campaign IDs",
                  unit="adv") as pbar:
            for adv_id in advertiser_ids:
                params = {"accountId": adv_id}
                if date_params:
                    params.update(date_params)
                params["limit"] = self.PAGE_LIMIT

                page = 1
                while True:
                    params["page"] = page
                    body = self.fetch_get(url, params)
                    if body is None:
                        break
                    records = self._extract_records(body)
                    if not records:
                        break
                    cids = [int(r["id"]) for r in records if "id" in r]
                    all_cids.extend(cids)
                    if len(records) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP_BETWEEN_CALLS)

                pbar.update(1)
                pbar.set_postfix_str(f"campaigns={len(all_cids)}")
                time.sleep(self.SLEEP_BETWEEN_CALLS)

        logger.success(f"Collected {len(all_cids)} campaign ID(s) total")
        return all_cids

    # ───────────────────────────────────────────────────────
    # FETCH ALL RECORDS FOR ONE ENDPOINT
    # ───────────────────────────────────────────────────────

    def fetch_endpoint_records(
        self,
        endpoint: type[TrafficEndpointConfig],
        parent_ids: list[int] | None = None,
        date_params: dict | None = None,
    ) -> list[dict]:
        path = endpoint.ENDPOINT_PATH
        scope = endpoint.SCOPE
        name = endpoint.ENDPOINT_NAME
        param_name = endpoint.PARAM_NAME
        id_in_path = endpoint.ID_IN_PATH
        supports_dates = endpoint.SUPPORTS_DATE_FILTER

        all_records = []

        # Build base params (date filter if supported)
        base_params = {}
        if supports_dates and date_params:
            base_params.update(date_params)

        # ── SEAT-LEVEL ──
        if scope == "seat":
            url = f"{self.BASE_URL}{path}"
            logger.info(f"[{name}] Fetching (seat-level): {path}")

            paginated_seat_eps = {
                "advertisers", "account_groups", "exchanges",
                "user_groups", "dooh_screen_lists",
                "dooh_screen_owners", "dooh_venue_types",
            }

            if name in paginated_seat_eps:
                records = self.fetch_all_pages(
                    url, base_params, desc=f"🔄 {name}"
                )
            else:
                body = self.fetch_get(url, base_params or None)
                records = self._extract_records(body)

            all_records.extend(records)
            logger.info(f"[{name}] Got {len(records)} record(s)")

        # ── ADVERTISER-LEVEL or CAMPAIGN-LEVEL ──
        elif scope in ("advertiser", "campaign"):
            if not parent_ids:
                logger.warning(f"[{name}] No parent IDs — skipping")
                return []

            logger.info(
                f"[{name}] Fetching ({scope}-level): {path} "
                f"× {len(parent_ids)} parent(s) ({param_name})"
            )

            with tqdm(total=len(parent_ids),
                      desc=f"🔄 {name}",
                      unit=param_name) as pbar:
                for pid in parent_ids:
                    try:
                        if id_in_path:
                            actual_url = f"{self.BASE_URL}{path.replace('{id}', str(pid))}"
                            params = dict(base_params)
                            records = self.fetch_all_pages(
                                actual_url, params,
                                desc=f"    {name}/{pid}"
                            )
                        else:
                            url = f"{self.BASE_URL}{path}"
                            params = {param_name: pid}
                            params.update(base_params)
                            records = self.fetch_all_pages(
                                url, params,
                                desc=f"    {name}/{pid}"
                            )

                        for r in records:
                            r[f"_parent_{param_name}"] = str(pid)

                        all_records.extend(records)
                        pbar.update(1)
                        pbar.set_postfix_str(f"total={len(all_records)}")
                        time.sleep(self.SLEEP_BETWEEN_CALLS)

                    except Exception as e:
                        logger.error(f"    {param_name}={pid} FAILED: {e}")
                        pbar.update(1)
                        continue

            logger.info(
                f"[{name}] Total: {len(all_records)} records "
                f"across {len(parent_ids)} parent(s)"
            )

        return all_records

    # ───────────────────────────────────────────────────────
    # TRANSFORM RECORDS → POLARS (via CSV buffer)
    # ───────────────────────────────────────────────────────

    def transform_to_polars(
        self,
        records: list[dict],
        report_type_value: str,
    ) -> pl.DataFrame:
        if not records:
            return pl.DataFrame()

        # 1. Flatten each record (all values → str)
        flat_records = [flatten_json(r) for r in records]

        # 2. Collect all unique column names preserving order
        all_cols = []
        seen = set()
        for rec in flat_records:
            for k in rec.keys():
                if k not in seen:
                    all_cols.append(k)
                    seen.add(k)

        # 3. Write to CSV in memory (guarantees uniform str types)
        csv_buffer = io.StringIO()
        writer = csv.DictWriter(
            csv_buffer, fieldnames=all_cols,
            extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(flat_records)

        # 4. Read back with Polars — infer_schema_length=0 = ALL Utf8
        df = pl.read_csv(
            io.StringIO(csv_buffer.getvalue()),
            infer_schema_length=0,
            null_values=["", "null", "NULL", "None", "N/A"],
        )
        logger.info(f"Flattened → {df.shape[0]} rows × {df.shape[1]} cols")

        # 5. Add metadata
        ingestion_ts = pendulum.now("UTC").to_iso8601_string()
        ingestion_date = pendulum.now("America/New_York").format("YYYY-MM-DD")
        df = df.with_columns([
            pl.lit(ingestion_date).alias("ingestion_date"),
            pl.lit(ingestion_ts).alias("ingestion_timestamp"),
            pl.lit(report_type_value).alias("report_type"),
        ])

        # 6. Sanitize column names
        rename_map = {}
        for col in df.columns:
            safe = sanitize_bq_column_name(col)
            if safe != col:
                rename_map[col] = safe
        if rename_map:
            logger.info(f"Renaming {len(rename_map)} columns for BQ:")
            for old, new in rename_map.items():
                logger.debug(f"    '{old}' → '{new}'")
            df = df.rename(rename_map)

        logger.success(f"Transform done | {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # ───────────────────────────────────────────────────────
    # RUN SINGLE ENDPOINT
    # ───────────────────────────────────────────────────────

    def run_endpoint(
        self,
        endpoint: type[TrafficEndpointConfig],
        output_dir: str = "output",
        api_type: str = API_TYPE,
        parent_ids: list[int] | None = None,
        date_params: dict | None = None,
    ) -> Optional[dict]:
        t0 = time.time()
        name = endpoint.ENDPOINT_NAME.upper()

        try:
            logger.info("=" * 60)
            logger.info(f"▶ [{name}] Starting traffic endpoint")
            logger.info("=" * 60)

            # 1. Fetch (paginated + date filtered)
            records = self.fetch_endpoint_records(
                endpoint, parent_ids, date_params
            )

            # 2. Transform → Polars (via CSV buffer)
            report_type_value = f"{api_type}_{endpoint.ENDPOINT_NAME}"
            df = self.transform_to_polars(records, report_type_value)

            if df.is_empty():
                logger.warning(f"[{name}] 0 records — nothing to save.")
                return {"df": df, "filepath": None, "day_str": None}

            # 3. Save CSV
            day_str = pendulum.now("America/New_York").format("YYYY-MM-DD")
            epoch = int(time.time())
            group_dir = os.path.join(output_dir, endpoint.ENDPOINT_NAME, day_str)
            os.makedirs(group_dir, exist_ok=True)
            filename = f"{endpoint.ENDPOINT_NAME}_{day_str}_{epoch}.csv"
            filepath = os.path.join(group_dir, filename)
            df.write_csv(filepath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name}] DONE | {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {elapsed:.1f}s → {os.path.abspath(filepath)}"
            )
            return {"df": df, "filepath": filepath, "day_str": day_str}

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"❌ [{name}] FAILED after {elapsed:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            return None

    # ───────────────────────────────────────────────────────
    # RUN ALL ENDPOINTS (hierarchical, with tqdm)
    # ───────────────────────────────────────────────────────

    def run_all_endpoints(
        self,
        endpoints: list[type[TrafficEndpointConfig]],
        output_dir: str = "output",
        api_type: str = API_TYPE,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> dict[str, Optional[dict]]:
        """
        Run all traffic endpoints respecting hierarchy.

        Args:
            start_time: ISO format e.g. "2025-04-01T00:00:00Z"
            end_time:   ISO format e.g. "2026-05-11T23:59:59Z"
                        Defaults: last 13 months → now
        """
        results: dict[str, Optional[dict]] = {}

        # ── Date range for filtering ──
        date_params = self.build_date_params(start_time, end_time)
        logger.info(
            f"📅 Date range: {date_params['startTime']} → {date_params['endTime']}"
        )

        # ── Separate by scope ──
        seat_eps = [e for e in endpoints if e.SCOPE == "seat"]
        adv_eps  = [e for e in endpoints if e.SCOPE == "advertiser"]
        camp_eps = [e for e in endpoints if e.SCOPE == "campaign"]

        total_eps = len(seat_eps) + len(adv_eps) + len(camp_eps)

        # ══════════════════════════════════════════════════════
        # PRE-STEP: Collect all parent IDs (raw, no Polars)
        # ══════════════════════════════════════════════════════
        advertiser_ids: list[int] = []
        campaign_ids: list[int] = []

        if adv_eps or camp_eps:
            logger.info("")
            logger.info("=" * 60)
            logger.info("📋 PRE-STEP: Collecting parent IDs")
            logger.info("=" * 60)

            advertiser_ids = self.collect_advertiser_ids(date_params)

            if camp_eps and advertiser_ids:
                campaign_ids = self.collect_campaign_ids(
                    advertiser_ids, date_params
                )

        # ══════════════════════════════════════════════════════
        # RUN ALL PHASES with master tqdm
        # ══════════════════════════════════════════════════════
        with tqdm(total=total_eps,
                  desc="🚀 Traffic Pipeline",
                  unit="endpoint") as master_pbar:

            # ── PHASE 1: Seat-level ──
            if seat_eps:
                logger.info("")
                logger.info("=" * 60)
                logger.info(f"📋 PHASE 1: {len(seat_eps)} seat-level endpoint(s)")
                logger.info("=" * 60)

                for ep in seat_eps:
                    result = self.run_endpoint(
                        ep, output_dir, api_type,
                        date_params=date_params
                    )
                    results[ep.ENDPOINT_NAME] = result
                    master_pbar.update(1)
                    master_pbar.set_postfix_str(ep.ENDPOINT_NAME)
                    time.sleep(self.SLEEP_BETWEEN_CALLS)

            # ── PHASE 2: Advertiser-level ──
            if adv_eps:
                if advertiser_ids:
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info(
                        f"📋 PHASE 2: {len(adv_eps)} advertiser-level endpoint(s) "
                        f"× {len(advertiser_ids)} advertisers"
                    )
                    logger.info("=" * 60)

                    for ep in adv_eps:
                        result = self.run_endpoint(
                            ep, output_dir, api_type,
                            advertiser_ids, date_params
                        )
                        results[ep.ENDPOINT_NAME] = result
                        master_pbar.update(1)
                        master_pbar.set_postfix_str(ep.ENDPOINT_NAME)
                        time.sleep(self.SLEEP_BETWEEN_CALLS)
                else:
                    logger.warning("⚠️  No advertiser IDs — skipping Phase 2")
                    for ep in adv_eps:
                        results[ep.ENDPOINT_NAME] = None
                        master_pbar.update(1)

            # ── PHASE 3: Campaign-level ──
            if camp_eps:
                if campaign_ids:
                    logger.info("")
                    logger.info("=" * 60)
                    logger.info(
                        f"📋 PHASE 3: {len(camp_eps)} campaign-level endpoint(s) "
                        f"× {len(campaign_ids)} campaigns"
                    )
                    logger.info("=" * 60)

                    for ep in camp_eps:
                        result = self.run_endpoint(
                            ep, output_dir, api_type,
                            campaign_ids, date_params
                        )
                        results[ep.ENDPOINT_NAME] = result
                        master_pbar.update(1)
                        master_pbar.set_postfix_str(ep.ENDPOINT_NAME)
                        time.sleep(self.SLEEP_BETWEEN_CALLS)
                else:
                    logger.warning("⚠️  No campaign IDs — skipping Phase 3")
                    for ep in camp_eps:
                        results[ep.ENDPOINT_NAME] = None
                        master_pbar.update(1)

        # ══════════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════════
        ok = sum(
            1 for v in results.values()
            if v is not None
            and v.get("df") is not None
            and not v["df"].is_empty()
        )
        fail = len(results) - ok

        logger.info("")
        logger.info("=" * 60)
        logger.info(
            f"📋 TRAFFIC SUMMARY: {ok} OK, {fail} fail/empty "
            f"out of {len(results)}"
        )
        for name, res in results.items():
            if res and res.get("df") is not None and not res["df"].is_empty():
                # rows = f"{res['df'].shape>8,} rows × {res['df'].shape[1]} cols"
                rows = f"{res['df'].shape[0]:>8,} rows"
            elif res and res.get("df") is not None:
                rows = "0 rows (empty)"
            else:
                rows = "ERROR/SKIPPED"
            logger.info(f"    {name:30s} → {rows}")
        logger.info("=" * 60)

        return results