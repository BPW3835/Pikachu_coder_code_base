"""
dsp_traffic.py
──────────────
Yahoo DSP Traffic API — production-ready.

Hierarchy:
    Seat (no params) → Advertiser (accountId) → Campaign (orderId) → Line (lineId)

Key design decisions (learned from POC):
    • NO date filters — Traffic API is metadata/snapshots, NOT time-series.
      POC confirmed: without date filters → 17 advertisers (ALL).
      With date filters → only 9 (recently modified). We want ALL.
    • IDs collected from raw JSON BEFORE any Polars transforms.
      If transform fails on one endpoint, downstream IDs are still intact.
    • Transform uses csv.DictWriter → pl.read_csv(infer_schema_length=0)
      which guarantees ALL columns are Utf8 (no type clashes).

Pipeline per endpoint:
    1. Paginated GET (rate-limited to 55/min, retry on 5xx/timeout)
    2. Flatten nested JSON → CSV buffer → Polars DataFrame (all STRING)
    3. Add metadata: ingestion_date, ingestion_timestamp, report_type
    4. Sanitize column names for BQ
    5. Save CSV

Returns same dict format as reporting.py:
    {"df": pl.DataFrame, "filepath": str, "day_str": str}

GCS / BQ uploads are handled by main.py via helpers.py.
"""

import io
import csv
import os
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
TZ = "America/New_York"


# ═══════════════════════════════════════════════════════════════
# ENDPOINT CONFIG — dynamic from metrics.py
# ═══════════════════════════════════════════════════════════════

class TrafficEndpointConfig:
    """Base class for a Traffic API endpoint descriptor."""
    ENDPOINT_NAME: str = "base"
    ENDPOINT_PATH: str = "/traffic/base"
    SCOPE: str = "seat"        # seat | advertiser | campaign | line
    PARAM_NAME: str = ""       # accountId | orderId | lineId


def _build_registry() -> dict[str, type[TrafficEndpointConfig]]:
    """Build endpoint classes from TRAFFIC_API_ENDPOINT_CONFIGS."""
    reg = {}
    for name, cfg in TRAFFIC_API_ENDPOINT_CONFIGS.items():
        cls = type(
            "".join(w.capitalize() for w in name.split("_")) + "Ep",
            (TrafficEndpointConfig,),
            {
                "ENDPOINT_NAME": name,
                "ENDPOINT_PATH": cfg["path"],
                "SCOPE":         cfg.get("scope", "seat"),
                "PARAM_NAME":    cfg.get("param_name", ""),
            },
        )
        reg[name] = cls
    return reg


ENDPOINT_REGISTRY = _build_registry()
ALL_TRAFFIC_ENDPOINTS = list(ENDPOINT_REGISTRY.values())


# ═══════════════════════════════════════════════════════════════
# YAHOO TRAFFIC CLASS
# ═══════════════════════════════════════════════════════════════

class YahooTraffic:
    """
    End-to-end Yahoo DSP Traffic API handler.

    Phases:
        PRE-STEP — Collect all parent IDs (raw JSON, no Polars)
        PHASE 1  — Seat-level endpoints (no params)
        PHASE 2  — Advertiser-level endpoints (per accountId)
        PHASE 3  — Campaign-level endpoints (per orderId)
        PHASE 4  — Line-level endpoints (per lineId)
    """

    BASE = "https://dspapi.admanagerplus.yahoo.com"
    RATE_LIMIT = 55      # max GET/min (under API quota of 60)
    SLEEP = 1.2          # seconds between consecutive calls
    PAGE_LIMIT = 100     # max records per page (API max)

    def __init__(self):
        self.helper = YahooHelpers()
        self._timestamps: list[float] = []
        logger.success("YahooTraffic initialised.")

    # ───────────────────────────────────────────────────────
    # AUTH HEADERS
    # ───────────────────────────────────────────────────────

    def _headers(self) -> dict:
        token = self.helper.get_access_token()
        return self.helper.build_auth_headers(token)

    # ───────────────────────────────────────────────────────
    # RATE LIMITER
    # ───────────────────────────────────────────────────────

    def _rate_wait(self):
        """Block if we're approaching the 60 GET/min limit."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.RATE_LIMIT:
            wait = 60 - (now - self._timestamps[0]) + 1
            if wait > 0:
                logger.info(f"⏱️  Rate limit: waiting {wait:.0f}s")
                time.sleep(wait)
        self._timestamps.append(time.time())

    # ───────────────────────────────────────────────────────
    # SINGLE GET — with retry
    # ───────────────────────────────────────────────────────

    def _get(self, url: str, params: dict | None = None,
             retries: int = 3) -> dict | None:
        """
        Single GET with rate limiting and retry.
        - 400/403 → return None immediately (bad param / no access)
        - 500     → return None immediately (server error)
        - 429     → sleep and retry
        - timeout → retry
        """
        for attempt in range(1, retries + 1):
            try:
                self._rate_wait()
                resp = requests.get(
                    url, headers=self._headers(),
                    params=params, timeout=60,
                )

                if resp.status_code == 429:
                    wait = random.randint(30, 60)
                    logger.warning(f"429 Rate limited — sleep {wait}s")
                    time.sleep(wait)
                    continue

                # Immediate bail on known-bad statuses
                if resp.status_code in (400, 403):
                    logger.warning(f"{resp.status_code}: {resp.text[:250]}")
                    return None

                if resp.status_code == 500:
                    logger.error(f"500: {resp.text[:250]}")
                    return None

                if resp.status_code != 200:
                    logger.error(f"{resp.status_code}: {resp.text[:250]}")
                    if attempt < retries:
                        time.sleep(5 * attempt)
                        continue
                    return None

                return resp.json()

            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(10)
                    continue
                return None

            except Exception as e:
                logger.error(f"GET error: {e}")
                if attempt < retries:
                    time.sleep(5 * attempt)
                    continue
                return None

        return None

    # ───────────────────────────────────────────────────────
    # EXTRACT RECORDS FROM RESPONSE JSON
    # ───────────────────────────────────────────────────────

    @staticmethod
    def _extract(body: dict | None) -> list[dict]:
        """
        Extract list of records from Traffic API JSON response.
        Handles:
          {"response": [...]}           → list of dicts
          {"response": {...}}           → single dict or nested lists
        """
        if not body:
            return []
        resp = body.get("response")
        if resp is None:
            return []

        if isinstance(resp, list):
            return [r for r in resp if isinstance(r, dict)]

        if isinstance(resp, dict):
            # Check for nested lists (e.g., seats, dictionary)
            lists = {
                k: v for k, v in resp.items()
                if isinstance(v, list)
            }
            if lists:
                out = []
                for key, items in lists.items():
                    for item in items:
                        if isinstance(item, dict):
                            item["_source_key"] = key
                            out.append(item)
                return out if out else [resp]
            return [resp]

        return []

    # ───────────────────────────────────────────────────────
    # PAGINATED FETCH — all pages with tqdm
    # ───────────────────────────────────────────────────────

    def _fetch_pages(self, url: str,
                     params: dict | None = None,
                     desc: str = "Fetching") -> list[dict]:
        """Paginate through a list endpoint. Returns all records."""
        params = dict(params) if params else {}
        params["limit"] = self.PAGE_LIMIT
        page = 1
        all_recs: list[dict] = []

        with tqdm(
            desc=desc, unit="rec", dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt} rec [{elapsed}]"
        ) as pbar:
            while True:
                params["page"] = page
                body = self._get(url, params)
                if body is None:
                    break

                recs = self._extract(body)
                if not recs:
                    break

                all_recs.extend(recs)
                pbar.update(len(recs))
                pbar.set_postfix_str(f"page={page}")

                if len(recs) < self.PAGE_LIMIT:
                    break

                page += 1
                time.sleep(self.SLEEP)

        return all_recs

    # ═══════════════════════════════════════════════════════
    # RAW ID COLLECTORS — no Polars, no transform
    # ═══════════════════════════════════════════════════════

    def _collect_ids(self, url: str,
                     id_field: str,
                     desc: str) -> list[int]:
        """Fetch all records from a seat-level endpoint and extract IDs."""
        recs = self._fetch_pages(url, params=None, desc=desc)
        ids = [int(r[id_field]) for r in recs if id_field in r]
        logger.success(f"  → {len(ids)} ID(s): {ids}")
        return ids

    def _collect_child_ids(
        self,
        url: str,
        param_name: str,
        parent_ids: list[int],
        id_field: str,
        desc: str,
    ) -> list[int]:
        """For each parent ID, paginate and collect all child IDs."""
        all_ids: list[int] = []
        with tqdm(
            total=len(parent_ids), desc=desc, unit="parent"
        ) as pbar:
            for pid in parent_ids:
                p: dict = {param_name: pid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    p["page"] = page
                    body = self._get(url, p)
                    if not body:
                        break
                    recs = self._extract(body)
                    if not recs:
                        break
                    ids = [
                        int(r[id_field])
                        for r in recs if id_field in r
                    ]
                    all_ids.extend(ids)
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP)
                pbar.update(1)
                pbar.set_postfix_str(f"total={len(all_ids)}")
                time.sleep(self.SLEEP)

        logger.success(f"  → {len(all_ids)} ID(s)")
        return all_ids

    # ═══════════════════════════════════════════════════════
    # FETCH ALL RECORDS FOR ONE ENDPOINT
    # ═══════════════════════════════════════════════════════

    def _fetch_endpoint(
        self,
        ep: type[TrafficEndpointConfig],
        parent_ids: list[int] | None,
    ) -> list[dict]:
        """
        Fetch all records for a single endpoint.
        Seat-level: single paginated call (NO params, NO date filter).
        Child-level: iterate over parent_ids, paginate each.
        """
        name = ep.ENDPOINT_NAME
        path = ep.ENDPOINT_PATH
        scope = ep.SCOPE
        param = ep.PARAM_NAME

        # ── SEAT-LEVEL ──
        if scope == "seat":
            url = f"{self.BASE}{path}"
            # Endpoints that return lists and support pagination
            paginated = {
                "advertisers", "account_groups",
                "dooh_screen_owners", "dooh_venue_types",
                "dooh_screen_lists",
            }
            if name in paginated:
                return self._fetch_pages(url, desc=f"🔄 {name}")
            else:
                # Simple single GET (seats, data_dictionary)
                body = self._get(url)
                recs = self._extract(body)
                logger.info(f"[{name}] {len(recs)} record(s)")
                return recs

        # ── CHILD-LEVEL (advertiser / campaign / line) ──
        if not parent_ids:
            logger.warning(f"[{name}] No parent IDs — skip")
            return []

        all_recs: list[dict] = []
        url = f"{self.BASE}{path}"

        with tqdm(
            total=len(parent_ids), desc=f"🔄 {name}", unit=param
        ) as pbar:
            for pid in parent_ids:
                p: dict = {param: pid, "limit": self.PAGE_LIMIT}
                page = 1
                while True:
                    p["page"] = page
                    body = self._get(url, p)
                    if not body:
                        break
                    recs = self._extract(body)
                    if not recs:
                        break
                    # Tag each record with its parent ID
                    for r in recs:
                        r[f"_parent_{param}"] = str(pid)
                    all_recs.extend(recs)
                    if len(recs) < self.PAGE_LIMIT:
                        break
                    page += 1
                    time.sleep(self.SLEEP)
                pbar.update(1)
                pbar.set_postfix_str(f"total={len(all_recs)}")
                time.sleep(self.SLEEP)

        logger.info(
            f"[{name}] {len(all_recs)} record(s) "
            f"across {len(parent_ids)} parent(s)"
        )
        return all_recs

    # ═══════════════════════════════════════════════════════
    # TRANSFORM → POLARS (via CSV buffer, guaranteed Utf8)
    # ═══════════════════════════════════════════════════════

    def _to_polars(
        self, records: list[dict], report_type: str
    ) -> pl.DataFrame:
        """
        Flatten JSON records → CSV in memory → Polars DataFrame.
        ALL columns are Utf8 (STRING in BQ). No type clashes.
        """
        if not records:
            return pl.DataFrame()

        # 1. Flatten each record
        flat = [flatten_json(r) for r in records]

        # 2. Collect all unique column names preserving insertion order
        cols: list[str] = []
        seen: set[str] = set()
        for rec in flat:
            for k in rec:
                if k not in seen:
                    cols.append(k)
                    seen.add(k)

        # 3. Write to CSV in memory (guarantees uniform str types)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf, fieldnames=cols,
            extrasaction="ignore", restval=""
        )
        writer.writeheader()
        writer.writerows(flat)

        # 4. Read back with Polars — infer_schema_length=0 = ALL Utf8
        df = pl.read_csv(
            io.StringIO(buf.getvalue()),
            infer_schema_length=0,
            null_values=["", "null", "NULL", "None", "N/A"],
        )

        # 5. Add metadata columns
        now_et = pendulum.now(TZ)
        now_utc = pendulum.now("UTC")
        df = df.with_columns([
            pl.lit(now_et.format("YYYY-MM-DD")).alias("ingestion_date"),
            pl.lit(now_utc.to_iso8601_string()).alias("ingestion_timestamp"),
            pl.lit(report_type).alias("report_type"),
        ])

        # 6. Sanitize column names for BigQuery
        renames: dict[str, str] = {}
        for c in df.columns:
            safe = sanitize_bq_column_name(c)
            if safe != c:
                renames[c] = safe
        if renames:
            df = df.rename(renames)

        logger.info(f"  → {df.shape[0]} rows × {df.shape[1]} cols")
        return df

    # ═══════════════════════════════════════════════════════
    # RUN SINGLE ENDPOINT (fetch + transform + save CSV)
    # ═══════════════════════════════════════════════════════

    def _run_one(
        self,
        ep: type[TrafficEndpointConfig],
        parent_ids: list[int] | None,
        output_dir: str,
    ) -> Optional[dict]:
        """Execute full pipeline for one endpoint."""
        t0 = time.time()
        name = ep.ENDPOINT_NAME

        try:
            logger.info("=" * 50)
            logger.info(f"▶ [{name.upper()}]")
            logger.info("=" * 50)

            # 1. Fetch all records
            recs = self._fetch_endpoint(ep, parent_ids)

            # 2. Transform → Polars
            report_type = f"{API_TYPE}_{name}"
            df = self._to_polars(recs, report_type)

            if df.is_empty():
                logger.warning(f"[{name}] 0 records — skip save")
                return {"df": df, "filepath": None, "day_str": None}

            # 3. Save CSV
            day = pendulum.now(TZ).format("YYYY-MM-DD")
            ep_dir = os.path.join(output_dir, name, day)
            os.makedirs(ep_dir, exist_ok=True)
            fname = f"{name}_{day}_{int(time.time())}.csv"
            filepath = os.path.join(ep_dir, fname)
            df.write_csv(filepath)

            elapsed = time.time() - t0
            logger.success(
                f"✅ [{name}] {df.shape[0]} rows × {df.shape[1]} cols "
                f"| {elapsed:.1f}s → {filepath}"
            )
            return {"df": df, "filepath": filepath, "day_str": day}

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(
                f"❌ [{name}] {elapsed:.1f}s: "
                f"{e}\n{traceback.format_exc()}"
            )
            return None

    # ═══════════════════════════════════════════════════════
    # RUN ALL — hierarchical pipeline
    # ═══════════════════════════════════════════════════════

    def run_all(
        self,
        endpoints: list[type[TrafficEndpointConfig]],
        output_dir: str,
    ) -> dict[str, Optional[dict]]:
        """
        Run all traffic endpoints respecting the hierarchy.

        Traffic API = metadata snapshots. NO date filters.
        Always fetches the full current state of all entities.
        """
        results: dict[str, Optional[dict]] = {}

        # ── Bucket endpoints by scope ──
        seat_eps = [e for e in endpoints if e.SCOPE == "seat"]
        adv_eps  = [e for e in endpoints if e.SCOPE == "advertiser"]
        camp_eps = [e for e in endpoints if e.SCOPE == "campaign"]
        line_eps = [e for e in endpoints if e.SCOPE == "line"]

        total = len(seat_eps) + len(adv_eps) + len(camp_eps) + len(line_eps)

        # ══════════════════════════════════════════════════
        # PRE-STEP: Collect all parent IDs (raw, no Polars)
        # NO date filters — get ALL entities
        # ══════════════════════════════════════════════════
        adv_ids:  list[int] = []
        camp_ids: list[int] = []
        line_ids: list[int] = []

        if adv_eps or camp_eps or line_eps:
            logger.info("")
            logger.info("=" * 50)
            logger.info("📋 PRE-STEP: Collecting advertiser IDs")
            logger.info("=" * 50)
            adv_ids = self._collect_ids(
                url=f"{self.BASE}/traffic/advertisers",
                id_field="id",
                desc="📋 Advertiser IDs",
            )

        if (camp_eps or line_eps) and adv_ids:
            logger.info("")
            logger.info("=" * 50)
            logger.info(
                f"📋 PRE-STEP: Collecting campaign IDs "
                f"({len(adv_ids)} advertisers)"
            )
            logger.info("=" * 50)
            camp_ids = self._collect_child_ids(
                url=f"{self.BASE}/traffic/campaigns",
                param_name="accountId",
                parent_ids=adv_ids,
                id_field="id",
                desc="📋 Campaign IDs",
            )

        if line_eps and camp_ids:
            logger.info("")
            logger.info("=" * 50)
            logger.info(
                f"📋 PRE-STEP: Collecting line IDs "
                f"({len(camp_ids)} campaigns)"
            )
            logger.info("=" * 50)
            line_ids = self._collect_child_ids(
                url=f"{self.BASE}/traffic/lines",
                param_name="orderId",
                parent_ids=camp_ids,
                id_field="id",
                desc="📋 Line IDs",
            )

        # ══════════════════════════════════════════════════
        # RUN ALL PHASES
        # ══════════════════════════════════════════════════
        phases = [
            ("SEAT",       seat_eps, None),
            ("ADVERTISER", adv_eps,  adv_ids),
            ("CAMPAIGN",   camp_eps, camp_ids),
            ("LINE",       line_eps, line_ids),
        ]

        with tqdm(
            total=total, desc="🚀 Traffic Pipeline", unit="ep"
        ) as master:
            for phase_name, eps, pids in phases:
                if not eps:
                    continue

                pid_count = len(pids) if pids else 0
                logger.info(f"\n{'=' * 50}")
                logger.info(
                    f"📋 {phase_name}: {len(eps)} endpoint(s)"
                    + (f" × {pid_count} parent(s)" if pids else "")
                )
                logger.info("=" * 50)

                # Skip child phases if no parent IDs
                if phase_name != "SEAT" and not pids:
                    logger.warning(
                        f"⚠️  No parent IDs — skipping {phase_name}"
                    )
                    for ep in eps:
                        results[ep.ENDPOINT_NAME] = None
                        master.update(1)
                    continue

                for ep in eps:
                    res = self._run_one(ep, pids, output_dir)
                    results[ep.ENDPOINT_NAME] = res
                    master.update(1)
                    master.set_postfix_str(ep.ENDPOINT_NAME)
                    time.sleep(self.SLEEP)

        # ══════════════════════════════════════════════════
        # SUMMARY
        # ══════════════════════════════════════════════════
        ok = sum(
            1 for v in results.values()
            if v and v.get("df") is not None and not v["df"].is_empty()
        )
        fail = len(results) - ok

        logger.info(f"\n{'=' * 50}")
        logger.info(
            f"📋 TRAFFIC SUMMARY: {ok} OK, {fail} fail/empty "
            f"out of {len(results)}"
        )
        for ep_name, res in results.items():
            if res and res.get("df") is not None and not res["df"].is_empty():
                info = (
                    f"{res['df'].shape[0]:>8,} rows "
                    f"× {res['df'].shape[1]} cols"
                )
            elif res and res.get("df") is not None:
                info = "0 rows (empty)"
            else:
                info = "ERROR/SKIPPED"
            logger.info(f"    {ep_name:25s} → {info}")
        logger.info("=" * 50)

        return results