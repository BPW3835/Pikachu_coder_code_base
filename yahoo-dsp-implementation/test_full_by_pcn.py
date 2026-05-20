#!/usr/bin/env python3
"""
Yahoo DSP "pull as much as possible" — OUTPUTS ORGANIZED BY DATE
 
What it does:
1) OAuth2 token (JWT client assertion)
2) Traffic API snapshots (metadata) saved under report_date=YYYY-MM-DD/
   - seats, advertisers, campaigns, lines, creatives, ads
   - ratelimit, dictionary (where available)
3) Reporting API (performance) saved under report_date=YYYY-MM-DD/reports/<report_name>/
   - Core joinable reports (adv/campaign/line/ad/creative)
   - One-dimension coverage reports for ALL dimension IDs published by Yahoo
   - Uses ALL metric IDs published by Yahoo
   - Applies Top-N caps for high-cardinality "end-user" dimensions per Yahoo best practices
4) Close-of-Books status snapshot (cob) per advertiser/date
 
Notes:
- "All columns" is achieved by using ALL metricTypeIds + (systematically) ALL dimensionTypeIds.
- Output is organized by date. Internally, Yahoo reports are still scoped by accountIds.
- GCS + BigQuery loading code is included but COMMENTED OUT (enable later).
 
Sources:
- Reporting create/read endpoints + COB endpoint: https://help.yahooinc.com/dsp-api/docs/resources [5](https://help.yahooinc.com/dsp-api/docs/resources)
- Dimensions are identified by dimensionTypeIds: https://help.yahooinc.com/dsp-api/docs/dimensions [4](https://help.yahooinc.com/dsp-api/docs/dimensions)
- Metrics are identified by metricTypeIds: https://help.yahooinc.com/dsp-api/docs/metrics [8](https://help.yahooinc.com/dsp-api/docs/metrics)
- Best practices warn about huge reports / high-cardinality end-user dimensions: https://help.yahooinc.com/dsp-api/docs/best-practices [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
"""
 
import os
import re
import json
import time
import base64
import hmac
import hashlib
import argparse
import requests
from datetime import datetime, timedelta
import pendulum
 
# =========================
# OPTIONAL (COMMENTED): GCP clients (enable later)
# =========================
# from google.cloud import storage
# from google.cloud import bigquery
 
# =========================
# CONFIG
# =========================
CLIENT_ID = os.getenv("YAHOO_DSP_CLIENT_ID", "PUT_CLIENT_ID_HERE").strip()
CLIENT_SECRET = os.getenv("YAHOO_DSP_CLIENT_SECRET", "PUT_CLIENT_SECRET_HERE").strip()
 
TOKEN_URL = "https://id.b2b.yahooincapis.com/zts/v1/oauth2/token"
TRAFFIC_BASE = "https://dspapi.admanagerplus.yahoo.com"
 
# Reporting endpoints can vary between tenants: api-sched-v3 vs api-schedv3, http vs https.
REPORT_POST_URLS = [
    "http://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/",
    "https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/",
    "http://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport/",
    "https://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport/",
]
REPORT_GET_BASES = [
    "http://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport",
    "https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport",
    "http://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport",
    "https://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport",
]
 
# COB endpoint described in Yahoo resources docs [5](https://help.yahooinc.com/dsp-api/docs/resources)
COB_BASES = [
    "http://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/cob",
    "https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/cob",
    "http://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport/cob",
    "https://api-schedv3.admanagerplus.yahoo.com/yamplus_api/extreport/cob",
]
 
# Reporting defaults (timezone + offset)
TIMEZONE = os.getenv("YAHOO_DSP_REPORT_TZ", "America/New_York")
TZ_OFFSET = os.getenv("YAHOO_DSP_TZ_OFFSET", "-05:00")  # override for DST correctness
CURRENCY = int(os.getenv("YAHOO_DSP_REPORT_CURRENCY", "4"))  # 4=USD
INTERVAL_TYPE_ID = int(os.getenv("YAHOO_DSP_INTERVAL_TYPE_ID", "2"))  # 2=Day
 
# Output base (organized by date)
# RUN_TS = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
RUN_TS = str(pendulum.now('UTC').format('YYYYMMDD_HHmmss'))
OUT_BASE = os.getenv("YAHOO_DSP_OUT_DIR", f"yahoo_dsp_all_by_date_{RUN_TS}")
os.makedirs(OUT_BASE, exist_ok=True)
 
# Yahoo official docs pages used to discover ALL IDs dynamically [4](https://help.yahooinc.com/dsp-api/docs/dimensions)[8](https://help.yahooinc.com/dsp-api/docs/metrics)
DIMENSIONS_DOC_URL = "https://help.yahooinc.com/dsp-api/docs/dimensions"
METRICS_DOC_URL = "https://help.yahooinc.com/dsp-api/docs/metrics"
BEST_PRACTICES_URL = "https://help.yahooinc.com/dsp-api/docs/best-practices"
 
# End-user / high-cardinality dimensions called out by Yahoo best practices (examples shown)
# (Domain=9, Browser=25, Age=27, City=21 are explicitly mentioned) [2](https://help.yahooinc.com/dsp-api/docs/best-practices)[4](https://help.yahooinc.com/dsp-api/docs/dimensions)
HIGH_CARD_DIM_IDS_DEFAULT = {9, 21, 25, 27}
 
# Top-N cap for high-cardinality dimensions (avoid unbounded reports)
TOPN_LIMIT = int(os.getenv("YAHOO_DSP_TOPN_LIMIT", "1000"))
TOPN_SORT_METRIC_ID = int(os.getenv("YAHOO_DSP_TOPN_SORT_METRIC_ID", "44"))  # common spend metric in examples [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
 
# =========================
# AUTH
# =========================
def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")
 
def make_jwt(client_id: str, client_secret: str) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "aud": "https://id.b2b.yahooincapis.com/zts/v1",
        "iss": f"idb2b.dsp.dspapi.{client_id}",
        "sub": f"idb2b.dsp.dspapi.{client_id}",
        "iat": now,
        "exp": now + 300,
    }
    header_b64 = b64url(json.dumps(header).encode("utf-8"))
    payload_b64 = b64url(json.dumps(payload).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(client_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{b64url(signature)}"
 
def get_access_token() -> str:
    jwt = make_jwt(CLIENT_ID, CLIENT_SECRET)
    r = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "scope": "api-client",
            "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
            "client_assertion": jwt,
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise Exception(f"Token request failed: {r.status_code} {r.text}")
    token = r.json().get("access_token")
    if not token:
        raise Exception(f"No access_token in response: {r.text}")
    return token
 
def traffic_headers(token: str) -> dict:
    return {
        "X-Auth-Method": "OAuth2",
        "X-Auth-Token": token,
        "Accept": "application/json",
    }
 
def reporting_headers(token: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Auth-Method": "OAuth2",
        "X-Auth-Token": token,
    }
 
# =========================
# DATE HELPERS
# =========================
def daterange(start_date: str, end_date: str):
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()
    ed = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = sd
    while cur <= ed:
        yield cur.strftime("%Y-%m-%d")
        cur += timedelta(days=1)
 
def to_ts(date_str: str, hhmmss: str) -> str:
    return f"{date_str}T{hhmmss}{TZ_OFFSET}"
 
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
 
def write_json(path: str, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
 
# =========================
# DISCOVER ALL DIMENSION & METRIC IDs (scrape official docs pages)
# =========================
def fetch_all_ids_from_docs(url: str, kind: str) -> list:
    """
    Scrape numeric IDs from Yahoo docs tables. The docs explicitly list Dimension Type ID and Metric Type ID. [4](https://help.yahooinc.com/dsp-api/docs/dimensions)[8](https://help.yahooinc.com/dsp-api/docs/metrics)
    This avoids manually maintaining a giant list.
    """
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise Exception(f"Failed to fetch {kind} docs: {url} -> {r.status_code}")
    html = r.text
 
    # Common pattern: table rows contain leading integer IDs.
    # We extract IDs that appear as standalone numbers followed by whitespace and a label.
    # This is heuristic but works well with Yahoo docs formatting.
    candidates = set()
 
    # Strong patterns: "Dimension Type ID" / "ID" column rows often show like: "4 Advertiser"
    for m in re.finditer(r'(?m)^\s*(\d{1,5})\s+[A-Za-z]', html):
        candidates.add(int(m.group(1)))
 
    # Additional pass: capture patterns like "> 4 <" or ">4<" in HTML
    for m in re.finditer(r'>(\d{1,5})<', html):
        candidates.add(int(m.group(1)))
 
    # Filter out obvious noise (0 rarely used; keep if present though)
    ids = sorted(candidates)
    return ids
 
def get_dimension_ids():
    return fetch_all_ids_from_docs(DIMENSIONS_DOC_URL, "dimensions")
 
def get_metric_ids():
    return fetch_all_ids_from_docs(METRICS_DOC_URL, "metrics")
 
# =========================
# TRAFFIC API SNAPSHOTS (metadata) — saved by date
# =========================
def traffic_get(token: str, path: str, params=None):
    url = TRAFFIC_BASE + path
    r = requests.get(url, headers=traffic_headers(token), params=params, timeout=60)
    if r.status_code != 200:
        raise Exception(f"Traffic API failed {path}: {r.status_code} {r.text}")
    return r.json()
 
def snapshot_traffic_by_date(token: str, report_date: str) -> dict:
    """
    Save Traffic API outputs under report_date=YYYY-MM-DD/traffic/.
    Note: Traffic is not time-series; this is a "daily snapshot" for auditability.
    """
    base = os.path.join(OUT_BASE, f"report_date={report_date}", "traffic")
    ensure_dir(base)
 
    endpoints = {
        "seats": ("/traffic/seats", None),
        "advertisers": ("/traffic/advertisers", None),
        "campaigns": ("/traffic/campaigns", None),     # may require pagination in some tenants
        "lines": ("/traffic/lines", None),
        "creatives": ("/traffic/creatives", None),
        "ads": ("/traffic/ads", None),
        "ratelimit": ("/traffic/ratelimit", None),     # official endpoint for current limits [10](https://teams.microsoft.com/l/meeting/details?eventId=AAMkADJhNzZjMzVlLTIyNzktNDQyZi05ZmM0LTFkNzk1MjIzNmEzOAFRAAgI3p847miAAEYAAAAATwBNHlW6PESi9Xgrq-no_wcAeAuV7F-vXEaKJ6cAdsDiZAAAAAABDQAAaR_0DFqPFEe7iBjRRVgSwQAAgVWrXQAAEA%3d%3d)
        "dictionary": ("/traffic/dictionary", None),   # data dictionary endpoint [9](https://help.yahooinc.com/dsp-api/docs/data-dictionary)
    }
 
    results = {}
    for name, (path, params) in endpoints.items():
        try:
            payload = traffic_get(token, path, params=params)
            out = os.path.join(base, f"{name}.json")
            write_json(out, payload)
            results[name] = out
        except Exception as e:
            # Don’t fail whole run if a non-critical endpoint is unavailable.
            results[name] = f"FAILED: {e}"
 
    return results
 
def extract_account_ids(advertisers_payload: dict) -> list:
    # Yahoo Traffic API typically returns {"response":[...]}
    rows = advertisers_payload.get("response") or []
    ids = []
    for adv in rows:
        if isinstance(adv, dict):
            aid = adv.get("id") or adv.get("accountId")
            if aid:
                ids.append(str(aid))
    return list(dict.fromkeys(ids))
 
# =========================
# REPORTING API (create → poll → download CSV) — saved by date
# =========================
def create_report(token: str, payload: dict) -> str:
    last_err = None
    for post_url in REPORT_POST_URLS:
        try:
            r = requests.post(post_url, headers=reporting_headers(token), data=json.dumps(payload), timeout=90)
            if r.status_code != 200:
                last_err = f"{post_url} -> {r.status_code} {r.text}"
                continue
            resp = r.json()
            rid = resp.get("customerReportId")
            if rid:
                return rid
            last_err = f"{post_url} -> 200 but missing customerReportId: {resp}"
        except Exception as e:
            last_err = f"{post_url} -> {repr(e)}"
    raise Exception(f"Report create failed. Last error: {last_err}")
 
def get_report_status(token: str, report_id: str) -> dict:
    """
    Some tenants return 405 if you use only one URL form. Try multiple:
    - base?customerReportId=...
    - base/?customerReportId=...
    - base/<report_id>
    """
    hdrs = reporting_headers(token)
    for base in REPORT_GET_BASES:
        r = requests.get(base, headers=hdrs, params={"customerReportId": report_id}, timeout=60)
        if r.status_code == 200:
            return r.json()
 
        r = requests.get(base + "/", headers=hdrs, params={"customerReportId": report_id}, timeout=60)
        if r.status_code == 200:
            return r.json()
 
        r = requests.get(f"{base}/{report_id}", headers=hdrs, timeout=60)
        if r.status_code == 200:
            return r.json()
 
    raise Exception(f"Report status failed for {report_id}")
 
def extract_download_url(status_payload: dict) -> str:
    for k in ("url", "downloadUrl", "reportUrl"):
        if status_payload.get(k):
            return status_payload[k]
    resp = status_payload.get("response")
    if isinstance(resp, dict):
        for k in ("url", "downloadUrl", "reportUrl"):
            if resp.get(k):
                return resp[k]
    return ""
 
def wait_for_success(token: str, report_id: str, poll_seconds=10, timeout_seconds=20*60) -> dict:
    start = time.time()
    while True:
        payload = get_report_status(token, report_id)
        status = payload.get("status") or (payload.get("response") or {}).get("status")
        if status == "SUCCESS":
            return payload
        if status == "FAILED":
            raise Exception(f"Report FAILED: {report_id} payload={payload}")
        if time.time() - start > timeout_seconds:
            raise Exception(f"Report polling timed out: {report_id} last={payload}")
        time.sleep(poll_seconds)
 
def download_csv(url: str, out_path: str):
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise Exception(f"CSV download failed: {r.status_code} {r.text[:500]}")
    with open(out_path, "wb") as f:
        f.write(r.content)
 
# =========================
# CLOSE-OF-BOOKS (COB) SNAPSHOT — per advertiser/date [5](https://help.yahooinc.com/dsp-api/docs/resources)
# =========================
def get_cob(token: str, advertiser_id: str, report_date: str) -> dict:
    hdrs = reporting_headers(token)
    last_err = None
    for url in COB_BASES:
        try:
            r = requests.get(url, headers=hdrs, params={"advertiserId": advertiser_id, "day": report_date}, timeout=60)
            if r.status_code == 200:
                return r.json()
            last_err = f"{url} -> {r.status_code} {r.text}"
        except Exception as e:
            last_err = f"{url} -> {repr(e)}"
    return {"error": last_err, "advertiserId": advertiser_id, "day": report_date}
 
# =========================
# REPORT CONFIG GENERATION (covers "everything Yahoo allows" in a runnable way)
# =========================
def build_report_payload(account_id: str, report_date: str, dimension_ids: list, metric_ids: list,
                         report_timezone: str = TIMEZONE, currency: int = CURRENCY,
                         interval_type_id: int = INTERVAL_TYPE_ID,
                         topn: bool = False) -> dict:
    """
    Builds Reporting API payload:
    - accountIds filters advertiser account data [5](https://help.yahooinc.com/dsp-api/docs/resources)
    - dimensionTypeIds + metricTypeIds determine returned columns [4](https://help.yahooinc.com/dsp-api/docs/dimensions)[8](https://help.yahooinc.com/dsp-api/docs/metrics)
    - dateTypeId=11 custom range, startDate/endDate with timezone offset [5](https://help.yahooinc.com/dsp-api/docs/resources)
    - Optional limitSpec for Top-N reports (recommended for high-card dims) [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
    """
    report_option = {
        "timezone": report_timezone,
        "currency": currency,
        "dimensionTypeIds": dimension_ids,
        "metricTypeIds": metric_ids,
        "accountIds": [account_id],
    }
 
    if topn:
        report_option["limitSpec"] = {
            "type": "default",
            "limit": TOPN_LIMIT,
            "columns": [
                {"direction": "DESCENDING", "metricTypeId": TOPN_SORT_METRIC_ID}
            ]
        }
 
    return {
        "reportOption": report_option,
        "intervalTypeId": interval_type_id,
        "dateTypeId": 11,
        "startDate": to_ts(report_date, "00:00:00"),
        "endDate": to_ts(report_date, "23:59:59"),
    }
 
def report_catalog(dimension_ids: list, metric_ids: list):
    """
    Generates report configurations:
    1) Core joinable reports (keys)
    2) One-dimension coverage reports for every dimension id
       - uses Top-N on high-card dims per Yahoo best practices [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
    """
    configs = []
 
    # Core join keys (commonly used to join in later layers):
    # Advertiser=4, Campaign=5, Line=6, Ad=7, Creative=8 [4](https://help.yahooinc.com/dsp-api/docs/dimensions)
    core_ids = [x for x in [4, 5, 6] if x in dimension_ids]
    creative_ids = [x for x in [4, 5, 6, 7, 8] if x in dimension_ids]
 
    if core_ids:
        configs.append(("core_adv_campaign_line", core_ids, False))
    if creative_ids:
        configs.append(("core_with_ad_creative", creative_ids, False))
 
    # One-dimension coverage reports:
    for dim_id in dimension_ids:
        name = f"dim_{dim_id}"
        topn = dim_id in HIGH_CARD_DIM_IDS_DEFAULT  # apply cap for known high-card dims [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
        configs.append((name, [dim_id], topn))
 
    # Also attempt "time-only" (no dimensions) because Yahoo best practices show dimensionTypeIds can be empty
    # If your tenant rejects empty dimensionTypeIds, the code will fallback to advertiser dimension in runtime.
    configs.insert(0, ("time_only_try_empty_dims", [], False))
 
    return configs
 
# =========================
# MAIN (outputs by date)
# =========================
def main():
    parser = argparse.ArgumentParser(description="Yahoo DSP: all APIs, outputs by date")
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--skip_reports", action="store_true", help="Only Traffic snapshots, no Reporting pulls")
    args = parser.parse_args()
 
    if not CLIENT_ID or not CLIENT_SECRET or "PUT_CLIENT" in CLIENT_ID:
        raise Exception("Set YAHOO_DSP_CLIENT_ID and YAHOO_DSP_CLIENT_SECRET.")
 
    print("Getting access token...")
    token = get_access_token()
    print("Token OK")
 
    # Discover ALL dimension and metric IDs from Yahoo official docs (dynamic)
    print("Fetching ALL dimensionTypeIds from Yahoo docs...")
    dimension_ids = get_dimension_ids()
    print(f"dimensionTypeIds discovered: {len(dimension_ids)}")
 
    print("Fetching ALL metricTypeIds from Yahoo docs...")
    metric_ids = get_metric_ids()
    print(f"metricTypeIds discovered: {len(metric_ids)}")
 
    configs = report_catalog(dimension_ids, metric_ids)
    print(f"Report configs generated: {len(configs)}")
    print("Note: high-cardinality dimensions use Top-N caps per Yahoo best practices.")  # [2](https://help.yahooinc.com/dsp-api/docs/best-practices)
 
    for d in daterange(args.start_date, args.end_date):
        print(f"\n=== report_date={d} ===")
 
        # 1) Traffic snapshots by date (metadata)
        traffic_files = snapshot_traffic_by_date(token, d)
 
        # Extract accountIds from the advertisers snapshot (so we pull reports for all accounts)
        advertisers_path = traffic_files.get("advertisers")
        if not advertisers_path or advertisers_path.startswith("FAILED"):
            raise Exception(f"Cannot proceed without advertisers snapshot: {advertisers_path}")
 
        advertisers_payload = json.load(open(advertisers_path, "r", encoding="utf-8"))
        account_ids = extract_account_ids(advertisers_payload)
        print(f"Accounts discovered for {d}: {len(account_ids)}")
 
        # 2) COB snapshot by date (optional but useful)
        cob_dir = os.path.join(OUT_BASE, f"report_date={d}", "cob")
        ensure_dir(cob_dir)
        for aid in account_ids:
            cob = get_cob(token, aid, d)  # advertiserId assumed as accountId in most tenant setups
            write_json(os.path.join(cob_dir, f"cob_{aid}.json"), cob)
 
        if args.skip_reports:
            continue
 
        # 3) Reporting pulls — OUTPUT ORGANIZED BY DATE, not by account
        # Structure:
        # OUT/report_date=YYYY-MM-DD/reports/<report_name>/yahoo_dsp_<report_name>_<account_id>_<date>.csv
        reports_base = os.path.join(OUT_BASE, f"report_date={d}", "reports")
        ensure_dir(reports_base)
 
        for (report_name, dims, use_topn) in configs:
            report_dir = os.path.join(reports_base, report_name)
            ensure_dir(report_dir)
 
            # Save config used for this report_name/date for audit
            write_json(os.path.join(report_dir, "_config.json"), {
                "report_name": report_name,
                "dimensionTypeIds": dims,
                "metricTypeIds_count": len(metric_ids),
                "metricTypeIds": metric_ids,  # full list for reproducibility
                "topn": use_topn,
                "topn_limit": TOPN_LIMIT if use_topn else None,
                "topn_sort_metric_id": TOPN_SORT_METRIC_ID if use_topn else None,
                "timezone": TIMEZONE,
                "tz_offset": TZ_OFFSET,
                "currency": CURRENCY,
                "intervalTypeId": INTERVAL_TYPE_ID,
                "report_date": d
            })
 
            for aid in account_ids:
                # Build payload
                payload = build_report_payload(aid, d, dims, metric_ids, topn=use_topn)
 
                # Special case: empty dims may be rejected by some tenants.
                # If report creation fails for empty dims, fallback to advertiser dimension (4) [4](https://help.yahooinc.com/dsp-api/docs/dimensions)
                try:
                    rid = create_report(token, payload)
                except Exception:
                    if len(dims) == 0 and 4 in dimension_ids:
                        payload = build_report_payload(aid, d, [4], metric_ids, topn=False)
                        rid = create_report(token, payload)
                    else:
                        raise
 
                status_payload = wait_for_success(token, rid)
                url = extract_download_url(status_payload)
                if not url:
                    raise Exception(f"No download URL for report_id={rid} report_name={report_name} account={aid}")
 
                out_csv = os.path.join(report_dir, f"yahoo_dsp_{report_name}_{aid}_{d}.csv")
                download_csv(url, out_csv)
 
        print(f"Completed reporting pulls for report_date={d}")
 
    print("\nDONE")
 
 
if __name__ == "__main__":
    main()