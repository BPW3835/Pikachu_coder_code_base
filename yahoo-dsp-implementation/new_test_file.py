import os
import json
import time
import base64
import hmac
import hashlib
import argparse
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from threading import Lock
 
# =========================
# CONFIG (edit only if needed)
# =========================
CLIENT_ID = os.getenv("YAHOO_DSP_CLIENT_ID", "PUT_CLIENT_ID")
CLIENT_SECRET = os.getenv("YAHOO_DSP_CLIENT_SECRET", "PUT_CLIENT_SECRET")
 
TOKEN_URL = "https://id.b2b.yahooincapis.com/zts/v1/oauth2/token"
TRAFFIC_BASE = "https://dspapi.admanagerplus.yahoo.com"
 
REPORT_POST_URL = "http://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/"
REPORT_STATUS_URL_BASE = "http://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport"
 
TIMEZONE = os.getenv("YAHOO_DSP_TIMEZONE", "America/New_York")
TZ_OFFSET = os.getenv("YAHOO_DSP_TZ_OFFSET", "-05:00")  # override if needed (DST-safe)
 
# ✅ STRICT CONFIG (your requirement-driven selection)
DIMENSIONS = [4, 5, 6, 7, 8, 19, 21, 25]
METRICS = [1, 2, 44, 11, 23, 25, 26, 27, 28, 29]
 
# Output
OUT_BASE = os.getenv("YAHOO_DSP_OUT_DIR", "output")
 
# =========================
# RATE LIMITER (Reporting POST)
# Keep below vendor quota to reduce throttling risk. [1](https://help.yahooinc.com/dsp-api/docs/limits)
# =========================
POSTS_PER_MIN = int(os.getenv("YAHOO_DSP_POSTS_PER_MIN", "18"))
POST_WINDOW_SEC = 60
_post_times = deque()
_post_lock = Lock()
 
def throttle_post():
    while True:
        with _post_lock:
            now = time.time()
            while _post_times and now - _post_times[0] > POST_WINDOW_SEC:
                _post_times.popleft()
            if len(_post_times) < POSTS_PER_MIN:
                _post_times.append(now)
                return
        time.sleep(0.25)
 
# =========================
# AUTH
# =========================
def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")
 
def get_token() -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {
        "aud": "https://id.b2b.yahooincapis.com/zts/v1",
        "iss": f"idb2b.dsp.dspapi.{CLIENT_ID}",
        "sub": f"idb2b.dsp.dspapi.{CLIENT_ID}",
        "iat": now,
        "exp": now + 300,
    }
 
    signing_input = f"{b64url(json.dumps(header).encode())}.{b64url(json.dumps(payload).encode())}"
    sig = hmac.new(CLIENT_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    jwt = f"{signing_input}.{b64url(sig)}"
 
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
    r.raise_for_status()
    return r.json()["access_token"]
 
def headers(token: str) -> dict:
    return {"Content-Type": "application/json", "X-Auth-Method": "OAuth2", "X-Auth-Token": token}
 
def traffic_headers(token: str) -> dict:
    return {"X-Auth-Method": "OAuth2", "X-Auth-Token": token}
 
# =========================
# DATE HELPERS
# =========================
def ts(date_str: str, hhmmss: str) -> str:
    return f"{date_str}T{hhmmss}{TZ_OFFSET}"
 
def date_chunks(start_date: str, end_date: str, chunk_days: int):
    s = datetime.strptime(start_date, "%Y-%m-%d").date()
    e = datetime.strptime(end_date, "%Y-%m-%d").date()
    cur = s
    while cur <= e:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), e)
        yield cur.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        cur = chunk_end + timedelta(days=1)
 
# =========================
# TRAFFIC API (Advertisers with pagination)
# Works for 100s/1000s of IDs.
# =========================
def traffic_get(token: str, path: str, params=None):
    url = f"{TRAFFIC_BASE}{path}"
    r = requests.get(url, headers=traffic_headers(token), params=params, timeout=60)
    r.raise_for_status()
    return r.json()
 
def get_all_advertisers(token: str, page_limit: int = 200, max_pages: int = 10000):
    all_rows = []
    for page in range(1, max_pages + 1):
        payload = traffic_get(token, "/traffic/advertisers", params={"page": page, "limit": page_limit})
        rows = payload.get("response", [])
        if not rows:
            break
        all_rows.extend(rows)
    return all_rows
 
def extract_account_ids(advertisers: list):
    ids = []
    for a in advertisers:
        if isinstance(a, dict) and "id" in a:
            ids.append(str(a["id"]))
    # de-dupe preserve order
    return list(dict.fromkeys(ids))
 
# =========================
# REPORTING API
# =========================
def create_report(token: str, account_id: str, start_date: str, end_date: str) -> str:
    payload = {
        "reportOption": {
            "timezone": TIMEZONE,
            "currency": 4,
            "dimensionTypeIds": DIMENSIONS,
            "metricTypeIds": METRICS,
            "accountIds": [account_id],
        },
        "intervalTypeId": 2,  # daily rows within range
        "dateTypeId": 11,     # custom range
        "startDate": ts(start_date, "00:00:00"),
        "endDate": ts(end_date, "23:59:59"),
    }
 
    throttle_post()
    r = requests.post(REPORT_POST_URL, headers=headers(token), json=payload, timeout=90)
    if r.status_code != 200:
        raise Exception(f"Create report failed: {r.status_code} {r.text}")
    return r.json()["customerReportId"]
 
def get_status(token: str, report_id: str) -> dict:
    url = f"{REPORT_STATUS_URL_BASE}/{report_id}"
    r = requests.get(url, headers=headers(token), timeout=60)
    if r.status_code != 200:
        raise Exception(f"Status failed: {r.status_code} {r.text}")
    return r.json()
 
def wait_report(token: str, report_id: str, max_attempts: int = 180, sleep_sec: int = 5):
    for attempt in range(1, max_attempts + 1):
        res = get_status(token, report_id)
        status = res.get("status") or res.get("response", {}).get("status")
        status_clean = str(status).upper()
 
        print(f"    ⏳ status={status} (attempt {attempt}/{max_attempts})")
 
        if status_clean == "SUCCESS":
            return res
        if status_clean == "FAILED":
            raise Exception(f"Report FAILED: {report_id}")
 
        time.sleep(sleep_sec)
 
    raise Exception(f"Timeout waiting for report: {report_id}")
 
def extract_url(payload: dict):
    for k in ("url", "downloadUrl", "reportUrl"):
        if payload.get(k):
            return payload[k]
    resp = payload.get("response")
    if isinstance(resp, dict):
        for k in ("url", "downloadUrl", "reportUrl"):
            if resp.get(k):
                return resp[k]
    return None
 
def download_csv(url: str, out_path: str):
    r = requests.get(url, timeout=300)
    if r.status_code != 200:
        raise Exception(f"Download failed: {r.status_code} {r.text[:400]}")
    with open(out_path, "wb") as f:
        f.write(r.content)
 
# =========================
# WORK UNIT
# =========================
def run_one(token: str, out_dir: str, account_id: str, start_date: str, end_date: str, resume: bool):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"yahoo_{account_id}_{start_date}_to_{end_date}.csv")
 
    # Resume mode: skip if already downloaded
    if resume and os.path.exists(out_file) and os.path.getsize(out_file) > 0:
        return {"account_id": account_id, "range": f"{start_date}..{end_date}", "skipped": True, "file": out_file}
 
    rid = create_report(token, account_id, start_date, end_date)
    wait_report(token, rid)
 
    # Fetch again to ensure URL is present (tenant behavior)
    final_status = get_status(token, rid)
    url = extract_url(final_status)
    if not url:
        raise Exception(f"Download URL missing for report_id={rid}")
 
    print("    🔗 downloading...")
    download_csv(url, out_file)
    print(f"    ✅ saved {out_file}")
 
    # =========================
    # GCP (COMMENTED) – enable later
    # Your RAW design is append-only + date-partitioned.
    # =========================
    # from google.cloud import storage
    # client = storage.Client()
    # bucket = client.bucket("YOUR_BUCKET")
    # blob = bucket.blob(f"yahoo_dsp/report_date={start_date}/yahoo_{account_id}_{start_date}_to_{end_date}.csv")
    # blob.upload_from_filename(out_file)
 
    return {"account_id": account_id, "range": f"{start_date}..{end_date}", "skipped": False, "file": out_file}
 
# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end_date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--chunk_days", type=int, default=7, help="Days per report (default 7)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument("--resume", action="store_true", help="Skip already-downloaded files")
    parser.add_argument("--max_accounts", type=int, default=0, help="For testing: limit accounts (0 = all)")
    args = parser.parse_args()
 
    token = get_token()
    print("✅ Token OK")
 
    advertisers = get_all_advertisers(token)
    account_ids = extract_account_ids(advertisers)
    if args.max_accounts and args.max_accounts > 0:
        account_ids = account_ids[:args.max_accounts]
 
    print(f"✅ Accounts discovered: {len(account_ids)}")
 
    for start_d, end_d in date_chunks(args.start_date, args.end_date, args.chunk_days):
        print(f"\n📅 Chunk: {start_d} → {end_d}")
        out_dir = os.path.join(OUT_BASE, f"report_range={start_d}_to_{end_d}")
 
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one, token, out_dir, aid, start_d, end_d, args.resume): aid
                for aid in account_ids
            }
 
            for fut in as_completed(futures):
                aid = futures[fut]
                try:
                    fut.result()
                except Exception as e:
                    print(f"  ❌ account {aid} failed for {start_d}..{end_d}: {e}")
 
    print("\n✅ DONE")
 
if __name__ == "__main__":
    main()