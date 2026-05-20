"""
metrics.py
──────────
Central registry for all Yahoo DSP API dimensions, metrics, and report configs.

Naming convention:
    {API_NAME}_{ENTITY}
    e.g. REPORTING_API_DIMENSION_CSV_COLUMNS
         REPORTING_API_METRIC_NAMES
         REPORTING_API_INTEGER_METRIC_IDS
         REPORTING_API_REPORT_GROUP_CONFIGS

When new APIs are added (Traffic API, Audience API, etc.),
follow the same pattern:
    TRAFFIC_API_...
    AUDIENCE_API_...
"""

# ═══════════════════════════════════════════════════════════════
# REPORTING API — DIMENSION TYPE ID → CSV COLUMN MAPPING
# ═══════════════════════════════════════════════════════════════

REPORTING_API_DIMENSION_CSV_COLUMNS: dict[int, list[tuple[str, str]]] = {
    3:    [("Creative Size Id", "STRING"), ("Creative Size", "STRING")],
    4:    [("Advertiser Id", "STRING"), ("Advertiser", "STRING")],
    5:    [("Campaign Id", "STRING"), ("Campaign Name", "STRING")],
    6:    [("Line Id", "STRING"), ("Line", "STRING")],
    7:    [("Ad Id", "STRING"), ("Ad", "STRING")],
    8:    [("Creative Id", "STRING"), ("Creative", "STRING")],
    9:    [("Domain Name", "STRING")],
    19:   [("Country Id", "STRING"), ("Country", "STRING")],
    20:   [("Region Id", "STRING"), ("Region", "STRING")],
    23:   [("Device Category Id", "STRING"), ("Device Category", "STRING")],
    25:   [("Browser Type ID", "STRING"), ("Browser Type Name", "STRING")],
    27:   [("Age ID", "STRING"), ("Age", "STRING")],
    28:   [("Gender ID", "STRING"), ("Gender", "STRING")],
    34:   [("Device Type Id", "STRING"), ("Device Type", "STRING")],
    46:   [("Beacon ID", "STRING"), ("Beacon", "STRING")],
    50:   [("Deal ID", "STRING"), ("Deal", "STRING")],
    54:   [("Beacon Type Id", "STRING")],
    55:   [("App Id", "STRING"), ("App Name", "STRING")],
    56:   [("Seat Id", "STRING"), ("Seat", "STRING")],
    63:   [("Media Type Id", "STRING"), ("Media Type", "STRING")],
    64:   [("Advertiser Group Id", "STRING"), ("Advertiser Group", "STRING")],
    106:  [("Inventory Type", "STRING")],
    122:  [("Mobile OS ID", "STRING"), ("Mobile OS", "STRING")],
    126:  [("Connection Type", "STRING")],
    147:  [("Days To Conversion", "STRING")],
    149:  [("Package Id", "STRING"), ("Package", "STRING")],
    153:  [("Connected TV Device", "STRING")],
    154:  [("Programmatic Guaranteed Impression", "STRING"), ("Auction Type", "STRING")],
    166:  [("Channel Type Id", "STRING")],
    167:  [("Video Content Length ID", "STRING"), ("Video Content Length", "STRING")],
    169:  [("Video Ad Placement ID", "STRING"), ("Video Ad Placement", "STRING")],
    171:  [("Ads.txt & App-ads.txt Declaration", "STRING")],
    174:  [("Ad Format Id", "STRING"), ("Ad Format", "STRING")],
    182:  [("Video Player Size", "STRING")],
    1028: [("Connected TV Make", "STRING")],
    1029: [("Connected TV Model", "STRING")],
    1033: [("Deal Type Description", "STRING")],
    1037: [("Video Creative Duration", "STRING")],
    1225: [("Line Channel Type", "STRING")],
    1226: [("Line Channel Subtype", "STRING")],
    1228: [("Exchange Id", "STRING"), ("Exchange", "STRING")],
    1240: [("Video Ad Placement (IAB) ID", "STRING"), ("Video Ad Placement IAB", "STRING")],
    1267: [("Campaign Client status", "STRING")],
}


# ═══════════════════════════════════════════════════════════════
# REPORTING API — METRIC TYPE ID → CSV COLUMN NAME
# ═══════════════════════════════════════════════════════════════

REPORTING_API_METRIC_NAMES: dict[int, str] = {
    1:   "Impressions",
    2:   "Clicks",
    11:  "CTR",
    23:  "Conversion",
    25:  "Start Views",
    26:  "25% Complete",
    27:  "50% Complete",
    28:  "75% Complete",
    29:  "100% Complete",
    30:  "Drop Rate at 0%",
    31:  "Drop Rate at 25%",
    32:  "Drop Rate at 50%",
    33:  "Drop Rate at 75%",
    41:  "CPC",
    42:  "Conversion Rate",
    43:  "CPA",
    44:  "Advertiser Spending",
    45:  "Advertiser eCPM",
    49:  "Network Profit",
    50:  "Network Profit %",
    51:  "Network eCPM",
    62:  "25% Completion Rate",
    63:  "50% Completion Rate",
    64:  "75% Completion Rate",
    65:  "100% Completion Rate",
    100: "Yahoo+ Measurable Impressions",
    101: "Yahoo+ Viewable Impressions",
    102: "Yahoo+ Viewability Rate",
    103: "Yahoo+ Nonviewable Impressions",
    104: "Yahoo+ Non-Measurable Impressions",
    105: "Yahoo+ Measurable Rate",
    106: "Impression Distribution Viewable",
    107: "Impression Distribution Nonviewable",
    108: "Impression Distribution Nonmeasurable",
    109: "View Through Conversion",
    110: "Click Through Conversion",
    111: "View Through ConversionRate",
    112: "Click Through ConversionRate",
    113: "View Through CPA",
    114: "Click Through CPA",
    129: "Seat Management Fee",
    137: "ROAS Action Value",
    138: "ROAS",
    168: "vCPM",
    169: "CPCV",
    171: "Cross device conversions",
    173: "Qualified Count",
    180: "Yahoo Data Fees",
    184: "Third Party Data Fees",
    188: "Order Management Fee",
    192: "Ad Serving Cost",
    196: "Ad Verification Cost",
    200: "Audience Measurement Cost",
    204: "Brand Safety Cost",
    208: "Viewability Cost",
    212: "Other Cost",
    216: "Total Variable Cost",
    254: "Bid Responses",
    260: "Total Vendor Fee",
    296: "Bid Price Without Multiplier",
    302: "Bid CPM",
    305: "Dynamic Conversion Value",
    317: "Very first conversion",
    322: "Programmatic Guaranteed Deal Fee",
    328: "3rd Party Conversion",
    329: "Audience Impressions",
    365: "CPV",
    382: "Bid Shading Cost Savings",
    386: "Companion Impressions",
    387: "Companion Clicks",
    426: "Estimated Billable Amount",
    430: "Unique People",
    446: "Unique Households",
    447: "Average Frequency Per HouseHold",
    448: "Average Impression Multiplier",
    461: "Very First View Through Conversion",
    462: "Very First Click Through Conversion",
    463: "Very First View Through Conversion CPA",
    464: "Very First Click Through Conversion CPA",
    465: "View Through Dynamic Conversion Value",
    466: "View Through Dynamic Conversion ROAS",
    467: "Click Through Dynamic Conversion Value",
    468: "Click Through Dynamic Conversion ROAS",
    469: "View Through Conversion Value",
    470: "View Through Conversion ROAS",
    471: "Click Through Conversion Value",
    472: "Click Through Conversion ROAS",
    473: "Very First View Through Conversion Value",
    474: "Very First View Through Conversion ROAS",
    475: "Very First Click Through Conversion Value",
    476: "Very First Click Through Conversion ROAS",
    523: "Average People Frequency",
    560: "Yahoo+ Demo Fee",
    564: "Yahoo Segment Fee",
    570: "Third Party Segment Fee",
    583: "Available Bid Requests",
    584: "Average Win Rate",
    714: "Total Data Fee",
    715: "Platform Fee",
    716: "Inventory eCPM",
    717: "Inventory Cost %",
    721: "Yahoo+ Viewable Impressions GroupM",
    723: "Yahoo+ Viewability Rate GroupM",
    724: "Inventory Cost",
}

# REPORTING API — Metrics whose BQ type should be INTEGER
REPORTING_API_INTEGER_METRIC_IDS: set[int] = {
    1, 2, 23, 25, 26, 27, 28, 29,
    100, 101, 103, 104, 109, 110,
    171, 173, 254, 317, 328, 329,
    386, 387, 430, 446, 461, 462, 583, 721,
}


# ═══════════════════════════════════════════════════════════════
# REPORTING API — REPORT GROUP DEFINITIONS
# ═══════════════════════════════════════════════════════════════

REPORTING_API_REPORT_GROUP_CONFIGS: dict[str, dict] = {

    "campaign_delivery": {
        "dimension_ids": [4, 5, 6, 149, 56, 64, 19, 63, 1225, 1267],
        "metric_ids":    [1, 2, 11, 23, 41, 42, 43, 44, 45, 137, 138, 171, 305, 317, 328],
    },

    "creative_performance": {
        "dimension_ids": [4, 5, 6, 7, 8, 3, 174],
        "metric_ids":    [1, 2, 11, 23, 44, 45, 41, 43],
    },

    "video_performance": {
        "dimension_ids": [4, 5, 6, 3, 169, 167, 182, 1037, 1240],
        "metric_ids":    [
            1, 25, 26, 27, 28, 29, 30, 31, 32, 33,
            62, 63, 64, 65, 168, 169, 365, 386, 387, 44,
        ],
    },

    "viewability": {
        "dimension_ids": [4, 5, 6, 34, 106, 19],
        "metric_ids":    [1, 100, 101, 102, 103, 104, 105, 106, 107, 108, 721, 723, 44],
    },

    "conversion_detail": {
        "dimension_ids": [4, 5, 6, 46, 54, 147],
        "metric_ids":    [
            23, 109, 110, 111, 112, 113, 114, 137, 138, 305,
            317, 461, 462, 463, 464, 465, 466, 467, 468,
            469, 470, 471, 472, 473, 474, 475, 476, 44,
        ],
    },

    "cost_fee": {
        "dimension_ids": [4, 5, 6, 149],
        "metric_ids":    [
            44, 49, 50, 51, 129, 180, 184, 188, 192, 196,
            200, 204, 208, 212, 216, 260, 322, 426, 560,
            564, 570, 714, 715, 716, 717, 724,
        ],
    },

    "geo_device": {
        "dimension_ids": [4, 5, 19, 20, 34, 23, 122, 126],
        "metric_ids":    [1, 2, 11, 23, 44, 45, 41, 43],
    },

    "supply_inventory": {
        "dimension_ids": [4, 5, 6, 1228, 1033, 154, 106, 55, 166, 171],
        "metric_ids":    [1, 2, 11, 44, 45, 23, 41, 43],
    },

    "audience_reach": {
        "dimension_ids": [4, 5, 6, 166, 1225],
        "metric_ids":    [1, 2, 44, 329, 430, 446, 447, 448, 523],
    },

    "bidding": {
        "dimension_ids": [4, 5, 6, 149],
        "metric_ids":    [1, 173, 583, 584],
        "data_source":   15,
        "schema_override": [
            {"name": "Day",                    "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser Id",          "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Advertiser",             "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Id",            "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Campaign Name",          "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line Id",                "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Line",                   "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package Id",             "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Package",                "type": "STRING",  "mode": "NULLABLE"},
            {"name": "Impressions",            "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Qualified Count",        "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Available Bid Requests", "type": "INTEGER", "mode": "NULLABLE"},
            {"name": "Average Win Rate",       "type": "FLOAT64", "mode": "NULLABLE"},
            {"name": "ingestion_timestamp",    "type": "STRING",  "mode": "NULLABLE"},
            {"name": "lookback_date_range",    "type": "STRING",  "mode": "NULLABLE"},
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# FUTURE APIs — add sections below following the same pattern
# ═══════════════════════════════════════════════════════════════
#
# TRAFFIC_API_DIMENSION_CSV_COLUMNS = { ... }
# TRAFFIC_API_METRIC_NAMES = { ... }
# TRAFFIC_API_INTEGER_METRIC_IDS = { ... }
# TRAFFIC_API_REPORT_GROUP_CONFIGS = { ... }
#
# AUDIENCE_API_DIMENSION_CSV_COLUMNS = { ... }





# ═══════════════════════════════════════════════════════════════
# TRAFFIC API — ENDPOINT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════
# scope:
#   "seat"       → fetched once at seat level
#   "advertiser" → fetched per advertiser (requires advertiser list)
#
# To add a new endpoint: just add an entry here.
# dsp_traffic.py picks it up automatically.

# ═══════════════════════════════════════════════════════════════
# TRAFFIC API — ENDPOINT CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════
#
# scope:
#   "seat"       → fetched once, no params required
#   "advertiser" → iterated per advertiser ID, uses param_name="accountId"
#   "campaign"   → iterated per campaign ID, uses param_name="orderId"
#
# param_name:
#   The query parameter name the Yahoo API expects for the parent ID.
#   Only needed for "advertiser" and "campaign" scoped endpoints.

TRAFFIC_API_ENDPOINT_CONFIGS: dict[str, dict] = {

    # ── SEAT-LEVEL ───────────────────────────────────────

    "seats": {
        "path": "/traffic/seats",
        "scope": "seat",
    },

    "account_groups": {
        "path": "/traffic/accountgroups",
        "scope": "seat",
    },

    "advertisers": {
        "path": "/traffic/advertisers",
        "scope": "seat",
        "supports_date_filter": True,
    },

    "data_dictionary": {
        "path": "/traffic/dictionary",
        "scope": "seat",
    },

    "exchanges": {
        "path": "/traffic/exchanges",
        "scope": "seat",
    },

    # "user_groups": {
    #     "path": "/traffic/usergroups",
    #     "scope": "seat",
    # },

    # "dooh_screen_owners": {
    #     "path": "/traffic/dooh/screenowners",
    #     "scope": "seat",
    # },

    # "dooh_venue_types": {
    #     "path": "/traffic/dooh/venuetypes",
    #     "scope": "seat",
    # },

    # "dooh_screen_lists": {
    #     "path": "/traffic/dooh/screenlists",
    #     "scope": "seat",
    # },

    # ── ADVERTISER-LEVEL (accountId) ─────────────────────

    "campaigns": {
        "path": "/traffic/campaigns",
        "scope": "advertiser",
        "param_name": "accountId",
        "supports_date_filter": True,
    },

    "creatives": {
        "path": "/traffic/creatives",
        "scope": "advertiser",
        "param_name": "accountId",
        "supports_date_filter": True,
    },

    "ads": {
        "path": "/traffic/ads",
        "scope": "advertiser",
        "param_name": "accountId",
        "supports_date_filter": True,
    },

    "pixels": {
        "path": "/traffic/pixels",
        "scope": "advertiser",
        "param_name": "accountId",
    },
    
    "user_groups": {
        "path": "/traffic/usergroups",
        "scope": "advertiser",
        "param_name": "accountId",
    },

    "deals": {
        "path": "/traffic/deals",
        "scope": "advertiser",
        "param_name": "accountId",
    },

    # "site_lists": {
    #     "path": "/traffic/sitelists",
    #     "scope": "advertiser",
    #     "param_name": "accountId",
    # },

    # "contextuals": {
    #     "path": "/traffic/contextuals",
    #     "scope": "advertiser",
    #     "param_name": "accountId",
    # },

    "beacon_lookups": {
    "path": "/traffic/advertisers/{id}/beaconlookups",
    "scope": "advertiser",
    "param_name": "accountId",
    "id_in_path": True
    },

    # ── CAMPAIGN-LEVEL (orderId) ─────────────────────────

    "packages": {
        "path": "/traffic/packages",
        "scope": "campaign",
        "param_name": "orderId",
        "supports_date_filter": True,
    },

    "lines": {
        "path": "/traffic/lines",
        "scope": "campaign",
        "param_name": "orderId",
        "supports_date_filter": True,
    },
}