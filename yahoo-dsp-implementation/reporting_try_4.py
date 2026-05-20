"""
reporting.py — Yahoo DSP Reporting API Module
===============================================
Handles report creation, polling, download, transformation (Polars),
and local / GCS / BigQuery persistence for the Yahoo DSP Reporting pipeline.

Author  : Priyank Chauhan
Python  : 3.13+
"""

from __future__ import annotations

import io
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Optional

import pendulum
import polars as pl
import requests
from loguru import logger
from pydantic import BaseModel, field_validator, model_validator
from tqdm import tqdm

from helpers import YahooHelpers

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       DIMENSION  REGISTRY                                  ║
# ║  Every dimension ID from the Yahoo DSP Reporting API documentation.        ║
# ║  Format  →  { dimension_type_id: "Dimension Type Name" }                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ALL_DIMENSIONS: dict[int, str] = {
    # ── Core Entity Dimensions ────────────────────────────────────────────────
    3:    "Creative Size",              # Returns Creative Size ID + Creative Size (e.g. 300x250)
    4:    "Advertiser",                 # Returns Advertiser ID + Advertiser name
    5:    "Campaign",                   # Returns Campaign ID + Campaign Name
    6:    "Line",                       # Returns Line ID + Line name
    7:    "Ad",                         # Returns Ad ID + Ad name
    8:    "Creative",                   # Returns Creative ID + Creative Name  (Seat only)
    9:    "Domain",                     # Internet domain of the impression      (Seat only, high cardinality)
    14:   "Advertiser Category ID",     # Category assigned by trafficker        (Seat only)

    # ── Geography Dimensions ──────────────────────────────────────────────────
    19:   "Country",                    # Country ID + Country name
    20:   "State",                      # State / Province name
    21:   "City",                       # City ID + City name
    22:   "Postal Code",               # Zip / Postal code     (500k row limit)
    282:  "DMA",                        # Designated Market Area (US only, Seat)

    # ── Device & Technology ───────────────────────────────────────────────────
    23:   "Device Category",            # Device Category ID + name (Desktop, Mobile, etc.)
    24:   "Mobile OS",                  # Mobile OS  (Seat only)
    25:   "Browser",                    # Browser name  (Seat only)
    27:   "ISP",                        # Internet Service Provider (Seat only)
    28:   "OS Version",                 # OS Version string  (Seat only)
    31:   "Mobile Model",              # Handset model  (Seat only)
    42:   "Device Platform",           # Platform (e.g. App / Web)  (Seat only)
    39:   "Connection Type",           # WiFi, 4G, etc.  (Seat only)
    46:   "Exchange",                   # SSP / Exchange name
    47:   "App/URL",                    # Individual app or URL  (Seat, 500k row limit)
    50:   "Inventory Type",            # OLV, CTV, Display, Audio, DOOH, etc.
    54:   "Inventory Source Type",     # Open Exchange vs. Deals
    55:   "Deal",                       # Deal ID + Deal name
    56:   "Deal Type",                 # DealType ID + name (Preferred Deal, PG, etc.)

    # ── Audience & Segment ────────────────────────────────────────────────────
    34:   "Segment",                    # Segment ID + Segment Name
    63:   "Data Provider",             # Data Provider ID + name
    64:   "Data Type",                 # Data Type ID + name (1P, 3P, etc.)
    66:   "Audience Type",             # Audience Type ID + name  (Seat only)

    # ── Frequency ─────────────────────────────────────────────────────────────
    # NOTE: Frequency Bucket dimensions are mutually exclusive with each other
    80:   "Line Frequency (Lifetime)", # Lifetime freq bucket at Line level   (250k row limit, Seat)
    81:   "Campaign Frequency (Lifetime)", # Lifetime freq bucket at Campaign (250k row limit, Seat)
    82:   "Advertiser Frequency (Lifetime)", # Lifetime freq at Advertiser    (250k row limit, Seat)

    # ── Native & Supply ───────────────────────────────────────────────────────
    73:   "Section",                    # Native section  (Seat only)
    74:   "Placement Type",            # Placement type   (Seat only)
    75:   "Native Supply Type",        # Supply type      (Seat only)

    # ── Custom / AMP-specific ─────────────────────────────────────────────────
    103:  "Line Type",                 # Line Type ID + name (Seat only)
    104:  "Channel",                   # Channel ID + Channel name
    106:  "Environment Type",          # Web vs In-App vs CTV, etc.
    107:  "Creative Type",             # Creative Type ID + name  (Display, Video, Audio, etc.)
    108:  "Creative Feature Type",     # Feature type ID + name   (Seat only)
    122:  "Partnership Type",          # Partnership ID + name    (Seat only)
    124:  "Age Group",                 # Age bracket  (Seat only)
    125:  "Gender",                    # Gender  (Seat only)
    126:  "Income Range",              # Income range  (Seat only)
    127:  "Parental Status",           # Parental status  (Seat only)
    128:  "Ethnicity",                 # Ethnicity  (Seat only)

    # ── Video / CTV / Audio ───────────────────────────────────────────────────
    147:  "Video Player Type",         # Video player type  (Seat only)
    149:  "Media Type",                # Media Type ID + name (Display / Video / Audio / Native / CTV / DOOH)
    150:  "Language",                  # Language ID + Language name  (Seat only)
    151:  "Content Rating",            # Content rating  (Seat only)
    153:  "Content Genre",             # Genre  (Seat only)
    154:  "Supply Type",               # Supply Type ID + name  (Seat only)
    155:  "Rewarded",                  # Rewarded ad indicator  (Seat only)
    158:  "Clickthrough Destination", # Clickthrough type (Seat only)
    162:  "Vast Placement Type",      # VAST placement  (Seat only)
    165:  "CTV Device Model",         # CTV device model  (Seat only)
    166:  "CTV App Name",             # CTV App  (Seat only)
    167:  "CTV App Bundle Name",      # CTV bundle  (Seat only)
    169:  "Ad Length",                 # Ad duration bucket  (Seat only)
    171:  "Program Type",             # Program type  (Seat only)
    174:  "Exchange Type",            # Exchange Type (Seat only)
    182:  "CTV Inventory Source Brand", # CTV brand  (Seat only)
    185:  "Audio Device Type",        # Audio device type  (Seat only)
    192:  "Content Network",          # Content network  (Seat only)
    193:  "Content Channel",          # Content channel  (Seat only)
    194:  "Content Series",           # Content series  (Seat only)
    195:  "Live Stream",              # Live stream flag  (Seat only)
    196:  "Distributor",              # Distributor name  (Seat only)
    197:  "Studio",                   # Studio name  (Seat only)
    198:  "Content Language",         # Language of content  (Seat only)
    201:  "Audio Inventory Source Brand", # Audio ISB  (Seat only)
    202:  "DOOH Venue Name",          # DOOH venue  (Seat only)
    203:  "DOOH Venue Type",          # DOOH venue type  (Seat only)
    204:  "DOOH Publisher Name",      # DOOH publisher  (Seat only)
    205:  "Bid Strategy",             # Bid strategy  (Seat only)
    206:  "Buying Entity",            # Buying entity  (Seat only)
    207:  "Bid Shading Strategy",     # Bid shading  (Seat only)

    # ── Conversion & Pixel ────────────────────────────────────────────────────
    1028: "Pixel Name",               # Conversion pixel  (Seat only)
    1029: "Conversion Event Name",    # Conversion event  (Seat only)
    1030: "Conversion Device Type",   # Device used for conversion  (Seat only)
    1031: "Conversion Tag Type",      # Tag type  (Seat only)
    1033: "Product",                  # Product ID + Product Name (Seat only)
    1035: "Conversion Source",        # Conversion source  (Seat only)
    1037: "Conversion Subtype",       # Subtype  (Seat only)
    1038: "Conversion Match Type",    # Match type  (Seat only)
    1039: "Conversion Environment Type", # Environment  (Seat only)
    1040: "Conversion Pixel Event Source", # Pixel event source  (Seat only)
    1041: "Dot Tag Conversion Attribution Type", # Dot tag attr  (Seat only)
    1042: "Conversion Attributable Media Type", # Attributable media  (Seat only)
    1043: "Conversion Attributable Country",  # Attr country  (Seat only)
    1044: "Conversion Attributable State",    # Attr state  (Seat only)
    1045: "Conversion Attributable City",     # Attr city  (Seat only)
    1046: "Conversion Attributable DMA",      # Attr DMA  (Seat only)
    1047: "Conversion Attributable Postal Code", # Attr postal  (Seat only)
    1048: "Conversion Attributable Browser",  # Attr browser  (Seat only)
    1049: "Conversion Attributable Device",   # Attr device  (Seat only)
    1050: "Conversion Attributable OS",       # Attr OS  (Seat only)
    1051: "Conversion Attributable Connection Type", # Attr conn  (Seat only)
    1052: "Conversion Attributable ISP",      # Attr ISP  (Seat only)
    1054: "Conversion Attributable Exchange", # Attr exchange  (Seat only)
    1056: "Conversion Attributable Line Frequency (Lifetime)", # Freq bucket (Seat, 250k)
    1057: "Conversion Attributable Campaign Frequency (Lifetime)", # (Seat, 250k)
    1058: "Conversion Attributable Advertiser Frequency (Lifetime)", # (Seat, 250k)
    1059: "Conversion Attributable User Day Frequency (Line)", # (Seat, 250k)
    1060: "Conversion Attributable User Day Frequency (Campaign)", # (Seat, 250k)
    1061: "Conversion Attributable User Day Frequency (Advertiser)", # (Seat, 250k)
    1062: "Conversion Attributable User Hour Frequency (Line)", # (Seat, 250k)
    1063: "Conversion Attributable User Hour Frequency (Campaign)", # (Seat, 250k)
    1064: "Conversion Attributable User Hour Frequency (Advertiser)", # (Seat, 250k)
    1065: "Conversion Attributable Week Frequency (Line)", # (Seat, 250k)
    1066: "Conversion Attributable Week Frequency (Campaign)", # (Seat, 250k)
    1067: "Conversion Attributable Week Frequency (Advertiser)", # (Seat, 250k)
    1068: "Conversion Attributable Month Frequency (Line)", # (Seat, 250k)
    1069: "Conversion Attributable Month Frequency (Campaign)", # (Seat, 250k)
    1070: "Conversion Attributable Month Frequency (Advertiser)", # (Seat, 250k)
    1071: "Conversion Attributable Flight Frequency (Line)", # (Seat, 250k)
    1072: "Conversion Attributable Flight Frequency (Campaign)", # (Seat, 250k)
    1073: "Conversion Attributable Flight Frequency (Advertiser)", # (Seat, 250k)
    1074: "Conversion Attributable Week Frequency (Line SoD)", # (Seat, 250k)
    1075: "Conversion Attributable Week Frequency (Campaign SoD)", # (Seat, 250k)
    1076: "Conversion Attributable Week Frequency (Advertiser SoD)", # (Seat, 250k)
    1077: "Conversion Attributable Month Frequency (Line SoD)", # (Seat, 250k)
    1078: "Conversion Attributable Month Frequency (Campaign SoD)", # (Seat, 250k)
    1079: "Conversion Attributable Month Frequency (Advertiser SoD)", # (Seat, 250k)
    1080: "Conversion Attributable Flight Frequency (Line SoD)", # (Seat, 250k)
    1081: "Conversion Attributable Flight Frequency (Campaign SoD)", # (Seat, 250k)
    1082: "Conversion Attributable Flight Frequency (Advertiser SoD)", # (Seat, 250k)
    1090: "Conversion Attributable Inventory Type",    # (Seat only)
    1091: "Conversion Attributable Domain",            # (Seat only)
    1093: "Conversion Attributable CTV Device Model",  # (Seat only)
    1094: "Conversion Attributable CTV App Name",      # (Seat only)
    1101: "Conversion Attributable App/URL",           # (Seat only, 500k row limit)
    1106: "Conversion Attributable Deal",              # (Seat only)
    1109: "Conversion Attributable Ad",                # (Seat only)
    1110: "Conversion Attributable Creative",          # (Seat only)
    1111: "Conversion Attributable Creative Size",     # (Seat only)
    1112: "Conversion Attributable Line Type",         # (Seat only)
    1113: "Conversion Attributable Channel",           # (Seat only)
    1114: "Conversion Attributable Inventory Source Type", # (Seat only)
    1115: "Conversion Attributable Deal Type",         # (Seat only)
    1116: "Conversion Attributable Environment Type",  # (Seat only)
    1117: "Conversion Attributable Mobile OS",         # (Seat only)
    1118: "Conversion Attributable Creative Type",     # (Seat only)
    1119: "Conversion Attributable Video Player Type", # (Seat only)
    1120: "Conversion Attributable Content Rating",    # (Seat only)
    1122: "Conversion Attributable Content Genre",     # (Seat only)
    1123: "Conversion Attributable Supply Type",       # (Seat only)
    1124: "Conversion Attributable Exchange Type",     # (Seat only)
    1125: "Conversion Attributable CTV Inventory Source Brand", # (Seat only)
    1126: "Conversion Attributable Age Group",         # (Seat only)
    1127: "Conversion Attributable Gender",            # (Seat only)
    1128: "Conversion Attributable Income Range",      # (Seat only)
    1143: "Conversion Attributable Vast Placement Type", # (Seat only)
    1147: "Conversion Attributable Ad Length",         # (Seat only)
    1148: "Conversion Attributable Program Type",      # (Seat only)
    1149: "Conversion Attributable Audio Device Type", # (Seat only)
    1150: "Conversion Attributable Audio Inventory Source Brand", # (Seat only)
    1163: "Conversion Attributable Bid Strategy",      # (Seat only)
    1165: "Conversion Attributable Content Network",   # (Seat only)

    # ── Reach / Unique ────────────────────────────────────────────────────────
    1221: "Unique Reach Country",          # (Seat only)
    1222: "Unique Reach State",            # (Seat only)
    1223: "Unique Reach City",             # (Seat only)
    1224: "Unique Reach DMA",              # (Seat only)
    1225: "Unique Reach Media Type",       # (Seat only)
    1226: "Unique Reach Mobile OS",        # (Seat only)
    1227: "Unique Reach Browser",          # (Seat only)
    1228: "Unique Reach Device Category",  # (Seat only)
    1230: "Unique Reach Exchange",         # (Seat only)
    1231: "Unique Reach Creative Size",    # (Seat only)
    1232: "Unique Reach Creative Type",    # (Seat only)
    1233: "Unique Reach Domain",           # (Seat only)
    1240: "Unique Reach Age Group",        # (Seat only)

    # ── IFO (Incremental Frequency Optimization) Dimensions ───────────────────
    # NOTE: These require a seat-level feature flag to be enabled.
    1241: "IFO Country",                   # (Seat only, IFO flag)
    1242: "IFO State",                     # (Seat only, IFO flag)
    1243: "IFO City",                      # (Seat only, IFO flag)
    1244: "IFO DMA",                       # (Seat only, IFO flag)
    1245: "IFO Media Type",               # (Seat only, IFO flag)
    1246: "IFO Device Category",           # (Seat only, IFO flag)
    1247: "IFO Creative Size",             # (Seat only, IFO flag)
    1270: "IFO Age Group",                 # (Seat only, IFO flag)
    1271: "IFO Gender",                    # (Seat only, IFO flag)
    1272: "IFO Income Range",              # (Seat only, IFO flag)
    1273: "IFO Ethnicity",                 # (Seat only, IFO flag)
    1274: "IFO Parental Status",           # (Seat only, IFO flag)

    # ── Additional / Later Additions ──────────────────────────────────────────
    1252: "Unique Reach Age Group 2",      # (Seat only)
    1253: "Unique Reach Gender",           # (Seat only)
    1254: "Unique Reach Income Range",     # (Seat only)
    1255: "Unique Reach Ethnicity",        # (Seat only)
    1256: "Unique Reach Parental Status",  # (Seat only)
    1257: "Unique Reach Inventory Type",   # (Seat only)
    1260: "Unique Reach Environment Type", # (Seat only)
    1261: "Unique Reach Channel",          # (Seat only)
    1262: "Unique Reach Exchange Type",    # (Seat only)
    1264: "Unique Reach Connection Type",  # (Seat only)
    1265: "Unique Reach CTV App Name",     # (Seat only)
    1266: "Unique Reach CTV Inventory Source Brand", # (Seat only)
    1267: "Unique Reach Audio Inventory Source Brand", # (Seat only)
    1268: "Unique Reach Audio Device Type", # (Seat only)
    1269: "Unique Reach DOOH Venue Type",  # (Seat only)
    1276: "Unique Reach Content Network",  # (Seat only)
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                       METRIC  REGISTRY (Standard)                          ║
# ║  All non-bidding, non-IFO metric IDs.                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

ALL_METRICS: dict[int, str] = {
    # ── Delivery ──────────────────────────────────────────────────────────────
    1:   "Impressions",
    2:   "Clicks",
    11:  "Post Click Conversions",
    23:  "Post Imp Conversions",
    25:  "Click Rate",                    # CTR (%)
    26:  "Post Imp Conversion Rate",
    27:  "Post Click Conversion Rate",
    28:  "Media Cost",                    # Advertiser currency
    29:  "Data Cost",
    30:  "Other Cost",
    31:  "Total Cost",
    32:  "Post Click Revenue",
    33:  "Post Imp Revenue",

    # ── CPx / ROI ─────────────────────────────────────────────────────────────
    41:  "eCPM",
    42:  "eCPC",
    43:  "Post Click CPA",
    44:  "Post Imp CPA",
    45:  "eCPA",
    49:  "Total Conversions",
    50:  "Total Revenue",
    51:  "ROAS",
    53:  "Unique Impressions",            # ⚠ 500k row limit
    62:  "Advertiser Spending",
    63:  "Gross Revenue",
    64:  "Net Revenue",
    65:  "Platform Access Fees",

    # ── Video ─────────────────────────────────────────────────────────────────
    100: "Video Views",
    101: "Video First Quartiles",
    102: "Video Midpoints",
    103: "Video Third Quartiles",
    104: "Video Completes",
    105: "Video Starts",
    106: "Video Start Rate",
    107: "Video First Quartile Rate",
    108: "Video Midpoint Rate",
    109: "Video Third Quartile Rate",
    110: "Video Complete Rate",
    111: "Video Close",
    112: "Video Skip",
    113: "VCR",                          # Video Completion Rate
    114: "CPCV",                         # Cost Per Completed View
    115: "Unique Video Views",           # ⚠ 500k row limit
    129: "Video Skip Rate",

    # ── Viewability ───────────────────────────────────────────────────────────
    134: "Measured Impressions",          # ⚠ 500k row limit
    135: "Viewable Impressions",          # ⚠ 500k row limit
    137: "Measurability Rate",
    138: "Viewability Rate",

    # ── Audio ─────────────────────────────────────────────────────────────────
    168: "Audio Starts",
    169: "Audio First Quartiles",
    171: "Audio Midpoints",
    180: "Audio Third Quartiles",
    184: "Audio Completes",
    188: "Audio Complete Rate",
    192: "Audio Start Rate",
    196: "Audio First Quartile Rate",
    200: "Audio Midpoint Rate",
    204: "Audio Third Quartile Rate",
    208: "Audio Close",
    212: "Audio Skip",
    216: "Audio Skip Rate",

    # ── Advanced Delivery / Profit ────────────────────────────────────────────
    260: "Supply Cost",
    305: "Gross Profit",
    317: "Partner Cost",
    322: "Customer Revenue",
    328: "Total Data Cost",
    329: "Platform Total Cost",
    330: "Customer Total Cost",
    334: "DSP Fees",
    338: "Brand Safety Cost",
    346: "Verification Cost",
    350: "Political Regulatory Cost",
    365: "Audience Data + DMP Cost",

    # ── Reach & Frequency ─────────────────────────────────────────────────────
    319: "Unique Reach Impressions",      # ⚠ 500k row limit
    320: "Unique Reach",                  # ⚠ 500k row limit
    321: "Average Frequency",             # ⚠ 500k row limit

    # ── Conversions Advanced ──────────────────────────────────────────────────
    386: "Post Click Activity Conversions",
    387: "Post Imp Activity Conversions",
    390: "Total Activity Conversions",
    391: "Post Click Activity Revenue",
    392: "Post Imp Activity Revenue",
    393: "Total Activity Revenue",
    394: "Post Click Activity CPA",
    395: "Post Imp Activity CPA",
    396: "Activity eCPA",
    397: "Post Click Activity ROAS",
    398: "Post Imp Activity ROAS",
    399: "Post Click Activity Unique Converters", # ⚠ 500k row limit
    400: "Post Imp Activity Unique Converters",   # ⚠ 500k row limit

    # ── Revenue Metrics ───────────────────────────────────────────────────────
    426: "Total CPA",
    430: "Total ROAS",
    431: "Post Click Unique Converters",  # ⚠ 500k row limit
    432: "Post Imp Unique Converters",    # ⚠ 500k row limit
    433: "Unique Converters",             # ⚠ 500k row limit

    # ── Additional Video / CTV ────────────────────────────────────────────────
    446: "Video Unmute",
    447: "Video Rewind",
    448: "Video Pause",
    453: "Video Companion Impressions",   # ⚠ 500k row limit
    455: "Video Companion Clicks",        # ⚠ 500k row limit
    456: "Video Companion Click-Through Rate", # ⚠ 500k row limit
    457: "Video Companion Views",         # ⚠ 500k row limit

    # ── CTV Advanced ──────────────────────────────────────────────────────────
    461: "Connected TV Impressions",
    462: "Connected TV eCPM",
    463: "Connected TV Total Cost",
    464: "Matchable Connected TV Impressions",
    465: "Connected TV Households Reached",
    466: "Connected TV Views",
    467: "Connected TV VCR",
    468: "Connected TV CPCV",
    469: "Matchable Connected TV Views",
    470: "Connected TV View Households Reached",
    471: "Connected TV Household Frequency",
    472: "Connected TV Household View Frequency",
    473: "Connected TV Impressions (Person Level)",
    474: "Connected TV People Reached",
    475: "Connected TV People Frequency",
    476: "Connected TV People View Frequency",

    # ── Misc / Newer ──────────────────────────────────────────────────────────
    523: "Amazon Conversions",
    560: "Companion CPCV",
    564: "Companion VCR",
    570: "Cost Per Unique Impression",

    # ── DOOH ──────────────────────────────────────────────────────────────────
    705: "DOOH Impressions",
    714: "People Impressions",
    715: "eCPM People",
    716: "People Reached",
    717: "People Frequency",
    721: "Household Impressions",
    723: "Household Reached",
    724: "Household Frequency",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  BIDDING-ONLY METRICS  (dataSource = 15)                   ║
# ║  These metrics ONLY work when dataSource=15 is specified in the payload.   ║
# ║  They can only be combined with dimensions: 4, 5, 6, 149.                 ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

BIDDING_ONLY_METRICS: dict[int, str] = {
    173: "Bids",
    254: "Bid Requests",
    296: "Win Rate",
    298: "Average Bid CPM",
    301: "Average Win CPM",
    302: "Filtered Bid Requests",
    304: "Bid Rate",
    371: "Pre-Ad Server Bid Requests",
    373: "Pre-Ad Server Filtered Bid Requests",
    375: "Pre-Ad Server Bids",
    377: "Pre-Ad Server Bid Rate",
    382: "Pre-Ad Server No Bid Rate",
    583: "Bid CPM p95",
    584: "Bid CPM p5",
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              IFO (Incremental Frequency Optimization) METRICS              ║
# ║  Require a seat-level feature flag to be enabled.                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

IFO_METRICS: dict[int, str] = {
    744: "IFO Unique Reach",
    745: "IFO Average Frequency",
    746: "IFO Unique Impressions",
    747: "IFO Unique Video Views",            # ⚠ 500k row limit
    748: "IFO Unique Audio Listens",          # ⚠ 500k row limit
    753: "IFO Incremental Reach",
    754: "IFO Incremental Impressions",
    755: "IFO Target Population Size",
    756: "IFO On-Target Impressions",
    757: "IFO On-Target Reach",
    758: "IFO On-Target Rate",
    759: "IFO On-Target Reach Rate",
    760: "IFO On-Target Average Frequency",
    761: "IFO On-Target eCPM",
    762: "IFO On-Target Reach eCPM",
    763: "IFO Reach Rate",
    764: "IFO Incremental Reach Rate",
    769: "IFO On-Target Incremental Reach",
    770: "IFO On-Target Incremental Reach Rate",
    771: "IFO Incremental Impressions Rate",
    772: "IFO On-Target Incremental Impressions",
    773: "IFO On-Target Incremental Impressions Rate",
}


# ── Convenience constant: IFO-specific Dimensions ────────────────────────────
IFO_DIMENSIONS: dict[int, str] = {
    k: v for k, v in ALL_DIMENSIONS.items()
    if v.startswith("IFO ")
}   # 1241,1242,1243,1244,1245,1246,1247,1270,1271,1272,1273,1274


# ── Metrics with row-limit restrictions (500k or 250k) ───────────────────────
ROW_LIMITED_METRICS: set[int] = {
    53,   # Unique Impressions
    115,  # Unique Video Views
    134,  # Measured Impressions
    135,  # Viewable Impressions
    319,  # Unique Reach Impressions
    320,  # Unique Reach
    321,  # Average Frequency
    399,  # Post Click Activity Unique Converters
    400,  # Post Imp Activity Unique Converters
    431,  # Post Click Unique Converters
    432,  # Post Imp Unique Converters
    433,  # Unique Converters
    453,  # Video Companion Impressions
    455,  # Video Companion Clicks
    456,  # Video Companion CTR
    457,  # Video Companion Views
    747,  # IFO Unique Video Views
    748,  # IFO Unique Audio Listens
}


# ── High-cardinality Dimensions (may hit 500k row limit quickly) ──────────────
HIGH_CARDINALITY_DIMENSIONS: set[int] = {9, 21, 22, 47, 282}


# ── Mutually-exclusive Frequency Bucket Dimension groups ──────────────────────
# Only ONE dimension from EACH group may appear in a single request.
FREQUENCY_BUCKET_IMPRESSION_DIMS: set[int] = {80, 81, 82}
FREQUENCY_BUCKET_CONVERSION_DIMS: set[int] = set(range(1056, 1083))  # 1056-1082


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                  DATE TYPE IDS  &  INTERVAL TYPE IDS                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DATE_TYPES: dict[int, str] = {
    1:  "Today",
    2:  "Yesterday",
    3:  "Last 7 Days",
    4:  "Current Month",
    5:  "Last Month",
    7:  "Life of Campaign",
    11: "Custom Range",          # Requires startDate + endDate
}

INTERVAL_TYPES: dict[int, str] = {
    1: "Cumulative",             # Single-row roll-up
    2: "Day",                    # Daily granularity  (populates Day column)
    5: "Month",                  # Monthly granularity (populates Month column)
}


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      PYDANTIC  VALIDATION  MODELS                          ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class FilterOptionValue(BaseModel):
    """Represents a single filter value (e.g., an advertiser ID to include/exclude)."""
    id: int | str
    name: Optional[str] = None


class FilterOption(BaseModel):
    """
    Represents a dimension filter applied to a report.
    dimensionTypeId  – the dimension to filter on (e.g. 4 = Advertiser).
    isExcluded       – True to *exclude* the listed values, False to *include*.
    includeValues    – list of values to include (or exclude).
    """
    dimensionTypeId: int
    isExcluded: bool = False
    includeValues: list[FilterOptionValue]


class SpecObject(BaseModel):
    """A single spec inside a HAVING clause (threshold on a metric)."""
    metricTypeId: int
    type: str            # e.g. "greaterThan", "lessThan"
    value: int | float


class HavingObject(BaseModel):
    """HAVING clause applied after aggregation."""
    type: str            # "and" | "or"
    havingSpecs: list[SpecObject]


class LimitSpecColumn(BaseModel):
    """Column to order by when using LimitSpec."""
    direction: str = "DESCENDING"
    metricTypeId: int


class LimitSpec(BaseModel):
    """Top-N / limit specification for the report result set."""
    columns: list[LimitSpecColumn]
    limit: int = 10000
    type: str = "default"


class ReportOption(BaseModel):
    """Core report options: dimensions, metrics, filters, currency, timezone."""
    timezone: str = "America/New_York"
    currency: int = 4                  # 4 = USD (advertiser currency codes)
    dimensionTypeIds: list[int]
    metricTypeIds: list[int]
    filterOptions: list[FilterOption] = []
    having: Optional[HavingObject] = None
    limitSpec: Optional[LimitSpec] = None
    accountIds: list[int] = []         # Seat / advertiser IDs to restrict to

    @field_validator("dimensionTypeIds")
    @classmethod
    def _at_least_one_dimension(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("dimensionTypeIds must contain at least one dimension.")
        return v

    @field_validator("metricTypeIds")
    @classmethod
    def _at_least_one_metric(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("metricTypeIds must contain at least one metric.")
        return v


class ReportPayload(BaseModel):
    """
    Full report creation payload.
    
    Rules
    -----
    * dateTypeId=11 (Custom Range) → ``startDate`` and ``endDate`` are **required**.
    * dataSource (optional) – set to 15 only when requesting bidding metrics.
    """
    reportOption: ReportOption
    dateTypeId: int
    intervalTypeId: int = 2            # Default: Day
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    dataSource: Optional[int] = None

    @model_validator(mode="after")
    def _validate_custom_range_dates(self) -> "ReportPayload":
        """If dateTypeId=11 (Custom Range), both startDate and endDate are mandatory."""
        if self.dateTypeId == 11:
            if not self.startDate or not self.endDate:
                raise ValueError(
                    "startDate and endDate are required when dateTypeId=11 (Custom Range)."
                )
        return self

    @field_validator("dataSource")
    @classmethod
    def _validate_datasource(cls, v: int | None) -> int | None:
        if v is not None and v != 15:
            raise ValueError("dataSource must be 15 (bidding data) or omitted entirely.")
        return v


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                      YAHOO  REPORTING  CLASS                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class YahooReporting:
    """
    Orchestrates the full Yahoo DSP Reporting lifecycle:
        1. Create a report  →  POST
        2. Poll until ready →  GET  (with tqdm progress)
        3. Download CSV     →  GET
        4. Transform        →  Polars DataFrame
        5. Persist          →  Local / GCS / BigQuery
    """

    # ── API Endpoints ─────────────────────────────────────────────────────────
    REPORT_ENDPOINT: str = (
        "https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/"
    )
    COB_ENDPOINT: str = (
        "https://api-sched-v3.admanagerplus.yahoo.com/yamplus_api/extreport/cob"
    )

    def __init__(self, helpers: YahooHelpers) -> None:
        """
        Parameters
        ----------
        helpers : YahooHelpers
            Instance of the helper class that provides auth-token management,
            GCS utilities, BQ utilities, and schema loading.
        """
        self.helpers = helpers
        logger.info("YahooReporting initialised with helper instance.")

    # ──────────────────────────────────────────────────────────────────────────
    #  AUTH  HEADER  BUILDER
    # ──────────────────────────────────────────────────────────────────────────
    def _auth_headers(self) -> dict[str, str]:
        """Build and return the authorisation headers required by the Yahoo DSP API."""
        token = self.helpers.get_access_token()
        return {
            "X-Auth-Method": "OAuth2",
            "X-Auth-Token": token,
            "Content-Type": "application/json",
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  1. CREATE REPORT
    # ──────────────────────────────────────────────────────────────────────────
    def create_report(self, payload: dict) -> str:
        """
        Submit a new report request to the Yahoo DSP Reporting API.

        Parameters
        ----------
        payload : dict
            Dictionary conforming to the ReportPayload schema.

        Returns
        -------
        str
            The ``customerReportId`` assigned by Yahoo.

        Raises
        ------
        ValueError
            If payload validation fails.
        requests.HTTPError
            If the API returns a non-2xx response.
        """
        try:
            # ── Validate payload with Pydantic ────────────────────────────────
            validated = ReportPayload(**payload)
            logger.info(
                f"Payload validated — dateTypeId={validated.dateTypeId}, "
                f"dims={len(validated.reportOption.dimensionTypeIds)}, "
                f"metrics={len(validated.reportOption.metricTypeIds)}"
            )

            # ── POST to Yahoo ─────────────────────────────────────────────────
            response = requests.post(
                self.REPORT_ENDPOINT,
                headers=self._auth_headers(),
                json=validated.model_dump(exclude_none=True),
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            report_id = data.get("customerReportId")
            if not report_id:
                raise ValueError(f"No customerReportId in API response: {data}")

            logger.success(f"Report created ─ customerReportId={report_id}")
            return report_id

        except Exception as exc:
            logger.error(f"create_report FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  2. POLL REPORT STATUS
    # ──────────────────────────────────────────────────────────────────────────
    def poll_report_status(
        self,
        report_id: str,
        max_wait: int = 1800,
        poll_interval: int = 15,
    ) -> dict:
        """
        Poll the report endpoint until status is SUCCESS or FAILED.

        Parameters
        ----------
        report_id : str
            The ``customerReportId`` returned by :meth:`create_report`.
        max_wait : int
            Maximum seconds to wait before raising a timeout error.
        poll_interval : int
            Seconds between successive GET requests.

        Returns
        -------
        dict
            Full API response on SUCCESS (contains ``url``, ``numRows``, etc.).

        Raises
        ------
        TimeoutError
            If the report does not complete within ``max_wait`` seconds.
        RuntimeError
            If the API reports FAILED status.
        """
        url = f"{self.REPORT_ENDPOINT}{report_id}"
        elapsed = 0

        logger.info(f"Polling report {report_id} (max {max_wait}s, interval {poll_interval}s) …")

        # tqdm progress bar capped at max_wait seconds
        with tqdm(total=max_wait, desc="⏳ Polling", unit="s", ncols=90) as pbar:
            while elapsed < max_wait:
                try:
                    resp = requests.get(url, headers=self._auth_headers(), timeout=60)
                    resp.raise_for_status()
                    data = resp.json()
                except requests.exceptions.HTTPError as http_err:
                    # Token may have expired — attempt refresh and retry
                    if resp.status_code == 401:
                        logger.warning("Token expired during poll — refreshing …")
                        self.helpers.refresh_access_token()
                        time.sleep(poll_interval)
                        elapsed += poll_interval
                        pbar.update(poll_interval)
                        continue
                    raise http_err
                except Exception as exc:
                    logger.error(f"Polling error: {exc}\n{traceback.format_exc()}")
                    raise

                status = data.get("status", "UNKNOWN").upper()
                logger.info(f"Report {report_id} — status={status} (elapsed {elapsed}s)")

                if status == "SUCCESS":
                    logger.success(
                        f"Report {report_id} READY — rows={data.get('numRows')} "
                        f"url={data.get('url', 'N/A')[:80]}…"
                    )
                    return data

                if status == "FAILED":
                    raise RuntimeError(
                        f"Report {report_id} FAILED. Response: {data}"
                    )

                # Still SUBMITTED / PROCESSING — wait and retry
                time.sleep(poll_interval)
                elapsed += poll_interval
                pbar.update(poll_interval)

        raise TimeoutError(
            f"Report {report_id} not ready within {max_wait}s — last status: {status}"
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  3. DOWNLOAD REPORT CSV
    # ──────────────────────────────────────────────────────────────────────────
    def download_report_csv(self, download_url: str) -> str:
        """
        Download the completed report CSV from the presigned URL.

        Parameters
        ----------
        download_url : str
            The ``url`` field from the SUCCESS status response.

        Returns
        -------
        str
            Raw CSV text.
        """
        try:
            logger.info(f"Downloading CSV from: {download_url[:100]}…")
            resp = requests.get(download_url, timeout=300)
            resp.raise_for_status()
            csv_text = resp.text
            logger.success(
                f"CSV downloaded — {len(csv_text):,} chars, "
                f"~{csv_text.count(chr(10)):,} lines."
            )
            return csv_text
        except Exception as exc:
            logger.error(f"download_report_csv FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  4. CHECK COB (Close of Books)
    # ──────────────────────────────────────────────────────────────────────────
    def check_cob(self, advertiser_id: int, day: str) -> bool:
        """
        Check whether a specific day has been closed (COB) for an advertiser.

        Parameters
        ----------
        advertiser_id : int
            Yahoo DSP advertiser ID.
        day : str
            Date in ``YYYYMMDD`` format (e.g. ``"20250501"``).

        Returns
        -------
        bool
            True if the day is closed, False otherwise.
        """
        try:
            params = {"advertiserId": advertiser_id, "day": day}
            resp = requests.get(
                self.COB_ENDPOINT,
                headers=self._auth_headers(),
                params=params,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            is_closed = str(data.get("dayClosed", "false")).lower() == "true"
            logger.info(
                f"COB check — advertiser={advertiser_id}, day={day}, "
                f"dayClosed={is_closed}"
            )
            return is_closed
        except Exception as exc:
            logger.error(f"check_cob FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  5. TRANSFORM CSV → POLARS DATAFRAME
    # ──────────────────────────────────────────────────────────────────────────
    def transform_to_polars(
        self,
        csv_text: str,
        schema: list[dict],
        lookback_range: str = "",
    ) -> pl.DataFrame:
        """
        Parse raw CSV text into a Polars DataFrame and cast columns to types
        that match the BigQuery target schema.

        Parameters
        ----------
        csv_text : str
            Raw CSV content from Yahoo.
        schema : list[dict]
            List of dicts ``{"name": str, "type": str, "mode": str}`` defining
            the BigQuery schema.  Supported types: STRING, INTEGER, INT64,
            FLOAT, FLOAT64, BOOLEAN.
        lookback_range : str
            Human-readable date-range label (e.g. ``"2024-04-05 → 2025-05-04"``).

        Returns
        -------
        pl.DataFrame
            Typed DataFrame with metadata columns appended.
        """
        try:
            logger.info("Transforming CSV → Polars DataFrame …")

            # ── Read raw CSV (all as Utf8 first) ─────────────────────────────
            df = pl.read_csv(
                io.StringIO(csv_text),
                infer_schema_length=0,       # treat everything as Utf8 initially
                null_values=["", "N/A", "null", "NULL", "-"],
            )
            logger.info(f"Raw DataFrame shape: {df.shape}")

            # ── Build a mapping: column_name → target polars dtype ────────────
            TYPE_MAP: dict[str, pl.DataType] = {
                "STRING":  pl.Utf8,
                "INTEGER": pl.Int64,
                "INT64":   pl.Int64,
                "FLOAT":   pl.Float64,
                "FLOAT64": pl.Float64,
                "BOOLEAN": pl.Boolean,
            }
            schema_map: dict[str, pl.DataType] = {}
            for col_def in schema:
                col_name = col_def["name"]
                bq_type = col_def.get("type", "STRING").upper()
                schema_map[col_name] = TYPE_MAP.get(bq_type, pl.Utf8)

            # ── Keep only columns that exist in BOTH the CSV and the schema ───
            schema_cols = set(schema_map.keys())
            csv_cols = set(df.columns)

            cols_to_keep = schema_cols & csv_cols
            cols_missing_in_csv = schema_cols - csv_cols
            cols_extra_in_csv = csv_cols - schema_cols

            if cols_missing_in_csv:
                logger.warning(
                    f"{len(cols_missing_in_csv)} schema column(s) not found in CSV "
                    f"(will be added as null): {sorted(cols_missing_in_csv)}"
                )
            if cols_extra_in_csv:
                logger.warning(
                    f"{len(cols_extra_in_csv)} CSV column(s) not in schema "
                    f"(will be dropped): {sorted(cols_extra_in_csv)}"
                )

            # Select only matching columns
            df = df.select([c for c in df.columns if c in cols_to_keep])

            # Add missing schema columns as null with correct type
            for col_name in cols_missing_in_csv:
                target_type = schema_map[col_name]
                df = df.with_columns(pl.lit(None).cast(target_type).alias(col_name))

            # ── Cast each column to the target dtype ──────────────────────────
            cast_exprs = []
            for col_name in df.columns:
                if col_name in schema_map:
                    target_type = schema_map[col_name]
                    cast_exprs.append(
                        pl.col(col_name).cast(target_type, strict=False).alias(col_name)
                    )
                else:
                    cast_exprs.append(pl.col(col_name))

            df = df.select(cast_exprs)

            # ── Append metadata columns ───────────────────────────────────────
            utc_now = datetime.now(timezone.utc).isoformat()
            df = df.with_columns([
                pl.lit(utc_now).alias("ingestion_timestamp"),
                pl.lit(lookback_range).alias("lookback_date_range"),
            ])

            logger.success(
                f"Transformation complete — final shape: {df.shape}, "
                f"columns: {df.columns}"
            )
            return df

        except Exception as exc:
            logger.error(f"transform_to_polars FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  6. DATAFRAME → CSV BYTES
    # ──────────────────────────────────────────────────────────────────────────
    def df_to_csv_bytes(self, df: pl.DataFrame) -> bytes:
        """
        Serialise a Polars DataFrame to CSV bytes (UTF-8).

        Parameters
        ----------
        df : pl.DataFrame

        Returns
        -------
        bytes
            CSV content encoded as UTF-8 bytes.
        """
        try:
            buf = io.BytesIO()
            df.write_csv(buf)
            csv_bytes = buf.getvalue()
            logger.info(f"DataFrame serialised to CSV — {len(csv_bytes):,} bytes.")
            return csv_bytes
        except Exception as exc:
            logger.error(f"df_to_csv_bytes FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  7. SAVE LOCAL
    # ──────────────────────────────────────────────────────────────────────────
    def save_local(self, data: bytes, filepath: str) -> str:
        """
        Persist raw bytes to the local filesystem.

        Parameters
        ----------
        data : bytes
            Content to write.
        filepath : str
            Destination path (directories are created automatically).

        Returns
        -------
        str
            Absolute path of the saved file.
        """
        try:
            os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(data)
            abs_path = os.path.abspath(filepath)
            logger.success(f"File saved locally → {abs_path} ({len(data):,} bytes)")
            return abs_path
        except Exception as exc:
            logger.error(f"save_local FAILED: {exc}\n{traceback.format_exc()}")
            raise

    # ──────────────────────────────────────────────────────────────────────────
    #  8. FULL PIPELINE ORCHESTRATOR
    # ──────────────────────────────────────────────────────────────────────────
    def run_pipeline(
        self,
        config: dict,
        schema: list[dict],
        output_dir: str = "output",
        gcs_bucket: str | None = None,
        gcs_path: str | None = None,
        bq_table: str | None = None,
        skip_cob: bool = False,
        advertiser_ids_for_cob: list[int] | None = None,
    ) -> pl.DataFrame:
        """
        End-to-end orchestration of the Yahoo DSP Reporting pipeline.

        Steps
        -----
        1. (Optional) Close-of-Books check for each advertiser.
        2. Create report via POST.
        3. Poll until SUCCESS.
        4. Download the CSV.
        5. Transform into a typed Polars DataFrame.
        6. Export DataFrame → CSV bytes.
        7. Save to local disk.
        8. (Optional) Upload CSV to GCS.
        9. (Optional) Load from GCS URI into BigQuery.

        Parameters
        ----------
        config : dict
            Full report payload dictionary (must be ReportPayload-compatible).
        schema : list[dict]
            BigQuery target schema (list of {"name", "type", "mode"} dicts).
        output_dir : str
            Local directory for saving the output CSV.
        gcs_bucket : str | None
            GCS bucket name (skip cloud upload if None).
        gcs_path : str | None
            GCS object path (e.g. ``"yahoo/reporting/daily/"``).
        bq_table : str | None
            BigQuery table ID ``project.dataset.table`` (skip BQ load if None).
        skip_cob : bool
            If True, skip the Close-of-Books check entirely.
        advertiser_ids_for_cob : list[int] | None
            Advertiser IDs to verify COB against. Defaults to None.

        Returns
        -------
        pl.DataFrame
            The final, typed DataFrame (after transformation).
        """
        pipeline_steps = [
            "COB Check", "Create Report", "Poll Status",
            "Download CSV", "Transform", "Export CSV",
            "Save Local", "Upload GCS", "Load BigQuery",
        ]

        try:
            with tqdm(total=len(pipeline_steps), desc="🚀 Pipeline", unit="step", ncols=100) as pbar:

                # ── Step 1: Close-of-Books Check ──────────────────────────────
                pbar.set_postfix_str("COB Check")
                if not skip_cob and advertiser_ids_for_cob:
                    cob_day = pendulum.yesterday("America/New_York").format("YYYYMMDD")
                    for adv_id in advertiser_ids_for_cob:
                        closed = self.check_cob(adv_id, cob_day)
                        if not closed:
                            logger.warning(
                                f"Advertiser {adv_id} day {cob_day} NOT closed — "
                                f"data may be incomplete."
                            )
                else:
                    logger.info("COB check skipped.")
                pbar.update(1)

                # ── Step 2: Create Report ─────────────────────────────────────
                pbar.set_postfix_str("Create Report")
                report_id = self.create_report(config)
                pbar.update(1)

                # ── Step 3: Poll Status ───────────────────────────────────────
                pbar.set_postfix_str("Poll Status")
                status_resp = self.poll_report_status(report_id)
                download_url = status_resp.get("url")
                if not download_url:
                    raise ValueError(
                        f"Report succeeded but no download URL returned: {status_resp}"
                    )
                pbar.update(1)

                # ── Step 4: Download CSV ──────────────────────────────────────
                pbar.set_postfix_str("Download CSV")
                csv_text = self.download_report_csv(download_url)
                pbar.update(1)

                # ── Step 5: Transform ─────────────────────────────────────────
                pbar.set_postfix_str("Transform")
                lookback = (
                    f"{config.get('startDate', 'N/A')} → {config.get('endDate', 'N/A')}"
                    if config.get("dateTypeId") == 11
                    else DATE_TYPES.get(config.get("dateTypeId", 0), "Unknown")
                )
                df = self.transform_to_polars(csv_text, schema, lookback_range=lookback)
                pbar.update(1)

                # ── Step 6: Export CSV Bytes ──────────────────────────────────
                pbar.set_postfix_str("Export CSV")
                csv_bytes = self.df_to_csv_bytes(df)
                pbar.update(1)

                # ── Step 7: Save Locally ──────────────────────────────────────
                pbar.set_postfix_str("Save Local")
                timestamp = pendulum.now("UTC").format("YYYYMMDD_HHmmss")
                filename = f"yahoo_dsp_report_{timestamp}.csv"
                filepath = os.path.join(output_dir, filename)
                self.save_local(csv_bytes, filepath)
                pbar.update(1)

                # ── Step 8: Upload to GCS (disabled for local dev) ────────────
                pbar.set_postfix_str("Upload GCS")
                if gcs_bucket and gcs_path:
                    gcs_uri = f"gs://{gcs_bucket}/{gcs_path}{filename}"
                    logger.info(f"Uploading to GCS → {gcs_uri}")
                    # self.helpers.upload_to_gcs(
                    #     bucket_name=gcs_bucket,
                    #     blob_path=f"{gcs_path}{filename}",
                    #     data=csv_bytes,
                    # )
                    # logger.success(f"GCS upload complete → {gcs_uri}")
                    logger.warning("GCS upload is commented out for local development.")
                else:
                    logger.info("GCS upload skipped (no bucket/path provided).")
                pbar.update(1)

                # ── Step 9: Load into BigQuery (disabled for local dev) ───────
                pbar.set_postfix_str("Load BigQuery")
                if bq_table and gcs_bucket and gcs_path:
                    gcs_uri = f"gs://{gcs_bucket}/{gcs_path}{filename}"
                    logger.info(f"Loading to BigQuery → {bq_table} from {gcs_uri}")
                    # self.helpers.load_gcs_to_bq(
                    #     gcs_uri=gcs_uri,
                    #     bq_table=bq_table,
                    #     schema=schema,
                    # )
                    # logger.success(f"BigQuery load complete → {bq_table}")
                    logger.warning("BQ load is commented out for local development.")
                else:
                    logger.info("BigQuery load skipped (no table/GCS config provided).")
                pbar.update(1)

            logger.success(
                f"✅ Pipeline complete — {df.shape[0]:,} rows, {df.shape[1]} cols, "
                f"saved → {filepath}"
            )
            return df

        except Exception as exc:
            logger.error(f"run_pipeline FAILED: {exc}\n{traceback.format_exc()}")
            raise
