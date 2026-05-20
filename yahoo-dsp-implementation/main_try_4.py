"""
main.py — Yahoo DSP Reporting Pipeline (Local Entry Point)
===========================================================
Two run modes:
  backfill : dateTypeId=11 (Custom Range) — 13 months lookback with Day granularity
  daily    : dateTypeId=2  (Yesterday)    — single-day extraction with COB check

Usage:
  python main.py --mode backfill
  python main.py --mode daily
  python main.py --mode daily --skip-cob

Environment variables required:
  YAHOO_DSP_CLIENT_ID
  YAHOO_DSP_CLIENT_SECRET

Author  : Priyank Chauhan
Python  : 3.13+
"""

from __future__ import annotations

import os
import sys
import argparse

import pendulum
from loguru import logger

from helpers import YahooHelpers
from reporting import YahooReporting


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         GLOBAL  CONFIGURATION                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

REPORT_TIMEZONE: str = "America/New_York"
ACCOUNT_IDS: list[int] = []            # ← Fill with your Yahoo DSP advertiser account IDs
OUTPUT_DIR: str = "output"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         DIMENSION  TYPE  IDS                               ║
# ║  Each entry maps to one or more columns in the CSV output.                 ║
# ║  "Day" column is populated automatically by intervalTypeId=2.              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

DIMENSION_TYPE_IDS: list[int] = [
    # ── Core Entity Dimensions ────────────────────────────────────────────────
    4,      # Advertiser           → Advertiser ID + Advertiser
    5,      # Campaign             → Campaign ID + Campaign Name
    6,      # Line                 → Line ID + Line
    7,      # Ad                   → Ad ID + Ad
    8,      # Creative             → Creative ID + Creative Name  (Seat only)
    3,      # Creative Size        → Creative Size ID + Creative Size

    # ── Geography ─────────────────────────────────────────────────────────────
    19,     # Country              → Country ID + Country
    20,     # State                → State

    # ── Device & Technology ───────────────────────────────────────────────────
    23,     # Device Category      → Device Category ID + Device Category Name

    # ── Audience / Segment ────────────────────────────────────────────────────
    34,     # Segment              → Segment ID + Segment Name
    63,     # Data Provider        → Data Provider ID + Data Provider
    64,     # Data Type            → Data Type ID + Data Type

    # ── Supply / Inventory ────────────────────────────────────────────────────
    46,     # Exchange             → Exchange ID + Exchange Name
    50,     # Inventory Type       → Inventory Type
    54,     # Inventory Source Type → Inventory Source Type
    55,     # Deal                 → Deal ID + Deal Name
    56,     # Deal Type            → Deal Type ID + Deal Type Name

    # ── Media / Channel ───────────────────────────────────────────────────────
    149,    # Media Type           → Media Type ID + Media Type
    104,    # Channel              → Channel ID + Channel Name
    106,    # Environment Type     → Environment Type
    107,    # Creative Type        → Creative Type ID + Creative Type
    154,    # Supply Type          → Supply Type ID + Supply Type
    166,    # CTV App Name         → CTV App Name
    174,    # Exchange Type        → Exchange Type

    # ── Product / Conversion ──────────────────────────────────────────────────
    1033,   # Product              → Product ID + Product Name

    # ── Unique Reach ──────────────────────────────────────────────────────────
    1225,   # Unique Reach Media Type     → Unique Reach Media Type
    1228,   # Unique Reach Device Category → Unique Reach Device Category
]

# ── HIGH-CARDINALITY DIMENSIONS (add with caution — use filterOptions or Top-N) ──
# 9,      # Domain              → Domain
# 21,     # City                → City ID + City               (500k row limit)
# 22,     # Postal Code         → Zip/Postal Code              (500k row limit)
# 47,     # App/URL             → App/URL                      (500k row limit)
# 282,    # DMA                 → DMA ID + DMA

# ── DEMOGRAPHIC DIMENSIONS (Seat only) ──
# 124,    # Age Group           → Age Group
# 125,    # Gender              → Gender
# 126,    # Income Range        → Income Range
# 127,    # Parental Status     → Parental Status
# 128,    # Ethnicity           → Ethnicity

# ── FREQUENCY BUCKET DIMENSIONS (MUTUALLY EXCLUSIVE — only ONE per request) ──
# 80,     # Line Frequency (Lifetime)       (250k row limit, Seat only)
# 81,     # Campaign Frequency (Lifetime)   (250k row limit, Seat only)
# 82,     # Advertiser Frequency (Lifetime) (250k row limit, Seat only)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                         METRIC  TYPE  IDS                                  ║
# ║  All standard metrics (non-bidding, non-IFO).                              ║
# ║  Row-limited metrics are commented out — enable only if row count < 500k.  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

METRIC_TYPE_IDS: list[int] = [
    # ── Delivery ──────────────────────────────────────────────────────────────
    1,      # Impressions
    2,      # Clicks
    25,     # Click Rate (CTR)
    28,     # Media Cost
    29,     # Data Cost
    30,     # Other Cost
    31,     # Total Cost
    41,     # eCPM
    42,     # eCPC
    62,     # Advertiser Spending

    # ── Conversions ───────────────────────────────────────────────────────────
    11,     # Post Click Conversions
    23,     # Post Imp Conversions
    26,     # Post Imp Conversion Rate
    27,     # Post Click Conversion Rate
    32,     # Post Click Revenue
    33,     # Post Imp Revenue
    43,     # Post Click CPA
    44,     # Post Imp CPA
    45,     # eCPA
    49,     # Total Conversions
    50,     # Total Revenue
    51,     # ROAS

    # ── Revenue / Profit ──────────────────────────────────────────────────────
    63,     # Gross Revenue
    64,     # Net Revenue
    65,     # Platform Access Fees

    # ── Video ─────────────────────────────────────────────────────────────────
    100,    # Video Views
    101,    # Video First Quartiles
    102,    # Video Midpoints
    103,    # Video Third Quartiles
    104,    # Video Completes
    105,    # Video Starts
    106,    # Video Start Rate
    107,    # Video First Quartile Rate
    108,    # Video Midpoint Rate
    109,    # Video Third Quartile Rate
    110,    # Video Complete Rate
    111,    # Video Close
    112,    # Video Skip
    113,    # VCR (Video Completion Rate)
    114,    # CPCV (Cost Per Completed View)
    129,    # Video Skip Rate

    # ── Viewability ───────────────────────────────────────────────────────────
    137,    # Measurability Rate
    138,    # Viewability Rate

    # ── Audio ─────────────────────────────────────────────────────────────────
    168,    # Audio Starts
    169,    # Audio First Quartiles
    171,    # Audio Midpoints
    180,    # Audio Third Quartiles
    184,    # Audio Completes
    188,    # Audio Complete Rate
    192,    # Audio Start Rate
    196,    # Audio First Quartile Rate
    200,    # Audio Midpoint Rate
    204,    # Audio Third Quartile Rate
    208,    # Audio Close
    212,    # Audio Skip
    216,    # Audio Skip Rate

    # ── Advanced Delivery / Profit ────────────────────────────────────────────
    260,    # Supply Cost
    305,    # Gross Profit
    317,    # Partner Cost
    322,    # Customer Revenue
    328,    # Total Data Cost
    329,    # Platform Total Cost
    330,    # Customer Total Cost
    334,    # DSP Fees
    338,    # Brand Safety Cost
    346,    # Verification Cost
    350,    # Political Regulatory Cost
    365,    # Audience Data + DMP Cost

    # ── Conversions Advanced ──────────────────────────────────────────────────
    386,    # Post Click Activity Conversions
    387,    # Post Imp Activity Conversions
    390,    # Total Activity Conversions
    391,    # Post Click Activity Revenue
    392,    # Post Imp Activity Revenue
    393,    # Total Activity Revenue
    394,    # Post Click Activity CPA
    395,    # Post Imp Activity CPA
    396,    # Activity eCPA
    397,    # Post Click Activity ROAS
    398,    # Post Imp Activity ROAS

    # ── Revenue / CPA Metrics ─────────────────────────────────────────────────
    426,    # Total CPA
    430,    # Total ROAS

    # ── Video Extended ────────────────────────────────────────────────────────
    446,    # Video Unmute
    447,    # Video Rewind
    448,    # Video Pause

    # ── CTV (Connected TV) ────────────────────────────────────────────────────
    461,    # Connected TV Impressions
    462,    # Connected TV eCPM
    463,    # Connected TV Total Cost
    464,    # Matchable Connected TV Impressions
    465,    # Connected TV Households Reached
    466,    # Connected TV Views
    467,    # Connected TV VCR
    468,    # Connected TV CPCV
    469,    # Matchable Connected TV Views
    470,    # Connected TV View Households Reached
    471,    # Connected TV Household Frequency
    472,    # Connected TV Household View Frequency
    473,    # Connected TV Impressions (Person Level)
    474,    # Connected TV People Reached
    475,    # Connected TV People Frequency
    476,    # Connected TV People View Frequency

    # ── Misc / Newer ──────────────────────────────────────────────────────────
    523,    # Amazon Conversions
    560,    # Companion CPCV
    564,    # Companion VCR
    570,    # Cost Per Unique Impression

    # ── DOOH & People-level ───────────────────────────────────────────────────
    705,    # DOOH Impressions
    714,    # People Impressions
    715,    # eCPM People
    716,    # People Reached
    717,    # People Frequency
    721,    # Household Impressions
    723,    # Household Reached
    724,    # Household Frequency
]

# ── ROW-LIMITED METRICS (500k / 250k row cap — enable only for low-cardinality reports) ──
# 53,     # Unique Impressions                        (500k row limit)
# 115,    # Unique Video Views                        (500k row limit)
# 134,    # Measured Impressions                      (500k row limit)
# 135,    # Viewable Impressions                      (500k row limit)
# 319,    # Unique Reach Impressions                  (500k row limit)
# 320,    # Unique Reach                              (500k row limit)
# 321,    # Average Frequency                         (500k row limit)
# 399,    # Post Click Activity Unique Converters     (500k row limit)
# 400,    # Post Imp Activity Unique Converters       (500k row limit)
# 431,    # Post Click Unique Converters              (500k row limit)
# 432,    # Post Imp Unique Converters                (500k row limit)
# 433,    # Unique Converters                         (500k row limit)
# 453,    # Video Companion Impressions               (500k row limit)
# 455,    # Video Companion Clicks                    (500k row limit)
# 456,    # Video Companion Click-Through Rate        (500k row limit)
# 457,    # Video Companion Views                     (500k row limit)

# ── BIDDING METRICS (require dataSource=15, only dims 4,5,6,149) ──
# 173,    # Bids
# 254,    # Bid Requests
# 296,    # Win Rate
# 298,    # Average Bid CPM
# 301,    # Average Win CPM
# 302,    # Filtered Bid Requests
# 304,    # Bid Rate
# 371,    # Pre-Ad Server Bid Requests
# 373,    # Pre-Ad Server Filtered Bid Requests
# 375,    # Pre-Ad Server Bids
# 377,    # Pre-Ad Server Bid Rate
# 382,    # Pre-Ad Server No Bid Rate
# 583,    # Bid CPM p95
# 584,    # Bid CPM p5

# ── IFO METRICS (require seat-level feature flag) ──
# 744,    # IFO Unique Reach
# 745,    # IFO Average Frequency
# 746,    # IFO Unique Impressions
# 747,    # IFO Unique Video Views                    (500k row limit)
# 748,    # IFO Unique Audio Listens                  (500k row limit)
# 753,    # IFO Incremental Reach
# 754,    # IFO Incremental Impressions
# 755,    # IFO Target Population Size
# 756,    # IFO On-Target Impressions
# 757,    # IFO On-Target Reach
# 758,    # IFO On-Target Rate
# 759,    # IFO On-Target Reach Rate
# 760,    # IFO On-Target Average Frequency
# 761,    # IFO On-Target eCPM
# 762,    # IFO On-Target Reach eCPM
# 763,    # IFO Reach Rate
# 764,    # IFO Incremental Reach Rate
# 769,    # IFO On-Target Incremental Reach
# 770,    # IFO On-Target Incremental Reach Rate
# 771,    # IFO Incremental Impressions Rate
# 772,    # IFO On-Target Incremental Impressions
# 773,    # IFO On-Target Incremental Impressions Rate


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     BIGQUERY  TARGET  SCHEMA                               ║
# ║  Column names MUST match the Yahoo CSV headers exactly.                    ║
# ║  Run once → inspect actual headers → adjust names if needed.              ║
# ║  NOTE: Some dimensions return two columns (ID + Name).                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

REPORT_SCHEMA: list[dict] = [
    # ── Day column (populated by intervalTypeId=2) ────────────────────────────
    {"name": "Day",                                      "type": "STRING",  "mode": "NULLABLE"},

    # ── Dimension columns ─────────────────────────────────────────────────────
    # Dim 4: Advertiser
    {"name": "Advertiser ID",                            "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Advertiser",                               "type": "STRING",  "mode": "NULLABLE"},
    # Dim 5: Campaign
    {"name": "Campaign ID",                              "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Campaign Name",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 6: Line
    {"name": "Line ID",                                  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Line",                                     "type": "STRING",  "mode": "NULLABLE"},
    # Dim 7: Ad
    {"name": "Ad ID",                                    "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Ad",                                       "type": "STRING",  "mode": "NULLABLE"},
    # Dim 8: Creative
    {"name": "Creative ID",                              "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Creative Name",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 3: Creative Size
    {"name": "Creative Size ID",                         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Creative Size",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 19: Country
    {"name": "Country ID",                               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Country",                                  "type": "STRING",  "mode": "NULLABLE"},
    # Dim 20: State
    {"name": "State",                                    "type": "STRING",  "mode": "NULLABLE"},
    # Dim 23: Device Category
    {"name": "Device Category ID",                       "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Device Category Name",                     "type": "STRING",  "mode": "NULLABLE"},
    # Dim 34: Segment
    {"name": "Segment ID",                               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Segment Name",                             "type": "STRING",  "mode": "NULLABLE"},
    # Dim 63: Data Provider
    {"name": "Data Provider ID",                         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Data Provider",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 64: Data Type
    {"name": "Data Type ID",                             "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Data Type",                                "type": "STRING",  "mode": "NULLABLE"},
    # Dim 46: Exchange
    {"name": "Exchange ID",                              "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Exchange",                                 "type": "STRING",  "mode": "NULLABLE"},
    # Dim 50: Inventory Type
    {"name": "Inventory Type",                           "type": "STRING",  "mode": "NULLABLE"},
    # Dim 54: Inventory Source Type
    {"name": "Inventory Source Type",                    "type": "STRING",  "mode": "NULLABLE"},
    # Dim 55: Deal
    {"name": "Deal ID",                                  "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Deal Name",                                "type": "STRING",  "mode": "NULLABLE"},
    # Dim 56: Deal Type
    {"name": "Deal Type ID",                             "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Deal Type",                                "type": "STRING",  "mode": "NULLABLE"},
    # Dim 149: Media Type
    {"name": "Media Type ID",                            "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Media Type",                               "type": "STRING",  "mode": "NULLABLE"},
    # Dim 104: Channel
    {"name": "Channel ID",                               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Channel Name",                             "type": "STRING",  "mode": "NULLABLE"},
    # Dim 106: Environment Type
    {"name": "Environment Type",                         "type": "STRING",  "mode": "NULLABLE"},
    # Dim 107: Creative Type
    {"name": "Creative Type ID",                         "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Creative Type",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 154: Supply Type
    {"name": "Supply Type ID",                           "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Supply Type",                              "type": "STRING",  "mode": "NULLABLE"},
    # Dim 166: CTV App Name
    {"name": "CTV App Name",                             "type": "STRING",  "mode": "NULLABLE"},
    # Dim 174: Exchange Type
    {"name": "Exchange Type",                            "type": "STRING",  "mode": "NULLABLE"},
    # Dim 1033: Product
    {"name": "Product ID",                               "type": "STRING",  "mode": "NULLABLE"},
    {"name": "Product Name",                             "type": "STRING",  "mode": "NULLABLE"},
    # Dim 1225: Unique Reach Media Type
    {"name": "Unique Reach Media Type",                  "type": "STRING",  "mode": "NULLABLE"},
    # Dim 1228: Unique Reach Device Category
    {"name": "Unique Reach Device Category",             "type": "STRING",  "mode": "NULLABLE"},

    # ── Metric columns (INTEGER for counts, FLOAT64 for rates/money) ─────────
    # Delivery
    {"name": "Impressions",                              "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Clicks",                                   "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Click Rate",                               "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Media Cost",                               "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Data Cost",                                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Other Cost",                               "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Total Cost",                               "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "eCPM",                                     "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "eCPC",                                     "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Advertiser Spending",                      "type": "FLOAT64", "mode": "NULLABLE"},
    # Conversions
    {"name": "Post Click Conversions",                   "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Post Imp Conversions",                     "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Post Imp Conversion Rate",                 "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Click Conversion Rate",               "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Click Revenue",                       "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Imp Revenue",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Click CPA",                           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Imp CPA",                             "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "eCPA",                                     "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Total Conversions",                        "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Total Revenue",                            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "ROAS",                                     "type": "FLOAT64", "mode": "NULLABLE"},
    # Revenue / Profit
    {"name": "Gross Revenue",                            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Net Revenue",                              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Platform Access Fees",                     "type": "FLOAT64", "mode": "NULLABLE"},
    # Video
    {"name": "Video Views",                              "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video First Quartiles",                    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Midpoints",                          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Third Quartiles",                    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Completes",                          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Starts",                             "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Start Rate",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video First Quartile Rate",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Midpoint Rate",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Third Quartile Rate",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Complete Rate",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Close",                              "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Skip",                               "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "VCR",                                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "CPCV",                                     "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Video Skip Rate",                          "type": "FLOAT64", "mode": "NULLABLE"},
    # Viewability
    {"name": "Measurability Rate",                       "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Viewability Rate",                         "type": "FLOAT64", "mode": "NULLABLE"},
    # Audio
    {"name": "Audio Starts",                             "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio First Quartiles",                    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Midpoints",                          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Third Quartiles",                    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Completes",                          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Complete Rate",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audio Start Rate",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audio First Quartile Rate",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audio Midpoint Rate",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audio Third Quartile Rate",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audio Close",                              "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Skip",                               "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Audio Skip Rate",                          "type": "FLOAT64", "mode": "NULLABLE"},
    # Advanced Delivery / Profit
    {"name": "Supply Cost",                              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Gross Profit",                             "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Partner Cost",                             "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Customer Revenue",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Total Data Cost",                          "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Platform Total Cost",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Customer Total Cost",                      "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "DSP Fees",                                 "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Brand Safety Cost",                        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Verification Cost",                        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Political Regulatory Cost",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Audience Data + DMP Cost",                 "type": "FLOAT64", "mode": "NULLABLE"},
    # Conversions Advanced
    {"name": "Post Click Activity Conversions",          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Post Imp Activity Conversions",            "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Total Activity Conversions",               "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Post Click Activity Revenue",              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Imp Activity Revenue",                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Total Activity Revenue",                   "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Click Activity CPA",                  "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Imp Activity CPA",                    "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Activity eCPA",                            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Click Activity ROAS",                 "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Post Imp Activity ROAS",                   "type": "FLOAT64", "mode": "NULLABLE"},
    # Revenue / CPA
    {"name": "Total CPA",                                "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Total ROAS",                               "type": "FLOAT64", "mode": "NULLABLE"},
    # Video Extended
    {"name": "Video Unmute",                             "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Rewind",                             "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Video Pause",                              "type": "INTEGER", "mode": "NULLABLE"},
    # CTV
    {"name": "Connected TV Impressions",                 "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV eCPM",                        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Connected TV Total Cost",                  "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Matchable Connected TV Impressions",       "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV Households Reached",          "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV Views",                       "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV VCR",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Connected TV CPCV",                        "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Matchable Connected TV Views",             "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV View Households Reached",     "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV Household Frequency",         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Connected TV Household View Frequency",    "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Connected TV Impressions (Person Level)",  "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV People Reached",              "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Connected TV People Frequency",            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Connected TV People View Frequency",       "type": "FLOAT64", "mode": "NULLABLE"},
    # Misc / Newer
    {"name": "Amazon Conversions",                       "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Companion CPCV",                           "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Companion VCR",                            "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Cost Per Unique Impression",               "type": "FLOAT64", "mode": "NULLABLE"},
    # DOOH & People-level
    {"name": "DOOH Impressions",                         "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "People Impressions",                       "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "eCPM People",                              "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "People Reached",                           "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "People Frequency",                         "type": "FLOAT64", "mode": "NULLABLE"},
    {"name": "Household Impressions",                    "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Household Reached",                        "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "Household Frequency",                      "type": "FLOAT64", "mode": "NULLABLE"},

    # ── Metadata columns (appended by pipeline) ──────────────────────────────
    {"name": "ingestion_timestamp",                      "type": "STRING",  "mode": "NULLABLE"},
    {"name": "lookback_date_range",                      "type": "STRING",  "mode": "NULLABLE"},
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                     CONFIG  BUILDERS                                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def build_backfill_config() -> dict:
    """
    Build report configuration for BACKFILL mode.

    - dateTypeId  : 11 (Custom Range)
    - Start date  : 13 months before yesterday
    - End date    : yesterday 23:59:59
    - Interval    : Day (intervalTypeId=2)

    Dates include the timezone offset so Yahoo resolves DST correctly.
    Pendulum handles DST transitions automatically.

    Returns
    -------
    dict
        Full report creation payload.
    """
    tz = pendulum.timezone(REPORT_TIMEZONE)
    end_dt = pendulum.yesterday(tz)
    start_dt = end_dt.subtract(months=13)

    # Format with timezone offset — pendulum resolves DST automatically
    # "2024-04-05T00:00:00-04:00" (EDT) or "2024-01-05T00:00:00-05:00" (EST)
    start_str = start_dt.start_of("day").format("YYYY-MM-DD[T]HH:mm:ssZ")
    end_str = end_dt.end_of("day").format("YYYY-MM-DD[T]HH:mm:ssZ")

    config = {
        "reportOption": {
            "timezone": REPORT_TIMEZONE,
            "currency": 4,                      # USD
            "dimensionTypeIds": DIMENSION_TYPE_IDS,
            "metricTypeIds": METRIC_TYPE_IDS,
            "filterOptions": [],
            "accountIds": ACCOUNT_IDS,
        },
        "dateTypeId": 11,                        # Custom Range
        "intervalTypeId": 2,                     # Day
        "startDate": start_str,
        "endDate": end_str,
    }

    logger.info(
        f"BACKFILL config built:\n"
        f"  Date range : {start_str} → {end_str}\n"
        f"  Interval   : Day (intervalTypeId=2)\n"
        f"  Timezone   : {REPORT_TIMEZONE}\n"
        f"  Dimensions : {len(DIMENSION_TYPE_IDS)}\n"
        f"  Metrics    : {len(METRIC_TYPE_IDS)}\n"
        f"  Accounts   : {ACCOUNT_IDS or 'all (default seat)'}"
    )
    return config


def build_daily_config() -> dict:
    """
    Build report configuration for DAILY mode.

    - dateTypeId  : 2 (Yesterday)
    - Interval    : Day (intervalTypeId=2)

    No startDate / endDate needed — "Yesterday" is relative to the timezone.

    Returns
    -------
    dict
        Full report creation payload.
    """
    config = {
        "reportOption": {
            "timezone": REPORT_TIMEZONE,
            "currency": 4,                      # USD
            "dimensionTypeIds": DIMENSION_TYPE_IDS,
            "metricTypeIds": METRIC_TYPE_IDS,
            "filterOptions": [],
            "accountIds": ACCOUNT_IDS,
        },
        "dateTypeId": 2,                         # Yesterday
        "intervalTypeId": 2,                     # Day
    }

    logger.info(
        f"DAILY config built:\n"
        f"  Date range : Yesterday (dateTypeId=2)\n"
        f"  Interval   : Day (intervalTypeId=2)\n"
        f"  Timezone   : {REPORT_TIMEZONE}\n"
        f"  Dimensions : {len(DIMENSION_TYPE_IDS)}\n"
        f"  Metrics    : {len(METRIC_TYPE_IDS)}\n"
        f"  Accounts   : {ACCOUNT_IDS or 'all (default seat)'}"
    )
    return config


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                           ENTRY  POINT                                     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    """CLI entry point for the Yahoo DSP Reporting pipeline."""

    # ── Argument parsing ──────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Yahoo DSP Reporting Pipeline — Local Development Mode",
    )
    parser.add_argument(
        "--mode",
        choices=["backfill", "daily"],
        default="daily",
        help="backfill = Custom Range (13-month lookback) | daily = Yesterday",
    )
    parser.add_argument(
        "--skip-cob",
        action="store_true",
        default=False,
        help="Skip the Close-of-Books (COB) check before extraction.",
    )
    args = parser.parse_args()

    # ── Validate environment variables ────────────────────────────────────────
    client_id = os.getenv("YAHOO_DSP_CLIENT_ID", "").strip()
    client_secret = os.getenv("YAHOO_DSP_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        logger.error(
            "Missing required environment variables: "
            "YAHOO_DSP_CLIENT_ID and/or YAHOO_DSP_CLIENT_SECRET.\n"
            "Set them before running:  "
            "export YAHOO_DSP_CLIENT_ID=xxx  &&  export YAHOO_DSP_CLIENT_SECRET=yyy"
        )
        sys.exit(1)

    logger.info(
        f"Starting Yahoo DSP Reporting Pipeline\n"
        f"  Mode       : {args.mode}\n"
        f"  Skip COB   : {args.skip_cob}\n"
        f"  Output dir : {OUTPUT_DIR}"
    )

    # ── Initialise helper and reporting instances ─────────────────────────────
    helpers = YahooHelpers()
    reporter = YahooReporting(helpers)

    # ── Build config based on run mode ────────────────────────────────────────
    if args.mode == "backfill":
        config = build_backfill_config()
    else:
        config = build_daily_config()

    # ── Execute the pipeline ──────────────────────────────────────────────────
    df = reporter.run_pipeline(
        config=config,
        schema=REPORT_SCHEMA,
        output_dir=OUTPUT_DIR,
        skip_cob=args.skip_cob,
        advertiser_ids_for_cob=ACCOUNT_IDS if ACCOUNT_IDS else None,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.success(
        f"✅ Pipeline complete — {df.shape[0]:,} rows × {df.shape[1]} cols\n"
        f"Files saved to: {os.path.abspath(OUTPUT_DIR)}/"
    )


if __name__ == "__main__":
    main()
