import json
from fastavro import writer, parse_schema


# -----------------------------
# BigQuery → Avro type mapping
# -----------------------------
def bq_to_avro_type(bq_field):
    bq_type = bq_field["type"]
    mode = bq_field["mode"]
    fields = bq_field.get("fields", [])

    # Map primitive types
    type_mapping = {
        "STRING": "string",
        "INTEGER": "long",
        "FLOAT": "double",
        "BOOLEAN": "boolean",
        "DATE": {"type": "int", "logicalType": "date"},
    }

    if bq_type == "RECORD":
        avro_type = {
            "type": "record",
            "name": bq_field["name"] + "_record",
            "fields": [
                {
                    "name": sub["name"],
                    "type": bq_to_avro_type(sub)
                }
                for sub in fields
            ],
        }
    else:
        avro_type = type_mapping.get(bq_type, "string")

    # Handle mode
    if mode == "REPEATED":
        avro_type = {"type": "array", "items": avro_type}

    elif mode == "NULLABLE":
        avro_type = ["null", avro_type]

    return avro_type


# -----------------------------
# Build full Avro schema
# -----------------------------
def build_avro_schema(bq_schema):
    return {
        "type": "record",
        "name": "RootRecord",
        "fields": [
            {
                "name": field["name"],
                "type": bq_to_avro_type(field),
                "default": None if field["mode"] == "NULLABLE" else None,
            }
            for field in bq_schema
        ],
    }


# -----------------------------
# Your BigQuery schema
# -----------------------------
bq_schema = [
  {
    "name": "event_date",
    "mode": "NULLABLE",
    "type": "DATE",
    "description": "",
    "fields": []
  },
  {
    "name": "property_id",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "property_name",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "account_id",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "account_name",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "event_timestamp",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "event_name",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "event_params",
    "mode": "REPEATED",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "key",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "value",
        "mode": "NULLABLE",
        "type": "RECORD",
        "description": "",
        "fields": [
          {
            "name": "string_value",
            "mode": "NULLABLE",
            "type": "STRING",
            "description": "",
            "fields": []
          },
          {
            "name": "int_value",
            "mode": "NULLABLE",
            "type": "INTEGER",
            "description": "",
            "fields": []
          },
          {
            "name": "float_value",
            "mode": "NULLABLE",
            "type": "FLOAT",
            "description": "",
            "fields": []
          },
          {
            "name": "double_value",
            "mode": "NULLABLE",
            "type": "FLOAT",
            "description": "",
            "fields": []
          }
        ]
      }
    ]
  },
  {
    "name": "event_previous_timestamp",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "event_value_in_usd",
    "mode": "NULLABLE",
    "type": "FLOAT",
    "description": "",
    "fields": []
  },
  {
    "name": "event_bundle_sequence_id",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "event_server_timestamp_offset",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "user_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "user_pseudo_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "privacy_info",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "analytics_storage",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "ads_storage",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "uses_transient_token",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "user_properties",
    "mode": "REPEATED",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "key",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "value",
        "mode": "NULLABLE",
        "type": "RECORD",
        "description": "",
        "fields": [
          {
            "name": "string_value",
            "mode": "NULLABLE",
            "type": "STRING",
            "description": "",
            "fields": []
          },
          {
            "name": "int_value",
            "mode": "NULLABLE",
            "type": "INTEGER",
            "description": "",
            "fields": []
          },
          {
            "name": "float_value",
            "mode": "NULLABLE",
            "type": "FLOAT",
            "description": "",
            "fields": []
          },
          {
            "name": "double_value",
            "mode": "NULLABLE",
            "type": "FLOAT",
            "description": "",
            "fields": []
          },
          {
            "name": "set_timestamp_micros",
            "mode": "NULLABLE",
            "type": "INTEGER",
            "description": "",
            "fields": []
          }
        ]
      }
    ]
  },
  {
    "name": "user_first_touch_timestamp",
    "mode": "NULLABLE",
    "type": "INTEGER",
    "description": "",
    "fields": []
  },
  {
    "name": "user_ltv",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "revenue",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "currency",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "device",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "category",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "mobile_brand_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "mobile_model_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "mobile_marketing_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "mobile_os_hardware_model",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "operating_system",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "operating_system_version",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "vendor_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "advertising_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "LANGUAGE",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "is_limited_ad_tracking",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "time_zone_offset_seconds",
        "mode": "NULLABLE",
        "type": "INTEGER",
        "description": "",
        "fields": []
      },
      {
        "name": "browser",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "browser_version",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "web_info",
        "mode": "NULLABLE",
        "type": "RECORD",
        "description": "",
        "fields": [
          {
            "name": "browser",
            "mode": "NULLABLE",
            "type": "STRING",
            "description": "",
            "fields": []
          },
          {
            "name": "browser_version",
            "mode": "NULLABLE",
            "type": "STRING",
            "description": "",
            "fields": []
          },
          {
            "name": "hostname",
            "mode": "NULLABLE",
            "type": "STRING",
            "description": "",
            "fields": []
          }
        ]
      }
    ]
  },
  {
    "name": "geo",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "city",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "country",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "continent",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "region",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "sub_continent",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "metro",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "app_info",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "version",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "install_store",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "firebase_app_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "install_source",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "traffic_source",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "medium",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "source",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "stream_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "platform",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "",
    "fields": []
  },
  {
    "name": "event_dimensions",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "hostname",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "ecommerce",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "total_item_quantity",
        "mode": "NULLABLE",
        "type": "INTEGER",
        "description": "",
        "fields": []
      },
      {
        "name": "purchase_revenue_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "purchase_revenue",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "refund_value_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "refund_value",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "shipping_value_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "shipping_value",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "tax_value_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "tax_value",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "unique_items",
        "mode": "NULLABLE",
        "type": "INTEGER",
        "description": "",
        "fields": []
      },
      {
        "name": "transaction_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "items",
    "mode": "REPEATED",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "item_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_brand",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_variant",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_category",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_category2",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_category3",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_category4",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_category5",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "price_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "price",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "quantity",
        "mode": "NULLABLE",
        "type": "INTEGER",
        "description": "",
        "fields": []
      },
      {
        "name": "item_revenue_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "item_revenue",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "item_refund_in_usd",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "item_refund",
        "mode": "NULLABLE",
        "type": "FLOAT",
        "description": "",
        "fields": []
      },
      {
        "name": "coupon",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "affiliation",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "location_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_list_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_list_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "item_list_index",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "promotion_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "promotion_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "creative_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "creative_slot",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "collected_traffic_source",
    "mode": "NULLABLE",
    "type": "RECORD",
    "description": "",
    "fields": [
      {
        "name": "manual_campaign_id",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "manual_campaign_name",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "manual_source",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "manual_medium",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "manual_term",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "manual_content",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "gclid",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "dclid",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      },
      {
        "name": "srsltid",
        "mode": "NULLABLE",
        "type": "STRING",
        "description": "",
        "fields": []
      }
    ]
  },
  {
    "name": "is_active_user",
    "mode": "NULLABLE",
    "type": "BOOLEAN",
    "description": "",
    "fields": []
  }
]   # <-- your large schema list


# Build Avro schema
avro_schema_dict = build_avro_schema(bq_schema)

# Parse with fastavro
parsed_schema = parse_schema(avro_schema_dict)


# -----------------------------
# Write empty Avro file (schema only)
# -----------------------------
with open("output.avro", "wb") as f_out:
    writer(f_out, parsed_schema, [])  # No data records yet


print("✅ Avro schema successfully generated and written to output.avro")
print(json.dumps(avro_schema_dict, indent=2))