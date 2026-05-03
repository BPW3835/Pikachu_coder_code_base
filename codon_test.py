# from time import time

# def fib(n):
#     return n if n < 2 else fib(n - 1) + fib(n - 2)

# t0 = time()
# ans = fib(40)
# t1 = time()
# print(f'Computed fib(40) = {ans} in {t1 - t0} seconds.')


import json
from pathlib import Path

INPUT_FILE = "sparsity_lookup_config.json"
OUTPUT_FILE = "sparsity_lookup_config.ndjson"

# Fields that are currently stored as JSON strings in your file
JSON_STRING_FIELDS = {
    "source_layer_datatype_mapping",
    "metric_columns",
    "target_layer_datatype_mapping",
    "source_to_target_column_mapping",
}

# Fields that should be booleans if they come as strings
BOOLEAN_FIELDS = {
    "is_active",
}


def normalize_record(record: dict) -> dict:
    """
    Normalize one record:
    - convert boolean-like strings to real booleans
    - convert JSON-string fields to actual dict objects where possible
    """
    normalized = dict(record)

    # Convert booleans stored as strings
    for field in BOOLEAN_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                normalized[field] = True
            elif lowered == "false":
                normalized[field] = False

    # Convert JSON strings to actual JSON objects
    for field in JSON_STRING_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            value = value.strip()
            if value:
                try:
                    normalized[field] = json.loads(value)
                except json.JSONDecodeError:
                    # keep original value if it is not valid JSON
                    normalized[field] = value

    return normalized


def convert_json_to_ndjson(input_path: str, output_path: str) -> None:
    """
    Convert standard JSON to NDJSON.
    Supports:
      - top-level array of objects
      - single object
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Normalize to a list of records
    if isinstance(data, dict):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        raise ValueError("Input JSON must be either an object or an array of objects.")

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("Each element in the JSON array must be an object.")
            normalized = normalize_record(record)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")

    print(f"NDJSON written to: {output_path}")


if __name__ == "__main__":
    convert_json_to_ndjson(INPUT_FILE, OUTPUT_FILE)