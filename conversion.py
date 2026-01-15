# import json
# with open('All_India_pincode_Boundary-19312.geojson', 'r') as f:
#     data = json.load(f)

# # Process the data as a standard Python dictionary/list
# # For example, to save the 'features' list to a new file:
# with open('output_pincode_lat_long_data.json', 'w') as f:
#     json.dump(data['features'], f, indent=4)




import json

def geojson_to_ndjson(input_path: str, output_path: str) -> None:
    """
    Convert GeoJSON FeatureCollection to NDJSON (one Feature per line)
    Compatible with BigQuery external JSON tables
    """

    with open(input_path, "r", encoding="utf-8") as infile:
        geojson = json.load(infile)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("Input file is not a GeoJSON FeatureCollection")

    features = geojson.get("features", [])
    if not features:
        raise ValueError("No features found in GeoJSON")

    with open(output_path, "w", encoding="utf-8") as outfile:
        for feature in features:
            outfile.write(json.dumps(feature, ensure_ascii=False))
            outfile.write("\n")

    print(f"Converted {len(features)} features → {output_path}")



geojson_to_ndjson(
    input_path="All_India_pincode_Boundary-19312.geojson",
    output_path="output_pincode_lat_long_data.ndjson"
)