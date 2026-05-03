import requests
import boto3
import json
import time

# --- CONFIGURATION ---
API_ENDPOINT = "http://127.0.0.1:5000/get-events"
STREAM_NAME = "ecommerce-raw-stream"
REGION_NAME = "us-east-1"

kinesis_client = boto3.client('kinesis', region_name=REGION_NAME)

def fetch_events_from_api():
    try:
        response = requests.get(API_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        return []
    
def stream_to_kinesis(events):
    if not events:
        return
    
    records = []
    for event in events:
        records.append({
            'Data': (json.dumps(event, ensure_ascii=False) + '\n').encode('utf-8'),
            'PartitionKey': event['customer_id']
        })
        
    try:
        response = kinesis_client.put_records(
            StreamName = STREAM_NAME,
            Records = records
        )
        failed_count = response.get('FailedRecordCount', 0)
        if failed_count > 0:
            print(f"Warning: {failed_count} records failed to upload.")
        else:
            print(f"Successfully sent {len(events)} events to Kinesis.")
    
    except Exception as e:
        print(f"Error sending to Kinesis: {e}")
        
def run_producer():
    
    try:
        while True:
            events = fetch_events_from_api()
            
            if events:
                stream_to_kinesis(events)
            
            time.sleep(2)
        
    except KeyboardInterrupt:
        print("\nProducer stopped by user.")
    
if __name__ == "__main__":
    run_producer()