from flask import Flask, jsonify
import random
import uuid
from datetime import datetime
from faker import Faker

app = Flask(__name__)
fake = Faker()

# --- MASTER DATA POOL (Consistency) ---
# Ensuring customer data remains consistent across sessions
CUSTOMERS = []
for i in range(1, 101):
    CUSTOMERS.append({
        "customer_id": f"CUST-{i:03d}",
        "customer_name": fake.name(),
        "email": fake.email(),
        "country": random.choice(["US", "UK"])
    })

PRODUCT_POOL = {
    "Electronics": ["Smartphone", "Laptop", "Smartwatch", "Headphones", "Tablet"],
    "Home & Kitchen": ["Coffee Maker", "Air Fryer", "Blender", "Toaster", "Vacuum Cleaner"],
    "Fashion": ["T-Shirt", "Jeans", "Sneakers", "Jacket", "Watch"],
    "Books": ["Sci-Fi Novel", "Cookbook", "History Book", "Biography", "Comic Book"]
}
CATEGORIES = list(PRODUCT_POOL.keys())

def generate_single_event():
    """Helper function to generate a single synthetic e-commerce event."""
    selected_customer = random.choice(CUSTOMERS)
    category = random.choice(CATEGORIES)
    product_name = random.choice(PRODUCT_POOL[category])
    
    # 10% chance of None for data quality testing purposes
    user_age = random.choice([None] + [random.randint(18, 75)] * 9)
    is_overdue = random.random() < 0.10
    
    return {
        "event_id": str(uuid.uuid4()),
        "customer_id": selected_customer["customer_id"],
        "customer_name": selected_customer["customer_name"],
        "email": selected_customer["email"],
        "phone_number": fake.phone_number(), 
        "country": selected_customer["country"],
        "age": user_age,
        "category": category,
        "product_name": product_name,
        "price": round(random.uniform(20.0, 1500.0), 2),
        "is_premium": random.choice([True, False]),
        "is_overdue": is_overdue,             
        "event_type": random.choice(['view', 'click', 'purchase']),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.route('/get-events', methods=['GET'])
def get_events():
    """Endpoint that returns a batch of 100 synthetic events."""
    batch_size = 100
    events_batch = [generate_single_event() for _ in range(batch_size)]
    return jsonify(events_batch)

@app.route('/', methods=['GET'])
def health_check():
    """Simple root endpoint for connectivity testing."""
    return jsonify({"status": "active", "endpoint": "/get-events"})

if __name__ == "__main__":
    # Server runs on port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
    
# http://127.0.0.1:5000/get-events