#!/bin/bash

# Script to run Ray examples

echo "Ray Large Data Processing Examples Runner"
echo "========================================"

# Activate virtual environment if it exists
if [ -d "ray_env" ]; then
    echo "Activating virtual environment..."
    source ray_env/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Display available examples
echo ""
echo "Available examples:"
echo "1. Simple Ray example (recommended starting point)"
echo "2. Basic Ray large data processing"
echo "3. Advanced Ray features with fault tolerance"
echo "4. Data science example with pandas/scikit-learn (may have issues)"

# Prompt user for selection
echo ""
read -p "Enter example number to run (1-4): " choice

case $choice in
    1)
        echo "Running simple Ray example..."
        python simple_ray_example.py
        ;;
    2)
        echo "Running basic Ray large data example..."
        python ray_large_data_example.py
        ;;
    3)
        echo "Running advanced Ray example..."
        python advanced_ray_example.py
        ;;
    4)
        echo "Running data science example..."
        echo "Note: This example may have serialization issues in some environments."
        python ray_data_science_example.py
        ;;
    *)
        echo "Invalid choice. Please run the script again and select 1-4."
        exit 1
        ;;
esac

echo ""
echo "Example completed!"