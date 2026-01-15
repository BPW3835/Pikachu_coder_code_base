"""Metrics for evals."""

import os

# Initialize an empty list to store metrics
metrics = []

# Define the directory where prompt files are stored
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Iterate through all files in the prompts directory
for file in os.listdir(PROMPTS_DIR):
    # Check if the file has a .md extension
    if file.endswith(".md"):
        # Read the content of the .md file and append it to the metrics list
        metrics.append({
            "name": file.replace(".md", ""),  # Use the file name (without extension) as the metric name
            "prompt": open(os.path.join(PROMPTS_DIR, file), "r").read()  # Read the file content as the prompt
        })
