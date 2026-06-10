# Ray Large Data Processing Example

This repository contains examples of how to use Ray to handle large data processing tasks in Python.

## Overview

The example demonstrates three key patterns for using Ray with large datasets:

1. **Task-based processing**: Distributing work across multiple Ray tasks
2. **Actor-based processing**: Using Ray actors for stateful computations
3. **Ray Data API**: Leveraging Ray's data processing capabilities for pandas DataFrames

## Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Example

Execute the main script:

```bash
python ray_large_data_example.py
```

## Key Concepts Demonstrated

### Task-Based Processing
- Breaking large datasets into chunks
- Submitting processing tasks asynchronously
- Collecting and aggregating results

### Actor-Based Processing
- Maintaining state across multiple operations
- Using actors for more complex processing workflows
- Retrieving accumulated results

### Ray Data API
- Converting pandas DataFrames to Ray datasets
- Performing distributed groupby operations
- Aggregating results efficiently

## Customization

You can adjust the following parameters in the script:
- `chunk_size`: Controls how large each data chunk is
- `size`: Controls the size of the generated sample data
- Processing functions: Modify to implement your specific data processing logic

## Performance Benefits

Using Ray for large data processing provides several advantages:
- Parallel execution across multiple CPU cores
- Efficient memory management
- Simplified distribution of work
- Reduced processing time for large datasets