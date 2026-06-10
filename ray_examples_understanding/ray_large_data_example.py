"""
Example of using Ray for handling large data processing in Python.
This script demonstrates various techniques for distributing computations 
and managing large datasets with Ray.
"""

import ray
import numpy as np
import pandas as pd
from typing import List
import time


# Initialize Ray
ray.init(ignore_reinit_error=True)


@ray.remote
def process_chunk(data_chunk: np.ndarray) -> dict:
    """
    Process a chunk of data remotely.
    
    Args:
        data_chunk: A numpy array representing a chunk of data
        
    Returns:
        Dictionary with processed results
    """
    # Simulate some computation
    mean_val = np.mean(data_chunk)
    sum_val = np.sum(data_chunk)
    max_val = np.max(data_chunk)
    min_val = np.min(data_chunk)
    
    # Add some artificial delay to simulate computation time
    time.sleep(0.1)
    
    return {
        "mean": mean_val,
        "sum": sum_val,
        "max": max_val,
        "min": min_val,
        "count": len(data_chunk)
    }


@ray.remote
class DataProcessorActor:
    """
    An actor class for stateful data processing.
    Actors maintain state between method calls.
    """
    
    def __init__(self):
        self.processed_count = 0
        self.results = []
    
    def process_data(self, data_chunk: np.ndarray) -> dict:
        """Process a chunk of data and maintain state."""
        result = {
            "chunk_id": self.processed_count,
            "mean": np.mean(data_chunk),
            "std": np.std(data_chunk),
            "shape": data_chunk.shape
        }
        
        self.results.append(result)
        self.processed_count += 1
        
        return result
    
    def get_stats(self) -> dict:
        """Return accumulated statistics."""
        return {
            "total_processed": self.processed_count,
            "results": self.results
        }


def process_large_array_with_tasks(data: np.ndarray, chunk_size: int = 10000) -> dict:
    """
    Process a large numpy array using Ray tasks.
    
    Args:
        data: Large numpy array to process
        chunk_size: Size of each chunk to process
        
    Returns:
        Aggregated results from all chunks
    """
    print(f"Processing array of shape {data.shape} with chunk size {chunk_size}")
    
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Split data into {len(chunks)} chunks")
    
    # Submit tasks for each chunk
    futures = [process_chunk.remote(chunk) for chunk in chunks]
    
    # Collect results
    results = ray.get(futures)
    
    # Aggregate results
    total_sum = sum(r["sum"] for r in results)
    total_count = sum(r["count"] for r in results)
    overall_mean = total_sum / total_count if total_count > 0 else 0
    overall_max = max(r["max"] for r in results)
    overall_min = min(r["min"] for r in results)
    
    return {
        "overall_mean": overall_mean,
        "overall_max": overall_max,
        "overall_min": overall_min,
        "total_count": total_count,
        "chunk_results": results
    }


def process_large_dataset_with_actor(data: np.ndarray, chunk_size: int = 10000) -> dict:
    """
    Process a large dataset using Ray actors for stateful processing.
    
    Args:
        data: Large numpy array to process
        chunk_size: Size of each chunk to process
        
    Returns:
        Final results from the actor
    """
    print(f"Processing dataset with actor, shape {data.shape} with chunk size {chunk_size}")
    
    # Create actor
    processor = DataProcessorActor.remote()
    
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Split data into {len(chunks)} chunks")
    
    # Process chunks with the actor
    futures = [processor.process_data.remote(chunk) for chunk in chunks]
    
    # Wait for all processing to complete
    ray.get(futures)
    
    # Get final stats
    stats = ray.get(processor.get_stats.remote())
    return stats


def create_sample_data(size: int = 1000000) -> np.ndarray:
    """Create sample numerical data for processing."""
    print(f"Creating sample data of size {size}")
    # Create a large array with some structure
    data = np.random.randn(size)
    # Add some trend
    data += np.linspace(0, 10, size)
    return data


def main():
    """Main function demonstrating different Ray usage patterns."""
    print("=" * 60)
    print("Ray Large Data Processing Examples")
    print("=" * 60)
    
    # Example 1: Process large array with tasks
    print("\n1. Processing large array with Ray tasks:")
    large_array = create_sample_data(100000)
    start_time = time.time()
    task_results = process_large_array_with_tasks(large_array, chunk_size=20000)
    task_time = time.time() - start_time
    print(f"Task-based processing took {task_time:.2f} seconds")
    print(f"Overall mean: {task_results['overall_mean']:.4f}")
    print(f"Value range: [{task_results['overall_min']:.4f}, {task_results['overall_max']:.4f}]")
    
    # Example 2: Process with actor
    print("\n2. Processing with Ray actor:")
    start_time = time.time()
    actor_results = process_large_dataset_with_actor(large_array, chunk_size=20000)
    actor_time = time.time() - start_time
    print(f"Actor-based processing took {actor_time:.2f} seconds")
    print(f"Total chunks processed: {actor_results['total_processed']}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()