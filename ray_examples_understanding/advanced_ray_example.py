"""
Advanced example of using Ray for handling very large data processing with fault tolerance.
This script demonstrates handling failures, checkpointing, and more sophisticated data processing patterns.
"""

import ray
import numpy as np
import time
import os
from typing import List, Dict, Any


# Initialize Ray
ray.init(ignore_reinit_error=True)


@ray.remote(max_retries=3)
def robust_process_chunk(data_chunk: np.ndarray, chunk_id: int) -> Dict[str, Any]:
    """
    Process a chunk of data with retry logic.
    
    Args:
        data_chunk: A numpy array representing a chunk of data
        chunk_id: Identifier for the chunk
        
    Returns:
        Dictionary with processed results
    """
    # Simulate potential failure
    if np.random.random() < 0.1:  # 10% chance of failure
        raise Exception(f"Simulated failure in chunk {chunk_id}")
    
    # Simulate computation
    mean_val = np.mean(data_chunk)
    sum_val = np.sum(data_chunk)
    max_val = np.max(data_chunk)
    min_val = np.min(data_chunk)
    
    # Add some artificial delay
    time.sleep(np.random.uniform(0.05, 0.2))
    
    return {
        "chunk_id": chunk_id,
        "mean": mean_val,
        "sum": sum_val,
        "max": max_val,
        "min": min_val,
        "count": len(data_chunk)
    }


@ray.remote(num_cpus=1, max_restarts=5)
class CheckpointedDataProcessor:
    """
    An actor class with checkpointing capability for large-scale data processing.
    """
    
    def __init__(self, checkpoint_interval: int = 10):
        self.processed_count = 0
        self.results = []
        self.checkpoint_interval = checkpoint_interval
        self.load_checkpoint()
    
    def process_batch(self, data_chunks: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Process a batch of data chunks."""
        batch_results = []
        
        for i, chunk in enumerate(data_chunks):
            result = {
                "batch_id": self.processed_count,
                "chunk_index": i,
                "mean": np.mean(chunk),
                "std": np.std(chunk),
                "median": np.median(chunk),
                "shape": chunk.shape
            }
            
            self.results.append(result)
            batch_results.append(result)
            self.processed_count += 1
            
            # Checkpoint periodically
            if self.processed_count % self.checkpoint_interval == 0:
                self.save_checkpoint()
                
        return batch_results
    
    def save_checkpoint(self):
        """Save processor state to disk."""
        checkpoint_data = {
            "processed_count": self.processed_count,
            "results": self.results
        }
        # In a real application, you would save this to persistent storage
        print(f"Checkpoint saved at processed_count={self.processed_count}")
    
    def load_checkpoint(self):
        """Load processor state from disk."""
        # In a real application, you would load from persistent storage
        print("Processor initialized")
    
    def get_stats(self) -> Dict[str, Any]:
        """Return accumulated statistics."""
        return {
            "total_processed": self.processed_count,
            "results": self.results[:5],  # Return first 5 results to keep output manageable
            "total_results": len(self.results)
        }


def process_massive_dataset_with_checkpoints(data: np.ndarray, chunk_size: int = 50000, 
                                           batch_size: int = 5) -> Dict[str, Any]:
    """
    Process a massive dataset using checkpointed actors.
    
    Args:
        data: Massive numpy array to process
        chunk_size: Size of each chunk
        batch_size: Number of chunks to process in each batch
        
    Returns:
        Final results from processing
    """
    print(f"Processing massive dataset of shape {data.shape}")
    print(f"Chunk size: {chunk_size}, Batch size: {batch_size}")
    
    # Split data into chunks
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    print(f"Split data into {len(chunks)} chunks")
    
    # Group chunks into batches
    batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    print(f"Grouped into {len(batches)} batches")
    
    # Create processor actor
    processor = CheckpointedDataProcessor.remote(checkpoint_interval=3)
    
    # Process batches
    batch_futures = []
    for i, batch in enumerate(batches):
        future = processor.process_batch.remote(batch)
        batch_futures.append(future)
        
        # Limit concurrent batches to avoid overwhelming the system
        if len(batch_futures) >= 3:
            # Wait for some batches to complete before submitting more
            ready_futures, remaining_futures = ray.wait(batch_futures, num_returns=1)
            batch_futures = remaining_futures
    
    # Wait for all batches to complete
    ray.get(batch_futures)
    
    # Get final stats
    stats = ray.get(processor.get_stats.remote())
    return stats


def process_with_simple_aggregation(chunks: List[np.ndarray]) -> Dict[str, Any]:
    """
    Process chunks and aggregate results simply.
    
    Args:
        chunks: List of data chunks to process
        
    Returns:
        Final aggregated result
    """
    # Submit all processing tasks
    futures = [robust_process_chunk.remote(chunk, i) for i, chunk in enumerate(chunks)]
    
    # Collect results
    try:
        results = ray.get(futures, timeout=30)  # 30 second timeout
        
        # Simple aggregation
        total_sum = sum(r.get("sum", 0) for r in results)
        total_count = sum(r.get("count", 0) for r in results)
        overall_mean = total_sum / total_count if total_count > 0 else 0
        overall_max = max(r.get("max", float('-inf')) for r in results)
        overall_min = min(r.get("min", float('inf')) for r in results)
        
        return {
            "overall_mean": overall_mean,
            "overall_max": overall_max,
            "overall_min": overall_min,
            "total_count": total_count,
            "chunk_count": len(results)
        }
    except Exception as e:
        print(f"Error in aggregation: {e}")
        return {"error": str(e)}


def create_massive_data(size: int = 10000000) -> np.ndarray:
    """Create massive sample data for processing."""
    print(f"Creating massive sample data of size {size:,}")
    # Create a large array with some structure
    data = np.random.randn(size)
    # Add some trend
    data += np.linspace(0, 100, size)
    return data


def main():
    """Main function demonstrating advanced Ray usage patterns."""
    print("=" * 70)
    print("Advanced Ray Large Data Processing Examples")
    print("=" * 70)
    
    # Create smaller dataset for demo to avoid memory issues
    massive_data = create_massive_data(1000000)  # 1 million elements instead of 10 million
    
    # Example 1: Robust processing with retries
    print("\n1. Processing with fault tolerance:")
    chunks = [massive_data[i:i + 100000] for i in range(0, len(massive_data), 100000)]
    print(f"Created {len(chunks)} chunks")
    
    # Submit tasks with retry logic
    futures = [robust_process_chunk.remote(chunk, i) for i, chunk in enumerate(chunks)]
    
    # Handle potential failures gracefully
    try:
        results = ray.get(futures, timeout=30)  # 30 second timeout
        print(f"Successfully processed {len(results)} chunks")
        
        # Calculate overall statistics
        total_sum = sum(r.get("sum", 0) for r in results)
        total_count = sum(r.get("count", 0) for r in results)
        overall_mean = total_sum / total_count if total_count > 0 else 0
        
        print(f"Overall mean: {overall_mean:.4f}")
        print("First few chunk results:")
        for i, result in enumerate(results[:3]):
            print(f"  Chunk {i}: mean={result.get('mean', 0):.4f}, "
                  f"count={result.get('count', 0)}")
                  
    except ray.exceptions.GetTimeoutError:
        print("Processing timed out - some tasks may still be running")
    except Exception as e:
        print(f"Error occurred: {e}")
        # In a real application, you might want to handle specific failures differently
    
    # Example 2: Processing with checkpointed actor
    print("\n2. Processing with checkpointed actor:")
    # Use a smaller subset for this demo
    subset_data = massive_data[:500000]  # 500k elements
    actor_results = process_massive_dataset_with_checkpoints(subset_data, 
                                                           chunk_size=50000, 
                                                           batch_size=3)
    print(f"Processed {actor_results['total_processed']} batches")
    print(f"Total results collected: {actor_results['total_results']}")
    print("Sample results:")
    for result in actor_results['results']:
        print(f"  Batch {result['batch_id']}: mean={result['mean']:.4f}")
    
    # Example 3: Simple aggregation for comparison
    print("\n3. Processing with result aggregation:")
    small_chunks = [massive_data[i:i + 50000] for i in range(0, 100000, 50000)]  # Just 2 chunks for demo
    aggregation_result = process_with_simple_aggregation(small_chunks)
    print("Aggregation result:", aggregation_result)
    
    # Shutdown Ray
    ray.shutdown()
    print("\n" + "=" * 70)
    print("Advanced demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()