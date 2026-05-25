"""
Simple example of using Ray for large data processing.
This script demonstrates basic Ray functionality without complex data serialization.
"""

import ray
import numpy as np
import time


# Initialize Ray
ray.init(ignore_reinit_error=True)


@ray.remote
def compute_statistics(data_chunk):
    """
    Compute basic statistics on a data chunk.
    
    Args:
        data_chunk: A numpy array
        
    Returns:
        Dictionary with statistics
    """
    return {
        "mean": np.mean(data_chunk),
        "std": np.std(data_chunk),
        "min": np.min(data_chunk),
        "max": np.max(data_chunk),
        "sum": np.sum(data_chunk),
        "size": len(data_chunk)
    }


@ray.remote
def monte_carlo_pi(n_samples):
    """
    Estimate Pi using Monte Carlo method.
    
    Args:
        n_samples: Number of samples
        
    Returns:
        Pi estimate
    """
    inside_circle = 0
    for _ in range(n_samples):
        x, y = np.random.random(), np.random.random()
        if x**2 + y**2 <= 1:
            inside_circle += 1
    return 4 * inside_circle / n_samples


@ray.remote
class ResultAggregator:
    """
    Actor to aggregate results from multiple tasks.
    """
    
    def __init__(self):
        self.results = []
        self.count = 0
    
    def add_result(self, result):
        """Add a result to the aggregator."""
        self.results.append(result)
        self.count += 1
        return self.count
    
    def get_final_result(self):
        """Calculate final aggregated result."""
        if not self.results:
            return None
            
        # Aggregate statistics
        total_sum = sum(r["sum"] for r in self.results if "sum" in r)
        total_size = sum(r["size"] for r in self.results if "size" in r)
        overall_mean = total_sum / total_size if total_size > 0 else 0
        
        # Find global min/max
        global_min = min(r["min"] for r in self.results if "min" in r)
        global_max = max(r["max"] for r in self.results if "max" in r)
        
        return {
            "overall_mean": overall_mean,
            "global_min": global_min,
            "global_max": global_max,
            "total_elements": total_size,
            "chunks_processed": len(self.results)
        }


def process_large_array_distributed(data, n_chunks=10):
    """
    Process a large array by distributing work across Ray tasks.
    
    Args:
        data: Large numpy array
        n_chunks: Number of chunks to split data into
        
    Returns:
        Aggregated results
    """
    print(f"Processing array of size {len(data):,} with {n_chunks} chunks")
    
    # Split data into chunks
    chunks = np.array_split(data, n_chunks)
    
    # Submit computation tasks
    futures = [compute_statistics.remote(chunk) for chunk in chunks]
    
    # Create aggregator actor
    aggregator = ResultAggregator.remote()
    
    # Add results to aggregator as they complete
    aggregation_futures = []
    for future in futures:
        # Get the result and add it to aggregator
        result = ray.get(future)
        aggregation_future = aggregator.add_result.remote(result)
        aggregation_futures.append(aggregation_future)
    
    # Wait for all aggregations to complete
    ray.get(aggregation_futures)
    
    # Get final result
    final_result = ray.get(aggregator.get_final_result.remote())
    return final_result


def parallel_monte_carlo_pi(n_samples_per_task=100000, n_tasks=10):
    """
    Perform Monte Carlo estimation of Pi in parallel.
    
    Args:
        n_samples_per_task: Samples per task
        n_tasks: Number of parallel tasks
        
    Returns:
        Average Pi estimate
    """
    print(f"Estimating Pi with {n_tasks} tasks of {n_samples_per_task:,} samples each")
    
    # Submit Monte Carlo tasks
    futures = [monte_carlo_pi.remote(n_samples_per_task) for _ in range(n_tasks)]
    
    # Collect results
    estimates = ray.get(futures)
    
    # Calculate average
    avg_estimate = np.mean(estimates)
    std_estimate = np.std(estimates)
    
    return avg_estimate, std_estimate, estimates


def create_large_dataset(size=10000000):
    """
    Create a large dataset for processing.
    
    Args:
        size: Size of the dataset
        
    Returns:
        Numpy array
    """
    print(f"Creating dataset with {size:,} elements")
    # Create array with some pattern
    data = np.random.randn(size)
    # Add trend
    data += np.linspace(0, 100, size)
    return data


def main():
    """Main function demonstrating Ray functionality."""
    print("=" * 60)
    print("Simple Ray Large Data Processing Examples")
    print("=" * 60)
    
    # Example 1: Distributed statistical computation
    print("\n1. Distributed statistical computation:")
    data = create_large_dataset(1000000)  # 1 million elements
    
    start_time = time.time()
    result = process_large_array_distributed(data, n_chunks=20)
    dist_time = time.time() - start_time
    
    print(f"Processing completed in {dist_time:.2f} seconds")
    print("Results:")
    print(f"  Overall mean: {result['overall_mean']:.4f}")
    print(f"  Value range: [{result['global_min']:.4f}, {result['global_max']:.4f}]")
    print(f"  Total elements: {result['total_elements']:,}")
    print(f"  Chunks processed: {result['chunks_processed']}")
    
    # Example 2: Parallel Monte Carlo simulation
    print("\n2. Parallel Monte Carlo Pi estimation:")
    start_time = time.time()
    pi_avg, pi_std, all_estimates = parallel_monte_carlo_pi(
        n_samples_per_task=500000, n_tasks=8
    )
    mc_time = time.time() - start_time
    
    print(f"Monte Carlo completed in {mc_time:.2f} seconds")
    print(f"Pi estimate: {pi_avg:.6f} ± {pi_std:.6f}")
    print(f"True Pi: {np.pi:.6f}")
    print(f"Error: {abs(pi_avg - np.pi):.6f}")
    print("Individual estimates:")
    for i, est in enumerate(all_estimates[:5]):  # Show first 5
        print(f"  Task {i+1}: {est:.6f}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()