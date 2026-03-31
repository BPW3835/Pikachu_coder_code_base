import ray
import time
import numpy as np

ray.init()

# ── Remote task: process one chunk ──────────────────────────────────────────
@ray.remote
def process_chunk(chunk: list[int]) -> dict:
    """Simulate CPU-heavy work on a data chunk."""
    time.sleep(0.5)
    return {
        "sum":  sum(chunk),
        "mean": np.mean(chunk),
        "size": len(chunk),
    }

# ── Actor: aggregate results as they arrive ──────────────────────────────────
@ray.remote
class ResultAggregator:
    def __init__(self):
        self.total_sum   = 0
        self.total_items = 0
        self.chunk_count = 0

    def add(self, result: dict):
        self.total_sum   += result["sum"]
        self.total_items += result["size"]
        self.chunk_count += 1

    def summary(self) -> dict:
        return {
            "chunks_processed": self.chunk_count,
            "total_items":      self.total_items,
            "global_mean":      self.total_sum / self.total_items,
        }

# ── Main driver ──────────────────────────────────────────────────────────────
data   = list(range(1000))
chunks = [data[i:i+100] for i in range(0, len(data), 100)]   # 10 chunks of 100

aggregator = ResultAggregator.remote()

# Dispatch all chunks in parallel
futures = [process_chunk.remote(chunk) for chunk in chunks]

# Stream results into the aggregator as each chunk finishes
for future in ray.get(futures):
    aggregator.add.remote(future)

summary = ray.get(aggregator.summary.remote())
print(summary)
# {'chunks_processed': 10, 'total_items': 1000, 'global_mean': 499.5}

ray.shutdown()