import random
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


configurations = list(range(1, 100))
random.shuffle(configurations)

NUM_WORKERS = os.cpu_count() or 4
BUCKET_SIZE = max(1, len(configurations) // NUM_WORKERS)

buckets = [
    configurations[i : i + BUCKET_SIZE]
    for i in range(0, len(configurations), BUCKET_SIZE)
]


def process_bucket(bucket: list[int]) -> list[int]:
    """Process one bucket of configurations (runs in a separate process)."""
    results = []
    for config_id in bucket:
        result = config_id * config_id
        results.append(result)
    return results


if __name__ == "__main__":
    print(f"CPU count: {NUM_WORKERS}")
    print(f"Configurations: {len(configurations)}")
    print(f"Buckets: {len(buckets)} (size ~{BUCKET_SIZE})")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_bucket, b): i for i, b in enumerate(buckets)}

        for future in as_completed(futures):
            bucket_idx = futures[future]
            results = future.result()
            print(f"Bucket {bucket_idx} done: {len(results)} results")


