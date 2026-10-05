"""
batching.py — Dynamic Batching for Inference
===============================================

Student implements:
  - Batcher: collects individual requests into batches
  - submit(): enqueue a request and return a future
  - worker loop: flush on max_batch_size or max_wait_ms
  - simulate_load(): Poisson arrival rate simulation

Systems engineering for inference throughput.
"""

import torch
import torch.nn as nn
import threading
import time
import queue
from concurrent.futures import Future
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────
# Dynamic Batcher
# ─────────────────────────────────────────────────────

class Batcher:
    """
    Dynamic batching: collect individual requests into batches
    to maximize GPU utilization.

    Policy: flush when EITHER:
      - max_batch_size requests collected, OR
      - max_wait_ms has elapsed since the first request in the batch

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement __init__, submit, _worker, shutdown.  ║
    ║                                                       ║
    ║  __init__(predict_fn, max_batch_size, max_wait_ms):    ║
    ║    1. self.predict_fn = predict_fn                      ║
    ║    2. self.max_batch_size = max_batch_size              ║
    ║    3. self.max_wait_ms = max_wait_ms                   ║
    ║    4. self._queue = queue.Queue()                       ║
    ║    5. self._running = True                              ║
    ║    6. self._thread = threading.Thread(                  ║
    ║         target=self._worker, daemon=True)               ║
    ║    7. self._thread.start()                              ║
    ║    8. Track stats: batch_sizes, wait_times              ║
    ║                                                       ║
    ║  submit(input_tensor) -> Future:                       ║
    ║    1. future = Future()                                ║
    ║    2. self._queue.put((input_tensor, future))           ║
    ║    3. return future                                     ║
    ║                                                       ║
    ║  _worker():                                            ║
    ║    while self._running:                                ║
    ║      1. Collect requests from queue                    ║
    ║      2. Start timer on first request                   ║
    ║      3. Keep collecting until:                          ║
    ║         - batch is full (max_batch_size), OR            ║
    ║         - max_wait_ms exceeded                         ║
    ║      4. Stack inputs: batch = torch.cat(inputs)         ║
    ║      5. results = self.predict_fn(batch)                ║
    ║      6. Split results and set each future's result      ║
    ║      7. Record batch size and wait time                 ║
    ║                                                       ║
    ║  shutdown():                                           ║
    ║    self._running = False                                ║
    ║    self._thread.join()                                  ║
    ╚═══════════════════════════════════════════════════════╝
    """

    def __init__(
        self,
        predict_fn: Callable[[torch.Tensor], torch.Tensor],
        max_batch_size: int = 32,
        max_wait_ms: float = 50.0,
    ):
        raise NotImplementedError("TODO: implement Batcher.__init__()")

    def submit(self, input_tensor: torch.Tensor) -> Future:
        raise NotImplementedError("TODO: implement Batcher.submit()")

    def _worker(self):
        raise NotImplementedError("TODO: implement Batcher._worker()")

    def shutdown(self):
        raise NotImplementedError("TODO: implement Batcher.shutdown()")

    def get_stats(self) -> Dict:
        """Return batching statistics. Provided."""
        if not hasattr(self, '_batch_sizes'):
            return {}
        return {
            'num_batches': len(self._batch_sizes),
            'avg_batch_size': np.mean(self._batch_sizes) if self._batch_sizes else 0,
            'max_batch_size_seen': max(self._batch_sizes) if self._batch_sizes else 0,
            'avg_wait_ms': np.mean(self._wait_times) * 1000 if self._wait_times else 0,
        }


# ─────────────────────────────────────────────────────
# Load Simulation
# ─────────────────────────────────────────────────────

def simulate_load(
    batcher: Batcher,
    input_shape: Tuple[int, ...],
    num_requests: int = 200,
    arrival_rate: float = 100.0,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, list]:
    """
    Simulate Poisson request arrivals and measure latency.

    ╔═══════════════════════════════════════════════════════╗
    ║  TODO: Implement this function.                       ║
    ║                                                       ║
    ║  Steps:                                               ║
    ║  1. Generate inter-arrival times:                      ║
    ║     np.random.exponential(1/arrival_rate, num_requests) ║
    ║  2. For each request:                                  ║
    ║     a. Wait for inter-arrival time                     ║
    ║     b. Create random input tensor                      ║
    ║     c. t0 = time.perf_counter()                        ║
    ║     d. future = batcher.submit(input_tensor)           ║
    ║     e. result = future.result()  # blocks until ready  ║
    ║     f. latency = time.perf_counter() - t0              ║
    ║     g. Record latency                                  ║
    ║  3. Return {'latencies_ms': [...], 'arrival_times': [..]}║
    ╚═══════════════════════════════════════════════════════╝
    """
    raise NotImplementedError("TODO: implement simulate_load()")


# ─────────────────────────────────────────────────────
# Visualization (provided)
# ─────────────────────────────────────────────────────

def plot_batching_results(results: Dict[str, Dict]):
    """
    Plot latency and throughput comparison for different batching policies.
    Provided utility.
    """
    configs = list(results.keys())
    p50s = [results[c]['p50_ms'] for c in configs]
    p95s = [results[c]['p95_ms'] for c in configs]
    throughputs = [results[c]['throughput'] for c in configs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = range(len(configs))
    width = 0.35
    ax1.bar([i - width/2 for i in x], p50s, width, label='p50', color='steelblue')
    ax1.bar([i + width/2 for i in x], p95s, width, label='p95', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=15)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency by Policy")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    ax2.bar(x, throughputs, color='#4CAF50')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=15)
    ax2.set_ylabel("Throughput (req/sec)")
    ax2.set_title("Throughput by Policy")
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()
