"""One-forward batched policy/value inference for concurrent MCTS workers."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic, perf_counter

import torch

from ai.network import KARDSNet


@dataclass
class _Request:
    state: torch.Tensor
    features: torch.Tensor
    mask: torch.Tensor
    future: Future


class _InferenceMetrics:
    def __init__(self) -> None:
        self.lock = Lock(); self.batches = self.requests = 0; self.batch_items = 0; self.wall_seconds = self.gpu_seconds = 0.0

    def add(self, size: int, wall: float, gpu: float = 0.0) -> None:
        with self.lock:
            self.batches += 1; self.requests += size; self.batch_items += size; self.wall_seconds += wall; self.gpu_seconds += gpu

    def snapshot(self) -> dict[str, float]:
        with self.lock:
            return {"gpu_forwards": float(self.batches), "inference_requests": float(self.requests),
                    "average_batch_size": self.batch_items / self.batches if self.batches else 0.0,
                    "inference_wall_seconds": self.wall_seconds, "gpu_seconds": self.gpu_seconds,
                    "inferences_per_second": self.requests / self.wall_seconds if self.wall_seconds else 0.0}


class BatchedInference:
    """Batch complete policy/value requests from concurrent local MCTS searches."""

    def __init__(self, model: KARDSNet, *, max_batch_size: int = 128, max_wait_ms: float = 5.0,
                 measure_gpu_time: bool = False) -> None:
        self.model, self.device = model.eval(), next(model.parameters()).device
        self.max_batch_size, self.max_wait_seconds = max_batch_size, max_wait_ms / 1000.0
        self.measure_gpu_time, self.metrics = measure_gpu_time, _InferenceMetrics()
        self._queue: Queue[_Request | None] = Queue(); self._closed = False
        self._thread = Thread(target=self._serve, name="kards-cuda-inference", daemon=True); self._thread.start()

    def __enter__(self) -> "BatchedInference": return self
    def __exit__(self, *_: object) -> None: self.close()
    def close(self) -> None:
        if not self._closed:
            self._closed = True; self._queue.put(None); self._thread.join()

    def evaluate(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._closed: raise RuntimeError("BatchedInference is closed")
        future: Future = Future()
        self._queue.put(_Request(state.detach().cpu(), features.detach().cpu(), mask.detach().cpu(), future))
        return future.result()

    def stats(self) -> dict[str, float]: return self.metrics.snapshot()

    def _serve(self) -> None:
        while True:
            first = self._queue.get()
            if first is None: return
            batch = [first]; deadline = monotonic() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                try: item = self._queue.get(timeout=max(0.0, deadline - monotonic()))
                except Empty: break
                if item is None: self._closed = True; break
                batch.append(item)
            self._run_batch(batch)
            if self._closed: return

    @torch.inference_mode()
    def _run_batch(self, requests: list[_Request]) -> None:
        try:
            states, features, mask, counts = _pack(requests, self.device)
            started = perf_counter(); gpu_started = None
            if self.measure_gpu_time and self.device.type == "cuda": gpu_started = torch.cuda.Event(True); gpu_ended = torch.cuda.Event(True); gpu_started.record()
            logits, values = self.model(states, features, mask)
            gpu_seconds = 0.0
            if self.measure_gpu_time and self.device.type == "cuda":
                gpu_ended.record(); gpu_ended.synchronize(); gpu_seconds = gpu_started.elapsed_time(gpu_ended) / 1000.0
            logits, values = logits.cpu(), values.cpu(); wall = perf_counter() - started
            self.metrics.add(len(requests), wall, gpu_seconds)
            for request, result, value, count in zip(requests, logits, values, counts): request.future.set_result((result[:count], value))
        except BaseException as error:
            for request in requests: request.future.set_exception(error)


class RemoteInferenceClient:
    """Synchronous request/reply client; each request returns policy and value together."""

    def __init__(self, request_queue, response_queue, worker_id: int = 0) -> None:
        self.request_queue, self.response_queue, self.worker_id, self._next_request = request_queue, response_queue, worker_id, 0

    def evaluate(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        request_id = self._next_request; self._next_request += 1
        self.request_queue.put((self.worker_id, request_id, state.tolist(), features.tolist(), mask.tolist()))
        response_id, payload = self.response_queue.get()
        if response_id != request_id: raise RuntimeError("Received an inference response for another request")
        if isinstance(payload, str): raise RuntimeError(payload)
        logits, value = payload
        return torch.tensor(logits, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)


class ProcessInferenceService:
    """GPU batch scheduler: wait briefly for workers, then run one combined forward."""

    def __init__(self, model: KARDSNet, request_queue, response_queues=None, *, max_batch_size: int = 128,
                 max_wait_ms: float = 5.0, measure_gpu_time: bool = False) -> None:
        self.model, self.device, self.request_queue = model.eval(), next(model.parameters()).device, request_queue
        self.response_queues, self.max_batch_size = response_queues or [], max_batch_size
        self.max_wait_seconds, self.measure_gpu_time, self.metrics = max_wait_ms / 1000.0, measure_gpu_time, _InferenceMetrics()
        self._closed = False; self._thread = Thread(target=self._serve, name="kards-process-cuda-inference", daemon=True); self._thread.start()

    def close(self) -> None:
        if not self._closed: self._closed = True; self.request_queue.put(None); self._thread.join()
    def stats(self) -> dict[str, float]: return self.metrics.snapshot()

    def _serve(self) -> None:
        while True:
            first = self.request_queue.get()
            if first is None: return
            batch = [first]; deadline = monotonic() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                try: item = self.request_queue.get(timeout=max(0.0, deadline - monotonic()))
                except Empty: break
                if item is None: self._closed = True; break
                batch.append(item)
            self._run_process_batch(batch)
            if self._closed: return

    @torch.inference_mode()
    def _run_process_batch(self, requests) -> None:
        try:
            states, features, mask, counts = _pack_process(requests, self.device)
            started = perf_counter(); gpu_seconds = 0.0
            if self.measure_gpu_time and self.device.type == "cuda": gpu_started = torch.cuda.Event(True); gpu_ended = torch.cuda.Event(True); gpu_started.record()
            logits, values = self.model(states, features, mask)
            if self.measure_gpu_time and self.device.type == "cuda": gpu_ended.record(); gpu_ended.synchronize(); gpu_seconds = gpu_started.elapsed_time(gpu_ended) / 1000.0
            logits, values = logits.cpu(), values.cpu(); self.metrics.add(len(requests), perf_counter() - started, gpu_seconds)
            for request, result, value, count in zip(requests, logits, values, counts): self.response_queues[request[0]].put((request[1], (result[:count].tolist(), float(value.item()))))
        except BaseException as error:
            for request in requests: self.response_queues[request[0]].put((request[1], f"Remote inference failed: {error}"))


def _pack(requests: list[_Request], device: torch.device):
    width = max(request.features.shape[0] for request in requests); dim = requests[0].features.shape[-1]
    states = torch.stack([request.state for request in requests]).to(device); features = torch.zeros((len(requests), width, dim), dtype=torch.float32, device=device); mask = torch.zeros((len(requests), width), dtype=torch.bool, device=device)
    counts = []
    for index, request in enumerate(requests):
        count = request.features.shape[0]; counts.append(count); features[index, :count] = request.features.to(device); mask[index, :count] = request.mask.to(device)
    return states, features, mask, counts


def _pack_process(requests, device: torch.device):
    width, dim = max(len(request[3]) for request in requests), len(requests[0][3][0])
    states = torch.tensor([request[2] for request in requests], dtype=torch.float32, device=device); features = torch.zeros((len(requests), width, dim), dtype=torch.float32, device=device); mask = torch.zeros((len(requests), width), dtype=torch.bool, device=device)
    counts = []
    for index, request in enumerate(requests):
        count = len(request[3]); counts.append(count); features[index, :count] = torch.tensor(request[3], dtype=torch.float32, device=device); mask[index, :count] = torch.tensor(request[4], dtype=torch.bool, device=device)
    return states, features, mask, counts
