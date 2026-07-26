"""One-forward batched policy/value inference for concurrent MCTS workers."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import shared_memory
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic, perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import torch

from ai.action_encoder import ACTION_FEATURE_DIM, MAX_ACTIONS
from ai.network import KARDSNet
from ai.observation import STATE_DIM


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


@dataclass(frozen=True)
class SharedInferenceSpec:
    """Names and fixed dimensions for cross-process inference ring buffers."""
    workers: int
    slots_per_worker: int
    state_name: str
    features_name: str
    mask_name: str
    count_name: str
    logits_name: str
    value_name: str


class SharedInferenceBuffers:
    """Fixed-size CPU buffers that avoid serializing inference tensors.

    Only the tiny ``(worker, slot, request_id)`` descriptor goes through a
    multiprocessing queue.  A slot remains owned by its worker until the
    reply arrives, so the CUDA service can safely read/write it without locks.
    """

    def __init__(self, spec: SharedInferenceSpec, *, create: bool = False) -> None:
        self.spec = spec
        shape = (spec.workers, spec.slots_per_worker)
        names = (spec.state_name, spec.features_name, spec.mask_name, spec.count_name, spec.logits_name, spec.value_name)
        sizes = (np.prod(shape) * STATE_DIM * 4, np.prod(shape) * MAX_ACTIONS * ACTION_FEATURE_DIM * 4,
                 np.prod(shape) * MAX_ACTIONS, np.prod(shape) * 4, np.prod(shape) * MAX_ACTIONS * 4, np.prod(shape) * 4)
        self._segments = [shared_memory.SharedMemory(name=name, create=create, size=int(size) if create else 0)
                          for name, size in zip(names, sizes)]
        self.states = np.ndarray((*shape, STATE_DIM), dtype=np.float32, buffer=self._segments[0].buf)
        self.features = np.ndarray((*shape, MAX_ACTIONS, ACTION_FEATURE_DIM), dtype=np.float32, buffer=self._segments[1].buf)
        self.masks = np.ndarray((*shape, MAX_ACTIONS), dtype=np.bool_, buffer=self._segments[2].buf)
        self.counts = np.ndarray(shape, dtype=np.int32, buffer=self._segments[3].buf)
        self.logits = np.ndarray((*shape, MAX_ACTIONS), dtype=np.float32, buffer=self._segments[4].buf)
        self.values = np.ndarray(shape, dtype=np.float32, buffer=self._segments[5].buf)
        if create:
            self.counts.fill(0)

    @classmethod
    def create(cls, workers: int, slots_per_worker: int = 32) -> "SharedInferenceBuffers":
        names = [f"kards_inf_{uuid4().hex}" for _ in range(6)]
        spec = SharedInferenceSpec(workers, slots_per_worker, *names)
        return cls(spec, create=True)

    def close(self) -> None:
        for segment in self._segments:
            segment.close()

    def unlink(self) -> None:
        for segment in self._segments:
            try: segment.unlink()
            except FileNotFoundError: pass


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
        return self.submit(state, features, mask).result()

    def submit(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> Future:
        if self._closed: raise RuntimeError("BatchedInference is closed")
        future: Future = Future()
        self._queue.put(_Request(state.detach().cpu(), features.detach().cpu(), mask.detach().cpu(), future))
        return future

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

    def __init__(self, request_queue, response_queue, worker_id: int = 0,
                 shared_spec: SharedInferenceSpec | None = None) -> None:
        self.request_queue, self.response_queue, self.worker_id, self._next_request = request_queue, response_queue, worker_id, 0
        self._pending: dict[int, Future] = {}
        self._submitted_at: dict[int, float] = {}
        self._free_slots = list(range(shared_spec.slots_per_worker - 1, -1, -1)) if shared_spec else []
        self._shared = SharedInferenceBuffers(shared_spec) if shared_spec else None
        self.metrics = {"request_transport_seconds": 0.0, "serialization_seconds": 0.0,
                        "deserialization_seconds": 0.0, "inference_wait_seconds": 0.0, "requests": 0.0}

    def close(self) -> None:
        if self._shared is not None:
            self._shared.close(); self._shared = None

    def stats(self) -> dict[str, float]:
        requests = self.metrics["requests"]
        return {**self.metrics, "average_request_transport_ms": 1000.0 * self.metrics["request_transport_seconds"] / requests if requests else 0.0,
                "average_serialization_ms": 1000.0 * self.metrics["serialization_seconds"] / requests if requests else 0.0,
                "average_deserialization_ms": 1000.0 * self.metrics["deserialization_seconds"] / requests if requests else 0.0,
                "average_inference_wait_ms": 1000.0 * self.metrics["inference_wait_seconds"] / requests if requests else 0.0}

    def evaluate(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        future = self.submit(state, features, mask)
        while not future.done():
            self.poll(block=True)
        return future.result()

    def submit(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> Future:
        request_id = self._next_request; self._next_request += 1
        future: Future = Future(); self._pending[request_id] = future
        self._submitted_at[request_id] = perf_counter(); self.metrics["requests"] += 1
        if self._shared is not None:
            # Async MCTS has a bounded number of pending leaves.  Slots are
            # intentionally reused only after ProcessInferenceService replies.
            if not self._free_slots:
                raise RuntimeError("Shared inference ring exhausted; increase slots_per_worker")
            slot = self._free_slots.pop(); started = perf_counter()
            count = int(features.shape[0])
            if count > MAX_ACTIONS:
                raise ValueError("Legal action count exceeds shared inference capacity")
            np.copyto(self._shared.states[self.worker_id, slot], state.detach().cpu().numpy())
            np.copyto(self._shared.features[self.worker_id, slot, :count], features.detach().cpu().numpy())
            np.copyto(self._shared.masks[self.worker_id, slot, :count], mask.detach().cpu().numpy())
            self._shared.counts[self.worker_id, slot] = count
            self.metrics["serialization_seconds"] += perf_counter() - started
            queued = perf_counter(); self.request_queue.put(("shared", self.worker_id, slot, request_id))
            self.metrics["request_transport_seconds"] += perf_counter() - queued
        else:
            started = perf_counter(); payload = (self.worker_id, request_id, state.tolist(), features.tolist(), mask.tolist())
            self.metrics["serialization_seconds"] += perf_counter() - started
            queued = perf_counter(); self.request_queue.put(payload); self.metrics["request_transport_seconds"] += perf_counter() - queued
        return future

    def poll(self, *, block: bool = False) -> int:
        """Resolve completed responses without blocking an async tree search."""
        completed = 0
        while True:
            try:
                response = self.response_queue.get() if block and not completed else self.response_queue.get_nowait()
            except Empty:
                return completed
            self._resolve_response(response)
            completed += 1

    def wait(self, timeout: float | None = None) -> int:
        """Block on the response queue once, then drain ready replies."""
        try:
            response = self.response_queue.get(timeout=timeout)
        except Empty:
            return 0
        self._resolve_response(response)
        return 1 + self.poll()

    def _resolve_response(self, response) -> None:
        if len(response) == 2:
            response_id, payload = response; slot = None
        else:
            response_id, slot, payload = response
        future = self._pending.pop(response_id, None)
        if future is None:
            raise RuntimeError("Received an inference response for an unknown request")
        self.metrics["inference_wait_seconds"] += perf_counter() - self._submitted_at.pop(response_id)
        if isinstance(payload, str):
            if slot is not None and self._shared is not None: self._free_slots.append(slot)
            future.set_exception(RuntimeError(payload))
        elif slot is not None and self._shared is not None:
            started = perf_counter(); count = int(self._shared.counts[self.worker_id, slot])
            logits = torch.from_numpy(self._shared.logits[self.worker_id, slot, :count].copy())
            value = torch.tensor(float(self._shared.values[self.worker_id, slot]), dtype=torch.float32)
            self._free_slots.append(slot); self.metrics["deserialization_seconds"] += perf_counter() - started
            future.set_result((logits, value))
        else:
            started = perf_counter(); logits, value = payload; future.set_result((torch.tensor(logits, dtype=torch.float32), torch.tensor(value, dtype=torch.float32)))
            self.metrics["deserialization_seconds"] += perf_counter() - started


class AsyncInferenceQueue:
    """Non-blocking adapter used by async MCTS to observe queue pressure."""

    def __init__(self, inference) -> None:
        self.inference, self.pending, self.pending_peak = inference, 0, 0

    def submit(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> Future:
        future = self.inference.submit(state, features, mask); self.pending += 1; self.pending_peak = max(self.pending_peak, self.pending)
        return future

    def poll(self) -> None:
        if hasattr(self.inference, "poll"): self.inference.poll()

    def wait(self, futures: list[Future], timeout: float) -> None:
        if hasattr(self.inference, "wait"):
            self.inference.wait(timeout)
        else:
            from concurrent.futures import wait
            wait(futures, timeout=timeout)

    def resolved(self, future: Future) -> tuple[torch.Tensor, torch.Tensor]:
        self.pending -= 1
        return future.result()


class ProcessInferenceService:
    """GPU batch scheduler: wait briefly for workers, then run one combined forward."""

    def __init__(self, model: KARDSNet, request_queue, response_queues=None, *, max_batch_size: int = 128,
                 max_wait_ms: float = 5.0, measure_gpu_time: bool = False,
                 shared_spec: SharedInferenceSpec | None = None) -> None:
        self.model, self.device, self.request_queue = model.eval(), next(model.parameters()).device, request_queue
        self.response_queues, self.max_batch_size = response_queues or [], max_batch_size
        self.max_wait_seconds, self.measure_gpu_time, self.metrics = max_wait_ms / 1000.0, measure_gpu_time, _InferenceMetrics()
        self.shared = SharedInferenceBuffers(shared_spec) if shared_spec else None
        self._closed = False; self._thread = Thread(target=self._serve, name="kards-process-cuda-inference", daemon=True); self._thread.start()

    def close(self) -> None:
        if not self._closed: self._closed = True; self.request_queue.put(None); self._thread.join()
        if self.shared is not None: self.shared.close(); self.shared = None
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
            states, features, mask, counts = _pack_process(requests, self.device, self.shared)
            started = perf_counter(); gpu_seconds = 0.0
            if self.measure_gpu_time and self.device.type == "cuda": gpu_started = torch.cuda.Event(True); gpu_ended = torch.cuda.Event(True); gpu_started.record()
            logits, values = self.model(states, features, mask)
            if self.measure_gpu_time and self.device.type == "cuda": gpu_ended.record(); gpu_ended.synchronize(); gpu_seconds = gpu_started.elapsed_time(gpu_ended) / 1000.0
            logits, values = logits.cpu(), values.cpu(); self.metrics.add(len(requests), perf_counter() - started, gpu_seconds)
            for request, result, value, count in zip(requests, logits, values, counts):
                if request[0] == "shared":
                    _, worker_id, slot, request_id = request
                    np.copyto(self.shared.logits[worker_id, slot, :count], result[:count].numpy())  # type: ignore[union-attr]
                    self.shared.values[worker_id, slot] = float(value.item())  # type: ignore[union-attr]
                    self.response_queues[worker_id].put((request_id, slot, None))
                else:
                    self.response_queues[request[0]].put((request[1], (result[:count].tolist(), float(value.item()))))
        except BaseException as error:
            for request in requests:
                if request[0] == "shared": self.response_queues[request[1]].put((request[3], request[2], f"Remote inference failed: {error}"))
                else: self.response_queues[request[0]].put((request[1], f"Remote inference failed: {error}"))


def _pack(requests: list[_Request], device: torch.device):
    width = max(request.features.shape[0] for request in requests); dim = requests[0].features.shape[-1]
    states = torch.stack([request.state for request in requests]).to(device); features = torch.zeros((len(requests), width, dim), dtype=torch.float32, device=device); mask = torch.zeros((len(requests), width), dtype=torch.bool, device=device)
    counts = []
    for index, request in enumerate(requests):
        count = request.features.shape[0]; counts.append(count); features[index, :count] = request.features.to(device); mask[index, :count] = request.mask.to(device)
    return states, features, mask, counts


def _pack_process(requests, device: torch.device, shared: SharedInferenceBuffers | None = None):
    def count(request) -> int:
        return int(shared.counts[request[1], request[2]]) if request[0] == "shared" else len(request[3])
    width = max(count(request) for request in requests)
    # Pinned staging allows the CUDA copy to overlap where the driver supports it.
    pin = device.type == "cuda"
    states_cpu = torch.empty((len(requests), STATE_DIM), dtype=torch.float32, pin_memory=pin)
    features_cpu = torch.zeros((len(requests), width, ACTION_FEATURE_DIM), dtype=torch.float32, pin_memory=pin)
    mask_cpu = torch.zeros((len(requests), width), dtype=torch.bool, pin_memory=pin)
    counts = []
    for index, request in enumerate(requests):
        size = count(request); counts.append(size)
        if request[0] == "shared":
            _, worker_id, slot, _ = request
            states_cpu[index].copy_(torch.from_numpy(shared.states[worker_id, slot]))  # type: ignore[union-attr]
            features_cpu[index, :size].copy_(torch.from_numpy(shared.features[worker_id, slot, :size]))  # type: ignore[union-attr]
            mask_cpu[index, :size].copy_(torch.from_numpy(shared.masks[worker_id, slot, :size]))  # type: ignore[union-attr]
        else:
            states_cpu[index].copy_(torch.tensor(request[2], dtype=torch.float32))
            features_cpu[index, :size].copy_(torch.tensor(request[3], dtype=torch.float32))
            mask_cpu[index, :size].copy_(torch.tensor(request[4], dtype=torch.bool))
    return (states_cpu.to(device, non_blocking=pin), features_cpu.to(device, non_blocking=pin),
            mask_cpu.to(device, non_blocking=pin), counts)
