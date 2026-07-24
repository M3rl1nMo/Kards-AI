"""Thread-safe batched inference for concurrent MCTS searches on one GPU."""
from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from time import monotonic
from typing import Literal

import torch

from ai.network import KARDSNet


@dataclass
class _Request:
    kind: Literal["policy", "value"]
    state: torch.Tensor
    features: torch.Tensor | None
    mask: torch.Tensor | None
    future: Future


class BatchedInference:
    """Own a single model execution thread and coalesce compatible MCTS calls.

    Callers block until their own result is ready, so a search tree observes
    exactly the same request/response ordering as local inference.  Only the
    independent tensor evaluations from separate searches share a GPU launch.
    """

    def __init__(self, model: KARDSNet, *, max_batch_size: int = 32, max_wait_ms: float = 2.0) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model = model.eval()
        self.device = next(model.parameters()).device
        self.max_batch_size = max_batch_size
        self.max_wait_seconds = max_wait_ms / 1000.0
        self._queue: Queue[_Request | None] = Queue()
        self._thread = Thread(target=self._serve, name="kards-cuda-inference", daemon=True)
        self._closed = False
        self._thread.start()

    def __enter__(self) -> "BatchedInference":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._queue.put(None)
            self._thread.join()

    def policy(self, state: torch.Tensor, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self._submit("policy", state, features, mask)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self._submit("value", state, None, None)

    def _submit(self, kind: Literal["policy", "value"], state: torch.Tensor,
                features: torch.Tensor | None, mask: torch.Tensor | None) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("BatchedInference is closed")
        future: Future = Future()
        self._queue.put(_Request(kind, state.detach().cpu(),
                                 features.detach().cpu() if features is not None else None,
                                 mask.detach().cpu() if mask is not None else None, future))
        return future.result()

    def _serve(self) -> None:
        while True:
            first = self._queue.get()
            if first is None:
                return
            batch = [first]
            deadline = monotonic() + self.max_wait_seconds
            while len(batch) < self.max_batch_size:
                try:
                    request = self._queue.get(timeout=max(0.0, deadline - monotonic()))
                except Empty:
                    break
                if request is None:
                    self._closed = True
                    break
                batch.append(request)
            self._run_batch(batch)
            if self._closed:
                return

    @torch.inference_mode()
    def _run_batch(self, requests: list[_Request]) -> None:
        for kind in ("policy", "value"):
            group = [request for request in requests if request.kind == kind]
            if not group:
                continue
            try:
                states = torch.stack([request.state for request in group]).to(self.device)
                if kind == "value":
                    values = self.model.value(states).cpu()
                    for request, value in zip(group, values):
                        request.future.set_result(value)
                    continue
                width = max(request.features.shape[0] for request in group if request.features is not None)
                features = torch.zeros((len(group), width, group[0].features.shape[-1]), dtype=torch.float32, device=self.device)
                mask = torch.zeros((len(group), width), dtype=torch.bool, device=self.device)
                for index, request in enumerate(group):
                    assert request.features is not None and request.mask is not None
                    count = request.features.shape[0]
                    features[index, :count] = request.features.to(self.device)
                    mask[index, :count] = request.mask.to(self.device)
                logits = self.model.policy(states, features, mask).cpu()
                for request, result in zip(group, logits):
                    assert request.features is not None
                    request.future.set_result(result[:request.features.shape[0]])
            except BaseException as error:
                for request in group:
                    request.future.set_exception(error)
