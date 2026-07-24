"""Policy/value update loop with CUDA and Apple MPS selection."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from ai.network import KARDSNet
from ai.replay_buffer import ReplayBuffer

def best_device() -> str:
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

class Trainer:
    def __init__(self, model: KARDSNet, learning_rate: float = 3e-4, device: str | None = None) -> None:
        self.device = device or best_device(); self.model = model.to(self.device); self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
    def train_batch(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, float]:
        batch = buffer.sample(batch_size)
        if not batch: raise ValueError("Replay buffer is empty")
        state = torch.tensor([x.state for x in batch], dtype=torch.float32, device=self.device)
        actions = torch.tensor([x.action_features for x in batch], dtype=torch.float32, device=self.device)
        mask = torch.tensor([x.legal_mask for x in batch], dtype=torch.bool, device=self.device)
        target_policy = torch.tensor([x.policy for x in batch], dtype=torch.float32, device=self.device)
        target_value = torch.tensor([x.value for x in batch], dtype=torch.float32, device=self.device)
        logits, value = self.model(state, actions, mask)
        policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        value_loss = F.mse_loss(value, target_value); loss = policy_loss + value_loss
        self.optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0); self.optimizer.step()
        return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item())}
