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
        self.cuda = str(self.device).startswith("cuda")
        if self.cuda:
            torch.set_float32_matmul_precision("high")
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda)
    def train_batch(self, buffer: ReplayBuffer, batch_size: int) -> dict[str, float]:
        batch = buffer.sample(batch_size)
        if not batch: raise ValueError("Replay buffer is empty")
        width = max(len(x.legal_mask) for x in batch)
        state = torch.tensor([x.state for x in batch], dtype=torch.float32, device=self.device)
        actions = torch.zeros((len(batch), width, len(batch[0].action_features[0])), dtype=torch.float32, device=self.device)
        mask = torch.zeros((len(batch), width), dtype=torch.bool, device=self.device)
        target_policy = torch.zeros((len(batch), width), dtype=torch.float32, device=self.device)
        for index, example in enumerate(batch):
            count = len(example.legal_mask)
            actions[index, :count] = torch.tensor(example.action_features, dtype=torch.float32, device=self.device)
            mask[index, :count] = torch.tensor(example.legal_mask, dtype=torch.bool, device=self.device)
            target_policy[index, :count] = torch.tensor(example.policy, dtype=torch.float32, device=self.device)
        target_value = torch.tensor([x.value for x in batch], dtype=torch.float32, device=self.device)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.cuda):
            logits, value = self.model(state, actions, mask)
            policy_loss = -(target_policy * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, target_value); loss = policy_loss + value_loss
        self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.scaler.step(self.optimizer); self.scaler.update()
        return {"loss": float(loss.item()), "policy_loss": float(policy_loss.item()), "value_loss": float(value_loss.item())}
