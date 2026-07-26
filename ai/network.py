"""PyTorch policy/value network scored against legal candidate actions."""
from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from ai.action_encoder import ACTION_FEATURE_DIM
from ai.observation import STATE_DIM


class KARDSNet(nn.Module):
    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_FEATURE_DIM, hidden_dim: int = 128) -> None:
        super().__init__()
        self.architecture = {"state_dim": state_dim, "action_dim": action_dim, "hidden_dim": hidden_dim}
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden_dim), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Tanh())

    @staticmethod
    def _batched_states(states: torch.Tensor) -> torch.Tensor:
        if states.ndim == 1: states = states.unsqueeze(0)
        return states

    @staticmethod
    def _batched_actions(action_features: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if action_features.ndim == 2: action_features = action_features.unsqueeze(0)
        if legal_mask.ndim == 1: legal_mask = legal_mask.unsqueeze(0)
        return action_features, legal_mask

    def _policy_from_state_hidden(self, state_hidden: torch.Tensor, action_features: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
        action_hidden = self.action_encoder(action_features)
        expanded = state_hidden.unsqueeze(1).expand(-1, action_features.shape[1], -1)
        logits = self.policy_head(torch.cat((expanded, action_hidden), dim=-1)).squeeze(-1)
        # ``-1e9`` overflows under CUDA autocast float16.  The finite minimum
        # for the active dtype is sufficient for masked softmax and keeps the
        # real mixed-precision training path valid.
        return logits.masked_fill(~legal_mask.bool(), torch.finfo(logits.dtype).min)

    def forward(self, states: torch.Tensor, action_features: torch.Tensor, legal_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        states = self._batched_states(states)
        action_features, legal_mask = self._batched_actions(action_features, legal_mask)
        state_hidden = self.state_encoder(states)
        return self._policy_from_state_hidden(state_hidden, action_features, legal_mask), self.value_head(state_hidden).squeeze(-1)

    def save_checkpoint(self, path: str | Path, **metadata: object) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "architecture": self.architecture, "metadata": metadata}, path)

    @classmethod
    def load_checkpoint(cls, path: str | Path, device: str | torch.device = "cpu") -> "KARDSNet":
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(**payload.get("architecture", {})); model.load_state_dict(payload["state_dict"]); model.to(device); return model
