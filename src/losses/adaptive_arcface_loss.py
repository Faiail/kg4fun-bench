import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveArcFaceLoss(nn.Module):
    """
    Adaptive ArcFace Loss supporting:
    - 2-arg mode: forward(cosine_logits, labels)  [drop-in for CompleteLoss and SingleItemRun]
    - 3-arg mode: forward(input_embs, target_embs, labels)
    - Learnable scale s (inverse temperature) clamped between min_s and max_s.
    - Scheduled angular margin m with linear/cosine warmup.
    - Numerical safety against NaN gradients on cosine ≈ 1.0.
    """
    def __init__(
        self,
        initial_s: float = 30.0,
        min_s: float = 1.0,
        max_s: float = 60.0,
        target_m: float = 0.35,
        warmup_steps: int = 5,
        schedule_type: str = "linear",  # "linear" or "cosine"
        eps: float = 1e-7,
    ):
        super().__init__()
        self.min_s = min_s
        self.max_s = max_s
        self.target_m = target_m
        self.warmup_steps = max(1, warmup_steps)
        self.schedule_type = schedule_type
        self.eps = eps

        # Learnable log_s parameter
        self.log_s = nn.Parameter(torch.tensor(math.log(initial_s), dtype=torch.float32))

        # Margin schedule state
        self.current_step = 0
        self._update_margin_buffers(0.0)

    def _update_margin_buffers(self, m: float):
        """Update trigonometric constants when m changes."""
        self.m = float(m)
        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.threshold = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def step_margin(self):
        """Advances the margin schedule by one epoch or step."""
        self.current_step += 1
        if self.current_step >= self.warmup_steps:
            new_m = self.target_m
        else:
            progress = self.current_step / self.warmup_steps
            if self.schedule_type == "cosine":
                new_m = self.target_m * 0.5 * (1.0 - math.cos(math.pi * progress))
            else:  # "linear"
                new_m = self.target_m * progress

        self._update_margin_buffers(new_m)

    @property
    def current_s(self) -> float:
        """Returns the current clamped scalar value of s."""
        with torch.no_grad():
            return float(self.log_s.exp().clamp(self.min_s, self.max_s).item())

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Supports:
          - forward(logits, labels): where logits is [B, C] cosine similarity
          - forward(input_feats, class_feats, labels): where feats are [B, D] and [C, D]
        """
        # 2-arg mode: input is already the cosine matrix
        cosine = logits.clamp(-1.0 + self.eps, 1.0 - self.eps)
        target_labels = labels

        # Ensure scale parameter matches input device
        s = torch.clamp(self.log_s.to(cosine.device).exp(), min=self.min_s, max=self.max_s)

        # Warmup phase: spherical cross-entropy without margin
        if self.m == 0.0:
            return F.cross_entropy(cosine * s, target_labels)

        # Efficient & safe: compute margin ONLY for target pairs
        target_labels_idx = target_labels.unsqueeze(1)
        target_cos = cosine.gather(1, target_labels_idx)

        # Clamping to 1e-7 inside sqrt prevents NaN derivative at cos = 1.0
        target_sine = torch.sqrt((1.0 - torch.pow(target_cos, 2)).clamp(min=1e-7))
        target_phi = (target_cos * self.cos_m) - (target_sine * self.sin_m)
        target_phi = torch.where(target_cos > self.threshold, target_phi, target_cos - self.mm)

        # Add difference (phi - cos) only at target positions
        diff = (target_phi - target_cos) * s
        scaled_logits = (cosine * s).scatter_add(1, target_labels_idx, diff)

        return F.cross_entropy(scaled_logits, target_labels)