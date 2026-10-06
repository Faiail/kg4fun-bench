from .registry import LOSS_REGISTRY
import torch


class CompleteLoss(torch.nn.Module):
    def __init__(
        self,
        keys: list[str],
        base_loss_name,
        base_loss_kwargs,
        alphas: dict[str, float] = None,
    ):
        super().__init__()
        self.criterions = torch.nn.ModuleDict(
            {k: LOSS_REGISTRY[base_loss_name](**base_loss_kwargs) for k in keys}
        )
        self.alphas = alphas if alphas is not None else {k: 1.0 for k in keys}

    def forward(
        self,
        input_emb_dict: dict[str, torch.Tensor],
        target_emb_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss = 0.0
        for k, v in self.criterions.items():
            total_loss = total_loss + (
                self.alphas[k] * v(input_emb_dict[k] @ target_emb_dict[k].T, labels[k])
            )
        return total_loss
