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
        self.criterions = torch.nn.ModuleDict()
        for k in keys:
            loss_name = (
                base_loss_name[k]
                if isinstance(base_loss_name, dict)
                else base_loss_name
            )
            if (
                isinstance(base_loss_kwargs, dict)
                and k in base_loss_kwargs
                and isinstance(base_loss_kwargs[k], dict)
            ):
                loss_kwargs = base_loss_kwargs[k]
            else:
                loss_kwargs = base_loss_kwargs
            self.criterions[k] = LOSS_REGISTRY[loss_name](**loss_kwargs)

        self.alphas = (
            alphas
            if alphas is not None
            else {"head": 0.5, "tail": 0.5, "rel": 1.0}
        )

    def forward(
        self,
        input_emb_dict: dict[str, torch.Tensor],
        target_emb_dict: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total_loss = 0.0
        for k, v in self.criterions.items():
            total_loss = total_loss + (
                self.alphas[k] * v(input_emb_dict[k], target_emb_dict[k], labels[k])
            )
        return total_loss

    def step_margin(self):
        for criterion in self.criterions.values():
            if hasattr(criterion, "step_margin"):
                criterion.step_margin()