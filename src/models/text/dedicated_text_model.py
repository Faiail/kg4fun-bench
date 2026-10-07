import torch
from transformers import AutoModel
from src.models.utils import ReturnKeys


class DedicatedTextEncoder(torch.nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.projector = torch.nn.ModuleDict(
            {
                k: torch.nn.Sequential(
                    torch.nn.Linear(bbone_hidden_size, final_hidden_size),
                    torch.nn.LayerNorm(final_hidden_size),
                )
                for k in [ReturnKeys.INPUT, ReturnKeys.TARGET]
            }
        )
        self.schema_hidden_size = final_hidden_size

    def encode_input(self, input_dict) -> torch.Tensor:
        output = self.encoder(**input_dict)
        cls_token = output.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(
            self.projector[ReturnKeys.INPUT](cls_token), p=2, dim=-1
        )

    def encode_target(self, target_dict) -> torch.Tensor:
        output = self.encoder(**target_dict)
        cls_token = output.last_hidden_state[:, 0]
        return torch.nn.functional.normalize(
            self.projector[ReturnKeys.TARGET](cls_token), p=2, dim=-1
        )

    def forward(self, input: dict, target: dict) -> dict[str, torch.Tensor]:
        input_emb = self.encode_input(input)
        target_emb = self.encode_target(target)
        return {ReturnKeys.INPUT: input_emb, ReturnKeys.TARGET: target_emb}
