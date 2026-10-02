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
        self.input_encoder = AutoModel.from_pretrained(**bbone_config)
        self.target_encoder = AutoModel.from_pretrained(**bbone_config)
        self.input_projector = torch.nn.Linear(bbone_hidden_size, final_hidden_size)
        self.target_projector = torch.nn.Linear(bbone_hidden_size, final_hidden_size)
        self.input_layernorm = torch.nn.LayerNorm(final_hidden_size)
        self.target_layernorm = torch.nn.LayerNorm(final_hidden_size)
        self.schema_hidden_size = final_hidden_size

    def encode_input(self, input_dict) -> torch.Tensor:
        output = self.input_encoder(**input_dict)
        cls_token = output.last_hidden_state[:, 0]
        return self.input_layernorm(self.input_projector(cls_token))

    def encode_target(self, target_dict) -> torch.Tensor:
        output = self.target_encoder(**target_dict)
        cls_token = output.last_hidden_state[:, 0]
        return self.target_layernorm(self.target_projector(cls_token))

    def forward(self, input: dict, target: dict) -> dict[str, torch.Tensor]:
        input_emb = self.encode_input(input)
        target_emb = self.encode_target(target)
        return {ReturnKeys.INPUT: input_emb, ReturnKeys.TARGET: target_emb}
