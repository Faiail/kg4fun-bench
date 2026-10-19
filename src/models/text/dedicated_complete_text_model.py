import torch
from transformers import AutoModel
from typing import Tuple
from src.models.utils import ReturnKeys


class DedicatedCompleteTextEncoder(torch.nn.Module):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(**bbone_config)
        self.node_projector = torch.nn.ModuleDict(
            {
                k: torch.nn.Sequential(
                    torch.nn.Linear(bbone_hidden_size, final_hidden_size),
                    torch.nn.LayerNorm(final_hidden_size),
                )
                for k in [ReturnKeys.INPUT, ReturnKeys.TARGET]
            }
        )
        self.edge_projector = torch.nn.ModuleDict(
            {
                k: torch.nn.Sequential(
                    torch.nn.Linear(bbone_hidden_size, final_hidden_size),
                    torch.nn.LayerNorm(final_hidden_size),
                )
                for k in [ReturnKeys.INPUT, ReturnKeys.TARGET]
            }
        )
        self.schema_hidden_size = final_hidden_size

    def encode_input(self, input_dict: dict, task: str = "node") -> torch.Tensor:
        output = self.encoder(**input_dict)
        cls_token = output.last_hidden_state[:, 0]
        proj = (
            self.edge_projector[ReturnKeys.INPUT]
            if task == "edge"
            else self.node_projector[ReturnKeys.INPUT]
        )
        return torch.nn.functional.normalize(proj(cls_token), p=2, dim=-1)

    def encode_target(self, target_dict: dict, task: str = "node") -> torch.Tensor:
        output = self.encoder(**target_dict)
        cls_token = output.last_hidden_state[:, 0]
        proj = (
            self.edge_projector[ReturnKeys.TARGET]
            if task == "edge"
            else self.node_projector[ReturnKeys.TARGET]
        )
        return torch.nn.functional.normalize(proj(cls_token), p=2, dim=-1)

    def forward(
        self,
        input_head,
        input_rel,
        input_tail,
        target_head,
        target_rel,
        target_tail,
    ) -> Tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        input_head_emb = self.encode_input(input_head, task="node")
        input_rel_emb = self.encode_input(input_rel, task="edge")
        input_tail_emb = self.encode_input(input_tail, task="node")
        target_head_emb = self.encode_target(target_head, task="node")
        target_rel_emb = self.encode_target(target_rel, task="edge")
        target_tail_emb = self.encode_target(target_tail, task="node")
        return (
            {
                ReturnKeys.HEAD: input_head_emb,
                ReturnKeys.REL: input_rel_emb,
                ReturnKeys.TAIL: input_tail_emb,
            },
            {
                ReturnKeys.HEAD: target_head_emb,
                ReturnKeys.REL: target_rel_emb,
                ReturnKeys.TAIL: target_tail_emb,
            },
        )
