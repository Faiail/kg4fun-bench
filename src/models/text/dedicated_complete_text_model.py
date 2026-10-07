import torch
from typing import Tuple
from src.models.utils import ReturnKeys
from .dedicated_text_model import DedicatedTextEncoder


class DedicatedCompleteTextEncoder(DedicatedTextEncoder):
    def __init__(
        self,
        bbone_config: dict,
        bbone_hidden_size: int,
        final_hidden_size: int,
    ):
        super().__init__(
            bbone_config=bbone_config,
            bbone_hidden_size=bbone_hidden_size,
            final_hidden_size=final_hidden_size,
        )

    def forward(
        self,
        input_head,
        input_rel,
        input_tail,
        target_head,
        target_rel,
        target_tail,
    ) -> Tuple[dict[str, torch.Tensor]]:
        input_head_emb = self.encode_input(input_head)
        input_rel_emb = self.encode_input(input_rel)
        input_tail_emb = self.encode_input(input_tail)
        target_head_emb = self.encode_target(target_head)
        target_rel_emb = self.encode_target(target_rel)
        target_tail_emb = self.encode_target(target_tail)
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
