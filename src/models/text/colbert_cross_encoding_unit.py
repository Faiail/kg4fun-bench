import torch
import torch.nn as nn
import torch.nn.functional as F


class ColBERTCrossEncodingUnit(nn.Module):
    """
    Late-Interaction MaxSim Unit based on Khattab & Zaharia (ColBERT, SIGIR 2020).
    
    Architecture:
    - Linear projection of tokens to unit_hidden_dim (e.g. 96 or 128) + L2 normalization.
    - Token-level MaxSim: For every valid input token, compute the maximum cosine
      similarity across all valid target tokens, then average across input tokens.
    - Zero cross-attention layers, highly efficient and focused on sharp keyword matching.
    """
    def __init__(
        self,
        hidden_dim: int,
        unit_hidden_dim: int,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.unit_hidden_dim = unit_hidden_dim
        self.proj_in = nn.Linear(hidden_dim, unit_hidden_dim)
        self.proj_tgt = nn.Linear(hidden_dim, unit_hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(
        self,
        input_cls: torch.Tensor,
        input_tokens: torch.Tensor,
        target_cls: torch.Tensor,
        target_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 1. Project and L2 normalize tokens
        h_in = F.normalize(self.dropout(self.proj_in(input_tokens)), p=2, dim=-1)   # (B_in, L_in, d)
        h_tgt = F.normalize(self.dropout(self.proj_tgt(target_tokens)), p=2, dim=-1) # (B_tgt, L_tgt, d)

        # 2. Token-to-token cosine similarity tensor with target padding masked out
        pad_tgt = ~target_mask.bool().unsqueeze(0).unsqueeze(2) # (1, B_tgt, 1, L_tgt)
        aff_cos = torch.matmul(
            h_in.unsqueeze(1),
            h_tgt.unsqueeze(0).transpose(-1, -2)
        ).masked_fill(pad_tgt, -1e4) # (B_in, B_tgt, L_in, L_tgt)

        # 3. MaxSim: Max over target tokens, mean over valid input tokens
        max_tgt = aff_cos.max(dim=-1).values # (B_in, B_tgt, L_in)
        m_in = input_mask.unsqueeze(1)       # (B_in, 1, L_in)
        in_lens = input_mask.sum(dim=-1, keepdim=True).clamp(min=1) # (B_in, 1)

        scores = (max_tgt * m_in).sum(dim=-1) / in_lens # (B_in, B_tgt)
        return scores

