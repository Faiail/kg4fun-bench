import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCrossEncodingUnit(nn.Module):
    """
    Bidirectional Multi-Grained Cross-Attention Unit with Late Interaction.
    
    Combines:
    1. Global-to-Global (G2G): CLS cosine similarity.
    2. Global-to-Local (T->I): Target queries Input tokens (Luo et al., MIMIC).
    3. Local-to-Global (I->T): Input queries Target tokens (Reciprocal BiDAF flow).
    4. Local-to-Local (L2L):   ColBERT MaxSim soft token alignment.
    5. Learnable Score Fusion: Dynamically weights multi-grained evidence.
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

        # Token projections
        self.proj_in = nn.Linear(hidden_dim, unit_hidden_dim)
        self.proj_tgt = nn.Linear(hidden_dim, unit_hidden_dim)

        # CLS projections
        self.proj_cls_in = nn.Linear(hidden_dim, unit_hidden_dim)
        self.proj_cls_tgt = nn.Linear(hidden_dim, unit_hidden_dim)

        self.norm_ctx_in = nn.LayerNorm(unit_hidden_dim)
        self.norm_ctx_tgt = nn.LayerNorm(unit_hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # Learnable score fusion initialized to uniform average
        self.score_fusion = nn.Linear(4, 1)
        nn.init.constant_(self.score_fusion.weight, 0.25)
        nn.init.zeros_(self.score_fusion.bias)

    def forward(
        self,
        input_cls: torch.Tensor,
        input_tokens: torch.Tensor,
        target_cls: torch.Tensor,
        target_tokens: torch.Tensor,
        input_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        B_in, L_in, _ = input_tokens.shape
        B_tgt, L_tgt, _ = target_tokens.shape

        # 1. Linear Projections
        h_in = self.dropout(self.proj_in(input_tokens))      # (B_in, L_in, d)
        h_tgt = self.dropout(self.proj_tgt(target_tokens))   # (B_tgt, L_tgt, d)
        c_in = self.dropout(self.proj_cls_in(input_cls))     # (B_in, d)
        c_tgt = self.dropout(self.proj_cls_tgt(target_cls))  # (B_tgt, d)

        # L2 Normalized representations for cosine alignment
        c_in_norm = F.normalize(c_in, p=2, dim=-1)
        c_tgt_norm = F.normalize(c_tgt, p=2, dim=-1)
        h_in_norm = F.normalize(h_in, p=2, dim=-1)
        h_tgt_norm = F.normalize(h_tgt, p=2, dim=-1)

        # 2. Global-to-Global (G2G) Cosine Similarity: (B_in, B_tgt)
        s_g2g = torch.matmul(c_in_norm, c_tgt_norm.transpose(-1, -2))

        # 3. Scaled Dot-Product Affinity Tensor for Cross-Attention
        scale = float(self.unit_hidden_dim) ** 0.5
        aff_raw = torch.matmul(
            h_in.unsqueeze(1),                   # (B_in, 1, L_in, d)
            h_tgt.unsqueeze(0).transpose(-1, -2) # (1, B_tgt, d, L_tgt)
        ) / scale                                # (B_in, B_tgt, L_in, L_tgt)

        # Broadcastable mask tensors
        pad_in = ~input_mask.bool().unsqueeze(1).unsqueeze(-1)  # (B_in, 1, L_in, 1)
        pad_tgt = ~target_mask.bool().unsqueeze(0).unsqueeze(2) # (1, B_tgt, 1, L_tgt)

        valid_in_ctx = input_mask.unsqueeze(1).unsqueeze(-1)    # (B_in, 1, L_in, 1)
        valid_tgt_ctx = target_mask.unsqueeze(0).unsqueeze(-1)  # (1, B_tgt, L_tgt, 1)

        in_lens = input_mask.sum(dim=-1, keepdim=True)          # (B_in, 1)
        tgt_lens = target_mask.sum(dim=-1, keepdim=True)        # (B_tgt, 1)

        # 4. Global-to-Local (Target queries Input tokens, Luo et al. MIMIC)
        # Softmax over L_in (dim 2)
        aff_t2i = aff_raw.masked_fill(pad_in, -1e4)
        attn_t2i = F.softmax(aff_t2i, dim=2)

        ctx_i = torch.matmul(
            h_in.unsqueeze(1).transpose(-1, -2), attn_t2i
        ).transpose(-1, -2)                                      # (B_in, B_tgt, L_tgt, d)

        # Masked average over valid target tokens (dim 2)
        ctx_i_pool = (ctx_i * valid_tgt_ctx).sum(dim=2) / tgt_lens.unsqueeze(0).clamp(min=1)
        ctx_i_pool = self.norm_ctx_in(ctx_i_pool)
        s_t2i = torch.sum(c_tgt_norm.unsqueeze(0) * F.normalize(ctx_i_pool, p=2, dim=-1), dim=-1)

        # 5. Local-to-Global (Input queries Target tokens, reciprocal flow)
        # Softmax over L_tgt (dim 3)
        aff_i2t = aff_raw.masked_fill(pad_tgt, -1e4)
        attn_i2t = F.softmax(aff_i2t, dim=3)

        ctx_t = torch.matmul(attn_i2t, h_tgt.unsqueeze(0))       # (B_in, B_tgt, L_in, d)

        # Masked average over valid input tokens (dim 2)
        ctx_t_pool = (ctx_t * valid_in_ctx).sum(dim=2) / in_lens.unsqueeze(1).clamp(min=1)
        ctx_t_pool = self.norm_ctx_tgt(ctx_t_pool)
        s_i2t = torch.sum(c_in_norm.unsqueeze(1) * F.normalize(ctx_t_pool, p=2, dim=-1), dim=-1)

        # 6. ColBERT MaxSim (Token-to-token fine alignment via normalized dot product)
        aff_cos = torch.matmul(
            h_in_norm.unsqueeze(1),
            h_tgt_norm.unsqueeze(0).transpose(-1, -2)
        ).masked_fill(pad_tgt, -1e4)                             # (B_in, B_tgt, L_in, L_tgt)

        max_tgt = aff_cos.max(dim=-1).values                     # (B_in, B_tgt, L_in)
        m_in = input_mask.unsqueeze(1)                           # (B_in, 1, L_in)
        s_maxsim = (max_tgt * m_in).sum(dim=-1) / in_lens.clamp(min=1) # (B_in, B_tgt)

        # 7. Multi-Grained Score Fusion
        features = torch.stack([s_g2g, s_t2i, s_i2t, s_maxsim], dim=-1) # (B_in, B_tgt, 4)
        matching_scores = self.score_fusion(features).squeeze(-1)        # (B_in, B_tgt)

        return matching_scores
