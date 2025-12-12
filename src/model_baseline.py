import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineTSFM(nn.Module):
    """
    Transformer-style baseline that only uses structured price & macro features.
    No RAG / neighbor memory.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        horizon: int = 5,
    ):
        super().__init__()
        self.horizon = horizon

        # Project raw features into model dimension
        self.linear_in = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simple head: pool over time → predict H-day returns
        self.pred_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, horizon),
        )

    def forward(self, x_struct: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_struct: (B, T, input_dim)
        Returns:
            (B, horizon) future returns
        """
        h = self.linear_in(x_struct)   # (B, T, d_model)
        h = self.encoder(h)            # (B, T, d_model)
        h = h.mean(dim=1)              # temporal pooling
        out = self.pred_head(h)        # (B, H)
        return out
