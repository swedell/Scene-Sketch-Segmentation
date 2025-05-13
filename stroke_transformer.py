import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class StrokeTransformerEncoder(nn.Module):
    def __init__(self, num_layers=6, d_model=256, nhead=8, dim_feedforward=512):
        super(StrokeTransformerEncoder, self).__init__()
        self.embedding = nn.Linear(3, d_model)  # Embed the 3D stroke data to 256D
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)  # Embed the stroke data
        x = x.permute(1, 0, 2)  # Transformer expects input as (S, N, E)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Permute back to (N, S, E)
        return x