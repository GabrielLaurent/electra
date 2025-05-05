# electra/model/discriminator.py
import torch
import torch.nn as nn
from electra.model.transformer import TransformerLayer

class Discriminator(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.transformer_layers = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        for layer in self.transformer_layers:
            x = layer(x, mask)
        output = self.output_layer(x)
        output = self.sigmoid(output)
        return output