# electra/model/generator.py
import torch
import torch.nn as nn
from electra.model.transformer import TransformerLayer

class Generator(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer_layers = nn.ModuleList([TransformerLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        for layer in self.transformer_layers:
            embedded = layer(embedded, mask)
        output = self.output_layer(embedded)
        return output