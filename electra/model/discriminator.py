import torch
import torch.nn as nn
from electra.model.transformer import TransformerLayer


class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_size)
        self.transformer_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.linear = nn.Linear(config.embedding_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids):
        """Forward pass of the discriminator.

        Args:
            input_ids: Input tensor of token IDs.

        Returns:
            A tensor of probabilities that each token is original.
        """
        embedded = self.embedding(input_ids)
        for layer in self.transformer_layers:
            embedded = layer(embedded)

        logits = self.linear(embedded)
        probs = self.sigmoid(logits)
        return probs