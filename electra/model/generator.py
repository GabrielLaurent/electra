import torch
import torch.nn as nn
from electra.model.transformer import TransformerEncoder


class Generator(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = TransformerEncoder(num_layers, hidden_size, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dense = nn.Linear(hidden_size, vocab_size)
        self.activation = nn.GELU()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings

    def forward(self, input_ids):
        # Input shape: (batch_size, sequence_length)
        # Output shape: (batch_size, sequence_length, vocab_size)
        embedded_output = self.embedding(input_ids)
        transformer_output = self.transformer(embedded_output)
        normalized_output = self.layer_norm(transformer_output)
        prediction_scores = self.dense(normalized_output)

        return prediction_scores


if __name__ == '__main__':
    # Example usage
    vocab_size = 10000
    hidden_size = 256
    num_layers = 2
    num_attention_heads = 4
    intermediate_size = 1024
    hidden_act = 'gelu'
    hidden_dropout_prob = 0.1
    attention_probs_dropout_prob = 0.1
    max_position_embeddings = 512

    generator = Generator(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_act=hidden_act,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
        max_position_embeddings=max_position_embeddings
    )

    # Create a dummy input
    batch_size = 32
    sequence_length = 128
    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_length))

    # Pass the input through the generator
    output = generator(input_ids)

    # Print the output shape
    print("Output shape:", output.shape)  # Expected output: torch.Size([32, 128, 10000])
