import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config.embedding_size, config.num_attention_heads, dropout=config.attention_dropout)
        self.feed_forward = FeedForward(config.embedding_size, config.intermediate_size, dropout=config.hidden_dropout)
        self.layer_norm1 = nn.LayerNorm(config.embedding_size)
        self.layer_norm2 = nn.LayerNorm(config.embedding_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, x):
        attention_output = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attention_output))
        feed_forward_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(feed_forward_output))
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads
        self.q_linear = nn.Linear(embedding_size, embedding_size)
        self.k_linear = nn.Linear(embedding_size, embedding_size)
        self.v_linear = nn.Linear(embedding_size, embedding_size)
        self.out_linear = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embedding_size)
        output = self.out_linear(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, embedding_size, intermediate_size, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, embedding_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x