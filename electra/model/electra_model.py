import torch
import torch.nn as nn

from electra.model.generator import Generator
from electra.model.discriminator import Discriminator


class ElectraModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)

    def forward(self, input_ids):
        generator_output = self.generator(input_ids)
        discriminator_output = self.discriminator(input_ids)
        return generator_output, discriminator_output


if __name__ == '__main__':
    # Example usage
    class Config:
        def __init__(self):
            self.vocab_size = 1000
            self.embedding_dim = 128
            self.hidden_size = 256
            self.num_layers = 2
            self.num_attention_heads = 4
            self.intermediate_size = 512
            self.max_position_embeddings = 512
            self.type_vocab_size = 2
            self.pad_token_id = 0

    config = Config()
    model = ElectraModel(config)

    # Generate some random input IDs
    input_ids = torch.randint(0, config.vocab_size, (1, 128))

    # Get the generator and discriminator outputs
    generator_output, discriminator_output = model(input_ids)

    print("Generator Output Shape:", generator_output.shape)
    print("Discriminator Output Shape:", discriminator_output.shape)