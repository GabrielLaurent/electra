import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from electra.model.electra_model import ElectraModel
from electra.data.datasets import PretrainingDataset  # Assuming you have this

class PreTrainer:
    def __init__(self, config, train_dataset, eval_dataset):
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.model = ElectraModel(config)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self, num_epochs):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            for batch in train_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                # Get Generator & Discriminator outputs
                gen_logits, disc_logits = self.model(input_ids)
                
                # Example losses, replace with actual ELECTRA loss
                generation_loss = nn.CrossEntropyLoss()(gen_logits.view(-1, self.config.vocab_size), input_ids.view(-1))
                discrimination_loss = nn.BCEWithLogitsLoss()(disc_logits.view(-1), torch.randint(0, 2, disc_logits.view(-1).shape, device=self.device).float())
                
                loss = generation_loss + discrimination_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(train_dataloader)}")

    def evaluate(self):
        self.model.eval()
        eval_dataloader = DataLoader(self.eval_dataset, batch_size=self.config.batch_size)
        total_loss = 0
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                gen_logits, disc_logits = self.model(input_ids)

                # Example losses, replace with actual ELECTRA loss
                generation_loss = nn.CrossEntropyLoss()(gen_logits.view(-1, self.config.vocab_size), input_ids.view(-1))
                discrimination_loss = nn.BCEWithLogitsLoss()(disc_logits.view(-1), torch.randint(0, 2, disc_logits.view(-1).shape, device=self.device).float())
                loss = generation_loss + discrimination_loss
                total_loss += loss.item()

        print(f"Evaluation Loss: {total_loss / len(eval_dataloader)}")

# Example Usage
if __name__ == '__main__':
    # Configure model
    class Config:
        def __init__(self):
            self.vocab_size = 10000
            self.embedding_dim = 128
            self.hidden_size = 256
            self.num_layers = 2
            self.num_attention_heads = 4
            self.intermediate_size = 512
            self.max_position_embeddings = 512
            self.type_vocab_size = 2
            self.pad_token_id = 0
            self.learning_rate = 1e-4
            self.batch_size = 32

    config = Config()

    # Create dummy datasets for example
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, vocab_size, seq_len, size):
            self.data = [{'input_ids': torch.randint(0, vocab_size, (seq_len,))} for _ in range(size)]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    train_dataset = DummyDataset(config.vocab_size, 128, 1000)
eval_dataset = DummyDataset(config.vocab_size, 128, 200)

    # Initialize and run the trainer
    trainer = PreTrainer(config, train_dataset, eval_dataset)
    trainer.train(num_epochs=2)
    trainer.evaluate()