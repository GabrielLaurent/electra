# examples/finetune.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

# Define a simple dataset for example purposes
class SimpleDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label)
        }


# Example fine-tuning script.  Fill in the details!
if __name__ == '__main__':
    # Configuration
    model_name = 'bert-base-uncased' # Or path to your pre-trained ELECTRA discriminator
    num_labels = 2  # Example: Binary classification
    batch_size = 32
    max_length = 128
    learning_rate = 2e-5
    epochs = 3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained model and tokenizer
    model = BertModel.from_pretrained(model_name).to(device) # Replace with ELECTRA Discriminator
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Add a classification head
    model.classifier = nn.Linear(model.config.hidden_size, num_labels).to(device)

    #Load some example data
    texts = ["This is a positive example", "This is a negative example"]
    labels = [1, 0]

    # Prepare DataLoader
    dataset = SimpleDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Training Loop (very basic example)
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)[0]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(outputs.view(-1, num_labels), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    print("Fine-tuning complete!")