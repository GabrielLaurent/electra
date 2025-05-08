import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
import random


class ELECTRADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, mask_prob):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.inputs = []
        self.labels = []

        self._load_data(file_path)

    def _load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            tokens = self.tokenizer.tokenize(line)
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # Truncate sequences that exceed the maximum length.
            if len(token_ids) > self.max_length - 2: # Account for [CLS] and [SEP] tokens
                token_ids = token_ids[:self.max_length - 2]
            
            # Add [CLS] and [SEP] tokens
            token_ids = [self.tokenizer.cls_token_id] + token_ids + [self.tokenizer.sep_token_id]
            
            # Pad the sequence to max_length
            padding_length = self.max_length - len(token_ids)
            token_ids = token_ids + [self.tokenizer.pad_token_id] * padding_length

            # Convert to torch tensor
            token_ids = torch.tensor(token_ids)

            masked_input, labels = self._mask_tokens(token_ids)
            self.inputs.append(masked_input)
            self.labels.append(labels)

    def _mask_tokens(self, inputs):
        # Create a copy of input IDs to store the labels (original tokens)
        labels = inputs.clone()
        # Get the vocabulary size
        vocab_size = self.tokenizer.vocab_size
        
        # Create a mask tensor
        mask = torch.rand(inputs.shape) < self.mask_prob

        # Do not mask special tokens
        mask[inputs == self.tokenizer.cls_token_id] = False
        mask[inputs == self.tokenizer.sep_token_id] = False
        mask[inputs == self.tokenizer.pad_token_id] = False

        # Get the indices where masking is True
        mask_indices = mask.nonzero(as_tuple=True)[0]
        
        # Save the original tokens which would be replaced
        labels[~mask] = -100  # Set unmasked tokens to -100, so they are ignored in loss calculation

        # Replace masked tokens with [MASK] token 80% of the time
        indices_replaced = torch.bernoulli(torch.full(mask_indices.shape, 0.8)).bool()
        inputs[mask_indices[indices_replaced]] = self.tokenizer.mask_token_id

        # Replace masked tokens with random word 10% of the time
        indices_random = torch.bernoulli(torch.full(mask_indices.shape, 0.5)).bool() & ~indices_replaced
        random_words = torch.randint(vocab_size, mask_indices[indices_random].shape, dtype=torch.long)
        inputs[mask_indices[indices_random]] = random_words

        # The rest of the time (10% of the time) we keep the masked token unchanged

        return inputs, labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]