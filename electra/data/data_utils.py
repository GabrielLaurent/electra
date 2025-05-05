# electra/data/data_utils.py
#Implement data loading and preprocessing functions

def mask_tokens(tokens, mask_token_id, vocab_size, mlm_probability=0.15):
    import torch
    labels = tokens.clone()
    probability_matrix = torch.full(labels.shape, mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    tokens[indices_mask] = mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_mask
    random_words = torch.randint(vocab_size, labels.shape, dtype=torch.long)
    tokens[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return tokens, labels
