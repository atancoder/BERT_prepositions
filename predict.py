import copy

import torch
from torch.nn import functional as F

from utils import batch_to_token_ids
from train import flatten_indices


def infer_masked_tokens(masked_sentences_tokens, predicted_tokens, masked_indices):
    B, T = masked_sentences_tokens.shape
    flattened_indices = flatten_indices(masked_indices, T)
    new_sentence_tokens = copy.deepcopy(masked_sentences_tokens).reshape(B*T)
    new_sentence_tokens[flattened_indices] = predicted_tokens.view(B*T)[flattened_indices]
    return new_sentence_tokens

def remove_pad(sentence):
    if "[PAD]" in sentence:
        pad_idx = sentence.index("[PAD]")
        return sentence[:pad_idx]
    return sentence

def predict(model, batch, tokenizer, context_length, device):
    (
        tokenized_sentences,
        attention_mask,
        masked_sentences_tokens,
        masked_indices,
    ) = batch_to_token_ids(batch["text"], tokenizer, context_length, device)

    model.eval()
    with torch.no_grad():
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(masked_sentences_tokens, src_key_padding_mask=attention_mask)
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        probabilities = F.softmax(logits, dim=-1)
        predicted_tokens = torch.multinomial(probabilities, num_samples=1)

        # replace predicted_tokens for masked_indices
        new_sentences_tokens = infer_masked_tokens(masked_sentences_tokens, predicted_tokens, masked_indices)
        new_sentences_tokens = new_sentences_tokens.reshape(B, T)

        orig_sentences = tokenizer.batch_decode(tokenized_sentences)
        masked_sentences = tokenizer.batch_decode(masked_sentences_tokens)
        new_sentences = tokenizer.batch_decode(new_sentences_tokens)

        for i in range(B):
            print(f"Orig sentence: {remove_pad(orig_sentences[i])}")
            print(f"Mask sentence: {remove_pad(masked_sentences[i])}")
            print(f"Pred sentence: {remove_pad(new_sentences[i])}\n\n")
