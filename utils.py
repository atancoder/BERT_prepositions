# type: ignore
import copy
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from prepositions import PREPOSITIONS_LIST

def create_padded_att_mask(attention_mask):
    """
    Attention mask is a 2D tensor representing 0s for padding and 1s for not padding
    For masking, we want to invert the 0s and 1s
    """
    return (1 - attention_mask).type(torch.float)
    
def flatten_indices(batch_indices: List[List[int]], context_length: int) -> List[int]:
    """
    Flattens the batch_indices list into 1D
    Assumes each batch has context_length indices

    e.g [[1,3], [0, 3]]  -> [1,3,4,7]

    [0,3] -> [4,7] because it's the 2nd batch
    """
    flattened_indices: List[int] = []
    for batch_id, batch in enumerate(batch_indices):
        batch_start_idx = batch_id * context_length
        new_indices = [batch_start_idx + idx for idx in batch]
        flattened_indices += new_indices
    return flattened_indices


def get_books_dataloader(batch_size, max_size=None):
    # Load the BookCorpus dataset
    bookcorpus = load_dataset("bookcorpus")

    # Access the train split
    train_dataset = bookcorpus["train"]

    random_indices = torch.randperm(len(train_dataset)).tolist()
    if max_size:
        random_indices = random_indices[:max_size]
    train_dataset = train_dataset.select(random_indices)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def apply_masking(token_ids, tokenizer, preposition_tokens_ids):
    """
    Masks out prepositions
    Returns the sentence token_ids with some [MASK] token_ids and 2D array of which token_ids
    were masked
    """
    token_ids = copy.deepcopy(token_ids)
    masked_indices = []
    for token_batch in token_ids:
        batch_masked_indicies = []
        for i in range(len(token_batch)):
            token_id = token_batch[i].item()
            if token_id == tokenizer.pad_token_id:
                break
            if token_id in preposition_tokens_ids:
                rand = torch.rand(1)
                if rand <= 0.5:
                    # mask out
                    token_batch[i] = tokenizer.mask_token_id
                elif rand <= 0.8:
                    token_batch[i] = torch.randint(tokenizer.vocab_size, (1,))
                else:
                    # Leave token untouched
                    pass
                batch_masked_indicies.append(i)
        masked_indices.append(batch_masked_indicies)
    return token_ids, masked_indices


def get_preposition_token_ids(tokenizer):
    return set(tokenizer.convert_tokens_to_ids(PREPOSITIONS_LIST))


def batch_to_token_ids(
    batch_sentences, tokenizer, context_length, device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, List[int]]:
    preposition_tokens_ids = get_preposition_token_ids(tokenizer)
    tokenized_sentences = tokenizer(
        batch_sentences,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=context_length,
    )
    attention_mask = create_padded_att_mask(tokenized_sentences.attention_mask)
    tokenized_sentences = tokenized_sentences.input_ids
    masked_sentences, masked_indices = apply_masking(
        tokenized_sentences,
        tokenizer,
        preposition_tokens_ids,
    )
    
    return (
        tokenized_sentences.to(device),
        attention_mask.to(device),
        masked_sentences.to(device),
        masked_indices,
    )
