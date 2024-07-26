import time
import shutil
from typing import List

import torch
from torch.nn.functional import cross_entropy

from utils import batch_to_token_ids, create_padded_att_mask


def flatten_indices(indices_2d: List[List[int]], seq_length: int) -> List[int]:
    """
    Flattens 2D array of indices into a 1D array of indices
    """
    indices_1d = []
    for batch_id, batch in enumerate(indices_2d):
        for idx in batch:
            indices_1d.append(idx + (batch_id * seq_length))
    return indices_1d


def compute_loss(model, batch_sentences, tokenizer, config, device, loss_fn):
    """
    Makes predictions and computes loss between predictions and labels
    only for masked_indices

    Flatten the masked indices so we can compute loss in parallel
    """
    (
        tokenized_sentences,
        masked_sentences,
        masked_indices,
    ) = batch_to_token_ids(batch_sentences, tokenizer, config["context_length"], device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        predictions = model(
            masked_sentences, src_key_padding_mask=create_padded_att_mask(tokenized_sentences.attention_mask)
        )

    B, T, _ = predictions.shape
    flattened_mask_indices = flatten_indices(masked_indices, T)
    predictions = predictions.reshape(B * T, -1)[flattened_mask_indices]
    orig_sentences_tokens = tokenized_sentences.input_ids.reshape(B * T)[flattened_mask_indices]
    return loss_fn(predictions, orig_sentences_tokens)


def train(
    model,
    dataloader,
    tokenizer,
    config,
    device,
    saved_model_file,
):
    start = time.time()
    size = len(dataloader.dataset)
    loss_fn = cross_entropy

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    model.train()
    for _ in range(config["epochs"]):
        for batch_id, batch in enumerate(dataloader):
            loss = compute_loss(
                model, batch["text"], tokenizer, config, device, loss_fn
            )

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_id % 10 == 0:
                loss, current = loss.item(), (batch_id + 1) * len(batch["text"])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print(f"{((time.time() - start) / 60):.2f} minutes have elapsed")
                if saved_model_file:
                    torch.save(model.state_dict(), "tmp_model.pt")
                    shutil.move("tmp_model.pt", saved_model_file)
                    print("Saved model")

def evaluate_model(model, tokenizer, dataloader, config, device) -> float:
    model.eval()
    total_loss = 0
    loss_fn = cross_entropy
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            loss = compute_loss(
                model, batch["text"], tokenizer, config, device, loss_fn
            )
            total_loss += loss.item()
    return total_loss / len(dataloader)
