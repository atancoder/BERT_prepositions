# type: ignore
"""
Uses official Transformer libraries from Pytorch
"""
from typing import Dict, Optional

import torch
from torch import nn


class BERTModel(nn.Module):
    def __init__(self, vocab_size: int, config: Dict, device: str):
        super().__init__()
        self.device = device
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=config["channels"]
        )
        self.positional_encodings = nn.Embedding(
            num_embeddings=config["max_context_length"],
            embedding_dim=config["channels"],
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config["channels"],
            nhead=config["num_heads"],
            dim_feedforward=config["channels"] * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=config["num_transformer_blocks"]
        )

        self.decoder = nn.Linear(
            in_features=config["channels"], out_features=vocab_size
        )
        torch.nn.init.normal_(
            self.decoder.weight, mean=0.0, std=1 / (config["channels"] ** 0.5)
        )
        # Share the same weights as the original embedding matrix
        # Works better in papers and reduces model size
        # Have embedder weight take after linear weight, b/c it's important for linear
        # weight to be initialized properly
        self.embedder.weight = self.decoder.weight

    def forward(
        self,
        input: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ):
        """
        Input is a batch of encodings
        For each batch, there are T encodings
        Each encodings consists of C channels

        BERT is similar to an autoencoder, where we wish to encode
        the parts of a sentence and then reconstruct them back into words
        """
        B, T = input.shape
        positional_embeddings = self.positional_encodings(
            torch.arange(T).to(self.device)
        )
        embeddings = self.embedder(input) + positional_embeddings
        output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # decode back into vocabulary
        output = self.decoder(output)
        return output
