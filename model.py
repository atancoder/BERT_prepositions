# type: ignore
from typing import Dict

import torch
from torch import nn
from torch.nn import functional


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
        self.transformer_blocks = nn.Sequential(
            *[
                Block(channels=config["channels"], num_heads=config["num_heads"])
                for _ in range(config["num_transformer_blocks"])
            ]
        )
        self.decoder = nn.Linear(
            in_features=config["channels"], out_features=vocab_size
        )

    def forward(self, input: torch.Tensor):
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
        output = self.transformer_blocks(embeddings)

        # decode back into vocabulary
        output = self.decoder(output)
        return output


class Block(nn.Module):
    """
    Transformer block responsible for encoding the input with context based on
    attention
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(channels, num_heads)
        self.ln1 = nn.LayerNorm(channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.ReLU(),
            nn.Linear(4 * channels, channels),
        )
        self.ln2 = nn.LayerNorm(channels)

    def forward(self, input: torch.Tensor):
        # sublayer 1
        output = input + self.multi_head_attention(input)  # Residual
        output = self.ln1(output)

        # sublayer 2
        output = output + self.feed_forward(output)  # Residual
        output = self.ln2(output)
        return output


class MultiHeadAttention(nn.Module):
    """
    Applies multihead attention to the encoding
    """

    def __init__(self, channels: int, num_heads: int):
        super().__init__()
        if channels % num_heads != 0:
            raise Exception("Num channels not divisble by num heads")
        subset_channels = channels // num_heads
        self.heads = nn.ModuleList(
            [SingleHeadAttention(channels, subset_channels) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(channels, channels)

    def forward(self, input: torch.Tensor):
        results = [head(input) for head in self.heads]
        output = torch.cat(results, dim=-1)  # dim = B,T,C
        output = self.linear(output)
        return output


class SingleHeadAttention(nn.Module):
    def __init__(self, orig_channels: int, subset_channels: int):
        super().__init__()
        self.orig_channels = orig_channels
        self.subset_channels = subset_channels
        self.query_transform = nn.Linear(orig_channels, subset_channels, bias=False)
        self.key_transform = nn.Linear(orig_channels, subset_channels, bias=False)
        self.value_transform = nn.Linear(orig_channels, subset_channels, bias=False)

    def forward(self, input: torch.Tensor):
        Q = self.query_transform(input)  # dim = B,T,subset_channels
        K = self.key_transform(input)  # dim = B,T,subset_channels
        V = self.value_transform(input)  # dim = B,T,subset_channels
        attention_mat = torch.matmul(Q, torch.transpose(K, 1, 2))  # dim = B,T,T
        attention_mat /= self.subset_channels**0.5
        softmax_att = functional.softmax(attention_mat, dim=1)
        output = torch.matmul(softmax_att, V)
        return output  # dim = B,T,subset_channels
