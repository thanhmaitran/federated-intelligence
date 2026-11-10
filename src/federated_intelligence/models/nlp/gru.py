"""Character-level GRU for Shakespeare next-character prediction."""

from __future__ import annotations

from torch import nn


class CharGRU(nn.Module):
    def __init__(
        self,
        vocab_size: int = 80,
        embedding_dim: int = 8,
        hidden_size: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len) of character ids -> logits for the next character.
        out, _ = self.gru(self.embedding(x))
        return self.fc(out[:, -1, :])
