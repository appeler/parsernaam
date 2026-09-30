"""Historical model and encoding for evaluation of pre-0.4 weights."""

from collections.abc import Sequence

import torch
from torch import nn


class LegacyLSTM(nn.Module):
    """Read the final padded timestep used by the original published weights."""

    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, num_layers: int
    ):
        """Reconstruct the original state-dictionary-compatible architecture."""
        super().__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Classify the hidden state after trailing padding."""
        outputs, _ = self.lstm(self.embedding(inputs))
        return self.softmax(self.fc(outputs[:, -1, :]))


def encode_legacy(names: Sequence[str], seq_len: int, vocabulary: str) -> torch.Tensor:
    """Encode names for the original published-weight comparison.

    Args:
        names: Names in the desired case.
        seq_len: Original model sequence length.
        vocabulary: Original character order.

    Returns:
        Tensor using the original shared unknown and padding token.
    """
    unknown = len(vocabulary) + 1
    result = torch.full((len(names), seq_len), unknown, dtype=torch.long)
    indices = {character: index for index, character in enumerate(vocabulary)}
    for row, name in enumerate(names):
        values = [indices.get(character, unknown) for character in name[:seq_len]]
        result[row, : len(values)] = torch.tensor(values, dtype=torch.long)
    return result
