"""LSTM model architecture used for name classification."""

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM neural network for name classification.

    A multi-layer LSTM network with embedding layer for character-level
    name classification. Supports both single name classification (first/last)
    and positional classification (first_last/last_first).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 1,
    ):
        """Initialize LSTM model.

        Args:
            input_size: Size of vocabulary (number of unique characters)
            hidden_size: Hidden layer dimension
            output_size: Number of output classes
            num_layers: Number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=0)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            input_tensor: Character indices with shape ``[batch, sequence]``.

        Returns:
            Log-softmax probabilities for each class [batch_size, num_classes]
        """
        embedded = self.embedding(input_tensor)
        lengths = input_tensor.ne(0).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.softmax(self.fc(hidden[-1]))
