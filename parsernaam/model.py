import torch
import torch.nn as nn
from typing import Optional


class LSTM(nn.Module):
    """LSTM neural network for name classification.
    
    A multi-layer LSTM network with embedding layer for character-level
    name classification. Supports both single name classification (first/last)
    and positional classification (first_last/last_first).
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """Initialize LSTM model.
        
        Args:
            input_size: Size of vocabulary (number of unique characters)
            hidden_size: Hidden layer dimension
            output_size: Number of output classes
            num_layers: Number of LSTM layers
        """
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The nn.Embedding layer returns a new tensor with dimension (sequence_length, 1, hidden_size)
        self.embedding = nn.Embedding(input_size, hidden_size)
        # LSTM layer expects a tensor of dimension (batch_size, sequence_length, hidden_size).
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            input: Input tensor of character indices [batch_size, sequence_length]
            
        Returns:
            Log-softmax probabilities for each class [batch_size, num_classes]
        """
        embedded = self.embedding(input.type(torch.IntTensor).to(input.device))
        # embedded = embedded.view(embedded.shape[0],-1,embedded.shape[3])
        h0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(input.device)
        c0 = torch.zeros(self.num_layers, embedded.size(0), self.hidden_size).to(input.device)
        out, _ = self.lstm(embedded, (h0, c0))
        out = out[:, -1, :]  # get the output of the last time step
        out = self.fc(out)
        out = self.softmax(out)
        return out
    