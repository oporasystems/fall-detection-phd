import torch.nn as nn
import torch
from training.utils.constants import max_sequence_length
import numpy as np


# Positional Encoding Class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=max_sequence_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: batch_size x seq_len x d_model
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x


# Multi-Scale Transformer Model
class MultiScaleTransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim=128, dropout=0.1):
        super(MultiScaleTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Define multiple scales
        self.scale_factors = [1, 2, 4]  # Adjust scales as needed

        self.transformers = nn.ModuleList()
        self.position_encodings = nn.ModuleList()

        for scale in self.scale_factors:
            # Create transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.transformers.append(transformer)

            # Positional encoding
            max_len = int(np.ceil(max_sequence_length / scale))
            pe = PositionalEncoding(hidden_dim, max_len=max_len)
            self.position_encodings.append(pe)

        # Linear layer to match the transformer d_model
        self.linear = nn.Linear(input_dim, hidden_dim)

        # Classification layer
        self.fc = nn.Linear(hidden_dim * len(self.scale_factors), num_classes)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: batch_size x seq_len x input_dim
        outputs = []
        for idx, scale in enumerate(self.scale_factors):
            # Downsample the sequence
            if scale > 1:
                x_scaled = x[:, ::scale, :]
            else:
                x_scaled = x
            # Linear projection
            x_scaled = self.linear(x_scaled)
            # Add positional encoding
            x_scaled = self.position_encodings[idx](x_scaled)
            # Pass through transformer
            x_scaled = self.transformers[idx](x_scaled)
            # Pooling (e.g., global average pooling)
            x_pooled = x_scaled.mean(dim=1)
            outputs.append(x_pooled)

        # Concatenate outputs from different scales
        x_concat = torch.cat(outputs, dim=1)  # Shape: batch_size x (hidden_dim * num_scales)
        x_concat = self.dropout_layer(self.relu(x_concat))
        # Classification layer
        x_out = self.fc(x_concat)
        return x_out