"""
Fingering prediction Transformer model.

Input: 26 tokens x 5 features
  - Previous (5): midi_offset, time_since, black_key, token_type=0, unused=-1
  - Current (1): pitch_class, 0, black_key, token_type=0.5, unused=-1
  - Lookahead (20): midi_offset, time_until, black_key, token_type=1, finger_hint

finger_hint: -1 = unknown, 0-1 = normalized finger (0-4 -> 0-1)

Output: single finger prediction (0-4)
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FingeringTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()

        # Input: 5 features per token
        self.input_proj = nn.Linear(5, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output: single finger prediction (5 classes)
        self.output_head = nn.Linear(d_model, 5)

        self.d_model = d_model

    def forward(self, tokens):
        """
        tokens: (batch, 26, 5) - 26 tokens, 5 features each
        Returns: (batch, 5) - 5 finger logits
        """
        x = self.input_proj(tokens)  # (batch, 26, d_model)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (batch, 26, d_model)

        # Use the current note token (index 5) for prediction
        current_token = x[:, 5, :]  # (batch, d_model)

        logits = self.output_head(current_token)  # (batch, 5)

        return logits
