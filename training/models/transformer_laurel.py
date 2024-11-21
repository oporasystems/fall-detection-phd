import torch
import torch.nn as nn

class LAuReL(nn.Module):
    def __init__(self, input_dim):
        super(LAuReL, self).__init__()
        self.alpha = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x, residual):
        """
        LAuReL: Learned Augmented Residual Layer
        Args:
            x: The output from the main transformation.
            residual: The input to the residual connection.
        """
        return self.alpha * x + self.beta * residual


class TransformerBlockWithLAuReL(nn.Module):
    def __init__(self, input_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlockWithLAuReL, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(input_dim)
        self.laurel1 = LAuReL(input_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim),
        )
        self.ln2 = nn.LayerNorm(input_dim)
        self.laurel2 = LAuReL(input_dim)

    def forward(self, x):
        # Self-attention block
        attn_output, _ = self.attention(x, x, x)
        x = self.laurel1(attn_output, self.ln1(x))

        # Feed-forward block
        ff_output = self.feed_forward(x)
        x = self.laurel2(ff_output, self.ln2(x))

        return x


class TransformerWithLAuReL(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, num_classes, hidden_dim, dropout):
        super(TransformerWithLAuReL, self).__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithLAuReL(input_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=0)  # Global average pooling for sequence data
        return self.fc(x)