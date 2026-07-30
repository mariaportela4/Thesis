import torch
import torch.nn as nn
import torch.nn.functional as F

"""
1-D Convolutional Neural Network for binary protein sequence classification.

The model treats each pre-computed embedding vector as a 1-D signal with a
single input channel and extracts local features through two convolutional
blocks before collapsing the sequence dimension with global max-pooling and
projecting to a single logit.

Architecture summary
--------------------
Input  : (batch_size, input_dim)            — flat embedding vector
Reshape: (batch_size, 1, input_dim)         — add channel dimension
Conv1  : 1  → 64  channels, kernel 5, same padding → BN → ReLU
Conv2  : 64 → 128 channels, kernel 5, same padding → BN → ReLU
Pool   : AdaptiveMaxPool1d(1)               — (batch_size, 128, 1)
Flatten: (batch_size, 128)
FC     : 128 → 64 → ReLU → Dropout(0.3) → 1 (raw logit)

The output is a single unbounded logit per sample. Apply
``torch.sigmoid`` for a probability or use ``nn.BCEWithLogitsLoss``
directly during training (numerically more stable than BCE + sigmoid).
"""

class CNNClassifier(nn.Module):
    """
    Binary classifier built on a two-block 1-D CNN.

    Each convolutional block applies a Conv1d layer followed by Batch
    Normalisation and ReLU activation. After the second block, Adaptive
    Max Pooling reduces the temporal dimension to a single value per
    channel, yielding a fixed-size feature vector regardless of the
    original embedding dimensionality. A small fully-connected head then
    maps this vector to a scalar logit.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding vector
        (e.g. 1024 for ProtBERT, 480–2560 for ESM-2).

    Attributes
    ----------
    conv1 : nn.Conv1d
        First convolutional layer: 1 → 64 channels, kernel size 5.
    bn1 : nn.BatchNorm1d
        Batch normalisation applied after ``conv1``.
    conv2 : nn.Conv1d
        Second convolutional layer: 64 → 128 channels, kernel size 5.
    bn2 : nn.BatchNorm1d
        Batch normalisation applied after ``conv2``.
    pool : nn.AdaptiveMaxPool1d
        Global max-pool that collapses the temporal dimension to 1,
        making the model input-size agnostic.
    classifier : nn.Sequential
        Fully-connected head: Linear(128→64) → ReLU → Dropout(0.3)
        → Linear(64→1).
    """

    def __init__(self, input_dim,  dropout=0.3):
        super(CNNClassifier, self).__init__()

        # --- Convolutional block 1 ---
        # padding=2 with kernel_size=5 preserves the temporal dimension
        # ("same" convolution), so no positional information is lost
        # before pooling.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

         # --- Convolutional block 2 ---
        # Doubles the number of feature maps to capture more complex
        # local patterns in the embedding space.
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)

        # --- Global max-pool ---
        # output_size=1 collapses the entire temporal dimension to a
        # single value per channel, selecting the strongest activation
        # regardless of where it occurs in the embedding vector.
        self.pool = nn.AdaptiveMaxPool1d(1)

         # --- Classification head ---
        # Dropout(0.3) regularises the penultimate representation to
        # reduce overfitting on small protein datasets.
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN classifier.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch_size, input_dim)``.

        Returns
        -------
        torch.Tensor
            Raw logit tensor of shape ``(batch_size, 1)``.
            Apply ``torch.sigmoid`` to convert to a probability, or
            pass directly to ``nn.BCEWithLogitsLoss``.
        """
        # Reshape: (batch_size, input_dim) → (batch_size, 1, input_dim)
        # The embedding vector is treated as a 1-D signal with one channel.
        x = x.unsqueeze(1)

        # Conv block 1: (B, 1, D) → (B, 64, D)
        x = F.relu(self.bn1(self.conv1(x)))

        # Conv block 2: (B, 64, D) → (B, 128, D)
        x = F.relu(self.bn2(self.conv2(x)))

        # Global max-pool: (B, 128, D) → (B, 128, 1)
        x = self.pool(x)

        # Remove the pooled temporal dimension: (B, 128, 1) → (B, 128)
        x = x.squeeze(-1)

        # FC head: (B, 128) → (B, 1)
        return self.classifier(x)
    
