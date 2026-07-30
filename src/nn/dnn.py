import torch
import torch.nn as nn

"""
Fully-connected Deep Neural Network for binary protein sequence classification.

The model maps a pre-computed protein embedding vector directly to a scalar
logit through four linear layers with progressively decreasing width. Batch
Normalisation and Dropout are applied after each hidden layer to stabilise
training and limit overfitting on small biological datasets

Architecture summary
--------------------
Input  : (batch_size, input_dim)   — flat embedding vector
Layer 1: Linear(input_dim → 512)  → BatchNorm1d → ReLU → Dropout(0.4)
Layer 2: Linear(512 → 256)        → BatchNorm1d → ReLU → Dropout(0.3)
Layer 3: Linear(256 → 64)         →               ReLU → Dropout(0.2)
Layer 4: Linear(64 → 1)           — raw logit

The output is a single unbounded logit per sample. Apply
``torch.sigmoid`` for a probability or pass directly to
``nn.BCEWithLogitsLoss`` during training (numerically more stable
than BCE + sigmoid).
"""

class DNNClassifier(nn.Module):
    """
    Binary classifier built on a four-layer fully-connected network.

    The network uses a funnel architecture (input_dim → 512 → 256 → 64 → 1)
    that progressively compresses the embedding into a scalar decision value.
    Batch Normalisation is applied to the first two hidden layers to reduce
    internal covariate shift and allow higher learning rates. Dropout rates
    decrease with depth (0.4 → 0.3 → 0.2), reflecting the smaller
    representation at each stage and reducing the risk of over-regularisation
    in the narrower layers.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input embedding vector
        (e.g. 1024 for ProtBERT, 480–2560 for ESM-2, 1024 for ProtT5).

    Attributes
    ----------
    model : nn.Sequential
        The full feed-forward stack from input to output logit.
    """
     
    def __init__(self, input_dim):
        super(DNNClassifier, self).__init__()

        self.model = nn.Sequential(
                # --- Hidden layer 1 ---
                # Large first layer captures broad patterns in the high-dimensional
                # embedding space. BatchNorm stabilises activations before ReLU.
                # Dropout(0.4) is highest here where overfitting risk is greatest.
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.4),

                # --- Hidden layer 2 ---
                # Halves the representation width; BatchNorm continues to
                # normalise. Dropout is reduced to 0.3 as the layer is narrower.
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),

                # --- Hidden layer 3 ---
                # Further compression to 64 units. BatchNorm is omitted here;
                # the small width makes it less beneficial. Dropout reduced to 0.2.
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Dropout(0.2),

                # --- Output layer ---
                # Projects to a single unbounded logit; no activation applied.
                # Use nn.BCEWithLogitsLoss during training or torch.sigmoid
                # at inference to obtain a class probability.
                nn.Linear(64, 1)
            )

    def forward(self, x):
            """
            Forward pass through the DNN classifier.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape ``(batch_size, input_dim)``.

            Returns
            -------
            torch.Tensor
                Raw logit tensor of shape ``(batch_size, 1)``.
                Apply ``torch.sigmoid`` to convert to a class probability,
                or pass directly to ``nn.BCEWithLogitsLoss``.
            """
            return self.model(x)