import re
import torch
import numpy as np

from transformers import (
    AutoTokenizer,
    EsmModel,
    BertModel,
    BertTokenizer,
    T5EncoderModel,
)

"""
Provides utilities for generating fixed-length numerical embeddings from
raw protein sequences using pre-trained transformer models.

Three model families are supported:
  - ProtBERT  (BERT-based,  model_type="bert")
  - ProtT5    (T5-based encoder, model_type="prott5")
  - ESM       (Meta's ESM family,  model_type="esm")

The embeddings are produced by mean-pooling the last hidden state over all
non-padding token positions, yielding a single vector per sequence regardless
of sequence length.
"""
# ---------- DEVICE SELECTION ----------

# Prefer CUDA (NVIDIA GPU) for maximum throughput.
# Fall back to MPS (Apple Silicon GPU) if available.
# Default to CPU if no accelerator is detected.

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def clean_protein_sequence(seq: str) -> str:
    """
    Sanitise a raw protein sequence for use with transformer models.

    Assumes ambiguous residues (B, Z, J, U, O) have already been normalised
    to 'X' upstream by normalize_ambiguous_aa.py. This function only strips
    whitespace/case and validates the sequence contains no unexpected
    characters — it does not perform ambiguous-residue normalisation itself,
    to avoid a second, divergent implementation of that logic.

    Raises
    ------
    ValueError
        If the sequence still contains raw ambiguous codes (B, Z, J, U, O),
        which signals that the input FASTA was not normalised upstream.
    """
    seq = seq.upper().strip()

    still_ambiguous = set('BZJUO') & set(seq)
    if still_ambiguous:
        raise ValueError(
            f"Sequence contains un-normalised ambiguous residues {still_ambiguous} — "
            f"expected input to already be normalised via normalize_ambiguous_aa.py"
        )

    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYX]", "", seq)  # drop anything not a valid IUPAC letter or X
    return seq


def format_for_protbert_or_prott5(seq: str) -> str:
    """
    Prepare a protein sequence for ProtBERT or ProtT5 tokenisation.

    Both models were trained on sequences where residues are separated by a
    single space (e.g. "M A G T K …"). This function first sanitises the
    sequence via :func:`clean_protein_sequence`, then inserts the required
    spaces.

    Parameters
    ----------
    seq : str
        Raw protein sequence string.

    Returns
    -------
    str
        Space-separated residue string ready for tokenisation
        (e.g. ``"M A G T K"``).
    """
    seq = clean_protein_sequence(seq)
    return " ".join(list(seq))

class ProteinEmbedder:
    """
    A unified interface for embedding protein sequences with transformer models.

    Wraps ProtBERT, ProtT5, and ESM model families behind a common API that
    handles tokenisation differences, device placement, and mean-pool
    aggregation automatically.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier or local path
        (e.g. ``"Rostlab/prot_bert"``).
    model_type : str
        Architecture family. Must be one of:
          - ``"bert"``   — BERT-based models (ProtBERT, ProtBERT-BFD, …)
          - ``"prott5"`` — T5 encoder-only models (ProtT5-XL-UniRef50, …)
          - ``"esm"``    — Meta ESM models (ESM-2, ESM-1b, …)

    Raises
    ------
    ValueError
        If ``model_type`` is not one of the three supported values.

    Attributes
    ----------
    model_name : str
        The Hugging Face identifier passed at construction time.
    model_type : str
        Lower-cased architecture family string.
    tokenizer : PreTrainedTokenizer
        Tokeniser loaded from the specified model hub path.
    model : PreTrainedModel
        Encoder model moved to the active :data:`DEVICE` and set to eval mode.
    """

    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type.lower()
        # Load the tokeniser and model that correspond to the chosen family.
        # Each family has different tokeniser classes and constructor options.
        if self.model_type == "bert":
             # ProtBERT uses a case-sensitive BertTokenizer;
             # do_lower_case=False preserves residue-letter casing.
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
            )
            self.model = BertModel.from_pretrained(model_name)

        elif self.model_type == "prott5":
            # ProtT5 requires the slow tokeniser (use_fast=False) and explicit
            # float32 to avoid precision mismatches on some hardware.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False,)
            self.model = T5EncoderModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )

        elif self.model_type == "esm":
            # ESM models are compatible with the generic AutoTokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = EsmModel.from_pretrained(model_name)

        else:
            raise ValueError(
                "model_type must be one of: 'bert', 'prott5', or 'esm'"
            )
        
        # Move model weights to the selected accelerator and disable
        # dropout / batch-norm training behaviour (eval mode).
        self.model.to(DEVICE)
        self.model.eval()
    
    def get_embedding_dim(self) -> int:
        """
        Return the dimensionality of the embeddings produced by this model.

        The value is read directly from the model's configuration so it stays
        correct regardless of which checkpoint is loaded.

        Returns
        -------
        int
            Size of the hidden (embedding) dimension
            (e.g. 1024 for ProtBERT, 1024 for ProtT5-XL, 480–2560 for ESM-2).
        """
        return self.model.config.hidden_size
    
    @torch.no_grad()
    def embed_sequence(self, seq: str, max_length: int = 1024) -> np.ndarray:
        """
        Compute a single fixed-length embedding for one protein sequence.

        The sequence is tokenised, fed through the encoder, and the resulting
        token-level hidden states are aggregated into a single vector via
        attention-mask-weighted mean pooling (padding tokens are excluded).

        Parameters
        ----------
        seq : str
            Raw protein sequence (any case; cleaned internally).
        max_length : int, optional
            Maximum token length passed to the tokeniser's truncation logic.
            Sequences longer than this will be silently truncated.
            Default is 1024.

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(hidden_size,)`` representing the
            mean-pooled embedding of the sequence.
        """
        # Apply the appropriate pre-processing for the chosen model family.
        if self.model_type in ["bert", "prott5"]:
            seq = format_for_protbert_or_prott5(seq)
        else:
            seq = clean_protein_sequence(seq)

        # Tokenise and move tensors to the active device.
        encoded = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,          # No padding for single sequences
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        # Forward pass — gradients disabled by the @torch.no_grad() decorator.
        outputs = self.model(**encoded)
        hidden  = outputs.last_hidden_state  # Shape: (1, seq_len, hidden_size)

        # Mean-pool over the sequence dimension, ignoring padding positions.
        # mask shape: (1, seq_len, 1) — broadcasts over the hidden dimension.
        mask   = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        # clamp(min=1e-9) guards against zero-length sequences causing NaN.

        # Remove the batch dimension and return a plain NumPy array.
        return pooled.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_sequences(self, sequences, max_length: int = 1024,) -> np.ndarray:
        """
        Compute embeddings for a collection of protein sequences.

        Sequences are embedded one at a time (no dynamic batching) and the
        results are stacked into a 2-D array. For large datasets consider
        batching upstream to improve GPU utilisation.

        Parameters
        ----------
        sequences : iterable of str
            Any iterable of raw protein sequence strings.
        max_length : int, optional
            Passed through to :meth:`embed_sequence`. Default is 1024.

        Returns
        -------
        np.ndarray
            2-D float32 array of shape ``(n_sequences, hidden_size)`` where
            each row is the mean-pooled embedding of the corresponding input
            sequence.
        """
        return np.vstack([self.embed_sequence(seq, max_length=max_length) for seq in sequences])