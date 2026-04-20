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

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def clean_protein_sequence(seq: str) -> str:
    seq = seq.upper().strip()
    seq = re.sub(r"[^ACDEFGHIKLMNPQRSTVWYUZOBX]", "", seq)
    seq = re.sub(r"[UZOB]", "X", seq)
    return seq


def format_for_protbert_or_prott5(seq: str) -> str:
    seq = clean_protein_sequence(seq)
    return " ".join(list(seq))


class ProteinEmbedder:
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type.lower()

        if self.model_type == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
            )
            self.model = BertModel.from_pretrained(model_name)

        elif self.model_type == "prott5":
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                do_lower_case=False,
                use_fast=False,
            )
            self.model = T5EncoderModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
            )

        elif self.model_type == "esm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = EsmModel.from_pretrained(model_name)

        else:
            raise ValueError(
                "model_type must be one of: 'bert', 'prott5', or 'esm'"
            )

        self.model.to(DEVICE)
        self.model.eval()

    @torch.no_grad()
    def embed_sequence(self, seq: str, max_length: int = 1024) -> np.ndarray:
        if self.model_type in ["protbert", "prott5"]:
            seq = format_for_protbert_or_prott5(seq)
        else:
            seq = clean_protein_sequence(seq)

        encoded = self.tokenizer(
            seq,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
        outputs = self.model(**encoded)

        hidden = outputs.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()

        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def embed_sequences(self, sequences, max_length: int = 1024):
        return np.vstack([self.embed_sequence(seq, max_length=max_length) for seq in sequences])