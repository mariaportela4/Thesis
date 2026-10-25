"""
deposcope_retrain.py
====================
Retrains DepoScope's  CNN classifier (Component 2) on our own dataset,
keeping the fine-tuned ESM-2 (Component 1) frozen as a feature extractor.

Source: https://github.com/dimiboeckaerts/DepoScope
Paper:  Concha-Eloko R, Stock M, De Baets B, Briers Y, Sanjuán R, Domingo-Calap P, Boeckaerts D. 
        DepoScope: Accurate phage depolymerase annotation and domain delineation using large language models. 
        PLoS Comput Biol. 2024 Aug 5;20(8):e1011831. doi: 10.1371/journal.pcbi.1011831.


Architecture (from DepoScope source):
  - ESM-2 (esm2_t12_35M, fine-tuned for token classification) → frozen
  - Dpo_classifier CNN head → trained on our data

Pipeline:
  1. Download DepoScope weights from Zenodo (if not present)
  2. Load train/test splits from final_dataset_40.csv
  3. Pre-compute ESM-2 token features for all sequences (cached to disk)
  4. Train CNN head on train split (with validation monitoring)
  5. Evaluate on test split → F1, Precision, Recall, AUC, Accuracy
  6. Save model + metrics to results/deposcope_retrain/

Usage:
    python src/deposcope_retrain.py

"""

import os
import subprocess
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, accuracy_score, classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForTokenClassification, AutoTokenizer
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

BASE_DIR        = Path("/home/mcarvalho/Thesis")
DATA_CSV        = BASE_DIR / "data" / "final_dataset_40.csv"
OUT_DIR         = BASE_DIR / "results" / "deposcope_retrain"
WEIGHTS_DIR     = BASE_DIR / "deposcope_weights"

ESM2_ZENODO_URL = "https://zenodo.org/records/10957073/files/esm2_t12_finetuned_depolymerases.zip"
DPO_ZENODO_URL  = "https://zenodo.org/records/10957073/files/Deposcope.esm2_t12_35M_UR50D.2203.full.model"

ESM2_CHECKPOINT = WEIGHTS_DIR / "esm2_t12_35M_UR50D__fulltrain__finetuneddepolymerase.2103.4_labels/checkpoint-2255"
DPO_MODEL_PATH  = WEIGHTS_DIR / "Deposcope.esm2_t12_35M_UR50D.2203.full.model"

FEATURE_CACHE   = OUT_DIR / "esm2_token_features.npz"   # cached pre-computed features

# Training hyperparameters (same spirit as DepoScope originals)
MAX_LENGTH   = 1024
BATCH_SIZE   = 16
NUM_EPOCHS   = 30
LR           = 1e-4
PATIENCE     = 5          # early stopping patience
THRESHOLD    = 0.5        # classification threshold

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# STEP 1 — DOWNLOAD WEIGHTS
# ─────────────────────────────────────────────────────────────

def download_weights():
    """Download DepoScope weights from Zenodo if not already present."""
    zip_path = WEIGHTS_DIR / "esm2_t12_finetuned_depolymerases.zip"

    # ESM-2 fine-tuned checkpoint
    if not ESM2_CHECKPOINT.exists():
        print("Downloading fine-tuned ESM-2 weights...")
        subprocess.run(["wget", "-q", "--show-progress",
                        ESM2_ZENODO_URL, "-O", str(zip_path)], check=True)
        print("Unzipping ESM-2 weights...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(WEIGHTS_DIR)], check=True)
        # remove __MACOSX artifact if present
        macosx = WEIGHTS_DIR / "__MACOSX"
        if macosx.exists():
            subprocess.run(["rm", "-rf", str(macosx)])
        zip_path.unlink()
        print(f"ESM-2 checkpoint ready at: {ESM2_CHECKPOINT}")
    else:
        print(f"ESM-2 checkpoint already exists: {ESM2_CHECKPOINT}")

    # DepoScope .model file
    if not DPO_MODEL_PATH.exists():
        print("Downloading DepoScope .model file...")
        subprocess.run(["wget", "-q", "--show-progress",
                        DPO_ZENODO_URL, "-O", str(DPO_MODEL_PATH)], check=True)
        print(f"DepoScope model ready at: {DPO_MODEL_PATH}")
    else:
        print(f"DepoScope .model already exists: {DPO_MODEL_PATH}")


# ─────────────────────────────────────────────────────────────
# STEP 2 — DEPOSCOPE ARCHITECTURE (from their source)
# ─────────────────────────────────────────────────────────────

class Dpo_classifier(nn.Module):
    """
    DepoScope's CNN classification head (Component 2).
    Taken directly from deposcope-predict.py — architecture unchanged.
    Only this component is trained; ESM-2 is frozen upstream.
    """
    def __init__(self):
        super().__init__()
        self.max_length = MAX_LENGTH
        self.conv1      = nn.Conv1d(1, 64, kernel_size=5, stride=1)
        self.conv2      = nn.Conv1d(64, 128, kernel_size=5, stride=1)
        # output length after 2 conv layers (no padding): max_length - 2*(5-1) = 1016
        self.fc1        = nn.Linear(128 * (self.max_length - 2 * (5 - 1)), 32)
        self.classifier = nn.Linear(32, 1)

    def forward(self, token_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_features: (batch, 1, max_length) float tensor of ESM-2 token IDs
        Returns:
            logits: (batch, 1)
        """
        out = F.relu(self.conv1(token_features))
        out = F.relu(self.conv2(out))
        out = out.view(token_features.size(0), -1)
        out = F.relu(self.fc1(out))
        return self.classifier(out)


# ─────────────────────────────────────────────────────────────
# STEP 3 — PRE-COMPUTE ESM-2 TOKEN FEATURES (frozen, cached)
# ─────────────────────────────────────────────────────────────

def pad_or_truncate(tokens: torch.Tensor, max_length: int) -> torch.Tensor:
    """Pad or truncate a (1, L) token tensor to (1, max_length)."""
    if tokens.size(1) < max_length:
        tokens = F.pad(tokens, (0, max_length - tokens.size(1)))
    elif tokens.size(1) > max_length:
        tokens = tokens[:, :max_length]
    return tokens


def compute_esm2_features(sequences: list[str],
                           tokenizer, esm2_model) -> np.ndarray:
    """
    Run each sequence through the frozen fine-tuned ESM-2 to get
    token-level label IDs (the intermediate representation DepoScope uses).

    Returns:
        features: np.ndarray of shape (N, max_length), dtype float32
    """
    esm2_model.eval()
    features = []

    for seq in tqdm(sequences, desc="Computing ESM-2 token features"):
        input_ids = tokenizer.encode(seq, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = esm2_model(input_ids)
            probs   = torch.nn.functional.softmax(outputs.logits, dim=-1)
            _, token_ids = torch.max(probs, dim=-1)   # (1, L)
        token_ids = pad_or_truncate(token_ids, MAX_LENGTH)  # (1, max_length)
        features.append(token_ids.cpu().numpy().astype(np.float32).squeeze(0))

    return np.stack(features, axis=0)   # (N, max_length)


def get_or_compute_features(df: pd.DataFrame,
                             tokenizer, esm2_model) -> dict[str, np.ndarray]:
    """
    Load pre-computed features from cache, or compute and cache them.
    Returns dict with keys 'train', 'test' mapping to feature arrays.
    """
    train_df = df[df["split"] == "train"]
    test_df  = df[df["split"] == "test"]
    if FEATURE_CACHE.exists():
        print(f"Loading cached ESM-2 features from {FEATURE_CACHE}")
        data = np.load(FEATURE_CACHE)
        cached_train, cached_test = data["train"], data["test"]

        # Guard against stale caches: if final_dataset_40.csv was regenerated,
        # the split sizes stay the same (1515/176) while the sequences behind
        # them change, so comparing lengths detects nothing. Compare identities.
        cached_train_ids = [str(x) for x in data["train_ids"]] if "train_ids" in data else None
        cached_test_ids = cached_test_ids = [str(x) for x in data["test_ids"]] if "test_ids" in data else None


        if (cached_train_ids != [str(x) for x in train_df["seq_id"]]
                or cached_test_ids != [str(x) for x in test_df["seq_id"]]):
            raise RuntimeError(
                f"Stale feature cache detected at {FEATURE_CACHE}: the cached "
                f"sequence IDs do not match the current {DATA_CSV.name}. The "
                f"features were computed for a different set of sequences, so "
                f"using them would pair each sequence's features with another "
                f"sequence's label. Delete the cache file and re-run."
            )
        return {"train": cached_train, "test": cached_test}
    
    print("Pre-computing ESM-2 token features (this runs once and is cached)...")

    train_feats = compute_esm2_features(train_df["sequence"].tolist(), tokenizer, esm2_model)
    test_feats  = compute_esm2_features(test_df["sequence"].tolist(),  tokenizer, esm2_model)

    np.savez(
        FEATURE_CACHE,
        train=train_feats,
        test=test_feats,
        train_ids=train_df["seq_id"].astype(str).to_numpy(dtype="U"),
        test_ids=test_df["seq_id"].astype(str).to_numpy(dtype="U"),
    )
    print(f"Features cached to {FEATURE_CACHE}")
    return {"train": train_feats, "test": test_feats}


# ─────────────────────────────────────────────────────────────
# STEP 4 — DATASET & TRAINING
# ─────────────────────────────────────────────────────────────

class TokenFeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32).unsqueeze(1)  # (N,1,L)
        self.labels   = torch.tensor(labels,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits = model(feats).squeeze(1)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            logits = model(feats).squeeze(1)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.array(all_probs), np.array(all_labels)


def compute_metrics(probs: np.ndarray, labels: np.ndarray,
                    threshold: float = THRESHOLD) -> dict:
    preds = (probs >= threshold).astype(int)
    return {
        "F1":        f1_score(labels, preds,        zero_division=0),
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall":    recall_score(labels, preds,    zero_division=0),
        "AUC":       roc_auc_score(labels, probs),
        "Accuracy":  accuracy_score(labels, preds),
    }


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # ── 1. Download weights ──────────────────────────────────
    download_weights()

    # ── 2. Load data ─────────────────────────────────────────
    print(f"\nLoading dataset from {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    print(f"  Total sequences: {len(df)}")
    print(f"  Train: {(df['split']=='train').sum()}, Test: {(df['split']=='test').sum()}")

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # ── 3. Load frozen ESM-2 ─────────────────────────────────
    print(f"\nLoading frozen ESM-2 from {ESM2_CHECKPOINT}")
    tokenizer   = AutoTokenizer.from_pretrained(str(ESM2_CHECKPOINT),
                                                local_files_only=True,
                                                clean_up_tokenization_spaces=False)
    esm2_model  = AutoModelForTokenClassification.from_pretrained(str(ESM2_CHECKPOINT),
                                                                   local_files_only=True)
    esm2_model  = esm2_model.to(DEVICE)
    esm2_model.eval()
    for param in esm2_model.parameters():   # freeze completely
        param.requires_grad = False
    print("  ESM-2 frozen — no gradients will flow through it.")

    # ── 4. Pre-compute / load token features ─────────────────
    feats = get_or_compute_features(df, tokenizer, esm2_model)
    train_feats  = feats["train"]
    test_feats   = feats["test"]
    train_labels = (train_df["label"] == "positive").astype(np.float32).values
    test_labels  = (test_df["label"] == "positive").astype(np.float32).values

    # ── 5. DataLoaders ───────────────────────────────────────
    assert len(train_feats) == len(train_labels), (
        f"train_feats ({len(train_feats)}) and train_labels ({len(train_labels)}) "
        f"length mismatch — feature cache is out of sync with the current split."
    )
    assert len(test_feats) == len(test_labels), (
        f"test_feats ({len(test_feats)}) and test_labels ({len(test_labels)}) "
        f"length mismatch — feature cache is out of sync with the current split."
    )

    # Early stopping uses the dataset's own validation partition, the same 88
    # sequences used by the six thesis models, so that every iteratively
    # trained model selects its checkpoint on identical data. The previous
    # version held out the last 10% of the training rows positionally; because
    # the table is ordered by class, that slice contained no positive sequence.
    valid_df   = df[df["split"] == "valid"].reset_index(drop=True)
    val_feats  = compute_esm2_features(valid_df["sequence"].tolist(),
                                       tokenizer, esm2_model)
    val_labels = (valid_df["label"] == "positive").astype(np.float32).values

    tr_feats, tr_labels = train_feats, train_labels

    assert 0 < val_labels.sum() < len(val_labels), \
        "validation partition is single-class -- early stopping would be meaningless"

    train_loader = DataLoader(TokenFeatureDataset(tr_feats, tr_labels),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TokenFeatureDataset(val_feats, val_labels),
                              batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TokenFeatureDataset(test_feats, test_labels),
                              batch_size=BATCH_SIZE, shuffle=False)

    # ── 6. Build CNN head ────────────────────────────────────
    model     = Dpo_classifier().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters (CNN head only): {total_params:,}")

    # ── 7. Training loop with early stopping ─────────────────
    print(f"\nTraining CNN head for up to {NUM_EPOCHS} epochs (patience={PATIENCE})...")
    best_val_loss  = float("inf")
    patience_count = 0
    best_model_path = OUT_DIR / "deposcope_retrained_cnn.pt"
    history = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss              = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_probs, val_lbls = evaluate(model, val_loader, criterion)
        val_metrics             = compute_metrics(val_probs, val_lbls)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()}
        })

        print(f"  Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_F1={val_metrics['F1']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✓ Best model saved (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # ── 8. Evaluate best model on test set ───────────────────
    print(f"\nLoading best checkpoint and evaluating on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
    _, test_probs, test_lbls = evaluate(model, test_loader, criterion)
    test_metrics = compute_metrics(test_probs, test_lbls)

    print("\n" + "="*50)
    print("TEST SET RESULTS — DepoScope (retrained CNN head)")
    print("="*50)
    for k, v in test_metrics.items():
        print(f"  {k:12s}: {v:.4f}")
    print("="*50)

    print("\nClassification Report:")
    preds = (test_probs >= THRESHOLD).astype(int)
    print(classification_report(test_lbls, preds,
                                target_names=["Non-DPO", "DPO"]))

    # ── 9. Save outputs ───────────────────────────────────────
    metrics_path = OUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    history_path = OUT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    probs_path = OUT_DIR / "test_probabilities.npy"
    np.save(probs_path, test_probs)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print(f"  Model:        {best_model_path.name}")
    print(f"  Metrics:      {metrics_path.name}")
    print(f"  History:      {history_path.name}")
    print(f"  Probs (.npy): {probs_path.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()