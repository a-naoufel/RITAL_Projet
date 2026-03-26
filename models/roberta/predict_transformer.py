"""
models/distilbert/predict_transformer.py
=========================================
Generates submission for Models 5a, 5b, or 6.
Reads best config from the corresponding scores.json.
Trains on full 2000 reviews, predicts testSentiment.txt.

Usage
-----
  python predict_transformer.py --model distilbert-base
  python predict_transformer.py --model distilbert-sst2
  python predict_transformer.py --model deberta

Output  →  submissions/preds_model5a_distilbert_base.txt   (etc.)
"""

import argparse, json, random, warnings
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

SEED    = 42
MAX_LEN = 256

MODEL_CONFIGS = {
    "distilbert-base": {
        "hf_name":    "distilbert-base-uncased",
        "scores_dir": "model5_distilbert_base",
        "out_file":   "preds_model5a_distilbert_base.txt",
        "batch_size": 16,
    },
    "distilbert-sst2": {
        "hf_name":    "distilbert-base-uncased-finetuned-sst-2-english",
        "scores_dir": "model5_distilbert_sst2",
        "out_file":   "preds_model5b_distilbert_sst2.txt",
        "batch_size": 16,
    },
    "deberta": {
        "hf_name":    "microsoft/deberta-v3-base",
        "scores_dir": "model6_deberta",
        "out_file":   "preds_model6_deberta.txt",
        "batch_size": 8,
    },
    "roberta": {
        "hf_name":    "textattack/roberta-base-SST-2",
        "scores_dir": "appendix_roberta",
        "out_file":   "preds_appendix_roberta.txt",
        "batch_size": 16,
    },
}

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"
TEST_FILE    = PROJECT_ROOT / "dataset" / "raw" / "testSentiment.txt"
OUT_DIR      = PROJECT_ROOT / "submissions"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def load_train(data_dir):
    pos = sorted((data_dir / "pos").glob("*.txt"))
    neg = sorted((data_dir / "neg").glob("*.txt"))
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Training data : {len(pos)} pos + {len(neg)} neg")
    return texts, labels


def load_test(test_file):
    text  = test_file.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    print(f"Test file     : {test_file.name}  ({len(lines)} lines)")
    return lines


class TextDS(Dataset):
    def __init__(self, texts, labels, tok):
        self.texts, self.labels, self.tok = texts, labels, tok
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc  = self.tok(self.texts[i], truncation=True,
                        max_length=MAX_LEN, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item


def make_collator(pad_id):
    def collate(batch):
        ml = max(x["input_ids"].shape[0] for x in batch)
        def pad(v, fill):
            out = torch.full((ml,), fill, dtype=v.dtype)
            out[:v.shape[0]] = v; return out
        r = {
            "input_ids":      torch.stack([pad(x["input_ids"],      pad_id) for x in batch]),
            "attention_mask": torch.stack([pad(x["attention_mask"], 0)      for x in batch]),
            "labels":         torch.stack([x["labels"] for x in batch]),
        }
        if "token_type_ids" in batch[0]:
            r["token_type_ids"] = torch.stack([pad(x["token_type_ids"], 0) for x in batch])
        return r
    return collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()),
                        required=True)
    args   = parser.parse_args()
    cfg    = MODEL_CONFIGS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Load best hyperparameters
    scores_path = PROJECT_ROOT / "results" / cfg["scores_dir"] / "scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(
            f"scores.json not found at {scores_path}\n"
            f"Run train_transformer_benchmark.py --model {args.model} first."
        )
    s = json.loads(scores_path.read_text())
    lr     = s["best_lr"]
    epochs = s["best_epochs"]
    print(f"Config  : lr={lr:.0e}, epochs={epochs}")
    print(f"CV acc  : {s['best_mean']:.4f} +/- {s['best_std']:.4f}")

    set_seed(SEED)
    X_train, y_train = load_train(DATA_DIR)
    test_lines       = load_test(TEST_FILE)

    tok   = AutoTokenizer.from_pretrained(cfg["hf_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["hf_name"], num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    collate  = make_collator(tok.pad_token_id or 0)
    train_dl = DataLoader(TextDS(X_train, y_train, tok),
                          batch_size=cfg["batch_size"],
                          shuffle=True, collate_fn=collate)

    total_steps  = epochs * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    print(f"\nFitting {cfg['hf_name']} on full {len(X_train)} reviews...")
    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            pbar.set_postfix(loss=f"{float(loss):.3f}")

    # Predict test set in batches
    print("\nPredicting test set...")
    model.eval()
    all_preds = []
    bs = cfg["batch_size"] * 2
    with torch.no_grad():
        for i in tqdm(range(0, len(test_lines), bs), desc="Predict"):
            batch_texts = test_lines[i:i+bs]
            enc = tok(batch_texts, truncation=True, padding=True,
                      max_length=MAX_LEN, return_tensors="pt").to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(**enc).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())

    out_file = OUT_DIR / cfg["out_file"]
    with out_file.open("w", encoding="utf-8") as f:
        for p in all_preds:
            f.write(("P" if p == 1 else "N") + "\n")

    n_pos = sum(p == 1 for p in all_preds)
    n_neg = sum(p == 0 for p in all_preds)
    print(f"\nWritten      : {out_file.resolve()}")
    print(f"Total        : {len(all_preds)}")
    print(f"Positive (P) : {n_pos}  ({100*n_pos/len(all_preds):.1f}%)")
    print(f"Negative (N) : {n_neg}  ({100*n_neg/len(all_preds):.1f}%)")
    written = sum(1 for _ in out_file.open())
    assert written == len(test_lines), \
        f"Line count mismatch: wrote {written}, expected {len(test_lines)}"
    print(f"Line count   : OK ({written} lines)")


if __name__ == "__main__":
    main()
