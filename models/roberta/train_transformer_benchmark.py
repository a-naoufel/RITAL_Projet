"""
models/distilbert/train_transformer_benchmark.py
=================================================
Models 5 & 6 — fine-tuned transformers
Covers: distilbert-base-uncased | distilbert-base-uncased-finetuned-sst-2-english | deberta-v3-base

Fixes vs existing train_distilbert.py
--------------------------------------
- Static padding (max_len=512 every batch)  →  dynamic padding per batch (collate_fn)
- No AMP                                    →  torch.cuda.amp autocast + GradScaler
- Single split                              →  3 seeds × 80/20 split → mean ± std
- Hardcoded model                           →  --model argument, all three share loop

Design
------
5-fold CV is impractical for transformers on CPU/single GPU (15+ runs per grid point).
Instead: 3 independent seeds on 80/20 split. Mean ± std is reported; this matches
the notation used for classical models while being realistic about compute budget.

Grid (per model)
----------------
lr      : model-specific (see MODEL_CONFIGS)
epochs  : [2, 3, 5]
batch   : fixed per model (see MODEL_CONFIGS)

Usage
-----
  # Run all three sequentially:
  python train_transformer_benchmark.py --model all

  # Run one at a time:
  python train_transformer_benchmark.py --model distilbert-base
  python train_transformer_benchmark.py --model distilbert-sst2
  python train_transformer_benchmark.py --model deberta

Outputs  →  results/model5_distilbert_base/      (or model5_distilbert_sst2 / model6_deberta)
  scores.json
  lr_curve.png
  confusion_matrix.png
  report_snippet.txt
"""

import argparse, json, time, random, warnings
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
SEED      = 42
N_SEEDS   = 3          # repeated runs for mean ± std
MAX_LEN   = 256        # truncate at 256 — covers >95% of movie reviews
PATIENCE  = 2          # early stopping patience (epochs)

MODEL_CONFIGS = {
    "distilbert-base": {
        "hf_name":   "distilbert-base-uncased",
        "out_dir":   "model5_distilbert_base",
        "lr_grid":   [1e-5, 2e-5, 3e-5, 5e-5],
        "epoch_grid":[2, 3, 5],
        "batch_size": 16,
        "report_id": "Model 5a — DistilBERT base uncased",
    },
    "distilbert-sst2": {
        "hf_name":   "distilbert-base-uncased-finetuned-sst-2-english",
        "out_dir":   "model5_distilbert_sst2",
        "lr_grid":   [5e-6, 1e-5, 2e-5, 3e-5],
        "epoch_grid":[2, 3, 5],
        "batch_size": 16,
        "report_id": "Model 5b — DistilBERT SST-2 (already sentiment-tuned)",
    },
    "deberta": {
        "hf_name":   "microsoft/deberta-v3-base",
        "out_dir":   "model6_deberta",
        "lr_grid":   [5e-6, 1e-5, 2e-5],
        "epoch_grid":[2, 3, 5],
        "batch_size": 8,    # DeBERTa is larger — smaller batch to fit VRAM
        "report_id": "Model 6 — DeBERTa-v3-base",
    },
    "roberta": {
        "hf_name":   "textattack/roberta-base-SST-2",
        "out_dir":   "appendix_roberta",
        "lr_grid":   [5e-6, 1e-5, 2e-5, 3e-5],
        "epoch_grid":[2, 3, 5],
        "batch_size": 16,
        "report_id": "Appendix — RoBERTa-base SST-2",
    },
}

HERE         = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
DATA_DIR     = PROJECT_ROOT / "dataset" / "raw" / "movies1000"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200,
    "font.family": "serif",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--",
})

# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── Data ───────────────────────────────────────────────────────────────────────
def load_docs(data_dir: Path):
    pos = sorted((data_dir / "pos").glob("*.txt"))
    neg = sorted((data_dir / "neg").glob("*.txt"))
    if not pos:
        raise FileNotFoundError(f"No reviews in {data_dir}/pos/")
    texts  = [p.read_text(encoding="utf-8", errors="ignore") for p in pos]
    texts += [p.read_text(encoding="utf-8", errors="ignore") for p in neg]
    labels = [1] * len(pos) + [0] * len(neg)
    print(f"Loaded {len(pos)} pos + {len(neg)} neg = {len(texts)} reviews")
    return texts, labels

# ── Dataset ────────────────────────────────────────────────────────────────────
class TextDS(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN):
        self.texts, self.labels = texts, labels
        self.tok, self.max_len  = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc  = self.tok(
            self.texts[i],
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

# ── Dynamic padding collator ───────────────────────────────────────────────────
def make_collator(pad_id):
    def collate(batch):
        max_len = max(x["input_ids"].shape[0] for x in batch)
        def pad(v, fill):
            out = torch.full((max_len,), fill, dtype=v.dtype)
            out[:v.shape[0]] = v
            return out
        input_ids = torch.stack([pad(x["input_ids"],      pad_id) for x in batch])
        attn_mask = torch.stack([pad(x["attention_mask"], 0)      for x in batch])
        labels    = torch.stack([x["labels"] for x in batch])
        result    = {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}
        # token_type_ids only present in some models (BERT, DeBERTa)
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.stack(
                [pad(x["token_type_ids"], 0) for x in batch]
            )
        return result
    return collate

# ── Training loop (one run, one lr, one epoch count) ──────────────────────────
def train_one_run(X, y, hf_name, lr, epochs, batch_size, seed, device):
    set_seed(seed)
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    tok   = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    collate = make_collator(tok.pad_token_id or 0)
    train_dl = DataLoader(TextDS(Xtr, ytr, tok), batch_size=batch_size,
                          shuffle=True,  collate_fn=collate)
    val_dl   = DataLoader(TextDS(Xva, yva, tok), batch_size=batch_size * 2,
                          shuffle=False, collate_fn=collate)

    total_steps  = epochs * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))
    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_acc, bad = 0.0, 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_dl, desc=f"  ep{ep}/{epochs} lr={lr:.0e} seed={seed}",
                    leave=False)
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            pbar.set_postfix(loss=f"{float(loss):.3f}")

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch  = {k: v.to(device) for k, v in batch.items()}
                preds  = model(**batch).logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total   += batch["labels"].size(0)
        val_acc = correct / total
        print(f"  ep{ep} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"  Early stop at epoch {ep}")
                break

    del model
    torch.cuda.empty_cache()
    return best_acc

# ── Grid search: lr × epochs, averaged over N_SEEDS ───────────────────────────
def run_grid(X, y, cfg, device):
    hf_name    = cfg["hf_name"]
    lr_grid    = cfg["lr_grid"]
    epoch_grid = cfg["epoch_grid"]
    batch_size = cfg["batch_size"]
    total      = len(lr_grid) * len(epoch_grid)

    print(f"\nGrid search: {total} configs × {N_SEEDS} seeds = {total*N_SEEDS} runs")
    print(f"{'lr':<10} {'ep':<5} {'mean':>8} {'std':>7}")
    print("-" * 36)

    results = []
    best    = {"mean": -1}

    for lr in lr_grid:
        for ep in epoch_grid:
            seed_accs = []
            for s in range(N_SEEDS):
                acc = train_one_run(X, y, hf_name, lr, ep, batch_size,
                                    seed=SEED + s, device=device)
                seed_accs.append(acc)

            mean_acc = float(np.mean(seed_accs))
            std_acc  = float(np.std(seed_accs))
            marker   = " <-" if mean_acc > best["mean"] else ""
            print(f"{lr:<10.0e} {ep:<5} {mean_acc:>8.4f} {std_acc:>7.4f}{marker}")

            entry = {"lr": lr, "epochs": ep, "mean": mean_acc, "std": std_acc,
                     "seed_accs": seed_accs}
            results.append(entry)
            if mean_acc > best["mean"]:
                best = entry.copy()

    return results, best

# ── Figures ────────────────────────────────────────────────────────────────────
def plot_lr_curve(results, best, cfg, out_dir):
    # One curve per epoch count, x-axis = lr
    epoch_vals = sorted(set(r["epochs"] for r in results))
    colors     = ["#1a5f9e", "#c0392b", "#27ae60"]

    fig, ax = plt.subplots(figsize=(7, 4))
    for col, ep in zip(colors, epoch_vals):
        sub    = [r for r in results if r["epochs"] == ep]
        lrs    = [r["lr"]   for r in sub]
        means  = [r["mean"] for r in sub]
        stds   = [r["std"]  for r in sub]
        ax.semilogx(lrs, means, "o-", color=col, lw=2, ms=5,
                    label=f"{ep} epochs")
        ax.fill_between(lrs,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.10, color=col)

    ax.axvline(best["lr"], color="#888", lw=1.2, ls="--",
               label=f"Best lr={best['lr']:.0e}, ep={best['epochs']}")
    ax.set_xlabel("Learning rate (log scale)", fontsize=11)
    ax.set_ylabel(f"Mean accuracy ({N_SEEDS} seeds)", fontsize=11)
    ax.set_title(f"{cfg['report_id']} — lr tuning", fontsize=11, pad=10)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = out_dir / "lr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_cm(cm, cfg, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels = ["Negative", "Positive"]
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Confusion matrix — {cfg['report_id']}", fontsize=10, pad=10)
    ax.grid(False)
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path = out_dir / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

# ── Final hold-out run at best config to get confusion matrix ─────────────────
def final_holdout_cm(X, y, cfg, best, device):
    """One run at best config + SEED=42 to get predictions for confusion matrix."""
    set_seed(SEED)
    hf_name = cfg["hf_name"]
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    tok   = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=2, ignore_mismatched_sizes=True
    ).to(device)

    collate  = make_collator(tok.pad_token_id or 0)
    train_dl = DataLoader(TextDS(Xtr, ytr, tok), batch_size=cfg["batch_size"],
                          shuffle=True, collate_fn=collate)
    val_dl   = DataLoader(TextDS(Xva, yva, tok), batch_size=cfg["batch_size"] * 2,
                          shuffle=False, collate_fn=collate)

    total_steps  = best["epochs"] * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))
    optimizer    = AdamW(model.parameters(), lr=best["lr"], weight_decay=0.01)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for ep in range(1, best["epochs"] + 1):
        model.train()
        for batch in tqdm(train_dl, desc=f"  Final run ep{ep}", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            preds = model(**batch).logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(batch["labels"].cpu().tolist())

    del model; torch.cuda.empty_cache()
    cm     = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds,
                                   target_names=["Negative", "Positive"])
    print(f"\nFinal hold-out accuracy: {accuracy_score(all_true, all_preds):.4f}")
    print(report)
    return cm, report

# ── Per-model runner ───────────────────────────────────────────────────────────
def run_model(model_key, X, y, device):
    cfg     = MODEL_CONFIGS[model_key]
    out_dir = PROJECT_ROOT / "results" / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    t0      = time.time()

    print(f"\n{'='*60}")
    print(f"  {cfg['report_id']}")
    print(f"  HF model : {cfg['hf_name']}")
    print(f"  Device   : {device}")
    print(f"{'='*60}")

    grid_results, best = run_grid(X, y, cfg, device)

    print(f"\nBest config : lr={best['lr']:.0e}, epochs={best['epochs']}")
    print(f"Mean acc    : {best['mean']:.4f} +/- {best['std']:.4f}")

    cm, clf_report = final_holdout_cm(X, y, cfg, best, device)

    plot_lr_curve(grid_results, best, cfg, out_dir)
    plot_cm(cm, cfg, out_dir)

    out = {
        "model":        cfg["report_id"],
        "hf_name":      cfg["hf_name"],
        "n_seeds":      N_SEEDS,
        "seed_base":    SEED,
        "grid":         grid_results,
        "best_lr":      best["lr"],
        "best_epochs":  best["epochs"],
        "best_mean":    best["mean"],
        "best_std":     best["std"],
        "best_seed_accs": best["seed_accs"],
        "holdout_cm":   cm.tolist(),
        "elapsed_s":    round(time.time() - t0, 1),
    }
    (out_dir / "scores.json").write_text(json.dumps(out, indent=2))

    snippet = (
        f"=== {cfg['report_id']} ===\n"
        f"lr={best['lr']:.0e}  epochs={best['epochs']}\n\n"
        f"Mean acc ({N_SEEDS} seeds) : {best['mean']:.4f} +/- {best['std']:.4f}\n"
        f"Per-seed              : {[round(a,4) for a in best['seed_accs']]}\n\n"
        f"Confusion matrix (hold-out seed=42):\n"
        f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
        f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"
        f"{clf_report}\n"
        f"Time : {round(time.time()-t0,1)}s\n"
    )
    (out_dir / "report_snippet.txt").write_text(snippet)
    print(snippet)
    print(f"Outputs -> {out_dir}")
    return out

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()) + ["all"],
        default="all",
        help="Which model(s) to run. 'all' runs sequentially.",
    )
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    X, y = load_docs(DATA_DIR)

    keys = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    for key in keys:
        run_model(key, X, y, device)
        print(f"\nFinished {key}. Moving to next model...\n")


if __name__ == "__main__":
    main()
