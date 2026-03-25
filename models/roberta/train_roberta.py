from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_docs(movies_dir: Path):
    pos = sorted((movies_dir / "pos").glob("*.txt"))
    neg = sorted((movies_dir / "neg").glob("*.txt"))
    X = [read_text(p) for p in pos] + [read_text(n) for n in neg]
    y = [1]*len(pos) + [0]*len(neg)
    return X, y

class TextDS(Dataset):
    def __init__(self, texts, labels, tok, max_len=256):
        self.texts, self.labels, self.tok, self.max_len = texts, labels, tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

def collate_dynamic(batch, pad_id):
    # dynamic padding for speed
    max_len = max(x["input_ids"].shape[0] for x in batch)
    def pad(v, fill):
        out = torch.full((max_len,), fill, dtype=v.dtype)
        out[: v.shape[0]] = v
        return out
    input_ids = torch.stack([pad(x["input_ids"], pad_id) for x in batch])
    attn = torch.stack([pad(x["attention_mask"], 0) for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        preds = logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / max(total, 1)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    SAVE_DIR = Path(__file__).resolve().parent / "roberta_sentiment"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_docs(DATA_DIR)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    model_name = "textattack/roberta-base-SST-2"
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    max_len = 256 * 2
    train_ds = TextDS(Xtr, ytr, tok, max_len=max_len)
    val_ds   = TextDS(Xva, yva, tok, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,
                              collate_fn=lambda b: collate_dynamic(b, tok.pad_token_id))
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False,
                              collate_fn=lambda b: collate_dynamic(b, tok.pad_token_id))

    epochs = 5
    lr = 1e-6
    wd = 0.01
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    sched = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    best = 0.0
    patience, bad = 2, 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            sched.step()
            pbar.set_postfix(loss=float(loss.detach()))

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}: val_acc={val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            bad = 0
            model.save_pretrained(SAVE_DIR)
            tok.save_pretrained(SAVE_DIR)
            print("Saved best to:", SAVE_DIR)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Best val acc:", best)

if __name__ == "__main__":
    main()