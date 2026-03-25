from pathlib import Path
import random
import numpy as np
import torch
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
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tok(
            self.texts[i],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds = out.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
    return correct / max(total, 1)

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    SAVE_DIR = Path(__file__).resolve().parent / "distilbert_sentiment"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_docs(DATA_DIR)

    # IMPORTANT: stratified split for stability
    Xtr, Xva, ytr, yva = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Start from a sentiment-ready checkpoint
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    max_len = 512
    train_ds = TextDS(Xtr, ytr, tokenizer, max_len=max_len)
    val_ds   = TextDS(Xva, yva, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

    epochs = 3
    lr = 1e-5
    wd = 0.01
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    total_steps = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_acc = 0.0
    patience = 2
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=float(loss.detach()))

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}: val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print("Saved best model to:", SAVE_DIR)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Best val acc:", best_acc)

if __name__ == "__main__":
    main()