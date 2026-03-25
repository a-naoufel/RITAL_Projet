from pathlib import Path
import re
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import GloVe
from tqdm import tqdm

SEED = 42
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def read_text(fp: Path) -> str:
    return fp.read_text(encoding="utf-8", errors="ignore")

def load_docs(movies_dir: Path):
    pos = sorted((movies_dir / "pos").glob("*.txt"))
    neg = sorted((movies_dir / "neg").glob("*.txt"))
    X = [read_text(p) for p in pos] + [read_text(n) for n in neg]
    y = [1]*len(pos) + [0]*len(neg)
    return X, y

def split_stratified(X, y, val_ratio=0.2):
    # simple stratified split
    idx_pos = [i for i, t in enumerate(y) if t == 1]
    idx_neg = [i for i, t in enumerate(y) if t == 0]
    random.shuffle(idx_pos); random.shuffle(idx_neg)
    nvp = int(len(idx_pos) * val_ratio)
    nvn = int(len(idx_neg) * val_ratio)
    val_idx = idx_pos[:nvp] + idx_neg[:nvn]
    tr_idx  = idx_pos[nvp:] + idx_neg[nvn:]
    random.shuffle(val_idx); random.shuffle(tr_idx)
    Xtr = [X[i] for i in tr_idx]; ytr = [y[i] for i in tr_idx]
    Xva = [X[i] for i in val_idx]; yva = [y[i] for i in val_idx]
    return (Xtr, ytr), (Xva, yva)

class Vocab:
    def __init__(self, min_freq=2, max_size=50000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.PAD = 0
        self.UNK = 1
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {t:i for i,t in enumerate(self.itos)}

    def build(self, tokenized_texts):
        c = Counter()
        for toks in tokenized_texts:
            c.update(toks)
        items = sorted(c.items(), key=lambda x:(-x[1], x[0]))
        items = [(t,f) for t,f in items if f >= self.min_freq]
        if self.max_size:
            items = items[: max(0, self.max_size - len(self.itos))]
        for tok,_ in items:
            if tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, toks):
        return [self.stoi.get(t, self.UNK) for t in toks]

    def __len__(self):
        return len(self.itos)

class ReviewDS(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len=400):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self): return len(self.texts)

    def __getitem__(self, i):
        toks = tokenize(self.texts[i])[: self.max_len]
        ids = self.vocab.encode(toks) or [self.vocab.UNK]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[i], dtype=torch.float32)

def collate(batch, pad_id=0):
    xs, ys = zip(*batch)
    lens = torch.tensor([len(x) for x in xs], dtype=torch.long)
    T = lens.max().item()
    X = torch.full((len(xs), T), pad_id, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :x.numel()] = x
    Y = torch.stack(ys)
    return X, lens, Y

class BiLSTM(nn.Module):
    def __init__(self, emb_matrix, hidden=256, dropout=0.3, pad_id=0):
        super().__init__()
        V, D = emb_matrix.shape
        self.embedding = nn.Embedding(V, D, padding_idx=pad_id)
        self.embedding.weight.data.copy_(emb_matrix)

        self.lstm = nn.LSTM(
            input_size=D,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # max pooling over time (often strong for sentiment)
        mask = (x != 0).unsqueeze(-1)  # [B,T,1]
        out = out.masked_fill(~mask, float("-inf"))
        pooled, _ = out.max(dim=1)     # [B,2H]
        logits = self.fc(self.dropout(pooled)).squeeze(1)
        return logits

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    for x,l,y in loader:
        x,l,y = x.to(device), l.to(device), y.to(device)
        logits = model(x,l)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * y.size(0)
        pred = (torch.sigmoid(logits) >= 0.5).long()
        correct += (pred == y.long()).sum().item()
        total += y.size(0)
    return total_loss/total, correct/total

def build_glove_matrix(vocab: Vocab, dim=100):
    glove = GloVe(name="6B", dim=dim)  # downloads on first use
    V = len(vocab)
    emb = torch.empty((V, dim))
    emb.normal_(mean=0.0, std=0.02)

    hits = 0
    for i, tok in enumerate(vocab.itos):
        if tok in glove.stoi:
            emb[i] = glove.vectors[glove.stoi[tok]]
            hits += 1

    # PAD should be zeros
    emb[vocab.PAD].zero_()
    print(f"GloVe coverage: {hits}/{V} = {hits/V:.2%}")
    return emb

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "dataset" / "movies1000"
    OUT_DIR = Path(__file__).resolve().parent / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT = Path(__file__).resolve().parent / "bilstm_glove.pt"

    X, y = load_docs(DATA_DIR)
    (Xtr, ytr), (Xva, yva) = split_stratified(X, y, val_ratio=0.2)

    tokenized = [tokenize(t) for t in Xtr]
    vocab = Vocab(min_freq=2, max_size=50000)
    vocab.build(tokenized)
    print("Vocab size:", len(vocab))

    max_len = 400
    train_ds = ReviewDS(Xtr, ytr, vocab, max_len=max_len)
    val_ds   = ReviewDS(Xva, yva, vocab, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id=vocab.PAD))
    val_loader   = DataLoader(val_ds, batch_size=64, shuffle=False,
                              collate_fn=lambda b: collate(b, pad_id=vocab.PAD))

    emb_matrix = build_glove_matrix(vocab, dim=100)
    model = BiLSTM(emb_matrix, hidden=128, dropout=0.5, pad_id=vocab.PAD).to(device)

    # Freeze embeddings for 1 epoch then unfreeze
    model.embedding.weight.requires_grad = False

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    patience, bad = 3, 0
    epochs = 10

    for ep in range(1, epochs+1):
        if ep == 2:
            model.embedding.weight.requires_grad = True
            print("Unfroze embeddings")

        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for x,l,yb in pbar:
            x,l,yb = x.to(device), l.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(x,l)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach()))

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {ep}: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad = 0
            torch.save({
                "model_state": model.state_dict(),
                "vocab": vocab.__dict__,
                "config": {"max_len": max_len, "emb_dim": 100, "hidden": 128, "dropout": 0.5},
            }, CKPT)
            print("Saved:", CKPT)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Best val acc:", best_acc)

if __name__ == "__main__":
    main()