from pathlib import Path
import re
import torch
import torch.nn as nn

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())

def load_lines_wc_like(fp: Path):
    text = fp.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines

class Vocab:
    def __init__(self):
        self.PAD = 0
        self.UNK = 1
        self.itos = []
        self.stoi = {}
    def encode(self, toks):
        return [self.stoi.get(t, self.UNK) for t in toks]
    def __len__(self): return len(self.itos)

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=100, hidden=128, dropout=0.5, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        mask = (x != 0).unsqueeze(-1)
        out = out.masked_fill(~mask, float("-inf"))
        pooled, _ = out.max(dim=1)
        return self.fc(self.dropout(pooled)).squeeze(1)

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_FILE = PROJECT_ROOT / "dataset" / "testSentiment.txt"
    OUT_DIR = Path(__file__).resolve().parent / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(Path(__file__).resolve().parent / "bilstm_glove.pt", map_location=device)
    vocab = Vocab()
    vocab.__dict__.update(ckpt["vocab"])
    cfg = ckpt["config"]

    model = BiLSTM(len(vocab), cfg["emb_dim"], cfg["hidden"], cfg["dropout"], vocab.PAD).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    max_len = cfg["max_len"]
    lines = load_lines_wc_like(TEST_FILE)

    out_path = OUT_DIR / "preds.txt"
    with out_path.open("w", encoding="utf-8") as f:
        for t in lines:
            toks = tokenize(t)[:max_len]
            ids = vocab.encode(toks) or [vocab.UNK]
            x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
            lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
            logit = model(x, lengths)
            prob = torch.sigmoid(logit).item()
            f.write(("P" if prob >= 0.5 else "N") + "\n")

    print("Wrote:", out_path)
    print("Total:", len(lines))

if __name__ == "__main__":
    main()