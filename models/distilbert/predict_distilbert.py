from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

def load_lines_wc_like(fp: Path):
    text = fp.read_text(encoding="utf-8", errors="ignore")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    TEST_FILE = PROJECT_ROOT / "dataset" / "testSentiment.txt"

    MODEL_DIR = Path(__file__).resolve().parent / "distilbert_sentiment"
    OUT_DIR = Path(__file__).resolve().parent / "outputs"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    lines = load_lines_wc_like(TEST_FILE)

    out_path = OUT_DIR / "preds.txt"
    batch_size = 32
    max_len = 256

    with out_path.open("w", encoding="utf-8") as f:
        for i in tqdm(range(0, len(lines), batch_size), desc="Predict"):
            batch = lines[i:i+batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_len,
                return_tensors="pt"
            ).to(device)

            logits = model(**enc).logits
            preds = logits.argmax(dim=-1).tolist()

            for p in preds:
                f.write(("P" if p == 1 else "N") + "\n")

    print("Wrote:", out_path)
    print("Total predictions:", len(lines))

if __name__ == "__main__":
    main()