from pathlib import Path
import os
import random
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW

# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_NEG_DIR = PROJECT_ROOT / "dataset" / "movies1000" / "neg"
TRAIN_POS_DIR = PROJECT_ROOT / "dataset" / "movies1000" / "pos"
TEST_FILE = PROJECT_ROOT / "dataset" / "testSentiment.txt"
RESULT_FILE = PROJECT_ROOT / "results.txt"

# Local model path
MODEL_NAME = str(PROJECT_ROOT / "models" / "roberta-imdb")

# Keep True if your machine cannot access Hugging Face
LOCAL_FILES_ONLY = True

MAX_LENGTH = 128
SEED = 42

NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
VAL_SIZE = 0.1
PATIENCE = 2
# =========================================================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_data():
    texts = []
    labels = []

    if not TRAIN_NEG_DIR.exists():
        raise FileNotFoundError(f"Missing negative training directory: {TRAIN_NEG_DIR}")
    if not TRAIN_POS_DIR.exists():
        raise FileNotFoundError(f"Missing positive training directory: {TRAIN_POS_DIR}")

    for path in sorted(TRAIN_NEG_DIR.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        labels.append(0)

    for path in sorted(TRAIN_POS_DIR.glob("*.txt")):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
        labels.append(1)

    if not texts:
        raise ValueError("No training files found in pos/neg directories.")

    return texts, labels


def load_test_data():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Missing test file: {TEST_FILE}")

    with TEST_FILE.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=-1)

            total_loss += loss.item()
            steps += 1

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(steps, 1)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def predict_model(model, dataloader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())

    return all_preds


def main():
    set_seed(SEED)

    if LOCAL_FILES_ONLY:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device              : {device}")

    texts, labels = load_training_data()
    test_texts = load_test_data()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts,
        labels,
        test_size=VAL_SIZE,
        random_state=SEED,
        stratify=labels,
    )

    print(f"Training examples   : {len(train_texts)}")
    print(f"Validation examples : {len(val_texts)}")
    print(f"Test examples       : {len(test_texts)}")
    print(f"Model               : {MODEL_NAME}")
    print(f"Local files only    : {LOCAL_FILES_ONLY}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model.to(device)

    train_dataset = ReviewDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    val_dataset = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    test_dataset = ReviewDataset(test_texts, None, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    best_val_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / max(train_steps, 1)
        val_loss, val_acc = evaluate_model(model, val_loader, device)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_val_acc = evaluate_model(model, val_loader, device)
    print(f"Best validation accuracy: {final_val_acc:.4f}")

    predictions = predict_model(model, test_loader, device)

    with RESULT_FILE.open("w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(("P" if pred == 1 else "N") + "\n")

    print(f"Done. Results written to: {RESULT_FILE}")


if __name__ == "__main__":
    main()
