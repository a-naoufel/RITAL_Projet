import re
import html
import unicodedata
import argparse
from pathlib import Path

BASIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by",
    "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in",
    "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "than", "too",
    "very", "can", "will", "just", "should", "now", "is", "am", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "having", "have", "has",
    "he", "she", "it", "they", "them", "his", "her", "their", "this", "that",
    "these", "those", "i", "you", "we", "me", "him", "my", "your", "our"
}
NEGATIONS_TO_KEEP = {"no", "not", "nor", "never", "n't"}


def preprocess_text(text: str, lowercase=True, remove_numbers=False, remove_stopwords=False) -> str:
    if not isinstance(text, str):
        return ""

    # Decode HTML entities
    text = html.unescape(text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Remove HTML tags like <br />
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove quotes
    text = re.sub(r"[\"“”‘’`]", " ", text)

    # Remove URLs and emails
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    # Replace dashes/slashes with spaces
    text = re.sub(r"[-_/]", " ", text)

    # Remove numbers if requested
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)

    # Remove punctuation except apostrophes inside words
    text = re.sub(r"[^\w\s']", " ", text)

    # Lowercase
    if lowercase:
        text = text.lower()

    # Remove standalone apostrophes
    text = re.sub(r"\b'\b", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords if requested
    if remove_stopwords:
        words = []
        for token in text.split():
            if token in NEGATIONS_TO_KEEP or token not in BASIC_STOPWORDS:
                words.append(token)
        text = " ".join(words)

    return text


def clean_file(input_file, output_file=None, lowercase=True, remove_numbers=False, remove_stopwords=False):
    input_path = Path(input_file)

    if output_file is None:
        output_path = input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}")
    else:
        output_path = Path(output_file)

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            cleaned_line = preprocess_text(
                line,
                lowercase=lowercase,
                remove_numbers=remove_numbers,
                remove_stopwords=remove_stopwords
            )
            fout.write(cleaned_line + "\n")

    print(f"Cleaned file written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Clean a text file for machine learning tasks.")
    parser.add_argument("input_file", help="Path to input text file")
    parser.add_argument("-o", "--output_file", help="Path to output cleaned file", default=None)
    parser.add_argument("--keep-case", action="store_true", help="Keep original letter case")
    parser.add_argument("--remove-numbers", action="store_true", help="Remove numbers")
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove stopwords")

    args = parser.parse_args()

    clean_file(
        input_file=args.input_file,
        output_file=args.output_file,
        lowercase=not args.keep_case,
        remove_numbers=args.remove_numbers,
        remove_stopwords=args.remove_stopwords
    )


if __name__ == "__main__":
    main()
