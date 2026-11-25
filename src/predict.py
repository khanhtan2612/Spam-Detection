"""Dự đoán một SMS bằng pipeline đã lưu, không fit lại.

Ví dụ: python -m src.predict --model outputs/models/svd --text "Call now to claim your prize"
"""

import argparse
import json
from pathlib import Path

from .pipeline import load_pipeline


def predict_message(model, text):
    """Trả dict gồm tin nhắn, nhãn dự đoán và P(spam)."""
    if not text.strip():
        raise ValueError("Tin nhắn không được rỗng.")
    probability = float(model.predict_proba([text])[0])
    return {"text": text, "prediction": "spam" if probability >= 0.5 else "ham",
            "spam_probability": round(probability, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/svd"))
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    result = predict_message(load_pipeline(args.model), args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
