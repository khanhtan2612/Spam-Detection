"""Dự đoán SMS bằng pipeline đã lưu, không fit lại."""

import argparse
import json
from pathlib import Path

import joblib


def predict_message(model, text):
    """Đưa SMS qua pipeline đã học; trả dict nội dung, nhãn và P(spam).

    Thứ tự cột xác suất theo model.classes_, nên tìm vị trí nhãn spam.
    Xác suất là ước lượng của classifier, chưa được hiệu chỉnh riêng.
    """
    if not text.strip():
        raise ValueError("Tin nhắn không được rỗng.")
    probability = model.predict_proba([text])[0]
    spam_index = list(model.classes_).index("spam")
    return {"text": text, "prediction": str(model.predict([text])[0]),
            "spam_probability": float(probability[spam_index])}


def main():
    """CLI chỉ load và dự đoán; không gọi fit và không cần dataset huấn luyện."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/svd.joblib"))
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    print(json.dumps(predict_message(joblib.load(args.model), args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
