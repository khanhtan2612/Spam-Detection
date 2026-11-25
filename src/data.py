"""Đọc dữ liệu SMS, làm sạch và chia train/validation/test theo từng lớp."""

import argparse
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_DATA = Path("data/raw/SMSSpamCollection")


def normalize_text(text):
    """Chữ thường và gộp khoảng trắng; dùng chung cho làm sạch, train và dự đoán."""
    return re.sub(r"\s+", " ", text).strip().lower()


def download_data(destination=DEFAULT_DATA):
    """Tải file SMSSpamCollection từ UCI nếu máy chưa có."""
    destination = Path(destination)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if Path(n).name == "SMSSpamCollection")
        destination.write_bytes(archive.read(member))
    print(f"Đã tải dữ liệu UCI vào {destination}")


def load_data(path=DEFAULT_DATA):
    """Đọc các dòng 'nhãn<TAB>tin nhắn'; trả (DataFrame đã làm sạch, thống kê)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Chạy: python -m src.data --download")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        label, separator, text = line.partition("\t")
        if not separator or label not in {"ham", "spam"}:
            raise ValueError(f"Dòng {line_number} không đúng định dạng nhãn<TAB>tin nhắn.")
        records.append({"source_id": line_number, "label": label, "text": text})
    return clean_data(pd.DataFrame(records, columns=["source_id", "label", "text"]))


def clean_data(raw):
    """Chuẩn hóa văn bản, bỏ tin rỗng, bỏ nội dung có hai nhãn mâu thuẫn, giữ một bản mỗi nội dung.

    Làm sạch trước khi chia tập để một nội dung không xuất hiện ở hai tập.
    """
    if not set(raw["label"]).issubset({"ham", "spam"}):
        raise ValueError("Chỉ chấp nhận nhãn ham/spam.")
    frame = raw.copy()
    frame["text"] = frame["text"].fillna("").map(normalize_text)
    empty = int(frame["text"].eq("").sum())
    frame = frame.loc[frame["text"].ne("")]
    conflicts = frame.groupby("text")["label"].nunique()
    conflicting_texts = conflicts[conflicts > 1].index
    conflicting_rows = int(frame["text"].isin(conflicting_texts).sum())
    frame = frame.loc[~frame["text"].isin(conflicting_texts)]
    duplicates = int(frame.duplicated("text").sum())
    frame = frame.drop_duplicates("text").reset_index(drop=True)
    summary = {
        "raw_rows": len(raw), "empty_removed": empty,
        "conflicting_rows_removed": conflicting_rows, "duplicates_removed": duplicates,
        "clean_rows": len(frame),
        "class_counts": {label: int((frame["label"] == label).sum()) for label in ("ham", "spam")},
    }
    return frame, summary


def stratified_split(df, ratios=(0.7, 0.15, 0.15), seed=42):
    """Chia df thành train/validation/test, giữ tỷ lệ ham/spam ở cả ba tập.

    Xáo từng lớp bằng np.random.default_rng(seed), cắt n_train = round(0.7·n), n_val = round(0.15·n),
    phần còn lại là test; cuối cùng xáo thứ tự trong mỗi tập.
    """
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1.0):
        raise ValueError("ratios phải có 3 số dương, tổng bằng 1.")
    rng = np.random.default_rng(seed)
    parts = {"train": [], "validation": [], "test": []}
    for label in sorted(df["label"].unique()):
        index = rng.permutation(df.index[df["label"] == label].to_numpy())
        n_train = round(ratios[0] * len(index))
        n_val = round(ratios[1] * len(index))
        parts["train"].append(index[:n_train])
        parts["validation"].append(index[n_train:n_train + n_val])
        parts["test"].append(index[n_train + n_val:])
    return {name: df.loc[rng.permutation(np.concatenate(ids))].copy() for name, ids in parts.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Tải dữ liệu UCI nếu chưa có.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.download:
        download_data(args.data)
    _, summary = load_data(args.data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
