"""Đọc dữ liệu UCI, loại trùng và chia tập trước khi học đặc trưng."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DEFAULT_DATA = Path("data/raw/SMSSpamCollection")


def normalize_text(text):
    """Dùng cùng một quy tắc lúc loại trùng, huấn luyện và dự đoán SMS mới."""
    return re.sub(r"\s+", " ", text).strip().lower()


def download_data(destination=DEFAULT_DATA):
    """Tải dữ liệu chính thức một lần; giữ README và mã SHA-256 của gói tải."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Đã có dữ liệu: {destination}; không tải lại.")
        return
    request = urllib.request.Request(DATA_URL, headers={"User-Agent": "Student-ML-Project/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if Path(n).name == "SMSSpamCollection")
        destination.write_bytes(archive.read(member))
        for member in archive.namelist():
            if Path(member).name.lower() == "readme":
                (destination.parent / "UCI_README.txt").write_bytes(archive.read(member))
    metadata = {"source_url": DATA_URL, "archive_sha256": hashlib.sha256(payload).hexdigest()}
    (destination.parent / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Đã tải dữ liệu UCI thật: {destination}")


def load_data(path=DEFAULT_DATA):
    """Đọc các dòng nhãn<TAB>SMS; trả (DataFrame đã sạch, thống kê làm sạch)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Thiếu {path}. Chạy python -m src.data --download hoặc đặt file UCI vào đây."
        )
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        label, separator, text = line.partition("\t")
        if not separator or label not in {"ham", "spam"}:
            raise ValueError(f"Nhãn/định dạng không hợp lệ tại dòng {line_number}.")
        records.append({"source_id": line_number, "label": label, "text": text})
    raw = pd.DataFrame(records, columns=["source_id", "label", "text"])
    return clean_data(raw)


def clean_data(raw):
    """Làm sạch trước chia tập; không học vocabulary, IDF hoặc hướng chiếu.

    Mỗi nội dung chuẩn hóa chỉ giữ một bản. Nếu nội dung có cả hai nhãn,
    loại toàn bộ nhóm vì không có căn cứ chọn nhãn đúng.
    """
    if not set(raw["label"]).issubset({"ham", "spam"}):
        raise ValueError("Chỉ chấp nhận nhãn ham/spam.")
    frame = raw.copy()
    frame["text"] = frame["text"].fillna("").map(normalize_text)
    # 1. Loại nội dung rỗng sau chuẩn hóa.
    empty = int(frame["text"].eq("").sum())
    frame = frame.loc[frame["text"].ne("")].copy()
    # 2. Loại nhóm có nhãn mâu thuẫn, rồi giữ một bản cho mỗi nội dung.
    conflicts = frame.groupby("text")["label"].nunique()
    conflicting_texts = conflicts[conflicts > 1].index
    conflicting_rows = int(frame["text"].isin(conflicting_texts).sum())
    frame = frame.loc[~frame["text"].isin(conflicting_texts)].copy()
    duplicates = int(frame.duplicated("text").sum())
    frame = frame.drop_duplicates("text").reset_index(drop=True)
    # 3. Lưu dấu vân tay để kiểm tra trùng và đối chiếu split trong outputs/results/.
    frame["text_sha256"] = frame["text"].map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    summary = {
        "raw_rows": len(raw), "empty_removed": empty,
        "conflicting_groups": len(conflicting_texts), "conflicting_rows_removed": conflicting_rows,
        "duplicates_removed": duplicates, "clean_rows": len(frame),
        "class_counts": {k: int(v) for k, v in frame["label"].value_counts().items()},
    }
    return frame, summary


def split_data(frame, seed=42):
    """Trả dict train/validation/test gần 70/15/15, giữ tỷ lệ ham/spam.

    Lần 1 giữ lại 30%; lần 2 chia phần này thành hai nửa validation và test.
    Không fit TF-IDF ở đây: chỉ chia các dòng văn bản đã làm sạch.
    """
    train, heldout = train_test_split(
        frame, test_size=0.30, stratify=frame["label"], random_state=seed
    )
    validation, test = train_test_split(
        heldout, test_size=0.50, stratify=heldout["label"], random_state=seed
    )
    return {"train": train.copy(), "validation": validation.copy(), "test": test.copy()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    if args.download:
        download_data(args.data)
    frame, summary = load_data(args.data)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
