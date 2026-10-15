"""Đồ án nhỏ: giảm chiều TF-IDF cho phát hiện SMS spam."""

import sys

# Windows có thể dùng cp1252 khi pipe output; dùng UTF-8 để in tiếng Việt.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")
