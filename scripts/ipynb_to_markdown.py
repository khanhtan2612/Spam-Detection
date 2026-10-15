"""Xuất notebook đã chạy thành báo cáo Markdown bằng thư viện chuẩn Python.

Ví dụ: python -m scripts.ipynb_to_markdown svd_pca_spam.ipynb
Chỉ chuyển nội dung và output có sẵn; chương trình không chạy lại notebook.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
import sys


ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)


def _text(value: str | list[str]) -> str:
    """Notebook có thể lưu source/output dưới dạng chuỗi hoặc danh sách dòng."""
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(line, str) for line in value):
        return "".join(value)
    raise ValueError("Source và output văn bản phải là chuỗi hoặc danh sách chuỗi.")


def _without_ansi(value: str | list[str]) -> str:
    return ANSI_ESCAPE.sub("", _text(value))


def _fenced(value: str, language: str) -> str:
    # Dùng hàng rào dài hơn backtick trong nội dung để Markdown không bị ngắt.
    longest = max((len(run) for run in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{value.rstrip(chr(10))}\n{fence}"


def _validate_cells(cells: list[dict]) -> None:
    """Không xuất báo cáo khi notebook còn lỗi hoặc chưa chạy hết code."""
    for index, cell in enumerate(cells, start=1):
        if cell.get("cell_type") != "code":
            continue
        source = _text(cell.get("source", ""))
        if source.strip() and cell.get("execution_count") is None:
            raise ValueError(
                f"Cell {index} chưa được chạy. Hãy chạy toàn bộ notebook trước khi xuất."
            )
        if any(output.get("output_type") == "error" for output in cell.get("outputs", [])):
            raise ValueError(
                f"Cell {index} có output lỗi. Hãy sửa lỗi và chạy lại trước khi xuất."
            )


def convert_notebook(
    notebook_path: str | Path,
    output_path: str | Path = "KET_QUA_MO_HINH.md",
) -> Path:
    """Giữ Markdown, code và output thật từ notebook đã chạy thành một báo cáo.

    Dùng lại PNG giống hệt đã có ở outputs/figures/ để tránh lưu ảnh trùng.
    Ảnh mới được lưu trong outputs/figures/notebook/. Các link đều tương đối.
    """
    notebook_path = Path(notebook_path)
    output_path = Path(output_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8-sig"))
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        raise ValueError("File notebook không có danh sách cells hợp lệ.")
    _validate_cells(cells)

    blocks: list[str] = []
    images: list[tuple[Path, bytes]] = []
    for cell_index, cell in enumerate(cells, start=1):
        source = _text(cell.get("source", ""))
        if cell.get("cell_type") == "markdown":
            if source.strip():
                blocks.append(source.rstrip("\n"))
            continue
        if cell.get("cell_type") != "code":
            continue
        if source.strip():
            blocks.append(_fenced(source, "python"))

        for output_index, output in enumerate(cell.get("outputs", []), start=1):
            if output.get("output_type") == "stream":
                content = _without_ansi(output.get("text", ""))
                if content.strip():
                    blocks.append(_fenced(content, "text"))
                continue
            if output.get("output_type") not in {"display_data", "execute_result"}:
                continue

            # Mỗi output chỉ chọn một dạng hiển thị để tránh bảng/ảnh bị lặp.
            data = output.get("data", {})
            if "text/markdown" in data:
                blocks.append(_without_ansi(data["text/markdown"]).rstrip("\n"))
            elif "image/png" in data:
                encoded_image = "".join(_text(data["image/png"]).split())
                image_bytes = base64.b64decode(encoded_image, validate=True)
                figure_dir = output_path.parent / "outputs" / "figures"
                # Notebook đã tạo các biểu đồ có tên rõ ràng; tái dùng đúng byte.
                image_path = next(
                    (path for path in sorted(figure_dir.glob("*.png"))
                     if path.read_bytes() == image_bytes),
                    None,
                )
                if image_path is None:
                    image_path = figure_dir / "notebook" / f"cell_{cell_index}_output_{output_index}.png"
                    images.append((image_path, image_bytes))
                relative_path = Path(
                    os.path.relpath(image_path, output_path.parent)
                ).as_posix()
                blocks.append(
                    f"![Kết quả cell {cell_index}, output {output_index}](<{relative_path}>)"
                )
            elif "text/plain" in data:
                content = _without_ansi(data["text/plain"])
                if content.strip():
                    blocks.append(_fenced(content, "text"))

    # Chỉ ghi file sau khi đã đọc và chuyển đổi hợp lệ toàn bộ notebook.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for image_path, image_bytes in images:
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(image_bytes)
    output_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(
        description="Chuyển notebook đã chạy thành báo cáo Markdown kèm output thật."
    )
    parser.add_argument(
        "notebook", nargs="?", default="svd_pca_spam.ipynb", help="Đường dẫn notebook."
    )
    parser.add_argument(
        "--output", default="KET_QUA_MO_HINH.md", help="Đường dẫn báo cáo Markdown."
    )
    args = parser.parse_args()
    report_path = convert_notebook(args.notebook, args.output)
    print(f"Đã xuất báo cáo: {report_path.resolve()}")


if __name__ == "__main__":
    main()
