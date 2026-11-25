"""Chạy notebook từ đầu bằng Jupyter rồi xuất báo cáo Markdown kèm output thật.

Ví dụ: python -m scripts.run_notebook
"""

import argparse
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient

from scripts.ipynb_to_markdown import convert_notebook


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path("svd_pca_spam.ipynb"))
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    root = notebook_path.parent

    # File tạm của Jupyter và matplotlib nằm trong project, có thể xóa.
    runtime = root / ".jupyter_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLCONFIGDIR"] = str(root / ".mplconfig")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    notebook = nbformat.read(notebook_path, as_version=4)
    # Mỗi ô tối đa 10 phút: ô chọn k và ô test chạy SVD/PCA với k = 100.
    client = NotebookClient(notebook, timeout=600, kernel_name="python3",
                            resources={"metadata": {"path": str(root)}})
    # Kernel dùng đúng Python đang chạy script này (không đăng ký kernel mới).
    client.create_kernel_manager().kernel_spec.argv[0] = sys.executable
    print(f"Đang chạy notebook bằng: {sys.executable}", flush=True)
    executed = client.execute()
    nbformat.write(executed, notebook_path)
    report = convert_notebook(notebook_path, root / "KET_QUA_MO_HINH.md")
    count = sum(cell.cell_type == "code" for cell in executed.cells)
    print(f"Đã chạy thành công {count} ô code: {notebook_path.name}")
    print(f"Báo cáo: {report}")


if __name__ == "__main__":
    main()
