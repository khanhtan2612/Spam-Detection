"""Chạy notebook từ đầu bằng Jupyter rồi xuất báo cáo Markdown kèm output thật."""

import argparse
import os
from pathlib import Path
import sys

from scripts.ipynb_to_markdown import convert_notebook
from scripts.notebook_setup import ensure_dependencies


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=Path("svd_pca_spam.ipynb"))
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    root = notebook_path.parent
    ensure_dependencies(root)

    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError as error:
        raise SystemExit(
            "Thiếu thư viện notebook. Chạy: "
            ".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt"
        ) from error

    # Các file kết nối Jupyter và cache biểu đồ ở trong project, có thể tạo lại.
    runtime = root / ".jupyter_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    os.environ["IPYTHONDIR"] = str(runtime / "ipython")
    os.environ["MPLCONFIGDIR"] = str(root / ".mplconfig")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    # Dùng đúng interpreter đang chạy script, tránh kernel Python hệ thống thiếu ML.
    # Chỉ đổi cấu hình trong bộ nhớ; không đăng ký kernel vào tài khoản người dùng.
    manager = client.create_kernel_manager()
    manager.kernel_spec.argv[0] = sys.executable
    print(f"Đang chạy notebook bằng: {sys.executable}", flush=True)
    executed = client.execute()
    nbformat.write(executed, notebook_path)
    report = convert_notebook(notebook_path, root / "KET_QUA_MO_HINH.md")
    count = sum(cell.cell_type == "code" for cell in executed.cells)
    print(f"Đã chạy thành công {count} ô code: {notebook_path.name}")
    print(f"Báo cáo có code, output và hình: {report}")


if __name__ == "__main__":
    main()
