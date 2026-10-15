"""Kiểm tra thư viện và cài bằng đúng Python đang chạy notebook."""

import importlib
import importlib.util
from pathlib import Path
import subprocess
import sys


REQUIRED_MODULES = {
    "joblib": "joblib",
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "threadpoolctl": "threadpoolctl",
    "IPython": "ipython",
    "ipykernel": "ipykernel",
    "nbclient": "nbclient",
    "nbformat": "nbformat",
}


def ensure_dependencies(project_root):
    """Cài requirements nếu thiếu thư viện; không chạy pip khi đã có đủ."""
    project_root = Path(project_root).resolve()
    requirements = project_root / "requirements.txt"
    missing = [
        package for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return

    if not requirements.is_file():
        raise FileNotFoundError(f"Không tìm thấy file thư viện: {requirements}")
    print("Đang cài thư viện còn thiếu: " + ", ".join(missing), flush=True)
    print(f"Python của notebook: {sys.executable}", flush=True)
    command = [sys.executable, "-m", "pip", "install", "-r", str(requirements)]
    try:
        subprocess.check_call(command, cwd=project_root)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Chưa cài được thư viện cho Python của notebook. "
            "Kiểm tra kết nối Internet và quyền cài đặt, rồi chạy lại ô này. "
            f'Lệnh cài: "{sys.executable}" -m pip install -r "{requirements}"'
        ) from error

    importlib.invalidate_caches()
    missing = [
        package for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise RuntimeError(
            f"Python {sys.executable} vẫn chưa nạp được: {', '.join(missing)}. "
            "Khởi động lại kernel rồi chạy lại các ô từ đầu."
        )
