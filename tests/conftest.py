import sys
from pathlib import Path


def pytest_configure() -> None:
    # Ensure `src/` is importable without requiring an editable install.
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

