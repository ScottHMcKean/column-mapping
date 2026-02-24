from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RepoPaths:
    repo_root_ws: str

    @property
    def repo_root_file(self) -> str:
        # Workspace files are readable via file:/
        return f"file:{self.repo_root_ws}"

    def data_file(self, relative_name: str) -> str:
        return f"{self.repo_root_file}/data/{relative_name}"


def get_repo_paths(dbutils) -> RepoPaths:
    """Resolve the repo root workspace path from the current notebook path."""
    nb_path = (
        dbutils.notebook.entry_point.getDbutils()
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    if not nb_path.startswith("/Workspace/"):
        nb_path = f"/Workspace{nb_path}"
    # Example notebook path:
    # /Workspace/Repos/user@example.com/column-mapping/notebooks/01_setup...
    parts = nb_path.split("/")
    if "notebooks" not in parts:
        # Best-effort fallback: assume notebook is at repo root
        repo_root_ws = "/".join(parts[:-1])
    else:
        notebooks_idx = parts.index("notebooks")
        repo_root_ws = "/".join(parts[:notebooks_idx])
    return RepoPaths(repo_root_ws=repo_root_ws)

