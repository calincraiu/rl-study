from pathlib import Path


def find_repo_root(start_path: Path) -> Path:
    for path in [start_path, *start_path.parents]:
        if (path / ".git").exists() or (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Repo root not found")