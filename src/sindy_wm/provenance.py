"""Recording which code and inputs produced a result.

Used wherever something is saved that should be traceable later: datasets
(see data/storage.py) and, later, model fits and other run outputs.

    code_info()           git commit, dirty flag, Python and package versions
    file_sha256(path)     hash of an input file, e.g. a scenario JSON
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

# Packages whose versions are recorded by default. Packages that are not
# installed are recorded as None.
TRACKED_PACKAGES = ("smaclite", "numpy", "pandas", "gymnasium", "pyarrow", "pysindy")


def code_info(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, Any]:
    """Git commit, dirty flag, and versions of Python and key packages.

    The commit only describes committed code. If the working tree has
    uncommitted changes, `git_dirty` is True and `git_changed_files` lists them.
    Git fields are None if git or the repository is not available.
    """
    repo_dir = Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], repo_dir)
    status = _git(["status", "--porcelain"], repo_dir)
    # Porcelain lines look like " M src/file.py": two status letters, a space, the path
    changed = [line[3:] for line in status.splitlines() if line.strip()] if status else []
    return {
        "git_commit": commit.strip() if commit else None,
        "git_dirty": bool(changed),
        "git_changed_files": changed,
        "python": platform.python_version(),
        "packages": {name: _package_version(name) for name in packages},
        "smaclite_commit": _vcs_commit("smaclite"),
    }


def file_sha256(path: str | Path) -> str:
    """SHA-256 hash of a file, e.g. a scenario JSON, to detect later edits."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a git command; None if git or the repository is not available."""
    try:
        result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _vcs_commit(name: str) -> str | None:
    """Commit of a package installed from git (read from its direct_url.json)."""
    try:
        text = metadata.distribution(name).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    if not text:
        return None
    return json.loads(text).get("vcs_info", {}).get("commit_id")
