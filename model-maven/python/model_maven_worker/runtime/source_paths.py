"""Identify editable project Python files without guessing from directory names."""

import sys
import sysconfig
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=32)
def _protected_paths(root, prefix):
    protected = set()
    if prefix != root and prefix.is_relative_to(root):
        protected.add(prefix)
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        location = sysconfig.get_path(key)
        if location:
            protected.add(Path(location).resolve())
    return tuple(protected)


@lru_cache(maxsize=16)
def _prefix_path(value):
    return Path(value).resolve()


def is_project_python_source(root, path, *, resolved=False):
    root = Path(root) if resolved else Path(root).resolve()
    path = Path(path) if resolved else Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    if path.suffix != ".py" or any(part in {".git", "__pycache__", ".model-maven"} for part in relative.parts):
        return False
    return not any(path.is_relative_to(directory) for directory in _protected_paths(root, _prefix_path(sys.prefix)))
