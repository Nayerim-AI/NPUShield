"""Load knowledge base files from kb/ directory into FTS5 index."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .kb_index import KBIndex


def load_kb(kb_dir: str | Path) -> KBIndex:
    """Scan kb/ directory and index all markdown, yaml, and json files."""
    kb_dir = Path(kb_dir)
    if not kb_dir.is_dir():
        raise FileNotFoundError(f"KB directory not found: {kb_dir}")

    kb = KBIndex(db_path=os.path.join(str(kb_dir), ".kb_cache.db"))
    index_path = kb_dir / ".kb_index.json"

    # Check cache: skip re-indexing if files haven't changed
    if _check_cache(kb_dir, index_path):
        # Load from cached KB file list
        cached = json.loads(index_path.read_text())
        for entry in cached["entries"]:
            text = (kb_dir / entry["path"]).read_text(encoding="utf-8") if (kb_dir / entry["path"]).exists() else entry["text"]
            kb.add_document(entry["path"], text)
        return kb

    # Index all files
    entries = []
    for file_path in sorted(kb_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix not in (".md", ".yaml", ".yml", ".json"):
            continue
        if file_path.name.startswith("."):
            continue

        rel_path = str(file_path.relative_to(kb_dir))
        text = file_path.read_text(encoding="utf-8")
        kb.add_document(rel_path, text)

        # Determine persona from first directory segment
        parts = rel_path.replace("\\", "/").split("/")
        persona = parts[0] if len(parts) >= 2 else "general"
        entries.append({
            "path": rel_path,
            "text": text,
            "persona": persona,
            "size": len(text),
        })

    # Save cache
    index_path.write_text(json.dumps({"entries": entries}, indent=2), encoding="utf-8")
    return kb


def _check_cache(kb_dir: Path, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    kb_mtimes = {}
    for f in sorted(kb_dir.rglob("*")):
        if f.is_file() and f.suffix in (".md", ".yaml", ".yml", ".json"):
            kb_mtimes[str(f.relative_to(kb_dir))] = f.stat().st_mtime
    try:
        cached = json.loads(cache_path.read_text())
        cached_entries = cached.get("entries", [])
        if len(kb_mtimes) != len(cached_entries):
            return False
        for entry in cached_entries:
            path = entry["path"]
            if path not in kb_mtimes:
                return False
        return True
    except (json.JSONDecodeError, KeyError, OSError):
        return False
