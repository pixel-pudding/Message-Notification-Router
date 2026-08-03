"""Stage 8 (locked architecture doc section 8): CHECKPOINT.

Simple append-only checkpoint so `python code/main.py` can resume a partial
run instead of re-processing (and re-spending Groq API calls on) messages
already completed. One message_id per line. Only meaningful together with
main.py's output.csv also being written incrementally in lockstep -- on
resume, rows already in output.csv from the prior run stay put and only
new rows get appended.
"""
from __future__ import annotations

from pathlib import Path

import config

CHECKPOINT_PATH = config.PROJECT_ROOT / "code" / ".cache" / "checkpoint.txt"


def save(message_id: str, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
    """Append message_id as a newly-completed line."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("a", encoding="utf-8") as f:
        f.write(message_id + "\n")


def load(checkpoint_path: Path = CHECKPOINT_PATH) -> set[str]:
    """Return the set of message_ids already completed in a prior run.
    Empty set if no checkpoint file exists yet.
    """
    if not checkpoint_path.exists():
        return set()
    with checkpoint_path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def clear(checkpoint_path: Path = CHECKPOINT_PATH) -> None:
    """Delete the checkpoint file -- used by --no-resume for a clean run."""
    checkpoint_path.unlink(missing_ok=True)
