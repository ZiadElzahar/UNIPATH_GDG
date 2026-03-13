"""Compatibility test entrypoint."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "test_rag.py"), run_name="__main__")
