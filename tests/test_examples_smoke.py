"""Smoke tests: the zero-dependency examples must always run."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(rel: str) -> None:
    r = subprocess.run([sys.executable, str(ROOT / rel)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"{rel} failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}"


def test_approximate_quickstart_lifecycle():
    _run("examples/approximate_reuse/quickstart_lifecycle.py")


def test_exact_prefix_quickstart_trie():
    _run("examples/exact_prefix_reuse/quickstart_trie.py")
