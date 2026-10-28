"""Smoke tests for the whole repository.

Every algorithm file is a runnable script. These tests run each one in a fresh
interpreter and check the contract the README promises: it exits cleanly, prints
something, and prints plain ASCII. They also check that the algorithm folders
and the links in the README stay in sync.

Run from the repository root:

    python -m pytest
"""
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_ALGORITHMS = 28


def _number(folder):
    return int(folder.name.split(".")[0])


FOLDERS = sorted(
    (p for p in ROOT.iterdir() if p.is_dir() and re.match(r"\d+\. ", p.name)),
    key=_number,
)


def _script(folder):
    scripts = sorted(folder.glob("_*.py"))
    assert len(scripts) == 1, f"{folder.name}: expected one _N_name.py file, found {len(scripts)}"
    return scripts[0]


def test_algorithm_numbering_is_contiguous():
    assert [_number(f) for f in FOLDERS] == list(range(1, EXPECTED_ALGORITHMS + 1))


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda f: f.name)
def test_folder_has_script_and_guide(folder):
    _script(folder)
    guides = sorted(folder.glob("_*.md"))
    assert len(guides) == 1, f"{folder.name}: expected one guide, found {len(guides)}"
    assert "Quick Start" in guides[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("folder", FOLDERS, ids=lambda f: f.name)
def test_demo_runs_cleanly(folder):
    result = subprocess.run(
        [sys.executable, "-B", str(_script(folder))],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip(), "the demo printed nothing"
    assert result.stdout.isascii(), "demo output must be plain ASCII"


def test_readme_links_resolve():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    targets = re.findall(r"\]\((?!https?://|#|mailto:)([^)\s]+)\)", readme)
    assert targets, "no relative links found in README.md"
    missing = [t for t in targets if not (ROOT / unquote(t.split("#")[0])).exists()]
    assert not missing, f"README links to files that do not exist: {missing}"
