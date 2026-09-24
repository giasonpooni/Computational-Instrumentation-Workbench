"""Regenerate docs/PIPELINES.md from the pipeline descriptors (the canonical data)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ciw.pipelines import check, render_catalog  # noqa: E402

(ROOT / "docs" / "PIPELINES.md").write_text(render_catalog(check()), encoding="utf-8")
print("docs/PIPELINES.md regenerated")
