"""Regenerate docs/PIPELINES.md from the descriptors and the CI gate registry (the canonical data)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ciw.pipelines import check, render_catalog  # noqa: E402
import ci_matrix  # noqa: E402


def render() -> str:
    return render_catalog(check()) + "\n" + ci_matrix.render_gates(ci_matrix.check())


if __name__ == "__main__":
    (ROOT / "docs" / "PIPELINES.md").write_text(render(), encoding="utf-8")
    print("docs/PIPELINES.md regenerated")
