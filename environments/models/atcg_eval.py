"""Run the in-tree evaluator from one isolated model environment."""

import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
PACKAGE_NAMES = ("atcg-sequence", "atcg-models", "atcg-runtime", "atcg-eval")

for package_name in PACKAGE_NAMES:
    sys.path.insert(0, str(REPOSITORY / "packages" / package_name / "src"))

from atcg.eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
