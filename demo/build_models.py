"""Pre-train the demo models into ./models so the deployed Space loads them
instantly instead of training on cold start.

    python build_models.py

The produced pickles are XGBoost/scikit-learn objects, so the Space must
install the same library versions they were built with — those are pinned in
requirements.txt. Re-run this whenever you bump those pins. The models/ folder
is not committed to git (see .gitignore); it is uploaded straight to the Space.
"""

from __future__ import annotations

from pathlib import Path

import logic

OUT = Path(__file__).parent / "models"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    logic.build_commit(save_to=OUT / "commit")
    logic.build_eta(save_to=OUT / "eta")
    print(f"wrote pre-trained models -> {OUT}")


if __name__ == "__main__":
    main()
