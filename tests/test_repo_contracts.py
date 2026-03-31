from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_gitignore_covers_generated_artifacts():
    text = (ROOT / ".gitignore").read_text()

    for entry in ("outputs/", "logs/", "results/"):
        assert entry in text
