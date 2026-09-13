"""Script to clean up duplicate/conflicting images in ../dataset to satisfy dataset integrity audits."""

from pathlib import Path

DATASET_ROOT = Path("../dataset").resolve()

FILES_TO_REMOVE = [
    # Cross-split or duplicate in Test/Train
    DATASET_ROOT / "Test" / "angry" / "Screenshot 2026-06-06 24666.png",
    DATASET_ROOT / "Train" / "angry" / "Screenshot 2026-06-06 23499.png",
    DATASET_ROOT / "Train" / "disgust" / "Screenshot 2026-06-06 44650.png",
    DATASET_ROOT / "Train" / "sad" / "Screenshot 2026-06-06 42453.png",
    DATASET_ROOT / "Train" / "sad" / "Screenshot 2026-06-06 44410.png",
    # Conflicting angry vs surprise duplicates
    DATASET_ROOT / "Train" / "angry" / "Screenshot 2026-06-19 115726.png",
    DATASET_ROOT / "Train" / "surprise" / "Screenshot 2026-06-19 115726.png",
    DATASET_ROOT / "Train" / "angry" / "Screenshot 2026-06-19 115733.png",
    DATASET_ROOT / "Train" / "surprise" / "Screenshot 2026-06-19 115733.png",
    DATASET_ROOT / "Train" / "angry" / "Screenshot 2026-06-19 115809.png",
    DATASET_ROOT / "Train" / "surprise" / "Screenshot 2026-06-19 115809.png",
    DATASET_ROOT / "Train" / "angry" / "Screenshot 2026-06-19 115948.png",
    DATASET_ROOT / "Train" / "surprise" / "Screenshot 2026-06-19 115948.png",
]


def clean_duplicates():
    removed = 0
    for p in FILES_TO_REMOVE:
        if p.exists():
            p.unlink()
            print(f"Removed: {p}")
            removed += 1
        else:
            print(f"Not found / already removed: {p.name}")
    print(f"Done. Removed {removed} duplicate/conflicting files.")


if __name__ == "__main__":
    clean_duplicates()
