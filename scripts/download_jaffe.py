"""Download JAFFE for noncommercial research and prepare optional training images.

Source terms prohibit redistributing the images. Keep raw and prepared files out
of GitHub and the public web. See docs/DATASETS.md for citation and use details.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://zenodo.org/records/14974867"
ARCHIVE_SHA256 = "6da27f5954f969c6f65d782911834dd66827ec3580e79b95073eb6cb93de5c3b"
README_SHA256 = "1a0299861d22830a434e161043c0fd61bb268f6fb758d7f1f1d8217d136c82ef"
CLASS_MAP = {
    "AN": "angry", "DI": "disgust", "FE": "fear", "HA": "happy",
    "NE": "neutral", "SA": "sad", "SU": "surprise",
}


def download_verified(raw_dir: Path, filename: str, expected_hash: str) -> bytes:
    target = raw_dir / filename
    if target.exists():
        payload = target.read_bytes()
    else:
        with urlopen(f"{SOURCE_PAGE}/files/{filename}?download=1", timeout=60) as response:
            payload = response.read(20_000_001)
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"Checksum mismatch for {filename}; inspect the release before use.")
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(payload)
    return payload


def prepare_jaffe(raw_dir: Path, output: Path) -> dict:
    """Retain subject identifiers; never generate a new test split implicitly."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty to avoid mixing datasets: {output}")
    download_verified(raw_dir, "README_FIRST.txt", README_SHA256)
    archive = download_verified(raw_dir, "jaffe.zip", ARCHIVE_SHA256)
    records = []
    with ZipFile(io.BytesIO(archive)) as zipped:
        members = [item for item in zipped.infolist() if item.filename.lower().endswith(".tiff")]
        if len(members) != 213:
            raise ValueError("Unexpected JAFFE image count.")
        for member in sorted(members, key=lambda item: item.filename):
            name = Path(member.filename).name
            subject, pose, *_ = name.split(".")
            label = CLASS_MAP[pose[:2]]
            relative_path = Path("Train") / label / f"jaffe_{Path(name).stem}.png"
            destination = output / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(io.BytesIO(zipped.read(member))) as image:
                image.convert("RGB").save(destination)
            records.append({
                "path": relative_path.as_posix(), "class": label,
                "subject_id": f"jaffe:{subject}", "source_file": name,
                "source": SOURCE_PAGE, "role": "supplemental_train_only",
            })
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metadata = {
        "source": SOURCE_PAGE, "archive_sha256": ARCHIVE_SHA256,
        "images": len(records), "subjects": len({row["subject_id"] for row in records}),
        "class_counts": dict(sorted(Counter(row["class"] for row in records).items())),
        "role": "supplemental_train_only", "missing_classes": ["contempt"],
        "redistribution_allowed": False,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPO_ROOT / "data/raw/jaffe")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data/supplemental/jaffe")
    args = parser.parse_args()
    print(json.dumps(prepare_jaffe(args.raw_dir, args.output), indent=2))


if __name__ == "__main__":
    main()
