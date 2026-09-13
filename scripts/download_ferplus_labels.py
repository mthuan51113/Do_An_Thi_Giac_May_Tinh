"""Download and verify Microsoft's public FER+ labels (images are separate)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"
EXPECTED_SHA256 = "9206e20d62f56475939f516847d61753e4860caeac9718b560129541b776fc2c"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data/raw/ferplus/fer2013new.csv"


def download_labels(output: Path) -> dict:
    """Fail closed if upstream content differs from the reviewed label release."""
    if output.exists():
        payload = output.read_bytes()
    else:
        with urlopen(SOURCE_URL, timeout=60) as response:
            payload = response.read(5_000_001)
    digest = hashlib.sha256(payload).hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError("FER+ checksum mismatch; inspect the upstream release before using it.")
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    required = {"Usage", "Image name", "contempt", "disgust", "unknown", "NF"}
    if len(rows) != 35_887 or not required.issubset(rows[0]):
        raise ValueError("Unexpected FER+ row count or column schema.")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists():
        output.write_bytes(payload)
    metadata = {
        "source": SOURCE_URL,
        "sha256": digest,
        "rows": len(rows),
        "bytes": len(payload),
        "contains_images": False,
        "image_source": "https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data",
        "split_counts": {
            usage: sum(row["Usage"] == usage for row in rows)
            for usage in ("Training", "PublicTest", "PrivateTest")
        },
    }
    output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(download_labels(args.output), indent=2))
    print("Labels only. Obtain fer2013.csv from the original image provider before training.")


if __name__ == "__main__":
    main()
