"""Copy existing official splits into a canonical, checked ImageFolder dataset."""

import argparse
import shutil
from pathlib import Path

from .data import scan_folder, split_directories, validate_records, write_json


def parse_source(value):
    name, separator, path = value.partition("=")
    if not separator or not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Use NAME=PATH, for example rafdb=D:/data/rafdb")
    return name.strip(), Path(path).resolve()


def prepare(sources, output, extra_train=(), subject_regex=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Output already exists; choose a new directory: {output}")
    names = [name for name, _ in [*sources, *extra_train]]
    if len(names) != len(set(names)):
        raise ValueError("Each source name must be unique.")
    records = []
    for name, root in [*sources, *extra_train]:
        if output == root or root in output.parents:
            raise ValueError("Prepared output must be outside source directories.")
        if ("ck+" in name.lower() or "ckplus" in name.lower()) and not subject_regex:
            raise ValueError("CK+ requires --subject-regex and subject-disjoint existing splits.")
        folders = {"train": root} if (name, root) in extra_train else split_directories(root)
        for split, folder in folders.items():
            records.extend(scan_folder(folder, split, name, subject_regex))
    classes = validate_records(records)
    # All integrity checks finish before any source is copied. Source files are untouched.
    for index, row in enumerate(records):
        path = Path(row["path"])
        relative = Path(row["split"]) / row["label"] / f"{index:07d}_{row['sha256'][:12]}{path.suffix.lower()}"
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        row["prepared_path"] = relative.as_posix()
    write_json(output / "manifest.json", {"schema_version": 1, "class_names": classes,
               "split_policy": "preserve_existing", "subject_regex": subject_regex,
               "records": records})
    return len(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=parse_source, action="append", required=True,
                        help="NAME=ROOT containing existing train/val/test folders; repeatable")
    parser.add_argument("--extra-train", type=parse_source, action="append", default=[],
                        help="NAME=IMAGEFOLDER of additional training-only images; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subject-regex", help="One capture group identifies subjects, e.g. '(S[0-9]+)'")
    args = parser.parse_args()
    try:
        count = prepare(args.source, args.output, args.extra_train, args.subject_regex)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"Prepared {count} images in {args.output}; official splits preserved.")


if __name__ == "__main__":
    main()
