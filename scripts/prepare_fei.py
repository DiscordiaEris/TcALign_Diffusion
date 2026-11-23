"""Convert the original color FEI photographs to the paired training layout."""

import argparse
import csv
import random
import re
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


# Target directions describe the direction the nose points in the image.
FRONTAL = 11
GROUPS = {
    "pose": {
        FRONTAL: "Turn the head to face straight forward.",
        3: "Turn the head toward the left side of the image.",
        4: "Turn the head slightly toward the left side of the image.",
        7: "Turn the head slightly toward the right side of the image.",
        8: "Turn the head toward the right side of the image.",
    },
    "expression": {
        FRONTAL: "Give the face a neutral expression.",
        12: "Make the person smile.",
    },
    "lighting": {
        FRONTAL: "Use bright, even lighting on the face.",
        13: "Dim the lighting on the face.",
        14: "Make the lighting on the face very dim.",
    },
}


def prepare(archive_path, output_dir, seed, validation_subjects):
    with ZipFile(archive_path) as archive:
        photos = {}
        for entry in archive.namelist():
            name = PurePosixPath(entry).name
            match = re.fullmatch(r"(\d+)-(\d\d)\.jpg", name, re.IGNORECASE)
            if not match:
                continue
            key = tuple(map(int, match.groups()))
            if key in photos:
                raise ValueError(f"Duplicate FEI photo index: {key}")
            photos[key] = entry

        subjects = sorted({subject for subject, _ in photos})
        if not 0 < validation_subjects < len(subjects):
            raise ValueError(f"Choose a validation size between 1 and {len(subjects) - 1}")
        required_shots = sorted({shot for group in GROUPS.values() for shot in group})
        missing = [(subject, shot) for subject in subjects for shot in required_shots
                   if (subject, shot) not in photos]
        if missing:
            raise ValueError(f"Missing {len(missing)} required original FEI photographs, starting at {missing[0]}")
        random.Random(seed).shuffle(subjects)

        for split, identities in (("train", subjects[validation_subjects:]),
                                  ("val", subjects[:validation_subjects])):
            directory = output_dir / "images" / split
            directory.mkdir(parents=True, exist_ok=True)
            rows = []
            for subject in sorted(identities):
                names = {shot: f"fei-{subject:03d}-{shot:02d}.jpg" for shot in required_shots}
                for shot, name in names.items():
                    path = directory / name
                    if not path.exists():
                        path.write_bytes(archive.read(photos[subject, shot]))
                for group in GROUPS.values():
                    for source in group:
                        for target, text in group.items():
                            if source != target:
                                rows.append((names[source], names[target], text))
            metadata = output_dir / f"metadata_{split}.csv"
            temporary = metadata.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("source", "target", "text"))
                writer.writerows(rows)
            temporary.replace(metadata)
            print(f"{split}: {len(identities)} identities, {len(rows)} photo pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/face_editing"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-subjects", type=int, default=20)
    args = parser.parse_args()
    prepare(args.archive, args.output, args.seed, args.validation_subjects)


if __name__ == "__main__":
    main()
