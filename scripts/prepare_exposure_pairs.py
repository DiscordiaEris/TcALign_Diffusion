"""Derive controlled global-exposure pairs without changing the original dataset.

This synthesizes RGB gain changes, not physical relighting. Original photographs
and valid reconstruction caches are hardlinked; use an output on the same file
system. Synthetic PNGs and caches are new files. Run explicitly with --root and
--output; importing this module does not prepare data.
"""

import argparse
import csv
import errno
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF


EXPOSURES = (
    ("bright", 1.0, "Use bright, even lighting on the face."),
    ("dim", 0.6, "Use dim lighting on the face."),
    ("very_dim", 0.3, "Use very dim lighting on the face."),
)
PHOTO_NAME = re.compile(
    r"(?P<identity>fei-\d+)-(?P<shot>\d{2})\.(?:jpg|jpeg|png)", re.IGNORECASE
)
LIGHTING_WORDS = re.compile(
    r"\b(?:bright|dim|dark|lighting|illumination|light|lit|relight|shadow|studio|natural|soft)\b",
    re.IGNORECASE,
)


def _cache_path(root, split, name, image_size):
    photo = root / "images" / split / name
    stat = photo.stat()
    key = hashlib.sha256(
        f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode()
    ).hexdigest()
    return root / "deep3d_params" / f"aligned_sh_v2_{image_size}" / split / f"{key}.npz"


def _load_cache(path):
    if not path.is_file():
        raise ValueError("missing_cache")
    try:
        with np.load(path, allow_pickle=False) as archive:
            cached = {key: archive[key].copy() for key in archive.files}
        params, alignment, rendered = (
            cached[key] for key in ("params", "alignment", "rendered")
        )
        if (
            params.shape != (254,)
            or not np.issubdtype(params.dtype, np.floating)
            or not np.isfinite(params).all()
        ):
            raise ValueError("expected 254 finite floating-point coefficients")
        if (
            alignment.shape != (2, 3)
            or not np.isfinite(alignment).all()
            or abs(np.linalg.det(alignment[:, :2])) < 1e-8
        ):
            raise ValueError("expected a finite invertible 2x3 alignment")
        if (
            rendered.ndim != 3
            or rendered.shape[-1] != 3
            or min(rendered.shape[:2]) < 1
            or rendered.dtype != np.uint8
        ):
            raise ValueError("expected a uint8 RGB render")
        return cached
    except (OSError, ValueError, KeyError, EOFError, BadZipFile) as error:
        raise ValueError(f"invalid_cache: {error}") from error


def _hardlink(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Linking the resolved file keeps the new image inside its image_dir.
        # FFHQDataset rejects outside-pointing image symlinks after resolve().
        os.link(source.resolve(), destination)
    except OSError as error:
        if error.errno == errno.EXDEV:
            raise ValueError(
                "Output must share a filesystem with the source for hardlinks"
            ) from error
        raise


def _photo(path, image_size):
    with Image.open(path) as original:
        image = TF.center_crop(
            TF.resize(original.convert("RGB"), image_size), image_size
        )
    return np.asarray(image).copy()


def _face_luminance(pixels, alignment):
    """Mean RGB luminance in the same 64px, border-padded crop as exposure loss."""
    image = torch.from_numpy(pixels.astype(np.float32) / 255).permute(2, 0, 1)[None]
    with torch.no_grad():
        grid = F.affine_grid(
            torch.as_tensor(alignment, dtype=torch.float32)[None],
            (1, 3, 64, 64),
            align_corners=False,
        )
        crop = F.grid_sample(
            image, grid, mode="bilinear", padding_mode="border", align_corners=False
        )
        weights = crop.new_tensor([0.2126, 0.7152, 0.0722])[None, :, None, None]
        return float((crop * weights).sum(dim=1).mean())


def _metadata(root, split):
    path = root / f"metadata_{split}.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        source_key = "source" if "source" in fields else "image"
        if not {source_key, "target", "text"}.issubset(fields):
            raise ValueError(f"{path}: expected source,target,text columns")
        rows = list(reader)
    for row in rows:
        if not all(row.get(key) for key in (source_key, "target", "text")):
            raise ValueError(f"{path}: empty source, target, or text")
    return fields, source_key, rows


def prepare(root, output, *, image_size=512):
    """Build a new dataset and return its persisted provenance manifest.

    No detector/model is run. Missing or invalid source caches skip synthetic
    pairs; non-lighting metadata is retained, including rows the loader may
    subsequently filter for missing reconstructions.
    """
    root, output = Path(root).resolve(), Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    output = output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the original dataset")
    if image_size < 1:
        raise ValueError("image_size must be positive")

    inputs = {}
    subjects = {}
    for split in ("train", "val"):
        directory = root / "images" / split
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        fields, source_key, rows = _metadata(root, split)
        photos = sorted(path for path in directory.rglob("*") if path.is_file())
        identities = {}
        names = {path.relative_to(directory).as_posix() for path in photos}
        for row in rows:
            if any(row[key].strip() not in names for key in (source_key, "target")):
                raise ValueError(
                    f"{split}: metadata references an absent original image"
                )
        for path in photos:
            match = PHOTO_NAME.fullmatch(path.name)
            if match:
                identity, shot = match.group("identity").lower(), int(
                    match.group("shot")
                )
                shots = identities.setdefault(identity, {})
                if shot in shots:
                    raise ValueError(
                        f"Duplicate {split} image for {identity} shot {shot}"
                    )
                shots[shot] = path.relative_to(directory).as_posix()
        subjects[split] = set(identities)
        inputs[split] = (fields, source_key, rows, photos, identities)
    if subjects["train"] & subjects["val"]:
        raise ValueError("Original train/val identity sets overlap")

    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(root),
        "output_root": str(output),
        "image_size": image_size,
        "complete": False,
        "method": {
            "description": "Synthetic global RGB exposure gain; not physical relighting",
            "reference": "Same identity shot 11, aligned 64x64 crop mean RGB luminance",
            "luminance_weights": [0.2126, 0.7152, 0.0722],
            "states": {state: factor for state, factor, _ in EXPOSURES},
            "gain": "reference_face_luminance * state_factor / source_face_luminance",
            "pixels": "Resize/center crop then multiply RGB [0,1] by gain, clip, round to uint8 PNG",
            "coefficients": "Copy source; lighting = (gamma + ambient) * gain - ambient",
            "ambient": {
                "indices": [0, 9, 18],
                "value": 0.8,
                "implementation": "Deep3DFR/models/bfm.py:compute_color, reshape RGB x 9",
            },
            "rendered_cache": "Approximate: source rendered uint8 RGB multiplied by gain and clipped",
            "original_files": "Hardlinks; original image bytes and valid caches are reused",
            "lighting_metadata_detection": LIGHTING_WORDS.pattern,
        },
        "splits": {},
        "original_files": [],
        "pairs": [],
        "skipped": [],
    }
    ambient = np.zeros(27, dtype=np.float32)
    ambient[[0, 9, 18]] = 0.8
    for split, (fields, source_key, rows, photos, identities) in inputs.items():
        image_dir = output / "images" / split
        cache_dir = output / "deep3d_params" / f"aligned_sh_v2_{image_size}" / split
        image_dir.mkdir(parents=True)
        cache_dir.mkdir(parents=True)
        valid = {}
        invalid = {}
        for photo in photos:
            name = photo.relative_to(root / "images" / split).as_posix()
            destination = image_dir / name
            _hardlink(photo, destination)
            cache = _cache_path(root, split, name, image_size)
            stat = photo.stat()
            record = {
                "split": split,
                "name": name,
                "source": str(photo),
                "size_bytes": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(photo.read_bytes()).hexdigest(),
            }
            try:
                _load_cache(cache)
            except ValueError as error:
                invalid[name] = str(error)
                record["cache_status"] = str(error)
            else:
                linked = _cache_path(output, split, name, image_size)
                _hardlink(cache, linked)
                valid[name] = linked
                record.update(
                    cache_status="linked",
                    source_cache=str(cache),
                    cache=linked.relative_to(output).as_posix(),
                    cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                )
            manifest["original_files"].append(record)

        retained = [row for row in rows if not LIGHTING_WORDS.search(row["text"])]
        generated = []
        for identity, shots in sorted(identities.items()):
            reference = shots.get(11)
            reference_mean = None
            if reference in valid:
                reference_cache = _load_cache(valid[reference])
                reference_mean = _face_luminance(
                    _photo(image_dir / reference, image_size),
                    reference_cache["alignment"],
                )
            for shot in (11, 13, 14):
                source = shots.get(shot)
                if source is None:
                    manifest["skipped"].append(
                        {
                            "split": split,
                            "identity": identity,
                            "source": None,
                            "shot": shot,
                            "reason": "missing_source_image",
                        }
                    )
                    continue
                reason = None
                if source not in valid:
                    reason = invalid[source]
                elif reference_mean is None:
                    reason = "missing_or_invalid_reference_cache"
                elif reference_mean <= 1e-6:
                    reason = "black_reference_crop"
                if reason:
                    manifest["skipped"].append(
                        {
                            "split": split,
                            "identity": identity,
                            "source": source,
                            "reason": reason,
                        }
                    )
                    continue
                source_cache = _load_cache(valid[source])
                pixels = _photo(image_dir / source, image_size)
                source_mean = _face_luminance(pixels, source_cache["alignment"])
                if source_mean <= 1e-6:
                    manifest["skipped"].append(
                        {
                            "split": split,
                            "identity": identity,
                            "source": source,
                            "reason": "black_source_crop",
                        }
                    )
                    continue
                for state, factor, text in EXPOSURES:
                    if shot == 11 and state == "bright":
                        continue
                    desired = reference_mean * factor
                    gain = desired / source_mean
                    floating = pixels.astype(np.float32) * gain
                    synthetic = np.rint(floating).clip(0, 255).astype(np.uint8)
                    name = (
                        Path("synthetic_exposure") / f"{Path(source).stem}__{state}.png"
                    ).as_posix()
                    destination = image_dir / name
                    destination.parent.mkdir(exist_ok=True)
                    if destination.exists():
                        raise FileExistsError(
                            f"Synthetic target name collides with an original: {destination}"
                        )
                    Image.fromarray(synthetic).save(destination, format="PNG")
                    target_cache = {
                        key: value.copy() for key, value in source_cache.items()
                    }
                    target_cache["params"][-27:] = (
                        source_cache["params"][-27:] + ambient
                    ) * gain - ambient
                    target_cache["rendered"] = (
                        np.rint(source_cache["rendered"].astype(np.float32) * gain)
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
                    cached_target = _cache_path(output, split, name, image_size)
                    np.savez(cached_target, **target_cache)
                    row = {field: "" for field in fields}
                    row.update({source_key: source, "target": name, "text": text})
                    generated.append(row)
                    manifest["pairs"].append(
                        {
                            "split": split,
                            "identity": identity,
                            "source": source,
                            "target": name,
                            "state": state,
                            "text": text,
                            "reference": reference,
                            "gain": gain,
                            "reference_face_luminance": reference_mean,
                            "source_face_luminance": source_mean,
                            "requested_target_face_luminance": desired,
                            "actual_target_face_luminance": _face_luminance(
                                synthetic, source_cache["alignment"]
                            ),
                            "clipped_channel_fraction": float((floating > 255).mean()),
                            "source_cache": valid[source]
                            .relative_to(output)
                            .as_posix(),
                            "target_cache": cached_target.relative_to(
                                output
                            ).as_posix(),
                            "target_sha256": hashlib.sha256(
                                destination.read_bytes()
                            ).hexdigest(),
                        }
                    )
        with (output / f"metadata_{split}.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(retained + generated)
        manifest["splits"][split] = {
            "identities": sorted(identities),
            "original_images": len(photos),
            "linked_valid_caches": len(valid),
            "invalid_or_missing_caches": len(invalid),
            "original_pairs": len(rows),
            "retained_nonlighting_pairs": len(retained),
            "removed_lighting_pairs": len(rows) - len(retained),
            "synthetic_pairs": len(generated),
            "output_pairs": len(retained) + len(generated),
            "source_metadata_sha256": hashlib.sha256(
                (root / f"metadata_{split}.csv").read_bytes()
            ).hexdigest(),
        }
    manifest["complete"] = True
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Existing paired dataset; read only"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory on the same filesystem",
    )
    parser.add_argument("--image-size", type=int, default=512)
    args = parser.parse_args(argv)
    report = prepare(args.root, args.output, image_size=args.image_size)
    for split, stats in report["splits"].items():
        print(
            f"{split}: retained {stats['retained_nonlighting_pairs']} non-lighting pairs; "
            f"replaced {stats['removed_lighting_pairs']} old lighting pairs with {stats['synthetic_pairs']} synthetic pairs"
        )
    print(f"Provenance: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
