import csv
import hashlib
import random
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm

from .transforms import align_paired_target, create_rendered_transform, create_transforms


# Short left/right commands follow the same image-coordinate convention as FEI
# metadata: the direction the nose points, rather than the subject's own side.
INSTRUCTION_PARAPHRASES = {
    "Turn the head toward the left side of the image.": (
        "Turn the head left.",
        "Look left.",
        "Rotate the head toward the left side of the image.",
        "Make the face look toward the left side of the image.",
        "Show the head turned toward the image's left side.",
    ),
    "Turn the head slightly toward the left side of the image.": (
        "Turn the head slightly left.",
        "Look slightly toward the left side of the image.",
        "Rotate the head slightly toward the image's left side.",
        "Show the face turned slightly toward the left side of the image.",
    ),
    "Turn the head slightly toward the right side of the image.": (
        "Turn the head slightly right.",
        "Look slightly toward the right side of the image.",
        "Rotate the head slightly toward the image's right side.",
        "Show the face turned slightly toward the right side of the image.",
    ),
    "Turn the head toward the right side of the image.": (
        "Turn the head right.",
        "Look right.",
        "Rotate the head toward the right side of the image.",
        "Make the face look toward the right side of the image.",
        "Show the head turned toward the image's right side.",
    ),
    "Turn the head to face straight forward.": (
        "Turn the head to a frontal pose.",
        "Show a front view of the face.",
        "Rotate the head to face straight ahead.",
        "Make the person look straight ahead.",
    ),
    "Make the person smile.": (
        "Make the face smile.",
        "Give the person a smiling expression.",
        "Add a smile to the face.",
        "Show a smiling face.",
        "Make the person happy.",
    ),
    "Give the face a neutral expression.": (
        "Make the expression neutral.",
        "Show a neutral face.",
        "Give the person a neutral facial expression.",
        "Set the face to a neutral expression.",
    ),
    "Use bright, even lighting on the face.": (
        "Make the lighting bright.",
        "Use bright lighting across the face.",
        "Illuminate the face with bright, even lighting.",
        "Give the face bright and even lighting.",
    ),
    "Dim the lighting on the face.": (
        "Use dim lighting on the face.",
        "Make the face lighting dim.",
        "Give the person dim lighting.",
        "Set the lighting to dim.",
    ),
    "Make the lighting on the face very dim.": (
        "Use very dim lighting on the face.",
        "Make the face lighting very dim.",
        "Give the person very dim lighting.",
        "Set the lighting to very dim.",
    ),
}


# Controlled exposures describe target states for all source exposures.
EXPOSURE_INSTRUCTION_ALIASES = {
    "Use dim lighting on the face.": "Dim the lighting on the face.",
    "Use very dim lighting on the face.": "Make the lighting on the face very dim.",
}


def augment_instruction(text):
    """Vary known paired prompts without inventing new target attributes."""
    alternatives = INSTRUCTION_PARAPHRASES.get(EXPOSURE_INSTRUCTION_ALIASES.get(text, text), ())
    return random.choice(tuple(dict.fromkeys((text, *alternatives)))) if alternatives else text


class FFHQDataset(Dataset):
    """Paired portraits and instructions, with cached 3D reconstructions."""

    def __init__(self, root_dir, transform=None, split="train", precompute_3d=True, config=None,
                 rendered_transform=None):
        self.transform = transform
        self.rendered_transform = rendered_transform
        self.config = config
        self.augment_instructions = split == "train" and getattr(config, "augment_instructions", False)
        root = Path(root_dir)
        self.image_dir = (root / "images" / split).resolve()
        metadata_path = root / f"metadata_{split}.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing paired metadata: {metadata_path} (source,target,text)")
        self.metadata = self._read_metadata(metadata_path)
        # Versioned and split-specific because older caches used a different alignment
        # and renderer, and both splits may contain identically named files.
        self.cache_dir = root / "deep3d_params" / f"aligned_sh_v2_{config.image_size}" / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if precompute_3d:
            self.precompute_3d_params()
        self.metadata = [
            (source, target, text) for source, target, text in self.metadata
            if self._cache_path(source).is_file() and self._cache_path(target).is_file()
        ]
        if not self.metadata:
            raise RuntimeError(f"No pairs in {metadata_path} have valid source and target 3D reconstructions")
        print(f"{split}: {len(self.metadata)} usable edit pairs", flush=True)

    def _read_metadata(self, path):
        rows = []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            source_column = "source" if "source" in (reader.fieldnames or ()) else "image"
            required = {source_column, "target", "text"}
            if not required.issubset(reader.fieldnames or ()):
                raise ValueError(f"{path}: required columns are source,target,text")
            for index, row in enumerate(reader, start=2):
                source, target, text = ((row[key] or "").strip()
                                        for key in (source_column, "target", "text"))
                if not source or not target or not text or source == target:
                    raise ValueError(f"{path}:{index}: expected distinct images and an edit instruction")
                for name in (source, target):
                    file = (self.image_dir / name).resolve()
                    if not file.is_relative_to(self.image_dir) or not file.is_file():
                        raise FileNotFoundError(f"{path}:{index}: invalid image path {name!r}")
                rows.append((source, target, text))
        if not rows:
            raise ValueError(f"No paired samples in {path}")
        return rows

    def _cache_path(self, name):
        source = self.image_dir / name
        stat = source.stat()
        key = hashlib.sha256(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"

    def _load_image(self, name):
        with Image.open(self.image_dir / name) as image:
            rgb = image.convert("RGB")
        return self.transform(rgb) if self.transform else rgb

    def precompute_3d_params(self):
        from utils.deep3d_utils import load_deep3d_model
        from utils.pytorch3d_renderer import PyTorch3DRenderer

        names = sorted({name for source, target, _ in self.metadata for name in (source, target)})
        pending = [(name, self._cache_path(name)) for name in names]
        pending = [(name, path) for name, path in pending if not path.is_file()]
        if not pending:
            return
        processor = load_deep3d_model(self.config)
        renderer = PyTorch3DRenderer(self.config)
        failed = []
        for name, destination in tqdm(pending, desc="Reconstructing paired portraits"):
            image = self._load_image(name).unsqueeze(0)
            try:
                params = processor.process(image)
            except ValueError as error:
                if "No facial landmarks" not in str(error):
                    raise
                failed.append(name)
                continue
            coefficients = np.concatenate([
                params[key].reshape(-1)
                for key in ("shape", "expression", "pose", "texture", "lighting")
            ]).astype(np.float32)
            if not np.isfinite(coefficients).all():
                raise ValueError(f"Non-finite 3D coefficients for {name}")
            rendered = renderer.tensor_to_pil(renderer.render_from_params_dict(params))
            with NamedTemporaryFile(dir=self.cache_dir, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez(handle, params=coefficients, rendered=np.asarray(rendered),
                         alignment=params["alignment"])
            temporary.replace(destination)
        if failed:
            failure_file = self.cache_dir / "failed_landmarks.txt"
            failure_file.write_text("\n".join(failed) + "\n", encoding="utf-8")
            print(f"No trained landmarks for {len(failed)} portraits. See {failure_file}.", flush=True)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        source, target, text = self.metadata[idx]
        if self.augment_instructions:
            text = augment_instruction(text)
        source_cache = self._cache_path(source)
        with np.load(source_cache) as cached:
            params = torch.from_numpy(cached["params"])
            rendered = Image.fromarray(cached["rendered"])
            source_alignment = torch.from_numpy(cached["alignment"])
        with np.load(self._cache_path(target)) as cached:
            target_params = torch.from_numpy(cached["params"])
            target_alignment = torch.from_numpy(cached["alignment"])
        source_image = self._load_image(source)
        target_image = self._load_image(target)
        if getattr(self.config, "align_paired_targets", False):
            target_image = align_paired_target(target_image, source_alignment, target_alignment)
            target_alignment = source_alignment.clone()
        return {
            "image": source_image,
            "target_image": target_image,
            "text": text,
            "image_name": str(source_cache),
            "params": params,
            "source_alignment": source_alignment,
            "target_params": target_params,
            "target_alignment": target_alignment,
            "rendered_image": self.rendered_transform(rendered),
        }


def create_dataset(config, split="train"):
    return FFHQDataset(
        config.data_root,
        transform=create_transforms(config.image_size, split="val"),
        split=split,
        config=config,
        rendered_transform=create_rendered_transform(config.render_image_size),
    )


def create_training_sampler(dataset, config):
    """Optionally give every edit instruction equal probability within an epoch."""
    if not getattr(config, "balance_instructions", False):
        return None
    counts = Counter(text for _, _, text in dataset.metadata)
    weights = [1 / counts[text] for _, _, text in dataset.metadata]
    generator = torch.Generator().manual_seed(config.seed)
    return WeightedRandomSampler(weights, len(dataset), replacement=True, generator=generator)
