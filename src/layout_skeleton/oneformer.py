"""Run OneFormer COCO inference and Appendix-A layout remapping."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "src.layout_skeleton"

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .labels import LAYOUT_LABELS, LAYOUT_PALETTE, appendix_layout_lut, label_contract


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class OneFormerError(RuntimeError):
    """Raised when semantic inference or artifact validation fails."""


@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_status(output_dir: Path, state: str, detail: str = "") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _combined_digest(records: list[dict[str, Any]], key: str) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record[key].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_camera(path: Path) -> CameraConfig:
    payload = _read_json(path)
    resolution = payload.get("pinhole_resolution")
    if not isinstance(resolution, list) or len(resolution) != 2:
        raise OneFormerError(f"camera file must contain pinhole_resolution: {path}")
    camera = CameraConfig(width=int(resolution[0]), height=int(resolution[1]))
    if min(camera.width, camera.height) <= 0:
        raise OneFormerError("camera width and height must be positive")
    return camera


def _resolve_manifest_artifact(manifest_path: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_file():
        return path
    # Downloaded manifests may contain source-server absolute paths. Preserve the
    # component-local suffix when possible.
    parts = path.parts
    if "mesh" in parts:
        suffix = Path(*parts[parts.index("mesh") + 1 :])
        candidate = manifest_path.parent / suffix
        if candidate.is_file():
            return candidate
    if not path.is_absolute():
        candidate = manifest_path.parent / path
        if candidate.is_file():
            return candidate
    return path


def _mesh_record(mesh_manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(mesh_manifest_path)
    if manifest.get("status") != "complete":
        raise OneFormerError("mesh manifest is not complete")
    record = manifest["outputs"]["poisson_mesh"]
    path = _resolve_manifest_artifact(mesh_manifest_path, record["path"])
    if not path.is_file() or _sha256(path) != record["sha256"]:
        raise OneFormerError(f"mesh Poisson artifact hash mismatch: {path}")
    resolved = dict(record)
    resolved["path"] = str(path)
    return resolved


def _verify_model_files(model_dir: Path) -> dict[str, dict[str, Any]]:
    required = (
        "config.json",
        "preprocessor_config.json",
        "pytorch_model.bin",
        "coco_panoptic.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    if not model_dir.is_dir():
        raise OneFormerError(f"OneFormer model directory is missing: {model_dir}")
    records: dict[str, dict[str, Any]] = {}
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise OneFormerError(f"OneFormer model file is missing: {path}")
        records[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return records


def _scan_images(images: Path, image_list: Path | None) -> list[dict[str, Any]]:
    if not images.is_dir():
        raise OneFormerError(f"image directory is missing: {images}")
    if image_list is None:
        paths = sorted(
            path
            for path in images.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    else:
        names = [line.strip() for line in image_list.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [images / name for name in names]
    if not paths:
        raise OneFormerError("no input images found")
    records = []
    for index, path in enumerate(paths):
        if not path.is_file():
            raise OneFormerError(f"input image is missing: {path}")
        records.append(
            {
                "index": index,
                "name": path.name,
                "path": str(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def _save_preview(image: Image.Image, layout: np.ndarray, output: Path) -> None:
    image_array = np.asarray(image, dtype=np.uint8)
    color = LAYOUT_PALETTE[layout]
    overlay = np.round(0.55 * image_array + 0.45 * color).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output)


def run_oneformer(
    images: Path,
    output_dir: Path,
    camera: CameraConfig,
    model_dir: Path,
    mesh_manifest: Path,
    image_list: Path | None = None,
    task: str = "semantic",
    preview_count: int = 12,
    random_seed: int = 0,
    command: list[str] | None = None,
) -> Path:
    """Run deterministic OneFormer semantic segmentation for every input frame."""

    images = images.expanduser().resolve()
    output_dir = output_dir.expanduser()
    model_dir = model_dir.expanduser().resolve()
    mesh_manifest = mesh_manifest.expanduser().resolve()
    if output_dir.exists():
        raise OneFormerError(f"OneFormer output already exists: {output_dir}")
    records = _scan_images(images, image_list.expanduser().resolve() if image_list else None)
    mesh = _mesh_record(mesh_manifest)
    model_files = _verify_model_files(model_dir)

    coco_dir = output_dir / "coco_id"
    layout_dir = output_dir / "layout_id"
    preview_dir = output_dir / "previews"
    coco_dir.mkdir(parents=True, exist_ok=False)
    layout_dir.mkdir()
    preview_dir.mkdir()
    _write_status(output_dir, "loading_model", str(model_dir))

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import torch
        from transformers import (
            CLIPTokenizer,
            OneFormerForUniversalSegmentation,
            OneFormerImageProcessor,
            OneFormerProcessor,
        )
    except ImportError as error:
        _write_status(output_dir, "failed", str(error))
        raise OneFormerError("PyTorch and transformers OneFormer support are required") from error
    if not torch.cuda.is_available():
        _write_status(output_dir, "failed", "CUDA is unavailable")
        raise OneFormerError("OneFormer requires CUDA")

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    log_path = output_dir / "inference.log"
    per_image_path = output_dir / "per_image.jsonl"
    labels_path = output_dir / "labels.json"

    try:
        image_processor = OneFormerImageProcessor.from_pretrained(
            model_dir, local_files_only=True, repo_path=str(model_dir)
        )
        tokenizer = CLIPTokenizer.from_pretrained(model_dir, local_files_only=True)
        processor = OneFormerProcessor(image_processor=image_processor, tokenizer=tokenizer)
        model = OneFormerForUniversalSegmentation.from_pretrained(
            model_dir, local_files_only=True
        ).cuda().eval()
        id2label = {int(key): value for key, value in model.config.id2label.items()}
        contract = label_contract(id2label)
        _write_json(labels_path, contract)
        lut = appendix_layout_lut()
        expected_size = (camera.width, camera.height)
        preview_indices = set(
            np.linspace(0, len(records) - 1, min(preview_count, len(records)), dtype=np.int64).tolist()
        )
        coco_histogram = np.zeros(133, dtype=np.int64)
        layout_histogram = np.zeros(len(LAYOUT_LABELS), dtype=np.int64)
        output_records: list[dict[str, Any]] = []
        torch.cuda.reset_peak_memory_stats()
        _write_status(output_dir, "inferencing", f"0/{len(records)}")
        with log_path.open("w", encoding="utf-8") as log, per_image_path.open("w", encoding="utf-8") as handle:
            log.write(f"model_dir: {model_dir}\nframes: {len(records)}\n")
            for index, source in enumerate(records):
                path = Path(source["path"])
                if _sha256(path) != source["sha256"]:
                    raise OneFormerError(f"input image hash changed: {path}")
                with Image.open(path) as loaded:
                    image = loaded.convert("RGB")
                if image.size != expected_size:
                    raise OneFormerError(f"input image dimension changed: {path}")
                frame_start = time.perf_counter()
                inputs = processor(images=image, task_inputs=[task], return_tensors="pt")
                inputs = {key: value.cuda() if hasattr(value, "cuda") else value for key, value in inputs.items()}
                with torch.inference_mode():
                    outputs = model(**inputs)
                semantic = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(expected_size[1], expected_size[0])]
                )[0]
                coco = semantic.to(device="cpu", dtype=torch.uint8).numpy()
                if coco.shape != (expected_size[1], expected_size[0]) or int(coco.max()) >= 133:
                    raise OneFormerError(f"invalid OneFormer output for {path.name}")
                layout = lut[coco]
                stem = path.stem
                coco_path = coco_dir / f"{stem}.png"
                layout_path = layout_dir / f"{stem}.png"
                Image.fromarray(coco, mode="L").save(coco_path)
                Image.fromarray(layout, mode="L").save(layout_path)
                if index in preview_indices:
                    _save_preview(image, layout, preview_dir / f"{stem}.jpg")
                coco_counts = np.bincount(coco.ravel(), minlength=133)
                layout_counts = np.bincount(layout.ravel(), minlength=len(LAYOUT_LABELS))
                coco_histogram += coco_counts
                layout_histogram += layout_counts
                record = {
                    "index": index,
                    "name": path.name,
                    "source_sha256": source["sha256"],
                    "coco_path": str(coco_path),
                    "coco_sha256": _sha256(coco_path),
                    "layout_path": str(layout_path),
                    "layout_sha256": _sha256(layout_path),
                    "coco_histogram": {str(i): int(v) for i, v in enumerate(coco_counts) if v},
                    "layout_histogram": {str(i): int(v) for i, v in enumerate(layout_counts) if v},
                    "elapsed_seconds": round(time.perf_counter() - frame_start, 6),
                }
                handle.write(json.dumps(record, sort_keys=True) + "\n")
                handle.flush()
                output_records.append(record)
                progress = f"{index + 1}/{len(records)}"
                log.write(f"{progress} {path.name} {record['elapsed_seconds']:.6f}s\n")
                log.flush()
                if (index + 1) % 10 == 0 or index + 1 == len(records):
                    _write_status(output_dir, "inferencing", progress)

        manifest = {
            "schema_version": 1,
            "component": "oneformer",
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": random_seed,
            "inputs": {
                "images": records,
                "mesh_manifest": {"path": str(mesh_manifest), "sha256": _sha256(mesh_manifest)},
                "poisson_mesh": mesh,
                "model_files": model_files,
            },
            "algorithm": {
                "implementation": "transformers OneFormerForUniversalSegmentation",
                "checkpoint": "oneformer_coco_swin_large",
                "task": task,
                "semantic_remapping": "HouseLayout3D Appendix A, Table 5",
                "layout_class_count": len(LAYOUT_LABELS),
            },
            "outputs": {
                "coco_id_dir": str(coco_dir),
                "layout_id_dir": str(layout_dir),
                "preview_dir": str(preview_dir),
                "preview_count": len(preview_indices),
                "labels": {"path": str(labels_path), "sha256": _sha256(labels_path)},
                "per_image": {"path": str(per_image_path), "sha256": _sha256(per_image_path)},
                "coco_combined_sha256": _combined_digest(output_records, "coco_sha256"),
                "layout_combined_sha256": _combined_digest(output_records, "layout_sha256"),
            },
            "statistics": {
                "coco_pixel_histogram": {str(i): int(v) for i, v in enumerate(coco_histogram) if v},
                "layout_pixel_histogram_named": {
                    label: int(layout_histogram[i]) for i, label in enumerate(LAYOUT_LABELS)
                },
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "validation": {
                "frame_count": len(output_records),
                "all_input_hashes_match": True,
                "all_output_shapes_match_camera": True,
                "all_coco_ids_in_range": True,
                "all_layout_ids_in_range": True,
                "one_output_pair_per_input": len(output_records) == len(records),
            },
            "environment": {
                "python": platform.python_version(),
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "warnings": [],
        }
        manifest_path = output_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(output_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(output_dir, "failed", str(error))
        if isinstance(error, OneFormerError):
            raise
        raise OneFormerError(str(error)) from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run OneFormer COCO segmentation and HouseLayout3D remapping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--image-list", type=Path)
    parser.add_argument("--mesh-manifest", type=Path, required=True)
    parser.add_argument("--camera", type=Path, default=Path("camera_param.json"))
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="semantic")
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    manifest = run_oneformer(
        images=args.images,
        output_dir=args.output,
        camera=load_camera(args.camera),
        model_dir=args.model_dir,
        mesh_manifest=args.mesh_manifest,
        image_list=args.image_list,
        task=args.task,
        preview_count=args.preview_count,
        random_seed=args.random_seed,
        command=sys.argv,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

