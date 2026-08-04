"""OneFormer COCO semantic inference and Appendix-A layout remapping."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import PipelineConfig
from .stages import Stage


class OneFormerStageError(RuntimeError):
    """Raised when semantic inference or artifact validation fails."""


LAYOUT_LABELS = (
    "wall",
    "ceiling",
    "floor",
    "surface",
    "inaccurate_window",
    "inaccurate_mirror",
    "inaccurate_outdoor",
    "stairs",
    "object",
)

# Table 7 of Appendix A. Everything not listed explicitly is an object.
APPENDIX_COCO_IDS: dict[str, frozenset[int]] = {
    "wall": frozenset({109, 110, 111, 112, 131}),
    "ceiling": frozenset({118}),
    "floor": frozenset({87, 122, 132}),
    "surface": frozenset({85, 86, 114, 120}),
    "inaccurate_window": frozenset({115}),
    "inaccurate_mirror": frozenset({93}),
    "inaccurate_outdoor": frozenset({90, 116, 119, 123, 125, 126}),
    "stairs": frozenset({106}),
}

LAYOUT_PALETTE = np.asarray(
    [
        [210, 68, 68],
        [90, 150, 230],
        [80, 185, 105],
        [238, 180, 70],
        [75, 205, 215],
        [205, 105, 220],
        [130, 130, 130],
        [245, 105, 30],
        [150, 105, 70],
    ],
    dtype=np.uint8,
)


def appendix_layout_lut() -> np.ndarray:
    """Return the exact 133-to-9 class remapping from Appendix A, Table 7."""

    lut = np.full(133, LAYOUT_LABELS.index("object"), dtype=np.uint8)
    assigned: set[int] = set()
    for layout_id, label in enumerate(LAYOUT_LABELS[:-1]):
        ids = APPENDIX_COCO_IDS[label]
        if assigned.intersection(ids):
            raise AssertionError("Appendix-A COCO mapping contains overlapping IDs")
        lut[list(ids)] = layout_id
        assigned.update(ids)
    return lut


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_digest(records: list[dict[str, Any]], key: str) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["name"].encode("utf-8"))
        digest.update(b"\0")
        digest.update(record[key].encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_status(stage_dir: Path, state: str, detail: str = "") -> None:
    _write_json(
        stage_dir / "STATUS.json",
        {
            "state": state,
            "detail": detail,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _verify_inputs(
    config: PipelineConfig, run_id: str
) -> tuple[list[dict[str, Any]], Path, dict[str, Any]]:
    run_dir = config.storage.outputs / config.scene / run_id
    input_manifest_path = run_dir / Stage.INPUT.value / "manifest.json"
    mesh_manifest_path = run_dir / Stage.MESH.value / "manifest.json"
    if not input_manifest_path.is_file() or not mesh_manifest_path.is_file():
        raise OneFormerStageError("completed 00_input and 04_mesh manifests are required")
    input_manifest = _read_json(input_manifest_path)
    mesh_manifest = _read_json(mesh_manifest_path)
    if input_manifest.get("status") != "complete":
        raise OneFormerStageError("00_input manifest is not complete")
    if mesh_manifest.get("status") != "complete":
        raise OneFormerStageError("04_mesh manifest is not complete")

    image_list = Path(input_manifest["outputs"]["image_list"])
    if not image_list.is_file():
        raise OneFormerStageError(f"frozen image list is missing: {image_list}")
    if _sha256(image_list) != input_manifest["outputs"]["image_list_sha256"]:
        raise OneFormerStageError("00_input image-list hash no longer matches")
    names = image_list.read_text(encoding="utf-8").splitlines()
    records = input_manifest.get("images", [])
    if names != [record.get("name") for record in records]:
        raise OneFormerStageError("00_input image records no longer match images.txt")

    mesh_record = mesh_manifest["outputs"]["poisson_mesh"]
    mesh_path = Path(mesh_record["path"])
    if not mesh_path.is_file() or _sha256(mesh_path) != mesh_record["sha256"]:
        raise OneFormerStageError("04_mesh Poisson artifact hash no longer matches")
    return records, mesh_manifest_path, mesh_record


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
        raise OneFormerStageError(f"OneFormer model directory is missing: {model_dir}")
    records: dict[str, dict[str, Any]] = {}
    for name in required:
        path = model_dir / name
        if not path.is_file():
            raise OneFormerStageError(f"OneFormer model file is missing: {path}")
        records[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return records


def _label_contract(id2label: dict[int, str]) -> dict[str, Any]:
    if set(id2label) != set(range(133)):
        raise OneFormerStageError("OneFormer checkpoint does not expose COCO's 133 IDs")
    expected = {
        85: "curtain",
        86: "door-stuff",
        87: "floor-wood",
        90: "gravel",
        93: "mirror-stuff",
        106: "stairs",
        109: "wall-brick",
        110: "wall-stone",
        111: "wall-tile",
        112: "wall-wood",
        114: "window-blind",
        115: "window-other",
        116: "tree-merged",
        118: "ceiling-merged",
        119: "sky-other-merged",
        120: "cabinet-merged",
        122: "floor-other-merged",
        123: "pavement-merged",
        125: "grass-merged",
        126: "dirt-merged",
        131: "wall-other-merged",
        132: "rug-merged",
    }
    mismatches = {
        str(class_id): {"expected": name, "actual": id2label[class_id]}
        for class_id, name in expected.items()
        if id2label[class_id] != name
    }
    if mismatches:
        raise OneFormerStageError(f"COCO label contract mismatch: {mismatches}")

    lut = appendix_layout_lut()
    coco_labels = []
    for class_id in range(133):
        layout_id = int(lut[class_id])
        coco_labels.append(
            {
                "id": class_id,
                "name": id2label[class_id],
                "layout_id": layout_id,
                "layout_name": LAYOUT_LABELS[layout_id],
            }
        )
    return {
        "source": "HouseLayout3D Appendix A, Table 7",
        "coco_label_count": 133,
        "layout_labels": [
            {
                "id": layout_id,
                "name": name,
                "color_rgb": LAYOUT_PALETTE[layout_id].tolist(),
                "coco_ids": [
                    entry["id"]
                    for entry in coco_labels
                    if entry["layout_id"] == layout_id
                ],
            }
            for layout_id, name in enumerate(LAYOUT_LABELS)
        ],
        "coco_labels": coco_labels,
    }


def _save_preview(image: Image.Image, layout: np.ndarray, output: Path) -> None:
    image_array = np.asarray(image, dtype=np.uint8)
    color = LAYOUT_PALETTE[layout]
    overlay = np.round(0.55 * image_array + 0.45 * color).astype(np.uint8)
    Image.fromarray(overlay, mode="RGB").save(output)


def run_oneformer(
    config: PipelineConfig,
    run_id: str,
    command: list[str] | None = None,
) -> Path:
    """Run deterministic OneFormer semantic segmentation for every frozen frame."""

    image_records, mesh_manifest_path, mesh_record = _verify_inputs(config, run_id)
    model_files = _verify_model_files(config.oneformer.model_dir)
    stage_dir = (
        config.storage.outputs / config.scene / run_id / Stage.ONEFORMER.value
    )
    if stage_dir.exists():
        raise OneFormerStageError(
            f"OneFormer stage already exists and will not be overwritten: {stage_dir}"
        )
    coco_dir = stage_dir / "coco_id"
    layout_dir = stage_dir / "layout_id"
    preview_dir = stage_dir / "previews"
    coco_dir.mkdir(parents=True, exist_ok=False)
    layout_dir.mkdir()
    preview_dir.mkdir()
    _write_status(stage_dir, "loading_model", str(config.oneformer.model_dir))

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(config.runtime.preferred_gpu))
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
        _write_status(stage_dir, "failed", str(error))
        raise OneFormerStageError(
            "PyTorch and transformers OneFormer support are required"
        ) from error

    if not torch.cuda.is_available():
        _write_status(stage_dir, "failed", "CUDA is unavailable")
        raise OneFormerStageError("OneFormer stage requires CUDA")
    random.seed(config.runtime.random_seed)
    np.random.seed(config.runtime.random_seed)
    torch.manual_seed(config.runtime.random_seed)
    torch.cuda.manual_seed_all(config.runtime.random_seed)

    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    log_path = stage_dir / "inference.log"
    records_path = stage_dir / "per_image.jsonl"
    labels_path = stage_dir / "labels.json"
    try:
        image_processor = OneFormerImageProcessor.from_pretrained(
            config.oneformer.model_dir,
            local_files_only=True,
            repo_path=str(config.oneformer.model_dir),
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            config.oneformer.model_dir, local_files_only=True
        )
        processor = OneFormerProcessor(
            image_processor=image_processor, tokenizer=tokenizer
        )
        model = OneFormerForUniversalSegmentation.from_pretrained(
            config.oneformer.model_dir, local_files_only=True
        ).cuda().eval()
        id2label = {int(key): value for key, value in model.config.id2label.items()}
        label_contract = _label_contract(id2label)
        _write_json(labels_path, label_contract)
        lut = appendix_layout_lut()
        expected_size = (config.input.camera.width, config.input.camera.height)
        coco_histogram = np.zeros(133, dtype=np.int64)
        layout_histogram = np.zeros(len(LAYOUT_LABELS), dtype=np.int64)
        output_records: list[dict[str, Any]] = []
        count = len(image_records)
        preview_indices = (
            set(
                np.linspace(
                    0,
                    count - 1,
                    min(config.oneformer.preview_count, count),
                    dtype=np.int64,
                ).tolist()
            )
            if count
            else set()
        )
        torch.cuda.reset_peak_memory_stats()
        _write_status(stage_dir, "inferencing", f"0/{count}")
        with log_path.open("w", encoding="utf-8") as log, records_path.open(
            "w", encoding="utf-8"
        ) as records_file:
            log.write(f"model_dir: {config.oneformer.model_dir}\n")
            log.write(f"frames: {count}\n")
            log.flush()
            for index, source_record in enumerate(image_records):
                image_path = Path(source_record["path"])
                if not image_path.is_file():
                    raise OneFormerStageError(f"input image is missing: {image_path}")
                if _sha256(image_path) != source_record["sha256"]:
                    raise OneFormerStageError(
                        f"input image hash no longer matches: {image_path}"
                    )
                with Image.open(image_path) as loaded:
                    image = loaded.convert("RGB")
                if image.size != expected_size:
                    raise OneFormerStageError(
                        f"input image dimension changed: {image_path}"
                    )
                frame_start = time.perf_counter()
                inputs = processor(
                    images=image,
                    task_inputs=[config.oneformer.task],
                    return_tensors="pt",
                )
                inputs = {
                    key: value.cuda() if hasattr(value, "cuda") else value
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    outputs = model(**inputs)
                semantic = processor.post_process_semantic_segmentation(
                    outputs, target_sizes=[(expected_size[1], expected_size[0])]
                )[0]
                coco = semantic.to(device="cpu", dtype=torch.uint8).numpy()
                if coco.shape != (expected_size[1], expected_size[0]):
                    raise OneFormerStageError(
                        f"OneFormer output has the wrong shape for {image_path.name}"
                    )
                if int(coco.max()) >= 133:
                    raise OneFormerStageError("OneFormer output contains a non-COCO ID")
                layout = lut[coco]
                stem = image_path.stem
                coco_path = coco_dir / f"{stem}.png"
                layout_path = layout_dir / f"{stem}.png"
                Image.fromarray(coco, mode="L").save(coco_path)
                Image.fromarray(layout, mode="L").save(layout_path)
                if index in preview_indices:
                    _save_preview(image, layout, preview_dir / f"{stem}.jpg")
                coco_counts = np.bincount(coco.ravel(), minlength=133)
                layout_counts = np.bincount(
                    layout.ravel(), minlength=len(LAYOUT_LABELS)
                )
                coco_histogram += coco_counts
                layout_histogram += layout_counts
                record = {
                    "index": index,
                    "name": image_path.name,
                    "source_sha256": source_record["sha256"],
                    "coco_path": str(coco_path),
                    "coco_sha256": _sha256(coco_path),
                    "layout_path": str(layout_path),
                    "layout_sha256": _sha256(layout_path),
                    "coco_histogram": {
                        str(class_id): int(value)
                        for class_id, value in enumerate(coco_counts)
                        if value
                    },
                    "layout_histogram": {
                        str(class_id): int(value)
                        for class_id, value in enumerate(layout_counts)
                        if value
                    },
                    "elapsed_seconds": round(time.perf_counter() - frame_start, 6),
                }
                records_file.write(json.dumps(record, sort_keys=True) + "\n")
                records_file.flush()
                output_records.append(record)
                progress = f"{index + 1}/{count}"
                log.write(
                    f"{progress} {image_path.name} "
                    f"{record['elapsed_seconds']:.6f}s\n"
                )
                log.flush()
                if (index + 1) % 10 == 0 or index + 1 == count:
                    _write_status(stage_dir, "inferencing", progress)

        finished_at = datetime.now(timezone.utc)
        manifest = {
            "schema_version": 1,
            "scene": config.scene,
            "run_id": run_id,
            "stage": Stage.ONEFORMER.value,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "finished_at": finished_at.isoformat(),
            "elapsed_seconds": round(time.perf_counter() - start, 6),
            "command": command if command is not None else sys.argv,
            "random_seed": config.runtime.random_seed,
            "inputs": {
                "image_count": len(image_records),
                "mesh_manifest": {
                    "path": str(mesh_manifest_path),
                    "sha256": _sha256(mesh_manifest_path),
                },
                "poisson_mesh": mesh_record,
                "model_files": model_files,
            },
            "algorithm": {
                "implementation": "transformers OneFormerForUniversalSegmentation",
                "checkpoint": "oneformer_coco_swin_large",
                "task": config.oneformer.task,
                "processor_loading": "offline local image-processor and CLIP tokenizer",
                "semantic_remapping": "HouseLayout3D Appendix A, Table 7",
                "coco_class_count": 133,
                "layout_class_count": len(LAYOUT_LABELS),
            },
            "outputs": {
                "coco_id_dir": str(coco_dir),
                "layout_id_dir": str(layout_dir),
                "preview_dir": str(preview_dir),
                "preview_count": len(preview_indices),
                "labels": {
                    "path": str(labels_path),
                    "sha256": _sha256(labels_path),
                },
                "per_image": {
                    "path": str(records_path),
                    "sha256": _sha256(records_path),
                },
                "coco_combined_sha256": _combined_digest(
                    output_records, "coco_sha256"
                ),
                "layout_combined_sha256": _combined_digest(
                    output_records, "layout_sha256"
                ),
            },
            "statistics": {
                "coco_pixel_histogram": {
                    str(class_id): int(value)
                    for class_id, value in enumerate(coco_histogram)
                    if value
                },
                "layout_pixel_histogram": {
                    str(class_id): int(value)
                    for class_id, value in enumerate(layout_histogram)
                },
                "layout_pixel_histogram_named": {
                    label: int(layout_histogram[index])
                    for index, label in enumerate(LAYOUT_LABELS)
                },
                "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
            },
            "validation": {
                "frame_count": len(output_records),
                "all_input_hashes_match": True,
                "all_output_shapes_match_camera": True,
                "all_coco_ids_in_range": True,
                "all_layout_ids_in_range": True,
                "one_output_pair_per_input": len(output_records) == len(image_records),
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
        manifest_path = stage_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        _write_status(stage_dir, "complete", str(manifest_path))
        return manifest_path
    except Exception as error:
        _write_status(stage_dir, "failed", str(error))
        if isinstance(error, OneFormerStageError):
            raise
        raise OneFormerStageError(str(error)) from error
