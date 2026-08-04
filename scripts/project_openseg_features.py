#!/usr/bin/env python3
"""Fuse sampled OpenSeg pixel features onto prototype triangles.

This script intentionally runs in the isolated TensorFlow/OpenSeg environment.
The layout process prepares only visible image coordinates, so full 640x640x768
feature maps never leave GPU memory or get written to disk.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--text-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-interval", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.checkpoint_interval <= 0:
        raise SystemExit("--checkpoint-interval must be positive")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    request = np.load(args.request, allow_pickle=False)
    image_paths = request["image_paths"].astype(str)
    frame_offsets = request["frame_offsets"].astype(np.int64)
    triangle_indices = request["triangle_indices"].astype(np.int64)
    image_rows = request["image_rows"].astype(np.float32)
    image_columns = request["image_columns"].astype(np.float32)
    triangle_count = int(request["triangle_count"])
    if len(frame_offsets) != len(image_paths) + 1:
        raise SystemExit("request frame offsets do not match image paths")
    if not (
        len(triangle_indices) == len(image_rows) == len(image_columns)
        == int(frame_offsets[-1])
    ):
        raise SystemExit("request sample arrays have inconsistent lengths")

    text_features = np.load(args.text_features, allow_pickle=False).astype(np.float32)
    if text_features.ndim != 2 or text_features.shape[1] != 768:
        raise SystemExit("text features must have shape (classes, 768)")
    text_tensor = tf.convert_to_tensor(text_features[None], dtype=tf.float32)

    model = tf.saved_model.load(str(args.model), tags=[tf.saved_model.SERVING])
    signature = model.signatures["serving_default"]

    feature_sums = np.zeros((triangle_count, 768), dtype=np.float32)
    feature_counts = np.zeros(triangle_count, dtype=np.uint32)
    first_frame = 0
    if args.resume and args.checkpoint.is_file():
        checkpoint = np.load(args.checkpoint, allow_pickle=False)
        feature_sums = checkpoint["feature_sums"].astype(np.float32)
        feature_counts = checkpoint["feature_counts"].astype(np.uint32)
        first_frame = int(checkpoint["next_frame"])
        if feature_sums.shape != (triangle_count, 768):
            raise SystemExit("checkpoint triangle shape differs from request")

    started = time.perf_counter()
    processed_samples = int(feature_counts.sum())
    for frame_index in range(first_frame, len(image_paths)):
        begin = int(frame_offsets[frame_index])
        end = int(frame_offsets[frame_index + 1])
        if end > begin:
            image_path = Path(image_paths[frame_index])
            result = signature(
                inp_image_bytes=tf.io.read_file(str(image_path)),
                inp_text_emb=text_tensor,
            )
            image_info = result["image_info"].numpy()
            scale_y, scale_x = image_info[2]
            offset_y, offset_x = image_info[3]
            crop_height = max(1, int(round(image_info[0, 0] * scale_y)))
            crop_width = max(1, int(round(image_info[0, 1] * scale_x)))
            rows = np.rint(image_rows[begin:end] * scale_y + offset_y).astype(np.int32)
            columns = np.rint(
                image_columns[begin:end] * scale_x + offset_x
            ).astype(np.int32)
            rows = np.clip(rows, 0, crop_height - 1)
            columns = np.clip(columns, 0, crop_width - 1)
            gather_indices = tf.convert_to_tensor(
                np.column_stack(
                    (np.zeros(len(rows), dtype=np.int32), rows, columns)
                ),
                dtype=tf.int32,
            )
            features = tf.gather_nd(
                result["ppixel_ave_feat"], gather_indices
            ).numpy().astype(np.float32)
            target_triangles = triangle_indices[begin:end]
            np.add.at(feature_sums, target_triangles, features)
            np.add.at(feature_counts, target_triangles, 1)
            processed_samples += len(target_triangles)

        next_frame = frame_index + 1
        elapsed = time.perf_counter() - started
        progress = {
            "state": "running",
            "frame": next_frame,
            "total_frames": len(image_paths),
            "percent": round(100.0 * next_frame / max(len(image_paths), 1), 3),
            "processed_samples": processed_samples,
            "triangles_with_features": int(np.count_nonzero(feature_counts)),
            "elapsed_seconds": round(elapsed, 3),
        }
        atomic_json(args.progress, progress)
        print(
            f"OpenSeg {next_frame}/{len(image_paths)} "
            f"samples={processed_samples} "
            f"triangles={progress['triangles_with_features']}",
            flush=True,
        )
        if next_frame % args.checkpoint_interval == 0:
            atomic_npz(
                args.checkpoint,
                feature_sums=feature_sums,
                feature_counts=feature_counts,
                next_frame=np.asarray(next_frame, dtype=np.int64),
            )

    denominator = np.maximum(feature_counts[:, None], 1).astype(np.float32)
    triangle_features = feature_sums / denominator
    norms = np.linalg.norm(triangle_features, axis=1, keepdims=True)
    triangle_features = np.divide(
        triangle_features,
        np.maximum(norms, 1.0e-12),
        out=np.zeros_like(triangle_features),
        where=norms > 0,
    )
    atomic_npz(
        args.output,
        triangle_features=triangle_features.astype(np.float32),
        triangle_feature_counts=feature_counts,
    )
    if args.checkpoint.exists():
        args.checkpoint.unlink()
    final_progress = {
        "state": "complete",
        "frame": len(image_paths),
        "total_frames": len(image_paths),
        "percent": 100.0,
        "processed_samples": int(feature_counts.sum()),
        "triangles_with_features": int(np.count_nonzero(feature_counts)),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "output": str(args.output),
    }
    atomic_json(args.progress, final_progress)
    print(json.dumps(final_progress, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
