# houselayout3d-repro

## Overview

This directory contains an independent reproduction of the MultiFloor3D method
described in the HouseLayout3D paper. The implementation is being delivered in
explicit approval-gated milestones. It does not import code from GRIP's
`src/` tree.

Current status: **formal stages 04-07 completed for r04**.

## Contents

- [Overview](#overview)
- [Repository contract](#repository-contract)
  - [Storage policy](#storage-policy)
  - [Output root](#output-root)
  - [Reproduction constraints](#reproduction-constraints)
  - [Related docs](#related-docs)
- [Data Preparation]()
- [Pipeline](#pipeline)
  - [Project skeleton check](#project-skeleton-check)
  - [Input freeze and COLMAP evidence](#input-freeze-and-colmap-evidence)
  - [Known-pose camera path](#known-pose-camera-path)
  - [Metric3D depth](#metric3d-depth)
  - [DN-Splatter training](#dn-splatter-training)
  - [Mesh export](#mesh-export)
  - [OneFormer semantics](#oneformer-semantics)
  - [Semantic skeleton](#semantic-skeleton)
  - [Polygon initialization](#polygon-initialization)
  - [Prototype fitting](#prototype-fitting)
  - [Stage 04-08 verification](#stage-04-08-verification)
  - [Scene graph](#scene-graph)
  - [Layout generation](#layout-generation)
  - [Final validation](#final-validation)
- [Environment smoke tests](#environment-smoke-tests)

## Repository contract

### Storage policy

Large or generated files live outside the checkout under:

```text
/tmp/tmp_data/GRIP-Layout/baselines/HouseLayout3D/
```

The local `data/`, `weights/`, `outputs/`, `cache/`, and `reference_files/`
entries are symlinks into that root. Pinned external repositories live in
`external/`; their exact revisions are recorded in
`references/external_revisions.json`.

### Output root

All formal results for this run are rooted at:

```text
/tmp/tmp_data/GRIP-Layout/baselines/HouseLayout3D/outputs/r04_front/r04-front-known-pose-v1/
```

### Reproduction constraints

- The production package is standalone and must not import GRIP internals.
- The active r04 configuration accepts RGB images, camera intrinsics, and the
  explicitly configured front-view `poses.csv`; it has no ground-truth input.
- Every pipeline stage writes a manifest and uses a separate output directory.
- No stage silently changes a model, threshold, input subset, or fallback
  algorithm after a failure.

### Related docs

- [Artifact contract](docs/ARTIFACT_CONTRACT.md)
- [Environment details](environments/README.md)
- [Unofficial source provenance](MultiFloor3D-unofficial/SOURCE_PROVENANCE.md)

## Data Preparation
### ASE
```bash
python data_prep/ase_transfer_single.py \
  --source ../layout_reconstruction/data/aria_ase/14240 \
  --dest-root data/aria_ase/14240
```

## Pipeline

### Project skeleton check

The project skeleton check uses only the host Python and PyYAML:

```bash
bash scripts/verify_skeleton.sh
```

### Input freeze and COLMAP evidence

Freeze and audit the r04 front-view input without reading the adjacent pose
file:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d prepare-input configs/r04_front.yaml --run-id r04-front-v1
```

Run the sparse reconstruction using only that audited image list:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-colmap configs/r04_front.yaml --run-id r04-front-v1
```

The preserved overlap-30 retry uses a separate run ID so it cannot overwrite
the default-overlap evidence:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d prepare-input configs/r04_front_overlap30.yaml \
  --run-id r04-front-overlap30-v1
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-colmap configs/r04_front_overlap30.yaml \
  --run-id r04-front-overlap30-v1
```

The stage writes `STATUS.json`, exact commands, separate logs, the COLMAP
database, all sparse components, and a `sparse/main` link to the largest model.

### Known-pose camera path

The active pose-backed path bypasses the rejected COLMAP reconstructions. It
audits the 469 known poses and exports metric Nerfstudio camera-to-world
transforms without pose centering or scaling:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d prepare-input configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d prepare-poses configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

`01_pose/transforms.json` converts only the camera axes from OpenCV to
Nerfstudio/OpenGL. World XYZ and translation units remain unchanged. Any
downstream Nerfstudio command must use `--auto-scale-poses False` and
`--orientation-method none`.

### Metric3D depth

Run pinned Metric3D v2 ViT-L inference on every approved frame:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-metric3d configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

`02_metric3d/depth` stores float32 metric depth maps. Auxiliary native
Metric3D normals and confidence maps are stored in compressed form under
`02_metric3d/geometry`; they are not silently treated as DN-Splatter normal
supervision.

### DN-Splatter training

Prepare the known-pose DN-Splatter dataset and train the official 30k-step
configuration:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /tmp/tmp_data/miniconda3/envs/nerfstudio/bin/python -m houselayout3d \
  prepare-dn-splatter configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /tmp/tmp_data/miniconda3/envs/nerfstudio/bin/python -m houselayout3d \
  train-dn-splatter configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

Since this run intentionally bypasses COLMAP, preparation unprojects the
Metric3D depths through the approved known poses to create a metric seed point
cloud. Training keeps camera optimization, auto-orientation, centering, and
pose auto-scaling disabled.

### Mesh export

Export the formal depth-and-normal Poisson mesh used by the downstream layout
pipeline:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src \
  /tmp/tmp_data/miniconda3/envs/nerfstudio/bin/python -m houselayout3d \
  run-mesh configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

This writes the oriented point cloud and Poisson surface to `04_mesh`. The
Open3D TSDF mesh under `03_dn_splatter/mesh_o3dtsdf` is retained only as an
auxiliary visualization artifact.

### OneFormer semantics

Run offline OneFormer COCO semantic segmentation and the exact Appendix-A
Table-7 remapping:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-oneformer configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

`05_oneformer/coco_id` preserves all 133 COCO semantic IDs;
`05_oneformer/layout_id` stores the nine layout classes needed downstream.
The model, tokenizer, and COCO metadata are loaded locally with network access
disabled.

### Semantic skeleton

Render final-checkpoint DN-Splatter depths and extract the semantic layout
skeleton:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-skeleton configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

This stage uses the paper's 5,000 random pixels per frame, accumulates each
back-projected semantic sample at its nearest mesh vertex, and applies three
levels of Superpoint Transformer Cut Pursuit. Its principal outputs are:

- `06_skeleton/semantic_mesh.ply`: the full, color-coded semantic mesh;
- `06_skeleton/ceiling_wall_floor_mesh.ply`: structural skeleton;
- `06_skeleton/objects_mesh.ply`: object subset used to close floor holes;
- `06_skeleton/stair_mesh.ply`: staircase subset;
- `06_skeleton/full_ray_origins.npy` and `full_ray_dests.npy`: optimization
  rays for the next prototype-fitting stages.

### Polygon initialization

Initialize planar polygons with Appendix A, Section C.1, Algorithm 1:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-polygon-init configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

The stage repeatedly selects the final-level superpoint with the most
unassigned vertices, fits a plane by RANSAC, finds all unassigned global plane
inliers, selects the mesh-edge connected component with maximum seed overlap,
and extracts polygon contours from triangle-boundary edges. Poisson
non-manifold boundary junctions are decomposed into simple closed contours;
disjoint outer contours become separate polygon records. Principal outputs:

- `07_polygon_init/clean_edge_mesh.ply`: structure vertices rectified onto
  their fitted planes, with vertex order preserved;
- `07_polygon_init/polygon_info.json`: the contour, plane, color, RDP mask,
  semantic, and compatibility metadata consumed by `fit_prototype.py`;
- `07_polygon_init/plane_components_mesh.ply`: color-coded accepted plane
  components;
- `07_polygon_init/polygon_boundaries.ply`: all extracted boundary lines.

The Appendix does not disclose `K`, RANSAC distance/iteration settings, or a
contour simplification tolerance. The reproduction therefore exposes the
measured defaults in YAML: `K=100`, 5 cm plane distance, 256 RANSAC iterations,
and 3 cm RDP epsilon.

### Prototype fitting

Freeze the Stage 08 inputs, then run the complete 4,000-step Matterport/Z-up
prototype optimizer from the byte-preserved unofficial source:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d prepare-prototype configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d fit-prototype configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

Preparation checks the source hashes and runtime, appends the zero-probability
`door` channel required only by the supplied optimizer, and explicitly creates
the simplified object mesh absent from the unofficial preprocessing script.
Fitting preserves every 100-step mesh/model checkpoint and the periodic
vertex, edge, and plane simplification diagnostics. Failed attempts are never
overwritten.

### Stage 04-08 verification

Verify every declared hash and the cross-stage array/mesh/hierarchy contracts:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  python scripts/verify_stages_04_06.py \
  outputs/r04_front/r04-front-known-pose-v1
```

Verify Stage 07 hashes, optimizer-required fields, contour indices/masks, and
rectified plane residuals independently:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  python scripts/verify_stage_07.py \
  outputs/r04_front/r04-front-known-pose-v1
```

After fitting completes, verify the frozen inputs/source, all 40 mesh and model
checkpoints, and the final fitted mesh independently:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  python scripts/verify_stage_08.py \
  outputs/r04_front/r04-front-known-pose-v1
```

### Scene graph

Construct the Appendix-D per-level room scene graph from the fitted prototype:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-scene-graph configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

Stage 09 groups floor polygons within 50 cm, assigns ceilings to the closest
next-lower level with at least 1 m clearance, constructs each floorplan, applies
the Appendix 2.5 m then 1.5 m bottleneck room segmentation, labels graph edges
as doors below 1.5 m or openings otherwise, projects OpenSeg features onto
visible prototype triangles, classifies rooms with CLIP, prunes outdoor leaf
nodes, and tests stair components against the 50 cm room-distance rule. Full
OpenSeg feature maps are never stored; sampled fusion checkpoints are written
every 25 frames.

Verify the scene graph hashes, paper thresholds, level grids, room geometries,
graph references, semantic feature arrays, and preview mesh independently:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  python scripts/verify_stage_09.py \
  outputs/r04_front/r04-front-known-pose-v1
```

### Layout generation

Generate the final 3D layout by extruding rooms, cutting graph openings and
doors, adding stair geometry, and detecting windows on the final walls:

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output \
  -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d run-layout configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

The formal outputs are `10_layout/attempt_*_complete/layout.ply` and
`layout.obj`. `layout_entities.json` stores dataset-style wall, floor, ceiling,
door, window, and stair polygons, while `final_scene_graph.json` links them to
rooms and graph edges. Closed pre-opening shells and the final room meshes with
intentional door/opening holes are retained under `rooms_closed/` and
`rooms_final/`.

### Final validation

Run the independent final validation and write the Stage11 report:

```bash
conda run -p /tmp/tmp_data/miniconda3/envs/houselayout3d-layout \
  houselayout3d validate-layout configs/r04_front_known_pose.yaml \
  --run-id r04-front-known-pose-v1
```

The machine-readable and concise reports are written to
`11_validation/attempt_*_complete/final_report.json` and `final_report.md`.

## Environment smoke tests

The three runtime environments and their roles are documented in
[environments/README.md](environments/README.md). To verify the pinned weights
and all three environments on GPU 1:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_test_envs.sh
```

This test performs real CUDA calls through DN-Splatter's nerfstudio
environment, PyTorch3D, FRNN, PyG, Superpoint Transformer cut-pursuit, and
TensorFlow/OpenSeg.
