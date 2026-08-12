# 4.2 Layout Skeleton Extraction Migration Plan

## Goal

Migrate paper Section 4.2 into the active `src` package. The migrated pipeline should start from the completed Section 4.1 artifacts and produce a semantic layout skeleton:

```text
RGB frames + poses + DN-Splatter checkpoint + Poisson mesh
  -> OneFormer per-frame semantic maps
  -> ray-based semantic votes on mesh vertices
  -> superpoint majority refinement
  -> structure/object/stair/inaccurate mesh subsets
```

This plan intentionally stops before Section 4.3. Polygon initialization, prototype fitting, scene graph creation, room extrusion, door/window/stair final layout generation are out of scope for this migration step.

## Current State

Active `src` currently covers Section 4.1 only:

- `src/rgb_to_mesh/colmap.py`: COLMAP pose reconstruction diagnostics.
- `src/rgb_to_mesh/metric3d.py`: Metric3D depth and normal inference.
- `src/rgb_to_mesh/dn_splatter.py`: DN-Splatter dataset preparation and training.
- `src/rgb_to_mesh/mesh.py`: DN-Splatter `gs-mesh dn` Poisson export.

For `data/insta360/r04`, the Section 4.1 mesh artifact is complete:

- `data/insta360/r04/mesh/export/DepthAndNormalMapsPoisson_pcd.ply`
- `data/insta360/r04/mesh/export/DepthAndNormalMapsPoisson_poisson_mesh.ply`
- `data/insta360/r04/mesh/manifest.json`

Backup implementations exist but are not active:

- `src_backup/oneformer.py`: OneFormer COCO inference plus Appendix-A layout remapping.
- `src_backup/skeleton.py`: DN-Splatter raw-depth rendering, semantic ray sampling, mesh vertex voting, superpoint aggregation, and filtered semantic meshes.
- `docs/backup/ARTIFACT_CONTRACT.md`: useful artifact contract for `oneformer` and `skeleton`.

## Paper Requirements

Section 4.2 says:

1. Run OneFormer on input images.
2. Map COCO classes into layout categories.
3. Back-project `M = 5000` randomly sampled pixels per image into 3D.
4. Assign each back-projected semantic point to the nearest mesh vertex and accumulate class votes.
5. Cluster the mesh into superpoints using Superpoint Transformer preprocessing.
6. Assign each vertex the majority label of its superpoint.
7. Extract:
   - structural components: wall, ceiling, floor, surface;
   - geometrically inaccurate surfaces: windows, mirrors, outdoor/noise;
   - objects;
   - stairs.

The migrated implementation should preserve this paper-level behavior and record any pragmatic deviations explicitly in manifests.

## Proposed Active Package Layout

Add a new package under `src/layout_skeleton/`:

```text
src/layout_skeleton/
  __init__.py
  labels.py
  oneformer.py
  skeleton.py
```

Keep Section 4.1 code under `src/rgb_to_mesh/`. Do not place 4.2 code in `rgb_to_mesh`, because after Poisson export the pipeline is no longer only RGB-to-mesh.

Update `src/__main__.py` to advertise the new scripts:

```text
python src/layout_skeleton/oneformer.py --images IMAGES --mesh-manifest MESH_MANIFEST --output OUTPUT
python src/layout_skeleton/skeleton.py --transforms TRANSFORMS --dn-splatter DN_SPLATTER --mesh MESH --oneformer ONEFORMER --output OUTPUT
```

The exact CLI can be adjusted during implementation, but it should avoid requiring the old monolithic `PipelineConfig`.

## Migration Unit 1: Labels and OneFormer

Create `src/layout_skeleton/labels.py` from the stable parts of `src_backup/oneformer.py`:

- `LAYOUT_LABELS`
- `APPENDIX_COCO_IDS`
- `LAYOUT_PALETTE`
- `appendix_layout_lut()`

Keep the nine intermediate labels used by the backup implementation:

```text
wall, ceiling, floor, surface,
inaccurate_window, inaccurate_mirror, inaccurate_outdoor,
stairs, object
```

Implement `src/layout_skeleton/oneformer.py` by migrating `run_oneformer` without the old repo-wide `PipelineConfig` dependency.

Inputs:

- image directory;
- optional image list, defaulting to sorted supported images;
- camera width/height validation, either from `camera_param.json` or explicit CLI args;
- completed mesh manifest for provenance;
- local OneFormer model directory;
- output directory;
- random seed;
- preview count.

Outputs:

```text
oneformer/
  STATUS.json
  manifest.json
  inference.log
  labels.json
  per_image.jsonl
  coco_id/<stem>.png
  layout_id/<stem>.png
  previews/<stem>.jpg
```

Validation:

- every input image hash is recorded;
- each output map has the expected camera resolution;
- COCO IDs are in `[0, 132]`;
- layout IDs are in `[0, 8]`;
- one COCO map and one layout map exists per image;
- `labels.json` records the Appendix-A remapping and OneFormer COCO label contract.

## Migration Unit 2: Skeleton Extraction

Implement `src/layout_skeleton/skeleton.py` by migrating the backup logic with explicit CLI inputs.

Inputs:

- completed DN-Splatter manifest or training config/checkpoint pair;
- completed mesh manifest or direct Poisson mesh path;
- completed OneFormer manifest;
- Nerfstudio `transforms.json`;
- camera intrinsics;
- `ns-render` executable;
- Superpoint Transformer repository path;
- output directory;
- random seed and GPU selection.

Processing stages:

1. Render DN-Splatter raw depth for the train split using `ns-render dataset --rendered-output-names raw-depth`.
2. Validate one rendered depth map per frame.
3. Sample exactly 5000 pixels per frame, uniformly without replacement.
4. Back-project valid depth samples into world coordinates using the frame transform and camera intrinsics.
5. Assign each valid point's layout label to the nearest Poisson mesh vertex.
6. Store vertex vote counts and ray-to-mesh distances.
7. Run Superpoint Transformer preprocessing and Cut Pursuit hierarchy on mesh vertices and colors.
8. Aggregate vertex votes per superpoint and assign majority semantic labels.
9. Use K=5 nearest-ray probability transfer only as a fallback for superpoints with zero paper votes.
10. Write semantic mesh and filtered mesh subsets.

Outputs:

```text
skeleton/
  STATUS.json
  manifest.json
  skeleton.log
  commands.json
  render_depth.log
  rendered_depth/
  mesh.ply
  semantic_mesh.ply
  ceiling_wall_floor_mesh.ply
  objects_mesh.ply
  stair_mesh.ply
  geometrically_inaccurate_mesh.ply
  sampled_semantic_points.ply
  rays_preview_20000.ply
  full_ray_origins.npy
  full_ray_dests.npy
  ray_is_valid.npy
  hard_labels_simplified_segmentations.npy
  ray_frame_row_column.npy
  vertex_vote_counts.npy
  ray_to_mesh_distance_meters.npy
  vertex_probabilities_knn5.npy
  vertex_probabilities.npy
  vertex_hard_assignments.npy
  simplified_segmentation_labels.npy
  spt/
    level_<n>_segmentation.npy
    level_<n>_segment_vote_counts.npy
    level_<n>_segment_probabilities_simplified.npy
    level_<n>_segment_hard_assignments_simplified.npy
    mesh_class_colored.ply
```

Validation:

- one depth map per frame;
- one semantic map per frame;
- frame order matches `transforms.json`;
- exact sample count equals `frame_count * 5000`;
- all mesh vertices receive final labels;
- all final labels are in range;
- three superpoint levels are produced, unless explicitly configured otherwise;
- structural, object, stair, and inaccurate mesh subsets are finite when non-empty;
- manifest records all input hashes, output hashes, algorithm parameters, environment, and warnings.

## CLI Design

Prefer explicit component CLIs over reintroducing the old monolithic config.

OneFormer example:

```bash
python src/layout_skeleton/oneformer.py \
  --images data/insta360/r04/images \
  --mesh-manifest data/insta360/r04/mesh/manifest.json \
  --camera camera_param.json \
  --model-dir pretrained_weights/oneformer_coco_swin_large \
  --output data/insta360/r04/oneformer
```

Skeleton example:

```bash
python src/layout_skeleton/skeleton.py \
  --transforms data/insta360/r04/dn_splatter/transforms.json \
  --dn-splatter data/insta360/r04/dn_splatter \
  --mesh-manifest data/insta360/r04/mesh/manifest.json \
  --oneformer data/insta360/r04/oneformer \
  --camera camera_param.json \
  --ns-render /path/to/ns-render \
  --superpoint-repo external/superpoint_transformer \
  --output data/insta360/r04/skeleton
```

During implementation, verify the actual DN-Splatter manifest paths for `transforms.json`, training config, and final checkpoint. The current local `r04` download only includes `mesh`, so the missing `images`, `dn_splatter`, Metric3D depth, and pose artifacts may need to be downloaded or regenerated before 4.2 can run locally.

## Dependency Notes

OneFormer runtime needs:

- CUDA PyTorch;
- `transformers`;
- `Pillow`;
- offline/local OneFormer checkpoint files.

Skeleton runtime needs:

- CUDA PyTorch;
- Nerfstudio/DN-Splatter `ns-render`;
- Open3D;
- NumPy/SciPy;
- Superpoint Transformer preprocessing dependencies.

The migration should fail fast with actionable messages when a runtime dependency or external repository is missing.

## Implementation Steps

1. Create `src/layout_skeleton/labels.py` and unit-test the COCO-to-layout LUT.
2. Migrate OneFormer inference into `src/layout_skeleton/oneformer.py` with explicit CLI args and manifest output.
3. Add lightweight tests for image-list ordering, label remapping, manifest shape, and output hash recording. Mock the model for unit tests if needed.
4. Migrate skeleton helper functions that do not depend on heavy runtimes: backprojection, ray sampling, vertex voting, label aggregation, mesh subset bookkeeping.
5. Add tests for backprojection math, sample count, vertex vote accumulation, zero-vote superpoint fallback, and filtered label groups.
6. Wire the heavy skeleton runtime: raw-depth render, Open3D mesh load/write, Superpoint Transformer hierarchy.
7. Add CLI smoke checks that validate missing inputs fail before running expensive GPU work.
8. Run OneFormer on a small frame subset first, then full `r04` once inputs are present.
9. Run skeleton extraction on `r04`.
10. Compare resulting `skeleton/manifest.json` against the artifact contract and inspect `semantic_mesh.ply`, `ceiling_wall_floor_mesh.ply`, `objects_mesh.ply`, and `stair_mesh.ply`.

## Acceptance Criteria

4.2 migration is complete when:

- active `src` contains `layout_skeleton/oneformer.py` and `layout_skeleton/skeleton.py`;
- both scripts can be invoked directly and are listed by `src/__main__.py`;
- `oneformer/manifest.json` completes for the selected scene;
- `skeleton/manifest.json` completes for the selected scene;
- skeleton output includes semantic mesh, structural mesh, object mesh, stair mesh if present, inaccurate mesh if present, superpoint arrays, ray arrays, and visualizations;
- manifests include enough hashes and parameters to rerun or audit the stage;
- no 4.3 prototype-fitting assumptions leak into the 4.2 outputs.

## Known Risks

- The current local `r04` folder only has `mesh`; 4.2 cannot run until images, transforms, DN-Splatter checkpoint/config, and OneFormer model files are available locally.
- Backup code assumes a monolithic `PipelineConfig`; active migration should remove that dependency rather than copying the old config tree wholesale.
- Superpoint Transformer can be brittle because it imports from a repository-local `src` package; isolate its `sys.path` insertion and document the exact external revision.
- OneFormer checkpoints are large and should stay outside git.
- `ns-render` output paths may vary by Nerfstudio version; implement robust discovery but reject ambiguous matches.
- Symlinked dataset paths from transferred mesh runtime may point to `/tmp/tmp_data/...`; do not rely on those symlinks as canonical local inputs.
