# 4.3 Polygon Initialization and Prototype Fitting Migration Plan

## Goal

Migrate paper Section 4.3 into the active `src` package. This stage consumes the Section 4.2 layout skeleton and produces a fitted layout prototype: a compact set of semantically labeled planar 3D polygons that better closes holes, connects nearby structure, and simplifies noisy mesh geometry.

The implementation should be split into two explicit components:

```text
4.2 skeleton
  -> polygon_init
  -> prototype_fitting
```

This plan stops before Section 4.4. Scene graph creation, room segmentation, room extrusion, final doors/windows/stairs entities, and dataset-format final layout export are out of scope.

## Paper Alignment

Section 4.3 requires:

1. Initialize a collection of planar 3D polygons `P` from the layout skeleton.
2. Fit one or more planes to each segmented superpoint from Section 4.2.
3. Optimize polygon vertex positions and plane equations with:
   - `Lgeo = Lprox + Lempty`;
   - `Lconnect`;
   - `Lsimple`.
4. Constrain each polygon to stay coplanar.
5. Allow and encourage shared vertices between polygons.
6. Periodically simplify by:
   - merging close vertices;
   - applying RDP to individual polygons;
   - merging close polygons with similar normals.
7. Close floor holes by projecting object triangles onto nearby floor polygons.
8. Extend walls/ceilings/floors where observed empty-space rays indicate the extension is plausible.

Appendix C.1 gives Algorithm 1 for polygon initialization. Appendix C.2 describes the trainable polygon set representation, planar constraints, shared vertices, and projection to one/two/three plane constraints.

## Current Source Material

Relevant existing code:

- `src_backup/polygon_init.py`
  - RANSAC plane fitting;
  - triangle connected-component extraction;
  - boundary loop extraction;
  - polygon metadata output;
  - rectified mesh output.
- `src_backup/prototype.py`
  - verifies skeleton and polygon initialization inputs;
  - freezes artifacts for the optimizer;
  - appends a zero-probability `door` channel for unofficial source compatibility;
  - probes prototype runtime dependencies;
  - executes `MultiFloor3D-unofficial/fit_prototype.py`.
- `src_backup/prototype_entry.py`
  - deterministic seeded launcher around the unofficial optimizer.
- `MultiFloor3D-unofficial/fit_prototype.py`
  - uploaded unofficial prototype optimizer entrypoint.
- `MultiFloor3D-unofficial/mesh_fitting_3D/`
  - differentiable polygon mesh representation, CGAL triangulation, losses, merge/split utilities.

The backup code still depends on the old monolithic `PipelineConfig`. The migration should remove that dependency and expose explicit CLI inputs, matching the active 4.1 and 4.2 style.

## Proposed Active Package Layout

Create a new active package:

```text
src/layout_prototype/
  __init__.py
  polygon_init.py
  prototype.py
  prototype_entry.py
```

Keep `layout_skeleton` focused on Section 4.2. Keep `layout_prototype` focused on Section 4.3.

Update `src/__main__.py` to advertise:

```text
python src/layout_prototype/polygon_init.py --skeleton SKELETON --output OUTPUT
python src/layout_prototype/prototype.py prepare --skeleton SKELETON --polygon-init POLYGON_INIT --source-repo SOURCE --output OUTPUT
python src/layout_prototype/prototype.py fit --prepared PROTOTYPE --source-repo SOURCE --output OUTPUT
```

The final CLI can differ, but it should not require repo-global YAML config.

## Component 1: Polygon Initialization

### Purpose

Implement Appendix C.1 Algorithm 1:

```text
layout skeleton mesh + final superpoint segmentation
  -> RANSAC plane components
  -> boundary polygons
  -> rectified planar mesh and polygon metadata
```

### Inputs

Required:

- completed `skeleton/manifest.json`;
- structural mesh from Section 4.2, normally `ceiling_wall_floor_mesh.ply`;
- structural per-vertex class probabilities from `ceiling_wall_floor_mesh_classes.npy`;
- final-level superpoint segmentation from `skeleton/spt/level_<final>_segmentation.npy`;
- vertex labels/probabilities if needed for semantic polygon labels.

Recommended CLI:

```bash
python src/layout_prototype/polygon_init.py \
  --skeleton data/insta360/r04/skeleton \
  --superpoint-level 3 \
  --plane-distance-threshold-meters 0.04 \
  --minimum-unassigned-vertices 100 \
  --ransac-iterations 256 \
  --rdp-epsilon-meters 0.03 \
  --output data/insta360/r04/polygon_init
```

### Algorithm

Follow Algorithm 1 exactly where possible:

1. Load the structural skeleton mesh.
2. Load final superpoint assignments and restrict them to the structural mesh vertices.
3. Mark every structural mesh vertex as unassigned.
4. While any superpoint has more than `K` unassigned vertices:
   - choose the superpoint with the most unassigned vertices;
   - fit a plane to its unassigned vertices using RANSAC;
   - find all unassigned vertices in the full structural mesh close to the plane;
   - split those inliers into mesh-edge connected components;
   - choose the connected component with maximum overlap with the seed superpoint;
   - project component vertices to the fitted plane;
   - extract boundary loops from selected component triangles;
   - simplify loops with RDP;
   - write polygon record(s).
5. Produce a rectified mesh where accepted component vertices are projected to their assigned plane.
6. Preserve rejected/unassigned diagnostics for auditing.

### Outputs

```text
polygon_init/
  STATUS.json
  manifest.json
  polygon_init.log
  assigned_plane_ids.npy
  plane_candidates.jsonl
  polygon_info.json
  rectified_mesh.ply
  plane_components_mesh.ply
  rejected_components.jsonl
```

The exact output names may follow `src_backup/polygon_init.py`, but the manifest must make them explicit.

### Validation

- skeleton manifest is complete and all declared hashes match;
- structural mesh is finite and non-empty;
- superpoint segmentation length matches skeleton mesh vertex count before structural filtering;
- each accepted polygon has:
  - at least three vertices;
  - finite plane equation;
  - non-zero area after projection to plane;
  - a source superpoint ID;
  - a connected component ID;
  - semantic probability summary;
- `assigned_plane_ids.npy` length matches structural mesh vertices;
- rectified mesh is finite;
- manifest records algorithm thresholds and counts:
  - accepted polygons;
  - rejected components;
  - assigned vertices;
  - unassigned vertices;
  - class histogram.

### Alignment Notes

Aligned with paper:

- RANSAC plane fitting per superpoint;
- global unassigned plane inliers;
- mesh-edge connected component;
- boundary polygon extraction.

Implementation choices:

- exact RANSAC iteration count;
- plane distance threshold;
- RDP simplification epsilon;
- branchy boundary loop handling;
- output PLY/JSON artifact structure.

## Component 2: Prototype Fitting

### Purpose

Optimize the initialized planar polygon set into the Section 4.3 layout prototype:

```text
polygon_init + skeleton geometry + empty-space rays + object mesh
  -> fitted prototype polygons
```

### Inputs

Required:

- completed `skeleton/manifest.json`;
- completed `polygon_init/manifest.json`;
- structural target mesh and class probabilities;
- object mesh from skeleton, when non-empty;
- semantic ray arrays:
  - `full_ray_origins.npy`;
  - `full_ray_dests.npy`;
  - `ray_is_valid.npy`;
  - `hard_labels_simplified_segmentations.npy`;
- class names from `simplified_segmentation_labels.npy`;
- polygon initialization artifacts:
  - `rectified_mesh.ply`;
  - `polygon_info.json`;
- unofficial source repository:
  - `MultiFloor3D-unofficial/fit_prototype.py`;
  - `MultiFloor3D-unofficial/mesh_fitting_3D/*`.

Recommended two-phase CLI:

```bash
python src/layout_prototype/prototype.py prepare \
  --skeleton data/insta360/r04/skeleton \
  --polygon-init data/insta360/r04/polygon_init \
  --source-repo MultiFloor3D-unofficial \
  --output data/insta360/r04/prototype
```

```bash
python src/layout_prototype/prototype.py fit \
  --prepared data/insta360/r04/prototype \
  --source-repo MultiFloor3D-unofficial \
  --python /path/to/prototype/python \
  --output data/insta360/r04/prototype
```

Use a `prepare` phase because the optimizer is long-running and sensitive to dependency/source state. Freezing inputs before execution makes failures auditable.

### Preparation Algorithm

1. Verify all skeleton and polygon-init input hashes.
2. Verify unofficial source files exist and record their SHA256 hashes.
3. Probe the prototype Python runtime:
   - CUDA PyTorch;
   - PyTorch3D;
   - Open3D;
   - Shapely;
   - scikit-learn;
   - RDP;
   - CGAL Python bindings.
4. Copy or hardlink frozen optimizer inputs into `prototype/frozen_inputs/`.
5. Prepare semantic class probabilities for unofficial source compatibility.
   - The backup implementation appends a zero-probability `door` channel because the unofficial optimizer expects a door class.
   - This is not a Section 4.3 paper requirement; record it as a compatibility warning.
6. Optionally simplify object mesh to a configured triangle budget for floor-hole projection.
7. Build and record the exact optimizer command.

### Fitting Algorithm

Use the unofficial optimizer as the first migration target rather than reimplementing differentiable polygon fitting from scratch.

Expected behavior from the source:

- build polygon set from `polygon_info.json` and rectified mesh;
- triangulate polygons with constrained Delaunay triangulation;
- keep trainable plane equations and vertex positions;
- project vertices to plane constraints when accessed;
- allow vertex sharing and multi-plane constraints;
- optimize losses corresponding to:
  - `Lprox`;
  - `Lempty`;
  - `Lconnect`;
  - `Lsimple`;
- periodically:
  - merge nearby vertices;
  - simplify polygon boundaries;
  - merge similar close planes;
- project object triangles to floor polygons for floor hole closing;
- extend wall/ceiling/floor edges when empty-space ray intersection density is low.

### Outputs

```text
prototype/
  STATUS.json
  manifest.json
  prepare_manifest.json
  commands.json
  frozen_inputs/
  attempts/
    attempt_000/
      STATUS.json
      fit.log
      checkpoints/
      final_model_state.*
      final_mesh.ply
      final_polygons.json
      diagnostics.json
```

If the unofficial optimizer writes different filenames, wrap or adapt outputs so `manifest.json` declares:

- final serialized polygon state;
- final prototype mesh;
- final semantic class names/probabilities;
- optimizer logs;
- checkpoints or at least final checkpoint;
- source file hashes;
- runtime dependency versions.

### Validation

- source repository files match recorded hashes or manifest clearly records changed hashes;
- runtime probe succeeds before long fit starts;
- frozen input hashes match the original manifests;
- final prototype mesh is finite and non-empty;
- final polygon state has:
  - plane equations;
  - vertex positions;
  - polygon-to-class labels/probabilities;
  - polygon topology;
- each polygon has at least three vertices and finite area;
- no NaN/Inf in optimized tensors/artifacts;
- optimizer exit code is zero;
- manifest records:
  - elapsed time;
  - iteration count;
  - checkpoint interval;
  - simplification/merge thresholds;
  - loss configuration;
  - warnings for any source compatibility adaptations.

### Alignment Notes

Aligned with paper:

- planar polygon set initialized from skeleton;
- gradient-based fitting of plane equations and vertex positions;
- coplanar constraints;
- shared vertices;
- `Lgeo`, `Lconnect`, `Lsimple` family;
- empty-space rays;
- floor hole closing via objects;
- wall/ceiling/floor extension.

Implementation choices or not fully paper-specified:

- exact unofficial optimizer implementation;
- source hash pinning;
- zero-probability `door` channel for optimizer compatibility;
- object mesh simplification target;
- exact checkpoint file format;
- exact dependency versions;
- exact simplification/merge thresholds where only described qualitatively in the paper.

## Implementation Steps

1. Create `src/layout_prototype/__init__.py`.
2. Migrate pure geometry helpers from `src_backup/polygon_init.py`:
   - `fit_plane_ransac`;
   - `plane_basis`;
   - `project_to_plane_2d`;
   - boundary loop extraction;
   - contour grouping and simplification.
3. Implement explicit CLI and manifest handling for `polygon_init.py`.
4. Add pure-function checks for:
   - RANSAC plane recovery;
   - boundary loop extraction;
   - projected polygon area;
   - superpoint/mesh length mismatch failure.
5. Run `polygon_init` on a small synthetic mesh fixture.
6. Run `polygon_init` on `r04` after 4.2 skeleton exists.
7. Create `src/layout_prototype/prototype_entry.py` from backup seeded launcher.
8. Implement `prototype.py prepare` with explicit inputs and source/runtime audits.
9. Implement `prototype.py fit` wrapper around `MultiFloor3D-unofficial/fit_prototype.py`.
10. Run runtime probe only, then a short debug fit if the unofficial source supports limited iterations.
11. Run full prototype fitting on `r04`.
12. Inspect final prototype mesh and polygon state before starting Section 4.4.

## Acceptance Criteria

Section 4.3 migration is complete when:

- `polygon_init/manifest.json` completes from a real 4.2 skeleton;
- `prototype/prepare_manifest.json` freezes and validates all optimizer inputs;
- `prototype/manifest.json` completes after fitting;
- final prototype artifacts are finite, hashed, and reloadable;
- no 4.4 scene graph or room extrusion assumptions are required to interpret the prototype;
- all paper deviations and unofficial-source compatibility shims are recorded in manifests.

## Known Risks

- 4.3 cannot run until 4.2 skeleton is complete.
- The unofficial optimizer may require a specialized Python environment with CGAL and PyTorch3D.
- The backup implementation pins source hashes; local source edits will need an explicit decision: update expected hashes or record unpinned source state.
- Boundary extraction from noisy connected components may produce branchy/non-simple loops; diagnostics should preserve rejected cases.
- Object mesh can be empty for some scenes; floor-hole projection must handle that without failing the whole prototype stage.
- Long optimization can fail late, so the prepare/frozen-input phase should be separate from fitting attempts.
