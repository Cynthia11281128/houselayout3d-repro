# Artifact contract

Each run is stored below `outputs/<scene>/<run-id>/`. A stage owns exactly one
directory and must not overwrite another stage's artifacts.

| Stage | Purpose |
| --- | --- |
| `00_input` | Immutable image/intrinsics manifest and exact approved image list |
| `01_pose` | Audited known poses and Nerfstudio `transforms.json` |
| `02_metric3d` | Metric depth plus auxiliary native normals and confidences |
| `03_dn_splatter` | Known-pose dataset, metric seed cloud, training config, and checkpoints |
| `04_mesh` | Poisson point cloud, mesh, and mesh validation |
| `05_oneformer` | Per-frame 133-class COCO IDs, Appendix-A nine-class IDs, previews, and immutable per-frame hashes |
| `06_skeleton` | Final-model raw depths, 5,000 rays/frame, nearest-vertex votes, three superpoint levels, labels, and structural/object/stair meshes |
| `07_polygon_init` | RANSAC planes, rectified mesh, and polygon metadata |
| `08_prototype` | Optimization checkpoints and fitted prototype |
| `09_scene_graph` | Levels, rooms, openings, stairs, and room semantics |
| `10_layout` | Final 3D entities and dataset-compatible export |
| `11_validation` | Tests, counts, topology checks, and final report |

Every completed stage must contain `manifest.json` with at least:

- schema version, scene, stage, status, and timestamps;
- input artifact paths and SHA256 digests;
- exact command, source revisions, environment, and random seed;
- output paths, counts, validation results, warnings, and elapsed time.

A stage is resumable only when its manifest has `status: complete` and all
declared output hashes still match. Failed or partial runs remain untouched for
inspection and use a new attempt directory.

`00_input/images.txt` is the only image list authorized for reconstruction.
The active r04 configuration consumes only the explicitly configured
`front/poses.csv`; adjacent trajectories and ground truth remain unauthorized.
The rejected `01_colmap` attempts remain preserved as diagnostic evidence but
are not part of `STAGE_ORDER`.

`06_skeleton` treats `04_mesh/export/DepthAndNormalMapsPoisson_poisson_mesh.ply`
as the only formal mesh input. It renders raw metric depth from the final
`03_dn_splatter` checkpoint. The primary semantic transfer follows the paper:
each valid back-projected point votes for its nearest mesh vertex, followed by
superpoint majority aggregation. The partial source's K=5 nearest-ray transfer
is preserved as a declared fallback only for a superpoint receiving no paper
votes.

`07_polygon_init` consumes only the structural subset, final-level superpoint
indices, and semantic labels declared by `06_skeleton`. Its plane loop follows
Appendix A Algorithm 1 exactly: RANSAC on the largest remaining superpoint,
global unassigned plane inliers, mesh-edge connected components, maximum seed
overlap, then the boundary of selected-component triangles. The rectified mesh
preserves the structural vertex order so every `polygon_info.json` contour
index addresses `clean_edge_mesh.ply` directly. A Poisson boundary junction
with degree other than two is decomposed in the fitted plane into simple closed
cycles instead of causing the full component to be discarded.

`08_prototype` freezes and hashes every Stage 06/07 optimizer input before the
long run. It executes the uploaded unofficial `fit_prototype.py` and its
`mesh_fitting_3D` modules byte-for-byte, through a deterministic seeded
launcher. Source-required compatibility inputs not produced by the published
preprocessing are recorded explicitly: a zero-probability `door` semantic
channel and a quadric-decimated 50,000-triangle object mesh. The completed
manifest requires mesh and serialized model-state checkpoints at every 100
steps from 0 through 3900, plus non-empty finite final mesh/model artifacts.

`09_scene_graph` consumes the final Stage 08 serialized polygon state rather
than inferring semantics from PLY colors. It implements Appendix D.1-D.5:
50 cm floor grouping, floor/ceiling BEV union, walls in the 0-2.5 m vertical
interval, 2.5 m then 1.5 m bottleneck room segmentation, the 1.5 m door/opening
rule, OpenSeg/CLIP room semantics and outdoor-leaf pruning, and the 50 cm stair
endpoint assignment rule. Its formal outputs are `levels.json`,
`rooms.geojson`, `scene_graph.json`, per-level raster grids, fused triangle and
room semantic features, stair diagnostics, and a color-coded room preview mesh.
Undisclosed raster/area/visibility defaults are explicit in YAML and recorded
in the stage manifest.

`10_layout` consumes the active Stage09 graph and implements Sec. 4.4 plus
Appendix D.6. It globally triangulates each room as a constrained planar graph
built from room boundaries, ceiling edges, and pairwise ceiling-plane
intersection lines; assigns triangles to upward ceilings; and extrudes floors,
ceilings, walls, discontinuities, doors, openings, and stairs. Window rays use
the paper's amended COCO set and retain only clusters with at least 10 points
and both dimensions above 30 cm. DBSCAN, outlier, voxel, and stair-visualization
constants omitted by the paper are explicit in YAML and the manifest. Outputs
include unified `layout.ply`/`layout.obj`, class meshes, per-room closed and
opened shells, dataset-style entity JSON, and the final scene graph.

`11_validation` independently reloads the Stage10 files and verifies every
declared digest, finite mesh geometry, pre-opening room topology, intentional
graph-opening boundaries, floor-triangulation area conservation, window
rectangles and thresholds, entity room references, and final scene-graph
links. It writes `final_report.json` plus a concise `final_report.md`; no ground
truth geometry is consumed.
