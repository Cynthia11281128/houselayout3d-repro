# Artifact Contract

Each run is stored below `outputs/<scene>/<run-id>/`. A component owns exactly one directory and must not overwrite another component's artifacts.

| Component | Purpose |
| --- | --- |
| `input` | Immutable image/intrinsics manifest and exact approved image list |
| `pose` | Audited known poses and Nerfstudio `transforms.json` |
| `colmap` | Diagnostic COLMAP database, sparse reconstruction output, commands, and logs |
| `metric3d` | Metric depth plus auxiliary native normals and confidences |
| `dn_splatter` | Known-pose dataset, metric seed cloud, training config, and checkpoints |
| `mesh` | Poisson point cloud, mesh, and mesh validation |
| `oneformer` | Per-frame COCO IDs, Appendix-A layout IDs, previews, and immutable per-frame hashes |
| `skeleton` | Rendered depths, semantic rays, superpoint labels, and structural/object/stair meshes |
| `polygon_init` | RANSAC planes, rectified mesh, and polygon metadata |
| `prototype` | Optimization checkpoints and fitted prototype |
| `scene_graph` | Levels, rooms, openings, stairs, and room semantics |
| `layout` | Final 3D entities and dataset-compatible export |
| `validation` | Counts, topology checks, and final report |

Every completed component must contain `manifest.json` with at least:

- schema version, scene, component, status, and timestamps;
- input artifact paths and SHA256 digests;
- exact command, source revisions, environment, and random seed when applicable;
- output paths, counts, validation results, warnings, and elapsed time.

A component is resumable only when its manifest has `status: complete` and all declared output hashes still match. Failed or partial runs remain untouched for inspection and use a new attempt directory.

`input/images.txt` is the only image list authorized for reconstruction. The active known-pose path consumes only the explicitly configured front-view `poses.csv`; adjacent trajectories and ground truth remain unauthorized.

`skeleton` treats `mesh/export/DepthAndNormalMapsPoisson_poisson_mesh.ply` as the formal mesh input and renders raw metric depth from the final `dn_splatter` checkpoint. Semantic transfer follows the paper: each valid back-projected point votes for its nearest mesh vertex, followed by superpoint majority aggregation.

`polygon_init` consumes the structural subset, final-level superpoint indices, and semantic labels declared by `skeleton`. Its plane loop follows Appendix A Algorithm 1: RANSAC on the largest remaining superpoint, global unassigned plane inliers, mesh-edge connected components, maximum seed overlap, then the boundary of selected-component triangles.

`prototype` freezes and hashes every `skeleton` and `polygon_init` optimizer input before the long run. It executes the uploaded unofficial `fit_prototype.py` and its `mesh_fitting_3D` modules through a deterministic seeded launcher.

`scene_graph` consumes the final prototype serialized polygon state rather than inferring semantics from PLY colors. It implements Appendix D.1-D.5 room extraction, OpenSeg/CLIP room semantics, outdoor-leaf pruning, and stair endpoint assignment.

`layout` consumes the active `scene_graph` output and implements Sec. 4.4 plus Appendix D.6. Outputs include unified `layout.ply`/`layout.obj`, class meshes, per-room closed and opened shells, dataset-style entity JSON, and the final scene graph.

`validation` independently reloads the layout files and verifies declared digests, finite mesh geometry, room topology, graph-opening boundaries, floor-triangulation area conservation, window rectangles and thresholds, entity room references, and final scene-graph links.
