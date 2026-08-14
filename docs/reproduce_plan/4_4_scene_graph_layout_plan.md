# 4.4 Single-Floor Scene Graph and Layout Export Migration Plan

## Goal

Migrate paper Section 4.4 into the active `src` package for the single-floor case. This stage consumes the Section 4.3 layout prototype and produces the final room-level scene graph and dataset-style 3D layout entities.

The implementation should be split into two explicit components:

```text
4.3 prototype
  -> scene_graph
  -> layout
```

This plan intentionally does not implement floor-level identification. The target scenes are treated as one building level by default. Appendix D.1 is therefore out of scope except for validating that the prototype contains enough floor geometry to define one usable level.

## Paper Alignment

Section 4.4 requires converting a semantically labeled prototype polygon set into a final layout represented as a scene graph:

1. Create 2D floorplans from prototype floor and ceiling polygons.
2. Segment each floorplan into rooms using walls.
3. Create a 2D scene graph with rooms as nodes and bottlenecks/openings as edges.
4. Classify and prune rooms using visual-language features.
5. Detect stairs and connect the affected rooms.
6. Extrude each room floorplan into a 3D room shell.
7. Insert openings for doors and stairs.
8. Detect windows from geometrically inaccurate window/outdoor evidence and attach them to walls.
9. Export final polygons for walls, floors, ceilings, doors, windows, and stairs.

For this reproduction step, item 1 is restricted to a single default level. There is no grouping of multiple building floors and no multi-floor stair edge between levels.

Appendix D details the relevant algorithms:

- D.2: creating a 2D floorplan from floor and ceiling polygons;
- D.3: room segmentation using Hov-SG's morphology-based algorithm;
- D.4: room classification and outdoor-leaf pruning;
- D.5: stair detection from the stair mesh;
- D.6: door and stair handling during floor extrusion.

Appendix D.1 is skipped because floor identification is not needed under the single-floor assumption.

## Current Source Material

Relevant active code:

- `src/prototype_fitting/prototype.py`
  - validates Section 4.3 inputs;
  - freezes optimizer inputs;
  - records final prototype mesh and serialized polygon state.
- `src/layout_skeleton/skeleton.py`
  - outputs semantic mesh subsets, including stairs, objects, inaccurate surfaces, and structural components;
  - records per-vertex semantic labels and ray artifacts.

Relevant external repositories already present in the workspace:

- `external/HOV-SG`
  - candidate source for morphology-based room segmentation;
  - candidate source for graph construction conventions.
- `external/OpenScene`, `external/openseg-tpu`, and `external/openai-clip`
  - candidate sources for OpenSeg/CLIP-style room classification if we implement Appendix D.4 fully.

There is currently no active `src/scene_graph` or `src/layout_export` package.

## Proposed Active Package Layout

Create two new active packages:

```text
src/scene_graph/
  __init__.py
  floorplan.py
  rooms.py
  openings.py
  stairs.py
  windows.py
  graph.py

src/layout_export/
  __init__.py
  extrusion.py
  entities.py
  mesh_io.py
  layout.py
```

Keep Section 4.4 graph extraction separate from final geometry export:

- `scene_graph` owns 2D floorplan construction, room cells, openings, room metadata, window candidates, and stair graph edges.
- `layout_export` owns 3D room extrusion, door/stair openings, final polygon entities, and mesh/JSON export.

Update `src/__main__.py` to advertise:

```text
python src/scene_graph/graph.py --prototype PROTOTYPE --skeleton SKELETON --output OUTPUT
python src/layout_export/layout.py --scene-graph SCENE_GRAPH --prototype PROTOTYPE --output OUTPUT
```

The final CLI can differ, but it should not require repo-global YAML config.

## Component 1: Scene Graph

### Purpose

Implement the single-floor subset of Section 4.4 and Appendix D.2-D.5:

```text
prototype polygons + skeleton semantic subsets
  -> one 2D floorplan
  -> room cells and opening edges
  -> optional room labels and pruning
  -> stair and window candidates
  -> scene graph manifest
```

### Inputs

Required:

- completed `prototype/manifest.json`;
- final prototype mesh, normally `prototype/fitted_mesh.ply`;
- final prototype polygon state, normally `prototype/polygon_set_3d.pt`;
- completed `skeleton/manifest.json`;
- structural semantic labels and class names from Section 4.2;
- stair mesh from Section 4.2, if non-empty;
- inaccurate window/outdoor semantic evidence from Section 4.2;
- original camera images and transforms if room classification or window evidence projection needs image-space features.

Recommended CLI:

```bash
python src/scene_graph/graph.py \
  --prototype data/insta360/r04/prototype \
  --skeleton data/insta360/r04/skeleton \
  --output data/insta360/r04/scene_graph
```

### Single-Floor Assumption

The component should not infer building levels. Instead:

1. Load all floor-classified prototype polygons.
2. Reject the run if no floor polygon exists.
3. Compute one representative floor elevation from the floor polygons, using an area-weighted mean or a robust median.
4. Assign all suitable ceiling polygons, wall polygons, rooms, openings, windows, and stairs to this one level.
5. Record `"single_floor_assumption": true` in the manifest.

If multiple separated floor elevations are visible, the component should warn but still select one default level unless an explicit `--floor-height` override is provided.

### Algorithm

#### 1. Load Prototype Geometry

Load the final prototype polygon state when possible. Prefer serialized topology and semantic class probabilities over PLY colors. Use the final prototype mesh only as a validation and visualization artifact.

Required normalized internal representation:

```text
PrototypePolygon:
  id
  class_name
  vertices_3d
  plane
  normal
  area
```

The implementation should preserve original polygon IDs so final entities can refer back to source prototype polygons.

#### 2. Create the Single 2D Floorplan

Project floor polygons to BEV using the global Z-up convention. Merge them with suitable ceiling polygons:

- floor polygons: all prototype polygons classified as `floor`;
- ceiling polygons: polygons classified as `ceiling` whose centroid is at least 1m above the representative floor height;
- optional surfaces: include `surface` only when its normal and elevation make it floor-like or ceiling-like, and record this as a warning.

Compute a robust 2D union using a geometry library such as Shapely. The output is one floorplan polygon or multipolygon. Preserve holes.

#### 3. Select Walls for Room Segmentation

Select wall-classified prototype polygons that:

- intersect the 2D floorplan in BEV;
- vertically overlap `[floor_height, floor_height + 2.5m]`;
- have mostly vertical normals.

Convert selected wall polygons to 2D wall masks or line barriers suitable for room segmentation.

#### 4. Segment Rooms

Apply a morphology-based room segmentation following Appendix D.3:

- first split with bottleneck width `2.5m`;
- then split with bottleneck width `1.5m`;
- create one room node per resulting cell;
- create graph edges for bottlenecks between cells;
- classify edges with width below `1.5m` as `door`, otherwise `opening`.

Implementation options:

- reuse Hov-SG's room segmentation code if it can be isolated cleanly;
- otherwise implement the morphology operations locally with OpenCV/scikit-image/Shapely while matching the documented bottleneck behavior.

Record the chosen implementation path in the manifest.

#### 5. Optional Room Classification and Pruning

Appendix D.4 uses OpenSeg features and CLIP room text embeddings. Implement this as a separate optional sub-step:

- `--enable-room-classification` computes room labels;
- default behavior can assign `"unknown"` to every room;
- outdoor-leaf pruning should run only when room classification is enabled and confidence is sufficient.

Room classes should match the paper where possible:

```text
bathroom, bedroom, living room, garage, entrance, kitchen, office,
stairs, gym, classroom, spa/sauna, mirror, grass/bushes/trees,
driveway, veranda/terrace/balcony, unknown
```

This keeps the graph component useful before the OpenSeg/CLIP dependency path is fully stabilized.

#### 6. Detect Stairs

For the single-floor case, stairs do not connect two inferred floor levels. Still detect stair geometry so the final layout can reserve stair openings and export stair entities.

Algorithm:

1. Load the stair mesh extracted by Section 4.2.
2. Cluster connected components.
3. Project each component to BEV.
4. Approximate each component with an oriented bounding rectangle.
5. Assign the rectangle to the nearest room or to all intersecting rooms.
6. Store it as a `stair_region` in the scene graph.

Do not create inter-floor edges unless a future multi-floor extension adds floor identification.

#### 7. Detect Windows

Window detection should be graph-side because final windows need wall and room references.

Use the paper's window evidence:

- inaccurate window class;
- outdoor/noise classes visible through windows;
- optionally curtains and window blinds for window detection;
- exclude mirrors from window detection.

Proposed algorithm:

1. Collect skeleton/prototype vertices or ray evidence for window/outdoor classes.
2. Assign evidence points to nearby wall polygons.
3. Cluster evidence per wall using DBSCAN.
4. Fit an axis-aligned rectangle per cluster in the wall plane.
5. Keep rectangles with width and height greater than `0.30m`.
6. Attach each window to the nearest room-side wall segment.

### Outputs

```text
scene_graph/
  STATUS.json
  manifest.json
  graph.json
  floorplan.geojson
  rooms.geojson
  openings.geojson
  windows.json
  stairs.json
  debug/
    room_segmentation.png
    floorplan_walls.png
```

`graph.json` should include:

```json
{
  "levels": [
    {
      "id": "level_0",
      "floor_height": 0.0,
      "floorplan_ref": "floorplan.geojson"
    }
  ],
  "rooms": [],
  "edges": [],
  "windows": [],
  "stair_regions": []
}
```

### Validation

- prototype and skeleton manifests are complete and hashes match;
- exactly one default level is declared;
- floorplan geometry is valid and non-empty;
- every room polygon lies within the floorplan, allowing small numeric tolerance;
- every opening references two valid rooms or one room plus exterior;
- door/opening widths are finite and positive;
- every window references a valid wall source and room when assignable;
- every stair region intersects at least one valid room or is recorded as unassigned with a warning;
- graph JSON is schema-valid and reloadable.

## Component 2: Layout Export

### Purpose

Convert the scene graph and prototype geometry into final Section 4.4 layout entities:

```text
scene graph + prototype polygons
  -> per-room 3D shells
  -> door/opening cutouts
  -> stair geometry
  -> window rectangles
  -> final layout mesh and entity JSON
```

### Inputs

Required:

- completed `scene_graph/manifest.json`;
- completed `prototype/manifest.json`;
- final prototype polygon state;
- final prototype mesh for validation and debug comparison.

Recommended CLI:

```bash
python src/layout_export/layout.py \
  --scene-graph data/insta360/r04/scene_graph \
  --prototype data/insta360/r04/prototype \
  --output data/insta360/r04/layout
```

### Algorithm

#### 1. Determine Room Heights

For each room:

- floor elevation is the single floor height from the scene graph;
- ceiling height should come from ceiling polygons over the room when available;
- fallback to a robust scene-level ceiling height;
- record fallback use per room.

#### 2. Extrude Room Floorplans

Extrude each 2D room cell into a closed 3D shell:

- floor face;
- ceiling face;
- wall faces along boundary segments.

Preserve room IDs on all generated faces.

#### 3. Insert Doors and Openings

For each graph edge:

- approximate the splitting boundary with an oriented 2D rectangle;
- use fixed door height `2.10m` for `door` edges;
- leave wider `opening` edges open across the configured opening height;
- add wall triangles above the door/opening where needed;
- avoid duplicating wall faces for shared room boundaries.

Door entities should reference:

- connected room IDs;
- 2D opening geometry;
- 3D rectangle or frame geometry;
- source graph edge ID.

#### 4. Insert Stair Openings and Stair Geometry

For each `stair_region`:

- subtract the stair footprint from intersecting room floorplans before extrusion when possible;
- add a stair entity using the oriented rectangle from the scene graph;
- in the single-floor case, represent the stair as a pitched or flat reserved region unless reliable endpoint heights are available;
- do not create a second floor shell.

#### 5. Insert Windows

For each detected window rectangle:

- create a 3D rectangular window entity on the assigned wall plane;
- clip or validate the rectangle against the wall face;
- reference the owning room and source wall/prototype polygon;
- preserve width/height thresholds used during detection.

#### 6. Export Final Entity Set

Final entities should cover:

```text
wall
floor
ceiling
door
opening
window
stairs
```

Each entity should include:

- stable ID;
- semantic class;
- room ID or graph edge ID;
- vertices in 3D;
- source prototype polygon IDs when available;
- source scene-graph object ID when available.

### Outputs

```text
layout/
  STATUS.json
  manifest.json
  layout.json
  layout.ply
  layout.obj
  entities/
    walls.json
    floors.json
    ceilings.json
    doors.json
    openings.json
    windows.json
    stairs.json
  meshes/
    walls.ply
    floors.ply
    ceilings.ply
    doors.ply
    windows.ply
    stairs.ply
  debug/
    room_shells/
```

`layout.json` should be the canonical machine-readable output. Mesh files are visualization/export artifacts.

### Validation

- scene graph manifest is complete and hashes match;
- every room produces finite floor, ceiling, and wall geometry;
- every generated polygon has at least three vertices and non-zero area;
- no NaN/Inf in exported vertices;
- door/opening/window/stair entities reference valid graph IDs;
- class-specific mesh files are reloadable and non-empty when the class exists;
- room shells are topologically consistent enough for downstream evaluation;
- manifest records any fallback ceiling heights, skipped windows, unassigned stairs, or invalid room cells.

## Implementation Steps

1. Create `src/scene_graph/__init__.py` and `src/layout_export/__init__.py`.
2. Implement prototype polygon-state loading with a stable internal `PrototypePolygon` representation.
3. Implement single-floor floorplan creation from floor and ceiling polygons.
4. Implement wall selection and BEV wall mask generation.
5. Isolate or reimplement Hov-SG-style morphology room segmentation.
6. Emit `scene_graph/graph.json`, GeoJSON debug outputs, and manifest validation.
7. Add optional room classification as a separate flag; keep default rooms as `unknown`.
8. Implement stair footprint extraction for single-floor stair regions.
9. Implement window detection and wall assignment.
10. Implement room shell extrusion.
11. Implement door/opening/stair cutouts during extrusion.
12. Implement final entity JSON and class mesh exports.
13. Update `src/__main__.py` and `docs/steps.md` with the new commands.
14. Add focused tests for floorplan union, room segmentation fixtures, opening width classification, extrusion, and entity schema validation.
15. Run the full 4.4 path on `r04` after a real 4.3 prototype manifest exists.

## Acceptance Criteria

Section 4.4 single-floor migration is complete when:

- `scene_graph/manifest.json` completes from a real 4.3 prototype;
- `scene_graph/graph.json` contains one level, valid room nodes, valid opening edges, and any detected windows/stairs;
- `layout/manifest.json` completes from the scene graph;
- `layout/layout.json` contains final wall, floor, ceiling, door/opening, window, and stair entities where present;
- exported meshes are finite, reloadable, and declared in the manifest with hashes;
- no floor-identification logic is required to interpret the output;
- all deviations from the multi-floor paper algorithm are recorded in the manifests.

## Known Risks

- The final prototype serialized state may need adapter code if the unofficial optimizer stores topology in a PyTorch object that is not stable across environments.
- Hov-SG room segmentation may be difficult to reuse directly; a local morphology implementation may be more reproducible.
- Single-floor assumptions can hide true split-level geometry; the manifest should warn when floor polygons have multiple height clusters.
- OpenSeg/CLIP classification is a heavy dependency path and should remain optional until the graph and layout geometry are stable.
- Window detection depends on noisy semantic evidence and may require conservative thresholds to avoid false positives.
- Door/opening cutouts can produce invalid polygons when room boundaries are very short or self-intersecting; validation should record skipped entities instead of failing late.
