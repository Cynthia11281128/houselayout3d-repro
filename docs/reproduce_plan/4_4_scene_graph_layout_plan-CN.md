# 4.4 单层场景图与布局导出迁移计划

## 目标

将论文 Section 4.4 迁移到 active `src` package，范围限定为单层场景。本阶段消费 Section 4.3 的 layout prototype，并产出最终的房间级 scene graph 和 dataset-style 3D layout entities。

实现应拆分为两个明确组件：

```text
4.3 prototype
  -> scene_graph
  -> layout
```

本计划有意不实现楼层级识别。目标场景默认视为一个 building level。因此 Appendix D.1 不在范围内，除非用于验证 prototype 中是否有足够的 floor geometry 来定义一个可用 level。

## 论文对齐

Section 4.4 要求将带语义标签的 prototype polygon set 转换为最终 layout，其表示形式是 scene graph：

1. 从 prototype floor 和 ceiling polygons 创建 2D floorplans。
2. 使用 walls 将每个 floorplan 分割成 rooms。
3. 创建 2D scene graph，其中 rooms 是节点，bottlenecks/openings 是边。
4. 使用 vision-language features 对 rooms 分类并剪枝。
5. 检测 stairs，并连接受影响的 rooms。
6. 将每个 room floorplan 挤出为 3D room shell。
7. 为 doors 和 stairs 插入 openings。
8. 从 geometrically inaccurate window/outdoor evidence 检测 windows，并将它们附着到 walls。
9. 导出 walls、floors、ceilings、doors、windows 和 stairs 的最终 polygons。

对本复现步骤，item 1 限定为单个默认 level。不进行多个 building floors 的分组，也不创建 multi-floor stair edge。

Appendix D 详细描述了相关算法：

- D.2：从 floor 和 ceiling polygons 创建 2D floorplan；
- D.3：使用 Hov-SG 的 morphology-based algorithm 做 room segmentation；
- D.4：room classification 和 outdoor-leaf pruning；
- D.5：从 stair mesh 检测 stairs；
- D.6：floor extrusion 期间处理 doors 和 stairs。

Appendix D.1 被跳过，因为在 single-floor assumption 下不需要 floor identification。

## 当前源码材料

相关 active code：

- `src/prototype_fitting/prototype.py`
  - 验证 Section 4.3 输入；
  - 冻结 optimizer inputs；
  - 记录最终 prototype mesh 和 serialized polygon state。
- `src/layout_skeleton/skeleton.py`
  - 输出 semantic mesh subsets，包括 stairs、objects、inaccurate surfaces 和 structural components；
  - 记录 per-vertex semantic labels 和 ray artifacts。

workspace 中已经存在的相关 external repositories：

- `external/HOV-SG`
  - morphology-based room segmentation 的候选来源；
  - graph construction conventions 的候选来源。
- `external/OpenScene`、`external/openseg-tpu` 和 `external/openai-clip`
  - 如果完整实现 Appendix D.4，可作为 OpenSeg/CLIP-style room classification 的候选来源。

当前还没有 active `src/scene_graph` 或 `src/layout_export` package。

## 建议的 Active Package Layout

创建两个新的 active packages：

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

将 Section 4.4 的 graph extraction 与最终 geometry export 分开：

- `scene_graph` 负责 2D floorplan construction、room cells、openings、room metadata、window candidates 和 stair graph edges。
- `layout_export` 负责 3D room extrusion、door/stair openings、final polygon entities 和 mesh/JSON export。

更新 `src/__main__.py`，展示：

```text
python src/scene_graph/graph.py --prototype PROTOTYPE --skeleton SKELETON --output OUTPUT
python src/layout_export/layout.py --scene-graph SCENE_GRAPH --prototype PROTOTYPE --output OUTPUT
```

最终 CLI 可以不同，但不应依赖 repo-global YAML config。

## Component 1: Scene Graph

### 目的

实现 Section 4.4 和 Appendix D.2-D.5 的 single-floor 子集：

```text
prototype polygons + skeleton semantic subsets
  -> one 2D floorplan
  -> room cells and opening edges
  -> optional room labels and pruning
  -> stair and window candidates
  -> scene graph manifest
```

### 输入

必需：

- completed `prototype/manifest.json`；
- final prototype mesh，通常是 `prototype/fitted_mesh.ply`；
- final prototype polygon state，通常是 `prototype/polygon_set_3d.pt`；
- completed `skeleton/manifest.json`；
- Section 4.2 的 structural semantic labels 和 class names；
- Section 4.2 的 stair mesh，如果非空；
- Section 4.2 的 inaccurate window/outdoor semantic evidence；
- 如果 room classification 或 window evidence projection 需要 image-space features，则需要 original camera images 和 transforms。

推荐 CLI：

```bash
python src/scene_graph/graph.py \
  --prototype data/insta360/r04/prototype \
  --skeleton data/insta360/r04/skeleton \
  --output data/insta360/r04/scene_graph
```

### 单层假设

该组件不应推断 building levels。取而代之：

1. 加载所有 floor-classified prototype polygons。
2. 如果不存在 floor polygon，则拒绝运行。
3. 从 floor polygons 计算一个 representative floor elevation，使用 area-weighted mean 或 robust median。
4. 将所有 suitable ceiling polygons、wall polygons、rooms、openings、windows 和 stairs 分配到这个单一 level。
5. 在 manifest 中记录 `"single_floor_assumption": true`。

如果出现多个分离的 floor elevations，组件应给出 warning，但除非提供显式 `--floor-height` override，否则仍选择一个默认 level。

### 算法

#### 1. 加载 Prototype Geometry

尽可能加载 final prototype polygon state。优先使用 serialized topology 和 semantic class probabilities，而不是 PLY colors。final prototype mesh 只作为 validation 和 visualization artifact 使用。

需要规范化的内部表示：

```text
PrototypePolygon:
  id
  class_name
  vertices_3d
  plane
  normal
  area
```

实现应保留 original polygon IDs，使 final entities 可以引用 source prototype polygons。

#### 2. 创建单个 2D Floorplan

使用 global Z-up convention 将 floor polygons 投影到 BEV。将其与 suitable ceiling polygons 合并：

- floor polygons：所有分类为 `floor` 的 prototype polygons；
- ceiling polygons：分类为 `ceiling`，且 centroid 至少高于 representative floor height 1m 的 polygons；
- optional surfaces：只有当 `surface` 的 normal 和 elevation 使其类似 floor 或 ceiling 时才包含，并将此记录为 warning。

使用 Shapely 等 geometry library 计算 robust 2D union。输出是一个 floorplan polygon 或 multipolygon。保留 holes。

#### 3. 为 Room Segmentation 选择 Walls

选择符合以下条件的 wall-classified prototype polygons：

- 在 BEV 中与 2D floorplan 相交；
- 在垂直方向上与 `[floor_height, floor_height + 2.5m]` 重叠；
- normals 基本垂直。

将选中的 wall polygons 转换为适用于 room segmentation 的 2D wall masks 或 line barriers。

#### 4. 分割 Rooms

按照 Appendix D.3 应用 morphology-based room segmentation：

- 先用 bottleneck width `2.5m` 分割；
- 再用 bottleneck width `1.5m` 分割；
- 为每个 resulting cell 创建一个 room node；
- 为 cells 之间的 bottlenecks 创建 graph edges；
- 将 width 小于 `1.5m` 的 edges 分类为 `door`，否则分类为 `opening`。

实现选项：

- 如果 Hov-SG 的 room segmentation code 能被干净地隔离，则复用它；
- 否则使用 OpenCV/scikit-image/Shapely 在本地实现 morphology operations，并匹配文档中的 bottleneck 行为。

在 manifest 中记录所选择的实现路径。

#### 5. 可选 Room Classification 和 Pruning

Appendix D.4 使用 OpenSeg features 和 CLIP room text embeddings。将其实现为独立的可选子步骤：

- `--enable-room-classification` 计算 room labels；
- 默认行为可以将每个 room 标为 `"unknown"`；
- outdoor-leaf pruning 只应在启用 room classification 且 confidence 足够时运行。

Room classes 应尽可能匹配论文：

```text
bathroom, bedroom, living room, garage, entrance, kitchen, office,
stairs, gym, classroom, spa/sauna, mirror, grass/bushes/trees,
driveway, veranda/terrace/balcony, unknown
```

这样可以在 OpenSeg/CLIP dependency path 完全稳定之前，保持 graph component 可用。

#### 6. 检测 Stairs

对于 single-floor case，stairs 不连接两个 inferred floor levels。仍然需要检测 stair geometry，使最终 layout 可以预留 stair openings 并导出 stair entities。

算法：

1. 加载 Section 4.2 提取的 stair mesh。
2. 聚类 connected components。
3. 将每个 component 投影到 BEV。
4. 用 oriented bounding rectangle 近似每个 component。
5. 将 rectangle 分配给最近的 room，或所有相交的 rooms。
6. 在 scene graph 中将其存储为 `stair_region`。

除非未来 multi-floor extension 添加 floor identification，否则不创建 inter-floor edges。

#### 7. 检测 Windows

Window detection 应放在 graph-side，因为 final windows 需要 wall 和 room references。

使用论文中的 window evidence：

- inaccurate window class；
- 通过 windows 可见的 outdoor/noise classes；
- 可选地将 curtains 和 window blinds 用于 window detection；
- 从 window detection 中排除 mirrors。

建议算法：

1. 收集 window/outdoor classes 的 skeleton/prototype vertices 或 ray evidence。
2. 将 evidence points 分配给附近 wall polygons。
3. 对每个 wall 使用 DBSCAN 聚类 evidence。
4. 在 wall plane 中为每个 cluster 拟合 axis-aligned rectangle。
5. 保留 width 和 height 都大于 `0.30m` 的 rectangles。
6. 将每个 window 附着到最近的 room-side wall segment。

### 输出

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

`graph.json` 应包含：

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

### 验证

- prototype 和 skeleton manifests complete，且 hashes match；
- 声明 exactly one default level；
- floorplan geometry valid 且 non-empty；
- 每个 room polygon 都位于 floorplan 内，允许 small numeric tolerance；
- 每个 opening 引用两个 valid rooms，或一个 room 加 exterior；
- door/opening widths finite 且 positive；
- 每个 window 在可分配时引用 valid wall source 和 room；
- 每个 stair region 至少与一个 valid room 相交，或作为 unassigned 记录 warning；
- graph JSON schema-valid 且 reloadable。

## Component 2: Layout Export

### 目的

将 scene graph 和 prototype geometry 转换为最终 Section 4.4 layout entities：

```text
scene graph + prototype polygons
  -> per-room 3D shells
  -> door/opening cutouts
  -> stair geometry
  -> window rectangles
  -> final layout mesh and entity JSON
```

### 输入

必需：

- completed `scene_graph/manifest.json`；
- completed `prototype/manifest.json`；
- final prototype polygon state；
- 用于 validation 和 debug comparison 的 final prototype mesh。

推荐 CLI：

```bash
python src/layout_export/layout.py \
  --scene-graph data/insta360/r04/scene_graph \
  --prototype data/insta360/r04/prototype \
  --output data/insta360/r04/layout
```

### 算法

#### 1. 确定 Room Heights

对每个 room：

- floor elevation 来自 scene graph 中的 single floor height；
- ceiling height 应优先来自 room 上方的 ceiling polygons；
- fallback 到 robust scene-level ceiling height；
- 逐 room 记录 fallback 使用情况。

#### 2. 挤出 Room Floorplans

将每个 2D room cell 挤出为 closed 3D shell：

- floor face；
- ceiling face；
- wall faces along boundary segments。

在所有 generated faces 上保留 room IDs。

#### 3. 插入 Doors 和 Openings

对每条 graph edge：

- 用 oriented 2D rectangle 近似 splitting boundary；
- 对 `door` edges 使用固定 door height `2.10m`；
- 对更宽的 `opening` edges，按 configured opening height 保持开放；
- 必要时添加 door/opening 上方的 wall triangles；
- 避免为 shared room boundaries 重复生成 wall faces。

Door entities 应引用：

- connected room IDs；
- 2D opening geometry；
- 3D rectangle 或 frame geometry；
- source graph edge ID。

#### 4. 插入 Stair Openings 和 Stair Geometry

对每个 `stair_region`：

- 如果可行，在 extrusion 前从相交 room floorplans 中减去 stair footprint；
- 使用 scene graph 中的 oriented rectangle 添加 stair entity；
- 在 single-floor case 中，除非有可靠 endpoint heights，否则将 stair 表示为 pitched 或 flat reserved region；
- 不创建第二个 floor shell。

#### 5. 插入 Windows

对每个 detected window rectangle：

- 在 assigned wall plane 上创建 3D rectangular window entity；
- 将 rectangle clip 或 validate 到 wall face；
- 引用 owning room 和 source wall/prototype polygon；
- 保留 detection 时使用的 width/height thresholds。

#### 6. 导出最终 Entity Set

Final entities 应覆盖：

```text
wall
floor
ceiling
door
opening
window
stairs
```

每个 entity 应包含：

- stable ID；
- semantic class；
- room ID 或 graph edge ID；
- 3D vertices；
- 可用时包含 source prototype polygon IDs；
- 可用时包含 source scene-graph object ID。

### 输出

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

`layout.json` 应作为 canonical machine-readable output。Mesh files 是 visualization/export artifacts。

### 验证

- scene graph manifest complete 且 hashes match；
- 每个 room 都生成 finite floor、ceiling 和 wall geometry；
- 每个 generated polygon 至少有三个 vertices，且 area 非零；
- exported vertices 中没有 NaN/Inf；
- door/opening/window/stair entities 引用 valid graph IDs；
- 当某个 class 存在时，对应 class-specific mesh files reloadable 且 non-empty；
- room shells 的 topology 对 downstream evaluation 足够一致；
- manifest 记录任何 fallback ceiling heights、skipped windows、unassigned stairs 或 invalid room cells。

## 实现步骤

1. 创建 `src/scene_graph/__init__.py` 和 `src/layout_export/__init__.py`。
2. 实现 prototype polygon-state loading，并使用稳定的内部 `PrototypePolygon` representation。
3. 实现从 floor 和 ceiling polygons 创建 single-floor floorplan。
4. 实现 wall selection 和 BEV wall mask generation。
5. 隔离或重新实现 Hov-SG-style morphology room segmentation。
6. 输出 `scene_graph/graph.json`、GeoJSON debug outputs 和 manifest validation。
7. 将 room classification 作为独立 flag 添加；默认保持 rooms 为 `unknown`。
8. 为 single-floor stair regions 实现 stair footprint extraction。
9. 实现 window detection 和 wall assignment。
10. 实现 room shell extrusion。
11. 实现 door/opening/stair cutouts during extrusion。
12. 实现 final entity JSON 和 class mesh exports。
13. 更新 `src/__main__.py` 和 `docs/steps.md`，加入新 commands。
14. 为 floorplan union、room segmentation fixtures、opening width classification、extrusion 和 entity schema validation 添加 focused tests。
15. 在真实 4.3 prototype manifest 存在后，对 `r04` 运行完整 4.4 path。

## 验收标准

Section 4.4 single-floor migration 完成的条件：

- `scene_graph/manifest.json` 可以从真实 4.3 prototype 完成；
- `scene_graph/graph.json` 包含 one level、valid room nodes、valid opening edges，以及任何 detected windows/stairs；
- `layout/manifest.json` 可以从 scene graph 完成；
- `layout/layout.json` 包含最终 wall、floor、ceiling、door/opening、window 和 stair entities（如果存在）；
- exported meshes finite、reloadable，并在 manifest 中声明 hashes；
- 不需要 floor-identification logic 即可解释输出；
- 所有与 multi-floor paper algorithm 的偏差都记录在 manifests 中。

## 已知风险

- 如果 unofficial optimizer 将 topology 存在跨环境不稳定的 PyTorch object 中，final prototype serialized state 可能需要 adapter code。
- Hov-SG room segmentation 可能难以直接复用；本地 morphology implementation 可能更可复现。
- Single-floor assumptions 可能掩盖真实 split-level geometry；当 floor polygons 有多个 height clusters 时，manifest 应给出 warning。
- OpenSeg/CLIP classification 是 heavy dependency path，在 graph 和 layout geometry 稳定前应保持 optional。
- Window detection 依赖 noisy semantic evidence，可能需要保守 thresholds 以避免 false positives。
- 当 room boundaries 很短或 self-intersecting 时，door/opening cutouts 可能产生 invalid polygons；validation 应记录 skipped entities，而不是在后期失败。
