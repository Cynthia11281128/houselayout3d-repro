# Inference code for MultiFloor3D: A Training-free Baseline for 3D Layout Estimation**

> **Note:** Obtaining the original Matterport3D dataset can take a while due to slow response times. While you wait, we provide a video of the fitting process and our complete predictions.

## Downloading the Matterport3D 2D Images

1. Follow the instructions at [https://niessner.github.io/Matterport/](https://niessner.github.io/Matterport/) to request and download the Matterport3D dataset.
2. Place the downloaded dataset in a directory of your choice, referenced below as `DATASET_ROOT`.

## Installation

Create the Conda environments required for the pipeline:

1. Follow **`./skeleton_extraction/OneFormer/INSTALL.md`** to create the **oneformer** environment.
2. Run **`./install.sh`** to create the **layout-estimation** environment.

## Running Inference

### 1. Configure Input / Output Paths

```bash
SCENE=WYY7iVyf5p8
DATASET_ROOT="/mnt/usb_ssd/bieriv/tmp/matterport_mesh_annot/${SCENE}"
OUTPUT_PATH="/mnt/usb_ssd/bieriv/segmented_matterport/suppl-matterport-oneformer/${SCENE}/all-classes"
mkdir -p "$OUTPUT_PATH"
```

### 2. Extract the Layout Skeleton

```bash
python extract_skeleton.py \
    --output-dir "${OUTPUT_PATH}/${SCENE}" \
    --poses-file "${DATASET_ROOT}/poses/${SCENE}.json" \
    --mesh-file  "${DATASET_ROOT}/matterport/v1/scans/${SCENE}/poisson_meshes/${SCENE}_10.ply"
```

### 3. Fit the Prototype

```bash
python fit_prototype.py \
    --scene-type matterport \
    --rectified-ply-path       "${OUTPUT_PATH}/clean_edge_mesh.ply" \
    --target-pcd-path          "${OUTPUT_PATH}/spt/ceiling_wall_floor_mesh.ply" \
    --target-vertex-classes    "${OUTPUT_PATH}/spt/ceiling_wall_floor_mesh_classes.npy" \
    --target-vertex-class-names "${OUTPUT_PATH}/simplified_segmentation_labels.npy" \
    --polygon-info-path        "${OUTPUT_PATH}/polygon_info.json" \
    --target-pcd-ray-origins   "${OUTPUT_PATH}/full_ray_origins.npy" \
    --target-pcd-ray-dests     "${OUTPUT_PATH}/full_ray_dests.npy" \
    --object-mesh              "${OUTPUT_PATH}/objects_mesh_simplified.ply" \
    --output-dir               "${OUTPUT_PATH}/all-classes" \
    --ray-classes              "${OUTPUT_PATH}/hard_labels_simplified_segmentations.npy"
```

### 4. Create the Scene Graph

```bash
python create_scene_graph.py \
    --scene                "$SCENE" \
    --base-polygon-set-path "$OUTPUT_PATH" \
    --output-dir            "$OUTPUT_PATH"
```
