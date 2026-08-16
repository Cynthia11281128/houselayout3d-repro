## Steps

### rgb-to-mesh
```bash
ROOT=data/insta360/r04

conda run -n houselayout3d-layout python src/rgb_to_mesh/metric3d.py \
    --images ${ROOT}/images \
    --output ${ROOT}/metric3d

conda run -n nerfstudio python src/rgb_to_mesh/dn_splatter.py \
    --output ${ROOT}/dn_splatter \
    --transforms ${ROOT}/transforms.json \
    --images ${ROOT}/images \
    --depth ${ROOT}/metric3d/depth

conda run -n nerfstudio python src/rgb_to_mesh/mesh.py \
    --dn-splatter ${ROOT}/dn_splatter \
    --output ${ROOT}/mesh
```

### layout-skeleton
```bash
ROOT=data/insta360/r04

conda run -n houselayout3d-layout python src/layout_skeleton/oneformer.py \
    --images ${ROOT}/images \
    --model-dir pretrained_weights/oneformer_coco_swin_large \
    --output ${ROOT}/oneformer

NS_RENDER=/tmp/tmp_data/miniconda3/envs/nerfstudio/bin/ns-render

conda run --no-capture-output -n houselayout3d-layout python src/layout_skeleton/skeleton.py \
      --transforms ${ROOT}/transforms.json \
      --dn-splatter ${ROOT}/dn_splatter \
      --mesh ${ROOT}/mesh/export/DepthAndNormalMapsPoisson_poisson_mesh.ply \
      --oneformer ${ROOT}/oneformer \
      --ns-render ${NS_RENDER} \
      --superpoint-repo external/superpoint-transformer \
      --output ${ROOT}/skeleton \
      --overwrite
```

### prototype-fitting
```bash
ROOT=data/insta360/r04

conda run --no-capture-output -n houselayout3d-layout python src/prototype_fitting/polygon_init.py \
    --skeleton ${ROOT}/skeleton \
    --superpoint-level 3 \
    --plane-distance-threshold-meters 0.04 \
    --minimum-unassigned-vertices 100 \
    --ransac-iterations 256 \
    --rdp-epsilon-meters 0.03 \
    --output ${ROOT}/polygon_init

conda run --no-capture-output -n houselayout3d-layout python src/prototype_fitting/prototype.py \
    --skeleton ${ROOT}/skeleton \
    --polygon-init ${ROOT}/polygon_init \
    --source-repo MultiFloor3D-unofficial \
    --python /tmp/tmp_data/miniconda3/envs/houselayout3d-layout/bin/python \
    --output ${ROOT}/prototype \
    --preferred-gpu 1
```

### scene-graph
```bash
ROOT=data/insta360/r04

conda run --no-capture-output -n houselayout3d-layout python src/scene_graph/graph.py \
    --prototype ${ROOT}/prototype \
    --skeleton ${ROOT}/skeleton \
    --output ${ROOT}/scene_graph
```

### layout-export
```bash
ROOT=data/insta360/r04

conda run --no-capture-output -n houselayout3d-layout python src/layout_export/layout.py \
    --scene-graph ${ROOT}/scene_graph \
    --prototype ${ROOT}/prototype \
    --output ${ROOT}/layout
```

