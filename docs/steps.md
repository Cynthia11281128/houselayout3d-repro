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
