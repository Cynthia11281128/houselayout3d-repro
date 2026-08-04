# Runtime environments

MultiFloor3D spans incompatible framework stacks, so the reproduction uses
three explicit environments instead of silently changing package versions.

| Role | Prefix | Core stack |
| --- | --- | --- |
| Reconstruction | `/tmp/tmp_data/miniconda3/envs/nerfstudio` | Python 3.8, PyTorch 2.1.2+cu118, nerfstudio 1.1.3, gsplat 1.0.0, editable DN-Splatter |
| Layout | `/tmp/tmp_data/miniconda3/envs/houselayout3d-layout` | Python 3.10, PyTorch 2.2.0+cu118, Metric3D, OneFormer, PyTorch3D, SPT, FRNN, HOV-SG, OpenAI CLIP |
| OpenSeg | `/tmp/tmp_data/miniconda3/envs/houselayout3d-openseg` | Python 3.10, TensorFlow 2.15.1, CUDA 12.2 runtime wheels |

The reconstruction environment is the user's existing nerfstudio environment.
DN-Splatter is installed editable from this reproduction's pinned checkout;
its existing PyTorch and nerfstudio versions are preserved.

The layout environment contains CUDA 11.8 development libraries and extensions
built for the RTX 4090 (`sm_89`). Its `.pth` file exposes the pinned Metric3D,
OneFormer, and Superpoint Transformer checkouts without copying their sources.
The `run-skeleton` command intentionally launches the pinned reconstruction
environment's `ns-render` for final-checkpoint raw depth, then continues in the
layout environment for ray voting and Superpoint Transformer preprocessing.

Exact key package versions are recorded in
`references/environment_versions.json`. Verify all environments from the
project root with:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/smoke_test_envs.sh
```

The smoke test intentionally fails if a required GPU, model file, entry point,
or compiled extension is missing. It does not download or modify anything.
