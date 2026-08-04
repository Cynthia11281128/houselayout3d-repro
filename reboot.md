This readme is used as a guide for codex to re-download required files to targeting folders after the server is rebooted / data is missing.

### Weights
All weights should be downloaded to `/tmp/tmp_data/weights` and then symlink to targeting path. 
1. [metric3D](https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_large_800k.pth)
2. [OneFormer COCO Swin-L HuggingFace model files](https://huggingface.co/shi-labs/oneformer_coco_swin_large)
3. [OneFormer COCO Swin-L original checkpoint](https://shi-labs.com/projects/oneformer/coco/150_16_swin_l_oneformer_coco_100ep.pth)
4. [OpenSeg saved_model.pb](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/colab/exported_model/saved_model.pb)
5. [OpenSeg variables.data](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/colab/exported_model/variables/variables.data-00000-of-00001)
6. [OpenSeg variables.index](https://storage.googleapis.com/cloud-tpu-checkpoints/detection/projects/openseg/colab/exported_model/variables/variables.index)
7. [OpenAI CLIP ViT-L/14@336px](https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt)
8. [Torchvision AlexNet LPIPS](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth)

### Environment
All environment should be setup in `/tmp/tmp_data/miniconda3`, this folder is symlinked to `~/miniconda3`.

1. [nerfstudio]
2. [houselayout3d-layout]
3. [houselayout3d-openseg]

For details, please refer to [environments](environments.md).
