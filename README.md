<br>

<video src="https://github.com/asamu12/CARIS/blob/main/issue/grasp.mp4" autoplay loop muted width="500">
  Your browser does not support the video tag.
</video>

https://github.com/user-attachments/assets/4be32150-1893-42ae-aa94-eeccbd94f33a



### Introduction

This project is a demo version, used for testing and verifying the effectiveness of RISgrasp. The Complete version will be uploaded after acceptance.
![Image](https://github.com/asamu12/CARIS/blob/main/fig/Overview.jpg)


## Environment
```bash
conda create -n DETRIS python=3.9.18 -y
conda activate DETRIS
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirement.txt
```

## Datasets
The detailed instruction is in [prepare_datasets.md](tools/prepare_datasets.md)

## Pretrained weights
Download the pretrained weights of DiNOv2-B, DiNOv2-L and ViT-B to pretrain
```bash
mkdir pretrain && cd pretrain
## DiNOv2-B
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth
## DiNOv2-L
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth
## ViT-B
wget https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

