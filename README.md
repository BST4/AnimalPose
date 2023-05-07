- [MLDA-BST4: Animal Pose Estimation](#mlda-bst4-animal-pose-estimation)
  - [Introduction](#introduction)
    - [Base Model](#base-model)
    - [Dataset](#dataset)
  - [Installation](#installation)
  - [Train](#train)
    - [Animal datasets (AP10K, APT36K)](#animal-datasets-ap10k-apt36k)
  - [Evaluation](#evaluation)
  - [Todo](#todo)
  - [Acknowledge](#acknowledge)
  - [Citing ViTPose](#citing-vitpose)

# MLDA-BST4: Animal Pose Estimation

## Introduction
In this project, our goal is to develop an API for a downstream animal classification task based on ViTPose, a state-of-the-art animal pose estimation algorithm. The API will provide users with a convenient and efficient way to extract pose features from animal images or videos, which can then be used for various downstream tasks, such as animal identification, behavior analysis, and disease diagnosis.

To achieve this goal, we will adapt ViTPose to output pose features that are compatible with the downstream classification task. We will also optimize the API for speed and scalability, so that it can handle large-scale datasets and real-time applications. Additionally, we will provide a user-friendly interface and comprehensive documentation to facilitate the adoption of our API by researchers and practitioners in the animal science community.

Overall, our project aims to bridge the gap between animal pose estimation and downstream tasks by providing a robust and accessible tool for animal classification and analysis.

### Base Model
ViTPose is a state-of-the-art algorithm for animal pose estimation, based on the Vision Transformer (ViT) architecture. ViTPose is designed to automatically detect and track the body parts of animals in images or videos, which is a challenging task due to the high variability in animal poses, appearances, and environmental conditions.

### Dataset
AP10K is a large-scale benchmark for general animal pose estimation, designed to facilitate research in this field. The dataset consists of 10,015 images collected and filtered from 23 animal families and 54 species, with high-quality keypoint annotations. Additionally, it contains another 50k images with family and species labels. The dataset can be used for supervised learning, cross-domain transfer learning, and intra- and inter-family domain generalization for unseen animals.

## Installation
We use PyTorch 1.9.0 or NGC docker 21.06, and mmcv 1.3.9 for the experiments.
```bash
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout v1.3.9
MMCV_WITH_OPS=1 pip install -e .
cd ..
git clone https://github.com/ViTAE-Transformer/ViTPose.git
cd ViTPose
pip install -v -e .
```

After install the two repos, install timm and einops, i.e.,
```bash
pip install timm==0.4.9 einops
```

## Train
### Animal datasets (AP10K, APT36K)

> Results on AP-10K test set

| Model | Dataset | Resolution | AP | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 71.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 74.5 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_base_ap10k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 80.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_large_ap10k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 82.4 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_huge_ap10k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

> Results on APT-36K val set

| Model | Dataset | Resolution | AP | config | weight |
| :----: | :----: | :----: | :----: | :----: | :----: |
| **ViTPose+-S** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 74.2 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_small_apt36k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccqO1JBHtBjNaeCbQ?e=ZN5NSz) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccrwORr61gT9E4n8g?e=kz9sz5) |
| **ViTPose+-B** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 75.9 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_base_apt36k_256x192.py)  | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccjj9lgPTlkGT1HTw?e=OlS5zv) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgcckRZk1bIAuRa_E1w?e=ylDB2G) |
| **ViTPose+-L** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 80.8 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_large_apt36k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgccp7HJf4QMeQQpeyA?e=JagPNt) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccs1SNFUGSTsmRJ8w?e=a9zKwZ) |
| **ViTPose+-H** | COCO+AIC+MPII+AP10K+APT36K+WholeBody | 256x192 | 82.3 | [config](configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/apt36k/ViTPose_huge_apt36k_256x192.py) | [log](https://1drv.ms/u/s!AimBgYV7JjTlgcclxZOlwRJdqpIIjA?e=nFQgVC) \| [Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgccoXv8rCUgVe7oD9Q?e=ZBw6gR) |

After downloading the ViTPose+ pretrained models, please first re-organize the pre-trained weights using

```bash
python tools/model_split.py --source <Pretrained PATH>
```

Conduct the experiments by running

```bash
# for single machine
bash tools/dist_train.sh <Config PATH> <NUM GPUs> --cfg-options model.pretrained=<Pretrained PATH> --seed 0

# for multiple machines
python -m torch.distributed.launch --nnodes <Num Machines> --node_rank <Rank of Machine> --nproc_per_node <GPUs Per Machine> --master_addr <Master Addr> --master_port <Master Port> tools/train.py <Config PATH> --cfg-options model.pretrained=<Pretrained PATH> --launcher pytorch --seed 0
```

## Evaluation

To test the pretrained models performance, please run 

```bash
bash tools/dist_test.sh <Config PATH> <Checkpoint PATH> <NUM GPUs>
```

## Todo

This repo current contains modifications including:

- [x] Upload configs and pretrained models

- [x] More models with SOTA results

- [x] Upload multi-task training config

## Acknowledge
We acknowledge the excellent implementation from [mmpose](https://github.com/open-mmlab/mmdetection) and [MAE](https://github.com/facebookresearch/mae).

## Citing ViTPose

For ViTPose

```
@inproceedings{
  xu2022vitpose,
  title={Vi{TP}ose: Simple Vision Transformer Baselines for Human Pose Estimation},
  author={Yufei Xu and Jing Zhang and Qiming Zhang and Dacheng Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022},
}
```

For ViTPose+

```
@article{xu2022vitpose+,
  title={ViTPose+: Vision Transformer Foundation Model for Generic Body Pose Estimation},
  author={Xu, Yufei and Zhang, Jing and Zhang, Qiming and Tao, Dacheng},
  journal={arXiv preprint arXiv:2212.04246},
  year={2022}
}
```

For ViTAE and ViTAEv2, please refer to:
```
@article{xu2021vitae,
  title={Vitae: Vision transformer advanced by exploring intrinsic inductive bias},
  author={Xu, Yufei and Zhang, Qiming and Zhang, Jing and Tao, Dacheng},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

@article{zhang2022vitaev2,
  title={ViTAEv2: Vision Transformer Advanced by Exploring Inductive Bias for Image Recognition and Beyond},
  author={Zhang, Qiming and Xu, Yufei and Zhang, Jing and Tao, Dacheng},
  journal={arXiv preprint arXiv:2202.10108},
  year={2022}
}
```
