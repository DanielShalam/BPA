# BPA: The balanced-pairwise-affinities Feature Transform (ICML 2024)

This repository contains the official PyTorch implementation of **BPA** (formerly SOT) — the **B**alanced-**P**airwise-**A**ffinities feature transform — as described in our paper [*The Balanced-Pairwise-Affinities Feature Transform*](https://arxiv.org/abs/2407.01467), presented at ICML 2024.

![BPA](bpa_workflow.png?raw=true)

BPA enhances the representation of a set of input features to support downstream tasks such as matching or grouping.

The transformed features capture both **direct** pairwise similarity and **third-party agreement** — how other instances in the set influence similarity between a given pair. This enables BPA to model higher-order relations effectively.

---

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5?p=the-self-optimal-transport-feature-transform)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5-1?p=the-self-optimal-transport-feature-transform)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5-1?p=the-self-optimal-transport-feature-transform)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5?p=the-self-optimal-transport-feature-transform)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=the-self-optimal-transport-feature-transform)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=the-self-optimal-transport-feature-transform)

---

## Few-Shot Classification Results

| Dataset      | Method                 | 5-Way 1-Shot | 5-Way 5-Shot |
|--------------|------------------------|--------------|--------------|
| MiniImagenet | PTMAP-BPA<sub>p</sub>  | 83.19        | 89.56        |
|              | PTMAP-BPA<sub>t</sub>  | 84.18        | 90.51        |
|              | PTMAP-SF-BPA           | 85.59        | 91.34        |
| CIFAR-FS     | PTMAP-BPA<sub>p</sub>  | 87.37        | 91.12        |
|              | PTMAP-SF-BPA           | 89.94        | 92.83        |
| CUB          | PTMAP-BPA<sub>p</sub>  | 91.90        | 94.63        |
|              | PTMAP-SF-BPA           | 95.80        | 97.12        |

---

## Setup

### Datasets
Instructions for downloading and preparing the few-shot classification datasets are available in the `datasets` directory.

### Pretrained Models
Most results in the paper use fine-tuned models:

- **WideResNet-28**: [PT-MAP checkpoint](https://drive.google.com/file/d/1wVJlDnU00Gurs0pw54ZMqf4XsWhJWHIh/view)
- **ResNet-12**: [FEAT checkpoint](https://github.com/Sha-Lab/FEAT)

### BPA Checkpoint
We provide a checkpoint for [PTMAP-BPA<sub>t</sub>](https://drive.google.com/file/d/1wjh_EBQPYYHFjqoqlKCitcG9mjgkWFRw/view?usp=sharing), which yields:

- **84.69%** accuracy for 5-way 1-shot (vs. 84.18% in the paper)  
- **90.30%** accuracy for 5-way 5-shot (vs. 90.51% in the paper)

---

## Usage

### Quick Start

BPA can be applied in just two lines of code:

```python
import torch
from bpa import BPA

x = torch.randn(100, 128)  # [n_samples, dim]
x = BPA()(x)               # Output shape: [n_samples, n_samples]
```

---

### Training PT-MAP-BPA<sub>t</sub> on MiniImagenet

1. [Download](https://drive.google.com/file/d/1wVJlDnU00Gurs0pw54ZMqf4XsWhJWHIh/view) the pretrained WRN feature extractor.
2. Create an empty `checkpoints` directory.
3. Extract the downloaded file into the `checkpoints` folder.

Run the following command to train BPA with PT-MAP:

```bash
python train.py \
  --sink_iters 5 \
  --distance_metric cosine \
  --ot_reg 0.2 \
  --method pt_map_bpa \
  --backbone wrn \
  --augment false \
  --lr 5e-5 \
  --weight_decay 0. \
  --max_epochs 50 \
  --train_way 10 \
  --val_way 5 \
  --num_shot 5 \
  --num_query 15 \
  --train_episodes 200 \
  --eval_episodes 400 \
  --checkpoint_dir ./checkpoints/pt_map_bpa \
  --data_path <your_dataset_path/miniimagenet/> \
  --pretrained_path ./checkpoints/miniImagenet/WideResNet28_10_S2M2_R/470.tar
```

Alternatively, use the [trained checkpoint](https://drive.google.com/file/d/1wjh_EBQPYYHFjqoqlKCitcG9mjgkWFRw/view?usp=sharing).

---

### Evaluation

You can evaluate using the [original PT-MAP repository](https://github.com/yhu01/PT-MAP):

1. Clone the PT-MAP repository.
2. Navigate to `method/pt_map/evaluation/` and replace the files with the ones from this repository.
3. Edit the following:
   - Set `checkpoint_dir` in `save_plk.py` to the location of your BPA checkpoint.
   - Update `_datasetFeaturesFiles` in `FSLTask.py` to point to your feature files.
4. Create feature files by running:
```bash
python save_plk.py --dataset miniImagenet --method S2M2_R --model WideResNet28_10
```
5. Then run the evaluation:
```bash
   python test_standard.py
``` 

---

#### Alternatively, evaluate directly within this repository

You can run evaluation directly using the same script as training, with a few extra flags:

```bash
python train.py \
  --eval true \
  --pretrained_path <path_to_checkpoint> \
  --backbone <backbone_name> \
  --test_episodes 2000 ```
```

Make sure to include the same arguments you used during training (e.g., `--method`, `--data_path`, etc.) so the evaluation runs consistently.

---

### Logging
We support logging and visualization via Weights & Biases (wandb). This helps track your training and evaluation metrics in real-time.

**To enable logging:**

1. Install the `wandb` package:
```bash
pip install wandb
```

2. Add the following flags to your command:
```bash
--wandb true --project <project_name> --entity <wandb_entity>
```

Replace `<project_name>` with your wandb project name, and `<wandb_entity>` with your wandb username or team name.

---

## Citation

<p>

#### If you find this repository useful in your research, please cite:
    @article{shalam2024balanced,
      title={The Balanced-Pairwise-Affinities Feature Transform},
      author={Shalam, Daniel and Korman, Simon},
      journal={arXiv preprint arXiv:2407.01467},
      year={2024}
    }
    
</p>

---

## Acknowledgment
[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://github.com/yhu01/PT-MAP)

[S2M2 Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087.pdf)

[Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions](https://arxiv.org/pdf/1812.03664.pdf)
