# BPA: The balanced-pairwise-affinities Feature Transform (ICML 2024)

This repository provides the official PyTorch implementation for **BPA** (former SOT) (The **B**alanced-**P**airwise-**A**ffinities),
as described in the paper [The Balanced-Pairwise-Affinities Feature Transform](https://arxiv.org/abs/2407.01467) (Accepted by ICML 2024).

![BPA](bpa_workflow.png?raw=true)

The Balanced-Pairwise-Affinities (BPA) feature
transform is designed to upgrade the features of a
set of input items to facilitate downstream matching or grouping related tasks.

The transformed set encodes a rich representation of high order relations between the instance features. 
Distances between transformed features capture their **direct** original similarity, and 
their **third party** 'agreement' regarding similarity to other features in the set. 


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5-1?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5-1?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=the-self-optimal-transport-feature-transform)

## Few-Shot Classification Results

| Dataset       | Method                 | 5-Way 1-Shot  | 5-Way 5-Shot  |
| ------------- |-------------           | ------------- | ------------- |
| MiniImagenet  | PTMAP-BPA<sub>p</sub>  | 83.19         | 89.56         |
|               | PTMAP-BPA<sub>t</sub>  | 84.18         | 90.51         |
|               | PTMAP-SF-BPA           | 85.59         | 91.34         |
|               |                        |               |               |
| CIFAR-FS      | PTMAP-BPA<sub>p</sub>  | 87.37         | 91.12         |
|               | PTMAP-SF-BPA           | 89.94         | 92.83         |
|               |                        |               |               |
| CUB           | PTMAP-BPA<sub>p</sub>  | 91.90         | 94.63         |
|               | PTMAP-SF-BPA           | 95.80         | 97.12         |

## Using BPA in your own project with only 2 lines of code
BPA flexibility allow to improve your set representation with only 2 additional lines of code!

```
import torch
from bpa import BPA

x = torch.randn(100, 128) # x is of shape [n_samples, dim]
x = BPA()(x)
# after BPA, x shape is [n_samples, n_samples]
```


## Running instructions
We provide the code for training and evaluating PT-MAP and ProtoNet with and without BPA.
Note that the results from the paper are not reproducible here. 
To fully reproduce the results, use the BPA as shown here, in the original repositories.

Find instructions on how to download the datasets under the datasets dir.

### Training
For now, you can choose between ProtoNet/PT-MAP including their BPA variations.

For example, to train our ProtoNet+BPA version on the MiniImagenet dataset using WideResnet as a backbone, run:

```
python train.py --data_path <yourdatasetpath/miniimagenet/> --backbone WRN --method proto_bpa --ot_reg 0.1 --max_epochs 200 --train_way 5 --scheduler step --step_size 40 --lr 0.0002  --augment false
```

We also support logging results into the cloud using the Wandb logger (highly suggested).

First, install it via:
```
pip install wandb
```

Then, set the following arguments:
```
--wandb true --project <project_name> --entity <wandb_entity>
```

### Fine-tuning
For WRN-28, we use the pretrained checkpoint from [Manifold Mixup repository](https://github.com/nupurkmr9/S2M2_fewshot).

For Resnet-12, we use the pretrained checkpoint from [FEAT](https://github.com/Sha-Lab/FEAT).

Dowload the weights according to the backbone you want and set:
```
--backbone <model name> --pretrained_path <./path>
```

### Evaluation
Run the same you used for training with:
```
--eval true --pretrained_path <./path> --backbone <backbone_name>
```
You can choose the number of episodes by modify
```
--test_episodes
```

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

## Acknowledgment
[Leveraging the Feature Distribution in Transfer-based Few-Shot Learning](https://github.com/yhu01/PT-MAP)

[S2M2 Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087.pdf)

[Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions](https://arxiv.org/pdf/1812.03664.pdf)
