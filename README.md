# SOT: The Self-Optimal-Transport Feature Transform

This repository provides the official PyTorch implementation and pretrained models for **SOT** (The **S**elf-**O**ptimal-**T**ransport), as described in the paper [The Self-Optimal-Transport Feature Transform](https://arxiv.org/abs/2204.03065).

![SOT](https://i.ibb.co/m8Nw7gx/SOT.png)

The Self-Optimal-Transport (SOT) feature transform is designed to upgrade the set of features of a data instance to facilitate downstream matching or grouping related tasks. 

The transformed set encodes a rich representation of high order relations between the instance features. Distances  between transformed features capture their **direct** original similarity and their **third party** 'agreement' regarding similarity to other features in the set. 

A particular min-cost-max-flow fractional matching problem, whose entropy regularized version can be approximated by an optimal transport (OT) optimization, results in our transductive transform which is efficient, differentiable, equivariant, parameterless and probabilistically interpretable.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cifar-fs-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cifar-fs-5-1?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5-1)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5-1?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-cub-200-5)](https://paperswithcode.com/sota/few-shot-image-classification-on-cub-200-5?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-2)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-2?p=the-self-optimal-transport-feature-transform)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/the-self-optimal-transport-feature-transform/few-shot-image-classification-on-mini-3)](https://paperswithcode.com/sota/few-shot-image-classification-on-mini-3?p=the-self-optimal-transport-feature-transform)

## Few-shot classification results

| First Header  | Second Header |
| ------------- | ------------- |
| Content Cell  | Content Cell  |
| Content Cell  | Content Cell  |

## Running instructions
### Clustering on the sphere
We provide the code to reproduce the syntethic expriment as described in the paper.
This can be beneficial in order to expriement with the SOT on a controlled data and can be used as a benchmark.
    
To run the expriemnt with the default arguments, simply run the script:

        syntethic_exp/eval_unit.py
        
The script includes a variety of paramters that controls the structre of the data as well as additional plots configurations.
    
### Few-Shot Classification

<details><summary>Dataset </summary>
<p>

    <details><summary>Datasets </summary>
    <p>
    </p>
</p>
</details>

<details><summary>Running PT-MAP-SOT<sub>p</sub> </summary>
<p>

Download and extract the featrues using the instructions on the [PT-MAP repository](https://github.com/yhu01/PT-MAP).
Then, run ....

</p>
</details>

<details><summary>Running PT-MAP-SOT<sub>t</sub> </summary>
<p>

Download the S2M2_R weights as described [PT-MAP repository](https://github.com/nupurkmr9/S2M2_fewshot)

</p>
</details>

<details><summary>Pretrained Models </summary>
<p>

All pretrained weights and features for the PT-MAP-SOT<sub>p</sub> expriment can be downloaded from the [PT-MAP repository](https://github.com/yhu01/PT-MAP)

</p>
</details>

## Citation

<p>

#### If you find this repository useful in your research, please cite:

    @article{shalam2022self,
      title={The Self-Optimal-Transport Feature Transform},
      author={Shalam, Daniel and Korman, Simon},
      journal={arXiv preprint arXiv:2204.03065},
      year={2022}
    }

</p>

## Acknowledgment
