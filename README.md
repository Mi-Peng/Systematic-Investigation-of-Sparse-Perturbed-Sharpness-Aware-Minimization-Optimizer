# Systematic-Investigation-of-Sparse-Perturbed-Sharpness-Aware-Minimization-Optimizer
This is the official implementation of paper [Systematic Investigation of Sparse Perturbed Sharpness-Aware Minimization Optimizer](https://arxiv.org/abs/2306.17504)


## Installation

<details open>
<summary>  Clone this repo  </summary>

```bash
git clone git@github.com:Mi-Peng/Systematic-Investigation-of-Sparse-Perturbed-Sharpness-Aware-Minimization-Optimizer.git
```
</details>

<details open>
<summary>  Create a virtual environment (e.g. Anaconda3) </summary>

```bash
conda create -n ssam python=3.8 -y
conda activate ssam
```
</details>

<details open>
<summary> Install the necessary packages </summary>

1. Pytorch

Install Pytorch following the [official installation instructions](https://pytorch.org/get-started/locally/).

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

2. cusparseLt

Details could be found in [cusparseLt.md](cusparseLt.md)

3. Install other packages
```bash
pip install einops
```

4. Install wandb(optional)

[Wandb](https://wandb.ai/site) makes it easy to track your experiments, manage & version your data. This is optional, codes run without wandb.
```bash
pip install wandb
```

5. Dataset preparation
We use CIFAR10, CIFAR100 and ImageNet in this repo.

For the CIFAR dataset, you don't need to do anything, pytorch will do the trivia about downloading.

For ImageNet dataset, we use standard ImageNet dataset, which could be found in http://image-net.org/. Your ImageNet file structure should look like:

```bash
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```
</details>

## Configuration
Details are in [configs/defaulf_cfg.py](configs/defaulf_cfg.py).

- `--dataset`: Currently supported choice include: `CIFAR10_base`, `CIFAR10_cutout`, `CIFAR100_base`, `CIFAR100_cutout` and `ImageNet_base`,.
- `--model`: Currently supported choice include: `resnet18`, `wideresnet28x10`, ...(See more in [models/\_\_init\_\_.py](models/__init__.py))
- `--opt`: How to update parameters. `--sgd` for SGD, `--sam-sgd` for SAM within SGD, `--ssamf-sgd` for Fisher-SparseSAM within SGD.
- `--pattern`. pattern of masks. Currently supported choice include: `unstructured`, `structured`, `nm`.
- `--n_structured` and `--m_structured`. Set `n` and `m` in `nm` pattern (Only works for `nm` pattern).

- `--implicit`. Whether use mask to calculate sparse perturbation implicitly, and must add argument `--samconv` or `--culinear ` to transform the backpropagation.
- `--samconv`. Transform the convolution layer for implicit sparse perturbation.(For ResNet)
- `--culinear`. Transform the linear layer for implicit sparse perturbation.(For vit_testspmm)
 

## Training

<details open>
<summary>  Training model on CIFAR10 with SGD (Taking ResNet18 as an example)</summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR10_cutout --datadir [Path2Data] \
  --opt sgd --lr 0.05 --weight_decay 5e-4 \
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SAM (Taking ResNet18 as an example)</summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt sam-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SSAM, <b>unstructured</b> mask, <b>explicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt ssamf-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern unstructured --sparsity 0.5 --num_samples 128 --update_freq 1 \
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SSAM, <b>structured</b> mask, <b>explicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt ssamf-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern structured --sparsity 0.5 --num_samples 128 --update_freq 1 \
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SSAM, <b>N:M</b> mask, <b>explicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt ssamf-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern nm --n_structured 2 --m_structured 4 --num_samples 128 --update_freq 1 \
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SSAM, <b>structured</b> mask, <b>implicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt sam-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern structured --sparsity 0.5 --num_samples 128 --update_freq 1 --implicit --samconv\
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Training model on CIFAR10 with SSAM, <b>N:M</b> mask, <b>implicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model resnet18 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt sam-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern nm --n_structured 2 --m_structured 4 --num_samples 128 --update_freq 1 --implicit --samconv\
  --seed 1234 --wandb
```
</details>

<details open>
<summary>  Test cusparseLt for ViT on CIFAR10 with SSAM <b>N:M</b> mask <b>implicit</b> sparse perturbation </summary>

```bash
python train.py \
  --model vit_testspmm --patch_size 1 --log_freq 1 \
  --dataset CIFAR100_cutout --datadir [Path2Data] \
  --opt sam-sgd --lr 0.05 --weight_decay 1e-3 --rho 0.2 \
  --pattern nm --n_structured 2 --m_structured 4 --num_samples 128 --update_freq 1 --implicit --culinear \
  --seed 1234 --wandb
```
</details>
