# Stable-Unlearnable-Example
This is the official implementation of [AAAI'24 "Stable Unlearnable Example: Enhancing the stableness of Unlearnable Examples via Stable Error-Minimizing Noise"](https://arxiv.org/abs/2311.13091). 

![SEM-framework](./SEM-framework.jpg)

## Requirements
- Python 3.8
- PyTorch 1.8.1
- Torchvision 0.9.1
- OpenCV 4.5.5

#### Install dependencies using pip

```shell
pip install -r requirements.txt
```

#### Install dependencies using Anaconda

```shell
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
conda install -c conda-forge opencv=4.5.5
```

## Quick Start

We give an example of creating stable unlearnable examples from CIFAR-10 dataset. More experiment examples can be found in [./scripts](./scripts).

#### Generate stable error-minimizing noise for CIFAR-10 dataset

```bash
python generate_stable_em.py \
    --arch resnet18 \
    --dataset cifar10 \
    --train-steps 5000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 2000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 8 \
    --pgd-steps 10 \
    --pgd-step-size 1.6 \
    --pgd-random-start \
    --atk-pgd-radius 4 \
    --atk-pgd-steps 10 \
    --atk-pgd-step-size 0.8 \
    --atk-pgd-random-start \
    --samp-num 5 \
    --report-freq 1000 \
    --save-freq 1000 \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/noise/sem8-4 \
    --save-name sem
```

#### Perform adversarial training on stable unlearnable examples

```bash
python train.py \
    --arch resnet18 \
    --dataset cifar10 \
    --train-steps 40000 \
    --batch-size 128 \
    --optim sgd \
    --lr 0.1 \
    --lr-decay-rate 0.1 \
    --lr-decay-freq 16000 \
    --weight-decay 5e-4 \
    --momentum 0.9 \
    --pgd-radius 4 \
    --pgd-steps 10 \
    --pgd-step-size 0.8 \
    --pgd-random-start \
    --report-freq 1000 \
    --save-freq 100000 \
    --noise-path ./exp_data/cifar10/noise/sem8-4/sem-fin-def-noise.pkl \
    --data-dir ./data \
    --save-dir ./exp_data/cifar10/train/sem8-4/r4 \
    --save-name train
```

## Citation

```
@article{liu2023stable,
  title={Stable Unlearnable Example: Enhancing the Robustness of Unlearnable Examples via Stable Error-Minimizing Noise},
  author={Liu, Yixin and Xu, Kaidi and Chen, Xun and Sun, Lichao},
  journal={arXiv preprint arXiv:2311.13091},
  year={2023}
}
```

## Acknowledgment
- Availability Attacks Create Shortcuts: [https://github.com/dayu11/Availability-Attacks-Create-Shortcuts](https://github.com/dayu11/Availability-Attacks-Create-Shortcuts)
- Robust Unlearnable Example: [https://github.com/fshp971/stable-unlearnable-examples](https://github.com/fshp971/stable-unlearnable-examples)
- Unlearnable examples: [https://github.com/HanxunH/Unlearnable-Examples](https://github.com/HanxunH/Unlearnable-Examples)
- Adversarial poisons: [https://github.com/lhfowl/adversarial_poisons](https://github.com/lhfowl/adversarial_poisons)
- Neural tangent generalization attacks: [https://github.com/lionelmessi6410/ntga](https://github.com/lionelmessi6410/ntga)


