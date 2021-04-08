# cspUTS-CVPR2021

## Introduction
This is a repo including the code and environment setup in CVPR 2021 compete in white box adversary attack. Our team got 63rd out of 1600 teams in this compete. The methods using PGD backbone and combine with the EOT and Auto Step Search.

Repo contains:
- Environments setup: the experiments environments are built on Colab. (GPU-16G)
- Codes supplied.

## Installation

1.Follow the installation instructions in repo [ares](https://github.com/thu-ml/ares/tree/main). Some installation guide:
``` shell
git clone https://github.com/thu-ml/ares
cd ares/
pip install -e .
```
The `requirements.txt` includes its dependencies, you might want to change PyTorch's version as well as TensorFlow 1's version. TensorFlow 1.13 or later should work fine. As for python version, Python 3.5 or later should work fine.

2. Installation of contest
Download `contest` folder from:
```shell
git clone https://github.com/thu-ml/ares/tree/contest/contest
```
Put the `contest` folder into the main folder of ARES. 

3.Download Datasets & Model Checkpoints
To download the CIFAR-10 dataset, please run:

``` shell
python3 ares/dataset/cifar10.py
```

To download the ImageNet dataset, please run:

``` shell
python3 ares/dataset/imagenet.py
```

for instructions.

ARES includes third party models' code in the `third_party/` directory as git submodules. Before you use these models, you need to initialize these submodules:

``` shell
git submodule init
git submodule update --depth 1
```

The `example/cifar10` directory and `example/imagenet` directories include wrappers for these models. Run the model's `.py` file to download its checkpoint or view instructions for downloading. For example, if you want to download the ResNet56 model's checkpoint, please run:

``` shell
python3 example/cifar10/resnet56.py
```
(*note: please check the model name and path carefully, make sure the model download are exactly the same! You can check the model path on `example/cifar10` directory and `example/imagenet` directories.)
(*To chnage the default model path, please run this code to change `ARES_RES_DIR` to the path you want to save all model and Data. Still make sure the path is correct!
```shell
import os
from ares import utils
os.environ["ARES_RES_DIR"] = '/content/drive/MyDrive/CVPR2021/ares/model'
A = utils.get_res_path('1')
print(A)
```
)

4. Quick Examples

ARES provides command line interface to run benchmarks. For example, to run distortion benchmark on ResNet56 model for CIFAR-10 dataset using CLI:

```shell
python3 -m ares.benchmark.distortion_cli --method mim --dataset cifar10 --offset 0 --count 1000 --output mim.npy example/cifar10/resnet56.py --distortion 0.1 --goal ut --distance-metric l_inf --batch-size 100 --iteration 10 --decay-factor 1.0 --logger
```

This command would find the minimal adversarial distortion achieved using the MIM attack with decay factor of 1.0 on the `example/cifar10/resnet56.py` model with L∞ distance and save the result to `mim.npy`.


## Citation

If you find ARES useful, you could cite our paper on benchmarking adversarial robustness using all models, all attacks & defenses supported in ARES. We provide a BibTeX entry of this paper below:

```
@inproceedings{dong2020benchmarking,
  title={Benchmarking Adversarial Robustness on Image Classification},
  author={Dong, Yinpeng and Fu, Qi-An and Yang, Xiao and Pang, Tianyu and Su, Hang and Xiao, Zihao and Zhu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={321--331},
  year={2020}
}
```
