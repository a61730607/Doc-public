# Pet安装

本文档包含了Pet及其依赖的安装（包括Pytorch）。

- Pet的介绍请参阅 [`README`](https://github.com/BUPT-PRIV/Pet-dev/blob/main/README.md)。

**环境要求:**

- NVIDIA GPU， Linux， Python3.6+。
- Pytorch 1.6.x-1.8.x （推荐Pytorch-1.8.2）
- 相关依赖 requirements.txt。
- CUDA 10.2, 11.x (推荐10.2或11.1)。

**注意：**

- Pet已被证实在CUDA >=10.2和CuDNN 7.5.1中可用。
- 请确保Pytorch及Cuda版本的兼容性。

## 安装Pytorch

使用Pet需要安装支持CUDA的Pytorch版本。请注意，安装的版本需要与环境中CUDA版本对应，具体请参考[pytorch](https://pytorch.org/get-started/locally/)。本指导以CUDA11.1环境为例。

```bash
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## 安装Pet

1.克隆Pet仓库：

```
git clone https://github.com/BUPT-PRIV/Pet-dev.git
```

2.安装 requirements.txt：

```
cd Pet-dev
pip3 install -r requirements.txt --user
```

3. make：

```
sh make.sh
```

## FAQ

Q: make过程遇到报错：`nvcc fatal : Unsupported gpu architecture 'compute_86'`

A: 在make之前，执行`export TORCH_CUDA_ARCH_LIST='8.0+PTX'`

