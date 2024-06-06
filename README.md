# Cambricon PyTorch

由于[官方的 Catch 仓库](https://github.com/Cambricon/catch)版本较旧，因此从 docker 镜像中提取了一份新的源码。

镜像版本：yellow.hub.cambricon.com/pytorch/pytorch:v1.17.0-torch1.13.1-ubuntu20.04-py310

(python 3.10 + pytorch 1.13.1 + torchvision 0.14.1)

## 前置条件

已成功安装 [cambricon-mlu-driver](https://sdk.cambricon.com/download?component_name=Driver)，使用 cnmon 命令可以成功看到 MLU 加速卡

## 安装步骤

使用 Ubuntu-20.04 + bash 作为示例

1. 使用虚拟环境（强烈推荐）

   ```
   conda create -n cambricon-pytorch python=3.10
   conda activate cambricon-pytorch
   ```

2. 下载源代码

   ```
   git clone --recurse-submodules https://github.com/jklincn/cambricon-pytorch.git
   cd cambricon-pytorch
   ```

3. 安装 Pytorch

   ```
   export PYTORCH_HOME=$(pwd)/pytorch
   bash catch/scripts/apply_patches_to_pytorch.sh
   cd pytorch 
   conda install cmake ninja intel::mkl-static intel::mkl-include
   pip install -r requirements.txt
   export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
   export USE_CUDA=0
   python setup.py install
   cd ..
   ```

4. 安装 Cambricon Neuware SDK

   注意：脚本仅支持 Ubuntu-18.04/20.04，其他发行版请从[官网](https://sdk.cambricon.com/download?component_name=Neuware+SDK)下载

   ```
   ./install_neuware.sh
   source ~/.bashrc
   conda activate cambricon-pytorch
   ```

5. 安装 Catch

   ```
   cd catch
   pip install -r requirements.txt
   python setup.py install
   cd ..
   ```

6. 安装 Torchvision

   ```
   pip install torchvision==0.14.1
   ```

7. 验证是否安装成功（会有一些 warning，但程序可以正常运行结束）

   ```
   python catch/examples/training/single_card_demo.py
   ```

## 测试

```
pip install matplotlib
python compare.py
```

compare.py 是寒武纪 MLU 单卡训练和多卡训练对比的示例程序，运行结束后会生成 4 张对比图（损失函数，平滑后的损失函数，训练时间，准确率）