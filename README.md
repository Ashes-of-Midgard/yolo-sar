# Installation

```
conda create -n yolo-sar python=3.11 -y
conda activate yolo-sar
# 这里我使用的是CUDA 11.8的环境，请根据CUDA版本选择对应的安装命令
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/Ashes-of-Midgard/yolo-sar.git
cd yolo-sar
pip install -e . -v
```

# Datasets & Weights

数据集从百度云盘下载，格式都经过处理

放置在
```
|- yolo-sar
    |- datasets
        |- OGSOD_yolo
        |- SAR_AIRcraft_yolo
        |- SSDD_yolo
        |- SARdet_100K_yolo
```

下载链接：
1. OGSOD_yolo: https://pan.baidu.com/s/1xnEms0zsWt96TuVukf2sqg?pwd=ptk3 提取码: ptk3
2. SAR_AIRcraft_yolo: https://pan.baidu.com/s/1bpNylbInJDFTYSefxuaKCg?pwd=8gmg 提取码: 8gmg
3. SSDD_yolo: https://pan.baidu.com/s/1j115aAKD3hWvdauUNn36eQ?pwd=erhv 提取码: erhv
4. SARdet_100K_yolo: https://pan.baidu.com/s/1UOVfJZpqcoX_AoCq6xBTzw?pwd=9nce 提取码: 9nce

训练过程中会自动下载yolo12m和yolo11n预训练权重，如果网络不畅，没法自动下载，请从此处下载: https://pan.baidu.com/s/1hDinFXOZOVSZvCmrwUigqg?pwd=apbn 提取码: apbn
并且放置在和train.py同一级目录下

要选择哪一个数据集进行训练，只需要找到对应数据集目录下面的```dataset.yaml```文件，复制该文件的路径替代掉```train.py```当中的
```
results = model.train(data="/root/yolo-sar/datasets/OGSOD_yolo/dataset.yaml", epochs=100)
```
还需要修改```dataset.yaml```当中的```path:```，将其修改为数据集所在的路径。


# Train

我们先用yolo12m作为基准模型，后续根据比赛具体评测设备调整。在OGSOD_yolo上微调100轮来评估各个模块是否有效，等实验出有效的模块以后再换到SARdet_100K_yolo进行完整的训练。

```
python train.py
```

如果只评测不训练，就把代码当中的
```
results = model.train(data="/root/yolo-sar/datasets/OGSOD_yolo/dataset.yaml", epochs=100)
```
注释掉。

