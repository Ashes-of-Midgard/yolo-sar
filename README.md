# 环境配置

```
conda create -n yolo-sar python=3.11 -y
conda activate yolo-sar
# 这里我使用的是CUDA 11.8的环境，请根据CUDA版本选择对应的安装命令
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install PyWavelets
git clone https://github.com/Ashes-of-Midgard/yolo-sar.git
cd yolo-sar
pip install -e . -v
```

# 任务：目标检测
## 数据集和预训练权重

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
并且放置在和train_det.py同一级目录下

要选择哪一个数据集进行训练，只需要找到对应数据集目录下面的```dataset.yaml```文件，复制该文件的路径替代掉```train_det.py```当中的
```
results = model.train(data="datasets/OGSOD_yolo/dataset.yaml", epochs=100)
```
还需要修改```dataset.yaml```当中的```path:```，将其修改为数据集所在的路径。


## 训练

我们先用yolo12m作为基准模型，后续根据比赛具体评测设备调整。在OGSOD_yolo上微调100轮来评估各个模块是否有效，等实验出有效的模块以后再换到SARdet_100K_yolo进行完整的训练。

```
python train_det.py
```

如果只评测不训练，就把代码当中的
```
results = model.train(data="datasets/OGSOD_yolo/dataset.yaml", epochs=100)
```
注释掉。

# 任务：旋转检测
## 数据集

数据集从百度云盘下载，格式都经过处理

放置在
```
|- yolo-sar
    |- datasets
        |- RSDD-SAR-yolo
        |- SRSDD-V1.0-yolo
        |- RSAR-yolo
```

下载链接：
1. RSDD-yolo:
2. SRSDD-V1.0-yolo:
3. RSAR-yolo:

训练过程中会自动下载yolo12m和yolo11n预训练权重，如果网络不畅，没法自动下载，请从此处下载: https://pan.baidu.com/s/1hDinFXOZOVSZvCmrwUigqg?pwd=apbn 提取码: apbn
并且放置在和train_obb.py同一级目录下

要选择哪一个数据集进行训练，只需要找到对应数据集目录下面的```dataset.yaml```文件，复制该文件的路径替代掉```train_obb.py```当中的
```
results = model.train(data="datasets/SRSDD-V1.0-yolo/dataset.yaml", epochs=100)
```
还需要修改```dataset.yaml```当中的```path:```，将其修改为数据集所在的路径。

## 训练

我们先用yolo12n-obb作为基准模型，后续根据比赛具体评测设备调整。在SRSDD-V1.0-yolo上训练100轮来评估各个模块是否有效，等实验出有效的模块以后再换到RSAR-yolo进行完整的训练。

```
python train_obb.py
```

如果只评测不训练，就把代码当中的
```
results = model.train(data="datasets/SRSDD-V1.0-yolo/dataset.yaml", epochs=100)
```
注释掉。

# 怎样修改代码

!!! 修改代码之前看这里 !!!

## 新增简单模块
如果新增的模块是即插即用类型的：即不需要模型使用特殊的训练流程（比如对抗学习），在训练和推理时模块的工作方式都一样，那么就可以通过配置文件添加到模型当中。

ultralytics库管理模型结构的工具是.yaml扩展名的配置文件。比如说文件```ultralytics/cfg/models/12/yolo12.yaml```记录了yolo12模型的结构。
```yaml
# YOLO12n backbone
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]] # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16
  - [-1, 4, A2C2f, [512, True, 4]]
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32
  - [-1, 4, A2C2f, [1024, True, 1]] # 8

# YOLO12n head
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, A2C2f, [512, False, -1]] # 11

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, A2C2f, [256, False, -1]] # 14

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 11], 1, Concat, [1]] # cat head P4
  - [-1, 2, A2C2f, [512, False, -1]] # 17

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 8], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]] # 20 (P5/32-large)

  - [[14, 17, 20], 1, Detect, [nc]] # Detect(P3, P4, P5)
```

其中每一行从左往右的内容依次表示：
1. 该层网络接收哪几层网络输出的结果作为输入。-1表示该层网络的前一层网络。如果是列表，则表示该层网络接收不止一个网络层的输出。网络的序号从第一行开始往下依次计数，起点是0.
2. 该层网络内部的基础模块数量。部分网络模块内部是由多个相同的子模块依次串联形成的，该参数指明了其内部子模块串联的数量。
3. 该层网络的类型。在初始化模型时，会根据这个字符串查找库当中对应的类型，并实例化。
4. 其他参数。这是一个列表，包含该层网络初始化过程中所需的其他参数。

要实现并插入新的模块，可以在目录```ultralytics/nn/modules```任意合适的位置定义模块类型，然后在```ultralytics/nn/modules/__init__.py```当中导入定义的新模块类型。最后在```ultralytics/nn/tasks.py```当中的函数```parse_model```里，增加解析新模块类型参数的方式。例如：
```python
def parse_model(d, ch, verbose=True):
    
    ...

    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):
        m = (
            getattr(torch.nn, m[3:])
            if "nn." in m
            else getattr(__import__("torchvision").ops, m[16:])
            if "torchvision.ops." in m
            else globals()[m]
        )  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        
        ...
        
        elif m is MyModule:
            # 在这里增加解析配置文件里参数并初始化MyModule实例的代码
```
要启用该模块，就在配置文件的正确位置写入该模块，将这个配置文件保存至指定的位置(推荐保存至```ultralytics/cfg/models/12/```)，命名方式推荐为"yolo12(-任务名)-自定义后缀.yaml"，如果是水平框检测模型，不需要指定任务名，但如果是旋转检测模型，需要指定任务名为"obb".
然后在```train_det.py```或```train_obb.py```当中读取该配置文件进行模型初始化，如：
```python
    model = YOLO("yolo12m-custom.yaml")
```
按照推荐方式存储和命名配置文件后，库函数能够自动解析跟随在yolo12后面的模型尺寸参数"n,s,m,l,x".

## 修改模型
有些模块与模型耦合程度比较高，它可能需要在训练和推理阶段分别执行不同的功能，这种情况下就需要对YOLO模型本身进行修改。

在```yolo-sar/ultralytics/nn/tasks.py```里面，继承```DetectionModel```，实现自定义的检测模型。重点修改它的```_pred_once```和```forward```函数，可以参考```DetectionModelSep```的实现方式。

现在假设你已经实现了```MyDetectionModel```子类。在```ultralytics/models/yolo/model.py```当中，继承```YOLO```类型，定义一个新的模型类型，记为```MyYOLO```。复制```YOLO```当中的```task_map```函数，修改返回字典的detect.model字段的值为```MyDetectionModel```（需要提前在文件头部导入），如下所示：
```python
class MyYOLO(YOLO):
    @property
    def task_map(self):
    return {
            ...
            "detect": {
                "model": MyDetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            ...
        }
```

最后，在训练时，如果要启用自定义的模型，则使用下述语句初始化模型
```python
    # model = YOLO("yolo12m.yaml")
    model = MyYOLO("yolo12m.yaml")
```

## 改变训练流程
想要改变训练流程，比如说增加对抗训练方式，请看下面的指南。

ultralytics库的训练流程是通过Trainer类实现的. 基类```BaseTrainer```定义在```ultralytics/engine/trainer.py```里。实际进行训练的时候，使用的是其子类```DetectionTrainer```, ```ClassificationTrainer```, ```SegmentationTrainer```等，我们本次比赛是检测任务，所以只分析```DetectionTrainer```。```DetectionTrainer```定义在```ultralytics/models/yolo/detect/train.py```里面。

对于这部分的改动，建议：
1. 如果是针对检测任务的特殊改动，可以在```ultralytics/models/yolo/detect/train.py```里面直接继承```DetectionTrainer```设计新的训练器。

2. 如果是不区分任务类型的改动，比如对抗训练，可以在```ultralytics/engine/trainer.py```里继承```BaseTrainer```，设计新的训练器```MyTrainer```. 然后在```ultralytics/models/yolo/detect/train.py```里导入并继承```MyTrainer```，设计```MyDetectionTrainer```.

改动训练器，关键是修改其```_do_train```函数，该函数里包括模型前向传播和反向梯度求导更新的代码。

修改完成后，无论是直接继承```DetectionTrainer```定义子类，还是继承新的训练器```CustomTrainer```，都应该在```ultralytics/models/yolo/detect/train.py```里有一个```MyDetectionTrainer```。将这个子类在文件```ultralytics/models/yolo/detect/__init__.py```当中导入，并写进列表```__ALL__```当中。确保可以在```yolo-sar/ultralytics/models/yolo/model.py```当中导入该类。

在```ultralytics/models/yolo/model.py```当中，继承```YOLO```类型，定义一个新的模型类型，记为```MyYOLO```。复制```YOLO```当中的```task_map```函数，修改返回字典的detect.trainer字段的值为```MyDetectionTrainer```，如下所示：
```python
class MyYOLO(YOLO):
    @property
    def task_map(self):
    return {
            ...
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.MyDetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            ...
        }
```

最后，在训练时，如果要启用自定义的训练流程，则使用下述语句初始化模型
```python
    # model = YOLO("yolo12m.yaml")
    model = MyYOLO("yolo12m.yaml")
```