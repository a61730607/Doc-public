# 在ImageNet数据集上训练ResNet50分类模型



## 1、介绍

​        本教程将介绍使用Pet训练以及测试ResNet50模型进行图像分类的主要步骤，在此我们会指导您如何通过组合Pet的提供组件来训练Mask R-CNN模型，在此我们仅讲解组件的调用，部分实现细节请查阅系统组件的相应部分。

​        在阅读本教程的之前我们强烈建议您阅读原始论文[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)[1]、[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)[2]和[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)[3]以了解更多关于ResNet的算法原理。 

## 2、快速开始 

### 2.1、数据准备

​        关于数据放置与数据制备的详细说明参见 [此处](../../usage/data_zh.md)。

- **数据放置**

​        Pet默认支持ImageNet数据集，当您想要使用ImageNet数据集进行您的模型训练和测试时，需要您将下载好的ImageNet数据集放在`$Pet/data/ILSVRC2017`文件夹下，文件夹结构如下

  ```shell
ILSVRC2017
├── Annotations
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   └── DET
│       ├── train
│       └── val
├── Annotations_refine
│   └── DET
│       ├── train
│       └── val
├── Data
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   ├── DET
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── DET_CROP
│       ├── train
│       └── val
├── ImageSets
│   ├── CLS-LOC
│   └── DET
└── log
    ├── DET
    └── VID
  ```

- **数据格式**

​        对于不同的视觉任务，Pet支持在多种数据集上进行模型的训练和测试，并且规定了Pet标准的数据集源文件的文件结构与标注的格式。

- **预训练模型权重/测试模型权重下载**

​        从Model Zoo中下载所需要的权重文件到"/ckpts/"相应目录下。

### 2.2、训练与测试

​     如果您已经有语义分割任务的训练经验，您可以直接在Pet中运行`$Pet/tools/vision/train_net.py`脚本开始训练您的ResNet50+DeepLab-v3模型。

如果您已经有分类任务的训练经验，您可以直接在Pet中运行`$Pet/tools/vision/train_net.py`脚本开始训练您的ResNet50模型。

- **训练**

​        直接在Pet中运行以下代码开始训练您的模型。

  ```shell
  # 指定GPU参数进行训练
  cd $Pet
  
  python tools/vision/train_net.py --cfg cfgs/tutorials/resnet50b_vision.yaml --gpu_id 0,1,2,3
  
  # --gpu_id参数为指定训练所用gpu，不指定默认训练所用gpu为8卡
  ```

- **测试**

​        训练好的模型自动存储在指定位置（`$Pet/ckpts/vision/ImageNet/resnet/resnet50b_vision`），对应的模型有对应的model_latest.pth文件，在Pet中运行以下代码开始测试您的模型。

  ```shell
  # 指定GPU参数进行训练
  cd $Pet
  
  python tools/vision/test_net.py --cfg cfgs/tutorials/resnet50b_vision.yaml --gpu_id 0,1,2,3
  
  # --gpu_id参数为指定测试所用gpu，不指定默认测试所用gpu为8卡
  ```

## 3、实验配置文件

​        Pet以yaml文件格式定义并存储本次实验配置信息，并根据任务类型和所用数据集等信息将其保存在cfgs目录下相应位置。关于配置系统（cfgs）的详细说明参见 [此处](../../usage/configs_zh.md)。故在进行任何与模型训练和测试有关的操作之前，需要指定一个yaml文件，明确在训练时对数据集、模型结构、优化策略以及训练时可以调节的重要参数的设置，本教程以`$Pet/cfgs/vision/ImageNet/resnet/resnet50b_vision.yaml`模型为例讲解训练过程中所需要的关键配置，这套配置将指导ResNet50模型以及测试的全部步骤与细节，全部参数设置详见 [此处]($Pet/cfgs/tutorials/resnet50b_vision.yaml)

​        此yaml包含的大致配置信息如下（所有默认基础配置可在pet/lib/config/目录下查询）

```python
MISC:# 基础配置 例如GPU数量
    ...
MODEL:# 模型配置 例如所采用的模型、网络结构
    ...
SOLVER:# 优化器及调度器配置 例如学习率、迭代次数、调度器类型等
    ...
DATA:# 数据相关配置  例如数据加载路径、标注格式等
    ...
TRAIN:# 训练配置  例如指定权重文件路径、指定训练集等
    ...
TEST:# 测试配置 例如指定测试集，指定图像大小调整的参数等
    ...
```

## 4、数据集准备与介绍

​        Pet默认支持ImageNet数据集，其余Pet支持的数据集详见 [此处](../../usage/data_zh.md)。确保ImageNet数据集已经存放在您的硬盘中并整理好文件结构。 

[ImageNet](https://image-net.org/challenges/LSVRC/2017/index.php) 大规模视觉识别挑战 2017 (ILSVRC2017)，分类和定位任务的数据将与[ILSVRC 2012](https://image-net.org/challenges/LSVRC/2012/index.php)保持不变。验证和测试数据将包括从 flickr 和其他搜索引擎收集的 150,000 张照片，手工标记有[1000 个对象类别](https://image-net.org/challenges/LSVRC/2017/browse-synsets.php)的存在或不存在. 这 1000 个对象类别包含 ImageNet 的内部节点和叶节点，但彼此不重叠。带有标签的 50,000 张图像的随机子集将作为开发套件中包含的验证数据以及 1000 个类别的列表发布。其余图片将用于评估，并在测试时无标签发布。训练数据是 ImageNet 的子集，包含 1000 个类别和 120 万张图像，将打包以便于下载。本次比赛的验证和测试数据不包含在 ImageNet 训练数据中。

​        在[官网IMAGENET](https://image-net.org/challenges/LSVRC/2017/)下载数据集`ILSVRC2017`。


​        yaml文件中关于数据集部分的配置如下

```yaml
DATA:
  PIXEL_MEAN: (0.485, 0.456, 0.406)  # 像素平均值（BGR顺序）作为元组
  PIXEL_STD: (0.229, 0.224, 0.225)  # 像素标准差（BGR顺序）作为元组
  DATASET_TYPE: "image_folder_dataset"  # 指定数据集类型
...
```

​        关于数据加载的详细教程与解释详见 [此处](../../usage/data_zh.md)

## 5、模型构建

​     以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，其包含了基础配置、模型配置，模型配置主要包括骨干网络配置与结构设置，以及对应任务的Head模块定义等基本配置信息，我们可以通过这些基础信息构建适应任务的模型。全部模型部分构建的yaml文件如下：

```yaml
MODEL:
  BACKBONE: "resnet" # 骨干网络配置
  GLOBAL_HEAD: 
    CLS:              #指定 head 部分为分类
      ONEHOT_ON: True  
  RESNET:   # 骨干网络RESNET的构建参数
    CLS_ON: True  # 分类
    LAYERS: (3, 4, 6, 3)
    STRIDE_3X3: True
    NORM: "BN"  # 批量归一化
    ZERO_INIT_RESIDUAL: False
    FREEZE_AT: 0
  ONEHOT: 
    NUM_CLASSES: 1000  # 分类的类别数量
```

关于使用模型的详细参数配置解释参见`$Pet/lib/config/model/backbone.py`,关于模型构建的详细介绍参见 [此处](../../usage/model_building_zh.md)。接下来将从主干网络和分类任务两个模块来详细分析此yaml文件中关于模型构建的参数定义。

### 5.1、创建主干网络

​     ResNet50主干网络模型构建的配置信息如下

```yaml
MODEL:  # 模型配置
  BACKBONE: "resNet"  # 骨干网络配置
  NECK: ""
...
ResNet:  # 骨干网络ResNet的结构设计
    LAYERS: (3, 4, 6, 3)  # 每一模块的层数，此处的参数设置为ResNet50
    STRIDE_3X3: True
    USE_3x3x3HEAD: True
    NORM: "SyncBN"
    STRIDE: 8
    FREEZE_AT: 0
...
```

### 5.2、创建分类任务网络

在yaml文件中设定任务关键字为CLS，表明任务为分类，`ONEHOT_ON: True`表明分类任务使用One_Hot方法，yaml文件中对这部分进行了以下定义：

```yaml
...
GLOBAL_HEAD:
    CLS:                     #指定 head 部分为分类
      ONEHOT_ON: True
...
```

根据yaml配置文件，通过GeneralizedCNN类实例化对应模型，并且在前向函数中控制数据流。具体代码在`$Pet/pet/vision/modeling/model_builder.py`中：

```python
from pet.vision.modeling.model_builder import GeneralizedCNN

class GeneralizedCNN(nn.Module):
    
        """ 视觉模型构建+前向函数定义    """
    def __init__(self, cfg: CfgNode) -> None:
        super(GeneralizedCNN, self).__init__()

        self.cfg = cfg
        # 构建backbone部分：ResNet
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = Backbone(cfg) 
        ...
        # Neck为""，无需构建此部分
        if cfg.MODEL.NECK: 
            Neck = registry.NECKS[cfg.MODEL.NECK]
            ...
        ...
        # 构建semseg的检测头
        if cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON:# cfg.infer_cfg()调用
            self.global_semseg = GlobalSemSeg(cfg, dim_in, spatial_in)
        ...
    ...
```

## 6、模型训练

### 6.1、加载训练数据

在训练开始前需要您将下载好的ImageNet数据集放在`$Pet/data/ILSVRC2017`文件夹下，文件夹结构如下

  ```shell
ILSVRC2017
├── Annotations
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   └── DET
│       ├── train
│       └── val
├── Annotations_refine
│   └── DET
│       ├── train
│       └── val
├── Data
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   ├── DET
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── DET_CROP
│       ├── train
│       └── val
├── ImageSets
│   ├── CLS-LOC
│   └── DET
└── log
    ├── DET
    └── VID
  ```


以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，在模型训练中的参数构建中指定了所用训练集等训练数据。关于数据加载的详细教程与解释详见 [此处](../../usage/data_zh.md)

```yaml
TRAIN:  # 训练参数设定
...
  DATASETS: ("imagenet1k_2017_train",)  #指定训练集
```

### 6.2、优化器与调度器的构建

迭代优化是训练深度学习模型的核心内容，迭代优化主要包括了优化器和调度器的参数设定。本教程以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，讲解优化器和调度器的配置。关于迭代优化部分的详细介绍参见 [此处](../../usage/solver_zh.md)

yaml文件中规定了优化器对基本学习率进行了设定，在优化器中对优化器类型、基本学习率、超参数进行了指定；在调度器中设定了最大迭代次数、SGD迭代次数、调度器类型。关于优化器与调度器的配置信息如下：

```yaml
SOLVER:
  OPTIMIZER:  # 优化器
    BASE_LR: 0.1  #设置初始学习率
    WEIGHT_DECAY_NORM: 0.0001  # 学习衰减率
  SCHEDULER:
    TOTAL_EPOCHS: 120  #设置总迭代轮数
    STEPS: (30, 60, 90)
    WARM_UP_EPOCHS: 0
```


关于优化器与调度器的构建详细配置解释参见`$Pet/lib/config/solver.py`.

在Pet的代码实现中，优化器和学习率调度器具体对应`Optimizer`和`Scheduler`两个基本Python操作类，两个Python类会在整个训练的过程中一直被用于指导模型的优化。通过解析配置文件相关参数，传给`Optimizer`类(`/pet/lib/utils/analyser.py`)和`LearningRateScheduler`类(`/pet/lib/utils/lr_scheduler.py`),从而构建优化器及调度器，仅在训练阶段使用，以下列出了`$pet/tools/vision/train_net.py`部分关于优化器与调度器的构建源码：

```python
from pet.lib.utils.optimizer import Optimizer
from pet.lib.utils.lr_scheduler import LearningRateScheduler

# 构建优化器
optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
optimizer = checkpointer.load_optimizer(optimizer)
...
# 构建调度器
scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
scheduler = checkpointer.load_scheduler(scheduler)
```


### 6.3、模型加载与保存

模型的加载与保存对网络训练十分重要，Pet定义了一个类`CheckPointer`用于相关功能的封装。以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，模型的加载主要需要确定模型参数的加载方式，加载预训练模型，加载模型参数；模型的保存主要包括模型参数保存、优化器与学习率调节器设置等。关于此部分的详细说明参见 [此处](../../usage/solver_zh.md)

关于模型加载与保存的完整代码请参考`pet/lib/utils/checkpointer.py`

在此yaml文件的设置中通过初始化权重文件所在路径来实现模型的加载，`$Pet/cfgs/tutorials/resnet50b_vision.yaml`中预训练模型字段缺省代表Pet将无预训练模型进行模型训练。

模型的保存主要通过设定参数SNAPSHOT_ITER与SNAPSHOT_EPOCHS来确定，SNAPSHOT_ITER指定了每训练迭代多少次保存一次参数，SNAPSHOT_EPOCHS指定了每训练多少个epochs保留一次参数，二者只能有一个生效。这使得Pet能在断点后继续进行训练，关于这部分的参数详见`$Pet/lib/config/solver.py`.

```python
# Snapshot (model checkpoint) period
SOLVER.SNAPSHOT_ITER = 10000
```

```python
# Snapshot (model checkpoint) period
SOLVER.SNAPSHOT_EPOCHS = 50
```

### 6.4、模型训练参数配置

以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，模型主要的训练流程有指定权重文件路径、指定训练集、指定训练过程中需要用到的数据预处理参数、指定图像增强参数、指定随机裁剪参数等，在该yaml文件中对这部分参数进行了指定。关于模型训练的详细说明参见 [此处](../../usage/training_zh.md)

关于训练部分的详细参数配置解释参见`$Pet/lib/config/data.py`

训练基本参数设定，包括batch size：

```yaml
TRAIN:  # 训练参数设定
...
  BATCH_SIZE: 256    # 训练最小batch size
...
```

预处理参数设定，包括图像增强，随机裁剪等参数设定：

```yaml
  ...
  TRANSFORMS: ("random_resized_crop", "random_horizontal_flip", "to_tensor", "normalize")   # 指定数据增强的方式
  RANDOM_RESIZED_CROP:  # crop 的参数设置
    SIZE: (224, 224)
  ...
```
关于模型训练的主要步骤包括创建模型、创建检查点、加载预训练权重或随机初始化、创建优化器、创建训练集与加载器、构建调度器、模型分布式等。以下代码列出了部分训练步骤，详细参见`$pet/tools/vision/train_net.py`。

```python
...
    # Create model
    model = GeneralizedCNN(cfg)
    logging_rank(model)
    ...

    # Create checkpointer
    checkpointer = CheckPointer(cfg.MISC.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME)

    # Load pre-trained weights or random initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MISC.CONV1_RGB2BGR)
    model.to(torch.device(cfg.MISC.DEVICE))

    ...

    # Create optimizer
    optimizer = Optimizer(model, cfg.SOLVER.OPTIMIZER).build()
    optimizer = checkpointer.load_optimizer(optimizer)
    logging_rank("The mismatch keys: {}".format(mismatch_params_filter(sorted(checkpointer.mismatch_keys))))

    ...
    
    # Create scheduler
    scheduler = LearningRateScheduler(optimizer, cfg.SOLVER, iter_per_epoch=iter_per_epoch)
    scheduler = checkpointer.load_scheduler(scheduler)
    
	...

    # Train
    train(cfg, model, train_loader, optimizer, scheduler, checkpointer, all_hooks)
```

## 7、模型测试

### 7.1、加载测试数据

在测试开始前需要您将下载好的ImageNet数据集放在`$Pet/data/ILSVRC2017`文件夹下，文件夹结构如下

  ```shell
ILSVRC2017
├── Annotations
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   └── DET
│       ├── train
│       └── val
├── Annotations_refine
│   └── DET
│       ├── train
│       └── val
├── Data
│   ├── CLS-LOC
│   │   ├── train
│   │   └── val
│   ├── DET
│   │   ├── test
│   │   ├── train
│   │   └── val
│   └── DET_CROP
│       ├── train
│       └── val
├── ImageSets
│   ├── CLS-LOC
│   └── DET
└── log
    ├── DET
    └── VID
  ```

以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，在模型测试中的参数构建中指定了所用测试集等训练数据。关于数据加载的详细教程与解释详见 [此处](../../usage/solver_zh.md)

```yaml
TEST:  # 测试参数设定
  DATASETS: ('imagenet1k_2017_val',)  # 指定测试集
```

### 7.2、模型测试

​        以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，模型测试过程中需要指定图像大小调整的参数等，这部分在yaml文件中有详细的配置，以下列出了此yaml文件中的关于测试参数设定细节。

关于测试部分的详细参数配置解释参见`$Pet/lib/config/data.py`

​        测试基本参数指定：

```yaml
TEST:  # 测试参数设定
...
  IMS_PER_GPU: 16  # 一张卡上16张图
...
```

​		测试过程中预处理参数指定，此处包括图像大小调整参数:

```yaml
...
RESIZE:
    SCALE: 256  # 测试期间resize图像大小
    MAX_SIZE: -1
  CENTER_CROP:
    CROP_SCALES: (224, 224)  # 测试期间crop图像大小
```

​        关于模型测试的主要步骤包括创建模型、加载模型、创建测试数据集与加载器、构建测试引擎等。此处列出部分源码作为解读，详细参见`$pet/tools/vision/test_net.py`。

```python
# Load model
    test_weights = get_weights(cfg.MISC.CKPT, cfg.TEST.WEIGHTS)
    load_weights(model, test_weights)
    model.eval()
    model.to(torch.device(cfg.MISC.DEVICE))

    # Create testing dataset and loader
    dataset = build_dataset(cfg, is_train=False)
    test_loader = make_test_data_loader(cfg, dataset)

    # Build hooks
    all_hooks = build_test_hooks(args.cfg_file.split("/")[-1], log_period=10, num_warmup=0)

    # Build test engine
    test_engine = TestEngine(cfg, model, dataset)

    # Test
    test(cfg, test_engine, test_loader, dataset, all_hooks)
```

### 7.3、模型评估（可视化与指标）

以`$Pet/cfgs/tutorials/resnet50b_vision.yaml`为例，模型的评估需要存储测试记录，设定评估参数，这部分在yaml文件中有详细的配置。关于模型评估的详细教程参见 [此处](../../usage/evaluation_zh.md)

关于评估部分的详细参数配置解释参见`$Pet/lib/config/config.py`

```yaml
EVAL:# 验证
    RECORD: [{"time": "20210727", "recorder": "user", "version": "0.7a",
            "cls": "top1/top5: 76.41/92.98",
            "mark": "torchvision/resnet50-19c8e357.pth; top1/top5: 76.13/92.86; FLOPs: 4087.1M; Params: 25.55M"}]
            # 测试记录存储，"time":测试时间；"recorder":测试者；"version":所用版本；"semseg": "mIoU/PixelACC/MeanACC/MeanF1Score:42.09/79.99/53.03/55.95":评估参数    
  #  测试记录存储，"time":测试时间；"recorder":测试者；"version": 所用版本；"cls": "top1/top5: 76.41/92.98":评估参数
```

- **可视化结果**

  分类模型暂无可视化结果，提供打印结果（包含类别及相应概率）。


## 参考文献

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016.

[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Identity Mappings in Deep Residual Networks. ECCV 2016.

[3] Saining Xie, Ross Girshick, Piotr Dolla ́r, Zhuowen Tu, Kaiming He. Aggregated Residual Transformations for Deep Neural Networks. CVPR 2017.

[4] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. ImageNet: A Large-Scale Hierarchical
Image Database. In CVPR09, 2009