

# 在CIHP数据集上训练Parsing教程
## 1、介绍

本部分以Parsing RCNN在CIHP数据集上的训练和测试为例，介绍使用Pet训练以及测试Parsing R-CNN模型进行人体部位分割的主要步骤。主要讲解组件的调用，部分实现细节请查阅系统组件的相应部分。在阅读本教程的之前我们强烈建议您阅读原始论文[Parsing R-CNN](https://arxiv.org/abs/1811.12596)[1]、[PGN](https://arxiv.org/pdf/1808.00157v1)[2]以了解更多关于Parsing R-CNN的算法原理。

## 2、快速开始
### 2.1、数据准备
   关于数据放置与数据制备的详细说明参见 [此处](../../usage/data_zh.md)
   - **数据放置**
Pet默认支持CIHP数据集，当您想要使用CIHP数据集进行您的模型训练和测试时，需要您将下载好的CIHP数据集放在`$Pet/data/CIHP`文件夹下，文件夹结构如下：
  ```
  CIHP
  	|--images
  		|--testing
  		|--training
  		|--validation
  		...
  	|--annotations
  		|--training
  		|--validation
  		|--visualization
  		...
  ```
- **数据格式**

  ​        对于不同的视觉任务，Pet支持在多种数据集上进行模型的训练和测试，并且规定了Pet标准的数据集源文件的文件结构与标注的格式。

- **预训练模型权重/测试模型权重下载**

  ​        从Model Zoo中下载所需要的权重文件到"/ckpts/"相应目录下  确保CIHP数据集已经存放在您的硬盘中，并按照数据制备中的文件结构整理好CIHP数据集的文件结构。
### 2.2、训练与测试
如果您具有丰富的人体部位分析算法的研究经验，您也可以直接在Pet中运行`$Pet/tools/train_net_all.py`脚本立即开始训练您的Parsing R-CNN模型.

- **训练**：
 直接在Pet中运行以下代码开始训练您的模型 

```
# 指定GPU参数进行训练
cd $Pet

 python tools/vision/train_net.py --cfg cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml  --gpu_id 0,1,2,3
 
# --gpu_id参数为指定训练所用gpu，不指定默认训练所用gpu为8卡
```

- **测试**： 
  训练好的模型自动存储在指定位置（`$Pet/ckpts/tutorials/CIHP/e2e_parsing_rcnn_R-50-FPN_1x_ms`），对应的模型有对应的model_latest.pth文件，在Pet中运行以下代码开始测试您的模型

  ```
  # 指定GPU参数进行训练
  cd $Pet
  
  python tools/vision/test_net.py --cfg cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml --gpu_id 0,1,2,3
  
  # --gpu_id参数为指定测试所用gpu，不指定默认测试所用gpu为8卡
  ```

## 3、构建实验配置文件
	
Pet以yaml文件格式定义并存储本次实验配置信息，并根据任务类型和所用数据集等信息将其保存在cfgs目录下相应位置。关于配置系统（cfgs）的详细说明参见 [此处](../../usage/configs_zh.md)。故在进行任何与模型训练和测试有关的操作之前，需要指定一个yaml文件，明确在训练时对数据集、模型结构、优化策略以及训练时可以调节的重要参数的设置，本教程以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`模型为例讲解训练过程中所需要的关键配置，这套配置将指导ResNet50+DeepLab-v3模型以及测试的全部步骤与细节，全部参数设置详见 [此处]($Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml)

​        此yaml包含的大致配置信息如下（所有默认基础配置可在pet/lib/config/目录下查询）

```
	MISC: # 基础配置
		  ...
	MODEL: # 模型配置
		  ...
	SOLVER: # 优化器及调度器配置
		  ...
	DATA: # 数据相关配置
		  ...
	TRAIN: # 训练配置
		  ...
	TEST: # 测试配置
```
## 4、数据准备与介绍
 Pet默认支持CIHP数据集，其余Pet支持的数据集详见 [此处](../../usage/data_zh.md)。确保CIHP数据集已经存放在您的硬盘中并整理好文件结构。
CIHP是一个目标建立在多人体解析研究的数据集，数据集中收集了多人体实例级的图像，并提供了实例级人体解析的标注。CIHP数据集包含28280个训练样本、5000张验证集和5000张测试集，共有38280个多人图像。本教程帮助您下载CIHP数据集并按照pet要求的标注格式进行数据集的转换。

Pet需要从CIHP中下载并解压下列文件：

* 图像数据下载到`$Pet/data/CIHP/images/`文件夹中
* 标签数据下载到`$Pet/data/CIHP/annotations/`文件夹中

| 文件名 | 大小 |
| :-----: | :-: |
| [instance-level_human_parsing.tar.gz](https://pan.baidu.com/s/1nvqmZBN#list/path=%2Fsharelink2787269280-523292635003760%2FLIP%2FCIHP&parentPath=%2Fsharelink2787269280-523292635003760) | 1.89GB |

Pet为用户提供了下载并提取CIHP数据集的脚本，用户可以通过下载并运行cihp.py完成CIHP数据集的准备。用户通过在终端中运行如下命令完成数据集的处理。

```
python cihp.py –dataset_dir $download path -target_dir $save path
```
        yaml文件中关于数据集部分的配置如下

```python
DATA:
  DATASET_TYPE: "coco_dataset"# 指定数据集类型
  PIXEL_MEAN: (0.485, 0.456, 0.406)# 像素平均值（BGR顺序）作为元组
  PIXEL_STD: (0.00392, 0.00392, 0.00392)# 像素标准差（BGR顺序）作为元组
```

​        关于数据加载的详细教程与解释详见 [此处](../../usage/data_zh.md)
## 5、模型构建
​ 以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，其包含了基础配置、模型配置，模型配置主要包括骨干网络配置与结构设置，以及对应任务的Head模块定义等基本配置信息，我们可以通过这些基础信息构建适应任务的模型。全部模型部分构建的yaml文件如下：

```python
MISC:# 基础配置
  CKPT: "ckpts/tutorials/CIHP/e2e_parsing_rcnn_R-50-FPN_1x_ms"#权重文件路径
MODEL:# 模型配置
  BACKBONE: "resnet"# 骨干网络配置
  NECK: "fpn"
  GLOBAL_HEAD:# 任务配置，本实验为Parsing
    DET:
      RPN_ON: True
  ROI_HEAD:
    FASTER_ON: True
    PARSING_ON: True
  RESNET:# 骨干网络ResNet的结构设计
    LAYERS: (3, 4, 6, 3)# 每一模块的层数，此处的参数设置为ResNet50
    STEM_WIDTH: 64
    USE_3x3x3HEAD: False
    WIDTH: 64
    BOTTLENECK: True
    STRIDE_3X3: False
    NORM: "FrozenBN"
  RPN:#区域建议网络
    ANCHOR_SIZES: (32, 64, 128, 256, 512)
    ANCHOR_STRIDE: (4, 8, 16, 32, 64)
    ASPECT_RATIOS: (0.5, 1.0, 2.0)
    FG_IOU_THRESHOLD: 0.7
    BG_IOU_THRESHOLD: 0.3
    BATCH_SIZE_PER_IMAGE: 256
    PRE_NMS_TOP_N_TRAIN: 2000
    PRE_NMS_TOP_N_TEST: 1000
    # when FPN on, it means per-image not per-batch
    POST_NMS_TOP_N_TRAIN: 1000
    POST_NMS_TOP_N_TEST: 1000
    NMS_THRESH: 0.7
    # SMOOTH_L1_BETA: 1 / 9
  FASTER:#faster网络配置
    NUM_CLASSES: 1 # remove background
    BOX_HEAD: "mlp_head"
    ROI_XFORM_METHOD: "ROIAlign"
    ROI_XFORM_RESOLUTION: (7, 7)
    ROI_XFORM_SAMPLING_RATIO: 2
    BBOX_REG_LOSS: "smooth_l1"
    FG_IOU_THRESHOLD: 0.5
    BG_IOU_THRESHOLD: 0.5
    BBOX_REG_WEIGHTS: (10., 10., 5., 5.)
    BATCH_SIZE_PER_IMAGE: 512
    POSITIVE_FRACTION: 0.25
    ROI_CLS_LOSS_WEIGHT: 1.0
    BBOX_REG_LOSS_WEIGHT: 1.0
    SCORE_THRESH: 0.05
    NMS_TH: 0.5
    TRAIN_ON_PRED_BOXES: False
    SMOOTH_L1_BETA: 1.0
  PARSING:#Parsing网络设置
    NUM_CLASSES: 1
    NUM_PARSING: 20
    ROI_XFORM_RESOLUTION: (14, 14)
    ROI_XFORM_SAMPLING_RATIO: 2
    RESOLUTION: (56, 56)
    LOSS_WEIGHT: 1.0
    ROI_SIZE_PER_IMG: 32
```

​关于使用模型的详细参数配置解释参见`$Pet/lib/config/model/backbone.py`,关于模型构建的详细介绍参见 [此处](../../usage/model_building_zh.md)。接下来将从主干网络和分割任务两个模块来详细分析此yaml文件中关于模型构建的参数定义。

### 5.1、创建主干网络

ResNet50主干网络模型构建的配置信息如下

```python
MISC:# 基础配置
  CKPT: "ckpts/tutorials/CIHP/e2e_parsing_rcnn_R-50-FPN_1x_ms"#权重文件路径
MODEL:# 模型配置
  BACKBONE: "resnet"# 骨干网络配置
  NECK: "fpn"
...
 RESNET:# 骨干网络ResNet的结构设计
    LAYERS: (3, 4, 6, 3)# 每一模块的层数，此处的参数设置为ResNet50
    STEM_WIDTH: 64
    USE_3x3x3HEAD: False
    WIDTH: 64
    BOTTLENECK: True
    STRIDE_3X3: False
    NORM: "FrozenBN"
...
```
### 5.2、创建Parsing任务网络

​在yaml文件中设定任务关键字为PARSING，表明任务目的，根据分割任务划分的Head模块为`FASTER_ON`与`PARSING_ON`。yaml文件中对这部分进行了以下定义，Pet使用`Generalized_RCNN`来搭建Parsing R-CNN网络。只需要在YAML文件中添加'PRCNN'参数，即可构建Parsing R-CNN网络的人体部位分析的分支网络：

```python
...
 GLOBAL_HEAD:# 任务配置，本实验为Parsing
    DET:
      RPN_ON: True
  ROI_HEAD:
    FASTER_ON: True
    PARSING_ON: True
  RESNET:# 骨干网络ResNet的结构设计
    LAYERS: (3, 4, 6, 3)# 每一模块的层数，此处的参数设置为ResNet50
    STEM_WIDTH: 64
    USE_3x3x3HEAD: False
    WIDTH: 64
    BOTTLENECK: True
    STRIDE_3X3: False
    NORM: "FrozenBN"
  RPN:#区域建议网络
    ANCHOR_SIZES: (32, 64, 128, 256, 512)
    ANCHOR_STRIDE: (4, 8, 16, 32, 64)
    ASPECT_RATIOS: (0.5, 1.0, 2.0)
    FG_IOU_THRESHOLD: 0.7
    BG_IOU_THRESHOLD: 0.3
    BATCH_SIZE_PER_IMAGE: 256
    PRE_NMS_TOP_N_TRAIN: 2000
    PRE_NMS_TOP_N_TEST: 1000
    # when FPN on, it means per-image not per-batch
    POST_NMS_TOP_N_TRAIN: 1000
    POST_NMS_TOP_N_TEST: 1000
    NMS_THRESH: 0.7
    # SMOOTH_L1_BETA: 1 / 9
  FASTER: #faster网络配置
    NUM_CLASSES: 1 # remove background
    BOX_HEAD: "mlp_head"
    ROI_XFORM_METHOD: "ROIAlign"
    ROI_XFORM_RESOLUTION: (7, 7)
    ROI_XFORM_SAMPLING_RATIO: 2
    BBOX_REG_LOSS: "smooth_l1"
    FG_IOU_THRESHOLD: 0.5
    BG_IOU_THRESHOLD: 0.5
    BBOX_REG_WEIGHTS: (10., 10., 5., 5.)
    BATCH_SIZE_PER_IMAGE: 512
    POSITIVE_FRACTION: 0.25
    ROI_CLS_LOSS_WEIGHT: 1.0
    BBOX_REG_LOSS_WEIGHT: 1.0
    SCORE_THRESH: 0.05
    NMS_TH: 0.5
    TRAIN_ON_PRED_BOXES: False
    SMOOTH_L1_BETA: 1.0
  PARSING: #Parsing网络设置
    NUM_CLASSES: 1
    NUM_PARSING: 20
    ROI_XFORM_RESOLUTION: (14, 14)
    ROI_XFORM_SAMPLING_RATIO: 2
    RESOLUTION: (56, 56)
    LOSS_WEIGHT: 1.0
    ROI_SIZE_PER_IMG: 32
...
```
根据yaml配置文件，通过GeneralizedCNN类实例化对应模型，并且在前向函数中控制数据流。具体代码在`$Pet/pet/vision/modeling/model_builder.py`中：

```python
from pet.vision.modeling.model_builder import GeneralizedCNN

class GeneralizedCNN(nn.Module):
    
        """ 视觉模型构建+前向函数定义    """
    def __init__(self, cfg: CfgNode) -> None:
        super(GeneralizedCNN, self).__init__()

        self.cfg = cfg
        # 构建backbone部分：ResNet
        Backbone = registry.BACKBONES[cfg.MODEL.BACKBONE]
        self.backbone = Backbone(cfg) 
        ...
        # Neck为""，无需构建此部分
        if cfg.MODEL.NECK: 
            Neck = registry.NECKS[cfg.MODEL.NECK]
            ...
        ...
        # 构建semseg的检测头
        if cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON:# cfg.infer_cfg()调用
            self.global_semseg = GlobalSemSeg(cfg, dim_in, spatial_in)
        ...
    ...
```
在`Generalized_RCNN`中`$/pet/vison/modeling/model_builder.py`，只需要在`Fast_RCNN`之后添加`Parsing_RCNN`分支网络的构建代码即可，其他部分与Mask R-CNN网络的构建方式类似。

Parsing R-CNN网络在除了包含区域建议网络（RPN)、特征金字塔网络（FPN）以及目标检测分支网络（FastRCNN）之外，最重要的分支网络就是**人体部位分析分支网络**（ParsingRCNN)。根据YAML文件的设置，`Parsing RCNN`网络的主体结构是`gce_head`，`$/pet/vison/roi_head/parsing/heads/gce_head.py``gce_head`主要由**几何与周边信息编码模块**（Geometric and Context Encoding， GCE）与**特征转换模块**组成。
```Python
@registry.PARSING_HEADS.register("gce_head")
class GCEHead(nn.Module):
    def __init__(self, cfg, dim_in, spatial_in):
        """
        Args:
            cfg (CfgNode)
            dim_in (list[int]): number of channels of the input feature.
            spatial_in (list[float])
        """
        super(GCEHead, self).__init__()
        self.dim_in = dim_in[-1]
        self.spatial_in = spatial_in

        method = cfg.MODEL.PARSING.ROI_XFORM_METHOD
        resolution = cfg.MODEL.PARSING.ROI_XFORM_RESOLUTION
        sampling_ratio = cfg.MODEL.PARSING.ROI_XFORM_SAMPLING_RATIO
        pooler = Pooler(
            method=method,
            output_size=resolution,
            scales=spatial_in,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_nl = cfg.MODEL.PARSING.GCE_HEAD.USE_NL
        norm = cfg.MODEL.PARSING.GCE_HEAD.NORM
        conv_dim = cfg.MODEL.PARSING.GCE_HEAD.CONV_DIM
        aspp_dim = cfg.MODEL.PARSING.GCE_HEAD.ASPP_DIM
        num_convs_before_aspp = cfg.MODEL.PARSING.GCE_HEAD.NUM_CONVS_BEFORE_ASPP
        aspp_dilation = cfg.MODEL.PARSING.GCE_HEAD.ASPP_DILATION
        num_convs_after_aspp = cfg.MODEL.PARSING.GCE_HEAD.NUM_CONVS_AFTER_ASPP

        # convx before aspp
        before_aspp_list = []
        for _ in range(num_convs_before_aspp):
            before_aspp_list.append(
                make_conv(self.dim_in, conv_dim, kernel_size=3, norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim
        self.conv_before_aspp = nn.Sequential(*before_aspp_list) if len(before_aspp_list) else None

        # aspp
        self.aspp = ASPP(self.dim_in, aspp_dim, dilations=aspp_dilation, norm=norm)
        self.dim_in = self.aspp.dim_out

        feat_list = [
            make_conv(self.dim_in, conv_dim, kernel_size=1, norm=make_norm(conv_dim, norm=norm), act=make_act())
        ]
        # non-local
        if use_nl:
            feat_list.append(
                NonLocal2d(conv_dim, int(conv_dim * cfg.MODEL.PARSING.GCE_HEAD.NL_RATIO), conv_dim, use_gn=True)
            )
        self.feat = nn.Sequential(*feat_list)
        self.dim_in = conv_dim

        # convx after aspp
        assert num_convs_after_aspp >= 1
        after_aspp_list = []
        for _ in range(num_convs_after_aspp):
            after_aspp_list.append(
                make_conv(self.dim_in, conv_dim, kernel_size=3, norm=make_norm(conv_dim, norm=norm), act=make_act())
            )
            self.dim_in = conv_dim
        self.conv_after_aspp = nn.Sequential(*after_aspp_list) if len(after_aspp_list) else None
        self.dim_out = [conv_dim]
        self.spatial_out = [resolution]

        self._init_weights()
```

* GCE模块的作用是丰富分支网络特征图的感受野并编码人体部位之间的几何与周边信息。人体部位相互之间的关系在人体部位分析任务中是十分重要的信息，简单地使用常规的卷积层堆叠的方式来构建ParsingRCNN分支网络并不能很好地捕获如左、右手，左、右脚以及不同人物的肢体之间的关系，一方面是由于感受野的不足，另一方面是因为常规卷积普遍地提取目标的视觉特征而没有将更多的注意力放在人体部位的关系捕获上。GCE模块使用[Non-local Neural Networks](https://arxiv.org/pdf/1711.07971.pdf)[3]中提出的Non-local结构来加强捕获这种属于人体部位之间的几何与上下文关系。

* Parsing R-CNN论文中对GCE模块中Non-local与ASPP结构的退化实验，Non-local模块只有在特征图具有丰富感受野的情况下才能发挥作用，单独使用Non-local模块并不能给人体部位分析模型的精度带来提升。ASPP模块被用来丰富特征图的感受野，单独使用ASPP没开来增大感受野也可以更好地捕获人体部位之间的几何与上下文关系。关于ASPP结构的详细信息请参考[Deeplab-v3](https://arxiv.org/pdf/1706.05587)[4]。

| | mIoU | AP@50 | AP@vol | PCP@50 |
| :----: | :--: | :---: | :----: | :----: |
| baseline | 50.7 | 47.9 | 47.6 | 49.7 |
| ASPP only | 51.9 | 51.1 | 48.3 | 51.4 |
| Non-local only | 50.5 | 47.0 | 47.6 | 48.9 |
| GCE | 52.7 | 53.2 | 49.7 | 52.6 |

注：基准实验采用[Mask R-CNN](https://arxiv.org/abs/1703.06870)[5]的网络结构

* 特征转换模块将从主干网络特征图上截取出来的特征向任务特征进行转换，由几个堆叠的卷积层即可完成；
## 6、模型训练
### 6.1、加载训练数据
在训练开始前需要您将下载好的CIHP数据集放在`$Pet/data/CIHP`文件夹下，文件夹结构如下

```
CIHP
	|--images
		|--testing
		|--training
		|--validation
		...
	|--annotations
		|--training
		|--validation
		|--visualization
		...
```

以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，在模型训练中的参数构建中指定了所用训练集等训练数据。关于数据加载的详细教程与解释详见 [此处](../../usage/data_zh.md)

```python
TRAIN:# 训练参数设定
...
    DATASETS: ("CIHP_train",)#指定训练集
```
### 6.2、优化器与调度器的构建
  迭代优化是训练深度学习模型的核心内容，迭代优化主要包括了优化器和调度器的参数设定。本教程以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，讲解优化器和调度器的配置。关于迭代优化部分的详细介绍参见 [此处](../../usage/solver_zh.md)

yaml文件中规定了优化器对基本学习率进行了设定，在优化器中对优化器类型、基本学习率、超参数进行了指定；在调度器中设定了最大迭代次数、SGD迭代次数、调度器类型。关于优化器与调度器的配置信息如下：

```python
SOLVER:
  OPTIMIZER:# 优化器
    TYPE:"SGD"# 指定优化器类型为SGD,其他还有"RMSPROP", "ADAM"等
    BASE_LR: 0.02# 基本学习率
    WEIGHT_DECAY: 0.0001# L2正则化超参数
    MOMENTUM: 0.9# 与SGD一起使用的动量
  SCHEDULER:
    TOTAL_ITERS: 45000# 20210 * 120 / 16 = 151575，最大迭代次数
    STEPS: (30000, 40000)# 调度器类型，这里使用的是POLY，其他还有"STEP", "COSINE", ...
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
模型的加载与保存对网络训练十分重要，Pet定义了一个类`CheckPointer`用于相关功能的封装。以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，模型的加载主要需要确定模型参数的加载方式，加载预训练模型，加载模型参数；模型的保存主要包括模型参数保存、优化器与学习率调节器设置等。关于此部分的详细说明参见 [此处](../../usage/solver_zh.md)

关于模型加载与保存的完整代码请参考`pet/lib/utils/checkpointer.py`

在此yaml文件的设置中通过初始化权重文件所在路径来实现模型的加载，以下列出了yaml文件中的模型加载初始化设定。

```python
TRAIN:# 训练参数设定
  WEIGHTS: "ckpts/vision/ImageNet/3rdparty/resnet/resnet50a_caffe/resnet50a_caffe_conv1-rgb.pth"# 指定权重文件路径
```

​        模型的保存主要通过设定参数SNAPSHOT_ITER与SNAPSHOT_EPOCHS来确定，SNAPSHOT_ITER指定了每训练迭代多少次保存一次参数，SNAPSHOT_EPOCHS指定了每训练多少个epochs保留一次参数，二者只能有一个生效。这使得Pet能在断点后继续进行训练，关于这部分的参数详见`$Pet/lib/config/solver.py`.

```python
# Snapshot (model checkpoint) period
SOLVER.SNAPSHOT_ITER = 5000
```
### 6.4、模型训练参数配置
以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，模型主要的训练流程有指定权重文件路径、指定训练集、指定训练过程中需要用到的数据预处理参数、指定图像增强参数、指定随机裁剪参数等，在该yaml文件中对这部分参数进行了指定。关于模型训练的详细说明参见 [此处](../../usage/training_zh.md)

关于训练部分的详细参数配置解释参见`$Pet/lib/config/data.py`

训练基本参数设定，包括batch size与分割数：

```python
TRAIN:# 训练参数设定
...
  BATCH_SIZE: 16# 训练最小batch size
  SIZE_DIVISIBILITY: 32# 指定每一个整理批次的分割数
...
```
预处理参数设定，包括图像增强，随机裁剪等参数设定：

```python
  ...
  TRANSFORMS: ("resize", "random_horizontal_flip", "to_tensor", "normalize")# 训练过程中需要用到的数据预处理参数
  COLOR_JITTER:# 图像增强参数
  RESIZE:
    SCALES_SAMPLING: "choice"# 训练期间最小最小尺寸的采样类型，这里使用的是"scale_factor"，其余还有"scale_factor", "range", .
    SCALES: (512, 640, 704, 768, 800, 864)
    MAX_SIZE: 1400
```
关于模型训练的主要步骤包括创建模型、创建检查点、加载预训练权重或随机初始化、创建优化器、创建训练集与加载器、构建调度器、模型分布式等。以下代码列出了部分训练步骤，详细参见`$pet/tools/vision/train_net.py`。

```python
...
    # Create model
    model = GeneralizedCNN(cfg)
    logging_rank(model)
    logging_rank(
        "Params: {} | FLOPs: {:.4f}M / Conv_FLOPs: {:.4f}M | Activations: {:.4f}M / Conv_Activations: {:.4f}M"
        .format(n_params, model_flops, conv_flops, model_activs, conv_activs)
    )

    # Create checkpointer
    checkpointer = CheckPointer(cfg.MISC.CKPT, weights_path=cfg.TRAIN.WEIGHTS, auto_resume=cfg.TRAIN.AUTO_RESUME)

    # Load pre-trained weights or random initialization
    model = checkpointer.load_model(model, convert_conv1=cfg.MISC.CONV1_RGB2BGR)
    model.to(torch.device(cfg.MISC.DEVICE))
    if cfg.MISC.DEVICE == "cuda" and cfg.MISC.CUDNN:
        cudnn.benchmark = True
        cudnn.enabled = True

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
完成了数据加载以及模型构建之后，我们需要在开始训练之前选择模型的优化策略，遵循Mask R-CNN的设置，在批次大小为16的情况下，设置初始学习率为0.02，训练45000次迭代，组合使用了学习率预热与阶段下降策略，分别在30000与40000次迭代时将学习率减小十倍。

在训练过程中，日志记录仪会在每若干次迭代后记录当前网络训练的迭代数、各项损失数值等训练信息，检查点组件会定期保存网络模型到配置系统中`cfg.CKPT`所设置的路径下。

根据`cfg.DISPLAY_ITER`设置的日志记录间隔，在训练过程中每经过20次迭代，日志记录仪会在终端中记录模型的训练状态。

```
[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 200/45000][lr: 0.009200][eta: 21:23:30]
	  total_loss: 1.690106 (1.417845), iter_time: 1.8643 (1.7190), data_time: 0.1549 (0.1443)
	  loss_parsing: 0.395894 (0.365891), loss_objectness: 0.252050 (0.210352), loss_classifier: 0.161344 (0.199260), loss_box_reg: 0.228464 (0.202087), loss_rpn_box_reg: 0.431002 (0.427271)
[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 220/45000][lr: 0.009920][eta: 21:29:40]
	  total_loss: 1.188639 (1.316550), iter_time: 2.0313 (1.7280), data_time: 0.1353 (0.1444)
	  loss_parsing: 0.395576 (0.342062), loss_objectness: 0.205645 (0.191415), loss_classifier: 0.199962 (0.190168), loss_box_reg: 0.156144 (0.187377), loss_rpn_box_reg: 0.411209 (0.438963)
[Training][e2e_parsing_rcnn_R-50-FPN-PSS-ERR-ASPPV3-PBD_1x_ms.yaml][iter: 240/45000][lr: 0.010640][eta: 21:37:11]
	  total_loss: 1.737057 (1.387051), iter_time: 1.8072 (1.7389), data_time: 0.1581 (0.1447)
	  loss_parsing: 0.347431 (0.351932), loss_objectness: 0.299453 (0.190103), loss_classifier: 0.196695 (0.190588), loss_box_reg: 0.149391 (0.185793), loss_rpn_box_reg: 0.479773 (0.427392)
```
## 7、模型测试
### 7.1、加载测试数据
在完成Parsing R-CNN模型的训练之后，我们使用`$Pet/tools/vision/test_net.py`在`CIHP_val`上评估模型的精度。同样需需要使用Dataloader来加载测试数据集，将图像的短边缩放到800像素，长边做同样尺度的缩放（长边最大1333像素）。
在测试开始前需要您将下载好的CIHP数据集放在`$Pet/data/CIHP`文件夹下，文件夹结构如下

```
CIHP
	|--images
		|--testing
		|--training
		|--validation
		...
	|--annotations
		|--training
		|--validation
		|--visualization
		...
```

以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，在模型测试中的参数构建中指定了所用测试集等训练数据。关于数据加载的详细教程与解释详见 [此处](../../usage/solver_zh.md)

```python
TEST:# 测试参数设定
  DATASETS: ("CIHP_val",)# 指定测试集
```
### 7.2、模型测试

​以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，模型测试过程中需要指定图像大小调整的参数等，这部分在yaml文件中有详细的配置，以下列出了此yaml文件中的关于测试参数设定细节。

关于测试部分的详细参数配置解释参见`$Pet/lib/config/data.py`

​测试基本参数指定：
测试过程中预处理参数指定，此处包括图像大小调整参数:

```python
...
```python
TEST:# 测试参数设定
...
RESIZE:
    SCALE: 800# 测试期间图像大小调整的参数，是图像最短边的像素大小
    MAX_SIZE: 1333
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
以`$Pet/cfgs/tutorials/e2e_parsing_rcnn_R-50-FPN_1x_ms_CIHP.yaml`为例，模型的评估需要存储测试记录，设定评估参数，这部分在yaml文件中有详细的配置。关于模型评估的详细教程参见 [此处](../../usage/evaluation_zh.md)

关于评估部分的详细参数配置解释参见`$Pet/lib/config/config.py`

```python
EVAL:# 验证
  RECORD: [{"time": , "recorder": , "version": "0.7a","parsing": "mark": ""}]# 测试记录存储，"time":测试时间；"recorder":测试者；"version":所用版本；"semseg": "mIoU/PixelACC/MeanACC/MeanF1Score:42.09/79.99/53.03/55.95":评估参数
```

在人体部位分析任务中，模型精度使用[MHP-v1](https://arxiv.org/pdf/1705.07206)[6]中提出的mIOU、APp（AP@50）、APP、vol（AP@vol)与PCP来评价模型精度。

平均交并比（mean Intersection Over Union，mIOU），在一张图片中所有人体部位类别的预测掩码与标注掩码的像素交并比的平均值。

基于人体部位的平均准确率(Average Precision based on Part，APp)，与实例分割任务基于整个区域的平均准确率不同，APp使用的是一个人体实例内不同部位的预测掩码与标注掩码之间的mIOU来判断一个预测的实例是否正确，AP@50代表当mIOU的阈值为0.5时的APp，AP@vol代表mIOU的阈值分别为0.1~0.9（间隔为0.1）时APp的平均值。

正确分析的人体部位百分比（Percentage of Correctly Parsed Body Parts，PCP)，APp将所有人体部位的准确率进行了均值计算，不能真正反映有多少人体部位被正确的预测。因此在每一个人物实例中，所有与标注掩码的像素IOU高于一定阈值的人体部位被认为是被正确预测，每一个人物实例计算一个PCP，整体的PCP是有所有人物实例的PCP的平均值。PCP@50代表IOU的阈值为0.5时的PCP。

- **可视化结果如下**：

![image](../../image_source/demo_parsing.png)

## 参考文献

[1] Lu Yang, Qing Song, Zhihui Wang and Ming Jiang. Parsing R-CNN for Instance-Level Human Analysis. CVPR 2019.

[2] K. Gong, X. Liang, Y. Li, Y. Chen, and L. Lin. Instance-level human parsing via part grouping network. ECCV 2018.

[3] Xiaolong Wang, Ross Girshick, Abhinav Gupta1, and Kaiming He. Non-localneural networks. CVPR 2018.

[4] Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam. Rethinking Atrous Convolution for Semantic Image Segmentation. arXiv:1706.05587.

[5] Kaiming He, Georgia Gkioxari, Piotr Dollár, Ross Girshick. Mask R-CNN. ICCV 2017.

[6] Jianshu Li, Jian Zhao, Yunchao Wei, Congyan Lang, Yidong Li, Terence Sim, Shuicheng Yan, Jiashi Feng. Multi-human parsing in the wild. arXiv:1705.07206, 2017.
