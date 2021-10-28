# 数据读取教程

## 介绍

Pet提供了一整套详细的数据读取流程，涵盖多个读取模块。通过
`tools/{type}/[project/subtask]/train_net.py`获取脚本信息，指定`tools/train_net_all.py`或`tools/test_net_all.py`实现训练或测试阶段的数据读取。下面将从通用训练脚本`tools/train_net_all.py`切入讲解数据读取流程，进而以`tools/vision/train_net.py`作为窗口介绍整套读取流程，主要涵盖数据制备与数据加载。

* **数据制备**:对于不同的视觉任务，Pet支持在多种数据集上进行模型的训练和测试，并且规定了Pet标准的数据集源文件的文件结构与标注的格式。
* **数据加载**:Pet实现了一套标准的数据载入接口，同时提供了多种在线数据增强方式如尺度变换、旋转、翻折等使得神经网络训练具有更好的泛化效果。
## 一、数据制备

数据制备组件包括了**数据集源文件的准备**以及**数据集注册**两个部分。遵循数据制备的标准即可在Pet中使用数据集进行模型的训练和测试。

### 数据集格式标准

Pet对所有视觉任务指定了一套标准的数据格式风格，配合高度一致的数据载入组件的工作方式与风格，确保能够高效的在数据集、模型上进行组合与切换。Pet对数据集格式的要求主要包括数据集源文件的文件结构和标注信息的存储形式两个方面。

根据计算机视觉任务的不同，数据集的规模、标注形式也不尽相同，遵循最小化差异原则，Pet对于数据集格式的标准被划分为两种基本风格：

* [ImageNet](http://www.image-net.org/challenges/LSVRC/)数据集风格主导的分类任务数据集格式标准。

* [MSCOCO](http://cocodataset.org/#home)数据集风格主导的实例分析任务的数据集格式标准。


#### 分类标准

目前主流的分类数据集当属`ImageNet`数据集，大量的视觉任务都依赖在`ImageNet`数据集上训练得到的分类模型进行迁移学习。因此对于分类任务，Pet目前主要提供`ImageNet`风格的分类数据集格式标准，对于[CIFAR](http://www.cs.toronto.edu/~kriz/cifar.html)数据集的训练与测试，Pet将之视为简单的快速开始，并未给予过多的关注。

训练数据文件夹的结构示例

```
└─(用户定义的数据集名称)
  └─train
    ├─n0001
    | ├─Image_00001.JPEG
    | ...
    | └─Image_99999.JPEG
    ...
    └─n9999
      ├─Image_00001.JPEG
      ...
      └─Image_99999.JPEG
```

文件结构中的`n0001`、`n9999`是存放图片数据的文件夹，用来表示其路径下数据的类别(使用`label_mapping.txt`文件映射到对应类别)。
在Pet中，默认将数据集路径设在`$Pet/data/`文件夹下，通过建立软连接的方式实现，这里给出`ILSVRC2017`加到`$Pet/data/`中的示例。

```
data
└─(用户定义的数据集名称)
  ├─train
  ├─val
  └─test
```

如果需要使用Pet在其他的分类数据集上进行研究，请先参考ImageNet的数据格式制备数据集，将自己的数据集通过加入到`$Pet/data`下，并参考数据集注册完成数据的注册。


#### 实例分析标准

`rcnn`，`ssd`，`pose`三个视觉模块包含了目标检测、实例分割、多人姿态估计、多人密集姿态估计、多人人体部位分析等基于实例的视觉任务的实现，而`MSCOCO`数据集是目前应用最广泛的实例级综合数据集，因此Pet中这三个视觉模块的数据制备组件以`MSCOCO2017`数据集的文件结构以及标注风格为主体数据集格式标准，并且使用高度统一格式的数据集能够充分发挥[cocoapi](https://github.com/cocodataset/cocoapi)的巨大便利性，对数据集进行高效的解析。

实例分析是Pet所有支持的视觉任务中，包含子任务最多的计算机视觉任务，所有实例分析任务在Pet下的数据制备标准均以MSCOCO2017为参考。MSCOCO数据集是目前最为流行的目标检测数据集，cocoapi也为数据集的解析、统计分析、可视化分析与算法评估提供了极大便利。

目标检测、实例分割、姿态估计是`MSCOCO`数据集所提供的几种官方标注，[人体密集姿态估计](http://densepose.org/)（DensePose）标注是由Facebook提供的MSCOCO数据集的扩充标注，选择了MSCOCO数据集中部分图片进行了人体密集姿态标注。它们是Pet所支持的重要功能，在此将几种实例分析任务的数据制备标准统一说明。

MSCOCO数据集的标注文件的标准格式请见[COCO官方文档](http://cocodataset.org/#format-data)。

根据Facebook开源的标注文件，人体密集姿态估计任务的标注文件`densepose_{dataset_name}_train/val/test.json`的格式与`MSCOCO`数据集相同，都包含有`images`、`categories`、`annotations`三个部分来分别存储图片、类别以及以实例为单位的标注信息，标注信息所包含的内容以及格式如下所示：

```
{
  "id": int,
  "iscrowd": 0 or 1,
  "category_id": int,
  "area": float,
  "num_keypoints": int,
  "bbox": [x, y, w, h],
  "keypoints": [x1, y1, v1, x2, y2, v2, ...],
  "segmentation": RLE or [polygon],
  "dp_x": [x1, x2, ...],
  "dp_y": [y1, y2, ...],
  "dp_U": [U1, U2, ...],
  "dp_V": [V1, V2, ...],
  "dp_I": [I1, I2, ...],
  "dp_masks":  RLE or [polygon],
  "image_id": int
}
```

在DensePose-COCO数据集标注内容中，包含了实例分析任务所有的标注内容，但对于目标检测、实例分割和人体姿态估计任务来说，只需要必要的外接框和任务对应的标注内容。

由于Pet采用`cocoapi`进行数据集解析，因此如果您需要在私人数据集上进行模型训练以及测试，则需要生成相应的`JSON`文件，`JSON`文件包含标注信息且格式与`MSCOCO`数据集中相应任务的标注文件相同。

参考`MSCOCO2017`数据集，Pet为实例分析的数据源文件规定了如下标准的文件结构：

```
└─MSCOCO2017(dataset name)
  ├─annotations(from annotations_trainval2017.zip)
  | ├─instances_train2017.json
  | ├─instances_val2017.json
  | ├─person_keypoints_train2017.json
  | ├─person_keypoints_val2017.json
  | ├─densepose_{dataset_name}_train.json
  | ├─densepose_{dataset_name}_val.json
  | ├─densepose_{dataset_name}_test.json
  | ├─image_info_test2017.json
  | └─image_info_test-dev2017.json
  ├─train2017
  | ├─000000000009.JPEG
  | ...
  | └─000000581929.JPEG
  ├─val2017
  | ├─000000000139.jpg
  | ...
  | └─000000581781.jpg
  └─test2017
    ├─000000000001.jpg
    ...
    └─000000581918.jpg
```

以上文件结构包括了检测、实例分割、姿态估计和人体密集姿态任务在内的数据文件结构。如果需要在`MSCOCO2017`数据集上训练实例分析任务模型，请下载数据集源文件并按此文件结构放置，如果您想在其他公开数据集或是自己的私人数据集上进行模型训练和测试，在生成相应的`JSON`标注文件后，也需要按上面的文件结构来放置数据集的源文件。


### 数据集注册

按照标准完成了的数据集制备之后，还需要在Pet中对数据集进行注册，才可以Pet中使用数据集进行模型的训练以及测试。

* 首先需要将数据集源文件软连接到`$Pet/data/`路径中，以MSCOCO数据集为例，通过如下指令建立源文件在Pet下的数据软连接：

```
ln -s /home/dataset_dir/MSCOCO2017  $Pet/data/coco
```

* 完成数据集源文件的软连接后，需要进一步在Pet中对数据集进行声明。在`$Pet/utils/data/catalog.py`中指定您数据集的图片文件、标注文件的路径，并设置数据集对应的关键词，如下所示：

```
'coco_2017_train': {
        _IM_DIR: _DATA_DIR + '/COCO/train2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_train2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 118287,
            'bbox': {
                'num_classes': 80,
                'num_instances': 860001,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 860001,
            },
        },
    },
    'coco_2017_val': {
        _IM_DIR: _DATA_DIR + '/COCO/val2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/instances_val2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 5000,
            'bbox': {
                'num_classes': 80,
                'num_instances': 36781,
            },
            'mask': {
                'num_classes': 80,
                'num_instances': 36781,
            },
        },
    },
    'coco_2017_test': {
        _IM_DIR: _DATA_DIR + '/COCO/test2017',
        _ANN_FN: _DATA_DIR + '/COCO/annotations/image_info_test2017.json',
        _ANN_TYPES: ('bbox', 'mask'),
        _ANN_FIELDS: {
            'num_images': 40670,
            'bbox': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
            'mask': {
                'num_classes': 80,
                'num_instances': -1,  # no annotations
            },
        },
    },
```


### Pet在不同数据集上进行的研究


|     视觉任务     | 数据集 |
| :------------:  | :---: |
|     图像分类     | cifar、ImageNet |
|     语义分割     | ADE2017 | 
| 目标检测、实例分析 | MSCOCO2017、VOC PASCAL、Densepose-COCO、MHP-v2、CIHP |
|      重识别      | DukeMTMC、Market1501、VehicleID |
|     姿态估计     | MSCOCO2017 keypoints |

遵循Pet所制定的数据制备组件的标准，我们在许多开源数据集上进行了大量的实验，已经在不同的任务上都训练出了最高水平精度的模型，同时提供这些模型的训练参数配置以及模型下载。

语义分割、目标检测、重识别任务下的一些数据集的标注格式可能与MSCOCO2017有很大不同，无法直接使用`cocoapi`进行标注信息的读取，因此需要将这些数据集的标注格式转换为COCO风格。Pet提供了一些数据集转换工具，可以将[VOC PASCAL](http://host.robots.ox.ac.uk/pascal/VOC/)、[CityScapes](https://www.cityscapes-dataset.com/)风格的标注文件转换成cocoapi可以读取的COCO-json标注。
## 二、数据加载
### 创建数据集类

数据加载是在深度学习模型训练过程中的关键环节，Pet为数据加载过程提供了完整且高效的读取加预处理的加载方式。在使用时，Pet通过调用`build_dataset`和`make_train_data_loader`两个函数完成数据的加载过程。

* build_dataset

以ssd中的dataloader使用为例，首先Pet在训练过程中调用`build_dataset`这一函数得到所需使用的数据集和数据集中的有用信息。同时按照配置文件中的要求调用`build_transforms`定义数据的预处理方式。
```Python
def build_dataset(cfg, is_train=True):
    dataset_list = cfg.TRAIN.DATASETS if is_train else cfg.TEST.DATASETS
    if not isinstance(dataset_list, (list, tuple)):
        raise TypeError(f"dataset_list should be a list of strings, got {dataset_list}.")
            datasets = []
    for dataset_name in dataset_list:
        if not contains(dataset_name):
            raise ValueError(f"Unknown dataset name: {dataset_name}.")

        root = get_im_dir(dataset_name)
        if not os.path.exists(root):
            raise NotADirectoryError(f"Im dir '{root}' not found.")

        ann_file = get_ann_fn(dataset_name)
        if not os.path.exists(ann_file):
            raise FileNotFoundError(f"Ann file '{ann_file}' not found.")

        unsupported_ann_types = ann_types.difference(get_ann_types(dataset_name))
        if len(unsupported_ann_types) > 0:
            raise ValueError(
                f"{dataset_name} does not support annotation types: {unsupported_ann_types}.")

        ann_fields = defaultdict(dict, get_ann_fields(dataset_name))
        if cfg.MODEL.HAS_MASK:
            ann_fields['mask'].update({'mask_format': cfg.DATA.FORMAT.MASK})
        if cfg.MODEL.GLOBAL_HEAD.SEMSEG_ON:
            ann_fields['semseg'].update({'semseg_format': cfg.DATA.FORMAT.SEMSEG,
                                         'label_format': cfg.DATA.FORMAT.SEMSEG_LABEL})
        if cfg.MODEL.ROI_HEAD.PARSING_ON:
            ann_fields['parsing'].update({'semseg_format': cfg.DATA.FORMAT.SEMSEG})
        ann_fields = dict(ann_fields)

        dataset = dataset_obj(root, ann_file, ann_types, ann_fields,
                              transforms=transforms,
                              is_train=is_train,
                              filter_invalid_ann=is_train,
                              filter_empty_ann=is_train,
                              filter_crowd_ann=True,    # TO Check, TODO: filter ignore
                              bbox_file=bbox_file,
                              image_thresh=image_thresh,
                              mosaic_prob=mosaic_prob)
        datasets.append(dataset)

        logging_rank(f"Creating dataset: {dataset_name}.")

    # concatenate all datasets into a single one
    if len(datasets) > 1:
        dataset = ConcatDataset(datasets)
        logging_rank(f"Concatenate datasets: {dataset_list}.")
    else:
        dataset = datasets[0]

    return dataset
```

* make_train_data_loader

在数据集构建完毕后利用torch提供的数据加载类`torch.utils.data.DataLoader`完成Pet的数据载入，同时对数据进行预处理。Pet为用户提供丰富的图像预处理方式，详情参考transforms介绍：

```Python
def make_train_data_loader(cfg, datasets, start_iter=1):
    assert len(datasets) > 0

    num_gpus = get_world_size()
    assert cfg.TRAIN.BATCH_SIZE % num_gpus == 0
    ims_per_gpu = int(cfg.TRAIN.BATCH_SIZE / num_gpus)
    shuffle = True
    num_workers = cfg.DATA.LOADER_THREADS
    drop_last = True if cfg.MISC.CUDNN else False

    # group images which have similar aspect ratio. In this case, we only
    # group in two cases: those with width / height > 1, and the other way around,
    # but the code supports more general grouping strategy
    aspect_grouping = cfg.DATA.SAMPLER.ASPECT_RATIO_GROUPING
    collator = BatchCollator(cfg.TRAIN.SIZE_DIVISIBILITY)

    sampler = make_data_sampler(cfg, datasets, shuffle)
    batch_sampler = make_batch_data_sampler(
        datasets, sampler, aspect_grouping, ims_per_gpu, drop_last=drop_last)

    if cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS is None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, iterations=cfg.SOLVER.SCHEDULER.TOTAL_ITERS, start_iter=start_iter)
    else:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, epochs=cfg.SOLVER.SCHEDULER.TOTAL_EPOCHS, start_iter=start_iter)

    data_loader = DataLoader(
        datasets,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )

    return data_loader
 ```
 
需要注意的是，`ssd`、`instance`在数据载入时图片大小均确定且一致，但由于`rcnn`的特殊性，Pet将`rcnn`下的数据采样器实现单独提出，以函数`make_batch_data_sampler`的形式置于`$Pet/pet/vision/datasets/dataset.py`下。其原因为在两阶段检测器的实现过程中，我们需要以每个batch中最大的图为基准，将其按比例缩放来确定所需张量维度，对于同一批次内其余图片在缩放的基础上在右侧和下侧进行补边操作，为了减少补边带来的计算量的增加，我们将图片分为**宽大于高**与**高大于宽**两种情况，将属于同一种类型的图片采集到同一个批次中进行数据加载。

```Python
def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, drop_last=False):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        group_ids = GroupedBatchSampler.get_group_ids(dataset, aspect_grouping)
        batch_sampler = GroupedBatchSampler(group_ids, sampler, images_per_batch, drop_last)
    else:
        batch_sampler = BatchSampler(sampler, images_per_batch, drop_last)
    return batch_sampler
```

Pet将所有任务共用的数据加载组件置于`$Pet/lib/data`下，包含以下内容：

```
├─lib
  ├─data
    ├─datasets
    ├─evalution
    ├─samplers
    ├─structures
    ├─transforms
    ├─collate_batch.py
    ├─dataset_catalog.py
```

* datasets

针对不同任务的特殊性，`datasets`提供了针对图像分类、目标检测和实例分析的三种数据读取方式，以`ImageFolderList`,`COCODataset`和`COCOInstanceDataset`,`cifar_dataset `四个独立的类来呈现，在后续不同任务的实现过程中，通过继承这三个类完成图像和标注数据的读取与载入。


* ImageFolderList

Pet针对图像分类任务的数据加载提供了获取图像和标签的类`ImageFolderList`,用户通过按照数据制备中分类任务要求的格式进行数据存放后，调用此类完成分类任务的数据加载过程。


 初始化

```Python
class ImageFolderDataset(data.Dataset):
    def __init__(self, root: str, ann_file: str, ann_types: Set[str],
                 ann_fields: Dict[str, Dict], transforms: Optional[Callable] = None,
                 is_train: bool = False, filter_invalid_ann: bool = True,
                 filter_crowd_ann: bool = True, load_bytes: bool = False, class_map: str = "", **kwargs) -> None:
        class_to_idx = None
        if class_map:
            class_to_idx = load_class_map(class_map, root)
        images, class_to_idx = find_images_and_targets(root, class_to_idx=class_to_idx)
        if len(images) == 0:
            raise RuntimeError(f'Found 0 images in subfolders of {root}. '
                               f'Supported image extensions are {", ".join(IMG_EXTENSIONS)}')
        self.root = root
        self.samples = images
        self.imgs = self.samples  # torchvision ImageFolder compat
        self.class_to_idx = class_to_idx
        self.load_bytes = load_bytes
        self.transform = transforms
```

* COCODataset

Pet针对目标检测任务如ssd和rcnn，提供了图像级标注文件解析读取的类`COCODataset`完成数据加载过程中图像级标注的载入。


 初始化

首先我们对`COCODataset`进行初始化：`COCODataset`继承了`torchvision.datasets.coco.CocoDetection`这一类，定义初始化参数包括：

> `ann_file`:标注文件路径
> `root`:图像数据路径
> `ann_types`:标注文件类型
> `transforms`:预处理方式

```Python
class COCODataset(CocoDetection)::
     def __init__(self, root: str, ann_file: str, ann_types: Set[str],
                 ann_fields: Dict[str, Dict], transforms: Optional[Callable] = None, is_train: bool = False,
                 filter_crowd_ann: bool = True, filter_invalid_ann: bool = True, filter_empty_ann: bool = True,
                 **kwargs) -> None:
        super(COCODataset, self).__init__(root, ann_file)
        coco = self.coco
        self.img_to_anns = [coco.imgToAnns[idx] for idx in self.ids]
        cat_ids = sorted(coco.getCatIds())
        self.classes = [c['name'] for c in coco.loadCats(cat_ids)] 
```


__getitem__

解析并获取数据集中每张图片的标注信息，针对现在主流算法如MaskRCNN会有关键点或实例分割的分支，Pet会根据配置文件中任务开关状态来获取图像、图像标注以及图像索引。

```Python
def __getitem__(self, idx: int) -> Tuple[Image.Image, ImageContainer, int]:
        img = self.pull_image(idx)
        anns = self.pull_target(idx)

        size = img.size
        target = ImageContainer(size)

        cat_ids_map = self.category_id_to_contiguous_id
        classes = [cat_ids_map[obj["category_id"]] for obj in anns]
        target['label'] = Label(size, classes)
```

* COCOInstanceDataset

`COCOInstanceDataset`与`COCODataset`类似，区别在于其继承了`torch.utils.data.Dataset`类完成针对实例的图像和标注的获取。

```Python
class ConcatDataset(_ConcatDataset):
def get_idxs(self, idx):
dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return dataset_idx, sample_idx

```
* cifar_dataset
Pet针对目标分类任务，提供了图像级标注文件解析读取的类cifar_dataset完成数据加载过程中图像级标注的载入。
```Python
def CifarDataset(root: str, ann_file: str, ann_types: Set[str],
                 ann_fields: Dict[str, Dict], transforms: Optional[Callable] = None,
                 is_train: bool = False, filter_invalid_ann: bool = True,
                 filter_crowd_ann: bool = True, load_bytes: bool = False, class_map: str = "", **kwargs) -> None:
    num_classes = ann_fields["cls"]["num_classes"]
    if num_classes == 10:#CIFAR10分类总数
        dataset = CIFAR10(root=root, train=is_train, transform=transforms)
    elif num_classes == 1000:#CIFAR100分类总数
        dataset = CIFAR100(root=root, train=is_train, transform=transforms)
    else:
        raise NotImplementedError

    return dataset
```

### 迭代加载器

`samplers`在继承`torch.utils.data.sampler`的基础上实现了基于batch、iteration、分布式和随机采样等数据采样方式。

* `DistributedSampler`:将数据加载限制为数据集子集的采样器。它的优势体现在和类  `torch.nn.parallel.DistributedDataParallel`结合使用。在这种情况下，每个进程可以将分布式采样器实例作为数据加载器采样器传递并加载原始数据集的一个子集。其初始化参数包括:

| 参数 | 参数解释 |
| :-: | :-: |
| dataset | 用于采样的数据集 |
| num_replicas | 参与分布式训练的进程数 |
| rank | 当前进程在所有进程中的序号 |
  
* `GroupedBatchSampler`:将采样器打包产生一个minibatch，强制将来自同一组的元素转为按照batch_size的大小输出。同时，该采样器提供的mini-batch将尽可能按照原始采样器的要求进行采样。其初始化参数包括:

| 参数 | 参数解释 |
| :-: | :-: |
| sampler | 用于采样的基础采样器 |
| batch_size | mini-batch的大小 |
| drop_uneven | 当其设置为True时，采样器将忽略大小小于batch_size的批次 |

* `IterationBasedBatchSampler`：将BatchSampler打包，从中重新采样，直到完成对指定的迭代次数采样完毕为止。

* `RangeSampler`：对数据集进行随机采样。
* `repeatFactor`: 重复因子训练采样器，设定每一轮图像集重复比例。


### 统一数据格式

* `structures`定义了对不同任务标注的处理及转换方式。对各类型标注如检测框、关键点、分割掩模等的处理方式均封装为Python类。

* `BoxList`:这一类表示一组边界框，这些边界框存储为Nx4的一组Tensor。为了唯一确定与每张图像相应的边界框，我们还存储了对应的图像尺寸。同时类中包含特定于每个边界框的额外信息，例如标签等。

* `ImageList`:将列表中可能大小不同的图像保存为单个张量的结构。其原理是将图像补边到相同的大小，并将每个图像的原始进行大小存储。

* `PersonKeypoints`：这一类继承了`Keypoints`类，完成对人体关键点的缩放，同时针对人体关键点检测中常用的水平翻转操作定义了关键点映射，实现数据加载过程中keypoints信息读取的转换。

* `HeatMapKeypoints`：这一类用于生成人体关键点热图，与`PersonKeypoints`有区别的是，此类中包含实例级别的人体关键点操作。

* `KeypointsInstance`：将针对人体实例关键点检测任务的所有基本操作封装为一类。

* `BinaryMaskList`：用于处理分割任务中，图像上所有目标的二值掩模。

* `PolygonInstance`：包含表示单个实例目标掩模的多边形。其实例化对象可以为一组多边形的集合。

* `PolygonList`：用于处理分割任务中，图像上以多边形形式标注的所有目标。

* `SegmentationMask`：用于存储图像中所有目标的分割标注信息，其中包括二值掩模和多边形掩模，完成`BinaryMaskList`和`PolygonList`提取出的标注信息的融合过程。


### 数据增强

`transforms`提供了丰富的针对object和instance的两种图像数据预处理方式。Pet将每一种图像和实例的预处理操作封装为一个Python类，并在后续预处理时实例化。

* 针对目标检测的图像预处理方式：

| 预处理操作 | 用途 |
| :-: | :-: |
| Compose | 将所有需要使用的预处理方式结合 |
| Resize | 对图片尺寸进行缩放 |
| RandomHorizontalFlip | 对图片进行水平翻转(镜像) |
| ColorJitter | 对图像亮度、饱和度、对比度的抖动 |
| ToTensor | 将图片转换为张量形式 |
| Normalize | 对图片进行归一化(减均值除方差) |
| RandomErasing | 把图像中一块矩形区域中的像素值替换为随机值 |
| SSD_ToTensor | ssd任务中将numpy形式存储的图片转换为张量形式 |
| SSD_Distort | ssd任务中对图像亮度、饱和度、对比度进行一定范围内的随机修改 |
| SSD_Mirror | ssd任务中对图片进行水平翻转(镜像) |
| SSD_Init | ssd任务中将rgb形式的图像通道转为bgr |
| SSD_Resize | ssd任务中对图片尺寸进行缩放 |
| SSD_Normalize | ssd任务中对图片进行归一化(减均值除方差) |
| SSD_CROP_EXPAND | ssd任务中对图像先进行随机裁剪，后将裁剪过的图片用0像素进行随机补边 |

* 针对实例分析的图像预处理方式：

| 预处理操作 | 用途 |
| :-: | :-: |
| Xy2Xyz | 将COCO关键点x,y,v形式的标注分解为x,y,z和vx,vy,vz |
| Box2CS | 将边界框转化为中心点和尺度的表示方法 |
| Scale | 对实例进行尺度变换 |
| Rotate | 对实例进行旋转 |
| Flip | 对实例进行水平翻装(镜像) |
| Affine | 旋转、翻转等操作通过affine应用于图片上 |
| InstanceOperate | 对于实例的操作如产生heatmap等 |
| BGR_Normalize | 对实例图像进行通道转换和归一化 |
| ExtractTargetTensor | 在pose任务中提取dataset类中所需的tensor属性值以用于__getitem__函数的传递 |



