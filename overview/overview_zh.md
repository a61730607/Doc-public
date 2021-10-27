# Pet整体介绍

在使用教程介绍之前，本文将对Pet理念及目录结构做简要介绍。 #TODO 补充一些说明文字。

## Pet理念

Pet在构建的过程中遵循一些理念，强烈建议您在使用之前阅读本篇内容。基于这些理念，Pet有机地将多种深度学习任务集成到一个平台中。

### 配置系统

Pet有一个非常强大的配置系统。通过这套配置系统，Pet可以控制所有的参数，包括模型结构、训练、测试、可视化等，Pet的配置系统可以管理所有支持的深度学习任务。具体介绍可参考[配置系统](../usage/configs_zh.md)。

### 任务分类

为将多种深度学习任务集成到Pet平台，同时保持代码的整洁性，Pet的许多目录和模块是基于任务进行分类的。Pet目前主要支持视觉任务，同时由于基于任务分类的理念，Pet保留了较强的可扩展性。为使目录及模块具备更强的逻辑与可读性，Pet使用两级分类模式。

- **vision**：视觉相关任务，遵循统一的流程。
  - 分类 **cl**a**s**sification （cls）
  - 检测 **det**ection （det）
  - 实例分割 **ins**tance **seg**mentation （insseg）
  - 语义分割 **sem**antic **seg**mentation （semsag）
  - 全精分割 **pan**optic **seg**mentation （panseg）
- **tasks**：基于某种具体任务，如OCR。由于任务具有一定复杂性，难以直接使用统一的vision进行模型构建或使用。
- **projects**：基于某种项目，如densepose。

### 模型构建

在vision中，所有任务的模型遵循统一的构建流程。视觉模型可大致分为三个部分：骨干网络Backbone、特征增强模块Neck、任务头Head。其中骨干网络是所有模型必须具备的，特征增强模块可选择，任务头是基于任务而引入的结构。这三个部分中，仅任务头与任务相关，所以在Pet中，为集成多种视觉任务，为每种任务构建一个统一的类进行管理。具体的模型构建流程请参考 [模型构建](../usage/model_building_zh.md)

## 目录结构

Pet主要的目录结构如下所示：

```
Pet-dev
|- cfgs # 配置文件
	|- vision
	|- tasks
	|- projects
|- pet  # Pet平台代码
	|- lib
	|- vision
	|- tasks
	|- projects
|- tools # 脚本工具
|- ckpts # 模型保存目录
|- tests # 测试脚本
```

cfgs、pet、ckpts的二级目录均遵循Pet理念中的任务分类，分别设置vision、tasks、projects目录。

- `cfgs/vision` 中的配置文件根据数据集进行划分。
- `pet`目录的`lib`存放多任务可复用代码，`vision`存放仅视觉任务使用的代码。

## 开发规范

强烈建议您遵循Pet理念及结构对Pet进行开发，这有助于代码风格的统一及整洁。 # TODO 补充

