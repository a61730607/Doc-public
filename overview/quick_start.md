# 快速开始
使用Pet训练CIFAR10分类模型

通过阅读本教程，您可以了解使用Pet在CIFAR10数据集上训练和测试一个分类器的简要步骤。如果想要了解更多内容，请点击以下链接：

* 在ImageNet数据集上训练分类模型的详细案例见[ImageNet分类教程](../tutorials/basic/cls_zh.md)。

下面介绍CIFAR分类训练和测试的流程：

## 准备数据

* 下载：训练前，先将[python版本的CIFAR数据](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)下载到本地并完成解压，CIFAR数据内容和类别如下：

![image](../image_source/cifar10_pic.png)

```
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz
```

* 路径：将数据以软连接的形式或直接存放到`$Pet/data`文件路径下，文件结构如下：

```
cifar
  |--cifar-10-batches-py
    |--data_batch_1
    |--data_batch_2
    |--data_batch_3
    |--data_batch_4
    |--data_batch_5
    |--test_batch
    ...
```

建立`$CIFAR`数据集路径到`$Pet/data`的软连接：

```
ln -s $CIFAR $Pet/data
```

## 训练

使用Pet训练和测试CIFAR10分类模型时，需要指定一个YAML文件，该文件里包含了模型所需的参数。Pet的配置系统详见：[配置系统](../usage/configs_zh.md)。这里以[$Pet/cfgs/tutorials/resnet110_Cifar10.yaml](https://github.com/BUPT-PRIV/Pet-dev/blob/main/cfgs/tutorials/resnet110_Cifar10.yaml)配置文件为例进行介绍。

启动训练：

```bash
cd $Pet
python tools/train_net_all.py --cfg cfgs/tutorials/resnet110_Cifar10.yaml # 默认8卡GPU训练
python tools/train_net_all.py --cfg cfgs/tutorials/resnet110_Cifar10.yaml --gpu_id 0 # 单GPU训练
```

在训练正常运行时，会在控制台输出如下的日志信息。

```bash
[10-26 11:41:27] |-[resnet110.yaml]-[iter: 20/63960]-[lr: 0.100000]-[eta: 2:49:11]
                 |-[max_mem: 169M]-[iter_time: 0.1588]-[data_time: 0.0239]
                 |-[total loss: 2.2620]
[10-26 11:41:29] |-[resnet110.yaml]-[iter: 40/63960]-[lr: 0.100000]-[eta: 2:28:50] 
                 |-[max_mem: 169M]-[iter_time: 0.1397]-[data_time: 0.0105]
                 |-[total loss: 2.0806]
[10-26 11:41:32] |-[resnet110.yaml]-[iter: 60/63960]-[lr: 0.100000]-[eta: 2:23:09]
                 |-[max_mem: 169M]-[iter_time: 0.1344]-[data_time: 0.0105]
                 |-[total loss: 2.0089]
[10-26 11:41:34] |-[resnet110.yaml]-[iter: 80/63960]-[lr: 0.100000]-[eta: 2:20:38]
                 |-[max_mem: 169M]-[iter_time: 0.1321]-[data_time: 0.0105]
                 |-[total loss: 1.9373]
[10-26 11:41:37] |-[resnet110.yaml]-[iter: 100/63960]-[lr: 0.100000]-[eta: 2:22:06]
                 |-[max_mem: 169M]-[iter_time: 0.1335]-[data_time: 0.0105]
                 |-[total loss: 1.9134]
[10-26 11:41:40] |-[resnet110.yaml]-[iter: 120/63960]-[lr: 0.100000]-[eta: 2:23:10]
                 |-[max_mem: 169M]-[iter_time: 0.1346]-[data_time: 0.0105]
                 |-[total loss: 1.8703]
[10-26 11:41:43] |-[resnet110.yaml]-[iter: 140/63960]-[lr: 0.100000]-[eta: 2:23:25]
                 |-[max_mem: 169M]-[iter_time: 0.1348]-[data_time: 0.0130]
                 |-[total loss: 1.8468]
[10-26 11:41:45] |-[resnet110.yaml]-[iter: 160/63960]-[lr: 0.100000]-[eta: 2:23:26]
                 |-[max_mem: 169M]-[iter_time: 0.1349]-[data_time: 0.0104]
                 |-[total loss: 1.8268]
[10-26 11:41:48] |-[resnet110.yaml]-[iter: 180/63960]-[lr: 0.100000]-[eta: 2:23:09]
                 |-[max_mem: 169M]-[iter_time: 0.1347]-[data_time: 0.0110]
                 |-[total loss: 1.7823]
[10-26 11:41:51] |-[resnet110.yaml]-[iter: 200/63960]-[lr: 0.100000]-[eta: 2:22:46]
                 |-[max_mem: 169M]-[iter_time: 0.1343]-[data_time: 0.0111]
                 |-[total loss: 1.7527]
                 
······

[10-26 14:12:13] INFO: Saving checkpoint done. And copy "model_latest.pth" to "model_iter63960.pth".
[10-26 14:12:13] INFO: Overall training speed: 63961 iterations in 2:17:49 (0.129293 s / it)
[10-26 14:12:13] INFO: Total training time: 2:18:23 (0:00:34 on hooks)
```

训练结束后，会将最终模型保存到`$Pet/ckpts/tutorials/Cifar/resnet110`路径下。

### 测试评估

接下来通过`$Pet/tools/test_net_all.py`进行测试评估。

测试用法示例：

```bash
python tools/test_net_all.py --cfg cfgs/tutorials/resnet110_Cifar10.yaml # 8卡评估
python tools/test_net_all.py --cfg cfgs/tutorials/resnet110_Cifar10.yaml --gpu_id 0 # 单GPU评估
```

测试结果：

```bash
[10-26 14:47:03] INFO: Loading from weights: ckpts/tutorials/Cifar/resnet110/model_latest.pth.
[10-26 14:47:03] INFO: Creating dataset: cifar10.
[10-26 14:47:04] INFO: [Testing][range:1-79 of 10000][10/79][0.049s = 0.021s + 0.029s + 0.000s][eta: 0:00:03]
[10-26 14:47:04] INFO: [Testing][range:1-79 of 10000][20/79][0.041s = 0.013s + 0.027s + 0.000s][eta: 0:00:02]
[10-26 14:47:04] INFO: [Testing][range:1-79 of 10000][30/79][0.037s = 0.011s + 0.026s + 0.000s][eta: 0:00:01]
[10-26 14:47:05] INFO: [Testing][range:1-79 of 10000][40/79][0.036s = 0.010s + 0.026s + 0.000s][eta: 0:00:01]
[10-26 14:47:05] INFO: [Testing][range:1-79 of 10000][50/79][0.036s = 0.009s + 0.027s + 0.000s][eta: 0:00:01]
[10-26 14:47:05] INFO: [Testing][range:1-79 of 10000][60/79][0.036s = 0.009s + 0.026s + 0.000s][eta: 0:00:00]
[10-26 14:47:06] INFO: [Testing][range:1-79 of 10000][70/79][0.035s = 0.009s + 0.026s + 0.000s][eta: 0:00:00]
[10-26 14:47:06] INFO: [Testing][range:1-79 of 10000][79/79][0.034s = 0.008s + 0.026s + 0.000s][eta: 0:00:00]
[10-26 14:47:06] INFO: Total inference time: 3.135s
[10-26 14:47:06] INFO: test_acc1: 94.28% | test_acc5: 99.88%
```

