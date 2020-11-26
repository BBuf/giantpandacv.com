# 前言
今天为大家介绍一篇CVPR 2018的一篇目标检测论文《Single-Shot Refinement Neural Network for Object Detection》，简称为RefineDet。RefineDet从网络结构入手，结合了one-stage目标检测算法和two-stage目标检测算法的优点重新设计了一个在精度和速度均为SOTA的目标检测网络。论文的思想值得仔细推敲，我们下面来一起看看。论文源码和一作开源的代码链接见附录。

# 背景
对目标检测而言，two-stage的算法如Faster-RCNN获得了最高的精度，而one-stage算法虽然精度低一些，但运行速度非常快。基于此背景，论文提出了一个新的目标检测器名为RefineDet。RefineDet有两个核心组成部分，分别是ARM模块（Anchor Refinement module）和ODM模块（Object Detection Module）。ARM模块旨在（1）过滤掉negative anchors，以减少分类器的搜索空间，（2）粗略调整anchors的位置和大小，为后续的回归提供更好的初始化。而ODM模块将修正后的Anchors作为输入，进一步改善回归和预测多级标签。同时论文还设计了TCB模块(Transfer Connection Block)来传输ARM的特征，用于ODM中预测目标的位置，大小和类别标签。并且整个网络是端到端的，使用多任务损失来优化。最后RefineDet在PASCAL VOC 2007/2012和MSCOCO数据集上达到SOTA精度和速度。

# 核心贡献

- 引入了two-stage目标检测器中的对box由粗到细的回归思想（典型的就是Faster-RCNN中先通过RPN得到粗粒度的box信息，然后再通过常规的回归支路进行建议不回归从而得到更准确的信息，从而获得了很高的精度）。
- 引入了类似于FPN网络的特征融合操作用于检测器，可以提高对小目标检测的能力，RefineDet检测的框架仍是为SSD。

# 网络结构

RefineDet的网络结构如Figure1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191222135107904.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到RefineDet的网络结构主要包含3个部分，即上面提到的ARM模块，ODM模块和Transfer Connection Block模块。接下来我们就先分析每个模块最后再宏观的分析网络结构。
## ARM模块
Anchor Refinement Module (ARM模块)，这个模块类似于Faster-RCNN中的RPN网络，主要用来得到候选框和去除一些负样本。其实说类似的原因主要是因为这里的输入利用了多层特征，而RPN网络的输入利用的是单层特征。ARM模块生成的候选框(ROI)会为后续的检测网络提供好的初始信息，这也是one-stage检测器和two-stage检测器的主要区别。
## TCB模块
Transfer Connection Block（TCB）模块是做特征的转换操作，即是将ARM部分输出的特征图转换为ODM部分的输入，从上面的Figure1可以看到这部分和FPN的做法一致，也是上采样+特征融合的思想。TCB模块的详细结构如Figure2所示，这里的上采样使用了步长为2的反卷积，并没有直接使用Upsampling。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191222141201626.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## ODM模块
Object Detection Module（ODM）模块就和SSD的检测头完全一样了，把经过TCB得到的各个不同尺度的特征接分类和回归头就可以了。唯一不同之处在于这部分的输入Anchors不再是人为设置的，而是由ARM模块得到的Refined Anchors boxes，类似RPN网络输出的Proposal（ROI）。更加值得注意的是，这里的浅层特征图（尺寸比较大的蓝色块）融合了高层特征图的信息，然后预测目标框是基于每层特征图进行，最后将各层结果再整合到一起。我们知道在原始的SSD中特征图是直接拿来做预测的，并没有特征图信息融合的过程，这是一个非常重要的差别，这种差别就导致了RefineDet对小目标检测的性能更好。

## 整体概述
拆解了RefineDet的几个子模块后，我们可以全局性的来看一下RefineDet的网络结构了。这个结构和two-stage目标检测器很像，即一个ARM模块做RPN要做的事，ODM模块做SSD要做的事。我们知道SSD是直接在人为设定的默认框上做回归，而RefineDet先通过ARM模块生成Refined Anchors boxes类似于RPN的Propsoal，然后在Refined Anchors boxes的基础上再进行回归，加上FPN结构的特征融合使得网络对小目标的检测也更加有效。

## 网络构建
以`ResNet101`，输入图像大小为`320`为例，在ARM模块的4个灰色的特征图的尺寸分别为`40x40`,`20x20`,`10x10`,`5x5`，其中前三个是`ResNet101`网络本身的输出层，最后`5x5`输出是另外添加的一个残差模块。有了主干网络后要做特征融合操作了，首先`5x5`的特征图经过一个上面介绍过的TCB模块`P6`得到对应大小的蓝色矩形块，P6代表的是最右边那个蓝色特征图，对于生成`P6`这条支路来说，没有反卷积操作。接着基于`10x10`的灰色特征图经过TCB模块后得到`P5`，即是左边数第`3`个蓝色特征图，这个时候TCB模块有反卷积上采样操作了，后面的依此类推就可以了。因为我讲解的时候打乱了论文的顺序，所以接下来再补充一下RefineDet的另外两个核心点，即双阶段级联回归和Anchor的负样本过滤。

- 双阶段级联回归
当前one-stage的目标检测器依靠具有不同尺度的特征图一步回归来预测目标，这在某些场景下是十分不准确的，例如小目标检测。因此，这篇论文提出了一个两步级联回归策略回归目标的位置和大小。首先，使用ARM模块调整Anchors的位置和大小，为ODM模块提供更好的初始化信息。然后将`N`个`Anchor boxes`和在特征图上划分的单元格关联起来。最初的位置每个`Anchor boxes`相对于其对应的单元格是固定的。在每个特征图单元格中，我们预测`Refined Anchors boxes`的四个偏移量（相对于原始`Anchors boxes`）和指示前景存在的两个置信度分数。最终，可以在每个特征图单元格上产生`n`个`Refined Anchors`。获得`Refined Anchors`后，我们将其传到ODM模块相应的特征图中，进一步生成目标类别和目标框位置。ARM和ODM中相应的特征图具有相同的维度。我们计算`Refined Anchors`的`c`个类别分数和四个准确的偏移量，产生`c + 4`的输出以完成检测任务。此过程类似于SSD 中的默认框。但与SSD 不同，RefineDet使用两步策略，即ARM生成`Refined Anchors boxes`，ODM采取其作为输入进一步检测，因此检测结果更精准，特别适用于小目标。

- 负样本Anchor过滤
one-stage目标检测器的精度比不过two-stage的一个重要原因是类别不平衡问题，之前介绍了Focal Loss, GIoU Loss，DIoU Loss都是为了改善这个问题，而在这篇论文中，使用的是`Negative Anchor`过滤。在训练阶段，针对ARM中的`Anchor`，如果`Negative Confidence`大于一个阈值$\theta$（$\theta=0.99$，经验值），那么在训练ODM时将它舍弃。也就是通过`Hard Negative Anchor`和`Refined anchor`来训练ODM。 与此同时，在预测阶段，如果`Refined Anchor Box`负置信度大于$\theta$，则在ODM进行检测时丢弃。

## 训练和测试细节
### 数据增强

这部分和SSD一致，不再赘述，感兴趣可以看看我写的那篇文章：[目标检测算法之SSD的数据增强策略](https://mp.weixin.qq.com/s/xYP4k8nDUS8bF3_nSLJqYg)

### Backbone网络
使用在`ILSVRC CLS-LOC`数据集上预训练的`VGG-16`和`ResNet-101`作为RefineDet中的骨干网络。RefineDet也可以在其他预训练网络上工作，如`Inception v2` ，`Inception ResNe`t和`ResNeXt101`。 与`DeepLab-LargeFOV`类似，通过子采样参数，将`VGG-16`的`fc6`和`fc7`转换成卷积层`conv_fc6`和`conv_fc7`。与其他层相比，`conv4_3`和`conv5_3`具有较大的方差，所以使用`L2标准化`，同时如果直接使用标准化会减慢训练速度，所以设置两个缩放系数初始值为`10`和`8`，可以在反向传播的过程中训练。 同时，为了捕捉高层次多种尺度的信息指导目标检测，还分别在剪裁的`VGG-16`和`ResNet101`的末尾添加了额外的卷积层（即`conv6_1`和`conv6_2`）和额外的残差模块（即`res6`）。
### L2norm的一个解释
l2norm：Conv4_3层将作为用于检测的第一个特征图,该层比较靠前，其norm较大，所以在其后面增加了一个L2 Normalization层，以保证和后面的检测层差异不是很大.这个和Batch Normalization层不太一样:其仅仅是对每个像素点在channle维度做归一化，归一化后一般设置一个可训练的放缩变量gamma.而Batch Normalization层是在[batch_size, width, height]三个维度上做归一化。

代码实现：

```
def l2norm(x,scale,trainable=True,scope='L2Normalization'):
    n_channels = x.get_shape().as_list()[-1] # 通道数
    l2_norm = tf.nn.l2_normalize(x,dim=[3],epsilon=1e-12) # 只对每个像素点在channels上做归一化
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.constant_initializer(scale),
                                trainable=trainable)
    return l2_norm * gamma
```

### Anchor设计和匹配策略
在`VGG-16`和`ResNet101`上选择尺寸分别为`8,16,32`和`64`像素步幅大小的特征层，与几种不同尺度的`Anchor`相关联进行预测。 每个特征图都与一个特定特征层`Anchor`的尺度（尺度是相应层步幅的4倍）和三个比率（`0.5,1.0`和`2.0`）相关联。匹配策略使用`IOU`阈值来确立，具体来说就是将每个`ground truth boxes`与具有最佳重叠分数的`anchor boxes`相匹配，然后匹配`anchor`重叠高于`0.5`的任何`ground truth boxes`。

### Hard Negtive Mining
我在SSD中已经详细讲解过了，不再赘述了。总结一句话就是把正负样本比例设置为`1:3`，当然负样本不是随机选的，而是根据`box`的分类`loss`排序来选的，按照指定比例选择`loss`最高的那些负样本即可。可以看我的这篇推文详细了解：[目标检测算法之SSD代码解析(万字长文超详细)](https://mp.weixin.qq.com/s/knbXiA3mUS3KCYoV0Rpbeg)
### 损失函数
RefineDet的损失函数主要包含ARM和ODM两部分。在ARM部分包含二分类损失损失`Lb`和回归损失`Lr`；同理在ODM部分包含多分类损失`Lm`和回归损失`Lr`。需要注意的是虽然RefineDet大致上是RPN网络和SSD的结合，但是在Faster R-CNN算法中RPN网络和检测网络的训练可以分开也可以end to end，而这里的训练方式就完全是end to end了，ARM和ODM两个部分的损失函数都是一起向前传递的。损失函数公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191222154851589.png)

其中$i$表示一个`batch`中的第几个`Anchor`，$l_i^*$表示$Anchor_i$对应的的ground truth bbox（GT）类别，$g_i^*$表示$Anchor_i$的GT框位置，$p_i$表示置信度，$x_i$表示ARM中`Anchor`的坐标。$c_i$表示预测的类别，$t_i$表示ODM中预测的坐标信息。$N_{arm}$和$N_{odm}$分别表示ARM和ODM中的`positive anchor`数量。$L_b$表示二分类损失，$L_m$表示多分类损失，$L_r$表示回归损失。$[l_i^*>=1]$表示如果negative confidence大于一个阈值$\theta$，那么返回1，否则返回0。如果$N_{arm}=0$设置$L_b(p_i,[l_i^*>=1])=0$和$L_r(x_i,g_i^*)=0$；如果$N_{odm}=0$，那么设置$L_m(x_i,l_i^*)=0$和$L_r(t_i,g_i^*)=0$。

### 训练超参数设置
用`xavier`方法随机初始化基于`VGG-16`的RefineDet的两个添加的卷积层中（`conv6_1`和`conv6_2`）的参数。对于基于`ResNet-101`的RefineDet，初始化参数来自具有标准的零均值高斯分布，残差模块（res6）的初始化方差为0.01。其他的一些参数设置为：
- batch_size： 32
- momentum：0.9
- weight decay：0.0005
- initial learing rate：$0.0001$
其他详细可以看代码实现。

### 推理阶段
首先，ARM过滤掉负置信度分数大于阈值$\theta$的`anchors`，`refine`剩余`anchors`的位置和大小。然后， ODM输出每个检测图像前400个高置信度的`anchors`。 最后，应用`NMS`，`jaccard`重叠率限定为`0.45` ，并保留前`200`个高置信度anchors，产生最终的检测结果。

# 实验
`Table1`是非常详细的实验结果对比，测试数据包括VOC2007和VOC2012数据集。以VGG-16为特征提取网络的`RefineDet320`在达到实时的前提下能在VOC 2007测试集上达到80以上的mAP，这个效果基本上是目前看到过的单模型在相同输入图像情况下的最好成绩了。表格中最后两行在算法名称后面多了`+`，表示采用`multi scale test`，因此效果会更好一点。 

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191222160533987.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论

论文提出了一种新的one-stage目标检测器，由两个相互连接模块组成，即ARM模块和ODM模块。使用`multi-task loss`对整个网络进行`end-to-end`训练。在PASCAL VOC 2007，PASCAL VOC 2012和MS COCO数据集上进行了几次实验，结果表明RefineDet实现了目标最先进的检测精度和高效率。

# 目标检测未来
论文提出了两个点，即计划使用RefineDet来检测一些其他特定类型的物体，例如行人，车辆和人脸，并在RefineDet中引入注意机制以进一步改善性能。

# 个人额外思考
TridenetNet+SSD？因为CVPR 2019的三叉戟网络提到FPN结构的精度是不如图像金字塔的，所以我们是否可以考虑将三叉戟网络的三个检测头放到SSD做一个更高精度的网络？

## 附录

论文地址：https://arxiv.org/abs/1711.06897

源码：https://github.com/sfzhang15/RefineDet

参考博客：https://blog.csdn.net/nwu_nbl/article/details/81110286

# 后记

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)