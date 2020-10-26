# 前言
前面已经讲解完了RCNN系列的三篇论文，目标检测项目也基本可以跑起来了。今天要讲的FPN也是Two Stage目标检测算法中非常值得推敲的论文，它进一步优化了Faster-RCNN，使得对小目标的检测效果更好，所以一起来看看吧。

# 背景
Faster-RCNN选取一个特征提取网络如VGG16做backbone，然后在高层特征（如VGG16后面的conv4）接RPN和检测头进行网络。正是由于Faster-RCNN基于图像的高级特征，这就导致对小目标的检测效果很差。而CV领域常用的处理尺度问题的办法就是特征金字塔，将原图以不同的比例采样，然后得到不同分辨率的图像进行训练和测试，在多数情况下确实是有效的。但是特征金字塔的时间开销非常大，导致在工程中应用是及其困难。FPN从新的角度出发提出了一个独特的特征金字塔网络来避免图像金字塔产生的超高计算量，同时可以较好的处理目标检测中的尺度变化问题，对小目标检测更鲁棒，同时在VOC和COCO数据集上MAP值均超过了Faster-RCNN。

# 简介
我们使用下图来阐释我们是如何处理尺度变化大的物体检测的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113140143419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- 上图(a)是处理这类问题最常用的方法，即特征金字塔，这种方法在传统的手动设计特征的方法中非常常用，例如DPM方法使用了接近10种不同的尺度获得了不错的效果。
- 上图(b)是在CNN提出之后出现的，因为神经网络模型对物体尺度本身有一定的鲁棒性，所以也取得了不错的性能，但最近的研究表明将特征金字塔和CNN结合仍可以提升性能，这说明基于单层特征的检测系统仍存在对尺度变化敏感的缺点。
- 上图(c)表示除了使用图像金字塔，我们可以使用深度学习本身的多层次结构来提取多尺度特征。最常见的就是SSD算法中利用多个特征层来分别做预测。但这种方式也有一些缺点就是浅层的语义特征比较弱，在处理小物体时表现得不够好。
- 上图(d)表示本文提出的FPN（Feature Pyramid Network ），它它能较好的让各个不同尺度的特征都具有较强的语义信息。FPN结合Faster RCNN可以在COCO物体检测比赛中取得当前单模型的最佳性能（SOTA）。另外，通过对比实验发现，FPN能让Faster RCNN中的RPN网络的召回率提高8个点；并且它也能使Fast RCNN的检测性能提升2.3个点（COCO）和3.8个点（VOC）。

# FPN结构
下图表示FPN的整体结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113141952128.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)我们可以看到FPN的整体结构分为**自底向上**和**自顶向下和侧向连接**的过程。接下来我们分别解释一下这两个关键部分。
### 自底向上
这一部分就是普通的特征提取网络，特征分辨率不断缩小，容易想到这个特征提取网络可以换成任意Backbone，并且CNN网络一般都是按照特征图大小分为不同的stage，每个stage的特征图长宽差距为2倍。在这个自底向上的结构中，一个stage对应特征金字塔的一个level。以我们要用的ResNet为例，选取conv2、conv3、conv4、conv5层的最后一个残差block层特征作为FPN的特征，记为{C2、C3、C4、C5}，也即是FPN网络的4个级别。这几个特征层相对于原图的步长分别为4、8、16、32。

### 自上向下和侧向连接
自上向下是特征图放大的过程，我们一般采用上采样来实现。FPN的巧妙之处就在于从高层特征上采样既可以利用顶层的高级语义特征（有助于分类）又可以利用底层的高分辨率信息（有助于定位）。上采样可以使用插值的方式实现。为了将高层语义特征和底层的精确定位能力结合，论文提出了类似于残差结构的侧向连接。向连接将上一层经过上采样后和当前层分辨率一致的特征，通过相加的方法进行融合。同时为了保持所有级别的特征层通道数都保持一致，这里使用1*1卷积来实现。在网上看到一张图，比较好的解释了这个过程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113143347813.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
FPN只是一个特征金字塔结构，需要配合其他目标检测算法才能使用。

# 实验
## 1.FPN对RPN网络的影响
如下表所示，论文做了6个实验。
- (a)基于conv4的RPN，原始原始的RPN。
- (b)基于conv5的RPN。
- (c) 完整FPN。
- (d)只用了自底向上的多层特征，没有自顶向下的特征。
- (e)用了自顶向下的特征，但不用侧向连接。
- (f)用了自顶向下的特征，也用了横向特征融合，但只用最后的P2做预测。（完整的预测是使用每一个level的特征${P_k}$做预测）

分析表格可知，自顶向下的特征、横向连接、尺度分离、多个层次的预测是提升FPN性能的关键。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113143938618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)为了更好的理解，放一张Faster-RCNN结合FPN的细致结构图如下，图可以在最下方的github工程找到：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113151827913.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
## 2.FPN对Fast RCNN的影响
使用和实验1相同的规则对Fast RCNN做了实验，结果如下表所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191113151958420.png)
# 结论
本文提出了一种简单、有效的建立特征金字塔的方式。它的使用对RPN方法和Fast/Faster RCNN方法都有极大的性能提升。另外，它的训练和测试时间和普通的Faster RCNN方法相差很小。因此，它可以作为图像特征金字塔的一种较好的替代。

# 源码
https://github.com/unsky/FPN