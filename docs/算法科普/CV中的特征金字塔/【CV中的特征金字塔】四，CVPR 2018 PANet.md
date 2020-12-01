# 1. 前言
PANet是CVPR 2018的一篇实例分割论文，作者来自港中文，北大，商汤和腾讯优图。论文全称为：**Path Aggregation Network for Instance Segmentation** ，即用于实例分割的路径聚合网络。PANet在Nask RCNN的基础上做了多处改进，充在COCO 2017实例分割比赛上夺冠，同时也是目标检测比赛的第二名。接下来就一起来看看吧。

# 2. 贡献
PANet整体上可以看做是对Mask-RCNN做了多个改进，充分的融合了特征，具体来说PANet的贡献可以总结为如下几点。
- **FPN**（这个已经有了，不算论文的贡献）。
- **Bottom-Up Path Augmentation**。
- **Adaptive Feature Pooling**。
- **Fully-Connected Fusion**。

接下来我们把这几个贡献点讲清楚就OK。

# 3. PANet整体结构
为了更好的去讲解上面几个小点，先看一下PANet的整体结构，如论文的Figure1所示。

![PANet 结构](https://img-blog.csdnimg.cn/20200301191526416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到PANet的结构的组成部分**就是我们讲的FPN，Bottom-Up Path Augmentation，Adaptive Feature Pooling，Fully-Connected Fusion这四个小模块**了，接下来我们详细讲解。

# 4. FPN
FPN是CVPR 2017的一篇论文，昨天已经讲过了这里就不多说了，讲解的推文地址如下：[【CV中的特征金字塔】二，Feature Pyramid Network](https://mp.weixin.qq.com/s/d2TSeKEZPmVy1wlbzp8BNQ)

# 5. Bottom-up Path Augmentation
Bottom-up Path Augemtation的提出主要是考虑到网络的浅层特征对于实例分割非常重要，不难想到浅层特征中包含大量边缘形状等特征，这对实例分割这种像素级别的分类任务是起到至关重要的作用的。因此，为了保留更多的浅层特征，论文引入了Bottom-up Path Augemtation。我们从Figure1中可以看到，作者用**红绿两个箭头**来表示这个结构是如何起作用的？

**红色的箭头表示在FPN中，因为要走自底向上的过程，浅层的特征传递到顶层需要经过几十个甚至上百个网络层，当然这取决于BackBone网络用的什么，因此经过这么多层传递之后，浅层的特征信息丢失就会比较严重。**  

绿色的箭头表作者添加了一个Bottom-up Path Augemtation结构，这个结构本身不到10层，这样浅层特征经过原始FPN中的横向连接到P2然后再从P2沿着Bottom-up Path Augemtation传递到顶层，经过的层数不到10层，能较好的保存浅层特征信息。**注意，这里的N2和P2表示同一个特征图。** 但N3,N4,N5和P3,P4,P5不一样，实际上N3,N4,N5是P3,P4,P5融合后的结果。

Bottom-up Path Augemtation的详细结构如Figure2所示，是一个常规的特征融合操作，这里展示的是$N_i$经过一个尺寸为$3\times 3$，步长为$2$的卷积之后，特征图尺寸减小为原来的一半然后和$P_{i+1}$这个特征图做add操作，得到的结果再经过一个卷积核尺寸为$3\times 3$，$stride=1$的卷积层得到$N_{i+1}$。

![Bottom-up Path Augemtation详细结构](https://img-blog.csdnimg.cn/20200301193655861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. Adaptive Feature Pooling
这一结构做的仍然是特征融合。论文指出，在Faster-RCNN系列的标检测或分割算法中，**RPN网络得到的ROI需要经过ROI Pooling或ROI Align提取ROI特征**，这一步操作中每个ROI所基于的特征都是单层特征，FPN同样也是基于单层特征，因为检测头是分别接在每个尺度上的。比如ResNet网络中常用的`res5`的输出。

在引入Adaptive Feature Pooling作者做了Figure3这个实验，可以看到图中有4条不同的曲线，分别对应了FPN网络中的4个特征层，然后每一层都会经过RPN网络获得ROI，因此**这4条曲线就对应了4个ROI集合**。图像中的横坐标表示的是ROI集合提取的不同层特征的占比。例如蓝色曲线表示level1的特征层，应该是尺度最小的ROI的集合，这一类型的ROI所提取的特征仅有30%是来自level1特征层的特征，剩下的70%均来自level2,3,4。其他level的特征层同理，因此就有了作者的这个思考和改进，也就是说对每个ROI提取不同层的特征并做融合，这对于提升模型效果显然是有好处的。 

![Adaptive Feature Pooling 灵感](https://img-blog.csdnimg.cn/20200301195824690.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

本文提出的**Adaptive Feature Pooling**则是将单层特征换成多层特征，即每个ROI需要和多层特征（论文中是4层）做ROI Align的操作，然后将得到的不同层的ROI特征融合在一起，这样每个ROI特征就融合了多层特征。

Adaptive Feature Pooling的详细结构如Figure6所示。

![Adaptive Feature Pooling详细结构](https://img-blog.csdnimg.cn/20200301194244274.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

RPN网络获得的每个ROI都要分别和$N_2,N_3,N_4,N_5$特征层做ROI Align操作，这样个ROI就提取到4个不同的特征图，然后将4个不同的特征图融合在一起就得到最终的特征，后续的分类和回归都是基于此最终的特征进行。 


# 7. Fully-Connected Fusion
PANet最后一个贡献是提出了Fully-connected Fusion，这是对原有的分割支路(FCN)引入一个前景二分类的全连接支路，通过融合这两条支路的输出得到更加精确的分割结果。这个模块的具体实现如Figure4所示。


![Fully-Connected Fusion模块](https://img-blog.csdnimg.cn/20200301200022182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

从图中可以看到这个结构主要是在原始的Mask支路（即带deconv那条支路）的基础上增加了下面那个支路做融合。增加的这个支路包含$2$个$3\times 3$的卷积层，然后接一个全连接层，再经过reshape操作得到维度和上面支路相同的前背景Mask，即是说下面这个支路做的就是前景和背景的二分类，输出维度类似于文中说的$28\times 28\times 1$。而上面的支路输出维度类似$28\times 28\times K$，其中$K$代表数据集目标类别数。最终，这两条支路的输出Mask做融合以获得更加精细的最终结果。

# 8. 跨卡训练BN
作者还提到PANet的训练使用了跨卡BN层计算，引入这个的原因主要是为了在训练过程中BN层的计算会更稳定。因为BN层的计算依赖batch_size的设置，设置得过小会导致BN层的参数不稳定，但Two-Stage的目标检测算法搭配caffe框架，其batch_size会非常小，因此跨卡BN是必须的。

# 9. 实验结果
下面的Table1展示了PANet和Mask-RCNN，COCO 2016的冠军 FCIS算法的分割效果对比。

![PANet的精度最高](https://img-blog.csdnimg.cn/20200301200937791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table2则展示了和Mask RCNN、FCIS、RentinaNet算法在COCO数据集上的检测效果对比，可以看到。精度提升是很大的。

![PANet在检测上也效果拔群](https://img-blog.csdnimg.cn/20200301201241604.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Tabel3则展示了这篇论文的消融实验，即文章提出的创新点带来的精度提升，其中RBL是这篇文章的BaseLine，也就是带FPN的Mask-RCNN。

![消融实验](https://img-blog.csdnimg.cn/20200301201531269.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

而Table6和Table7也值得关注，因为这里透漏了在实例分割中的一些涨点方法：

- Deformable Convolution（DCN）可变性卷积，这天生就适合分割任务吧？
- Testing tricks，提了2.5个mAP，主要包括Multi-Scale Tesing，这个比较耗时，但是效果一般都不差。
- Horizontal Flip Tesing，不了解。
- larger model 更深更宽的网络一般精度更好。
- ensemble 模型融合，融合多个了ResNeXt-101、ResNet-269、SE-ResNeXt-101等网络的结果。 

![冠军Tricks](https://img-blog.csdnimg.cn/20200301202013208.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 10. 附录

- 论文原文：https://arxiv.org/pdf/1803.01534.pdf
- 代码实现：https://github.com/ShuLiu1993/PANet
- 参考资料：https://blog.csdn.net/u014380165/article/details/81273343

# 11. 推荐阅读

- [【CV中的特征金字塔】一，工程价值极大的ASFF](https://mp.weixin.qq.com/s/2f6ovZ117wKTbZvv2uRwdA)
- [【CV中的特征金字塔】二，Feature Pyramid Network](https://mp.weixin.qq.com/s/d2TSeKEZPmVy1wlbzp8BNQ)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)