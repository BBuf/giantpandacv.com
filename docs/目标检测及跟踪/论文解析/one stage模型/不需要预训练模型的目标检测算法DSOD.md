> 论文：Learning Deeply Supervised Object Detectors from Scratch 

# 1. 前言
DSOD(Deeply Supervised Object Detectors)是ICCV 2017的一篇文章，它表达了一个非常有意思的东西。这篇论文不是从目标检测的高mAP值或者速度更快出发，而是从另外一个角度切入来说明fine-tune后的检测模型和直接训练的检测模型的差距其实是可以减少的，也即是说一些检测模型可以摆脱fine-tune这一过程，并且相比于fine-tune训练出来的模型效果并不会变差。

# 2. 介绍
DSOD这一算法是在SSD的基础上进行的改进，可以简单的看成：

**DSOD=SSD+DenseNet**

作者在论文中提到他也实验了从$0$开始训练**Region Proposal Based**的检测算法比如Faster RCNN，R-FCN等，但这些模型很难收敛。而One-Stage的目标检测算法比如SSD却可以收敛，虽然效果很一般，因此最后作者使用SSD作为了这篇论文接下来讨论的BaseLine。

然后本文基于SSD改进的DSOD在VOC2007 trainval和2012 trainval数据集上训练模型，然后在VOC2007 testset上测试的表现(77.7%mAP)超过了使用fine-tune策略的SSD300S（69.6%mAP）和SSD300（75.8mAP），原文是这样描述的。
> Our DSOD300 achieves 77.7% mAP, which is much better than the SSD300S that is trained from scratch using VGG16 (69.6%) without deep supervision. It is also much better than the fine-tuned results by SSD300 (75.8%)


# 3. 出发点
这篇文章的出发点是什么呢？作者认为几乎的所有检测网络都在使用fine-tune这一技巧，那么一定要用fine-tune吗？作者列出来了3个原因来尝试说明fine-tune不是必须的。原因如下：
- 预训练的模型一般是在分类图像数据集比如Imagenet上训练的，不一定可以迁移到检测模型的数据上（比如医学图像）。
- 预训练的模型，其结构都是固定的，因此如果想要再修改的话比较麻烦。
- 预训练的分类网络的训练目标一般和检测目标不一致，因此预训练的模型对于检测算法而言不一定是最优的选择。

基于上面这几点原因，论文提出了一个从$0开$始的检测模型DSOD，我们接下来看看是怎么设计的吧。

# 4. DSOD网络结构

下面的Figure1分别展示了SSD的整体结构和DSOD的整体结构。

![SSD和DSOD网络结构](https://img-blog.csdnimg.cn/20200329191909856.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Figure1左图的`plain connection`表示SSD网络中的特征融合操作，这里对于$300\times 300$的输入图像来说，一共融合了$6$种不同scale的特征。在每个虚线矩形框内都有一个$1\times 1$的卷积和一个$3\times 3$的卷积操作，这也可以被叫作BottleNeck，也就是$1\times 1$卷积主要起到降维从而降低$3\times 3$卷积计算量的作用。

Figure1右图的`dense connection`表示本文的DSOD引入了DenseNet的思想。`dense connection`左边的虚线框部分和`plain connection`右边的虚线框部分结构一致，区别在于里面的`channel`个数，`dense connection`中$3\times 3$卷积的`channel`个数是`plain connection`中$3\times 3$卷积的一半。主要是因为在`plain connection`中每个BottleNeck的输入是前面一个BottleNeck的输出，而在`dense connection`中，**每个bottleneck的输入是前面所有bottleneck的输出的concate**，所以为了降低计算量减少了通道数。

同时，`dense connection`部分右边的矩形框是下采样模块，包含一个$2\times 2$的最大池化层（降采样作用）和一个$1\times 1$的卷积层（降低`channel`个数的作用），作者也提到先进行降采样再进行$1\times 1$卷积可以进一步减少计算量。

因此可以看出DSOD即是**SSD+DenseNet**的结果。

下面的Table1详细展示了DSOS网络的结构。其中Dense Block就是DenseNet网络的子模块，然后`stem block`由$3\times 3$卷积和$2\times 2$池化组成，后面实验证明了这个`stem block`可以提高mAP值。

![DSOD详细结构](https://img-blog.csdnimg.cn/20200329193308666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 5. 实验结果
在看实验结果之前如果你对DenseNet有点忘记了，建议点下面的链接复习一下。[卷积神经网络学习路线（十二）| 继往开来的DenseNet](https://mp.weixin.qq.com/s/UP_OhkKiIwTSgkrqcEvL5g) 。下面的Table3展示了在VOC2007 testset上不同参数的实验结果。

![第3，4行的对比可以看出BottleNeck的channel个数越多，mAP相对越高。第5、6行的对比可以看出growth rate从16变成48，可以提高4.8%的mAP。第6，9行的对比可以看出stem block的存在可以提高2.8%的mAP。](https://img-blog.csdnimg.cn/20200329212010248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table4则展示了更丰富的结果。

![可以看出SSD在没有预训练模型的情况下也是可以收敛的，不过效果一般。但如果使用本文的DSOD则可以达到使用预训练模型效果，并且结果还偏好一点](https://img-blog.csdnimg.cn/20200329212937392.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

更多实验结果请参考原文。

# 6. 总结
DSOD是在SSD的基础上结合了DenseNet的思想，使得网络可以在不使用预训练模型的条件下收敛得和使用预训练模型的BaseLine模型一样好，另外DenseNet的引入也使得相比SSD来说DSOD的参数数量大大减少，注意参数量不等于推理速度会变快。如果专注于特殊图像检测或者难以搞定预训练模型的场景这篇文章的思想是值得借鉴的。

# 7. 参考
- https://blog.csdn.net/u014380165/article/details/77110702
- 论文链接：https://arxiv.org/abs/1708.01241
- 代码：https://github.com/szq0214/DSOD

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)