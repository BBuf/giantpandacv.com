![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216203640616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 前言
今天为大家介绍一篇图森科技在ICCV 2019的目标检测论文《Scale-Aware Trident Networks for Object Detection》，简称TridentNet，中文翻译为三叉戟网络。论文地址见附录。

# 背景
我们知道在目标检测任务中，尺度变化一直是很关键的问题。针对尺度变化问题，也有很多的方案被提出，如Figure 1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216204007395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中(a)图表示多尺度图像金字塔网络，直接对图像进行不同尺度的缩放。(b)是FPN网络，借鉴了金字塔结构将分辨率信息和语义信息相结合，克服不同层，不同尺度带来的问题。这两种方法的目的都是让模型对尺寸不同的目标具有不同不同的感受野。除此之外还有SNIP，SNIP主要包含了两个改进点：

1、为了减少domain-shift，在梯度回传的时候只将和预训练模型所基于的训练数据尺寸相应的ROI的梯度进行回传。

2、借鉴了多尺度训练思想，引入图像金字塔来处理数据集种不同尺寸的数据。虽然图像金字塔的效率比较低，但通过对原图不同比例的缩放，充分利用了模型的表征能力。相比之下，FPN产生多层次的特征，但也牺牲了不同尺度下的特征一致性。总结一下就是，图像金字塔虽然慢但是精度高。而FPN虽然快，但相比于图像金字塔牺牲了精度。有没有办法将这两者统一起来呢？TridentNet横空出世。

# 主要贡献
论文提出了TridentNet，基于ResNet-101骨架网络在COCO数据集上达到了单模型48.4的准确率，刷新COCO纪录。TridentNet的主要贡献在于：

- 首次提出感受野(receptive filed)对目标检测任务中不同尺度大小目标的影响，并进行相关实验验证。
- 提出了适应于多尺度的目标检测框架TridentNet。
- 使用参数共享的方法，提出了训练3个分支，但测试时只使用其中一个分支，这样保证前向推理的时候不会有额外的参数和计算量增加。
- 使用ResNet-101为backbone的TridentNet在COCO数据集上达到了48.4的map，真正的SOTA。

# 膨胀卷积
假设膨胀率为$ds$，使用的卷积核大小为$3\times 3$，则使用膨胀卷积的感受野大小为$3+2\times 2\times (ds-1)$，例如：

- $ds=1$，表示不进行膨胀，感受野大小为$3\times 3$
- $ds=2$，表示膨胀卷积系数为$2$，感受野大小为$7\times 7$
- $ds=4$，表示膨胀卷积系数为$4$，感受野大小为$15\times 15$
从Table1可以看到，随着感受野的增大，小目标的检测准确性也开始下降，但是大目标的检测准确性开始上升，Table1如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216210653828.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 网络结构
Figure2展示了TridentNet的网络结构。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216210916195.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

TridentNet模块包含3个完全一样的分支，唯一不同的只是膨胀卷积的膨胀速率。从上到下，膨胀率分别为1,2,3，分别检测小，中，大的目标。TridentNet的构造和改进是通过将一些常规卷积快替换为三叉戟块(Trident Block)。其中Trident Block由多个平行分支组成，除了膨胀卷积的膨胀速率之外，每个分支和原始的卷积块有完全相同的结构。下面以ResNet为例，对于一个残差模块，包括三个卷积核即$1\times 1$，$3\times 3$，$1\times 1$。详细结构如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216211812532.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到三个分支由于膨胀卷积的膨胀速率不同使得网络拥有了更多可供选择的感受野。通常用三叉戟块替换主干的最后一个阶段的卷积块，因为三叉戟块产生的感受野有较大的差异，足以执行目标检测任务。同时，三叉戟块的三个分支之间的权重是共享的，只有膨胀速率不同，这种设置使得权值共享更加简单。权值共享有以下几个优点：

- 可以不增加额外的参数量。
- 和论文的出发点一致，即不同尺度的物体应该以同样的表征能力进行统一的转化。
- 在不同的感受野下，可以对不同尺度范围训练相同的参数。

# 训练和测试
TridentNet在训练过程中会对每一个分支进行优化。因此，需要对目标的ground truth的大小进行测试，即：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216221313150.png)

其中，$w$，$h$代表ground truth的宽和高。$l_i$，$u_i$代表实验中定义的第$i$个分支的目标最小面积和最大面积
。在COCO数据集上分别为$32\times 32$和$96\times 96$。基于此公式实现目标匹配原则，即小的目标走第一个分支，中等目标走第二个分支，大的目标走第三个分支。而不是所有的目标都走一个分支，这样就可以有针对性的训练。
在测试时，只用中间的分支进行推理，然后对结果进行NMS后处理，最后输出预测的目标信息。当然这样做会带来一些精度损失，大概在0.5-1个map值，但这样的好处在于不会引入额外的参数，不会增加额外的计算量。

# 实验结果
本文的另外一个值得称道的地方就是实验做的非常棒。首先来看Multi-branch ，Weight-sharing， Scale-aware有效性证明，如Table2所示。可以看到都比baseline好并且当三部分都加上的时候性能达到最高。注意一下，Table2是在CICO验证集上进行的测试。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216222034496.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后Table3展示了TridentNet模块分支个数对AP值的影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216222529443.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后TridentNet模块在ResNet不同block中的实验结果如Table4所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216222640853.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后论文探索了TridentNet各个分支的检测精度，如Table5所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216223255981.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后论文还给出了TridentNet 中间分支在coco测试的结果，如Table6所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191216223455329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

上面的实验都是在COCO验证集上的结果，接下来给出模型在COCO测试集上相对于其他SOAT网络的精度，如Table 7所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121622360272.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 总结
好了，做一个简短的总结吧。TridentNet是一种尺度敏感的检测网络，并且训练过程也要进行多尺度训练。检测准确性很高，并且不会有额外的参数，额外的计算量，是ASPP结构的灵活应用。

# 附录
论文原文：https://arxiv.org/pdf/1901.01892.pdf
官方实现：https://github.com/TuSimple/simpledet/tree/master/models/tridentnet

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)