# 摘要

近些年来，因为弱监督目标检测仅需要**图片分类级别**的label受到了人们广泛的关注，其代价是准确率一定程度的下降。本文提出了一个简单而有效的**弱监督协作目标检测框架**，基于**共享部分特征**，**增强预测相关性**来同时训练强，弱监督两个检测网络。弱监督目标检测网络采取类似WSDDN的结构，强监督目标检测网络采取类似Faster-RNN的结构。最终在数据集上证明了框架的有效性

# 补充WSDDN

WSDDN全称是Weakly Supervised Deep Detection Network，即弱监督深度检测网络。

只依靠image级别的label来对其训练
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020072511260884.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)


整个结构主要是以图像分类和目标检测这两个框架

左边是一个**预训练好的CNN**

右边则是一个**类似FasterRCNN结构**，通过感兴趣池化以及SPP空间金字塔池化，得到**区域(region)级别的特征**

然后**分支成两个数据流**，其中一个数据流使用之前预训练好的CNN做一个**分类**，另外一个数据流则做**检测**任务

其中分类的结果使用的softmax，形式如下，**表示每个区域各个类别的概率**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112629234.png)


而检测流中也用的是softmax，形式如下，**表示每个类别中更具有信息的是哪个区域**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112909702.png)
最后得到了分类和检测结果，通过**矩阵内积(即对应元素相乘)**将两个softmax结果融合

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112915391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725185443861.png)


1-n代表图片级别的label个数，1-C代表是类别个数，Φk的值域在(0, 1)之间，可以认为是类别K出现在image_xi中的概率，如果大于0.5则表示出现，因此后面减了个1/2。最后对所有loss求和

# 整体构造
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112921640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

我们分别有弱检测器记为Dw，强检测器Ds

Dw通过弱监督，如右上角的图片分类任务进行训练，由于Ds没有直接的监督所以不能直接训练。因此我们要求Ds, Dw需要输出**相似的锚框预测**

通过最后的**cosistency Loss**来约束两个检测器的预测。**另外由于两个检测器的目标任务相同，我们在中间共享了提取的特征。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112928214.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

上图是更为详细的结构

整体网络基于**VGG16**搭建

然后分支成两条路，**红色的是强监督检测器，蓝色的是弱监督检测器**

红色的支路则是**类似FasterRCNN结构**，**最终输出分类和回归框**

蓝色的支路先经过**Selective Search  Window**算法，将图片划分成各个小区域。送入SPP金字塔池化结构，后续的全连接层通过两个分支，分别得到分类和回归框结果。**这里要注意的是FC6和FC7这两个是共享网络参数的**。这两个支路结果融合，并与上面红色支路，做了一个**Prediction Consistency Loss**，进一步约束两者输出相同的预测。另外还需要根据图片label，得到图片级别的**分类损失**，对整个弱监督网络进行优化

# 损失计算

相较于WSDDN，该文章的一个创新点也是增强了一个Loss计算来约束两个检测器

首先弱监督检测网络最后对目标输出一个图片类别预测，**这里没有采取WSDDN中使用预训练CNN对其进行判别。而是直接通过给定的图像标签来计算损失**，损失函数也是使用**二分类交叉熵损失函数**。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112935787.png)


第二就是文章里提到的**consistency loss**了，训练整个框架的一大难点是在不需要检测标签情况下，为强检测器定义一个loss来帮助优化。考虑到Ds，Dw两个检测器**最终的目标是输出预测框及类别**，我们提出**使用输出的一致性**来训练Ds检测器

而cosistency loss又由三部分组成
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725112940916.png)


第一部分**CPinter表示Dw和Ds对框类别预测的交叉熵损失函数**，其中pjc表示Dw输出，pic表示Ds输出

第二部分**CPinner表示Ds对框类别预测的损失函数**，由于训练前期，**Dw是一个弱检测器，会引来部分噪声**。所以引入一个CPinner对前期训练做一个约束，两者分配的比例通过β来控制，**β默认设置为0.8**

第三部分**CLinter则是通过Smooth L1 Loss来对回归框预测进行**，**并赋予权重Pj**

整个loss前面还有一个Iij系数，**如果两者预测框的IOU大于0.5则置为1，其他情况则置为0**，进一步刻画两个检测器输出一致性

# 实验结果

实验采取的是SGD优化器，以及β=0.8的超参数设置

在PASCAL VOC 2007 数据集上进行实验

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725111616324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)


**Iw指的是初始弱检测器**

**CLw指的是经过这一框架训练的弱检测器**

**CLs指的是经过这一框架训练的强检测器**

**CSs指的是强监督下的强检测器**

可以看到通过弱监督得到的强检测器效果提升还是很明显的，mAP一下涨了好几个点，并且弱检测器的性能也不差，得到了很大的提升

以下是效果图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200725111601449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

# 总结

WSCL的整体框架思路还是很清晰明了的。沿用了上一代WSDDN大部分的结构。去除了预训练CNN，统一使用图像标签，并引入了一个Prediction Consistency Loss，巧妙的将强，弱两个检测器结合起来，形成监督。考虑到前期弱检测器因性能不够引入噪声，也对其中的权重做出了适当的调整。最后的实验结果也是能看的出来弱监督对检测器带来的提升是挺不错的