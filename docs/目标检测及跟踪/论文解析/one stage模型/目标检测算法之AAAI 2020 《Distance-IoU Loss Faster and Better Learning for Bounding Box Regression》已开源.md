# 前言
前面介绍了CVPR 2019的GIoU Loss，推文地址如下：[点这里](https://mp.weixin.qq.com/s/CNVgrIkv8hVyLRhMuQ40EA)，同时GIoU Loss里面也引入了IoU Loss，这个首先需要明确一下。然后AAAI 2020来了一个新的Loss，即是本文要介绍的DIoU Loss。论文原文地址见附录。

# 背景
我们先来回归一下IoU Loss和GIoU Loss。IoU Loss可以表示为：$L_{IoU}=1-\frac{|B \bigcap B^{gt}|}{|B\bigcup B^{gt}|}$，从IoU的角度来解决回归损失，但它的缺点是当两个框不想交时，IOU-Loss始终为1，无法给出优化方向。因此GIoU来了，GIoU可以用下面的公式表示：

$L_{GIoU}=1-IoU+\frac{|C-B \cup B^{gt}|}{|C|}$

可以看到GIoU在IoU的基础上添加了一项，其中$C$表示包含两个框的最小矩形，这样就可以优化两个框不相交的情况。但GIoU仍然存在一个问题是，当两个框相交时，GIoU损失退化为了IoU损失。导致在预测框bbox和ground truth bbox包含的时候优化变得非常困难，特别是在水平和垂直方向收敛难。这里一个猜想是在水平和垂直方向，$C$值的增长没有在其他方向快，因此对这两个方向的惩罚力度不够，导致放慢了收敛速度。如论文的Figure2和Fiigure4所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215214706803.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215225755374.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

同时为了更加形象的说明这一点，我们看一下论文的Figure1：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215230031927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中Figure1中的上面3张图表示GIoU的回归过程，其中绿色框为目标框，黑色框为Anchor，蓝色框为不同迭代次数后，Anchor的偏移结果。第2行的3张图则表示了DIoU的回归过程。从图中可以看到，在GIoU的回归过程中，从损失函数的形式我们发现，当IoU为0时，GIoU会先尽可能让anchor能够和目标框产生重叠，之后GIoU会渐渐退化成IoU回归策略，因此整个过程会非常缓慢而且存在发散的风险。而DIoU考虑到anchor和目标之间的中心点距离，可以更快更有效更稳定的进行回归。
# 问题提出
基于上诉分析，作者提出了如下两个问题：

- 一，直接最小化Anchor和目标框之间的归一化距离以达到更快的收敛速度是否可行？
- 二，如何使回归损失在与目标框有重叠甚至有包含关系时更准确，收敛更快？

# DIoU Loss
论文为了解决第一个问题，提出了**Distance-IoU Loss**(DIoU Loss)。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215220032147.png)

这个损失函数中，$b$，$b^{gt}$分别代表了Anchor框和目标框的中心点，$p$代表计算两个中心点的欧式距离，$c$代表的是可以同时覆盖Anchor框和目标框的最小矩形的对角线距离。因此DIoU中对Anchor框和目标框之间的归一化距离进行了建模。直观展示如Figure 5所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215221343124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

DIoU Loss的优点如下：

- 和GIoU Loss类似，DIoU Loss在和目标框不重叠时，仍然可以为边界框提供移动方向。
- DIoU Loss可以直接最小化两个目标框的距离，因此比GIoU Loss收敛快得多。
- 对于包含两个框在水平方向和垂直方向上这种情况，DIoU Loss可以使回归Loss 下降非常快，而GIoU Loss几乎退化为IoU Loss。

# CIoU Loss
为了回答第二个问题，作者提出了**Complete-IoU Loss**。一个好的目标框回归损失应该考虑三个重要的几何因素：重叠面积，中心点距离，长宽比。GIoU为了归一化坐标尺度，利用IOU并初步解决了IoU为0无法优化的问题。然后DIoU损失在GIoU Loss的基础上考虑了边界框的重叠面积和中心点距离。所以还有最后一个点上面的Loss没有考虑到，即Anchor的长宽比和目标框之间的长宽比的一致性。基于这一点，论文提出了CIoU Loss。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215223039116.png)

从上面的损失可以看到，CIoU比DIoU多了$\alpha$和$v$这两个参数。其中$\alpha$是用来平衡比例的系数，$v$是用来衡量Anchor框和目标框之间的比例一致性。它们的公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215223405498.png)![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121522342215.png)

然后在对$w$和$h$求导的时候，公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215224025726.png)

因为$w^2+h^2$这一项在计算的时候会变得很小，因为$w$,$h$的取值范围是$[0,1]$。而在回归问题中回归很大的值是很难的，因此一般都会对原始的$w$，$h$分别处以原图像的长宽。所以论文直接将$w^2+h^2$设为常数1$1$，这样不会导致梯度的方向改变，虽然值变了，但这可以加快收敛。

从$\alpha$的定义式来看，损失函数会更加倾向于往重叠区域增多的方向优化，尤其是IoU为0的情况，这满足我们的要求。
同时，在进行nms阶段，一般的评判标准是IOU，这个地方作者推荐替换为DIOU，这样考虑了中心点距离这一个信息，效果又有一定的提升。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121522375496.png)

# 可视化实验
论文做了一个有趣的实验来探索IoU和GIoU存在的问题，我觉得还是有必要介绍一下，实验如Figure3所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215225022140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中，绿色的框代表回归任务需要回归的7个不同尺度的目标框，7个目标框的中心坐标都是`[10,10]`。蓝色的点代表了所有Anchor的中心点，各个方向都有，各个距离也都有。一共有$5000$个蓝色点，有$5000\times 7\times 7$个Anchor框，并且每个框都需要回归到$7$个目标框去，因此一共有$5000\times 7\times 7\times 7$个回归等式。从Figure3(b)中我们可以看到在训练同样的代数后（$200$次），三个loss最终每个Anchor的误差分布。从IoU误差的曲线我们可以发现，Anchor越靠近边缘，误差越大，那些与目标框没有重叠的Anchor基本无法回归。从GIoU误差的曲线我们可以发现，对于一些没有重叠的Anchor，GIoU的表现要比IoU更好。但是由于GIoU仍然严重的依赖IoU，因此在两个垂直方向，误差很大，基本很难收敛，这就是GIoU不稳定的原因。从DIoU误差的曲线我们可以发现，对于不同距离，方向，面积和比例的Anchor，DIoU都能做到较好的回归。CIoU类似。
# 实验结论
Table1给出了分别在YOLOv3上使用IoU Loss，GIoU Loss，DIoU Loss, CIoU Loss获得的AP值。可以看到CIoU Loss涨了快3个点，证明了这种Loss的有效性。遗憾的是论文没有给出其他的对比数据了，究竟有没有用欢迎大家去试，个人认为这种带有工程性Trick的论文是最值得去尝试的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191215223557744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 附录

论文原文：https://arxiv.org/pdf/1911.08287.pdf
参考博客：https://blog.csdn.net/qiu931110/article/details/103330107
源码实现：https://github.com/Zzh-tju/DIoU-darknet
# 后记
关于DIoU Loss和CIoU Loss就介绍到这里了，希望对大家目标检测升点有帮助。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)