# 前言
继续Anchor-Free探索。前面介绍了[【目标检测Anchor-Free】ECCV 2018 CornerNet](https://mp.weixin.qq.com/s/cKOna7GfTwl1X1sgYNXcEg)，相信大家对Anchor-Free目标检测算法有基本的认识和理解了。但是在介绍这个论文的时候最后提到CornerNet最大的瓶颈在于角点检测的不准确，这篇文章主要针对这一点进行了改进，提出了ExtremeNet。论文原文和代码见附录。

# 介绍
这篇论文提出了一种新的Anchor-Free的目标检测网络，借鉴了CornerNet的思想，并对其进行改进，取得了不错的效果。这篇论文的思路是不再像CornerNet那样去检测目标的左上角和右下角，而是检测目标的4个极值点，即最上点，最下点，最左点，最右点，然后这4个点包围的框就是目标框。在COCO数据集上取得了43.7%的精度，速度大概300ms一张，没有什么优势，但这篇论文的思想仍然值得我们一看。下面的FIgure1展示了ExtremeNet的实例检测结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123181641654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

ExtremeNet的大致流程是使用关键点检测算法，对每个类别预测4张极值点热力图和1张中心点热力图，然后通过暴力枚举找到有效的点组。CornerNet是第一个将关键点用于目标检测的算法，ExtremeNet和它主要有2处不同，第一点是CornerNet的角点通常是在目标的外部，往往没有强烈的局部特征。而ExtremeNet的极值点就在物体的边缘上，视觉上很好辨认。第二点是ExtremeNet纯粹依赖于几何关系进行极值点分组，没有隐含的特征学习，效果更好。

# 算法准备工作
**极值和中心点**  传统的目标检测标注方法通常是使用矩形框对目标进行标注，而矩形框一般用左上角和右下角两个点来表示。而在ExtremeNet中使用四点标注法，即用一个目标上下左右四个方向的极值来标注，并且通过这4个极值点也可以算出目标的中心点。

**关键点检测**  通常为每个关键点生成一个通道的heatmap，每个channel对应1个分类。损失函数可以使用L2损失，或者使用逐像素的交叉熵损失，记预测的Heatmap为$\hat{Y}$，$Y$是ground truth的heatmap，这是一个Gaussian map，在ground truth关键点处具有峰值。使用104层的Hourglass网络作为Backbone网络，生成大小为$H\times W$的heatmap，即$\hat{Y}\in (0,1)^{H\times W}$。ground truth的heatmap是多峰值的heatmap，每个关键点决定对应高斯核的均值，而高斯核的标准差则可以取一个固定值或者正比于目标的尺寸。

**CornerNet** 上一篇推文讲了，就不再赘述。

**Deep Extreme Cut** CVPR 2018提出的论文，给定一副图像，和若干个极值点，即可得到一个类别未知分割mask。如果需要得到更加精细的分割效果的话，正好可以接在ExtremeNet的输出之后，便可以轻松得到分割结果。这篇论文的地址也放在附录。

# ExtremeNet目标检测
ExtremeNet的输出通道是$5\times C + 4 \times 2$。其中$C$表示一共有多少类目标，对于每个类别预测一张heatmap和一张center map因此是5张。然后对于每种极值点heatmap，再预测两张offset map（分别对应X和Y方向），注意是所有类别共享并且center map没有，因此只有$4\times 2$张。整个流程如Figure3所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123195315547.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 中心点分组
参考CornerNet，首先提取heatmap中的极值点，极值点定义为$3\times 3$滑动窗口中的极大值并且响应大于预先设定的阈值$t_p$。然后对于每一种极值点组合（进行合适的剪枝以减小遍历规模），计算它们的中心点，如果center map对应位置上的响应超过了预先设定的阈值$t_c$，则将这5个点作为一个备选，该备选组合的分数为5个对应点的分数平均值。论文中设置$t_p=0.1$并且$t_c=0.1$，一个可视化的过程如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123200315172.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

具体的算法流程如Algorithm1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123200411111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
## Ghost box suppression
通过上面的算法有可能会得到一些高置信度但是是假阳性的检测结果。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123200836914.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

上面的三个实线框表示三个ground truth bbox，而下面的虚线bbox则是ghost bbox，可以发现在训练过程中是可以知道下面的box是假阳性检测，而在测试过程中是并不知道的。为了对这一情况进行处理，论文使用了soft NMS算法，根据Algorithm1，假设最后预测出的一个bbox，其分数为$s_0$，所有被这个bbox包含的其它bbox的得分之和超出$s_0$的三倍，那么这个bbox的score降为原来的一半即$s_0/2$，后面再执行置信度阈值为0.5的nms就可以把这种ghost bbox滤掉了。为什么3个就行？从上图可以看到至少要有3个bbox才可以产生ghost bbox的情况，并且只能是奇数个才能出现这种情况，如果是比3更大的奇数也就是说需要很多个目标的中心点在同一水平或者垂直线上，这在现实情况中是几乎不会出现的。

## 边缘聚合
极值点的定义不唯一，这导致如果目标沿着水平或者垂直方向边缘形成极值点的话（如汽车顶部），那么该边缘的点可能都会被当成极值点。网络会对沿着目标任何对齐的边缘产生**弱响应**，而非单个强响应。这会导致两个问题：一是，较弱的响应可能低于预设的极值点阈值，导致漏掉所有的点；二是，即使侥幸超过了阈值，但其score可能还是会比稍微旋转过的目标低（在两个方向上都有较大的响应）。

为了解决这个问题，论文提出使用边缘聚合（edge aggregation）方法。对于每个极值点，其响应都是局部窗口最大值，对于left/right方向的极点，按照垂直方向进行聚合，对top/bottom方向的角点，则按照水平方向进行聚合。聚合的时候注意是聚合那些score单调递减的极点，并且在这个方向上有局部最小score时停止聚合，这是为了避免多个目标bbox 沿轴排列，从而将不同目标的极点score聚合到一起。

令$m$为top或者bottom类型的极点，记$N_i^m=\hat{Y_{m_x+i,m_y}}$表示包含$m$这个极点的水平线段在heatmap上的响应值。在令$i_0<0<i_1$表示左右两个最近极小值点，即$N_{i_0-1}^m>N_{i_0}^m$，然后极点$m$的score被更新为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020012320524533.png)

其中，$\lambda_{aggr}$是聚合权重，论文取0.1。效果如Figure4所示：

![(a)是原始heatmap图，在边上的模型预测响应较弱，经过edge aggregation后，边的中间点处的响应得到加强。](https://img-blog.csdnimg.cn/20200123210527587.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## Extreme Instance Segmentation
基于ExtremeNet检测到的4个极值点，通过在每两个极值点之间插入中间点，来补成八边形掩模。具体做法是：首先根据4个极值点找到外接矩形；然后对每个极值点，在其所属的矩形边上，沿着两个方向各延长矩形边的1/8；最后将8个点连接起来，如果遇到了矩形边界则截断，得到最后的八边形分割掩模估计结果。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123211013103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

同时将ExtremeNet获得的极值点传到上面提到的Deep Extreme Cut网络可以获得一个类别未知的分割Mask，注意**类别其实在ExtremeNet中已经知道了**，这就相当于一个双阶段的实例分割。效果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123211228473.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


# 实验结果
不多讲了，直接贴一下核心结果图，在MSCOCO上map值达到了43.7%，相比于CornerNet涨点5.3%，还是不错的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200123211319103.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论
这篇论文改善了CornerNet角点预测困难的问题，提出的ExtremeNet不仅仅在精度上有较大提升并且可以用来做分割，思想是蛮有趣的。

# 附录

- 论文原文：https://arxiv.org/pdf/1901.08043.pdf
- Deep Extreme Cut论文：https://arxiv.org/abs/1711.09081
- 代码：https://github.com/xingyizhou/ExtremeNet
- 参考1：https://blog.csdn.net/sinat_37532065/article/details/86693930
- 参考2：https://zhuanlan.zhihu.com/p/67386907

# 同期文章

- [目标检测算法之Anchor Free的起源：CVPR 2015 DenseBox](https://mp.weixin.qq.com/s/gYq7IFDiWrLDjP6219U6xA)
- [【目标检测Anchor-Free】ECCV 2018 CornerNet](https://mp.weixin.qq.com/s/cKOna7GfTwl1X1sgYNXcEg)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)