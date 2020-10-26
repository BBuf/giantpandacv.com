# 1. 前言
看过前几天[【CNN调参】目标检测算法优化技巧](https://mp.weixin.qq.com/s/sSxbqLNVV7kwCLHRCOiM8w)的同学应该知道，ASFF的作者在构建Stronger YOLOV3 BaseLine的时候就用到了Guided Anchoring这种Trick。这篇论文题目为《Region Proposal by Guided Anchoring》，中了CVPR 2019。这篇论文提出了一种新的Anchor生成方法Guided Anchoring，不同于以前的固定Anchor或者根据数据进行聚类Anchor的做法，通过Guided Anchoring可以预测Anchor的形状和位置得到稀疏的Anchor，另外作者提出了Feature Adaption模块使得Anchor和特征更加匹配。论文作者也在知乎上清晰的介绍了这项工作，感兴趣可以去看看作者是如何思考的。地址为：`https://zhuanlan.zhihu.com/p/55854246`。

# 2. Guided Anchoring
像SSD，Faster-RCNN，YOLOV3等多种基于Anchor的目标检测网络都是将一系列形状和宽高比不同的Anchor预设在特征图的每个像素点上，但这会产生两个明显的问题。
- 位置上：Anchor是密集的，分布在图像中的各个地方，但是图像中大部分区域都是不包含物体的，因此大多数Anchor是无效的，所以我们希望有一种方法可以得到和目标所在位置匹配的稀疏Anchor。
- 形状上：Anchor通常是预设好或者按照数据集聚类来的，这两种方式都是固定的。并不一定能完全贴合实际并且对特殊大小或长宽比很大的目标检测鲁棒性就会变差。

基于上面的两个问题，本文提出了Guided Anchoring方法。我们一般用$4$个变量$(x,y,w,h)$来描述一个Anchor，即中心点坐标和长宽，可以写成如下的联合概率分布：

$p(x,y,w,h|I)=p(x,y|I)p(w,h|x,y,I)$

从这个公式我们可以看出，Anchor在不同的位置有不同的出现概率$p(x,y|I)$，即Anchor只应当在特定的位置出现。并且每个位置上Anchor的宽高$w,h$应该和它的位置有关系。

基于这一猜想，论文提出了Guided Anchoring。整体网络结构如Figure1所示。

![论文的整体网络结构](https://img-blog.csdnimg.cn/20200326170353311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到这个网络整体上应用了FPN，然后在每层输出的特征图上用蓝色框部分的Guided Anchoring之后，获得了Anchor和新的特征图，再进一步得到检测结果。

而在Guided Anchoring结构中又分为两个分支，一个分支用来预测Anchor的位置和大小，另外一个分支用于产生新的用于检测目标的特征图。在生成Anchor的分支中又分为了两个子分支，即特征$F_1$经过两个$1\times 1$卷积分别得到$w\times h\times 1$和$w\times h\times 2$的特征图来预测Anchor的位置以及宽高。同时论文注意到因为不同位置的Anchor大小是不同的，所以对应到特征图$F_1$上的范围也应该有差别，因此作者通过一个Feature Adaptation模块，将Anchor的形状信息融合到原来的特征图$F_1$中，得到新的特征图$F_1^{'}$用于最后的检测。


 # 3. Anchor位置预测

 Anchor位置的预测部分是想获得一个和$F_1$特征图尺寸一样的概率图$p(.|F_1|)$，并且在训练的时候Anchor应该尽可能的和GT的中心重合来获得更大的IOU，因此作者将每个GT划分成了三种类型的区域，即物体中心区域，外围区域和忽略区域。大概思路就是将GT框的中心一小块对应在特征图$F_1$上的区域标记为物体中心区域，在训练的时候作为正样本，其余区域按照离中心的距离被标记为忽略或者负样本。通过位置预测，就可以筛选出一小部分区域作为Anchor的候选中心点位置，使得Anchor的数量大大降低。在前向推理的时候，可以采用masked conv代替普通的卷积，只在有Anchor的地方进行计算，可以加速推理速度。

![定义物体中心区域，外围区域和忽略区域](https://img-blog.csdnimg.cn/20200326174111221.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

另外值得一提的是在训练位置预测分支的时候，使用了Focal Loss。因为，位置预测分支实际上是一个二分类问题，即预测哪些像素应该是中心点来生产Anchor。

# 4. Anchor形状预测
Anchor的形状(宽高比)预测分支是给定Anchor中心点，预测最好的长宽，这是一个回归问题。一个简单的想法是如果我们要预测Anchor的宽高，我们首先应该算出和当前GT框最匹配的宽高，然后可以通过Smooth L1/L2这种损失来监督网络学习Anchor的宽和高就好了。但是这个和当前Anchor最匹配的宽和高是很难计算的，因此论文的想法是通过IOU来监督网络学习使得IOU最大的Anchor的宽和高，同时因为IOU是可导的，所以可以直接参与到反向传播中。一个Anchor和GT框的IOU可以表示为vIOU，计算公式如下：

![Anchor和GT的IOU计算](https://img-blog.csdnimg.cn/20200326180019803.png)

但是这里还有一个问题，那就是之前我们固定了Anchor的大小，可以用和GT最大IOU的那个Anchor来匹配就可以了。但是现在Anchor的长宽是未知的，我们又不能穷举所有的可能的$w$和$h$来计算和GT框最大IOU的Anchor，因此这里采用了一种近似的方式，即采样几种可能的$w$和$h$来估计vIOU。论文采样了$9$对$w$和$h$来估计vIOU，并且经过实验证明网络对于具体要采样多少对$w,h$这个超参数不敏感。注意一下下面公式(6)的Loss叫作Bounded IOU Loss，是为了减少计算量提出的，这和IOU损失原理一样。

最后，形状预测的损失函数表示如下：

![形状预测的损失函数](https://img-blog.csdnimg.cn/20200326180639996.png)

# 5. 生成Anchor
在获得Anchor的位置和宽高比后我们就可以生成Anchor了，如Figure4所示。可以看到这时候的Anchor是稀疏的并且每个位置都不一样。

![生成Anchor](https://img-blog.csdnimg.cn/2020032620172695.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

使用生成的Anchor代替常规的划窗方式，AR(平均召回率)相比普通的RPN可以有4个点的提升，而代价仅仅是增加了2个$1\times 1$卷积。

# 6. Feature Adaption模块
上面提到过，这个模块主要是将Anchor的形状信息融合到特征中。我们知道RPN中的Anchor使用了划窗的方式均匀分布在整张图像中，因为Anchor是均匀的，所以说每个Anchor都可以用同样感受野大小的特征图来做预测。但是应用了Guided Anchoring之后，Anchor的大小是可变的，因此每个Anchor对应的位置的感受野大小应当有所区别才能体现出Anchor大小不同的优势。

为了做到这一点，作者使用了$3\times 3$的可变形卷积$N_T$，并且为了融合每个Anchor的形状信息，将$w\times h\times 2$经过$1\times 1$卷积得到$N_T$的偏移量。这样做的好处是特征的有效范围和Anchor形状更加接近，不同的位置可以应用不同形状的Anchor了。

下面的Table4展示了Feature Adaption可以提升4个点的AR，还是非常可观的。

![消融实验](https://img-blog.csdnimg.cn/20200326202814248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 7. 高质量候选框的正确打开方式
作者发现通过Guided Anchoring的方式可以得到很多高质量的候选区域，但是给检测器带来的提升效果却十分有限（Faster-RCNN提升1个点左右）。后面作者发现通过Guided Anchoring方式得到的候选框有两个特点：
- 候选框中正样本的比例比传统算法更高。
- 候选框的IOU普遍变得更大了。

为了可以应用好这些高质量的候选框，作者提出了以下策略：

- 减少RPN产生的候选框的数量。
- 提高IOU阈值。

最后使用这种方式，在Faster R-CNN上获得了2.7%的mAP值提升。细节请看Table7。

![高质量候选框的正确打开方式](https://img-blog.csdnimg.cn/20200326203301348.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 8. 总结
这篇文章还是非常有意思的，作者在讨论区提到这种方法在实际应用中的提升相比你在自己数据集上进行聚类Anchor来得更加Solid。并且作者在mmdection已经实现了这一方法，开箱即用。

# 9. 参考
- https://blog.csdn.net/watermelon1123/article/details/89847184
- https://zhuanlan.zhihu.com/p/55854246
- 论文原文：https://arxiv.org/pdf/1901.03278.pdf

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)