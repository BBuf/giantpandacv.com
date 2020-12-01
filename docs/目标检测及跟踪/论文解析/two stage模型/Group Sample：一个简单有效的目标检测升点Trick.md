# 1. 前言
今天为大家介绍一个CVPR 2019提出的一个有趣的用于人脸检测的算法，这个算法也可以推广到通用目标检测中，它和OHEM，Focal Loss有异曲同工之妙。论文地址为：`http://openaccess.thecvf.com/content_CVPR_2019/papers/Ming_Group_Sampling_for_Scale_Invariant_Face_Detection_CVPR_2019_paper.pdf`。

# 2. 出发点

这篇文章的出发点是，我们在做人脸检测或者通用目标检测时，都会碰到数据集中目标尺寸差异较大的情况，这种情况对检测带来的最直观的影响就是小目标的检测效果不好。为了缓解这个问题[【CV中的特征金字塔】二，Feature Pyramid Network](https://mp.weixin.qq.com/s/d2TSeKEZPmVy1wlbzp8BNQ) 被提出，并在小目标上的效果提升明显。但是，**FPN为什么有效呢？**

为了解答这个问题，论文做了大量实验。通过实验结果的对比发现，将FPN修改为基于单个特征层进行预测同样可以取得接近多个特征层的效果，而修改前后不同尺寸的样本数量分布非常类似。接着，论文继续探索了不同尺寸的样本数量分布对检测效果的影响，最终得到下面的结论：

**不同尺寸的样本数量不均衡是导致检测模型效果不好的重要原因，这个样本既包括正样本，又包括负样本。**

基于此出发点，论文提出了Group Sample方法使得不同尺寸的样本数量达到均衡，明显提升了目标检测的效果。

# 3. 关键实验

下面的Figure1列出了列出了5个网络，这5个网络是论文的后续实验和论文创新点介绍的核心部分：

![Figure1](https://img-blog.csdnimg.cn/20200723202117162.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

首先，`(a)`和`(b)`分别代表RPN和FPN网络，其中RPN网络是基于单个特征层(**即(a)图中的C4**)进行预测的，而FPN是基于多个特征层进行预测的，也就是`(b)`中的P2到P5，灰色框中的数字如**128/32** ，前面的数字代表Anchor的尺寸，后面的数字表示铺设Anchor时的`stride`，这两个变量都是针对原图来说的。然后，`(c)`和`(b)`的区别主要在于`(c)`的预测层从多个变成1个。而`(d)`和`(c)`的差别在于`(d)`中不同特征层铺设的Anchor的`stride`都全部相同，最后`(e)`是这篇论文提出的Group Sample算法，主要是在`(d)`的基础上增加了Anchor的Group Sample操作。

因此，Figure1中的5种网络主要有以下两个关键差别：

- Anchor铺设的特征层数量不同。
- Anchor铺设时设置的Stride不同。

下面的Table1给出了这两种差别对目标检测模型效果的影响：


![Table1](https://img-blog.csdnimg.cn/20200723204347706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)



然后我们可以从这个表中获得一些结论：

- **从FPN和FPN-finest-stride来看，二者的区别主要是用于预测的特征层数量不同，而Table1中两者的效果却非常接近。** 这个实验作者想证明的是FPN之所以有效，并不是因为预测特征层的数量增多导致的，而是深层和浅层特征的融合。
- 从FPN-finest-stride和FPN-finest来看，它们的主要区别是铺设Anchor的`stride`不同，FPN-finest采用完全相同的`stride`，所以不同尺寸的负样本数量比较均衡，不同尺寸的正样本数量不均衡（类似RPN）。**而FPN-finest-stride中不同尺寸的正负样本数量差异都很大，这个实验目的就是观察这种差异带来的影响有多大。从结果来看，这种差异带来的影响很大。**
- 从FPN-finest-sampleing和FPN-finest-stride来看，这里是为了探索人工干预不同尺寸的样本分布对结果影响有多大，可以看到效果提升是比较明显的。所谓人工干预就是让不同尺寸的正负样本数量分布均衡。

基于这些实验和一些先验知识，作者为每一个实验都给出了一个解释，可以结合Figure2来进行理解：

![关于5种网络结构的不同尺度Anchor占比图](https://img-blog.csdnimg.cn/20200723210946631.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


- **结论1**： **RPN中不同尺寸的负样本比例基本均衡，但是正样本占比随着尺寸的增大而增大，** 原因比较容易理解，RPN在铺设anchor时候，不同尺寸的anchor数量都是相同的（因为stride相同），但是因为大尺寸的anchor更容易成为正样本，因此尺寸越大的正样本占比越大。
- **结论2**：FPN基于多个特征层铺设Anchor，可以看Figure1(b)。P2层铺设尺寸为16，stride为4的anchor；P3层铺设尺寸为32，stride为8的anchor；显然在其他参数都相同的情况下，P2层的anchor数量是P3层anchor数量的4倍，因此可以**看到Figure2(b)中的柱形图基本满足这样的4倍关系**，其他层同理有这种关系。
- **结论三**：FPN-finest和FPN-finest-stride的差别主要在于铺设anchor的stride不同，这就使得它们在不同尺寸的样本分布上存在较大差异。**FPN-finest-stride的不同尺寸特征图的正负样本分布都是不均衡的且负样本的差异更大，另外由于步长的原因，不管正样本还是负样本都是小尺寸的占比更大，因此FPN-finest-stride在小尺寸人脸（hard任务）上的效果要优于FPN-finest。** **FPN-finest的不同尺寸的负样本分布是均衡的，但是不同尺寸的正样本分布是不均衡的，而且正样本中大尺寸的占比要远大于小尺寸，因此FPN-finest在大尺寸人脸（Easy，Midium任务）上的效果要优于FPN-finest-stride。**


基于上面的实验结论，作者猜测不同尺寸的样本数量（包括正负样本）不均衡是导致目标检测效果不佳的重要原因，因此就尝试人工干预不同尺寸的样本分布，使得不同尺寸的正负样本均分布均衡，如`Figure2(e)`所示，并且效果也比之前提升了很多。

# 4. Group Sample的做法

Group Sample的具体做法就是**将正负样本按照尺寸大小分成不同的组，然后随机采样正样本使得每个组的正样本数量相同，同时因为每个组的样本数量是预先设定好的固定值（比如$128、256$），因此不同组的负样本数量也是相同的，这样就完成了人为干预的过程。**

# 5. 和Focal Loss，OHEM对比

Group Sample的做法和OHEM，Focal Loss有异曲同工之妙，但是OHEM和Focal Loss主要解决的是样本不均衡问题，即从图像样本数量的角度出发，通过给较难的样本更大的损失惩罚来实现。

本文的Group Sample虽然也是解决样本不平衡问题，但是这里的不平衡指的是**不同尺寸之间的样本不均衡** ，另一方面这里的样本是候选框(Anchor)层面，而不是图像样本层面的。

# 6. 实验结果
论文中的Table4展示了Group Sample和OHEM、focal loss的效果对比，提升还是很明显的，其他更多实验结果可以看原始论文。


![在同一数据集上，Group Sample和OHEM、focal loss的效果对比，提升还是很明显的](https://img-blog.csdnimg.cn/20200723221334575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure5展示了本文的Group Sample在WIDER FACE上的测试结果，可以看到结果也是非常不错的。

![在WIDER FACE上的表现](https://img-blog.csdnimg.cn/2020072322271616.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

比较遗憾的事，截至到今天(2020/7/23)，这篇论文仍然没有放出源码，感兴趣的读者只能摸索着去复现了。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)