# 前言
关于YOLOv3可以看一下我前面的推文讲解：[YOLOV3点这里](https://mp.weixin.qq.com/s/Rgw4mgwaYpQ00WAI26uiiw) 。前面提到过，YOLOv3在实时性和精确性都是做的比较好的，并在工业界被广泛应用。当前的目标检测算法，大都在网络结构，Anchor，IOU上做了大量的文章，而在检测框的可靠性上做文章的却非常少。所以，Gaussian YOLOv3它来了。论文地址为：[点这里](https://arxiv.org/abs/1904.04620) 。并且作者也开放了源码，地址为：[点这里](https://github.com/jwchoi384/Gaussian_YOLOv3)。所以本文就带大家来学习一下这个算法。
# 算法原理
## YOLOv3回顾

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207201619999.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

对于左图，就是YOLOv3的网络结构。可以看到YOLO V3整体使用了特征金字塔的结构，使得网络在3个尺度上执行目标检测任务，可以适应各种不同大小的目标。并且使用了跳跃连接skip shortcut防止因为网络过深而发生梯度消失，无法收敛。YOLOV3还使用了上采样操作，并将大特征图和小特征图上采样的特征图进行concat，使得网络既可以包含高层特征的高级语义信息又可以保留低层特征的物体位置信息，对目标检测任务起到促进作用。

而右图也就是YOLOv3中的输出层，可以看到YOLOv3会在三个特征层分别输出，输出信息为目标的坐标位置，目标是前景的置信度，目标属于某个特定类别的置信度。对于对于每个尺度分支而言，在每个grid cell中会预测出三个结果（每个尺度下会有三个anchor）。将三个尺度的结果合并，进行非极大值抑制（NMS）后，输出最终的检测结果。

## YOLOv3可能存在的问题？
从上面的回顾中可以看到，YOLOV3的目标类别是一个概率值来评价的，而目标的框只有位置信息$(x,y,w,h)$却没有概率值，也就是说我们无法知道当前目标框的可靠性。这就是YOLOv3存在的问题，我们无法评价目标框的可靠性。所以这篇论文以这位切入点提出了Gaussian YOLOv3.即利用Guassian模型对网络输出进行建模，在基本不改变YOLOv3结构和计算量的情况下，能够输出每个预测框的可靠性，并且在算法总体性能上提升了3个点的MAP。

## Gaussian YOLOv3
将原始的YOLOv3的目标框输出加入高斯模型后，网络的输出变成了下图这样。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207221330873.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

原始的YOLOv3对于三个不同尺度的feature map，每个点输出$3\times ((t_x, t_y, t_w, t_h) + obj_score + class_score)$个信息，其中$class_score$包含了类别的可靠性信息，$obj_score$包含了是否是目标的可靠性信息，而关于边界框，我们只有关于坐标的相关信息$t_x,t_y,t_w,t_h$，但这些坐标信息并不能表示该bbox的可靠性。基于此，论文提出了将高斯模型用到bbox的坐标预测上，通过高斯模型的标准差来估计坐标信息的可靠性。加入了高斯模型后，bbox的输出变成了：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208141229743.png)其中$\mu$代表均值，$\sum$表示标准差（方差），考虑到YOLOv3的一贯做法，我们需要将上面的参数做以下变换：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208141519710.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

通过这一系列等式(2),(3),(4)得到的$u_{tx},\mu_{ty},\mu_{tw},\mu_{th}$可以直接当做计算bbox回归的坐标来用（因为$\mu_{tx}$就是$tx$的最大似然估计）。

注意，这里对$\hat{u_{tx}}，\hat{u_{ty}}$做了sigmoid操作将值限定在(0,1)范围是YOLOv3的直接坐标预测做法，即是说每个grid检测的物体中心必须落在当前grid内。这一点可以看我之前的推文解释：[点这里](https://mp.weixin.qq.com/s/UHSHdGL3GucmUozglmZESw)。而$\hat{u_{tw}}$和$\hat{u_th}$没有做sigmoid操作的原因是因为长宽的尺度变化可能操作1，这个做法和YOLOv3一致。标准差$\hat{\delta_{tx}}$,$\hat{\delta_{ty}}$,$\hat{\delta_{tw}}$,$\hat{\delta_{th}}$也通过sigmoid将值限定在（0,1）范围内，这是因为标准差表明了点坐标的可靠性，0表示非常可靠，1表示不可靠（因为对于高斯分布，方差越大，则一定程度上说明这个分布的变化比较大，即是$\mu_{tx}$这个bbox的估计结果越不可靠）。

做了上面的铺垫后就可以引出论文的损失函数了。现在对于网络输出的每个bbox坐标都满足均值为$\mu$，方差为$\sigma$的高斯，因此论文中使用了NLL_LOSS，即是negative log likelihood loss。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208142346276.png)

其中：![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208142558205.png)这个式子表示了对于bbox的ground truth框的每一个值在均值为$u_{tx}$和房差为$\sigma_{tx}$的高斯分布下的值$x_{ijk}^G$，其中高斯分布的密度函数是：
$f(x)=\frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})$。其中每个$x$对应的输出值$f(x)$是在该点的概率密度值，这个值可以近似表示概率，但是它实际上不是概率（其实概率密度函数是概率分布函数的导数）。所以我们的目标是希望ground truth 的概率密度函数值在网络的所有输出值的均值$\mu_{tx}$和方差$\delta_{tx}$构成的高斯分布中是最大的，即表明网络输出的高斯分布和真实标签是最相符的。那么当$\mu_{tx}$和$\sigma_{tx}$构成的分布和真实的lable分布越接近时，$N$就会越大，$log(N...)$也会越大，那么前面取负号，整个损失就会越小。
损失函数还有一个权重惩罚系数$\gamma_{ijk}$，计算公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208145122568.png)

其中$w_{scale}$使用ground truth的长宽来计算的。然后$\delta_{ijk}^{obj}$表示当ground truth和当前的Anchor的IOU大于一定阈值时取1，如果没有合适的Anchor和Ground Truth就取0。

注意上面只是针对了bbox的x坐标进行了讲解，其他的一样类推即可。还有在目标检测预测阶段，bbox的每一类特定的置信度就要用下面的公式来计算了：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208145020680.png)

相比原来的YOLO系列，预测部分增加了一个$Uncertainty_{aver}$坐标可靠性系数，这个系数是bbox的四个标准差$\sigma_{tx},\sigma_{ty},\sigma_{tw},\sigma_{th}$的平均值，这也是前面为什么对标准差做了sigmoid操作的原因。

# 实验结果
在KITTI和BDD上验证结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191208150425390.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到Gaussian YOLOv3比YOLOv3的效果提升了很多，并且加入了Gaussian之后的YOLO效果在同等速度下达到了最优。并且文中提到Gaussian YOLOv3有效的减少约%40的False Positive，提高了约5%的True Positive。

# 代码推荐
建议跑AlexAB版本的darknet里面的GaussianYOLOv3，非常简单和准确。地址如下：[点这里](https://github.com/AlexeyAB/darknet/blob/master/cfg/Gaussian_yolov3_BDD.cfg)
# 后记
今天介绍了Gaussian YOLOv3，个人觉得这篇论文的思想是值得点赞的。从bbox的可靠性做文章有可能会是目标检测算法优化的又一重要方向。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS01M2E3NWVmOTQ2YjA0OTE3LnBuZw?x-oss-process=image/format,png)