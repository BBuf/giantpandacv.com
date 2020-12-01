# 前言
这篇论文仍然是瞄准了One-Stage目标检测算法中的正负样本不均衡问题，上周我们介绍He Kaiming等人提出的Focal Loss，推文地址如下：https://mp.weixin.qq.com/s/2VZ_RC0iDvL-UcToEi93og 来解决样本不均衡的问题。但这篇论文提出，Focal Loss实际上是有问题的，论文论述了该问题并提出了GHM Loss更好的解决One-Stage目标检测算法中的正负样本不均衡问题。论文地址为：https://arxiv.org/pdf/1811.05181.pdf。github开源地址为：https://github.com/libuyu/GHM_Detection
# 梯度均衡机制(GHM)
首先论文引入了一个统计对象：梯度模长(gradient norm)。考虑一个简单的二元交叉熵函数(binar cross entropy loss)：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207122920785.png)

其中$p=sigmoid(x)$是模型预测的样本的类别概率，而$p^{*}$是标签信息，这样可以球处于其对$x$的梯度：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207123936995.png)

所以，论文定义了一个梯度模长为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120712401455.png)

直观来看，$g$表示了样本的真实值和预测值的距离。看下论文的Figure2，表示的是一个One-satge模型收敛后画出的梯度模长分布图。Figure2如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207130858369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

上图横坐标表示gradient norm，纵坐标表示数据分布的比例，做了对数尺度缩放，显然非常靠近y轴表示的是easy examples，非常靠近$x=1$轴的表示very hard examples，中间部分的表示hard example。 重新标注下上图即：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207131239636.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

注意到途中绿框部分即为very hard example，论文认为这部分样本对模型的提升作用是没有帮助的，但这部分样本同样也有着较大的比例。根据Focal Loss的观点，我们班应该关注的是介于红框和绿框之间的样本，而Focal Loss缺不能解决这一点，而是把very hard examples也考虑在内了。这些very hard examples也被称为离群点，也就是说如果一个模型强行拟合了离群点，模型的泛化能力会变差，所以这篇论文提出了GHM Loss抑制离群点，取得了比Focal Loss更好的效果。

基于上面的分析，论文提出了**梯度均衡机制**(GHM)，即根据样本梯度模长分布的比例，进行一个相应的标准化(normalization)，使得各种类型的样本对模型参数的更新有更加均衡的贡献，进行让模型训练更高效可靠。由于梯度均衡本质上是对不同样本产生的梯度进行一个加权，进而改变它们的贡献量，而这个权重加在损失函数上也可以达到同样的效果，此研究中，梯度均衡机制便是通过重构损失函数来实现的。为了更加清楚的描述新的损失函数，论文定义了**梯度密度**(gradient density)这一概念。仿照物理上对于密度的定义（单位体积内的质量），论文把梯度密度定义为单位取值区域内分布的样本数量。
首先定义**梯度密度函数**（Gradient density function）

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120713215367.png)

其中$g_k$表示第$k$个样本的梯度，而且：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207132416582.png)

所以梯度密度函数$GD(g)$就表示梯度落在区域$[g-\frac{\epsilon}{2}，g+\frac{\epsilon}{2}]$的样本数量。再定义度密度协调参数$\beta$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207132851979.png)

其中$N$代表样本数量，是为了保证均匀分布或只划分一个单位区域时，该权值为 1，即 loss 不变。
综上，我们可以看出**梯度密度大的样本的权重会被降低，密度小的样本的权重会增加**。于是把GHM的思想应用于分别应用于分类和回归上就形成了GHM-C和GHM-R。

# 用于分类的GHM Loss

把GHM应用于分类的loss上即为GHM-C，定义如下所示:

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207133117739.png)

根据GHM-C的计算公式可以看出，候选样本的中的简单负样本和非常困难的异常样本(离群点)的权重都会被降低，即loss会被降低，对于模型训练的影响也会被大大减少，正常困难样本的权重得到提升，这样模型就会更加专注于那些更有效的正常困难样本，以提升模型的性能。GHM-C loss对模型梯度的修正效果如下图所示，横轴表示原始的**梯度loss**，纵轴表示修正后的。由于样本的极度不均衡，论文中所有的图纵坐标都是取对数画的图。注意这是Loss曲线，和上面的梯度模长曲线要加以区别。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207133617427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

从上图可以看出，GHM-C和Focal Loss（FL）都对easy example做了很好的抑制，而GHM-C和Focal Loss在对very hard examples上有更好的抑制效果。同时因为原始定义的梯度密度函数计算计算太复杂，所以论文给出了一个**梯度密度函数简化版本**：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120713434939.png)

其中$ind(g)=t，(t-1)\epsilon<=g<=t\epsilon$。然后再结合密度协调函数$\hat{\beta}$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207134934978.png)

得到$\hat{L_{GHM-C}}$为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207135052194.png)

# 用于回归的GHM Loss
GHM的思想同样适用于Anchor的坐标回归。坐标回归的loss常用Smooth L1 Loss，如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207135420416.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207135442186.png)

其中，$t=(t_x,t_y,t_w,t_h)$表示模型的预测坐标偏移值，$t^*=(t_x^*,t_y^*,t_w^*,w_h^*)$表示anchor相当于Ground Truth的实际坐标偏移量，$\delta$表示$SL_1$函数的分界点，常取$\frac{1}{9}$。定义$d=t_i-t_i^{*}$，则$SL_1$的梯度为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207140539827.png)

其中$sgn$表示符号函数，当$|d|>=\delta$时，所有样本梯度绝对值都为1，这使我们无法通过梯度来区分样本，同时d理论上可以到无穷大，这也使我们无法根据梯度来估计一些example输出贡献度。基于此观察，论文对Smooth L1损失函数做了修正得到$ASL_1$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207141028307.png)

在论文中，$\mu=0.02$。
$ASL_1$和Smooh L1损失有相似的性质，并且梯度为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207141212453.png)

论文把$|\frac{d}{\sqrt{d^2+\mu^2}}|$定义为梯度模长（gradient norm），则$ASL_1$的梯度模长和样本部分的关系如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207141456565.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

由于坐标回归都是正样本，所以简单样本的数量相对并不是很多。而且不同于简单负样本的分类对模型起反作用，简单正样本的回归梯度对模型十分重要。但是同样也可以看出来，存在相当数量的异常样本的回归梯度值很大。（图上最靠右的部分）。所以使用GHM的思想来修正loss函数，可以得到：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207141620521.png)

以达到对离群点的抑制作用。
GHM-R Loss对于回归梯度的修正效果如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207142627953.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到，GHM-R loss加大了简单样本和正常困难样本的权重，大大降低了异常样本的权重，使模型的训练更加合理。

# 实验结论
因为GHM-C和GHM-R是定义的损失函数，因此可以非常方便的嵌入到很多目标检测方法中，作者以focal loss（大概是以RetinaNet作为baseline），对交叉熵，focal loss和GHM-C做了对比，发现GHM-C在focal loss 的基础上在AP上提升了0.2个百分点。如表4所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207141911624.png)

如果再用GHM-R代替双阶段检测器中的Smooth L1损失，那么AP值又会有提示。如表7所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207142029828.png)

如果同时把GHM-R Loss和GHM-C Loss用到目标检测器中，AP值有1-2个点提升。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207142450373.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 后记
论文的大概思想就是这样，对于样本有不均衡的场景，我认为这个Loss是比较值得尝试的。

# 参考文章
https://blog.csdn.net/watermelon1123/article/details/89362220
https://zhuanlan.zhihu.com/p/80594704

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)