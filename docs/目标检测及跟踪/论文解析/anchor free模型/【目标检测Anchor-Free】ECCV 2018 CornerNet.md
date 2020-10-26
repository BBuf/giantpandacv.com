# 前言
继续来探索Anchor-Free目标检测算法，前面讲了Anchor-Free的起源 [目标检测算法之Anchor Free的起源：CVPR 2015 DenseBox](https://mp.weixin.qq.com/s/gYq7IFDiWrLDjP6219U6xA) ，其实同期另外一个有名的目标检测算法YOLOV1也是Anchor-Free系列的了。Anchor-Free系列相比于Anchor-Based的发展是较慢的，在2018-2019年才开始火起来。今天为大家介绍一下ECCV 2018的CornerNet，全称为：Detecting Objects as Paired Keypoints 。论文原文和代码见附录链接。

# 贡献
- 提出通过检测bbox的一对角点来检测出目标。
- 提出Corner Pooling，来更好的定位bbox的角点。

# 介绍
Anchor-Based的目标检测算法相信看了公众号目标检测栏目的话已经了解了很多算法了，这些算法都是基于成千上万个Anchor框来进行预测，无论是one-satge还是two-stage的Anchor-Based目标检测算法，都取得了巨大的成功。但是Anchor-Based目标检测算法有两个缺点，其一是Anchor boxes的数量需要非常大，如在DSSD算法中超过了40000，在RetinaNet中超过了100000，当然最后只有一小部分anchor boxes会和GT框重合，这就带来了严重的正负样本不均衡问题，针对这一问题也有很多方法被提出如OHEM，Focal Loss, GHM Loss等等。而另外一个缺点是Anchor-Based的算法引入了很多超参数和设计选择，如Anchor的个数，宽高比，大小等等，当于多尺度结合的时候网络会变得更加复杂。

因此，针对以上缺点，这篇论文提出了CornerNet，这是一种新的one-stage的Anchor-Free目标检测算法。论文将对一个目标的检测看成一对关键点（左上和右下）的检测。具体来说，论文使用单个卷积神经网络来预测同一物体类别的所有实例的左上角的热力图，所有右下角的热力图，以及每个检测到的角点的嵌入向量。嵌入向量用于对属于同一目标的一对角点进行分组，本方法极大的简化了网络的输出，并且不需要设计Anchor boxes，Figure1展示了本方法的整体流程：


![我们将一个目标检测为一对组合在一起的边界框角点。 卷积网络输出一个左上角热图和一个右下角热图，并输出每个检测到的角点的嵌入矢量。 训练网络以预测属于同一目标的角点的类似嵌入。](https://img-blog.csdnimg.cn/2020012113540051.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

而CornerNet的另外一个创新点是Corner Pooling，这是一种新型的池化层，可以帮助卷积神经网络更好的定位边界框的角点。如Figure2所示，目标边界框的一角通常是在目标之外。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121141859159.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

因此，以top-left角点为例，对每个通道，分别提取特征图对应位置处水平和垂直方法的最大值，然后求和。如Figure3所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121142022470.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# CornerNet
## 概述
Figure 4展示了CornerNet网络的大致结构，论文使用Hourglass（沙漏）网络作为CornerNet的Backbone网络。沙漏网络之后是两个预测模块，一个模块用于预测左上角，一个模块用于预测右下角。每个模块都有自己的Corner Pooling模块，在预测热力图、嵌入和偏移之前，池化来自沙漏网络的特征。和其它的目标检测器不同，论文不使用不同尺度的特征来检测不同大小的目标，只将两个模块用于沙漏网络的输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121142449387.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
## 检测角点
我们预测两组热力图，一组用于左上角角点，一组用于右下角角点。每一组热力图有$C$个通道，其中$C$是类别数（不包括背景），并且大小为$H\times W$。每个通道都是一个二进制掩码，用于表示该类的角点位置。对于每个角点，有一个ground-truth正位置，其他所有位置都是负位置。在训练期间，我们没有同等地惩罚负位置，而是减少对正位置半径内的负位置给予的惩罚。这是因为如果一对假角点检测器靠近它们各自的ground-truth位置，它仍然可以产生一个与ground-truth充分重叠的边界框，如Figure5所示。我们通过确保半径内的一堆点生成的边界框和ground-truth边界框的$IOU>=t$（在所有实验中把$t$设置为$0.7$）来确定物体的大小，从而确定半径。给定半径，惩罚的减少量由非标准化的2维高斯分布$e^{-\frac{x^2+y^2}{2\sigma^2}}$给出，其中心位于正位置，$\sigma$是半径的$1/3$。

![图5.用于训练的“Ground-truth”热图。在正位置半径范围内（橙色圆圈）的方框（绿色虚线矩形）仍然与地ground-truth（红色实心矩形）有很大的重叠。](https://img-blog.csdnimg.cn/2020012115193757.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

$p_{cij}$为预测图中$c$类位置$(i,j)$的得分，$y_{cij}$为用非标准化高斯分布增强的ground-truth热力图。论文设计了一个Focal Loss的变体损失：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121153124864.png)

其中$N$表示图像中目标的数量，$\alpha$和$\beta$控制每个像素点贡献的超参数（在所有实验中将$\alpha$设为2，$\beta$设为4）。利用$y_{cij}$中编码的高斯凸点,$(1-y_{cij})$这一项减少了ground-truth像素点周围的惩罚权重。同时，图像中的位置$(x,y)$被映射到热力图中的位置为$([\frac{x}{n}],[\frac{y}{n}])$，其中$n$表示下采样因子。当我们将热力图中的位置重新映射回输入图像时，可能会存在像素偏移，这会极大影响小边界框和ground-truth之间的IOU值。为了解决这个问题，论文提出预测位置偏移，以稍微调整角点位置，然后再将它们映射回输入分辨率，如公式(2)所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121155352413.png)

其中$o_k$表示偏移量，$x_k$和$y_k$是角点$k$的$x$和$y$坐标。特别地，我们预测所有类别的左上角共享一组偏移，另一组由右下角共享。对于训练，我们在ground-truth角点位置应用Smooth L1 Loss：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121155820608.png)

## 分组角点
图像中可能出现多个目标，因此可能会检测到多个左上角和右下角。我们需要确定左上角和右下角的一对角点是否来自同一目标边界框。论文受到Newell等人提出的用于多人姿态估计任务的关联嵌入方法的启发，Newell等人检测人的关节点，并为每个检测到的关节生成嵌入向量。他们根据嵌入向量之间的距离将节点分组，关联嵌入的思想也适用于我们的任务。 网络预测每个检测到的角点的嵌入向量，使得如果左上角和右下角属于同一个边界框，则它们的嵌入之间的距离应该小。 然后，我们可以根据左上角和右下角嵌入之间的距离对角点进行分组。 嵌入的实际值并不重要，我们仅使用嵌入之间的距离来对角点进行分组。论文和Newell他们一样使用1维嵌入，$e_{tk}$表示角点$k$的左上的嵌入，$e_{bk}$表示右下的嵌入。我们使用"pull"损失来训练网络对角点进行分组，并且用"push"损失来分离角点。如公式(4)和(5)所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121160921952.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中$e_k$是$e_{tk}$和$e_{bk}$的平均值，我们在所有的实验中将$\Delta$设置为$1$。和损失偏移类似，仅仅在ground-truth角点位置应用这个损失函数。

## Corner Pooling
如Figure2 所示，通常没有局部视觉证据表明存在角点。要确定像素是否为左上角，我们需要水平地向右看目标的最上面边界，垂直地向底部看物体的最左边边界。因此，我们提出Corner Pooling通过编码显式先验知识来更好地定位角点。

假设我们要确定位置$(i,j)$的像素是不是左上角角点，设$f_{tij}$和$f_{lij}$分别为$(i,j)$位置中$f_t$和$f_l$的特征向量。对于$H\times W$的特征图，Corner Pooling时层首先最大池化$f_t$中在$(i,j)$和$(i,H)$之间所有的特征值，使之成为特征向量$t_{ij}$。另外，最大池化$f_l$中在$(i,j)$和$(W,j)$之间的所有特征值，使之成为特征向量$l_{ij}$。最后把$t_{ij}$和$f_{ij}$加在一起，如公式(6)和(7)所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121163432124.png)

在这里，我们应用了一个elementwise最大化操作。动态规划可以有效地计算$t_{ij}$和$l_{ij}$，如Figure6所示：

![图6。左上角的池化层可以非常有效地实现。我们从右到左扫描水平最大池化，从下到上扫描垂直最大池化。然后我们相加两个经过最大池化的特征映射。](https://img-blog.csdnimg.cn/20200121164414410.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

我们以类似的方式定义右下角池化层。最大池化$(0,j)$与$(i,j)$之间所有得特征值，$(i,0)$和$(i,j)$之间的所有特征值，然后将池化结果相加。Corner Pooling层用于预测模块来预测热力图，嵌入和偏移量。

![图7。预测模块从一个修改后的残块开始，其中我们将第一个卷积模块替换为corner pooling模块。修改后的残差块后面跟着一个卷积模块。我们有多个分支用于预测热图、嵌入和偏移量。](https://img-blog.csdnimg.cn/20200121165052244.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

预测模块的第一部分是残差模块的修改版，修改后的残差模块中，将第一个$3\times 3$卷积替换为一个Corner Pooling模块。这个残差模块首先通过具有$128$个通道的$2$个$3\times 3$卷积模块的主干网络进行特征提取，然后应用一个Cornet Pooling层。残差模块之后，我们将池化特征输入到具有$256$个通道的$3\times 3$的Conv+BN层，同时为这一层加上瓶颈结构。修改后的残差模块后面接一个具有$256$个通道的$3\times 3$卷积模块和$256$个通道的$3$个Conv-ReLU-Conv来产生热力图，嵌入和偏移量。



## 沙漏网络
CornerNet的Backbone使用Hourglass网络，这个网络最早在人体姿态估计任务中被引入。Hourglass网络是全卷积网络，包含一个或者多个Hourglass模块。输入图片的尺寸是$511\times 511\times 3$，经过一个`conv(7x7-c128-s2-p3)`的卷积模块，以及一个`residual`（其中卷积层的核为`3x3-cl256-s2-p1`），因此这个时候特征图大小就变成了$128\times 128\times 256$，这个特征图作为Hourglass网络的输入。

而在Hourglass网络中首先使用一些卷积核池化层对输入特征进行下采样（注意论文指出这里没有使用池化层来做下采样，而是只使用步长为$2$卷积层），然后再上采样使得特征图的分辨率回到原始大小，由于Max-Pooling层会丢失细节信息，所以增加跳跃连接层将低级特征信息带到上采样特征图中，因此hourglass不仅仅结合了局部特征还结合了全局特征，当堆叠多个hourglass模块时就可以重复这个过程，从而补货更加高级的特征。

Hourglass网络结构图如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121175219706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 实验
## 训练细节
论文在Pytorch上实现了CornerNet，代码地址见附录。网络是在默认的Pytorch设置下随机初始化的，没有在任何外部数据集上预训练，另外因为使用了Focal Loss，所以应该按照RetinaNet论文中指出的方式来对某些卷积层的偏置进行初始化。在训练时，设置了网络的输入分辨率$511\times 511$，所以输出分辨率为$128\times 128$。为了减少过拟合，论文使用了标准的数据增强技术，包括随机水平翻转、随机缩放、随机裁剪和随机色彩抖动，其中包括调整图像的亮度，饱和度和对比度。 使用Adam来优化完整的训练损失：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121182432138.png)

其中$\alpha，\beta，\gamma$分别是pull，push和offset的权重。 我们将$\alpha$和$\beta$都设置为$0.1$，将$\gamma$设置为$1$。`batch_size`设置成`49`，并在10个Titan X（PASCAL）GPU上训练网络（主GPU 4个图像，其余GPU每个GPU 5个图像）。为了节省GPU资源，在论文的 ablation experiments（即模型简化测试，去掉该结构的网络与加上该结构的网络所得到的结果进行对比）中，我们训练网络，进行`250k`次迭代，初始学习率为$2.5\times 10^{-4}$。当将论文的结果和其他检测器比较时，论文额外训练网络，进行`250k`迭代，并到最后`50k`次迭代时，将学习率降为$2.5\times 10^{-5}$。
## 测试细节
测试时主要有3个步骤：
- 如何判断某个位置是角点？首先执行nms，对得到的两组热力图（注意热力图就是某个特定位置属于某个类别角点的概率）应用`3x3`大小并且`stride=1,pad=1`的`maxpooling`，不改变特征图大小，保留值保持不变，值改变了的则全部置为0。然后选择top N，这个操作是在所有分类下（分类不独立）进行，选择top N的时候同时保持这些角点的对应分类。
- 左上角点和右下角点如何配对？参考分组角点那一节的分析，用嵌入距离来判断。
- 再次选择top K的角点对，并微调坐标位置，得到最后的结果。

## MSCOCO
论文在MSCOCO数据集上评测了这个算法。MS COCO包含80k图像用于训练，40k图像用于验证，20k图像用于测试。 训练集中的所有图像和验证集中的35k图像用于训练。 验证集中剩余的5k图像用于超参数搜索和ablation study。 测试集上的所有结果都将提交给外部服务器进行评估。 为了与其他检测器进行公平比较，我们在test-dev集上记录了我们的主要结果。 MS COCO在不同IoU上使用平均精度(AP)、在不同物体尺寸上使用AP作为主要评估指标。

## 消融研究
- Corner Pooling。Corner Pooling是CornerNet的关键组成部分。 为了理解它对性能的贡献，训练了另一个具有相同数量参数但没有corner Pooling的网络。可以看到对中型和大型目标特别有用，它们的AP分别提高了2.4％和3.7％。 这是预料中的，因为中型和大型目标的最顶部，最底部，最左边，最右边得边界可能更远离角点位置。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121184328995.png)

- 减少对负位置的惩罚。（减少对正位置周围目标半径范围内的负位置给出的惩罚）即上面的分组角点，为了探索这个操作对结果的影响，训练一个没有减少惩罚的网络和另一个固定半径为2.5的网络。 我们在验证集上将它们与CornerNet进行比较。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121184552510.png)

可以看到，一个固定半径相比于BaseLine模型$AP$值提升了2.7%，$AP^m$提升了1.5%，$AP^l$提升了5.3%。而基于物体半径可以进一步将$AP$提高2.9%，$AP^m$增加2.6％，$AP^l$增加6.5％。此外，我们看到减少惩罚特别有利于大中型目标。

- **误差分析。** 这一节是比较有趣的，上面都在说这个算法的优点和原理，这个实验来分析这个算法的瓶颈。CornerNet同时输出热图，偏移和嵌入，所有这些都会影响检测性能。 如果错过任何一个角，都将会丢失一个目标；需要精确的偏移来生成紧密的边界框。不正确的嵌入将导致许多错误的边界框。 为了理解每个部件如何影响最终误差，我们通过将预测的热图和偏移替换为ground-truth，并在验证集上评估性能，以此来执行误差分析。实验结果如Table6所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121185528820.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到，单独使用ground-truth的热力图可以将AP从38.5％提高到74.0％。而$AP$,$AP^m$,$AP^l$分别增长43.1％，40.9％和30.1％。 如果我们用ground-truth偏移代替预测的偏移量，则AP进一步从74.0%增加到了87.1%，这表明CornerNet的主要瓶颈是检测角点。下图展示一些错误的例子。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121190110202.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 和最先进的目标检测器相比
在MS COCO testdev上，将CornerNet与其他最先进的检测器进行比较，结果如Table7所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020012118590345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

通过多尺度评估，CornerNet实现了42.1％的AP值，精度是one-stage算法中的SOTA。下面看一些可视化例子。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200121190025294.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论
这篇论文提出了CornerNet，这是一种新的目标检测方法，可以将边界框检测为成对的角点。 并且在MS COCO上对CornerNet进行评估，并展示出了SOTA结果。

# 附录
- 论文原文：https://arxiv.org/pdf/1808.01244.pdf
- 参考1：https://blog.csdn.net/weixin_40414267/article/details/82379793
- 参考2：https://zhuanlan.zhihu.com/p/66406815
- 代码实现：https://github.com/princeton-vl/CornerNet


# 同期文章
[目标检测算法之Anchor Free的起源：CVPR 2015 DenseBox](https://mp.weixin.qq.com/s/gYq7IFDiWrLDjP6219U6xA)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)