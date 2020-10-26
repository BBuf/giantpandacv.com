# 前言
深度残差金字塔网络是CVPR2017年的一篇文章，由韩国科学技术院的Dongyoon Han, Jiwhan Kim发表，改善了ResNet。其改用加法金字塔来逐步增加维度，还用了零填充直连的恒等映射，网络更宽，准确度更高，超过了DenseNet，泛化能力更强。论文原文见附录。

# 介绍
近年来，深度卷积神经网络在图像分类任务中表现出了卓越的性能。通常，深度神经网络结构是由大量的卷积层堆叠而成，并且使用池化不断的减小图片的分辨率。同时，特征映射维度在下采样的地方急剧增长，这对于确保性能是必要的，因为它增加了高级属性的多样性，这也适用于残差网络，并且与其性能密切相关。在这篇论文中，作者提出并不是网络在执行下采样的单元处急剧增加特征图的尺寸，而是逐渐的增加所有单元的特征尺寸，以尽可能多地涉及位置。我们对这种网络设计进行了深入讨论，证明了其是提高泛化能力的有效手段。此外，论文提供了一种新的残差单元，能够通过使用本文的新网络架构进一步提高分类精度。在CIFAR-10,CIFAR-100和ImageNet数据集的实验证明，和原始的ResNet相比，我们的网络具有更高的精度和泛化能力。
# 网络结构
- **金字塔(瓶颈)残差单元。**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822103233432.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
可以看到相对于传统的残差模块，金字塔残差单元的各个单元的维度逐渐增加，直到出现下采样的剩余单元。
- **深度残差金字塔网络结构。**
深度残差金字塔网络结构如Table1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822103756517.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
其中$\alpha$代表扩展因子，$N_n$代表一个group中有多少个`block`，下采样在`conv3_1`和`conv4_1`处进行，步长为`2`。 
-  **深度残差金字塔网络通道变化公式**
金字塔网络每一层的通道数和网络深度有关，论文提到了2种通道增长方式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110230437753.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110230504670.png)

其中$N=N_2+N_3+N_4$，式子(2)为加法金字塔，式子(3)为乘法金字塔，$\alpha$是超参数扩展因子，$k$是当前网络的层数，控制超参数可使金字塔更宽或更细，但高度不变。两者方式的比较如Figure2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822104757311.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
Figure2(a)为加法的PyramidNet，Figure2(b)为乘法的PyramidNet，Figure2(c)是加法金字塔网络和乘法金字塔网络的对比。加法金字塔网络的特征映射维数呈线性增长，而乘法网络的特征映射维数呈几何急剧增长。乘法金字塔网络中输入端层的维数缓慢增加，输出端层的维数急剧增加，这个过程类似于VGG和ResNet等原始的深度网络架构。

- **加法金字塔网络和乘法金字塔网络的对比。**
对比结果如Figure7所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822105120845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
这个实验指出，当参数个数较少时，加法和乘法金字塔网络的性能基本相同，因为这两种网络架构没有显著的结构差异。而随着参数数量的增加，它们开始在特征图维度配置方面显示出更显著的差异，可以看出加法金字塔表现更好。由于加法的特征映射维数呈线性增长，与乘法相比，**输入附近层**的特征映射维数更大，**输出附近层**的特征映射维数更小。所以增加输入端附近层的模型容量将比使用传统的特征映射维数乘法缩放方法带来更好的性能改进。

# 金字塔网络其他Trick
-  **零填充的直连恒等映射。**
零填充是为了保证金字塔的形状。具体方法如Figure5所示，Figure5 (a)是带有零填充恒等映射的直连残差单元，Figure5(b)是对Figure5(a)的展开表示，它构成了一个直连和普通网络混合的残差网络。Table2表明(b)这种零填充直连恒等映射精度最好。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822105538121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019082211291027.png)
- **BN层和激活层怎么放？**
不是很好解释，论文实验了在去掉某些BN和ReLU后的不同典型结构的精度，如Figure6所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822113543147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
这里解释一下，`(a)`表示原始的预激活ResNets，`(b)`表示去除第一个ReLU的预激活ResNets， `(c)`表示在预激活ResNets的最后一个卷积层之后重新分配一个BN层，`(d)`表示对预激活ResNets去除第一个ReLU，在最后一个卷积层之后重新分配一个BN层。Table3展示了对上诉不同方法的实验结果，使用Figure6中的结构`d`可以提高性能。因此，只要使用适当数量的ReLUs来保证特征空间流形的非线性，就可以去除剩余的ReLUs来提高网络性能。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822113708588.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 实验结果

实验结果如Table4，5所示，在CIFAR100效果很好超过了80%，一般比较好的都在80%左右，而ResNet才不到75%。精度是当之无愧的SOTA。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822113906725.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190822113923173.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 后记
PyramidNet效果很好，但实际用的不是特别多，$\alpha$设的大的时候网络会很宽，而$Pooling$为了缩小尺寸，卷积计算量减少，有利于实用。其实用的不多基本是受限于计算资源，我相信硬件的进一步发展这个网络仍有机会大放异彩。

# 附录
- 论文原文：https://arxiv.org/abs/1610.02915
- 代码实现：https://github.com/jhkim89/PyramidNet-caffe
- https://zhuanlan.zhihu.com/p/68413130

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)