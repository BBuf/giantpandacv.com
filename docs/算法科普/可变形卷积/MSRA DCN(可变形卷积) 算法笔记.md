# 1. 前言
Deformable Convlolutional Networks是ICCV 2017一篇用在检测任务上的论文，它的出发点是在图像任务中目标的尺寸，形状变化不一，虽然目前（指代论文发表以前）的CNN对这些变化有一定的鲁棒性，但并不太够。因此论文以此为切入点，通过在卷积层中插入offset（可变形卷积）和在ROI Pooling层中插入offset（可变形ROI Pooling）来增强网络的特征提取能力。这个offset的作用是使网络在提取特征时更多的把注意力聚焦到和训练目标有关的位置上，可以更好的覆盖不同尺寸和形状的目标，并且offset也是在监督信息的指导下进行学习的，因此不用像数据增强一样需要大量先验知识才可以获得更好的效果，整体结构也保持了end2end。

# 2. 算法原理
## 2.1 可变形卷积示意图
下面的Figure2展示了可变形卷积的示意图：


![Deformable Convolution的示意图](https://img-blog.csdnimg.cn/20200518211407800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到可变形卷积的结构可以分为上下两个部分，上面那部分是基于输入的特征图生成offset，而下面那部分是基于特征图和offset通过可变形卷积获得输出特征图。

假设输入的特征图宽高分别为$w$，$h$，下面那部分的卷积核尺寸是$k_h$和$k_w$，那么上面那部分卷积层的卷积核数量应该是$2\times k_h\times k_w$，其中$2$代表$x$，$y$两个方向的offset。

并且，这里输出特征图的维度和输入特征图的维度一样，那么offset的维度就是$[batch, 2\times k_h\times k_w, h, w]$，假设下面那部分设置了`group`参数（代码实现中默认为$4$），那么第一部分的卷积核数量就是$2\times kh \times kw \times group$，即每一个`group`共用一套offset。下面的可变形卷积可以看作先基于上面那部分生成的offset做了一个插值操作，然后再执行普通的卷积。

## 2.2 普通卷积和可变形卷积的差异

下面的Figure5形象的展示了普通的卷积和可变形卷积的差异：


![常规卷积和可变形卷积的差异](https://img-blog.csdnimg.cn/202005182125356.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

图中以两个$3\times 3$卷积为例，可以看到对于普通卷积来说，卷积操作的位置都是固定的。而可变形卷积因为引入了offset，所以卷积操作的位置会在监督信息的指导下进行选择，可以较好的适应目标的各种尺寸，形状，因此提取的特征更加丰富并更能集中到目标本身。

## 2.3 可变形卷积的可视化效果
下面的Fiugure6展示了可变形卷积的可视化效果。

![可变形卷积可视化效果](https://img-blog.csdnimg.cn/20200518212911687.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到每张图像中都有红色和绿色的点，并且绿色点只有一个，这个点表示的是`conv5`输出特征图上的一个位置，往前$3$个卷积层理论上可以得到$9^3=729$个红点，也即是卷积计算的区域，但是这$729$个点中有一些是越界的，所以图中显示的点会少于$729$个。

可以看到当绿色点在目标上时，红色点所在区域也集中在目标位置，并且基本能够覆盖不同尺寸的目标，因此可变形卷积可以提取更有效的特征。

而当绿色点在背景上时，红色点所在的区域比较发散，这应该是不断向外寻找并确认该区域是否为背景区域的过程。

## 2.4 可变形ROI Pooling
下面的Figure3展示了可变形ROI Pooling的示意图：


![可变形ROI Pooling的示意图](https://img-blog.csdnimg.cn/20200518213741797.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

整体上仍然可以分成上下两部分，上面那部分是基于输入特征图生成offset，下面的部分是基于输入特征图和offset通过可变形RoI Pooling生成输出RoI特征图。

上面部分先通过常规的ROI Pooling获得ROI特征图，这里假设每个RoI划分成$k\times k$个bin，Figure3中是划分成$3\times 3$个bin，然后通过一个输出节点数为$k\times k\times 2$的全连接层获得offset信息，最后再reshape成维度为$[batch, 2, k, k]$的offset。

下面部分仍然先基于offset进行插值操作，然后再执行普通的ROI Pooling，这样就完成了可变形ROI Pooling。可以看出可变形ROI Pooling相对于普通ROI Pooling来说，ROI中的每个bin的位置都是根据监督信息来学习的，而不是固定划分的。

## 2.5 可变形PSROI Pooling
下面的Figure4展示了可变形PSROI Pooling的示意图：

![可变形PSROI Pooling的示意图](https://img-blog.csdnimg.cn/20200518214308395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

PSROI Pooling是R-FCN中提出来，文章见[目标检测算法之NIPS 2016 R-FCN（来自微软何凯明团队） ](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&amp;mid=2247484442&amp;idx=1&amp;sn=b8ae9870bb6caf28cfa3720b636d1ff4&amp;chksm=9f80be8ca8f7379a81102b12958413b055dc21dd682144e08830a1738178639950a80d9eec25&token=1734809540&lang=zh_CN#rd) 。这里仍然分成上下两部分。

上面那部分先通过卷积核数量为$2\times k\times k\times (C+1)$的卷积层获得输出特征图，其中$k\times k$表示bin的数量，$C$表示目标类别数，$1$表示背景，然后基于该特征图通过PS RoI pooling操作得到输出维度为$[batch, 2\times(C+1), k, k]$的offset。

下面那部分先通过卷积核数量为$k\times k\times (C+1)$的卷积层获得输出特征图，这是R-FCN中的操作，然后基于该特征图和第一部分输出的offset执行可变形PSROI Pooling操作。可变形PSROI Pooling也可以看做是先插值再进行PS ROIPooling操作。

Figure7展示了可变形PSROI Pooling在实际图像上的效果，每张图中都有一个ROI框(黄色框)和$3\times 3$个bin（红色框），在普通的PSROI Pooling中这$9$个bin的位置应该是均匀划分的，但在可变形PSROI Pooling中这些区域是集中在目标区域的，这说明可变形结构可以让网络的注意力更集中于目标区域。

![可变形PSROI Pooling在实际图像上的效果](https://img-blog.csdnimg.cn/20200518215225416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


## 2.6 代码实现细节
在代码实现时，主网络使用了ResNet101，但是原本`stride=32`的`res5`部分修改为`stride=16`，同时可变形卷积也只在`res5`部分的$3\times 3$卷积添加，另外为了弥补修改`stride`带来的感受野减小，在`res5`的可变形卷积部分将`dilate`参数设置为2。

本来想在这里贴代码理解的，但介于篇幅原因，我就后面的推文再介绍一下Pytorch代码理解和实现吧。（逃）


# 3. 实验结果
下面的Table1是PASCAL VOC上的实验结果，包含在多种图像任务和网络的不同阶段添加可变形卷积层的差异。

![PASCAL VOC上的实验结果](https://img-blog.csdnimg.cn/20200518215918945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到，将`res5-3`的三个卷积层换成可变形卷积层就有明显的效果提升并且效果基本饱和了。

Table3主要是和atrous convolution作对比，因为atrous convolution也是增加了普通卷积操作的感受野，所以这个对比实验是在证明都增加感受野的同时，以固定方式扩大感受野和更加灵活地聚焦到目标区域谁更优。

![和atrous convolution作对比](https://img-blog.csdnimg.cn/20200518220211309.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

接下来Table4展示了模型的大小和速度对比，可以看到可变形卷积并没有带来太大的计算复杂度增长。


![模型的大小和速度对比](https://img-blog.csdnimg.cn/20200518220316136.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

最后，Table5展示了在COCO数据集上关于添加可变形卷积结构的效果对比，可以看到可变形卷积的提升还是非常明显的。

![在COCO数据集上关于添加可变形卷积结构的效果对比](https://img-blog.csdnimg.cn/20200518220450994.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 4. 结论
看完这篇论文只能说MSRA果然名不虚传，实际上我理解的DCN也是在调感受野，只不过它的感受野是可变的，这就对图像数据中的目标本身就有大量变化的情况有很大好处了。个人感觉，这确实是目标检测领域近年来非常solid的一个工作了。

# 5. 参考
- 论文原文：https://arxiv.org/pdf/1703.06211.pdf
- 代码：https://github.com/msracver/Deformable-ConvNets
- 参考博客：https://blog.csdn.net/u014380165/article/details/84894089

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)