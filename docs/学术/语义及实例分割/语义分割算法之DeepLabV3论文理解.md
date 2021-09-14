# 论文地址
https://arxiv.org/abs/1706.05587

# 摘要
本文首先回顾了空洞卷积在语义分割中的应用，这是一种显式调整滤波器感受野和控制网络特征响应分辨率的有效工具。为了解决多尺度分割对象的问题，我们设计了采用级联或并行多个不同膨胀系数的空洞卷积模块，以更好的捕获上下文语义信息。此外，我们扩充了在DeepLab V2中提出的ASPP模块，进一步提升了它的性能。并且我们还分享了一些训练系统方面的经验和一些实施方面的细节。

# 介绍

作者提到DeepLab系列面临三大挑战：

**挑战一**：为分类任务设计的DCNN中的多次Max-Pooling和全连接层会导致空间信息的丢失。在DeepLabV1中引入了空洞卷积来增加输出的分辨率，以保留更多的空间信息。

**挑战二**：图像存在多尺度问题，有大有小。一种常见的处理方法是图像金字塔，即将原图resize到不同尺度，输入到相同的网络，获得不同的feature map，然后做融合，这种方法的确可以提升准确率，然而带来的另外一个问题就是速度太慢。DeepLab v2为了解决这一问题，引入了ASPP（atrous spatial pyramid pooling）模块，即是将feature map通过并联的采用不同膨胀速率的空洞卷积层，并将输出结果融合来得到图像的分割结果。

**挑战三**：分割结果不够精细的问题。这个和DeepLabV1的处理方式一样，在后处理过程使用全连接CRF精细化分割结果。

在本文中，我们我们重新讨论了在级联模块和空间金字塔池化框架下应用空洞卷积，这能够有效地扩大滤波器的感受野，将多尺度的上下文结合起来。特别地，我们提出的模块由具有不同采样率的空洞卷积和BN层组成，对于训练非常重要。我们实验了级联和并行的方式来部署ASPP模块。还有一个重要的问题是，采用采样率非常大的3*3空洞卷积，由于图像边界效应，不能捕捉图像的大范围信息，也即是原文说的会退化成1*1卷积，所以论文在这里提出在ASPP模块中加入图像级特征。此外，我们详细介绍了实现的细节，并分享了训练模型的经验，还包括一种简单而有效的引导方法，用于处理稀有和精细标注的对象。

# 相关工作
很多工作已经证明了全局特征或上下文的语义信息有助于语义分割。在本文中，一共讨论了四种利用上下文信息进行语义分割的全卷积网络(FCNs)，如Figure 2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106152312682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**图像金字塔(Image pyramid)：** 多个尺度的图片输入到一个相同的网络中。小尺寸的输入有助于理解长距离的语义，大尺寸的输入有助于修正细节。使用拉普拉斯金字塔对输入图像进行变换，将不同尺度的图片输入到DCNN，并将所有比例的特征图合并。有人将多尺度的输入按顺序从粗到细依次应用，也有人将输入直接调整成不同的大小，并融合所有大小的特征。这类模型的主要缺点是由于GPU内存有限，较大较深的CNN不方便使用，因此通常在推理阶段应用。

**编码器-解码器(Encoder-Decoder)**  该模型由两部分组成：(a)编码器中，
特征映射的空间维度逐渐减少，从而更容易捕获较长范围内的信息；(b)解码器中，目标细节和空间维度逐渐恢复。例如，有人用反卷积来学习对低分辨率特征响应进行上采样。SegNet复用编码器中的池化索引，学习额外的卷积层来平滑特征响应。U-Net将编码器中的特征层通过跳跃连接添加到相应的解码器激活层中。LRR使用了一个拉普拉斯金字塔重建网络。最近，RefineNet等证明了基于编码-解码结构的有效性。这类模型也在目标检测的领域得到了应用。

**上下文模块（Context module）** 包含了额外的模块，采用级联的方式，用来编码远距离上下文信息。一种有效的方法是合并Dense CRF到DCNN中，共同训练DCNN和CRF。

**空间金字塔池化(Spatial pyramid pooling)** 空间金字塔池化可以在多个范围内捕捉上下文信息。DeepLabv V2提出了空洞卷积空间金字塔池化(ASPP)，使用不同采样率的并行空洞卷积层才捕获多尺度信息。PSPNet在不同网格尺度上执行空间池化，并在多个语义分割数据集上获得出色的性能。还有其他基于LSTM的方法聚合全局信息。

# 方法
这里主要回顾如何应用atrous convolution来提取紧凑的特征，以进行语义分割； 然后介绍在串行和并行中采用atrous convolution的模块。
**1. Atrous Convolution for Dense Feature Extraction**
假设2维信号，针对每个位置i，对应的输出y， 以及filter w，对于输入feature map x进行 atrous convlution 计算：  

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106160829346.png)

其中x代表输入信号，w是卷积核系数，y是输出，其中k是输入信号维度，r是膨胀速率，如果r等于就退化为标准卷积。

**Going Deeper with Atrous Convolution**

以串行方式设计atrous convolution 模块，复制ResNet的最后一个block，如下图的block4，并将复制后的blocks以串行方式级联。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106161941484.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

对于Figure3的图a，每一个块都有三个3×3卷积。除了最后一个块，其余的模块中最后的一个卷积步长为2，类似于原来的ResNet。这么做背后的动机是，引入的stride使得更深的模块更容易捕获长距离的信息。整个图像的特征都可以汇聚在最后一个小分辨率的特征图中。然而，我们发现续的stride(stride过大对于短距离的语义信息捕获很不利)对语义分割是有害的，会造成细节信息的丢失(如下表)。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106162636499.png)

因此我们使用了不同采样率的空洞卷积。如Figure 3(b)中，输出步幅为out_stride = 16。这样可以在不增加参数量和计算量的同时有效缩小步幅。

**Multi-grid Method**

对block4～block7 采用不同atrous rates。即，定义Multi_Grid=(r1,r2,r3)为block4～block7的三个卷积层的 unit rates。卷积层最终的atrous rate等于unit rate和对应的rate的乘积。例如，当output_stride=16， Multi_Grid=(1,2,4)时, block4中three convolutions的rate分别为：rates=2∗(1,2,4) = (2,4,8)。


**ASPP 模块**
DeepLab V3中将BN层加入到V2提出的ASPP模块中。具有不同atrous rates的ASPP模块可以有效的捕获多尺度信息。不过，论文发现，随着sampling rate的增加，有效filter特征权重（即有效特征区域，而不是补零区域的权重）的数量会变小。如下图所示，当采用具有不同atrous rates的3×3 filter应用到65×65 feature map时，在rate值接近于feature map 大小的极端情况，该3×3 filter不能捕获整个图像内容，而退化成了一个简单的1×1 filter， 因为只有中心 filter 权重才是有效的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106165445239.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了解决这一问题，并将全局内容信息整合到模型中，则采用图像级特征。即，采用全局平均池化(global average pooling)对模型的 feature map 进行处理，将得到的图像级特征输入到一个 1×1 convolution with 256 filters(加入 batch normalization)中，然后将特征进行双线性上采样(bilinearly upsample)到特定的空间维度。最后，论文改进了ASPP， 即: 
(a) 当output_stride=16时，包括一个 1×1 convolution 和三个3×3 convolutions，其中3×3 convolutions的 rates=(6,12,18)，(所有的filter个数为256，并加入batch normalization)。 需要注意的是，当output_stride=8时，rates将加倍。
(b) 图像级特征, 如 Figure5。

连接所有分支的最终特征，输入到另一个 1×1 convolution(所有的filter个数也为256，并加入batch normalization)，再进入最终的 1×1 convolution，得到 logits 结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110618410951.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 实验结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191106184332393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

文章还给出了一些可视化结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019110618444130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论

提出的模型DeepLab V3采用atrous convolution的上采样滤波器提取稠密特征映射和去捕获大范围的上下文信息。具体来说，编码多尺度信息，提出了级联模块逐步翻倍的atrous rates，提出了ASPP模块增强图像级的特征，探讨了多采样率和有效视场下的滤波器特性。实验结果表明，该模型在Pascal voc 2012语义图像分割基准上比以前的DeppLab版本有了明显的改进，并取得了SOAT精度。