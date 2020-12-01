# 前言
前面介绍了CornerNet和ExtremeNet，今天要介绍的是CVPR 2019一篇大名鼎鼎的Anchor-Free目标检测论文《CenterNet: Keypoint Triplets for Object Detectiontection》，这篇论文由中科院，牛津大学以及华为诺亚方舟实验室联合提出。是One-Stage目标检测算法中精度最高的算法。值得注意的是CenterNet是在之前介绍的CornerNet上进行了改进，CornerNet我们已经介绍过了，可以去看往期文章。本论文的地址以及官方代码地址见附录。

# 介绍
传统的基于关键点的目标检测方法如CornerNet就是利用目标左上角的角点和右下角的角点来确定目标，但在确定目标的过程中无法很好的利用目标内部的特征，导致产生了很多误检测（在讲CorenerNet的时候已经提到CornerNet最大的瓶颈是角点检测不准确，这正是因为它提出的Corner Pooling更加关注目标的边缘信息，而对目标内部的感知能力不强）。为了改善这一缺点，CenterNet提出使用左上角，中心，右下角三个关键点来确定一个目标，使网络花费很小的代价就具有了感知物体内部的能力，从而可以有效的抑制误检。同时，为了更好的检测中心点和角点，论文提出了Cascade Cornet Pooling和Center Pooling来提取中心点和角点的特征。CenterNet在MSCOCO数据集上获得了47%的mAP值，是One-Stage目标检测算法中的精度最高的。论文中CenterNet提到了三种用于目标检测的网络，这三种网络都是编码解码(encoder-decoder)的结构： 

- Resnet-18 with up-convolutional layers : 28.1% coco and 142 FPS 
- DLA-34 : 37.4% COCOAP and 52 FPS
- Hourglass-104 : 45.1% COCOAP and 1.4 FPS

# 原理
上面已经提到了CornerNet的缺点，即全局信息获取能力弱，无法很好的对同一目标的两个角点进行分组。如Figure1的上面两张图所示，前100个预测框中存在大量长宽不协调的误检，这是因为CornerNet无法感知物体内部的信息，这一个问题可以借助互补信息来解决如在Anchor-Based目标检测算法中设定一个长宽比，而CornerNet是无法解决的。因此，CenterNet新预测了一个目标中心点作为互补信息，并且提出了Center Pooling和Cascade Cornet Pooling来更好的提取中心点和角点的特征。如Figure1下方的两张图所示，预测框和GT框有高IOU并且GT的中心在预测框的中心区域，那么这个预测框更有可能是正确的，所以可以通过判断一个候选框的区域中心是否包含一个同类物体的中心点来决定它是否正确。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125181056401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125181115681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125181123207.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 方法
## 基准线和动机
本论文使用CornerNet作为BaseLine，其中CornerNet使用heatmaps、embeddings、offsets来确定目标的位置，具体可以见同期文章中对CornerNet的解读，为了对误检问题进行量化这篇论文提出一个新的衡量指标$FD$，计算方式为$FD=1-AP$，其中$AP$表示IOU分别取$[0.05,...0.95]$共$10$个值下的平均精度。CornerNet的误检情况见Table1，容易看出$FD=37.8$，而$FD_5=32.7$，这告诉我们即使把GT框和预测框的匹配条件限制得很死（只有那些与GT框的IoU< 0.05的预测框才被认定为错误目标框），100个预测框中仍然平均有32.7 个错误目标框。而小尺度的目标框其$FD_s$更是达到了60.3。因此，让网络具有具有感知物体内部的能力非常重要，所以CenterNet额外预测目标内部的点来添加这一能力。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125181800998.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## CenterNet
下面的Figure2是CenterNet的结构图。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125182238391.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

网络通过Center Pooling和Cascade Corner Pooling分别得到角点热力图和中心点热力图，用来预测关键点的位置。得到角点的位置和类别后，通过Offsets将角点的位置映射到输入图片的对应位置，然后通过 embedings 判断哪两个角点属于同一个物体，以便组成一个检测框。这个过程其实就是CornerNet的组合过程，CenterNet的不同之处在于它还预测了一个中心点，对于每个目标框定义一个中心区域，通过判断每个目标框的中心区域是否含有中心点，若有则保留，并且此时框的置信度分数为中心点、左上角点和右下角点的置信度分数的平均值；若无则去除，使得网络具备感知目标区域内部信息的能力，能够有效去除错误的目标框。

另外一个问题是如何定义中心区域，如果定义的中心区域太小会导致很多小尺度的错误目标框无法被去除，而中心区域太大会导致很多大尺度的错误目标框无法去除，为了解决这一问题论文提出了尺度可调节的中心区域定义法。具体如公式(1)所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125183055135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中$n$的值根据边界框是否大于$150$进行设置为$3$或$5$，可视化效果如Figure3所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125183250247.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 丰富中心点和角点特征

- **Center Pooling**：一个物体的中心点不一定含有可以和其它类别有很大区分性的语义信息（例如人的头部含有很强的易区分于其它类别的语义信息，但是人这个物体的中心点基本位于身体的中部）。下面的Figure4(a)表示Center Pooling的原理，Center Pooling提取中心点水平方向和垂直方向的最大值并相加，给中心点提供除了所处位置以外的信息，这使得中心点有机会获得更易于区分于其他类别的语义信息。Center Pooling 可通过不同方向上的Corner Pooling 的组合实现，例如一个水平方向上的取最大值操作可由Left Pooling 和Right Pooling通过串联实现。同理，一个垂直方向上的取最大值操作可由Top Pooling 和Bottom Pooling通过串联实现，具体操作如Figure5（a）所示，特征图两个分支分别经过一个$3\times3 Conv-BN-ReLU$，然后做水平方向和垂直方向的Corner Pooling，最后再相加得到结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125184902248.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125184920740.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

- **Cascade Corner Pooling**：这一模块用于预测目标的左上角和右下角角点，一般情况下角点位于物体外部，所处位置并不含有关联物体的语义信息，这为角点的检测带来了困难。Figure4（b）是CornerNet中的做法即提取物体边界最大值进行相加，该方法只能提供关联物体边缘语义信息，对于更加丰富的物体内部语义信息则很难提取到。所以这篇论文提出了Cascade Corner Pooling，原理如Figure5（b）所示，它首先提取出目标边缘最大值，然后在边缘最大值处继续向物体内部（如Figure4(c)所示）提取最大值，并和边界最大值相加，以此给角点提供更丰富的关联目标语义信息。Figure5(b)展示了Cascade Top Corner Pooling的原理。这里需要注意一下Cascade Corner Pooling只是为了通过内部信息丰富角点特征，也就是级联不同方向的Corner Pooling实现内部信息的叠加，最终的目的还是要预测角点，所以最终左上角点通过Cascade Top Corner Pooling+Cascade Left Corner Pooling实现，右下角点通过Cascade Right Corner Pooling+Cascade Bottom Corner Pooling实现。

## 训练和推理
CenterNet的损失函数如下所示，由角点位置损失(Focal Loss)，中心点位置损失，embedding损失，offsets损失(L1损失)组成，结构和CornerNet类似，只是增加了中心点损失项，$\alpha$，$\beta$和$\gamma$分别为$0.1$,$0.1$和$1$。`batch_size`设置为$48$，在Tesla V100(32G) GPU上迭代480K次，前450K次学习率为$2.5\times 10^{-4}$，后30K次学习率为$2.5\times 10^{-5}$。代码使用Pytorch框架实现，没有预训练，输入输出分辨率都是$511\times 511$，损失函数优化策略为Adam。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125201932336.png)

# 实验结果
最终的实验结果如Table2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125202516371.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到CenterNet获得了47%的mAP，超过了所有的One-Stage算法，领先幅度越5%，并且精度和Two-Stage的目标检测算法的最好结果也是接近的。

下面的Table3是CenterNet 与 CornerNet 的单独对比，可以看出在MS COCO数据集上CenterNet消除大量误检框，尤其是在小物体上。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125214605594.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure6展示了CenterNet和CornerNet的对比结果。(a) 和 (b) 表明 CenterNet 能有效去除小尺度的错误目标框。(c) 和 (d) 表明 CenterNet 能有效去除中等尺度和大尺度的错误目标框。（e）是否采用Center Pooling检测中心点。（f）对比分别使用Corner Pooling和Cascade Corner Pooling检测角点。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125214819867.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure7展示了置信度在0.5以上的目标框分布情况，可以看到CenterNet去除了大量错误的目标框，因此即使在目标框的置信度比较低的情况下，依然可以保证较好的效果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125215026526.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table4是消融实验。分别说明了本文提出的CRE（中心点加入），CTP（中心点池化），CCP（级联角点池化）的有效性。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125215355958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

最后Table5是错误实验分析，将检测的中心点用真实的中心点代替，实验结果表明中心点的检测准确度还有很大的提升空间。同时该结果还表明要想更进一步的提升检测精度，需要进一步提升中心点的检测精度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200125215435352.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论
这篇论文在CornerNet的基础上增加了一个中心点来消除误检框，基本想法来源于：“网络具备感知物体内部信息的能力”，并且论文提出的这一思想也可以用于其他的Anchor-Based或者Anchor-Free的目标检测算法中带来效果提升。

# 附录

- 论文原文：https://arxiv.org/abs/1904.08189
- 代码地址：https://github.com/Duankaiwen/CenterNet
- 参考1：https://zhuanlan.zhihu.com/p/66048276
- 参考2：https://www.cnblogs.com/gawain-ma/p/10882113.html

# 同期文章

- [目标检测算法之Anchor Free的起源：CVPR 2015 DenseBox](https://mp.weixin.qq.com/s/gYq7IFDiWrLDjP6219U6xA)
- [【目标检测Anchor-Free】ECCV 2018 CornerNet](https://mp.weixin.qq.com/s/cKOna7GfTwl1X1sgYNXcEg)
- [【目标检测Anchor-Free】CVPR 2019 ExtremeNet（相比CornerNet涨点5.3%）](https://mp.weixin.qq.com/s/Sj0zgcFFt_W9yZy37oENUw)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)