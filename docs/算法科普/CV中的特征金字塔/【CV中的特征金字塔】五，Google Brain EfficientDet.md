![在COCO数据集上实现了精度和运算量的最好Trade-Off](https://img-blog.csdnimg.cn/20200307102304701.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 1. 前言
上周六解读了Google Brain在2019年的大作EfficientNet，可以在这个链接找到：[卷积神经网络学习路线（二十二）| Google Brain EfficientNet](https://mp.weixin.qq.com/s/uqnpIHQyZjRijwPgzYHEGg)。紧接着Google Brain又提出了这篇EfficientDet一举刷新MS COCO数据集的目标检测精度，今天就一起来看看这篇论文的核心思想吧。论文原文见附录，代码实现官方没开源，文后有一个别人复现的链接。

# 2. 摘要
模型的效率在计算机视觉中的地位越来越重要，这篇论文系统的研究了用于目标检测的各种神经网络结构设计选择，并提出了提高效率的几种关键优化方法。首先，论文提出了一个基于加权的双向特征金字塔网络(BiFPN)允许简单快速的进行多尺度特征融合。其次，论文提出了一种复合尺度扩张方法，该方法可以统一地对所有的Backbone网络，特征网络和预测网络的分辨率，深度和宽度进行缩放。基于这些优化，本文的新网络被称为EfficientDet。特别地，本文的EfficientDer-D7以52M的参数量和326B FLOPs的运算量在MS COCO数据集上实现了SOTA的51.0 mAP，比当前精度最高的检测器mAP值高3个点，速度快4倍，且参数量少9.3倍。

# 3. 贡献
EfficientDet是在EfficientNet的基础上针对目标检测任务提出的，它的贡献可以总结为如下几点：
- 论文提出了新的多尺度特征金字塔结构BiFPN(Bi-directional feature pyramid network)，即是在我们上次介绍到的[【CV中的特征金字塔】四，CVPR 2018 PANet](https://mp.weixin.qq.com/s/bUU4VaYQL80nzw3kBF-nXQ) 的基础上引入了横向直连。
- 论文还提出了weighted-BiFPN，即在不同scale的特征进行融合时引入注意力机制来对不同来源的特征进行权重调整(per-feature / per-channel / pei-pixel)，然而，这个带来的提升相比Bi-FPN来说是比较小的。
- 仿造EfficientNet的复合缩放方法，对检测网络的各个部分进行复合缩放(输入图像大小，backbone的深度、宽度，BiFPN的深度（侧向级联层数），cls/reg head的深度)。同时由于检测网络中的变量更多，所以没有使用grid search，而是基于**经验**进行了实验。

# 4. Bi-FPN的结构
如下图所示，BiFPN在Simplifield 的基础上增加了横向直连。

![Figure2 BiFPN与其他的特征融合方法的比较](https://img-blog.csdnimg.cn/2020030710394510.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

同时，作者观察到以前的征融合方法对所有输入特征一视同仁，在BiFPN中则引入了加权策略，下边介绍论文提出来的加权策略，也可以看作一种Attention机制。具体来说特征融合方法可以分成以下几种：
- **无界融合。** 公式可以表示为：$O=\sum_{i}w_i*I_i$，其中$w_i$可以是一个标量(对每个特征而言)，可以是一个向量(对每一个通道)，也可以是一个多维Tensor(对没有个像素)。
- **基于Softmax的融合。** 上面的融合方法缺点很明显，即如果不对$w_i$做限制容易导致训练不稳定，于是很自然的想到对每一个权重使用`softmax`。公式如下：$O=\sum_i\frac{e^{w_i}}{\sum_je^{w_j}}$
- **快速限制融合。** 在上面的融合中由于计算`softmax`比较慢，于是作者提出了快速的限制融合方法，公式如下：$O=\sum_{i}\frac{w_i}{\epsilon+\sum_jw_j}$。为了保证weight大于0，在weight前使用ReLU激活函数。以Figure2中的第6层为例，公式如下：

![在上图BiFPN结构中第６层中的加权特征融合](https://img-blog.csdnimg.cn/20200307110645382.png)

# 5. EfficientDet结构
## 5.1 EfficientDet 网络结构
EfficientDet的网络结构如Figure3所示，使用了EfficientNet和Bi-FPN，最后接上分类头和回归头即可。

![EfficientDet网络结构](https://img-blog.csdnimg.cn/20200307111430313.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 5.2 模型复合扩张
模型复合扩张请读一下之前对EfficientNet的解读。论文将EfficientDet的模型复合扩张分成以下几个部分。
- **对于Backbone网络**。直接采用EfficientNet-B0 to B6中的复合系数，并采用 EfficientNet作为backbone。

- **对于BiFPN 网络**：

![对于BiFPN network中width和depth的设置](https://img-blog.csdnimg.cn/20200307112127195.png)

- **对于Box/class 预测网络**:

![对Box/class prediction network中的depth的设置](https://img-blog.csdnimg.cn/20200307112208418.png)

- **对于输入图像的分辨率**（必须是$2^7＝128$的倍数）：

![对于Input image resolution的设置](https://img-blog.csdnimg.cn/20200307112255909.png)

详细的复合系数设置汇总到Table1中了。

![EfficientDet各个模型扩张复合系数表](https://img-blog.csdnimg.cn/20200307112432732.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 实验结果
在MS COCO数据集上和其他流行的检测网络的详细对比结果如Table2所示。

![EfficientDet在COCO的表现](https://img-blog.csdnimg.cn/20200307112750984.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

模型大小和推理延迟的比较如Figure4所示。

![模型大小和推理延迟](https://img-blog.csdnimg.cn/20200307112908769.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

不同特征融合方式的对比实验结果如Table5所示。

![不同特征融合方式的对比实验结果](https://img-blog.csdnimg.cn/20200307113037143.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![不同缩放方式的比较结果，本文复合融合是最强的](https://img-blog.csdnimg.cn/20200307113131772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 7. 结论
从结果看起来是非常牛逼的，不过具体用起来怎么样我们暂时也是不知道的，毕竟没有开源。等开源之后，如果真的好用，将是对目标检测领域的一个极大冲击，很可能在工业界大展身手。

# 8. 附录
- 论文原文：https://arxiv.org/abs/1911.09070
- 非官方复现：https://github.com/xuannianz/EfficientDet
- 参考：https://zhuanlan.zhihu.com/p/96773680

# 9. 推荐阅读
- [【CV中的特征金字塔】一，工程价值极大的ASFF](https://mp.weixin.qq.com/s/2f6ovZ117wKTbZvv2uRwdA)
- [【CV中的特征金字塔】二，Feature Pyramid Network](https://mp.weixin.qq.com/s/d2TSeKEZPmVy1wlbzp8BNQ)
- [【CV中的特征金字塔】三，两阶段实时检测网络ThunderNet](https://mp.weixin.qq.com/s/LX8pFMsDT21QNXtnXJIjXA)
- [【CV中的特征金字塔】四，CVPR 2018 PANet](https://mp.weixin.qq.com/s/bUU4VaYQL80nzw3kBF-nXQ)



---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)