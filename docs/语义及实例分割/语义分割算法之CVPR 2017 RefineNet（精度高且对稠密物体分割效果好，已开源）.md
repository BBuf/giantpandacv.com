![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121721535891.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 前言
前面介绍了很多目标检测的算法，为了平衡一下，今天介绍一个语义分割算法QAQ。这是来自CVPR 2017的RefineNet，在PSCAL VOC 2012上达到了83.4的mIOU，是一个非常优秀的网络，同时其论文的思想也能对我们做工程做学术给予启发，我们来一起看看吧。对了，说一句题外话，卷积神经网络系列已经更新了5篇，接下来打算更新20篇左右从2012年到现在非常经典的CNN网络，一起来学习CNN呀。这篇论文的地址见附录。

# 背景
当前流行的CNN如VGG，ResNet等由于池化层和卷积步长的存在，特征图分辨率越来越小，导致损失了一些细节信息，我们在[卷积神经网络学习路线（一）| 卷积神经网络的组件以及卷积层是如何在图像中起作用的？](https://mp.weixin.qq.com/s/MxYjW02rWfRKPMwez02wFA) 介绍过低层的特征图有丰富的细节信息，而高层特征图则有更加抽象的语义信息。对于分类问题来说，只要高级语义信息足够强，模型就能表现得好。而对于稠密预测如逐像素的语义分割问题来说，除了高级语义特征之外，还需要低层的细节信息。针对这一问题，有很多方法被提出。例如以SegNet结构为典型的编解码结构，使用反卷积来恢复图像分辨率，但却依然很难恢复细节信息。还有Unet为代表的使用跳跃连接来产生高分辨率的预测。以及DeepLab系列使用空洞卷积保持分辨率，增大感受野，但这种做法增加了计算量，并且空洞卷积容易损失一些高层语义信息。

# 主要贡献
针对上面的背景，RefineNet被提出。论文认为高级语义特征有助于分类识别，而低层语义特征有助于生成清晰，详细的边界。因此作者在Unet的基础上进行魔改，产生了今天要介绍的RefineNet。它的主要贡献是：
- 提出一种多路径精细化的语义分割网络RefineNet。这个网络可以利用多个层次的特征，使得语义分割精度更高。
- RefineNe的使用了大量的残差结构，使得网络梯度不容易发散，训练起来更加容易和高效。
- 提出了一个Chained Residual Pooling模块，可以从一个大的图像区域捕捉背景上下文信息。

# 网络结构
论文提出的网络结构可以分为两段分别对应于U-Net中的向下和向上两个过程。其中向下的过程以ResNet为基础，向上的过程使用了新提出的RefineNet为基础，并将ResNet中的低层特征和当前RefineNet的特征加以fusion。整体框架如Figure2(c)所示。其中左边的4组特征是从ResNet的4个Block中取出的。值得一提的是，具体用多少个特征，以及用多少个RefineNet级联是可以灵活改变的，论文中也有相关实验。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217212202805.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# RefineNet结构
RefineNet的细节结构如Figure3所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217213127909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
RefineNet的结构可以分为3个部分，首先不同尺度的输入特征首先经过2个残差模块的处理，然后将不同尺寸的特征进行融合，当然如果只有一个输入尺度，该模块则可以省去。所有特征均UpSampling到和低层特征分辨率一样的尺寸，然后进行加和，其中采样之前的卷积模块是为了调整不同特征的通道数。最后一个是这篇论文的另外一个创新点，即一个链式的池化模块。其设计的本意是想让侧支上的一系列池化(尺寸比较大，步长为1)来获取背景信息，然后主支上的ReLU以在不显著影响梯度流通的情况下提高后续pooling的性能，同时不让网络的训练对学习率很敏感。
最后网络再经过一个残差模块即得到RefineNet的输出结果。

# 一张带有Tensor尺寸的结构图
此图来自CSDN的gqixl博主，十分感谢，侵删。原图地址见附录。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217214615721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 实验结果
论文在很多数据集上做了实验，都取得了当时的SOTA精度。下面仅仅给一下PASCAL VOC 2012的测试结果如Table5所示，其他测试结果请参考原文。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217214914678.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
可以看到RefineNet在PASCAL VOC 2017上取得了83.4的mIOU，证明了该网络的有效性。下面再给一下RefineNet在VOC2012和Cityscapes数据集上的一些可视化结果，如图Figure5和Figure6所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121721513666.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191217215150211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 附录
论文原文：https://arxiv.org/pdf/1611.06612.pdf
参考文章：https://blog.csdn.net/gqixf/article/details/82911220
Pytorch源码实现：https://github.com/thomasjpfan/pytorch_refinenet
# 后记
今天为大家介绍了CVPR 2017的RefineNet，希望能为大家做语义分割任务提供一个新的的思路，那么今天就介绍到这里啦。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)