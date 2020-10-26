> 论文全称：Parallel Feature Pyramid Network for Object Detection  
>   论文链接：http://openaccess.thecvf.com/content_ECCV_2018/papers/Seung-Wook_Kim_Parallel_Feature_Pyramid_ECCV_2018_paper.pdf

# 1. 前言
今天来学习一下这篇ECCV 2018的网络PFPNet，它借鉴了SPP的思想并通过MSCA（多尺度语义融合）模块来进行特征融合，进而提出了PFPNet来提升目标检测算法的效果。PFPNet在结构上借鉴了SSD，而在特征融合上借鉴了SPP思想加宽了网络，同时这里提出的MSCA模块完成了类似于FPN的特征融合，最后基于融合后的特征再进行检测，最终PFPNet在多个BenchMark上获得了和[CVPR 2018 RefineDet](https://mp.weixin.qq.com/s/4PQQwDGGyiK_w1TBgYBDIg)相似的性能。

# 2. 网络结构上的改进
下面的Figure1展示了目标检测算法在网络结构上经历的一些优化过程（注意这是截至到PFPNet以前）。

![标检测算法在网络结构方面的优化历程](https://img-blog.csdnimg.cn/20200319170350593.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

`Figure1(a)`上半部分的结构是直接通过一个网络结构得到特征，并基于该特征进行预测。而下半部分的结构则引入了特征金字塔，基于多个特征层进行预测，这和SSD的思想类似。

`Figure1(b) `中的2个网络则是引入了top-down结构和融合操作，也就是从高层到浅层的特征融合过程（Hourglass网络），**这样预测层既有浅层的目标的位置信息（有利于定位）又有高层的目标的语义信息（有利于目标的识别），因此能有效提高目标的检测效果（尤其是小目标）** 。然后这两个网络的区别是第一个是单级预测，第二个是多级预测类似FPN。

`Figure1(c)`展示了SPP网络。即对同一个特征图进行不同尺寸的池化操作获得不同尺度的特征图，将不同尺度的特征图concate之后再做预测，这样融合了多个尺度的特征也可以对检测效果有所提升。

`Figure1(d)`展示了这篇文章提出的PFPNet的结构示意图。首先也是通过SPP得到不同尺度的特征图，然后基于这些特征图**通过MSCA模块得到融合后的特征**，最后基于融合后的多层特征做预测。 

# 3. PFPNet的整体结构
下面的Figure3展示了PFPNet的整体结构。


![Figure 3](https://img-blog.csdnimg.cn/20200319173443802.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
对于原始的输入图片，先通过一个BackBone网络提取特征，假设提取到的特征通道数是$D$，然后将提取到的特征送入SPP网络获得不同尺度的特征图，用$N$表示尺度的数量这里位$3$，得到的特征图通道数用$C_H$表示，当然$C_H=D$。然后再通过一个通道缩减操作将(b)中的特征图的通道缩减，也即是Bottleneck操作，缩减后的通道数用$C_L$表示，公式是：$C_L=D/(N-1)$。然后再通过MSCA操作得到融合后的特征图（对应(d)操作），得到的特征通道数是$C_p$，最后基于融合后的多个特征图进行预测。

# 4. MSCA模块
下面的Figure4是MSCA模块的结构示意图。MSCA主要是基于特征通道的concate操作，但是输入特征有些特殊。

![MSCA模块](https://img-blog.csdnimg.cn/20200319175049768.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
在Figure4中，当得到P1这个融合后的特征时是用$F_H(1)$、下采样后的$F_L(0)$、上采样后的$F_L(2)$进行concate后得到的。

**这个地方有个问题，为什么是使用$F_L(1)$而不是$F_H(1)$呢？**

论文说的是相同尺度的特征信息要足够多，而$F_H$部分得到的特征是未经过通道缩减的，因此在得到某一个尺度的预测层特征时，被融合的对应尺度特征都是采用$F_H$部分的输出特征，而不是$F_L$部分的输出特征。相比之下，不同尺度的待融合特征采用$F_L$部分的输出特征，相当于补充信息。

# 5. 实验结果
下面的Table3展示了PFPNet在VOC数据集上的测试结果，需要说明的是PFPNet-S300表示anchor的初始化和SSD算法一样，PFPNet-R320表示初始anchor采用RefineDet算法的ARM模块得到的refine后的anchor来初始化。关于RefineDet请看：[目标检测算法之CVPR 2018 RefineDet](https://mp.weixin.qq.com/s/4PQQwDGGyiK_w1TBgYBDIg)


![Table3](https://img-blog.csdnimg.cn/20200319181056891.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
可以看到PFPNet-S300和PFPNet-S512的性能和RefineDet差不多。

下面的Table4则展示了PFPNet在COCO数据集上的测试结果。

![Table4](https://img-blog.csdnimg.cn/2020031918122056.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 参考
- https://blog.csdn.net/u014380165/article/details/82468725

# 7. 推荐阅读
- [【CV中的特征金字塔】一，工程价值极大的ASFF](https://mp.weixin.qq.com/s/2f6ovZ117wKTbZvv2uRwdA)
- [【CV中的特征金字塔】二，Feature Pyramid Network](https://mp.weixin.qq.com/s/d2TSeKEZPmVy1wlbzp8BNQ)
- [【CV中的特征金字塔】三，两阶段实时检测网络ThunderNet](https://mp.weixin.qq.com/s/LX8pFMsDT21QNXtnXJIjXA)
- [【CV中的特征金字塔】四，CVPR 2018 PANet](https://mp.weixin.qq.com/s/bUU4VaYQL80nzw3kBF-nXQ)
- [【CV中的特征金字塔】五，Google Brain EfficientDet](https://mp.weixin.qq.com/s/ughHo9Q1L0c4stZIxs-3ow)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)