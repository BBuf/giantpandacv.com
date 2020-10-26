> 看了不少的目标检测论文了，个人认为多数论文的出发点就两个，一是感受野，二是特征融合。此外，解决数据不平衡和轻量化也是另外两个重要的方向。今天要讲解的RFBNet就是从感受野角度来改善了SSD检测器。
# 1. 前言
今天为大家科普一篇ECCV 2018的一篇目标检测网络RFBNet，论文全名为：**Receptive Field Block Net for Accurate and Fast Object Detection**  。这篇论文主要的贡献点主要是在SSD网络中提出了一个**Receptive Field Block (RFB)** 模块，RFB模块主要是在Inception的基础上加入了空洞卷积层从而有效的增大了感受野。另外，RFB模块是嵌在SSD上的，所以检测的速度比较快，精度比SSD更高。

# 2. RFB模块
RFB模块的效果示意图如Figure2所示，其中虚线部分就是指RFB模块。

![Figure2. RFB效果示意图](https://img-blog.csdnimg.cn/20200303163329165.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

RFB模块主要有两个特点：

- RFB模块有多个分支，每个分支的第一层都由特定大小卷积核的卷积核构成，例如图上的$1\times 1$, $3\times 3$，$5\times 5$。
- RFB模块引入了空洞卷积，主要作用是为了增加感受野，空洞卷积之前是应用在分割网络DeepLab中，这里将其应用在检测任务中，以获得更大的感受野，可以更好的编码空间长距离语义。

在RFB模块中，最后将不同尺寸和感受野的输出特征图进行Concat操作，以达到融合不同特征的目的。在Figure2中，RFB模块中使用三种不同大小和颜色的输出叠加来展示。在Figure2的最后一列中将融合后的特征与人类视觉感受野做对比，从图中看出是非常接近的，这也是这篇论文的出发点。

# 3. 两种RFB结构示意图
下面的Figure4展示了RFBNet的两种结构。


![RFBNet的两种结构](https://img-blog.csdnimg.cn/2020030316511242.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

- Figure4(a)表示RFB结构，整体上借鉴了Inception的思想。主要不同点在于引入3个空洞卷积层。
- Figure4(b)表示RFB-s结构。RFB-s和RFB相比主要有两个改进，一方面用$3\times 3$卷积层代替$5\times 5$卷积层，另一方面用$1\times 3$和$3\times 1$卷积层代替$3\times 3$卷积层，主要目的应该是为了减少计算量，类似Inception后面的版本对Inception结构的改进。

# 4. RFBNet-300的结构
下面的Figure5代表RFBNet-300的整体结构图，基本是照搬了SSD，主要有2点不同：

- BackBone部分用两个RFB结构替换原来新增的两层。
- `conv4_3`和`conv7_fc`在接预测层之前分别接RFB-s和RFB结构。

![RFBNet-300的网络结构](https://img-blog.csdnimg.cn/20200303170634648.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 5. 实验结果
下面的Table1展示了RFBNet-300在PASCAL VOC2007 test dev上的测试结果，训练集基于2007和2012的trainval。RFBNet-300在mAP和FPS两方面效果都不错。 

![RFB-Net在PASCAL VOC2007 test dev上的测试结果](https://img-blog.csdnimg.cn/20200303171628909.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

而下面的Table4是在COCO test dev 2015数据集上的测试结果。最后的RFBNet-512E做了两个改变：
- 对`conv7_fc`的输出特征做了`up-sample`，然后和`conv4_3`的输出特征做融合，基于融合后的特征做预测。这种做法实际上是借鉴了FPN算法的思想。
- RFB结构中增加了`7x7`大小的卷积分支。这两点改进对效果的提升有一定帮助，而且带来的计算量也少。

![在COCO test dev 2015数据集上的测试结果](https://img-blog.csdnimg.cn/20200303172951950.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure6展示了和RFBNet同时期的一些目标检测算法在COCO test-dev数据集上关于效果和速度的直观对比，可以看到RFBNet的速度和精度平衡还是不错的。

![RFBNet同时期的一些目标检测算法在COCO test-dev数据集上关于效果和速度的直观对比](https://img-blog.csdnimg.cn/20200303173409838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 后记
RFBNet说白了就是空洞卷积的应用，虽然看起来论文比较水，但至少给我们提供了一个重要信息，在检测中调感受野是行之有效的。

# 7. 附录
- 论文原文：https://arxiv.org/pdf/1711.07767.pdf
- 代码实现：https://github.com/ruinmessi/RFBNet
- 参考：https://blog.csdn.net/u014380165/article/details/81556769

# 8. 推荐阅读
[目标检测和感受野的总结和想法](https://mp.weixin.qq.com/s/9169hhoJwYd0VckNt8VDLg)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)