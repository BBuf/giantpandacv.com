> 为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码在文章末尾，感兴趣可以加入。
# 1. 前言
Single Stage Headless Face Detector（SSH）是ICCV 2017提出的一个人脸检测算法，它有效提高了人脸检测的效果，主要的改进点包括多尺度检测，引入更多的上下文信息，损失函数的分组传递等等，论文相对比较简单，获得的效果也还不错（从Wider Face的结果来看，和前几天介绍的[在小尺寸人脸检测上发力的S3FD](https://mp.weixin.qq.com/s/XrCY91IrfKBVOKPeIcJOGA) 差不多）。


# 2. 网络结构
SSH算法的网络结构如Figure2所示：

![Figure2 SSH算法的网络结构](https://img-blog.csdnimg.cn/20200517165352674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

SSH算法是在VGG的基础上进行了改进，创新点主要有两个，即**尺度不变性和引入更多的上下文信息**。

在Figure2中，**尺度不变性**是通过不同尺度的检测层来完成的，和SSD，YOLOV3等目标检测算法类似。具体来说一共有$3$个尺寸的检测模块(**detection module**)，检测模块 M1，M2，M3的`stride`分别为$8$，$16$，$32$，从图中也可以看出M1主要用来检测小尺寸人脸，M2主要用来检测中等尺寸人脸，M3主要用来检测大尺寸人脸。每个检测模块都包含了分类(Scores)和回归(Boxes)两个分支。检测模块M2是直接接在VGG的`conv5_3`层后面，而检测模块M1的输出包含了较多的特征融合和维度缩减(从$512$->$128$)操作，从而减少计算量。

而**引入更多的上下文信息** 是通过在检测模块中插入上下文模块(context module)实现的，上下文模块的结构如Figure4所示，它是通过将原始的特征分别接一个$3\times 3$卷积的支路和$2$个$3\times 3$卷积的支路从而为特征图带来不同的感受野，达到丰富语义信息的目的：

![通过增大感受野引入更多的上下文信息](https://img-blog.csdnimg.cn/20200517170139663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 3. 创新点详解
刚才提到，SSH算法的创新点就$3$个，即新的检测模块，上下文模块以及损失函数的分组传递，接下来我们就再盘点一下：

## 3.1 检测模块

下面的Figure3是检测模块的示意图：

![检测模块结构](https://img-blog.csdnimg.cn/20200517171033623.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

分类和回归的特征图是融合了普通卷积层和上下文模块输出的结果。分类和回归支路输出的$K$表示特征图上每个点都设置了$K$个Anchor，这$K$个Anchor的宽高比例都是$1:1$，论文说增加宽高比例对于人脸检测的效果没有提示还会增加Anchor的数量。


## 3.2 上下文模块
下面的Figure4是上下文模块的示意图：

![上下文模块结构](https://img-blog.csdnimg.cn/20200517173754463.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

作者原本是通过引入卷积核尺寸较大的卷积层例如$5\times 5$和$7\times 7$来增大感受野，从而引入更多的上下文信息。但为了减少计算量，作者借鉴了GoogleNet中用多个$3\times 3$卷积代替$5\times 5$卷积或者$7\times 7$卷积的做法，于是最终上下文模块的结构就如Figure4所示了，另外注意到开头的$3\times 3$卷积是共享的。

## 3.3 损失函数的分组传递
SSH算法还对损失函数做了一些调整，公式如下所示：

![损失函数](https://img-blog.csdnimg.cn/20200517174749185.png)

分类的损失函数还是采用常用的二分类损失函数，其中$A_k$表示属于检测模块$M_k$的Anchor，而回归损失部分多了一个$I(g_i=1)$，意思对于不同尺度的检测模块来说，只回传对应尺度的Anchor损失，这就实现了第二节中提到的M1主要用来检测小人脸，M2主要用来检测中等尺寸人脸，M3主要用来检测大尺寸人脸的目的。另外，在引入OHEM算法时也是针对不同尺度的检测模块分别进行的。


# 4. 实验结果
下面的Table1展示了不同的人脸检测算法在Wider FACE数据集上的效果对比。HR算法的输入为图像金字塔，可以看到不使用图像金字塔的SSH算法效果都超过了相同特征提取网络的HR算法。最后一行的SSH(VGG-16)+Pyramid表示的是输入为图像金字塔，可以看到准确率进一步提升了。


![SSH算法在Wider FACE上的实验结果](https://img-blog.csdnimg.cn/20200517175800752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table2展示了在NVIDIA Quadro P6000 GPU上的速度测试，结果如下：

![速度测试结果](https://img-blog.csdnimg.cn/20200517180113680.png)

# 5. 总结
这篇文章介绍了一下用于人脸检测的SSH算法，它提出的上下文模块和损失函数的分组传递还是比较有意思的，论文的精度也说明这几个创新点是有用的。但是论文给出的实验结果比较少，所以我们无法判断每个Trick对结果的影响幅度到底多大，这是比较遗憾的。

# 6. 附录
- HR算法：https://blog.csdn.net/wfei101/article/details/80932095
- 参考：https://blog.csdn.net/u014380165/article/details/83590831
- 论文原文：https://arxiv.org/pdf/1708.03979.pdf
- 代码：https://github.com/mahyarnajibi/SSH

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)