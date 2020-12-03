# 前言
MobileNetV3是Google继MobileNet V1和MobileNet V2后的新作，主要使用了网络搜索算法(用NAS通过优化每个网络块来搜索全局网络结构，用NetAdapt算法搜索每个层的滤波器数量)，同时在MobileNet V2网络结构基础上进行改进，并引入了SE模块（我们已经讲过了SENet，[【cv中的Attention机制】最简单最易实现的SE模块](https://mp.weixin.qq.com/s/MwgRF1toP3P-gBPhX3A-eA)）和提出了H-Swish激活函数。论文原文见附录。
# 关键点
## 1. 引入SE模块
下面的Figure3表示了MobileNet V2 Bottleneck的原始网络结构，然后Figure4表示在MobileNet V2 Bottleneck的基础上添加了一个SE模块。因为SE结构会消耗一定的时间，SE瓶颈的大小与卷积瓶颈的大小有关，我们将它们全部替换为固定为膨胀层通道数的1/4。这样做可以在适当增加参数数量的情况下提高精度，并且没有明显的延迟成本。并且SE模块被放在了Depthwise卷积后面。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813103127930.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 2. 更改网络末端计算量大的层
MobileNetV2的inverted bottleneck结构是使用了`1*1`卷积作为最后一层，以扩展到高维度的特征空间(也就是下图中的320->1280那一层的`1*1`卷积)。这一层的计算量是比较大的。MobileNetV3为了减少延迟并保留高维特性，将该`1*1`层移到最终的平均池化之后(960->Avg Pool->`1*1` Conv)。现在计算的最后一组特征图从`7*7`变成了`1*1`，可以大幅度减少计算量。最后再去掉了Inverted Bottleneck中的Depthwise和`1*1`降维的层，在保证精度的情况下大概降低了15%的运行时间。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813110625940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 3. 更改初始卷积核的个数
修改网络头部卷积核通道数的数量，Mobilenet v2中使用的是$32\times 3\times 3$，作者发现，其实$32$可以再降低一点，所以这里改成了$16$，在保证了精度的前提下，降低了$3ms$的速度。

## 4. H-Swish 激活函数
Mobilenet V3引入了新的非线性激活函数：H-Wwish。它是最近的Swish非线性函数的改进版本，计算速度比Swish更快(但比ReLU慢)，更易于量化，精度上没有差异。其中Swish激活函数的公式如下:

$swish(x) = x*\delta(x)$

其中$\delta(x)$是`sigmoid`激活函数，而H-Swish的公式如下:

$h-swish(x)=x\frac{ReLU(x+3)}{6}$

简单说下，Swish激活函数相对于ReLU来说提高了精度，但因为Sigmoid函数而计算量较大。而H-swish函数将Sigmoid函数替换为分段线性函数，使用的ReLU6在众多深度学习框架都可以实现，同时在量化时降低了数值的精度损失。下面这张图提到使用H-Swish在量化的时候可以提升15%的精度，还是比较吸引人的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813111736731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 5. NAS搜索全局结构和NetAdapt搜索层结构
刚才已经提到MobileNet V3模块是参考了深度可分离卷积，MobileNetV2的具有线性瓶颈的反向残差结构(the inverted residual with linear bottleneck)以及MnasNe+SE的自动搜索模型。实际上上面的1-4点都是建立在使用NAS和NetAdapt搜索出MobileNet V3的基础结构结构之上的，自动搜索的算法我不太了解，感兴趣的可以去查看原文或者查阅资料。

# 网络结构
开头提到这篇论文提出了2种结构，一种Small，一种Large。结构如Table1和Table2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813112115659.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126181157955.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

MobileNet V3-Small网络结构图可视化结果见推文最后的图片。

# 实验
分类都在ImageNet上面进行测试，并将准确率与各种资源使用度量(如推理时间和乘法加法(MAdds))进行比较。推理时间在谷歌Pixel-1/2/3系列手机上使用TFLite运行测试，都使用单线程大内核。下面的Figure1展示了性能和速度的比较结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126214708854.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure2是运算量和准确率的比较。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126215152548.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table3是分类性能和推理速度的比较，而Table4是量化后的结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126221450569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126221714798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Figure 9展示了不同组件的引入是如何影响了延迟/准确度的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126222012241.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table6是在SSDLite中替换Backbone，在MSCOCO数据集上的比较结果。在通道缩减的情况下，MobileNetV3-Large（V3+）比具有几乎相同mAP值的MobileNetV2快25%。然后在相同的推理速度下，MobileNetV3-Small比MobileNetV2和MnasNet的mAP值高2.4和0.5。

# 补充
在知乎上看到一个回答，蛮有趣的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813113046673.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190813113100859.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论
基本上把MobileNet V3除了搜索网络结构的部分说完了，但是似乎这次Google开源的这个V3没有达到业界的预期吧。并且这篇论文给人的感觉是网络本身和相关的Trick很容易懂，但是具体是怎么搜索出V3以及预训练模型未开源这些问题仍会使我们一脸懵。但如果从工程角度来讲，毕竟使用简单，效果好对我们也足够了。

# 参考文章

- 论文原文：https://arxiv.org/pdf/1905.02244.pdf
- 源码实现：https://github.com/xiaolai-sqlai/mobilenetv3
- 参考资料1：https://zhuanlan.zhihu.com/p/69315156
- 参考资料2：http://tongtianta.site/paper/27865



文章最下方附有  MobileNetV3 Small结构可视化图

# 推荐阅读

- [快2020年了，你还在为深度学习调参而烦恼吗？](https://mp.weixin.qq.com/s/WU-21QtSlUKqyuH6Bw1IYg)

- [卷积神经网络学习路线（一）| 卷积神经网络的组件以及卷积层是如何在图像中起作用的？](https://mp.weixin.qq.com/s/MxYjW02rWfRKPMwez02wFA)

- [卷积神经网络学习路线（二）| 卷积层有哪些参数及常用卷积核类型盘点？](https://mp.weixin.qq.com/s/I2BTot_BbmR4xcArpo4mbQ)

- [卷积神经网络学习路线（三）| 盘点不同类型的池化层、1*1卷积的作用和卷积核是否一定越大越好？](https://mp.weixin.qq.com/s/bxJmHnqV46avOttAFhk28A)

- [卷积神经网络学习路线（四）| 如何减少卷积层计算量，使用宽卷积的好处及转置卷积中的棋盘效应？](https://mp.weixin.qq.com/s/Cv68oXVdB6pg_4Q_vd_9eQ)

- [卷积神经网络学习路线（五）| 卷积神经网络参数设置，提高泛化能力？](https://mp.weixin.qq.com/s/RwG1aEL2j6G-MAQRy-BEDw)

- [卷积神经网络学习路线（六）| 经典网络回顾之LeNet](https://mp.weixin.qq.com/s/oqX9h1amyalfMlHmxEg76A)
- [卷积神经网络学习路线（七）| 经典网络回顾之AlexNet](https://mp.weixin.qq.com/s/4nTRYbIZOLcMdqYpRpui6A)
- [卷积神经网络学习路线（八）| 经典网络回顾之ZFNet和VGGNet](https://mp.weixin.qq.com/s/0hQhG4Gg5AjpBUR6poVz-Q)
- [卷积神经网络学习路线（九）| 经典网络回顾之GoogLeNet系列](https://mp.weixin.qq.com/s/mXhVMHBsxrQQf_MV4_7iaw)
- [卷积神经网络学习路线（十）| 里程碑式创新的ResNet](https://mp.weixin.qq.com/s/op1ERa4GIlcbCgxFRsENdw)
- [卷积神经网络学习路线（十一）| Stochastic Depth（随机深度网络）](https://mp.weixin.qq.com/s/3mndBm86qamoy4Gn5mBLfA)
- [卷积神经网络学习路线（十二）| 继往开来的DenseNet](https://mp.weixin.qq.com/s/UP_OhkKiIwTSgkrqcEvL5g)
- [卷积神经网络学习路线（十三）| CVPR2017 Deep Pyramidal Residual Networks](https://mp.weixin.qq.com/s/CdNgtBaUIBKuzCpbxy1PXw)
- [卷积神经网络学习路线（十四） | CVPR 2017 ResNeXt（ResNet进化版）](https://mp.weixin.qq.com/s/EwQNrfhFc61lyfpaBvyKJg)
- [卷积神经网络学习路线（十五） | NIPS 2017 DPN双路网络](https://mp.weixin.qq.com/s/DaFlvbu7toR83I2M1qjSzA)
- [卷积神经网络学习路线（十六） | ICLR 2017 SqueezeNet](https://mp.weixin.qq.com/s/gMNtQvW_20O0XaNwLHS3xw)
- [卷积神经网络学习路线（十七） | Google CVPR 2017 MobileNet V1](https://mp.weixin.qq.com/s/425qUjREw-AdoySKirwD1w)
- [卷积神经网络学习路线（十八） | Google CVPR 2018 MobileNet V2](https://mp.weixin.qq.com/s/RKGJFSEuPPdWMQtI6goWzA)
- [卷积神经网络学习路线（十九） | 旷世科技 2017 ShuffleNetV1](https://mp.weixin.qq.com/s/jfBk6EX3HUu9wUIgxehKHA)
---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# MobileNetV3 Small结构可视化图

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200126182117839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)