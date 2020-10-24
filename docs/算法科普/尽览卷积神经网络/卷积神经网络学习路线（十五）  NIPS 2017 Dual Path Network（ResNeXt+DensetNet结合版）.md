# 前言
前面已经讲了ResNet，ResNeXt，以及DenseNet，讲解的原文都可以在文后找到。今天要介绍的DPN（双路网络）是2017年由颜水成老师提出的，作者的简介如下大家可以感受一下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114194516207.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
DPN就是在ResNeXt和DenseNet的基础上，融合这两个网络的核心思想而成，论文原文见附录。

# 贡献
- DPN的Dual Path（双路）结构结合了ResNeXt（残差分组卷积）和DenseNet（稠密连接）两种思想。即可以利用残差网络的跳跃连接对特征进行复用，又可以利用密集连接路径持续探索新特征。
- 使用了分组卷积，降低了计算量。
- 性能超越ResNeXt和DenseNet，可以做检测分割任务新BackBone。

# 网络结构
DPN的网络结构如Table1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114200056498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
可以看到DPN的网络结构和ResNeXt的网络结构很类似。最开始是一个$7\times 7$卷积层，接着就是一个最大池化层，再然后是四个`stage`，再接一个全局平均池化以及全连接层，最后是`softmax`层。整体结构是这样，重点就在每个`stage`具体是怎么变化的了，接下来我们就一起来理解一下。

上面说了DPN网络就是把ResNeXt和DenseNet融合成1个网络，因此这里首先介绍一下这篇论文是如何表达ResNeXt和DenseNet的，具体如Figure2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114203830968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- 其中Figure2（a）就是ResNeXt的一部分，其中(a)的左边这个矩形的小**凸起**部分就代表每个残差模块输入输出，对于一个输入`x`，分两条路走，一条是`x`本身，即残差模块中的跳跃连接，而另一条就是经过`1x1`卷积+`3x3`卷积+`1x1`卷积（即瓶颈结构），然后把这两条路获得结果做一个求和即可，也即是文中的`+`符号，得到下一个残差模块的输入。

- 而Figure2（b）是DenseNet的一部分，其中(b)左边的多边形**凸起**代表每个密集连接模块输入输出，对于输入`x`只有一条路，即经过几层卷积之后和`x`做一个通道合并（`concat`），得到的输出又是下一个密集连接模块的输入，这样每一个密集连接模块的输入都在不断累加，可以看到这个多边形越来越宽。
- Figure2（c）通过在(b)中的模块之间共享相同输出的第一个`1x1`卷积，密集连接模块退化成了残差模块，虚线圈起来的部分就是残差模块。
- Figure2（d）是双路径体系结构（DPN），将同一层上ResNet的输出和DenseNet的输出按元素相加，再整体做卷积，然后将结果按原先通道分配的情况来分割又各分给残差模块和密集连接模块来连接，既有按元素相加又有通道相加，这就是DPN的双路径。
- Figure2（e）实际上和Figure2（d）是工程等价的，其中`~`表示分割操作，`+`表示元素级加法。一个模块中先连接，然后整体做卷积，再分开，分开之后残差模块的和残差模块相连，密集连接模块和密集连接模块相连，连完之后得到新的特征层。这一块完成后再做些卷积、pooling等，然后继续作为下一个模块的输入。

**需要注意的一点是，上面的所有的`3x3`卷积都是分组卷积，所以DPN是ResNeXt和DenseNet的结合，而不是ResNet。**

# 一个有意思的类比
这篇论文还有一个有意思的类比，即**ResNet VS RNN** 和 **DenseNet VS HORNN**。
- **ResNet VS RNN。** ResNet可以促进特征复用，减少特征冗余，因为ResNet可以通过跳跃连接获得没有信息冗余的直连映射部分，然后对冗余的信息进行信息提取和过滤，提取出有用的信息就是残差，这其实和RNN的有一点像，如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114211948576.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- **DenseNet VS HORNN**。DenseNet因为提前连接的特征又经过了卷积，所以可以学到新特征。这和HORNN比较类似，如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114212549637.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
因此DPN就有一种类似于RNN的特征重复使用以及探索新特征的功能，获得了性能提升。

# 实验

Table2是在ImageNet-1k数据集上和SOTA网络结构的对比结果，可以看到DPN的模型更小，GFLOPs和准确率方面都会更好。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114212928384.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)而Figure3是关于训练速度和参数量的对比，如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114213059898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
将DPN作为目标检测或者语义分割的Backbone也获得了性能提升：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200114213340353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 结论
DPN将ResNeXt和DenseNet相结合，使得网络对特征的利用更加充分，获得了比ResNeXt和DenseNet更好的效果，论文的思想是值得借鉴的，不过似乎工程上用的比较少，我猜测原因还是这个连接太复杂了吧。

# 附录
- 论文原文：https://arxiv.org/pdf/1707.01629.pdf
- 参考：https://blog.csdn.net/u014380165/article/details/75676216
- Mxnet代码：https://github.com/cypw/DPNs

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

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)