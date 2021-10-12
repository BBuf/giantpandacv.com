# 0. 简介

![论文题目&作者团队](https://img-blog.csdnimg.cn/20200608205124788.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

今天给大家解读的是最近一篇关于Deepfake检测的论文，出自阿里巴巴，小米AI lab联合出品的FakeSpotter，由于以往的Deepfake检测网络鲁棒性并不高，这篇文章探讨一个更简单的方式，增强网络的鲁棒性。


# 1. 前言
近些年来，各种各样的GAN网络在图片生成上取得巨大成功，然后现有的检测器还不足以完全面对GAN网络的挑战。本文提出了一种**基于人类神经行为来辨别真假人脸**的模型，我们推测每一层神经元激活函数可能提取到更多微小的特征，而这些特征对于真假人脸识别是十分重要的。后续我们也在4款SOTA GAN模型下做了实验，验证了我们的猜想。

# 2. 方法
## 2.1 Insight

神经元覆盖技术被广泛应用于传统DNN的内部行为，当**输出值大于阈值的激活神经元**，被激活的神经元将作为输入的**另一种形式**，将学习的内容一层层保存在网络中

而前人也有一些工作针对**关键的激活神经层**用于检测对抗例子。

我们工作的灵感来自分层激活的神经元，它捕捉输入的微妙特征，可以用来寻找真实和合成的面部图像之间的差异。


## 2.2 模拟神经元行为

![FakeSpotter检测框架](https://img-blog.csdnimg.cn/20200608205452965.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

上图是FakeSpotter检测框架，与传统框架不同的是**根据每层神经元激活特性**来进行人脸分析。

在传统DNN中，每一层神经元是否被激活**取决于他的输出值是否高于阈值Threshold**

我们提出了一种确立阈值的策略，公式如下

![本文提出的确立阈值的策略](https://img-blog.csdnimg.cn/20200608205544666.png)

分式上面是**各个神经元输出值之和**

|N|代表**当前层神经元的总数目**

|T|代表**当前层输入的个数**

最后**通过这个阈值来决定这个神经元是否被激活**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200608205734926.png)

下图是描述这两种策略的算法

![Algorithm 1](https://img-blog.csdnimg.cn/20200608205811582.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 3. 其它实现细节

- 优化器是**动量为0.9的SGD**，起始学习率为0.0001。
- 损失函数采用**二分类交叉熵**损失binary cross-entropy。
- CNN架构采用的是Vgg-Face，将骨干网络替换为**ResNet50**，并且带有我们的**MNC策略**。
- 设计了**五层全连接网络**来作为最后的二分类网络。


# 4. 实验表现
我们通过压缩，模糊，缩放，加噪声来评价模型的鲁棒性。

![通过压缩，模糊，缩放，加噪声来评价模型的鲁棒性](https://img-blog.csdnimg.cn/20200608205910344.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到在检测DFDC这个数据集上表现并不是那么好。

因为该数据集有**人脸替换，声音替换**这两种类型，而声音替换是超过FakeSpotter这种基于图像的检测框架范围内了。


![其它检测模型在Cele-DFv2数据集上加入前面提到的四种操作的表现](https://img-blog.csdnimg.cn/2020060821005681.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到FakeSpotter仍然保持较好的检测率。


# 5. 总结
在DeepFake检测领域中，关键一个问题是**模型的鲁棒性**，一个训练好的模型可能换到另外一个数据集就失效了。 该工作受DNN神经元激活层的启发，将激活层输出值，平均到每个神经元上，作为一个阈值加入到整体网络进行训练，而最后在多个模型实验下，也表明这种**基于激活层阈值策略**，能得到更多细微特征，进一步提高模型鲁棒性。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)