# 1. 常规卷积

![常规卷积](https://img-blog.csdnimg.cn/20200422202503445.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

相信接触过卷积神经网络的都知道常规卷积的操作

我们通过$N$个$D_k\times D_k$大小的卷积核

卷积出来的结果为$D_g \times D_g \times N$

现在我们来计算一下常规卷积的计算开销（以最简单的`stride`（步长）为$1$的情况进行讨论）

![卷积层每一步操作的开销](https://img-blog.csdnimg.cn/20200422202638672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

一个卷积核做一次卷积操作需要的开销为$D_k\times D_k\times M$，这里$D_k$是卷积核大小，$M$则为通道数。

一个卷积核完整地卷积完一次所需开销为$D_g\times D_g\times D_k\times D_k \times M$，这里$D_g$指的是卷积完成后的特征图长宽。

我们一共使用$N$个这样的卷积核进行卷积计算，因此最后总的计算开销为：

$N \times D_g \times D_g \times D_k \times D_k\times M$

# 2. Depthwise卷积
下面我们来看以下Depthwise卷积的步骤

在常规的卷积中，我们的卷积核都对每个通道进行了一次计算

而Depthwise卷积只对一个通道进行卷积计算，因此Depthwise卷积是不会改变原始特征图的通道数目


![深度可分离卷积的Depthwise卷积步骤](https://img-blog.csdnimg.cn/20200422203954747.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

原始特征图大小为$D_f\times D_f\times M$

我们需要用相同通道数目个卷积核进行卷积

换句话说就是用$M$个$D_k\times D_k \times 1$的卷积核分别对一个通道进行卷积计算

![M个Dk*DK*1的卷积核分别对一个通道进行卷积计算](https://img-blog.csdnimg.cn/20200422204208946.png)

# 3. Pointwise卷积
上面的Depthwise卷积有一个问题是，它只让卷积核单独对一个通道进行计算，但是各个通道之间的信息并没有达到交换，从而在网络后续信息流动中会损失通道之间的信息
因此我们加入一个Pointwise操作，来进一步融合通道之间的信息这个操作也十分简单，就是常规的$1\times 1$卷积核，形如下图：

![1x1卷积核](https://img-blog.csdnimg.cn/20200422204459285.png)

通过这样一个$1\times 1$卷积核，我们能在尽可能减少计算量的情况下，加强通道之间信息的交互

接下来我们继续推导下Pointwise卷积核这部分的计算量

假设一共有$N$个$1\times 1$的卷积核

![Pointwise卷积示意图](https://img-blog.csdnimg.cn/20200422204620231.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

原始特征图为$D_g \times D_g \times M$

一个卷积核计算开销为：$M$

一个卷积核卷积完原始特征图的计算开销为：$M\times D_g\times D_g$

我们一共使用$N$个卷积核，所以总开销为：

$N\times M\times D_g\times D_g$

最后我们来看一下Depthwise和Pointwise这两部分的计算量：

![Depthwise和Pointwise这两部分的计算量](https://img-blog.csdnimg.cn/20200422205008420.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 4. 和传统卷积相比

![和传统卷积相比，深度可分离卷积的计算量减少了N倍](https://img-blog.csdnimg.cn/20200422205045649.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到Depthwise + Pointwise卷积的计算量相较于传统卷积减少了$N$倍！

在达到相同目的（即对相邻元素以及通道之间信息进行计算）下，Dpethwise + Pointwise能极大减少卷积计算量，这也导致了大量移动端网络上都采用了这种卷积结构，再加上模型蒸馏，剪枝，能让移动端更高效的计算，推理。


# 5. 一个延伸
另外华为最近出了一篇GhostNet文章，对移动端网络感兴趣的可以去看看，该文章同时结合了常规卷积和Depthwise卷积，通过普通卷积得出一系列特征图，再通过Depthwise方式在这一系列特征图得出另外一部分特征图，最后**concate**到一起：

![GhostNet的模块](https://img-blog.csdnimg.cn/20200422205249395.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 代码实现
部分框架有提供Depthwise卷积API，这里就不再赘述了。

其他框架可以参考下面我的PaddlePaddle框架代码来实现：

```python
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, BatchNorm


class DepthwiseConv(fluid.dygraph.Layer):
    """
    通过分组卷积实现Depthwise卷积
    卷积前后通道数不变
    """

    def __init__(self, size, stride):
        """
        :param inputs: 输入张量
        :param size: 卷积核大小
        """
        super(DepthwiseConv, self).__init__()
        self.size = size
        self.stride = stride
        self.padding = (int(self.size - self.stride) + 1) // 2

    def forward(self, inputs):
        channels = inputs.shape[1]  # 获取输入通道数
        # print(channels)
        depthwise = Conv2D(num_channels=channels, filter_size=self.size, stride=self.stride,
                           padding=(self.padding, self.padding), num_filters=channels)
        out = depthwise(inputs)
        return out
```

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)