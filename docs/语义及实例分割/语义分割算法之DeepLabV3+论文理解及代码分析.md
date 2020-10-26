# 前言
之前讲了DeepLabV1,V2,V3三个算法，DeepLab系列语义分割还剩下最后一个DeepLabV3+，以后有没有++,+++现在还不清楚，我们先来解读一下这篇论文并分析一下源码吧。论文地址：https://arxiv.org/pdf/1802.02611.pdf
# 背景
语义分割主要面临两个问题，第一是物体的多尺度问题，第二是DCNN的多次下采样会造成特征图分辨率变小，导致预测精度降低，边界信息丢失。DeepLab V3设计的ASPP模块较好的解决了第一个问题，而这里要介绍的DeepLabv3+则主要是为了解决第2个问题的。
我们知道从DeepLabV1系列引入空洞卷积开始，我们就一直在解决第2个问题呀，为什么现在还有问题呢？
我们考虑一下前面的代码解析推文的DeepLab系列网络的代码实现，地址如下：https://mp.weixin.qq.com/s/0dS0Isj2oCo_CF7p4riSCA 。对于DeepLabV3，如果Backbone为ResNet101，Stride=16将造成后面9层的特征图变大，后面9层的计算量变为原来的4倍大。而如果采用Stride=8，则后面78层的计算量都会变得很大。这就造成了DeepLabV3如果应用在大分辨率图像时非常耗时。所以为了改善这个缺点，DeepLabV3+来了。
# 算法原理
DeepLabV3+主要有两个创新点。
## 编解码器
为了解决上面提到的DeepLabV3在分辨率图像的耗时过多的问题，DeepLabV3+在DeepLabV3的基础上加入了编码器。具体操作见论文中的下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118170543614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)其中，(a)代表SPP结构，其中的8x是直接双线性插值操作，不用参与训练。(b)是编解码器，融集合了高层和低层信息。(c)是DeepLabv3+采取的结构。

我们来看一下DeepLabV3+的完整网络结构来更好的理解这点：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118170833600.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
对于编码器部分，实际上就是DeepLabV3网络。首先选一个低层级的feature用1 * 1的卷积进行通道压缩（原本为256通道，或者512通道），目的是减少低层级的比重。论文认为编码器得到的feature具有更丰富的信息，所以编码器的feature应该有更高的比重。 这样做有利于训练。 
对于解码器部分，直接将编码器的输出上采样4倍，使其分辨率和低层级的feature一致。举个例子，如果采用resnet `conv2` 输出的feature，则这里要$\times 4$上采样。将两种feature连接后，再进行一次$3 \times 3$的卷积（细化作用），然后再次上采样就得到了像素级的预测。

实验结果表明，这种结构在Stride=16时有很高的精度速度又很快。stride=8相对来说只获得了一点点精度的提升，但增加了很多的计算量。 

## 更改主干网络

论文受到近期MSRA组在Xception上改进工作可变形卷积([Deformable-ConvNets](https://arxiv.org/pdf/1703.06211.pdf))启发，Deformable-ConvNets对Xception做了改进，能够进一步提升模型学习能力，新的结构如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118173622497.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
最终，论文使用了如下的改进：
- 更深的Xception结构，不同的地方在于不修改entry flow network的结构，为了快速计算和有效的使用内存
- 所有的max pooling结构被stride=2的深度可分离卷积代替
- 每个3x3的depthwise convolution都跟BN和Relu

最后将改进后的Xception作为encodet主干网络，替换原本DeepLabv3的ResNet101。
# 实验
论文使用modified aligned Xception改进后的ResNet-101，在ImageNet-1K上做预训练，通过扩张卷积做密集的特征提取。采用DeepLabv3的训练方式(poly学习策略，crop$513\times 513$)。注意在decoder模块同样包含BN层。


## 使用1*1卷积少来自低级feature的通道数
上面提到过，为了评估在低级特征使用1*1卷积降维到固定维度的性能，做了如下对比实验：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118174623614.png)
实验中取了$Conv2$尺度为$[3\times3,256]$的输出，降维后的通道数在32和48之间最佳，最终选择了48。
## 使用3*3卷积逐步获取分割结果
编解码特征图融合后经过了$3\times 3$卷积，论文探索了这个卷积的不同结构对结果的影响：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118174911940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

最终，选择了使用两组$3\times 3$卷积。这个表格的最后一项代表实验了如果使用$Conv2$和$Conv3$同时预测，$Conv2$上采样2倍后与$Conv3$结合，再上采样2倍的结果对比，这并没有提升显著的提升性能，考虑到计算资源的限制，论文最终采样简单的decoder方案，即我们看到的DeepLabV+的网络结构图。

# Backbone为ResNet101
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118175825113.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# Backbone为Xception
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118175946170.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
这里可以看到使用深度分离卷积可以显著降低计算消耗。

与其他先进模型在VOC12的测试集上对比：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118180240598.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)在目标边界上的提升，使用trimap实验模型在分割边界的准确度。计算边界周围扩展频带(称为trimap)内的mIoU。结果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191118180300853.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
与双线性上采样相比，加decoder的有明显的提升。trimap越小效果越明显。Fig5右图可视化了结果。

# 结论
论文提出的DeepLabv3+是encoder-decoder架构，其中encoder架构采用Deeplabv3，decoder采用一个简单却有效的模块用于恢复目标边界细节。并可使用空洞卷积在指定计算资源下控制feature的分辨率。论文探索了Xception和深度分离卷积在模型上的使用，进一步提高模型的速度和性能。模型在VOC2012上获得了SOAT。Google出品，必出精品，这网络真的牛。

# 代码实现

结合一下网络结构图还有我上次的代码解析[点这里](https://mp.weixin.qq.com/s/0dS0Isj2oCo_CF7p4riSCA) 看，就很容易了。
```python

from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deeplabv3 import _ASPP
from .resnet import _ConvBnReLU, _ResLayer, _Stem


class DeepLabV3Plus(nn.Module):
    """
    DeepLab v3+: Dilated ResNet with multi-grid + improved ASPP + decoder
    """

    def __init__(self, n_classes, n_blocks, atrous_rates, multi_grids, output_stride):
        super(DeepLabV3Plus, self).__init__()

        # Stride and dilation
        if output_stride == 8:
            s = [1, 2, 1, 1]
            d = [1, 1, 2, 4]
        elif output_stride == 16:
            s = [1, 2, 2, 1]
            d = [1, 1, 1, 2]

        # Encoder
        ch = [64 * 2 ** p for p in range(6)]
        self.layer1 = _Stem(ch[0])
        self.layer2 = _ResLayer(n_blocks[0], ch[0], ch[2], s[0], d[0])
        self.layer3 = _ResLayer(n_blocks[1], ch[2], ch[3], s[1], d[1])
        self.layer4 = _ResLayer(n_blocks[2], ch[3], ch[4], s[2], d[2])
        self.layer5 = _ResLayer(n_blocks[3], ch[4], ch[5], s[3], d[3], multi_grids)
        self.aspp = _ASPP(ch[5], 256, atrous_rates)
        concat_ch = 256 * (len(atrous_rates) + 2)
        self.add_module("fc1", _ConvBnReLU(concat_ch, 256, 1, 1, 0, 1))

        # Decoder
        self.reduce = _ConvBnReLU(256, 48, 1, 1, 0, 1)
        self.fc2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv1", _ConvBnReLU(304, 256, 3, 1, 1, 1)),
                    ("conv2", _ConvBnReLU(256, 256, 3, 1, 1, 1)),
                    ("conv3", nn.Conv2d(256, n_classes, kernel_size=1)),
                ]
            )
        )

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h_ = self.reduce(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.layer5(h)
        h = self.aspp(h)
        h = self.fc1(h)
        h = F.interpolate(h, size=h_.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat((h, h_), dim=1)
        h = self.fc2(h)
        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        return h


if __name__ == "__main__":
    model = DeepLabV3Plus(
        n_classes=21,
        n_blocks=[3, 4, 23, 3],
        atrous_rates=[6, 12, 18],
        multi_grids=[1, 2, 4],
        output_stride=16,
    )
    model.eval()
    image = torch.randn(1, 3, 513, 513)

    print(model)
    print("input:", image.shape)
print("output:", model(image).shape)
```

# 参考文章
https://blog.csdn.net/u011974639/article/details/79518175

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPadaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS01M2E3NWVmOTQ2YjA0OTE3LnBuZw?x-oss-process=image/format,png)