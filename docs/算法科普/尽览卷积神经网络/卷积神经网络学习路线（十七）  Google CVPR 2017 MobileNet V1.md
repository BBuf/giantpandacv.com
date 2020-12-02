# 前言
这是卷积神经网络的第十七篇文章，Google 2017年发表的MobileNet V1，其核心思想就是提出了深度可分离卷积来代替标准卷积，将标准卷积分成Depthwise+Pointwise两部分，来构建轻量级的深层神经网络，这一网络即使现在也是非常常用的。论文原文地址和代码实现见附录。

# 网络结构
在本节中，论文首先描述MobileNet V1的核心层----深度可分离卷积层。然后介绍MobileNet V1网络结构，并以两个模型缩小参数（`Width Multiplier`和`Resolution Multiplier`）作为总结。
## 深度可分离卷积
MobileNet V1模型基于深度可分离卷积，它是`factorized convolutions`的一种，而`factorized convolutions`将标准化卷积分解为深度卷积和$1\times 1$卷积（`pointwise convolution`）。对于MobileNet V1，深度卷积将单个滤波器应用到每一个输入通道。然后，点卷积用$1\times 1$卷积来组合深度卷积的输出。我们知道，标准卷积是直接将输入通过卷积的方式组合成一组新的输出。而深度可分离卷积则将其分成两层，一层用于卷积，一层用于组合。这种分解过程能极大减少计算量和模型大小。Figure 2展示了如何将一个标准卷积分解为深度卷积和$1\times 1$的点卷积。

![这里写图片描述](https://img-blog.csdn.net/20180411204344224?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

**标准卷积**使用一个和输入数据通道数相同的卷积核执行逐个通道卷积后求和获得一个通道的输出，计算量为$M\times D_k \times D_k$，其中$M$代表输入的通道数，$D_k$为卷积核的宽和高，一个卷积核处理输入数据时的计算量为：
$D_k \times D_k \times M \times D_F \times D_F$
其中$D_F$为输入的宽和高，如果在某一个卷积层使用$N$个卷积核，那么这一卷积层的计算量为：
$D_k \times D_k \times N \times M \times D_F \times D_F$

**深度可分离卷积**首先使用一组**通道数**为$1$的卷积核，每次只处理一个输入通道，并且这一组卷积核的**个数**是和输入通道数相同的。执行完上面的深度卷积后，再使用通道数为输入数据通道数大小的的$1\times 1$卷积来组合之前输出的特征图，将最终输出通道数变为一个指定的数量。

从理论上来看，一组和输入通道数相同的2D卷积核（通道数为$1$，即深度卷积或者说分组卷积）的运算量为：

$D_k\times D_k\times M \times D_F \times D_F$

而3D（标准卷积）的$1\times 1$卷积核的运算量为：

$N\times M \times D_F \times D_F$

因此这种组合方式的计算量为：

$D_k \times D_k \times M \times D_F \times D_F + N \times M \times D_F \times D_F$

因此，深度可分离卷积相比于标准卷积计算量的比例为：

$\frac{D_k \times D_k \times M \times D_F \times D_F + N \times M \times D_F \times D_F}{D_k\times D_k\times M \times D_F \times D_F}=$
$\frac{1}{N}+\frac{1}{D_k^2}$

然后举个例子，给定三个通道的$224 \times 224$的图像，VGG16网络的第$3$个卷积层$conv2_1$输入的是尺寸为$112\times 112$的特征图，通道数为$64$，卷积核尺寸为$3\times 3$，卷积核个数为$128$，标准卷积计算量按照上面的公式计算就是：

$3\times 3\times 128\times 64\times 112\times 112=924844032$

如果采用深度可分离卷积，计算量为：

$3\times \times 64\times 112\times 112 + 128\times 64\times 112\times 112=109985792$

这两者计算量的比值为：$\frac{109985792}{924844032}=0.1189$

可以看到将一个标准卷积换成深度可分离卷积之后模型的计算量减少了$9$倍。

## MobileNet V1 网络结构

MobileNet网络结构是以深度可分离卷积为基础单元建立的，其中第一层是标准卷积。MobileNet V1的完整网络结构定义在Table1中。所有卷积层后面都接了BN层和ReLU非线性激活层，但最后的全连接层例外，它没有非线性激活函数，是直接馈送到`Softmax`层进行分类。在Figure 3还比较了标准卷积和深度可分离卷积（都跟着BN层和ReLU层）的区别。深度可分离卷积和第一个标准卷积均可以处理下采样问题，并且在最后一个全连接层前面还接了一个全局平均池化层，将特征图的长宽变成$1\times 1$。如果将`Depthwise`卷积层和`Pointwise`卷积层算成不同层的话，MobileNet V1一共有`28`层。

![这里写图片描述](https://img-blog.csdn.net/2018041209413749?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

![这里写图片描述](https://img-blog.csdn.net/20180412094327244?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

当然，仅仅通过减少网络乘加操作对于网络加速是不够的，确保这些操作能够有效实现也很重要。例如，非结构化的稀疏矩阵操作通常不比密集矩阵运算快，除非是非常稀疏的矩阵。而MobileNet V1的模型结构几乎将全部的计算复杂度放到了$1\times 1$卷积中。这可以通过高度优化的通用矩阵乘法（GEMM）来实现。通常标准卷积如$3\times 3$卷积，$5\times 5$卷积由GEMM算法实现，但需要使用`im2col`在内存中对数据进行重排，以将其映射到GEMM可以使用的方式。这个方法在流行的`Caffe`深度学习框架中正在使用。$1\times 1$的卷积不需要在内存中重排就可以直接被GEMM（最优化的数值线性代数算法之一）实现。MobileNet V1的$1\times 1$卷积占了总计算量的`95%`，并且也占了`75%`的参数（见Table 2）。而几乎所有的额外参数都在全连接层。

![这里写图片描述](https://img-blog.csdn.net/20180412095057321?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 模型瘦身因子
虽然MobileNet V1网络结构已经很小并且高效，但特定场景或应用程序中可能需要更小更快的模型。为了构建这些较小且计算量较少的模型，论文引入了一个非常简单的参数$\alpha$，称为`Width multiplier`。这个参数的作用是在每层均匀地缩减网络宽度。对于一个给定的卷积层和$\alpha$，输入通道的数量从$M$变成$\alpha M$，输出通道的数量从$N$变成$\alpha N$。深度可分离卷积的计算复杂度变成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200119211536224.png)

其中$\alpha \in (0, 1]$，通常设为$1，0.75，0.5和0.25$。$\alpha = 1$表示基准MobileNet V1，而$\alpha < 1$则表示瘦身的MobileNet V1。$\alpha$有减少计算复杂度和参数数量（大概$\alpha ^ 2$倍）的作用。并且`Width Multiplier`参数可以应用于任何模型结构，以定义一个具有可接受准确率，速度和大小的新模型。

##  分辨率缩减因子
这里要介绍的第二个超参数是分辨率缩减因子$\rho$，又叫`Resolution multiplier`。我们将其应用于输入图像，并且每个层的特征图分辨率随之被减去相同的$\rho$倍。实际操作的时候，我们通过输入分辨率隐式的设置$\rho$。$\rho=1$是最基础的MobileNet V1模型，而$\rho < 1$示瘦身的MobileNet V1。`Resolution multiplier`同样可以减少$\rho^2$的计算复杂度。


我们以MobileNet V1中的一个典型的卷积层为例，看看深度可分离卷积，`Width Multiplier`和`Resolution Multiplier`具体降低了多少计算复杂度和参数量。Table 3展示了将上述Trick应用于该层，得到的计算复杂度(FLPOs)和参数数量(Params)对比结果。

![这里写图片描述](https://img-blog.csdn.net/2018041209592969?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70) 

# 实验结果
Table4-7展示了上面介绍的一系列参数和深度可分离卷积在ImageNet上的精度以及模型的参数量/计算量 的对比结果。

![这里写图片描述](https://img-blog.csdn.net/2018041210093457?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

下面的Figure 4展示了在ImageNet，$16$个模型的准确率（$\alpha \in(1,0.75,0.5,0.25)$，$\rho \in (224,192,160,128)$）和计算复杂度的关系。结果近似为`log`关系，并在$α=0.25$有一个显著降低。

![这里写图片描述](https://img-blog.csdn.net/2018041210151239?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Figure 5展示了在ImageNet上，$16$个模型的准确率($\alpha \in(1,0.75,0.5,0.25)$，$\rho \in (224,192,160,128)$)和参数数量的关系。不同颜色代表不同的分辨率。参数数量不会因输入分辨率不同而不同。

![这里写图片描述](https://img-blog.csdn.net/20180412101602739?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Table 8比较了基准的的MobileNet V1和的GoogleNet以及VGG16的一些指标。可以看到MobileNet V1和VGG16准确率相近，但模型大小小了32倍，计算量也小了$27$倍。MobileNet V1比GoogleNet更加精确，并且模型大小更小，计算复杂度也少了$2.5$倍。

![这里写图片描述](https://img-blog.csdn.net/20180412101827513?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

Table 9比较了瘦身后的（`Width multiplier` $\alpha = 0.5$ ，输入分辨率$160\times 160$）的MobileNet V1。瘦身后的MobileNet V1准确率比AlexNet好大约`4%`，但模型大小小了`45`倍，计算复杂度也小了$9.4$倍。在模型尺寸差不多的情况下，它比`SqueezeNet`有更高的准确率，而计算复杂度降低了$22$倍。

MobileNet V1还可以作为Backbone网络应用在现代目标检测系统中。根据2016 COCO冠军的工作，论文报告了针对COCO数据集使用MobileNet V1做Backbone进行目标检测的结果。从Table 13中可以看到，使用相同Faster-RCNN和SSD框架，MobileNet V1和VGG，InceptionV2分别做检测模型的Backbone进行了比较。其中，**SSD**使用了$300 \times 300$的输入分辨率（SSD300），而Faster-RCNN使用了$600$和$300$的分辨率（**Faster-RCNN300,Faster-RCNN600**）。这个Faster-RCNN模型每张图考虑$300$个候选框。模型通过COCO数据集上训练（排除8000张minival图），然后在**minival**测试集上测试。对于这两个框架，MobileNet V1可以实现与其他Backbone网络的相似结果，但计算复杂度和模型大小都小了非常多。Figure6展示了一个使用SSD做模板检测的可视化例子。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200119214517461.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

FaceNet模型是最先进的人脸识别模型。它基于**TripletLoss**构建人脸特征。为了构建一个移动端的FaceNet模型，论文使用`Distillation`方法去训练MobileNet FaceNet模型，实验结果可以在Table 14中看到。

![这里写图片描述](https://img-blog.csdn.net/20180412103215806?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

# 结论
这篇论文提出了一种基于深度可分离卷积的新型模型架构，称为MobileNet V1。接着讨论了一些可以得到高效模型的重要设计方法并展示了如何通过使用`Width Multiplier`和`Resolution Multiplier`来构建更小更快的MobileNet V1模型，以合理的精度损失来减少模型尺寸和提高模型运行速度。最后，将不同的MobileNet V1结构与SOTA的网络模型进行比较，展示了尺寸，速度和精度方面的差异。

# Pytorch代码实现

```cpp
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
```


# 附录

- 论文原文：https://arxiv.org/pdf/1704.04861.pdf
- 参考：https://baijiahao.baidu.com/s?id=1566004753349359&wfr=spider&for=pc
- 代码实现：https://github.com/kuangliu/pytorch-cifar

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

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)