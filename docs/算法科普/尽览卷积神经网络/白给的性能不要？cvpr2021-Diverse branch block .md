- 论文地址：[Diverse Branch Block: Building a Convolution as an Inception-like Unit](https://arxiv.org/abs/2103.13425)
- 官方代码：[DingXiaoH/DiverseBranchBlock](https://github.com/DingXiaoH/DiverseBranchBlock)

# 引言

本文是继前作ACNet的又一次对网络结构重参数化的探索，我们设计了一个类似Inception的模块，**以多分支的结构丰富卷积块的特征空间**，各分支结构包括平均池化，多尺度卷积等。最后在推理阶段前，**把多分支结构中进行重参数化，融合成一个主分支**。这样能在相同的推理速度下，“白嫖”模型性能

![DBB结构转换](https://img-blog.csdnimg.cn/20210329105732434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

# 卷积的性质

常规卷积核本质上也是一个张量，其形状为（输出通道数，输入通道数，卷积核大小，卷积核大小）
$$
C_{out}，C_{in}，KernelSize, KernelSize
$$
而卷积操作本质上也是一个线性操作，因此卷积在某些情况下具备一些线性的性质

## 可加性

可加性即在两个**卷积核形状一致**的情况下，卷积结果满足可加性
即
$$
Input \otimes F_1 + Input \otimes F_2 = Input \otimes (F_1+F_2)
$$
其中 $F_1$ 和 $F_2$ 分别表示两个独立的卷积操作

## 同质性

即
$$
Input \otimes (p*F_1) =p* (Input \otimes F_1)
$$
后续我们针对多分支结构的转换都是基于这两种基本性质来操作的

![论文提及的6种转换](https://img-blog.csdnimg.cn/20210329111340501.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

# 转换1： Conv-BN融合

在CNN中，卷积层和BN层经常是成对出现的，我们可以把BN的参数融入到卷积层里（**这里偷懒，直接复制粘贴以前RepVGG写的推导了**）
卷积层公式为
$$
Conv(x) = W(x)+b
$$
BN层公式为
$$
BN(x) = \gamma*\frac{(x-mean)}{\sqrt{var}}+\beta
$$
将卷积层结果带入到BN公式中
$$
BN(Conv(x)) = \gamma*\frac{W(x)+b-mean}{\sqrt{var}}+\beta
$$
化简为
$$
BN(Conv(x)) = \frac{\gamma*W(x)}{\sqrt{var}}+(\frac{\gamma*(b-mean)}{\sqrt{var}}+\beta)
$$
这其实就是一个卷积层，只不过权重考虑了BN的参数
令
$$
W_{fused}=\frac{\gamma*W}{\sqrt{var}} \\
B_{fused}=\frac{\gamma*(b-mean)}{\sqrt{var}}+\beta
$$
融合的结果就是
$$
BN(Conv(x)) = W_{fused}(x)+B_{fused}
$$

# 转换2 分支相加

这就利用到我们前面讲的卷积可加性，这也比较好理解，我们可以看一段基于oneflow框架的验证代码

```python
import oneflow as flow 
import oneflow.typing as tp 
import numpy as np 
from typing import Tuple


@flow.global_function()
def conv_add(x: tp.Numpy.Placeholder(shape=(1, 2, 4, 4)))->Tuple[tp.Numpy, tp.Numpy]: 
    conv1 = flow.layers.conv2d(x, 4, kernel_size=3, padding="SAME", name="conv1")
    conv2 = flow.layers.conv2d(x, 4, kernel_size=3, padding="SAME", name="conv2")
    # Merge Add
    conv_merge_add = flow.layers.conv2d(x, 4, kernel_size=3, padding="SAME", name="conv_merge_add")
    return conv1 + conv2, conv_merge_add

x = np.ones(shape=(1, 2, 4, 4)).astype(np.float32)
weight_1 = np.random.randn(4, 2, 3, 3).astype(np.float32)
weight_2 = np.random.randn(4, 2, 3, 3).astype(np.float32)

# Load numpy weight
flow.load_variables({"conv1-weight": weight_1, "conv2-weight": weight_2, "conv_merge_add-weight": weight_1+weight_2})

original_conv_add, merge_conv_add = conv_add(x)

print("Conv1 + Conv2 output is: ", original_conv_add)
print("Merge Add output is: ", merge_conv_add)
print("Is Match: ", np.allclose(original_conv_add, merge_conv_add, atol=1e-5))
```

这里我们定义了一个方法，conv1和conv2分别表示两个独立的卷积操作，最后相加返回。而conv3表示的是融合后的卷积操作。 定义好后，我们将设定好的权重分别导入给conv1和conv2，然后将相加后的权重，导入给conv3，最后用`np.allclose`来验证结果是否准确

# 转换3 序列卷积融合

在网络设计中，我们也会用到1x1卷积接3x3卷积这种设计（如ResNet的BottleNeck块），它能调整通道数，减少一定的参数量。

其原始公式如下
$$
F_1(D,C,1,1) \\
F_2(E,D,K,K) \\
Out=F2 \otimes (F1 \otimes Input)
$$
我们假设输入是一个三通道的图片，1x1卷积的输出通道为2，3x3卷积的输出通道为4，那么图示如下

![1x1接3x3](https://img-blog.csdnimg.cn/20210329112155937.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

作者提出了这么一个转换方法，**首先将1x1卷积核的第零维和第一维互相调换位置**
$$
Transpose(F_1):F_1(D, C, 1, 1) ->F_1(C, D, 1, 1)
$$
然后3x3卷积核权重与转置后的"1x1卷积核"进行卷积操作
$$
F_2 \otimes Transpose(F_1) 形状为(E, C, K, K)
$$

![1x1和KxK卷积转换](https://img-blog.csdnimg.cn/2021032911375140.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

最后输入与其做卷积操作，整个流程可以写为
$$
Input \otimes F_2 \otimes (Transpose(F_1))
$$
这里我也简单写了一个测试代码

```python
import oneflow as flow 
import oneflow.typing as tp 
import numpy as np 
from typing import Tuple 


@flow.global_function()
def conv2d_Job(x: tp.Numpy.Placeholder((1, 3, 4, 4))) -> Tuple[tp.Numpy, tp.Numpy]:
    weight_1x1 = flow.get_variable(
        name="weight1x1",
        shape=[2, 3, 1, 1], # [O_c, I_c, ksize, ksize]
        initializer=flow.ones_initializer(),
    )
    weight_3x3 = flow.get_variable(
        name="weight3x3",
        shape=[4, 2, 3, 3], # [O_c, I_c, ksize, ksize]
        initializer=flow.ones_initializer(),
    )

    conv_1x1 = flow.nn.conv2d(x, weight_1x1, strides=1, padding=(0, 0, 0, 0), name="conv1x1")
    conv_1x1_3x3 = flow.nn.conv2d(conv_1x1, weight_3x3, strides=1, padding=(0, 0, 0, 0), name="conv3x3")
    
    weight_1x1_transposed = flow.transpose(weight_1x1, [1, 0, 2, 3]) # [2, 3, 1, 1] -> [3, 2, 1, 1]
    weight_merge = flow.nn.conv2d(weight_3x3, weight_1x1_transposed, strides=1, padding=(0, 0, 0, 0), name="weight_merge") # [4, 3, 3, 3]
    conv_merge = flow.nn.conv2d(x, weight_merge, strides=1, padding=(0, 0, 0, 0), name="conv_merge")

    return conv_1x1_3x3, conv_merge

x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
weight_1x1 = np.random.randn(2, 3, 1, 1).astype(np.float32)
weight_3x3 = np.random.randn(4, 2, 3, 3).astype(np.float32)

# Load numpy weight
flow.load_variables({"weight1x1": weight_1x1, "weight3x3": weight_3x3})

conv1x1_3x3, conv_merge = conv2d_Job(x)

print("Conv 1x1 and 3x3 is: ", conv1x1_3x3)
print("Merge Conv: ", conv_merge)

print("Is Match: ", np.allclose(conv1x1_3x3, conv_merge, atol=1e-5))
```

# 转换4 拼接融合

在Inception模块中，我们经常会用到的一个操作就是concat，将各分支的特征，在通道维上进行拼接。

我们也可以将多个卷积拼接转换为一个卷积操作，**只需要将多个卷积核权重在输出通道维度上进行拼接即可**，下面是一个示例代码

```python
import oneflow as flow 
import oneflow.typing as tp 
import numpy as np 
from typing import Tuple


@flow.global_function()
def conv_concat(x: tp.Numpy.Placeholder(shape=(1, 1, 4, 4)))->Tuple[tp.Numpy, tp.Numpy]: 
    conv1 = flow.layers.conv2d(x, 2, kernel_size=3, padding="SAME", name="conv1")
    conv2 = flow.layers.conv2d(x, 2, kernel_size=3, padding="SAME", name="conv2")
    # Merge Concat
    conv_merge_concat = flow.layers.conv2d(x, 4, kernel_size=3, padding="SAME", name="conv_merge_concat")
    return flow.concat([conv1, conv2], axis=1), conv_merge_concat

x = np.ones(shape=(1, 1, 4, 4)).astype(np.float32)
weight_1 = np.random.randn(2, 1, 3, 3).astype(np.float32)
weight_2 = np.random.randn(2, 1, 3, 3).astype(np.float32)

flow.load_variables({"conv1-weight": weight_1, "conv2-weight": weight_2, "conv_merge_concat-weight": np.concatenate([weight_1, weight_2], axis=0)})

original_conv_concat, merge_conv_concat = conv_concat(x)

print("Conv1 concat Conv2 output is: ", original_conv_concat)
print("Merge Concat output is: ", merge_conv_concat)
print("Is Match: ", np.allclose(original_conv_concat, merge_conv_concat, atol=1e-5))
```

# 转换5 平均池化层转换

我们简单回顾一下平均池化层操作，它也是一个滑动窗口，对特征图进行滑动，将窗口内的元素求出均值。**与卷积层不一样的是，池化层是针对各个输入通道的（如Depthwise卷积），而卷积层会将所有输入通道的结果相加**。一个平均池化层的示意图如下：

![平均池化层](https://img-blog.csdnimg.cn/20210329114930491.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

那其实平均池化层是可以等价一个**固定权重**的卷积层，假设平均池化层窗口大小为3x3，那么我**可以设置3x3卷积层权重为 1/9**，滑动过去就是取平均。另外要注意的是卷积层会将所有输入通道结果相加，**所以我们需要对当前输入通道设置固定的权重，对其他通道权重设置为0**。

![卷积层替换平均池化层](https://img-blog.csdnimg.cn/2021032911530135.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

另外补充一下，由于**最大池化层是一个非线性的操作**，所以是不能用卷积层替换的
下面是测试代码：

```python
import oneflow as flow 
import oneflow.typing as tp 
import numpy as np 
from typing import Tuple

@flow.global_function()
def avg_pool(x: tp.Numpy.Placeholder(shape=(1, 3, 4, 4)))->Tuple[tp.Numpy, tp.Numpy]: 
    avg_pool_out = flow.nn.avg_pool2d(x, ksize=3, strides=1, padding=(0, 0, 0, 0))
    # Use conv to instead average pool
    conv_avg_pool = flow.layers.conv2d(x, 3, kernel_size=3, strides=1, name="conv_avg")
    return avg_pool_out, conv_avg_pool

x = np.ones(shape=(1, 3, 4, 4)).astype(np.float32)
weight = np.zeros(shape=(3, 3, 3, 3)).astype(np.float32)

for i in range(3): 
    weight[i, i, :, :] = 1 / 9 # Set 3x3 kernel weight value as 1/9

# Load numpy weight
flow.load_variables({"conv_avg-weight": weight})

avg_pool_out, conv_avg_pool = avg_pool(x)

print("Average Pool output is: ", avg_pool_out)
print("Conv Average Pool output is: ", conv_avg_pool)
print("Is Match: ", np.allclose(avg_pool_out, conv_avg_pool, atol=1e-5))
```

# 转换6 多尺度卷积融合

这部分其实就是ACNet的思想，存在一个卷积核
$$
kh × kw (kh ≤ K, kw ≤ K)
$$
那么我们可以把卷积核周围补0，来等效替代KxK卷积核
下面是一个示意图

![多尺度卷积融合](https://img-blog.csdnimg.cn/2021032911593567.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

# Diverse Branch Block结构

介绍完六种等价转换方式后，我们简单看下DBB结构

![DBB结构](https://img-blog.csdnimg.cn/2021032912011094.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

其中一共有四个分支，分别是

- 1x1 卷积分支
- 1x1 - KxK卷积分支
- 1x1 - 平均池化分支
- KxK 卷积分支
  启发于Inception模块，各操作有不同的感受野以及计算复杂度，能够极大丰富整个模块的特征空间

因为最后都可以等价转换为一个KxK卷积，作者后续实验就是将这个Block替换到骨干网络中的KxK卷积部分。

# 实验

![实验](https://img-blog.csdnimg.cn/20210329120452305.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

作者基于ImageNet上，和前作ACNet在相同的超参数，数据增广条件下进行了对比。可以看到比起ACNet还有一定程度的提升，反正最后都能融合，性能能白嫖一点是一点。

![消融实验](https://img-blog.csdnimg.cn/20210329120714615.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70)

作者也针对DBB模块的各个路径做了消融实验，可以看到每个分支都能对模型性能有一定的提升，最后集合起来性能最好。

# 总结

作者在CVPR2021的一系列网络重参数化工作解读也算画上了句号，他本人也在实验DBB模块和RepVGG结合会不会有更强的性能。个人感觉这篇文章实用性很大，能白嫖模型性能就白嫖。而且DBB模块的潜力还很大，作者提出的六种等价转换方法都有一定的普适性，说不定后续会有NAS搜索DBB模块的工作。作者这两篇重参数化工作RepVGG和Diverse Branch Block都做的十分好，涉及到的一些转换方法也能加深我们对卷积操作的理解，推荐各位去拜读一下~

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)