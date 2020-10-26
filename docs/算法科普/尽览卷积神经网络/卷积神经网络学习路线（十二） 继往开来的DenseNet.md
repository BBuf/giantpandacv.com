# 前言
这是卷积的第十二篇文章，主要为大家介绍一下DenseNet，值得一提的是DenseNet的作者也是上一篇[卷积神经网络学习路线（十一）| Stochastic Depth（随机深度网络）](https://mp.weixin.qq.com/s/3mndBm86qamoy4Gn5mBLfA)论文的作者，即清华的黄高。相比于里程碑式创新的ResNet来讲，DenseNet的作用或许用既往开来来形容是最合适不过了。论文原文地址见附录。

# 介绍
论文先讲到了先前的网络因为使用了`shortcut`连接，网络已经变得越来越深了。接着引入了论文要介绍的`DenseNet`，正是利用了`shortcut`连接的思想，每一层都将前面所有层的特征图作为输入，最后使用`concatenate`来聚合信息。实验显示，`DenseNet`减轻了梯度消失问题，增大了特征重用，大大减少了参数量。
`Figure 1`是`DenseNet`的一个组件（`dense block`），整个网络是由多个这种组件堆叠出来的。可以看到`DenseNet`使用了`concatenate`来聚合不同的特征图，类似于`ResNet`残差的思想，提高了网络的信息和梯度流动，使得网络更加容易训练。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802154601611.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
`Figure 2`展示了使用3个`dense block`搭建出来的`DenseNet`网络：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802154957498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 相关工作
- **通过级联来加深网络。**
	- 80年代的级联结构和DenseNet结构有点相似，但那时的主要目的是为了逐层训练多层感知机。
	- 最近，提出使用batch梯度下降训练全连接级联网络。虽然在小数据集上有效，但这个方法最多只适合有几百个参数的网络。
	- 何凯明等人提出的ResNet。

- **通过shortcut连接加深网络。**
	- Highway网络是第一个将网络深度做到100+的，使用了gating mapping。
	- ResNet在Highway的基础上，将gating mapping换成了identity mapping。
	- Stochastic depth ResNet通过随机dropout掉一些identity mapping来强制学习，这表明，ResNet中有很多冗余层，DenseNet就是受到这个启发来做的。
- **通过加宽网络来使网络更深。**
	- GoogleNet使用Inception模块加深了网络
	- WRN加宽了ResNet
	- FractalNet 也加宽了网络
- **提高特征重用。**
	- 相比于通过加深，加宽网络来增强表示能力，DenseNet关注特征重用。dense架构容易训练，并且参数更少。特征图谱通过 concat 聚合可以增加后面层输入的变化，提高效率。
	- Inception 系列网络中也有用`concatenate`来聚合信息，但DenseNet更加简单高效。

- **其他工作。**
	- NIN 将微型 mlp 结构引入 conv 来提取更加复杂的特征。
	- Deeply Supervised Network (DSN) 添加辅助 loss 来增强前层的梯度。
	- Ladder Networks 在 自动编码器 中引入了横向连接。
	- Deeply-Fused Nets (DFNs) 提高信息流。

# 实现方法
对于一个卷积神经网络，假设输入图像$x_0$。该网络包含L层，每一层都实现了一个非线性变换$H_i(.)$，其中$i$表示第$i$层。$H_i(.)$可以是一个组合操作，如`BN, ReLU, Conv`，将第$i$层的输出记作$x_i$。

## 稠密连接
为了进一步改善网络层之间的信息交流流，论文提出了不同的连接模式：即引入从任何层到所有后续层的直接连接。结果，第$i$层得到了之前所有层的特征映射$x_0,x_1,...,x_{i-1}$作为输入：$x_i=H_i([x_0,x_1,...,x_{i-1}])$，其中$[x_0,x_1,...,x_{i-1}]$表示特征映射的级联。


## 复合函数
定义$H_i(.)$为三个连续操作的组合，即：$BN+ReLU+Conv。$

## 池化层
`DenseNet`使用了$2\times 2$的平均池化做特征下采样。
## 增长率
当每个$H_i$都产生$k$个特征映射时，它表示第$i$层有$k_0+k*(i-1)$个输入特征，$k_0$表示输入层的通道数。`DenseNet`与已存在架构不同之处在于`DenseNet`可以有很窄的层，例如: $k = 12$。其中参数$k$称为网络的增长率。下表展示了不同深度的DenseNet网络结构，其中$k=32$：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802173405615.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
## 为什么DenseNet结构有效？

对此的一种解释是`DenseNet`中的每个层都可以访问对应块中所有前面的特征映射，因此可以访问网络的“集体知识”。我们可以将特征映射看作网络的全局状态。每个层将自己的$k$个特征映射添加到这个状态。增长速度控制着每一层新信息对全局状态的贡献。全局状态一旦写入，就可以从网络中的任何地方访问，并且与传统网络体系不同，不需要逐层复制它。
## 瓶颈层
虽然每一层只产生$k$个输出特征映射，但它通道具有更多的输入。有文章指出，在每个$3\times 3$卷积之前可以引入$1\times 1$卷积层作为瓶颈层，以减少输入特征映射的数量，从而提高计算效率。实验发现这种设计对于`DenseNet`特别有效，并将具有瓶颈层的网络称为`DenseNet-B`，即具有`BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)`组件的的$H_i$。
## 压缩
为了进一步提高模型的紧凑性，可以减少过度层上的特征映射的数量。如果一个`dense block`包含`m`个特征映射，可以让其紧跟着的变化层生成$\theta \times m$个输出特征映射，其中$0<\theta<=1$作为压缩因子。当$\theta=1$时，跨转换层的特征映射的数量保持不变。

# 实验
通过`Table 2`可以看出`DenseNet`在准确率和参数量上取得了较好的平衡，精度上全面超越`ResNet`网络。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802180049163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)`Figure 3`对比了`ResNet`和`DenseNet`参数量和`FLOPS`是如何影响测试错误率的，可以看出相同准确率时`DenseNet` 的参数更少，推理时的计算量也更小。 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802180134898.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
从`Figure 4`可以看出，在相同性能下`DenseNet`的参数量是`ResNet`的三分之一；`1001`层的`pre-activation ResNet`（参数为`10M`）的性能和`100`层的`DenseNet` （参数为`0.8M`）相当。说明`DenseNet`的参数利用效率更高。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190802180205336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 结论
这篇论文论文提出了一个新的网络结构`DenseNet`，解决了`ResNet`遗留的网络层冗余的问题，引入了具有相同特征映射大小的任意两个层之间的直接连接。我们发现，`DenseNet`可以自然地扩展到数百个层，且没有表现出优化困难。`DenseNet`趋向于随着参数量的增加，在精度上也产了对应的提高，并没有任何性能下降和过拟合的情况。但是根据天下没有免费的午餐定理，`DenseNet`有一个恐怖的缺点就是内存占用极高，比较考验硬件，另外`DenseNet`和`ResNet`一样仍存在调参困难的问题。

# 附录
- 论文原文：https://arxiv.org/pdf/1608.06993.pdf
- 参考：https://www.cnblogs.com/zhhfan/p/10187634.html



---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)