【GiantPandaCV导语】这是最近百度的一篇网络结构设计文章，该网络结构是**手工设计**得来，主要改进在对特征图**多级划分卷积，拼接**，提升了网络的精度，同时也降低了推理时间。**个人感觉是res2net，ghostnet的结合**，并且训练阶段没引入过多的trick，最后的**实验结果很惊艳**，或许是炼丹的一个好选择。

![CNN模型图像分类准确度，推理速度比较](https://img-blog.csdnimg.cn/20201107105353406.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

# 前言

在该工作内，我们发现多层级的特征对视觉任务效果提升明显，因此设计了一个即插即用的多级分离模块（Hierarchical-Split Block）。HS-Block包含了多个层级特征图的分离和拼接，我们将其替换到Resnet中，提升效果明显，top-1准确率能达到81.28%，同时在其他任务如目标检测，分割表现也十分出色。

# 介绍

这篇工作里面我们考虑以下3个问题

1. 如何避免在特征图产生**冗余信息**
2. 如何在**不增加计算复杂度**前提下，让网络学习到**更强的特征表达**
3. 如何得到更好的精度同时，**保持较快的推理速度**

基于上述3个问题，我们设计了HS-Block来生成多尺度特征图。

在原始的特征图，**我们会分成 S 组，每一组有W个通道（也即W个特征图）。**

![HS-Block模块示意图](https://img-blog.csdnimg.cn/2020110711104014.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

**只有第一组的能直接送入到最后的输出中**

从第二组开始操作就开始不一样了：

- 第二组的特征图经过3x3卷积后，得到了**图示的黄色方块**
- 然后它会被**分离成两组子特征图**
- 一组子特征图**拼接到最后的输出**
- 另一组子特征图**拼接到第三组特征图**，如此循环往复

可以看到这个操作还是比较复杂的，但基本都是按照上述那四条规则循环进行。

最后拼接完的特征图我们会**再用1x1卷积进行特征融合**

# 灵感来源

在论文内，作者也说了灵感来自于res2net和Ghostnet（解析可参考 [华为GhostNet网络](https://mp.weixin.qq.com/s/1Fw5bp5cf_zElcH-IVtgqA)

我们先来分别回顾这两个网络主要的模块

Res2Net的模块结构如下：

![Res2Net模块示意图](https://img-blog.csdnimg.cn/2020110711220364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

Res2Net的模块也是将一组特征图分离成4组，分层做卷积。
特殊点在于第二组特征图经过3x3卷积后，**是以 elementwise-add 的形式加到第三组特征图上**。
类似的，第三组特征图经过3x3卷积后，也以elementwise-add的形式加到第四组特征图上。

下面我们看看GhostNet模块：

![GhostModule模块示意图](https://img-blog.csdnimg.cn/20201107112559721.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

GhostNet的设计思想是**为了减少特征图冗余，部分特征图可以在已有特征图上经过卷积操作得到**。因此它先通过一个卷积模块得到黄色部分，再在黄色部分中进一步生成特征图（原论文是Depthwise Conv），最后两者拼接到一起。

# 设计思想

HS-Block的多尺度分离，卷积的做法。**能容不同组的特征图享受到不同尺度的感受野**。

在前面的特征图，拼接进来的特征卷积次数较少，**感受野较小，能关注细节信息**。

在后面的特征图，拼接进来的特征卷积次数较多，**感受野较大，能关注全局信息**

这部分思想来源于res2net，通过不同感受野，增加特征的丰富性。

那么引入了另外一个问题，**res2net的做法在通道数量较大的时候，计算复杂度也随之增加**。那么是不是所有特征图都是必须的呢？在GhostNet里面有观察到并不是

![特征图冗余](https://img-blog.csdnimg.cn/20201107113352899.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

**部分特征图是可以在已有特征图上生成**，因此借助GhostNet的思想，**我们在模块内将 S2 组卷积得到的特征图，部分拼接到 S3 组，实现了特征图的复用**，后续组也是这样一个做法，成功用 GhostNet 思想降低了计算复杂度。

# 分析计算复杂度

常规的卷积核参数计算是

$$
param = C_{in}*K_{size}[0]*K_{size}[1]*C_{out}
$$

在resnet的BottleNeck结构里，输入前后通道数不变，而在我们这里，是分成S组，每组W个通道，换句话说，我们输入通道是

$$
C_{in} = W*S
$$

而我们使用的卷积核长宽都是一致的，替换到上面公式就是

$$
param_{normal} = W*S*K*K*W*S
$$

现在我们看下HS-Block这里的计算量
由于第一组特征图直接拼到后面，所以没有计算量

$$
param = 0, \space(i=1)
$$

后面的特征图，输入通道需要concat，前面一半的特征图
因此在原始的W上，还有额外加多一项，输入通道为：

$$
C_{in} = (W + \frac{2^{i-1}-1}{2^{i-1}}W)
$$

**这里输入通道推导可能光看公式不太清楚，其实可以手推一下，基本的情况是**

$$
S2: (W + \frac{1}{2}W) 这里的一半通道来自S1 \\
S3: (W + \frac{1}{2}C_{s2}) = (W+\frac{1}{2}(W + \frac{1}{2}W)) = (W+\frac{3}{4}W) \\
$$

以此类推
最后每一组卷积的参数为

$$
param_i = K*K*W*W*( \frac{2^{i-1}-1}{2^{i-1}}+1) \space(1<i<=S)
$$

**注意到上面是每一组卷积的参数，总参数我们需要累加起来**

$$
param_{total} = K*K*W*W*\sum_{n=1}^{s-1}(\frac{2^{n-1}-1}{2^{n-1}}+1)
$$

其中1是常数项，累加s-1次，和为s-1，我们单独提出来

$$
param_{total} = K*K*W*W*[\sum_{n=1}^{s-1}(\frac{2^{n-1}-1}{2^{n-1}})+s-1]
$$

而累加项中

$$
\frac{2^{n-1}-1}{2^{n-1}} = 1-\frac{1}{2^{n-1}}
$$

很容易得知

$$
1-\frac{1}{2^{n-1}} < 1
$$

因此我们有推出不等式

$$
\sum_{n=1}^{s-1}(\frac{2^{n-1}-1}{2^{n-1}}) <\sum_{n=1}^{s-1}(1) = s-1
$$

我们代入前面公式有

$$
param_{total} < K*K*W*W*(S-1+S-1) = K*K*W*W*(2S-2)
$$

与常规卷积参数比较

$$
param_{normal} = K*K*W*S*S < param_{total} < K*K*W*(2S-2)
$$

可以看到参数量确实是减少了

# 实验结果

![图像分类实验结果](https://img-blog.csdnimg.cn/20201107131701449.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

可以看到最后的实验，得益于合理的设计思想，HS-ResNet的性能和速度都很快。

![目标检测实验](https://img-blog.csdnimg.cn/20201107131933688.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

![图像分割实验](https://img-blog.csdnimg.cn/20201107132033351.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

在目标检测，分割等其他任务中表现也十分优秀

![不同分组实验性能](https://img-blog.csdnimg.cn/20201107132100838.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

作者也在最后探讨了下分组数对于性能，推理速度的影响，可以看到这个模块也是十分灵活，性能和速度都是可以自行调整的

# 最后

目前Paddle官方还在做后续实验，相关代码模型还没放出来。作者的思想也不是很复杂，将res2net和ghostnet的思想融合进来，也保留了灵活性，拓展性，相信这个模型还会有更大的潜力。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)