# 1. 前言
这是卷积神经网络学习路线的的第二十二篇文章，要介绍的是2019年Google Brain的大作EfficientNet，论文全名为**EfficientNet：Rethinking Model Scaling for Convolutional Neural Networks**。

这篇论文系统的研究了网络的深度(Depth)，宽度(Width)和分辨率(Resolution)对网络性能的影响（MobileNetV1也从这些角度去尝试给网络瘦身），然后提出了一个新的缩放方法，即使用简单且高效的符合系数均匀的缩放深度/宽度/分辨率的所有尺寸，在MobileNets和ResNets上证明了其有效性。

进一步，论文是用神经结构搜索来设计了新的baseline并进行扩展获得了一系列模型，称为EfficientNets，比之前的ConvNets更加准确和高效。**其中EfficientNet-B7实现了ImageNet的SOTA，即84.4%的top1准确率和97.1%的top-5准确率，同时这个模型比现存的最好的卷积神经网络的模型大小小8.4倍，速度快6.1倍。**

# 2. 问题引入
在这篇论文之前已经有相当多的CNN工作通过增加网络的深度，宽度，分辨率等去提升模型性能。尽管可以任意的缩放二维或者三维，但任意的缩放都需要复杂的手工调整，并且常常产生更差的精度和效率。基于此，本论文重新思考和研究了CNN的缩放问题，即是否存在一种公式化的方法缩放CNN，从而实现更高的精度和效率？作者的研究表明，平衡网络宽度/深度/分辨率的所有维度是至关重要的，并且可以通过简单的按比例缩放每个维度来实现这种平衡。基于这个思考，论文提出了一种简单有效的复合缩放方法。如Figure2所示。

![模型缩放方法](https://img-blog.csdnimg.cn/20200229110618893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

比如，我们想使用$2^N$倍计算资源，可以简单的通过$\alpha^N$去增加深度，$\beta^N$去增加宽度，$\gamma^N$去增加分辨率，其中$\alpha$，$\beta$，$\gamma$是通过始小模型上进行小网格搜索确定的常系数。

直观来说，这种复合缩放的方法是有意义的，例如输入图像更大，则网络需要更多层来增加感受野，并且需要更多的通道来捕获更大图像上的更细粒度的语义信息。

论文指出，模型扩展的有效性很大程度上式取决于BaseLine，因此论文使用NAS来搜索新的BaseLine，并将其扩展为一系列模型，称为EfficientNets。

![这里展示了EfficientNets和当前一些SOTA网络在ImageNet上的比较，可以看到EfficientNets的参数量减少了很多倍，同时精度差不多甚至更高。](https://img-blog.csdnimg.cn/20200229111222416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 3. 复合模型缩放
## 3.1 问题定义
对于CNN的第$i$层可以定义为一个函数：$Y_i=F_i(X_i)$，其中$F_i$为一个操作(operrator)，$Y_i$为输出张量，$(X_i)$为输入张量，其形状可以表示为$<H_i，W_i, C_i>$，其中$H_i$和$W_i$为空间维度，$C_i$为通道数。一个CNN网络$N$可以表示为：

![CNN表达式](https://img-blog.csdnimg.cn/20200229111738646.png)

其中$F_i^{L_i}$表示$F_i$层在第$i$个阶段被重复了$L_i$次。

模型缩放和传统的CNN设计去寻找最优结构$F_i$不一样，模型缩放是为了寻找$F_i$的最佳宽度$C_i$，深度$L_i$和分辨率$(H_i,W_i)$，通过固定住$F_i$，模型缩放简化了资源限制问题。为了进一步简化搜索空间，论文限制所有层必须以恒定比率均匀缩放。在给定资源限制的情况下，去最大化模型的精度，就变成了以下优化问题：

![EfficieneNet优化问题](https://img-blog.csdnimg.cn/20200229112232246.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 3.2 维度缩放
在上面的优化问题中最大的苦难是$d$，$w$，$r$相互依赖，并且值在不同的资源约束下会发生变化。因此，传统方法主要在这三个维度之一进行缩放CNN：

- 深度($d$)：缩放深度是最常见的方法。这基于更深的网络可以捕获更丰富、更复杂的特征，并且可以很好地泛化新任务。 然而由于梯度消失问题，更深层次的网络也更难训练。虽然跨层连接和批量归一化等可以缓解训练问题，但非常深的网络的准确度增益会减少：例如，ResNet-1000和ResNet-101有相似精度。
- 宽度($w$)。对于小模型，缩放宽度是最常用的。这基于更宽的网络往往能够捕获更细粒度的特征，并且更容易训练。 然而，极宽但浅的网络往往难以捕获更高级别的特征。 
- 分辨率($r$)。 **感觉多尺度训练是最常用的了？** 这基于使用更高分辨率的输入图像，网络可以捕获更细粒度的特征。 

下面的Figure3展示了缩放单个维度会模型精度的影响。

![Figure3](https://img-blog.csdnimg.cn/20200229112856603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

从上图作者得到**观察一**：扩展网络宽度、深度或分辨率的任何维度都可以提高准确性，但对于更大的模型，精度增益会降低。

## 3.3 复合缩放
作者进一步观察到，网络宽度，深度和分辨率的缩放尺度不是独立的。直观来说，对于更高分辨率的图像，应该相应的增加网络深度，这样较大的感受野可以帮助捕获包含更高分辨率图像更多像素的特征。 同时，还应该增加网络深度，以便在高分辨率图像中捕获具有更多像素的更细粒度的特征。这些自觉表明我们需要协调和平衡不同的缩放尺度，而不是传统的单维缩放。 然后论文做了Figure4所示的实验，验证了不同网络深度和分辨率下的宽度缩放产生的影响。可以看到，在不改变深度(`d=1.0`)和分辨率(`r=1.0`)的情况下缩放网络宽度`w`，则精度会很快达到饱和。当CNN拥有更深（`d = 2.0`）和更高分辨率（`r = 2.0`），宽度缩放在相同的`FLOPS`成本下实现了更好的精度。 

![Figure4](https://img-blog.csdnimg.cn/20200229113549122.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后论文就提出了**观察2**：为了追求更高的准确性和效率，在ConvNet缩放期间平衡网络宽度，深度和分辨率的维度是非常重要的。

基于观察1和观察2，这篇论文提出了一个新的复合缩放方法，以公式的形式，使用一个复合系数$\phi$统一缩放网络宽度、深度和分辨率，如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200229114043741.png)

其中$\alpha$，$\beta$，$\gamma$是可以通过小网格搜索确定的常数。$\phi$是我们自己指定的系数，控制有多少资源可用于模型缩放，而$\alpha$，$\beta$，$\gamma$指定如何将这些额外的资源分配给网络的宽度，深度和分辨率。需要注意的是，常规卷积运算的FLOPS和$d$，$w^2$，$r^2$成正比，即倍增网络深度将使FLOPS加倍，但网络宽度或分辨率倍增会使FLOPS增加四倍。 因为卷积神经网络中主要的计算成本由卷积运算产生，因此使用等式3对ConvNet进行缩放，即将通过$(\alpha * \beta^2 * \gamma^2)^{\phi}$近似地表示增加总FLOPS。在论文中，作者约束$\alpha* \beta^2 * \gamma^2)^{\phi} \approx2$，使得对于任何新的$\phi$，总FLOPS增加约$2^{\phi}$。

# 4. EfficientNet结构
因为模型缩放不会改变BaseLine模型的操作$\hat{F_{i}}$，因此拥有良好的BaseLine也很关键。论文除了用现有的CNN评估这个缩放方法，还利用MnasNet搜索了一个新的BaseLine模型来评估，称为EfficientNet-B0，结构如下：

![EfficientNet-B0](https://img-blog.csdnimg.cn/2020022911494892.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

获得了BaseLine EfficientNet-B0后，通过下面两个步骤来进行模型缩放：

- 步骤一：首先确定$\phi=1$，假设有两倍的可用资源，并根据公式2和3进行$\alpha$,$\beta$,$\gamma$的小网格搜索。作者找到了EfficientNet-B0满足约束的最佳值$\alpha=1.2$，$\beta=1.1$，$\gamma=1.15$。
- 步骤二：将$\alpha$,$\beta$,$\gamma$固定为常数，并使用公式3扩展具有不同$\phi$的基线网络，以获得EfficientNet-B1至B7，见Table2。


![EfficientB0-B7](https://img-blog.csdnimg.cn/20200229115341188.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

具有类似的top-1 / top-5精度的CNN被组合在一起以进行效率比较。 与现有的CNN模型相比，EfficientNet模型可以将参数和FLOPS降低一个数量级（参数减少高达8.4倍，FLOPS减少高达16倍）。

# 5. 实验
## 5.1 缩放MobileNets和ResNets
Table3展示了以不同的方式缩放在ImageNet上的结果，与其他单维缩放方法相比，这个复合缩放方法提高了所有这些模型的准确性，表明了缩放方法对一般现有的CNN模型的有效性。

![Table3](https://img-blog.csdnimg.cn/20200229115619645.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 5.2 EfficientNet的ImageNet结果
Table2展示了从相同的BaseLine 网络EfficientNet-B0开始缩放的所有EfficientNet模型的性能。EfficientNet模型通常使用比其他CNN模型少一个数量级的参数和FLOPS，但具有相似的精度。 特别是，EfficientNet-B7在66M参数和37B FLOPS下达到84.4％top1 / 97.1％top-5精度，比之前最好的GPipe更精确但小8.4倍。Figure 1（上面有）和Figure 5显示了现有的SOTA CNN模型的参数精度和FLOPS精度曲线，其中缩放的EfficientNet模型比其他CNN模型具有更少的参数和FLOPS，且实现了更高的精度。 

![Figure5](https://img-blog.csdnimg.cn/20200229120047755.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

另外， 作者还验证了计算成本，Table4展示了20次运行的平均延迟，快了5-6倍。

![Tabel4](https://img-blog.csdnimg.cn/20200229120150509.png)

## 5.3 EfficientNet迁移学习的结果
论文还在常用的迁移学习数据集上评估了EfficientNet，论文使用了ImageNet预训练模型并对新的数据集进行微调，数据集相关信息如Table6所示。


![Table6](https://img-blog.csdnimg.cn/20200229120414427.png)

下面的Table5展示了迁移学习的性能，可以看到与公开可用模型相比，(1)EfficientNet模型减少了平均4.7倍（最多21倍）的参数，同时实现了更高的精度。 (2)EfficientNet模型在8个数据集中有5个仍然超过了它们的准确度，且使用的参数减少了9.6倍。

![Table5](https://img-blog.csdnimg.cn/20200229120558973.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Figure6则展示了各种模型的精度-参数曲线。

![Figure6](https://img-blog.csdnimg.cn/20200229120721364.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 讨论
所有的缩放方法都是以更多的FLOPs为代价来提升精度，Figure8说明了复合缩放的重要性。

![复合模型效果提升巨大](https://img-blog.csdnimg.cn/20200229120905555.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

论文还用了类激活图来分析不同缩放方法的有效性，Figure 7的类激活图说明了具有复合缩放的模型倾向于关注具有更多目标细节的更相关区域，而其他模型要么缺少目标细节，要么无法捕获图像中的所有目标。


![类别激活图](https://img-blog.csdnimg.cn/20200229121021812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 7. 结论
这篇论文系统的研究CNN模型的缩放，并明确仔细平衡网络宽度、深度和分辨率的重要性。因此论文提出了一种简单而高效的复合缩放方法，使我们能够以更原则的方式轻松地将基线CNN模型缩放到任何目标资源约束，同时保持模型效率。

# 8. 思考
这篇论文除了提出需要仔细平衡网络宽度、深度和分辨率的重要性，还使用NAS搜索出了一个搞笑的EfficientNet BaseLine，并且似乎论文的缩放方法在这个BaseLine上的提升是最大的，在其它的网络上则没有那么高了。。。这个实验做的非常漂亮，但我我这种穷人玩家肯定玩不动，也不要说复现了。。。猜测是动用了TPU？对代码有兴趣可以去研究源码。

# 9. 附录

- 原文链接：https://arxiv.org/pdf/1905.11946v1.pdf
- 代码：https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet


# 10. 推荐阅读

- [快2020年了，你还在为深度学习调参而烦恼吗？](https://mp.weixin.qq.com/s/WU-21QtSlUKqyuH6Bw1IYg)
- [将卷积神经网络视作泛函拟合](https://mp.weixin.qq.com/s/uF4dG0hzMNjVEA7Vkd6F-g)
- [在2020年，你觉得计算机视觉有哪些值得研究的领域？](https://mp.weixin.qq.com/s/KHZ1kfS6joACF3q_MeG2vw)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)