> 论文：XNOR-Net: ImageNet Classification Using Binary Convolutional Neural Networks 
> 链接：https://arxiv.org/abs/1603.05279
> 代码：http://allenai.org/plato/xnornet
# 1. 前言
前面已经介绍了2篇低比特量化的相关文章，分别为：[基于Pytorch构建一个可训练的BNN](https://mp.weixin.qq.com/s/UMnROIUiW2PPR8vIg8W6OQ) 以及 [基于Pytorch构建三值化网络TWN](https://mp.weixin.qq.com/s/aP6zaZwHRRNOTG5icZ-Kcg) 。在讲解那2篇文章的时候可能读者会发现某些小的知识点出现的比较突兀，今天要介绍的这一篇论文可以看做对前两篇文章的理论部分进行查漏补缺。

# 2. 概述
这篇论文中提出了两种二值化网络**Binary-Weight-Networks和XNOR-Networks** 。BNN在前些日子讲过了，今天再回顾一下然后和XNOR-Networks做一个对比。下面的Figure1展示了这两种网络和标准CNN的差别，这里使用的是AlexNet作为基础网络。

![BNN/XNOR-Net和CNN的对比](https://img-blog.csdnimg.cn/20200718220241169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
BNN是通过对**权重W**做二值化操作来达到**减小模型存储空间**的目的，准确率影响并不明显（56.7%->56.8%，但从实验部分ResNet18上的结果来看影响还是比较大的）。而XNOR-Networks对**权重W和激活值I** 同时做了二值化，既可以减少模型存储空间，又可以加速模型，但是准确率降了很多。

本文介绍的这两种网络里面提到的**权重**指的是卷积层的权重或者全连接层的权重，由于全连接可以用$1\times 1$卷积代替，所以这里统一认为是卷积层的权重。


# 3. Binary-Weight-Networks
**Binary-Weight-Networks**的目的是将权重**W**的值都用二值来表示，也即是说W的值要么为-1，要么为1。在网络的前向传播和反向传播中一直遵循这个规则，只不过在参数更新的时候还是使用原来的权重W，因为更新参数需要更高的精度（位宽，例如float就是32位）。

下面来看一下论文的公式推导过程以及如何实现权重的二值化，首先约定卷积层的操作可以用**I*W**来表示，其中**I**表示输入，维度是$c\times w_{in}\times h_{in}$，其中**W**表示卷积核的权重，维度是$c\times w\times h$。那么当使用二值卷积核B乘以一个尺度参数$\alpha$代替原来的卷积核W，就可以获得下面的等式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200718230137817.png)

中间那个圆圈带加号的符号表示没有乘积的卷积运算，这里$W \approx \alpha B$

注意，这里的$\alpha$默认是个正数，$\alpha$和$B$是相对的，因为如果$\alpha$和B都取相反数它们的乘积是不变的。

现在希望用一个尺度参数$\alpha$和二值矩阵B来代替原来的权重W，那么我们肯定想前者尽可能的等于后者，因此就有了下面的优化目标函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200718231128711.png)

我们的目的是让这个$J$越小越好，这种情况的下$\alpha$和B就是我们需要的值，另外这里将矩阵$W$和$B$变换成了向量，也即是说W和B的维度都是$1*n$，其中$n=c*w*h$。

接下来，我们的目的就是求出$\alpha$和$B$的最优值，使其满足上面的优化目标。我们可以将上面的优化函数展开，写成下面的形式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719185017644.png)

因为B是一个$1*n$的向量，里面的值要么是$-1$，要么是$1$，因此：

$B^TB=n$

即一个常量。并且W是已知的，所以：$W^T*W$仍然是一个常量。另外，$\alpha$是一个正数，这些常量在优化函数中都可以去掉，不影响优化的最终结果。就可以得到B的最优值的计算公式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719200908915.png)

显然，根据这个式子，**B的最优值就是W的符号**。即：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719211733336.png)

**也就是说，当W中某个位置的值是正数时，B中对应位置的值就是+1，反之为-1。**（这不就是之前我们将基于Pytorch构建BNN中使用的基本原理吗）

知道了B的最优值之后，接下来就可以求$\alpha$的最优值。我们使用上面第二个J函数表达式对$\alpha$求导，并让导数等于0，从而获得最优值，等式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719211918309.png)

然后将前面获取到的$B^*$带入上面的等式，就可以获得下面这个求$\alpha$最优值的最终式子：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719212016535.png)

也就是说$\alpha$的最优值是W的每个元素的绝对值之和的均值，其中  **|| ||** 表示L1范数。

最后**Binary-Weight-Networks** 算法可以总结如下：

![使用BNN训练一个CNN](https://img-blog.csdnimg.cn/20200719214730436.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

第一个for循环是遍历所有的层，第二个for循环是遍历某一层的所有卷积核。通过获得$A_{lk}$和$B_{lk}$就可以近似等于原始的权重$W_{lk}$了，**另外在backward过程中的梯度计算也是基于二值权重**。

# 4. XNOR-Networks
前面的BNN只是将权重W的值用二值表示，而下面要引入的XNOR-Networks不仅将权重W用二值表示，并且也将输入用二值表示。XNOR即同或门，假设输入是0或1，那么当两个输入相同时输出为1，当两个输入不同时输出为0。

然后我们知道卷积就是使用卷积核去点乘输入的某个区域获得最后的特征值，这里假设输入为X，卷积核是W，那么我们就希望得到$\beta,H,\alpha,B$使得下式成立：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719220326838.png)

即用$\beta H$近似输入X，另外：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719221641872.png)

然后就有了下面这个优化等式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719221715771.png)

其中圆圈里面带一个点的符号代表element-wise product，即点乘运算。如果用Y代替XW，**C代替HB**，$\gamma=\beta \alpha$，那么优化等式可以重写为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719221857788.png)

然后这个式子和第三节的优化等式形式类似，即：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200718231128711.png)

因此可以利用Binary-Weight-Networks的推导结果来解这个优化式子。根据前面$\alpha$和$B$的计算公式，可以得到C和$\gamma$的计算公式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719222128481.png)![在这里插入图片描述](https://img-blog.csdnimg.cn/20200719222153284.png)

这里因为$|X_i|$和$|W_i|$是相互独立的，所以 可以直接拆开。

**上面两个等式右边的结果就是4个参数的最优解。**

下面的Figure2展示了具体的二值化操作。第一行即是Binary-Weight-Networks的最优值求解。第二行是XNOR-Networks的最优值求解，不过由于存在重复计算，所以采用第三行的方式，其中$c$表示通道数，A表示通过对输入I求均值得到的。第四行和第三行的含义一样，更加完整的表达XNOR-Net的计算过程，这里K就是第三行计算得到的K，中括号里面的内容就是最优的C即$C^*$。

**然后第四行里面的*符号代表的是convolutional opration using XNOR and bitcount operation**。即如果两个二值矩阵之间的点乘可以将点乘换成XNOR-Bitcounting操作，**将32位float数之间的操作变成1位的XNOR门操作，这就是加速的核心点**。

![XNOR-Net的具体二值化操作](https://img-blog.csdnimg.cn/20200719224007163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure3右侧展示了添加XNOR量化方法后的网络结构：

![XNOR-Net和传统CNN的对比](https://img-blog.csdnimg.cn/20200719224545401.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 5. 实验结果
下面的Table1是上面两种二值化方式和同期其它的二值化方式的对比结果，其中Full-Precision是不经过二值化的模型准确率。

![本文的二值化方法和其它的方法对比](https://img-blog.csdnimg.cn/20200719224747899.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后下面还提供了本文的两种二值化方式和正常的ResNet-18的准确率对比，可以看出在ResNet18上**Binary-Weight-Networks对准确率的影响更大** （和AlexNet相比）。这个原因大概是因为ResNet的残差结构导致的。

![本文的两种二值化方式和正常的ResNet-18的准确率对比](https://img-blog.csdnimg.cn/20200719224950912.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 总结
本文从理论角度提出了两种二值化方法，即BWN和XNOR-Net，背后的理论推导相对是比较简单的，但没看这篇文章去理解之前已经发过的几篇文章是相对困难的，所以希望这篇文章会给对低比特量化感兴趣的读者带来帮助。同时，从实验结果仍然可以看到BNN要落地还是有一大段路（准确率掉得厉害），笔者暂时也了解不多，后面学习一些更solid的方法也会及时在公众号更新。

最后再提一句，看完这篇文章详细读者都知道为什么之前构建BNN的时候使用根据符号来进行二值化了的吧，个人认为任何一算法背后的理论支撑都是不可或缺的。

# 7. 参考
- https://blog.csdn.net/u014380165/article/details/77731595

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)