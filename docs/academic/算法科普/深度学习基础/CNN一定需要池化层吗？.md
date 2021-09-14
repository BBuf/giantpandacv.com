## 太长不看版
这是一篇ICLR2015的论文，文章针对现有网络设计，探索了最大池化层是否能被卷积层给完全替代。作者在 CIFAR10/100 数据集上设计了一系列对比实验，从而得出可以在不损失精度下，将最大池化层替换成卷积层。

# 前言
在现有的网络结构设计指导下，似乎卷积层后跟一个池化层下采样，已经是一个准则。我们重新思考了现有SOTA网络，并得出结论最大池化层是能被卷积层给替代。我们设计了一系列小网络，并提出了一种新的**反卷积方法**来去可视化CNN学习到的特征

# 模型描述
为了理解池化层和卷积层为什么有效，我们返回到公式里面
我们令 **f为特征图**，W, H, N分别是特征图的**宽，高，通道数**
对于一般的池化窗口为K的p范数下采样，我们有

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020081715535734.png#pic_center)

而对于一般的卷积层，我们需要设定一个权重，进行相乘，并将多个通道结果进行相加。最后再通过激活函数进行激活，形式如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817155744798.png#pic_center)

**θ代表的是卷积核权重**， 从1到N求和，**代表是对多个特征图的卷积窗口进行求和**

这里有个细节需要强调下：**池化层是分别对每张特征图做池化/P范数操作**
而卷积层在多通道情况下，**是通过相加各个特征图来进行特征融合**

因此在比较这两个公式后，论文里也说到：**池化层可以看作是一种 特征级别上的卷积，其激活函数为对应的p范数**

分析完上述公式后，作者假定了池化层有效的几个因素
1. **P范数形式能增加CNN的平移不变性**，这里存疑我后续会解释
2. 池化层的下采样，能为后续的卷积操作**提供更大的感受野**
3. 池化层仅仅是在特征图上操作，不会带来额外的参数，**因此有助于整个网络优化过程**

我们假设第二点是提升CNN性能的关键。我们有以下两个选择来替代池化层
1. 去除掉池化层，将卷积层的步长变为2。这种方法参数量与此前一致
2. 用步长为2的池化层，来替代池化层。由于引入新的卷积层，参数量会适当增加

考虑到3x3卷积叠加能达到5x5卷积的感受野，减少大量参数，我们也将其加入到实验对比。因此我们的网络设计如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020081716555511.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)


### 补充P范数
P范数定义如下

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817155506542.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

即求**某个范围内x的P次方和**，最后**再开P次方**

那**如果是平均池化**，我们可以看作是P=1的范数下采样，前面需要乘上一个系数 **K分之一**

### CNN平移不变性的存疑
具体可以参考下 [证伪：CNN中的图片平移不变性](https://zhuanlan.zhihu.com/p/38024868)
简单来说，我们下采样因子是固定的，常用的我们都是步长为2的操作，来进行叠加，缩小特征图分辨率

举个例子，1张224x224的图片，经过多次下采样至 7x7，那么整个采样因子就是 32x32。为了保证平移不变性，我需要让物体平移距离是32的整数倍。

换句话说，我相当于将整个图片划分成了32x32个小格子，物体需要落到这个格子里，才能具有不变性。这个概率是

$$
P = 1 / (32*32)
$$

# 实验
我们在CIFAR10, CIFAR100, ImageNet2012数据集上进行测试
## 实验设置
我们在先前的Model C上，又引申出三种模型

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817170341447.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

1. 第一个模型，将每一层最后一个卷积层步长设置为2，去除掉了池化层
2. 第三个模型，将最大池化层以步长为2的卷积层替代
3. 第二个模型，为了与第三个模型做对比，在保持相同卷积层时，用最大池化层下采样
因为原始的ModelC，已经包含了两次卷积层+一次池化层的结构。所以针对第一个模型就不再多设置一个模型对比
## 实验结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817171115676.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

首先我们来看下Model A,B,C
最主要的区别就是拿3x3卷积来替代大卷积核和部分1x1卷积。
事实证明使用3x3卷积替代大卷积核，能得到性能上的提升

再B，C两组中， ALL-CNN都得到了最好的效果。
而在A组，池化的效果比Strided的效果更好。**为了保证不是因为参数量增加而引起的，我们也对比了ALL-CNN-A**。事实证明池化操作是能提高网络性能的

后续，我们针对训练过程是否加入图像增广也做了一组实验

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817201437399.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

在这两种情况下，网络表现的都比前面的要好很多
我们将这个最佳模型，放到CIFAR100进一步进行测试

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200817202055573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)

可以看到我们的效果依然很好，超越了前面的网络模型。
仅仅被Fractional Pooling超过，但这个模型参数过大，大概50M左右的参数。
所以我们的模型表现是十分不错的。

# 笔者理解
在现代的网络设计中，池化层出现的越来越少了。早在几年前，这个问题还是很令人深思的。
郑安坤大佬也做过一系列实验[CNN真的需要下采样吗](https://zhuanlan.zhihu.com/p/94477174)，并且后面也探讨了maxpooling。

我也是比较同意文中的观点，因为无论是均值池化层还是最大池化层，都可以对应传统图像处理的一个人为设计的滤波器。如果没有下采样操作，它完全是可以做一个抑制图像噪声的操作在。
并且池化层是与多通道之间没有关系的，只是在单一特征图上做。

我会认为在浅层特征图中，空间相关特征比较明显，可以使用池化层。
在高维特征图上，特征经过编码后，空间相关不太明显，这时候用卷积层做下采样会比较好。

而且还是需要具体任务具体分析，比如在CVPR2020的[Small Big Net](https://mp.weixin.qq.com/s?__biz=MzA4MjY4NTk0NQ==&mid=2247488413&idx=1&sn=f032a7a29c9d34eee79b4c13fed8de02&chksm=9f80a90ba8f7201daf223182b4f62b741e50727c88967fefd6ee143c70c74d8cac633e57dd3d&mpshare=1&scene=1&srcid=0730MHbXzWWHvFamf52HbwXo&sharer_sharetime=1597669556188&sharer_shareid=e41a096be0d8cd906e18224f9bb5c2a8&key=de879d1d09e2a8b72003f264798f11019e57e45e9b0248510be411eab6005ecd6030f7e32780c77ef20546eb25721f352c1faf063472e7cd88c2b26db3e772100f2ecf885e08f528c92c95850f352345813e32b4d8072ca1f694ac99f096b5c1699ef6d6a925bb7e53e9163ecf9083e2c7d8ce742f2c95d840377bfcd7e8501e&ascene=1&uin=MTYwODQ4NTY2MQ==&devicetype=Windows%2010%20x64&version=62090523&lang=zh_CN&exportkey=A/4/C%2bjGhOyqrOCRnVuZ/3Y=&pass_ticket=yxPZ4JnyAn5tcu2fH60Q2zDBIwL7hINFYQw43TiZLpQmFJhehNOh/LrWeOu//OU8)

在时间维度上对视频帧应用最大池化来提取特征信息，因此我还是认为池化操作是有其必要性的。