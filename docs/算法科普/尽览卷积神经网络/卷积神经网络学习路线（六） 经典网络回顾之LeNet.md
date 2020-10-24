![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218212108283.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)开篇的这张图代表ILSVRC历年的Top-5错误率，我会按照以上经典网络出现的时间顺序对他们进行介绍，同时穿插一些其他的经典CNN网络。
# 前言
这是卷积神经网络学习路线的第六篇文章，前面五篇文章从细节，超参数调节，网络解释性方面阐述了卷积神经网络。从这篇文章开始，卷积神经网络学习路线就开始代领大家一起探索从1998年到2019年的20多种经典的网络，体会每种网络的前世今身以及包含的深邃思想。本节就带大家来探索一下LeNet。
# 背景&贡献
LeNet是CNN之父Yann LeCun在1998提出来的，LeNet通过巧妙的设计，利用卷积、参数共享、下采样等操作提取特征，避免了大量的计算成本，最后再使用全连接神经网络进行分类识别，这个网络也是近20年来大量神经网络架构的起源。
# 网络结构
LeNet-5是LeNet系列最新的卷积神经网络，设计用于识别机器打印的字符，LeNet-5的网络结构如下图所示。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191218205810465.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)我们可以具体分析一下：
- 首先输入图像是单通道的$28\times 28$大小的图像，用caffe中的Blob表示的话，维度就是$[B,1,28,28]$。其中$B$代表`batch_size`。
- 第一个卷积层`conv1`所用的卷积核尺寸为$5\times 5$，滑动步长为$1$，卷积核数目为$20$，那么经过这一层后图像的尺寸变成$28-5+1=24$，输出特征图的维度即为$[B,20,24,24]$。
- 第一个池化层的池化核尺寸为$2\times 2$，步长$2$，这是没有重叠的`max pooling`，池化操作后，图像尺寸减半，变为$12\times 12$，输出特征图维度为$[B,20,12,12]$。
- 第二个卷积层`conv2`的卷积核尺寸为$5\times 5$，步长$1$，卷积核数目为$50$，卷积后图像尺寸变为$12-5+1=8$，输出特征图维度为$[B,50,8,8]$。
- 第二个池化层`pool2`池化核尺寸为$2\times 2$，步长$2$，这是没有重叠的`max pooling`，池化操作后，图像尺寸减半，变为$4\times 4$，输出矩阵为$[B,50,4,4]$。
- `pool2`后面接全连接层`fc1`，神经元数目为$500$，再接`relu`激活函数。
- 最后再接`fc2`，神经元个数为$10$，得到$10$维的特征向量，用于$10$个数字的分类训练，再送入`softmaxt`分类，得到分类结果的概率`output`。

# Keras实现
我们来看一个`keras`的`LeNet-5`实现，

```
def LeNet():
    model = Sequential()
    model.add(Conv2D(32,(5,5),strides=(1,1),input_shape=(28,28,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model
```

没错就是这么简单。。。

# 后记
因为LeNet是最简单的CNN网络，所以就讲得比较简短。虽然LeNet结构简单，准确率在解决现代很多视觉任务时已经力不从心了，但LeNet是CNN的开创者，我们仍然应该给予足够的尊重。

# 卷积神经网络学习路线往期文章
[卷积神经网络学习路线（一）| 卷积神经网络的组件以及卷积层是如何在图像中起作用的？](https://mp.weixin.qq.com/s/MxYjW02rWfRKPMwez02wFA)

[卷积神经网络学习路线（二）| 卷积层有哪些参数及常用卷积核类型盘点？](https://mp.weixin.qq.com/s/I2BTot_BbmR4xcArpo4mbQ)

[卷积神经网络学习路线（三）| 盘点不同类型的池化层、1*1卷积的作用和卷积核是否一定越大越好？](https://mp.weixin.qq.com/s/bxJmHnqV46avOttAFhk28A)

[卷积神经网络学习路线（四）| 如何减少卷积层计算量，使用宽卷积的好处及转置卷积中的棋盘效应？](https://mp.weixin.qq.com/s/Cv68oXVdB6pg_4Q_vd_9eQ)

[卷积神经网络学习路线（五）| 卷积神经网络参数设置，提高泛化能力？](https://mp.weixin.qq.com/s/RwG1aEL2j6G-MAQRy-BEDw)

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)