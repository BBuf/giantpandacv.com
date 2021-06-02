
# [Deblurring by Realistic Blurring](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9156306)论文解读

【GiantPandaCV导语】本文主要对2020年CVPR中由澳大利亚国立大学和腾讯AI Lab联合发表的Deblurring by Realistic Blurring论文进行解读，从研究背景、网络结构、损失函数、实验结果方面对其进行简要介绍。

## 引言

图像去模糊方法主要包含盲去模糊(blind deblurring)和非盲去模糊(non-blind deblurring)，区别在于模糊核是否已知。传统的图像去模糊算法利用了多种先验，如：全变差(total variation)、重尾梯度先验(heavy-tailed gradient prior)等。近期随着深度学习技术的迅猛发展，也提出了基于CNN、基于GAN和基于RNN的用于图像去模糊的方法，比较著名的如DeblurGAN和DeblurGAN-v2等，而这些方法都专注于从模糊图像中恢复清晰图像本身，而忽略了图像模糊这个源头，因此并没有对图像模糊过程进行建模；同时由于数据集的稀缺，很多方法都采用了数据增强的方式来增加数据样本，但是大多数情况下合成的模糊图像与真实图像相差甚远。
因此这篇文章从图像模糊过程本身和合成图像真实性出发，提出了一种全新的GAN网络，包含learning-to-Blur GAN(BGAN)和learning-to-DeBlur GAN(DBGAN)两个子网络，其中BGAN的作用是学习图像模糊过程本身以及生成更符合自然模糊图像的合成图像，用于DBGAN训练；DBGAN的作用是进行去模糊操作，从模糊图像中恢复清晰图像。

## 网络简介

### 网络结构

如下图所示，BGAN的生成器和DBGAN的生成器网络结构非常类似，BGAN的生成器包含3个独立的卷积层，以及9个ResBlock，每个ResBlock包含5个3x3的卷积层以及4个ReLU激活层，DBGAN的生成器同样包含3个独立的卷积层，但是ResBlock的个数为16个。BGAN和DBGAN的鉴别器都采用的VGG-19网络。

![BAGN和DBGAN网络结构](https://img-blog.csdnimg.cn/20210508092122239.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxOTM2MTkx,size_16,color_FFFFFF,t_70)

受到SinGAN工作的启发，本文中也采用了同样的工作原理，在BGAN的生成器中随机添加噪声；首先从正态分布中采样长度为4的向量，再将其在空间维度重复128x128次得到4x128x128的噪声图，将其与公开数据集中获得的清晰图像一起作为BGAN生成器的输入，最终得到的输出结果为逼真的合成模糊图像。为了更好的满足GAN网络的训练，文中还收集了额外的真实世界模糊图像数据集用以BGAN鉴别器训练，以得到更加真实的合成图像。
DBGAN的输入为BGAN生成的模糊图像，其目的是为了从模糊图像中恢复得到清晰的图像，因此DBGAN的输出即为清晰图像。

### 损失函数

BGAN网络的损失函数包含Perceptual loss和Adversarial loss两部分，其中Perceptual loss针对输入的清晰图像和合成的模糊图像，Adversarial loss针对合成的模糊图像和新搜集的真实世界模糊图像，BGAN的目的就是模拟真实模糊图像的模糊过程，以得到逼真的合成模糊图像。
DBGAN网络的损失函数包含Perceptual loss、Content loss和Adversarial loss三部分，在BGAN的基础上增加了MSE loss作为Content loss，DBGAN的目的是从输入的模糊图像中恢复出清晰图像。两个网络的Adversarial loss与以往的GAN对抗损失有所区别，文章中提出了一种新的对抗损失：Relativistic Blur Loss(RBL/RDBL).
传统的对抗损失的目的是训练鉴别器区别真假的能力，反过来使得生成器能够生成更加”真“的图像，也就是将原来标记为”0“的生成图像训练的更加像真实世界图像，促使其标记值趋近于”1“；如下公式所示：

![传统对抗损失函数](https://img-blog.csdnimg.cn/20210508092213418.PNG)

而文章采用的RBL损失函数则是将标记值为”0“的合成图像和标记值为”1“的真实图像都推向0.5，互补的更新生成器参数，能够生成更加逼真的合成图像，如下图所示：

![RBL模糊过程简介](https://img-blog.csdnimg.cn/20210508092257818.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxOTM2MTkx,size_16,color_FFFFFF,t_70)

基于此观点，RBL(RDBL)损失函数可以描述为：

![RBL损失函数构造](https://img-blog.csdnimg.cn/20210508092530735.PNG)

![RDBL损失函数构造](https://img-blog.csdnimg.cn/20210508092549488.PNG)

所以BGAN和DBGAN的整体损失函数为:

![BGAN和DBGAN整体损失函数构造](https://img-blog.csdnimg.cn/20210508092605728.PNG)

### 实施细节
在训练BGAN和DBGAN时，采用了均值为0标准差为0.01的高斯分布来初始化权重；设置batch size=4，patchsize=128x128；学习率初始为0.0001，在损失函数值收敛后降为0.0000001；RBL和RDBL损失中的超参数α和β分别为0.005、0.01.

## 实验结果

作者在图像去模糊非常流行的GoPro数据集上进行实验，获得了SOTA的结果：

![GoPro数据集上的定量结果比较](https://img-blog.csdnimg.cn/20210508092624385.PNG)

在峰值信噪比(PSNR)和图像相似性程度(SSIM)上都获得了一定提升。视觉效果如下图所示：

![GoPro数据集上的视觉效果展示](https://img-blog.csdnimg.cn/20210508092716387.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxOTM2MTkx,size_16,color_FFFFFF,t_70)

从左到右分别为模糊图像、Nah等人的算法、Tao等人的算法和本文的方法，可以看出本文的方法在细节表现上更为出色，结果更加清晰。

## 总结

这篇文章的创新点主要在于设计了一个可以生成逼真合成模糊图像的GAN网络，并提出了一种新的对抗损失函数；其实很多领域都可以借鉴这篇文章的方法来生成数据集；如：低光图像增强、图像去雾、图像去雨等，可以很大程度减少搜集数据的难度，总体来说，这篇文章在数据集生成上开辟了一条新的道路，很有借鉴意义！

(如有不同见解可联系：j1269998232@163.com)

作者：H Jiang

时间：2021/5/6