# AAAI'22 | Panini-Net | 基于GAN先验的退化感知特征插值人脸修复网络

Paper:AAAI'22 - Panini-Net: GAN Prior Based Degradation-Aware Feature Interpolation for Face Restoration

Authors:Yinhuai Wang, Yujie Hu, Jian Zhang

## 背景介绍
人脸修复是一种典型的ill-posed问题、可逆图像修复问题，其解不唯一且必存在。高度退化和多退化的场景下，高质量的人脸修复明显更具有挑战性。传统深度学习方法利用成对的数据集训练模型从而获得处理该退化问题的能力，这些普通范式虽然在全局人脸结构上有不错的效果，但是明显在修复结果的细节丰富度上并不理想。因此，近一两年，顶会上出现了不少基于GAN先验特征的人脸修复方法。这些方法通过将退化的人脸图像编码到训练好的GAN网络的潜在空间中，利用隐藏在GAN网络中丰富的图像先验知识，来获得更好的人脸修复细节。但是，常见的GAN网络的latent features维度并不高，这些特征的空间表达能力也不佳，难以完整捕获退化人脸图像的面部结构，造成恢复结果的身份信息与原图并不一致，即方法结果的保真度较低。 为了进一步、更完整得捕获输入退化图像的面部特征，一些方法不仅将降质人脸图像编码到潜空间中，而且还将外部特征（例如从降质人脸图像中提取的特征）与 GAN 先验特征融合起来，以实现更好的身份一致性。然而，它们并没有提供明确的降质感知的特征融合设计，因此在面对不同的、多变的退化时，修复效果的鲁棒性并不理想。

受最近对比学习范式的启发，作者团队首先提出一种**无监督的退化表征学习策略**，旨在利用**对比学习和视觉注意力的最新进展**，预训练一个退化表示编码器（DRE）。DRE 提取输入退化人脸图像的退化表征，作为全局条件指导恢复过程。此外，作者还提出了一种新的退化感知特征插值（DAFI）模块，可以根据退化表征动态融合 GAN 先验特征和从退化人脸图像中提取的特征。作者团队进一步提出了一种新的网络，以集成这些设计用于人脸修复任务。**由于选择和融合不同来源的特征的思路类似于制作 panini （帕尼尼，KFC早餐经典食品）的方式，因此将这个网络称为 Panini-Net。该网络可以根据退化程度动态调整融合的特征比例，以实现更好的修复性能。**



## 方法介绍
下面将逐一介绍Panini-Net的各个模块，最后将总结该方法的重点内容。
![Panini-Net架构图。 它由图像特征提取模块（Image Feature Extraction Module）、退化感知特征插值模块组成(DAFI) 模块，以预训练的 StyleGAN2 作为 GAN Prior 模块 (GPM)。 给定退化的人脸图像作为输入，图像特征提取模块提取特征,并预测latent code,该latent code可以从 GPM 中粗略地获取类似的高质量人脸。 然后，使用 DAFI 块逐步对特征做插值处理从而合并退化人脸图像的有效结构信息。预训练的退化表示编码器 (DRE) 将退化表示编码为向量，其可以被视为指导 DAFI 块进行恢复的全局条件。](https://img-blog.csdnimg.cn/f7591df0193b4ccca90578b6169c8d6a.png)


### GAN Prior Module 
Panini-Net采用预训练的StyleGAN2的生成器来作为GAN先验模块，如上图中所示，该模块从一个可学习的常量特征$F_{init}$开始，逐渐通过一系列的GAN blocks来生成分层的高质量特征图，从而将其配合退化感知特征插值模块
，通过动态的特征融合来矫正面部结构。

### Unsupervised Degradation Representation Learning for Degradation Representation
![退化表示的无监督退化表示学习策略。对于每次迭代，随机生成一组新的退化参数，并在两个高质量图像上操作它们从而得到不同的新 HQ 图像生成正例对。让队列中的历史图像成为反例，以鼓励学习退化而不是内容。](https://img-blog.csdnimg.cn/eb9387bc84c34602adb8672a5d4bdd3b.png)

无监督表征学习（Unsupervised Degradation Representation Learning ）用于图像修复（超分）其实不是一个比较新的idea，之前cvpr‘21的超分工作DASR，以及cvpr’22的AirNet都有类似的范式来作为方案的核心。不过在Panini-Net中，该部分还是挺不一样的。具体来说，如上图所示，先在两个不同的高质量人脸图像上应用同一组退化参数来得到两个内容不同、退化模式不同的退化图像，随后利用MoCo范式来执行对比学习，所利用的约束也是常见的InfoNCE loss，从而鼓励学习退化而不是内容。
cvpr‘21的超分工作DASR，以及cvpr’22的AirNet的论文名字：
Unsupervised Degradation Representation Learning for Blind Super-Resolution (CVPR'21)
All-in-one image restoration for unknown corruption (CVPR'22)

### Degradation-aware Feature Interpolation (DAFI) block
![退化感知特征插值 (DAFI) 块，无监督退化特征学习方式训练得到的encdoer从退化图像中抽取出V_{DR}作为退化的判别表征，该表征可以作为一种“condition”来生成自适应的channel-wise mask。可以从上图中看出，mask由一个mlp子网络和softmax来生成。这个mask 将用于动态特征插值从而辅助特征的融合。](https://img-blog.csdnimg.cn/9e6dcaf5e57644f58e1a5d210e814415.png)


在获得退化的判别表征后，Panini-Net将其作为一个全局的退化“condition”从而指导退化修复，具体来说，通过如上图所示的mask，该mask的size为： $ （B,C,1）$,即channel-wise的形式。将每个mask元素用于对应的融合特征通道的插值权重。 通过如下插值公式，来利用该mask来灵活的动态融合不同特征：

$$
\mathbf{F}_{D A F I}^i=\mathbf{F}_{G P B}^i \odot \mathbf{m a s k}_i+\mathbf{F}_{I F E}^i \odot\left(1-\mathbf{m a s k}_i\right),
$$
其中$\odot$表示channel-wise上的点积。

## 实验分析&视觉效果对比
![16xSR设定下的视觉对比图，可以看到PaniniNet很好的修复了退化图像的细节信息，保真度也非常不错。](https://img-blog.csdnimg.cn/e99c4224737c46e791ab564edd250ba5.png)

![消融实验](https://img-blog.csdnimg.cn/8d18f6eef3b4493e8be1d7e52d457e04.png)

作者在正文消融实验部分重点探讨了利用DAFI模块作为fusion操作的增益，并对Panini-Net的关键超参做了剖析。对fusion操作的探讨，主要是和直接利用concat+conv来fusion的常见操作做了对比，模型剖析部分则重点关注退化水平与插值比率的超参关系。

作者发现DAFI模块可以更好的保留GAN先验特征中的细节信息，而global condition guidance可以帮助DAFI更好的去fusion特征。当退化严重时，Panini-Net可以动态增加GAN-Prior的使用比例。

## 结论
这篇论文重点关注如何更好的引入GAN Prior从而帮助人脸图像修复问题，作者通过无监督表征学习和结合mask策略的插值（特征融合）模块来将GAN prior动态的引入到修复网络中，实现了非常不错的修复效果。
