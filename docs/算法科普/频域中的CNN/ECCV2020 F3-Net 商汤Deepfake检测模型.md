# 前言 
这篇论文是商汤团队在ECCV2020的一个工作：[Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues](https://arxiv.org/abs/2007.09355)，通过引入两种提取频域特征的方法**FAD (Frequency-Aware Decomposition) 和LFS (Local Frequency Statistics)** ，并设计了一个 MixBlock 来融合双路网络的特征，从而在频域内实现对Deepfake的检测

# 介绍
随着Deepfake技术不断迭代，检测合成人脸的挑战也越来越多。虽然已有的基于RGB色彩空间的检测技术准确率不错，**但是实际中，这些视频随着流媒体传播，视频通常会被多次压缩**，而在较低质量的视频中，要想进行检测就比较困难，这也一定程度上启发我们去**挖掘频域**内的信息。 

那么问题来了，**我们如何才能把频域信息引入到CNN中**？传统的FFT和DCT**不满足平移不变性和局部一致性**，因此直接放在CNN**可能是不可行的**。

我们提出了两种频率特征，从一个角度来看，**我们可以根据分离的频率分量重组回原来的图片**，因此第一种频率特征也就可以被设计出来，我们可以对**分离的频率分量经过一定处理，再重组回图片**，最终的结果也适合输入进CNN当中。这个角度本质上是在RGB空间上描述了频率信息，而不是直接给CNN输入频率信息。这也启发了我们的第二种频率特征，**在每个局部空间（patch）内，统计频率信息，并计算其平均频率响应**。这些统计量可以重组成**多通道特征图，通道数目取决于频带数目**
![图A是不同分辨率的真假人脸，图B是该工作设计的两种频域方法，图C是多个模型ROC曲线](https://img-blog.csdnimg.cn/20201001173924573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
如上图，在低质量(Low Quality)图片中，两种图片都很真实，但是在局部的频率统计量(LFS)中，呈现出明显的差异，这也就很自然能被检测出来。

基于上述两种特征，我们设计了 **Frequency in Face Forgery Network**（$F^3Net$)，第一个频率特征为**FAD（Frequency-aware Image Decomposition）**，第二个频率特征为**LFS（Local Frequency Statistics）**。因为这两个特征是相辅相成的，我们还设计了一种融合模块**MixBlock**来融合其在双路网络中的特征。整体流程如下图所示
![整个算法的简单流程](https://img-blog.csdnimg.cn/20201001174527929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
# FAD
以往的工作采用的是人工设计频域滤波器，但这**无法完全覆盖所有的图像模式**，并且**固定的滤波器很难自适应的捕捉到图像中伪造的模式**。因此我们提出了自适应的滤波方法，具体做法如下：
- 设计N个二分类滤波器（也就是所谓的**掩码mask**）$\{{f_{base}^i}\}_{i=1}^{N}$，将图像的频率**分为低，中，高三个频带**。
- **为了让其具备自适应能力，我们额外设计三个可学习的滤波器**$\{{f_{w}^i}\}_{i=1}^{N}$。然后分别将这两种滤波器结合在一起，公式如下

$$
f_{base}^i + \sigma(f_{w}^i) \\ 
$$
其中 $\sigma$ 为 归一化，目的是限制其值在 (-1, 1) 之间
$$
\sigma = \frac{1-e^{-x}}{1+e^{-x}}
$$
- 我们将这两个滤波器应用在DCT变换后
- 最后做反DCT，将图像重组回来

总的公式如下
$$
yi = D^{-1}\{ D(x)\odot[f_{base}^i + \sigma(f_{w}^i)], i=1,2,..,N\}
$$
**其中D代表DCT变换，$D^{-1}$代表反DCT变换**


FAD的流程如下
![FAD特征提取流程](https://img-blog.csdnimg.cn/20201001181758249.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
其中b图是将2维频谱展开成1维的形式，我们可以看到第一个滤波器取的是整个频段的1/16，而第二个滤波器取得是1/16~1/8，第三个滤波器则取得是剩下的7/8。
# LFS
前面的FAD尽管提取到了频域特征，但它最后是通过反DCT变换，转化到RGB空间上，输入进CNN。**这些信息并不是直接的频域信息**，因此我们提出了局部频域特征**local frequency statistics**（LFS)，它能满足RGB图片的**平移不变性以及局部一致性**。
![LFS特征提取流程](https://img-blog.csdnimg.cn/20201001182532460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
具体流程如下
1. 对输入的图片采用**滑窗DCT**（Silde Window DCT)，从而提取局部的频率响应。
2. 计算一系列可学习频带的**频率响应均值**
3.将频率统计信息重新组合为与**输入图像共享相同布局**的**多通道空间映射**
$$
q = log_{10} ||D(p) \odot[h_{base}^i + \sigma(h_{w}^i)]||, i ={1, 2, ..., M}
$$
其中log10是为了调整数值级别，D是滑窗DCT变换。

跟FAD一样，我们这里也设计了二分类滤波器和可学习滤波器，操作流程跟FAD完全一样，这里就不展开了。

**对于每个滑窗**w**中的局部统计信息**q**，经过上述变换被转换为 $1\times1\times M$的向量**

在我们的实验中，我们将每个**滑窗大小设置为10x10**，**步长为2**，**频带数目为6**。一张299 x 299 x 3的图片输入进来将被转换为 149 x 149 x 6。
# MixBlock
虽然这两种频率特征不同，但具有一致性，都是从DCT变换，并经过滤波器进行不同频率分离。因此我们设计了一种MixBlock来在双路网络中融合两者的特征。
![MixBlock结构图](https://img-blog.csdnimg.cn/2020100118420671.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
1. FAD和LFS共同输入进卷积里，得到一个**AttentionMap**
2. FAD和LFS分别与AttentionMap相乘得到$F_{attention}$和$L_{attention}$
3. $F_{attention}$与$LFS$相加，$L_{attention}$与$FAD$相加，完成特征融合。

论文里双路网络都采用的是Xception网络，该网络一共有12个Block，我们将融合模块**分别放置在第7个和第12个XceptionBlock**里，对中，高层特征进行融合操作
# 实验
![实验图片](https://img-blog.csdnimg.cn/20201001185022329.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
这张对比图很好的表现了F3-Net在低质量图片中的表现，可见在频域内做检测确实有更好的抗压缩性能。
![实验图片](https://img-blog.csdnimg.cn/20201001185122848.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
在不同数据集上表现也比较稳定，没有因数据集的分布产生较大波动
![消融实验](https://img-blog.csdnimg.cn/20201001185158459.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDEwNjkyOA==,size_16,color_FFFFFF,t_70#pic_center)
最后作者也设计了一系列消融实验，来表明各个模块的有效性。
# 总结
这篇工作还是挺有意思的，不同于以往传统频域特征，它选择将传统和深度学习进行结合，为可学习的滤波器设定一定约束，从而根据不同图像自适应分离出频率信息。Deepfake的一大难点就是对低质量，多次压缩图片的检测，因为在RGB图片上是很难发现的。最终的实验也表明该方法的有效性，坐等商汤开源代码


