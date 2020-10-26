# 0. 前言
熟悉人脸相关业务的读者应该对下面这个3D人脸模型比较熟悉：

![3D人脸模型](https://img-blog.csdnimg.cn/20200620162018319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到在3D空间中人脸的位姿主要包含三种：

- 平面内旋转角（左右歪头问题）：roll。
- 平面外左右旋转（正脸，侧脸问题）：yaw。
- 平面外俯仰（仰头，低头问题）：pitch。

然后现在的很多人脸检测器比如我们介绍过的MTCNN，FaceBoxes，RetinaFace等等都实现了高精度的实时人脸检测，但这些算法往往都是在直立的人脸上表现很好，在角度极端的情况下表现不好。通过上面的3D模型我们想到，人除了正坐和站立，还有各种各样的姿态，如Figure1所示，导致人脸的平面旋转角度(roll)的范围是整个平面内（0-360度），注意这里我们没有考虑yaw和pitch，也就是说PCN这个算法是用来解决任意平面角度的人脸检测问题。注意在论文中角度的简称是（rotation-in-place（RIP）angle）即RIP。

![人脸可能有各种平面内的旋转角度](https://img-blog.csdnimg.cn/20200620163021134.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


# 1. 介绍
基于CNN的人脸检测器受益于CNN强大的非线性特征表达能力，但在检测旋转人脸时效果一般，因为各个角度旋转的人脸在特征在模型训练时不容易收敛，目前已有三种针对旋转人脸检测的方案：**数据扩充**、**分而治之**、**旋转角度探测器**(rotation router)。

**数据扩充**：最简单粗暴也最直观的方法，将包含向上的人头图像均匀地做360°全角度旋转生成训练数据，再用一个性能强劲的模型学习，现有的upright人脸检测模型可以直接学习，无需额外操作。但是，为了拟合如此旋转角度的人脸场景，模型性能需要比较强悍，耗时较大，就无法实时了。如Figure2所示：

![Figure2](https://img-blog.csdnimg.cn/20200620163545672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**分而治之**：训练多个检测器，每个检测器只检测一小部分偏转角度的人脸，所有检测器结果融合后，就可以覆盖所有方向的人脸。每个检测器只需要检测特定旋转范围内的人脸，对每个检测器而言，只需要一个耗时少的浅层模型即可。但是所有检测器都需要跑一遍，整体耗时就增加了。另外，角度是一个360度的分类或者回归问题，容错空间太大，如下图所示，直接预测存在误差较大的可能。

![Figure7](https://img-blog.csdnimg.cn/20200620164224737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**旋转角度探测器**。用一个CNN(rotation router)计算出旋转人脸的偏转角，将旋转的人脸按偏转角校准至向上后，再使用现有的upright face detector检测校准后的人脸candidates即可，这符合认知常识，添加一个rotation router计算人脸偏转角度即可，不需要额外开销。但是精准的人脸角度计算很有挑战性，为了精准的计算人脸偏转角，通常都需要使用性能强悍的CNN，耗时就又成为了瓶颈。

前面的三种方法要么精度不高要么耗时很大，因此作者就提出了这个PCN，怎么做的呢？**既然利用rotation router想一步到位计算精准的人脸偏转角度有难度，那么我们渐进式地基于cascade从粗到精一步一步计算。第一层网络先初略判断一个偏转角，再校准一下。第二层网络同样操作，进一步校准，以减少人脸偏转角度范围。第三层网络精准计算偏转角度，基于前两步骤校准后，再使用第三层网络直接输出人脸分类、偏转角度、bbox即可。** 整体下来模型耗时也少，可以实时。有没有感觉到这个算法好像和MTCNN的过程超级像？所以首先stage1就是对face candidates(类似mtcnn图像金字塔+滑窗)筛选candidates(face/non-face二分类)，将方向朝下人脸校准为方向朝上人脸(updown clip即可)，stage2与step1类似，人脸筛选(face/non-face二分类)+将step1中的upright人脸进一步校准至 [-45; 45]，最后stage3输出人脸分类、偏转角度(the continuouts precise RIP angle)、bbox即可。**可以看到，只有在stage3才做精准预测，stage1、2只做+-90°、+-180°旋转，这也保证了整个算法的时间消耗低，做到了实时。** 接下来作者总结了PCN的特点和优势点如下：
- PCN渐进式地分步校准人脸偏转角度，每个小步骤都是一个基于浅层CNN的简单任务，最终可以让偏转的人脸角度逐步降低并校准，准确率高、效果好、耗时少。
- step1、2只做粗校准(如下->上180°、左->右90°校准)，优势有二：1 粗校准操作耗时少，易实现；2 对角度的粗分类也容易实现，可以直接在人脸分类、bbox回归的multi-task中加一个分支即可
- 在两个有挑战的数据集上-----多角度旋转的FDDB+作者手工筛选并标注的wider face test子集上(multi-oriented FDDB and a challenging subset of WIDER FACE containing rotated faces in the wild)，本方案取得了不错的效果。

# 2. PCN详细介绍
## 2.1 整体介绍

PCN包括了3个阶段，每个阶段都做了人脸和非人脸分类，人脸bounding box的回归，人脸偏转角度计算。其中stage1和stage2只做离散分类的角度估计，stage3做连续回归的角度细估计，对人脸方向的校准(stage1,stage2,旋转人脸180度，90度等)属于后操作，也就是说在校准网络结束后做，使之渐进的校准为一个朝上的人人脸。如果使用一个模型预测各种旋转角度的人脸，可能在精度和耗时上都有损耗，所以该论文提出将校准过程分为3个渐进式步骤。在stage1和stage2上只做粗略的方向分类(离散的方向分类，如180,-180，90)，最后stage3做连续的方向回归，输出校准后的人脸偏转角度，因为偏转角度已经校准到-45到45范围，所以直接使用人脸检测器检测出人脸，不用再接校准操作,PCN已经可以在CPU上达到实时。因为是渐进式的校准人脸角度，逐渐降低人脸的练准度，所以这种方法可以处理任何角度旋转的人脸。在人脸数据集FDDB和wider face test(作者自己制作的)均取得了不错的效果。

下面的Figure3展示了PCN的大致过程：

![Figure3  PCN的概述。PCN-1首先鉴别人脸并把朝下的人脸校准为朝上，将RIP角度范围从[-180°，180°]减半到[-90°，90°]。 旋转后的候选窗被PCN-2进一步区分并校准到[-45°，45°]的直立范围，并将RIP范围再缩小一半。 最后PCN-3确定每个候选是否人脸并预测精确的RIP角度。](https://img-blog.csdnimg.cn/20200620164633400.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

如Figure6所示，PCN逐渐校准每个候选框的RIP方向使其直立以更好的区分人脸和非人脸。下面我们就分别解释一下每个Stage的细节。

![PCN模型的三个阶段的详细CNN结构。 “Conv”，“MP”，“InnerProduct”和“ReLU”分别表示卷积层，最大池化层，内积层和Relu层。 （k×k, s）表示内核大小为k，步幅为s。](https://img-blog.csdnimg.cn/20200620165539663.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 2.2 第一个stage的PCN

对于每个滑窗输入$x$，PCN1做三件事情：face/non-face分类、bbox回归、校准：

$[f; t; g] = F1(x)$

$F1$：stage1的CNN模型

$f$：face confidence score，用于区分face/non-face

$t$：bbox回归向量

$g$：方向得分(0~1二分类问题、输出up、down即可)

**第一个损失函数**，区分face/non-face：

$L_{cls} = ylogf + (1-y)log(1-f)$

如果$x$是人脸$y=1$，否则$y=0$。

**第二个损失函数**，尝试回归人脸的bounding box。

$L_{reg}(t, t^*) = S(t - t^*)$

其中$t$和$t^*$代表了真实的回归框和预测的回归框，S代表损失距离Smooth L1 loss，t和$t^*$可以用回归框的左上角，长宽(这里长等于宽)这3个参数来描述，写成公式就是：

$t_w = w^*/w$

$t_a = (a^* + 0.5w^* - a - 0.5w)/w^*$

$t_b = (b^* + 0.5w^* -b - 0.5w)/w^*$

其中$a,b,w$分别代表了回归框左上角坐标$(x,y)$和高宽$width$。

**第三个损失函数**，对PCN1来说，就是简单的up-down二分类问题，使用softmax即可。公式为：

$L_{cal}(t,t^*)=ylog(g)+(1-y)log(1-g)$。

**整个PCN1的损失函数为：**

![PCN1的损失函数](https://img-blog.csdnimg.cn/20181031231710448.png)

其中$\lambda$是各个loss的weight！

以上操作的意思：PCN1可以类似adaboost一样，第一步大量去除容易分类的fp candidates(face/non-face)，再做一次bbox归回，最后根据up-down分类结果，对candidates做upright flip，确保所有人脸bbox都是朝上，经此操作，人脸旋转角度变为[-90, 90]。将常用的upright人脸数据集做[-180, 180]旋转，以扩充为旋转数据集。在训练阶段，有3种类型的数据：

pos samples：iou vs gt > 0.7

neg samples：iou vs gt < 0.3

suspected samples：iou vs gt ∈ (0.4, 0.7)

face/non-face classification：pos & neg；

bbox regression && calibration：pos & suspected；

特别地，对于calibration网络，pos & suspected samples：

face-up：RIP angles ∈ (-65, 65)

face-down：RIP angles ∈ (-180, -115) & (115, 180)

不在此角度范围内的RIP angles不用于训练calibration分支。


## 2.3 第二个stage的PCN
 这个阶段和第一个stage很类似，唯一不同就是在calibration分支是一个三分类问题[-90;-45], [-45;45], or [45;90]，将常用的upright人脸数据集做[-90, 90]旋转，以扩充为旋转数据集。calibration：pos & suspected；
calibration分支分类id含义：
0：[-90, -60]，需要+90
1：[-30, 30]，不用做操作
2：[60, 90]，-90
不在此范围内的数据，不考虑用于训练。

## 2.4 第3个stage的PCN
经过stage1、2两波操作，人脸RIP已经被校准至[-45,45]之间(calibrated to an upright quarter of RIP angle range)，此时人脸已经比较容易检测，使用PCN-3的网络就可以准确检测并回归人脸bbox。最终人脸角度把三个阶段的计算角度结果累加即可得到最终的旋转角度。这部分原理看图：

![从粗到细的级联回归方式预测RIP角度。 候选窗口的RIP角度，即θRIP，是来自三个阶段的预测RIP角度的总和，即θRIP=θ1+θ2+θ3。 特别是，θ1只有0°或180°两个值，θ2只有三个值，0°，90°或-90°，θ3是[-45°，45°]范围内的连续值。](https://img-blog.csdnimg.cn/20181031230947516.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

最终的人脸角度计算公式为：

$\theta_{RIP}=\theta_1+\theta_2+\theta_3$


## 2.5 PCN是如何实现精度和速度提升的
- 在早期阶段只预测粗糙的RIP角度，增强对多样性样本的鲁棒性，并且有利于后续步骤。
- 然后通过逐步减少RIP的范围减少人脸和非人脸的分类误差，从而提高了检测精度。
- 将难度较大的旋转角度预测分解为多个小任务，每一个任务都比较简单，这使得校准的整体难度降低。
- 在前两个阶段先用小的CNN过滤掉那些简单的负样本，再用大的CNN鉴别难负样本，可以大大提高提高检测速度。
- 基于粗糙RIP预测的校准可以通过三次翻转原始图像来有效实现，这几乎不会增加额外的时间成本。 具体而言，将原始图像旋转-90°，90°和180°以获得向左，向右，向下的图片， 如Figure5所示，0°，-90°，90°和180°的窗口可以分别从原始，向左，向右，向下的图片中截取得到。

![Figure5](https://img-blog.csdnimg.cn/20200620174902830.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 3. 结果
下面的Table1展示了在带角度的FDDB数据集上的精度和速度比较。

![实验结果](https://img-blog.csdnimg.cn/20200620175112292.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Figure10还展示了在Wider Face上的一些可视化结果：

![Figure10](https://img-blog.csdnimg.cn/20200620175208337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 4. 参考
- https://arxiv.org/pdf/1804.06039.pdf
- https://blog.csdn.net/qq_14845119/article/details/80225036
- https://zhuanlan.zhihu.com/p/36303792
- 官方代码链接：https://github.com/MagicCharles/FaceKit/tree/master/PCN（里面还附带了各种版本的代码链接，包括可以Arm端部署的NCNN版代码）

---------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)