## Dynamic Anchor Learning for Arbitrary-Oriented Object Detection

介绍一篇自己AAAI2021的目标检测工作：Dynamic Anchor Learning for Arbitrary-Oriented Object Detection。主要是讨论旋转目标检测中anchor匹配机制的问题和一些思考。

arxiv论文地址：https://arxiv.org/abs/2012.04150

代码：https://github.com/ming71/DAL

### 1. Motivation

基于anchor的算法在训练时首先根据将预设的anchor和目标根据IoU大小进行空间匹配，以一定的阈值（如0.5）选出合适数目的anchor作为正样本用于回归分配的物体。但是这会导致两个问题：

* **进一步加剧的正负样本不平衡**。对于旋转目标检测而言，预设旋转anchor要额外引入角度先验，使得预设的anchor数目成倍增加。此外，旋转anchor角度稍微偏离gt会导致IoU急剧下降，所以预设的角度参数很多。（例如旋转文本检测RRD设置13个角度，RRPN每个位置54个anchor）。
* **分类回归的不一致**。当前很多工作讨论这个问题，即预测结果的分类得分和定位精度不一致，导致通过NMS阶段以及根据分类conf选检测结果的时候有可能选出定位不准的，而遗漏抑制了定位好的anchor。目前工作的解决方法大致可以分为两类：网络结构入手和label assignment优化，这里不再赘述。

### 2. Discussion

用旋转RetinaNet在HRSC2016数据集上实验可视化检测结果发现，很如下图b，多低质量的负样本居然能够准确回归出目标位置，但是由于被分为负样本，分类置信必然不高，不会被检测输出；如图a，一些高质量正样本anchor反而可能输出低质量的定位结果。

![](https://img-blog.csdnimg.cn/20201216112135845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

为了进一步验证这种现象是否具有普遍性，统计了训练过程的所有样本IoU分布，以及分类回归分数散点图，结果如下图。我们将anchor和gt的IoU称为输入IoU，pred box和gt的IoU称为输出IoU。从中看出：

* 74%左右的正样本anchor回归的pred box后依然是高质量样本（IoU>0.5）；近一半的高质量样本回归自负样本，这说明负样本还有很大的利用空间，当前基于输入IoU的label assignment选正样本的效率并不高，有待优化。
* 图c说明，当前的基于输入IoU的标签分配会诱导分类分数和anchor初始定位能力成正相关。而我们期望的结果是pred box的分类回归能力成正相关。从这里可以认为**基于输入IoU的标签分配是导致分类回归不一致的根源之一**。这个很好理解，划分样本的时候指定的初始对齐很好的为正样本，其回归后就算产生了不好的预测结果，分类置信还是很高，因为分类回归任务是解耦的；反之很多初始对齐不好的anchor被分成负样本，即使能预测好，由于分数很低，无法在inference被输出。
* 进一步统计了预测结果的分布如d，可以看到在低IoU区间分类器表现还行，能有效区分负样本，但是高IoU区间如0.7以上，分类器对样本质量的区分能力有限。【**问**：表面上右半区密密麻麻好像分类器完全gg的样子，但是我们正常检测器并没有出现分类回归的异常高分box的定位一般也不赖，为什么？一是由于很多的IoU 0.5以上的点都是负样本的，即使定位准根本不会被关注到；二是预测的结果中，只要有高质量的能被输出就行了，其他都会被NMS掉，体现在图中就是右上角可以密密麻麻无所谓，但右下角绝不能有太多点。】

![](https://img-blog.csdnimg.cn/20201216112148458.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

### 3. Method
#### 3.1 Analysis

首先是baseline用的是附加角度回归的reitnanet。

直观来说，输出IoU能够直接反映预测框的定位能力，那么直接用输出IoU来反馈地选取正样本不就能实现分类回归的一致吗？但是进行实验发现，网络根本不能收敛。即便是在训练较好的模型上finetune，模型性能依然会劣化发散。推测是两种情况导致的：

* 输入IoU大，但是输出IoU小的anchor并不应该被划分为负样本，其更大概率还是正样本的，这部分容易学习的样本丢失严重影响收敛
* 任何样本都可能在训练过程中随机匹配到一个目标，但不应该因此直接确信为正样本，这在训练早期尤为严重

例如，一个anchor回归前IoU为0.4，回归后IoU是0.9，我们可以认为这是一个潜在高质量样本；但是如果一个anchor回归前是0，回归后0.9，他基本不可能是正样本，不该参与loss计算。反之亦然。可见这么简单的思路没有人采用，不是没人想到，而是真的不行。相似的label assignment工作中，即使利用了输出IoU也是用各种加权或者loss等强约束确保可以收敛，有一个只利用输出IoU进行feedback的工作，但是我复现的时候有很多问题，实验部分会介绍。

#### 3.2 Dynamic Anchor Selection

可以理解为输入IoU是目标的空间对齐（spatial alignment），而输出IoU是由于定位物体所需重要特征的捕捉能力决定的，可以理解为特征对齐（feature alignment）能力。据此定义了匹配度（matching degree）如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112158301.png#pic_center)

前两项比较好理解，通过输入IoU表征的空间对齐能力对anchor的定位性能作先验缓和上面的两种情况以稳定训练过程。第三项表征的是回归前后的不稳定性作用，对回归前后的变化进行惩罚。实际上这一项是有效利用输出IoU的关键，后面的实验会证明这一点。自己私下的实验中发现，有了$fa$和$u$两项(即$\alpha=0$)实际上就能实现超越输入IoU的labelassignment了，但是输出IoU很不稳定，参数比较难调，而加入空间先验后稳定了很多，效果也能保持很好的水平。

这个不确定性惩罚项$u$有很多表征形式，之前试过各种复杂花哨的表征和加权变换，虽然相对现有形式有所提升但是提升空间不大。没必要搞得故弄玄虚的，所以最后还是保留了这种最简单的方式。

有了新的正样本选择标准，直接进行正常的样本划分就能选出高质量的正样本。这里还能进一步结合一些现有的采样和分配策略进一步获得更好的效果（如ATSS等），论文没有展示这部分实验可以自己尝试。学习策略上，在训练前期为了避免输出IoU的不稳定影响，采取逐渐加大空间对齐影响系数，直至设定值。实验证明这个策略不影响最终效果，只是加速收敛。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020121611233259.png#pic_center)

#### 3.3 Matching-Sensitive Loss

得到匹配度矩阵后，我们可以将其加权到训练loss中，核心思想是增强分类器对高质量样本的识别效果，从而解决Motivation中提到的分类回归不一致的问题。具体而言就是将匹配度矩阵进行补偿，最大值补偿到1，补偿值加权到正样本上去，使之更多地关注高性能样本的分类和回归性能。一开始我采用的是直接将md加权到loss，效果很差找问题调了一段时间才解决。补偿加权的方法相比直接加权有两个好处：

* 避免分类器对匹配度划定的负样本进行不合理的关注。比如在md补偿的策略下，一个低质量正样本可能导致很大的补偿值从而带来一堆低质量正样本；
* 由于匹配度是介于[0,1]，直接加权将导致正样本被进一步稀释；
* 确保分类和回归任务对补偿的anchor的足够关注，每个样本至少有一个anchor能学好。例如对于某个gt最大md为0.2，直接加权将导致其loss贡献很小，本来就难匹配的目标更难学了。

损失函数表示和具体分析如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112205984.png#pic_center)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112212696.png#pic_center)

对于分类任务而言，如果正样本全部置为1，就无法区分高效md不同的样本了（显然md=0.1和md=0.9的样本被分为正样本的概率不应该一样）【**问**：检测器正常情况下就是这么做的，咋就没你这么多事？不是不能区分。可行的原理是通过不断的学习，优化下降loss来对“边缘”程度不同的hard example进行判断。所以我们看到定位能力和分类分数常常不会差得很远，这都是反复的优化的结果。显然这种策略有效但是很笨，其中显然还有东西可以做，可以大大提高训练效率】。所以这里通过构造带有定位潜力信息的md补偿矩阵来加权loss进一步关注高质量正样本的学习情况，使得分类得分更加准确有效，提高NMS的准确性。

对于回归任务而言，早在cascade RCNN就提出过高精度anchor对loss贡献相对小而难以优化的问题，还有很多工作从IoU分布、重采样、归一化等方法缓解这个问题。这里我们采用的还是匹配度信息，方法也是很质朴的对正样本re-weight；只不过加权关注的不再是空间对齐的anchor，而是对根据md度量的高质量样本给予更多的关注。

采用匹配度敏感的loss能够有效增强检测器的精确定位样本的区分能力，如下图所示，左边是正常训练检测器，可以看到定位精度上高性能部分的额区分度很低，但是加了MSL后样本的定位性能大大提高，同时分类分数也对应提高，越往右上角颜色越深，分类分数和定位性能有较好的关联性。很多分析由于原论文篇幅有限没有展开。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112219823.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

### 4. Experiment

旋转目标的实验上采取了四个数据集：三个遥感数据集DOTA，HRSC2016，UCAS-AOD和一个文本检测数据集ICDAR2015。同时在水平目标数据集上也进行了泛化性实验，采用的是ICDAR2013，NWPU VHR-10和VOC2007。为了证明我们的方法能够有效提取高质量的anchor，从而减少旋转目标检测中anchor的预设，缓和不平衡问题，我们在特征图每个位置仅仅使用了3个水平的anchor，文本检测由于目标宽高全都很常悬殊，采用5个ratio。

#### 4.1 Ablation

##### 4.1.1 Component-wise evaluation

with input IoU是正常的label assignment方式，相比之下，引入output IoU后性能反而略有下降，特征对齐的信息并不能被有效利用。实际上这里的$\alpha$设置为0.1，因为输出IoU实际上非常不稳定，比例一旦增大就会导致训练不收敛。然而在采用不确定性惩罚后，即使是很简单的直接作差形式性能也能有很大提高，相比前一个模型直接提升了7%。加入MSL后，mAP进一步得到提高达到88.6。同时，相比AP50的2.7%涨幅，AP75提高了9.9%，可见MSL对高精度定位大有裨益。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020121611222732.png#pic_center)

##### 4.1.2 Hyper-parameters

匹配度的设置实际上引入了两个额外的超参数，因此补充了敏感性实验如下。表中的结果和分析结论基本一致：当不确定性抑制项存在，$\alpha$合理减小会使特征对齐的影响增大，同时mAP增大，说明能够输出IoU能够有效指导样本划分过程。如果$\alpha$过大将使得空间对齐能力占主导地位，输出IoU带来的不稳定减小，此时不应施加很强的扰动惩罚，即减小$\gamma$的值，如$\alpha$=0.9时，$\gamma$从3减小到5，mAP从70.1提升到83.5。

此外，正如上面所说，其实$sa$也就是正常的IoU完全可以不要，仅靠$fa$和$u$就能取得更好的结果，但是很敏感anchor设置和$\gamma$取值以及$u$的形式，比较难调。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112233829.png#pic_center)

这为匹配度公式的参数调试提供一种思路：$\alpha$和$\gamma$应该同时增大或减小。此外观察不难发现，$\alpha$在0.3-0.5的效果是最好的，$\gamma$也大致可以确定。这个结论在其他数据集和其他anchor设置上也验证过，依然成立，论文篇幅有限这里没有予以展示。但是说实话，取值而言会有偏移，虽然不会太多但是还是要调一调，这个有点不快乐。

#### 4.2 Experiment Results

##### 4.2.1 Comparison with other sampling methods

表中列举的都是**自己复现的结果**，采用各自论文的思想但是由于原论文都不是做旋转检测的，并不完全一致。值得一提的还是HAMbox。很多朋友问我，我说发生了肾莫事，我一看截图，啊，原来左天，塔门有人复现不出hambox的效果，不收敛或者下降。我这里直接复现其实也不收敛，最后的85.7结果是在收敛模型上finetune，并且没有直接用output IoU，而是加了input IoU调制；为了防止大量低质量compensated anchor，设置的每个物体只匹配三个anchor；并且用上类似curriculum learning的方式慢慢加大系数才学收敛的，收敛后效果也害行。比较意外的是atss，没想到效果这么好，稍微调一调就86+。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112241136.png#pic_center)

##### 4.2.2 Results on DOTA

RetinaNet在DOTA的结果是唯一一处用的不是我开源的代码，而是临时写的基于mmdetection的，因为我的代码优化不行占显存速度慢，顺便也在s2anet大佬肩膀上试了试。如果问为肾莫没在DOTA上做ablation，只能说硬件显卡不够，大意了，跑dota就很费劲了，调了好久，我只能耗子尾汁。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112247705.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

##### 4.2.3 Results on HRSC2016

稳定涨了几个点，也是这个数据集小容易调。注意啊，很快啊，只放了3anchor就能比肩一些多anchor的的，速度大概24FPS。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112254434.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

##### 4.2.3 Results on UCAS-AOD

这个上面比较意外的是AP75，提升非常大，达到惊人的74，甚至有点不对劲。但是可视化DAL前后的结果，发现也在情理之中。这个数据集比较简单，尤其是飞机，是个旋转检测器都一副要上90封顶的样子。只是baseline很多定位性能卡在AP75坎的检测结果全被MSL优化后抬上来了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112301926.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

##### 4.2.4 Results on ICDAR2015

其实可以看出这个方法并不万能，在文本检测器上还是不行，文本检测的难点不只是匹配。baseline只有77.5的情况下，反复调参加通用trick也只能到81.5，加了多尺度测试勉强提点到82+，和当前先进的检测器还是有相当的差距。但是可以移植到一些sota检测器上去，也会有所提升。（BTW，文本和通用旋转检测确实不太一样，要实现较高F1只是解决旋转问题远远不够。例如之前写了个Cascade-RetinaNet在HRSC2016的baseline加点简单增强就有85+，但是移植到IC15上印象中裸跑才不到75。）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112309334.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

#### 4.3 Experiments on HBB dataset

其实这个方法是通用的，只是旋转目标匹配难匹配，所以提升更大更明显。于是如下，可以看出在三个数据集都有提升涨点稳定。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201216112315988.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L21pbmdxaTE5OTY=,size_16,color_FFFFFF,t_70#pic_center)

最后这次的四个reviewer都很认真细致，提出了很多有帮助的建议，还是比较走运的。

开源程序或者论文上有问题疑惑欢迎提问和联系，遥感旋转目标检测相关方向的同学欢迎一起交流学习。 