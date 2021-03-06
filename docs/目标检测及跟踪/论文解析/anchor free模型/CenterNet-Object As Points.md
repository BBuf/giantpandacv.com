# Objects as Points

ZHOU X, WANG D, KRäHENBüHL P 2019a. Objects as Points. ArXiv [J], abs/1904.07850.

## 摘要

检测任务将目标视为图片中的与坐标对齐的框。大多数成功的目标检测器都详尽的枚举潜在的目标位置，然后对每个位置进行分类。这种方法是浪费的、低效的、需要额外的后处理工作。在这篇文章中，我们另辟蹊径，将目标建模为一个单独的点-即目标框的中心点。本文提出的检测器使用**关键点估计**来寻找中心点，然后回归出目标的其他属性，比如大小、3D位置、方向、姿态等。基于中心点的方法，CenterNet，是一个端到端的、可微分的检测器，要比同等条件下基于目标框的检测器更简单，更快速，更准确。在MSCOCO数据及上，CenterNet能达到更好做到速度-精度权衡。142FPS速度实现了28.1%的AP，52FPS的速度实现了37.4%AP，1.4FPS的速度(多尺度测试)实现了45.1%的AP。在KITTI基准上以相同的方法来估计3D检测，在COCO关键点数据集上估计人体姿态，我们的方法与传统的多阶段方法相比，更具有竞争优势并且可以实时运行。

## 1. 介绍

目标检测驱动了很多视觉任务比如实力分割、姿态估计、跟踪、动作识别，也存在许多下游的应用比如视频监控、自动驾驶、视觉问答。目前，目标检测器使用的是紧紧围绕目标的一个坐标对齐的外包矩形框。这种方法将目标检测简化为大量潜在的物体外包矩形框的图像分类问题。对于每个外包矩形框来说，分类器决定了这个图像的内容是前景还是背景。

单阶段检测器将一系列可能的外包矩形框（也就是锚框）的复杂排列在图像中进行划窗，直接对其进行分类而不需要关注对应框中的内容。两阶段的检测器则为每个潜在的框重新计算了图像特征，然后分类这些特征。后处理也称之为非极大抑制算法会移除同一个个体的重复的检测框。后处理是不可微分的、不可训练，因此大多数目标检测器并不是可以端到端的训练。然而，过去五年中，这个想法达到了经验上的成功。基于划窗的目标检测器有一些浪费，因为他们需要枚举所有的目标可能存在的位置和维度。

在这篇文章中，我们提供了一个更加简单更加有效的选择。我们使用目标中心点作为一个目标的代表。其他属性，比如目标的大小，维度，3D区间，朝向和姿态等信息可以直接通过图像特征进行回归。目标检测在这种情况下就更类似一个标准的关键点估计问题。我们将输入图片喂到全卷积网络从而输出一个热度图。热图的峰值代表目标的中心点。每个峰值的图像特征会负责预测目标框的高和宽。魔性的训练是通过密集的有监督训练。推理过程是一个单网络的前向传播，不需要非极大抑制算法进行后处理。

![我们将目标框建模成一个点。目标框的大小和其他属性信息可以通过中心点的特征进行推断](CenterNet-Object%20As%20Points.assets/image-20200528140028620.png)

我们的方法具有普适性，可以以很小的代价扩展到其他任务中。我们在3D目标检测和多人姿态估计进行了试验，这些任务中不同的就是除了中心点以外的额外的输出。对于3D检测框估计问题来说，我们需要回归目标的绝对深度，3D检测框维度和目标朝向。对于人体姿态估计，我们将二维关节点位置看做来自中心的偏移量，直接回归到中心点位置。

CenterNet非常简单，能以非常高的速度运行。通过一个简单的ResNet18层和上采样鞥，我们的网络可以在COCO上以142FPS的速度达到28.1%的AP。通过精心设计的关键点检测网络DLA-34，能够在COCO上以52FPS的速度达到37.4%的AP。如果使用SOTA关键点估计网络Hourglass-104，并且结合上多尺度测试，我们的玩了个可以在COCO上以1.4FPS的速度达到45.1%。在3D 检测框估计的问题和人体姿态估计方面，我们以更高的推理速度可以和最新的技术进行竞争。

代码开源在：https://github.com/xingyizhou/CenterNet

## 2. 相关工作

**基于区域分类的目标检测算法** 最初最为成功的深度目标检测器R-CNN是通过从大量的候选区域中枚举目标的位置，然后将对应区域的图片抠出来，然后使用深度神经网络进行分类。Fast-RCNN则直接扣取对应区域的特征来节省计算量。然后，两种方法都依赖于缓慢的底层区域建议的方法。



**隐含对象的目标检测算法** Faster RCNN在检测网络中产生区域建议。 它在一个低分辨率的图像网格周围采样固定形状的边界框(锚) ，并分类为“前景或非前景”。 如果重叠IOU>0.7, 锚标被标记为前景，如果背景重叠0.3，则被认为是背景忽略。 每个生成的区域方案再次进行分类。如果将提出的分类器转化为多分类，这就构成了单阶段检测器的基础。Anchor先验、不同的特征图分辨率、损失函数重权衡等都是一阶段检测器的改进点。

我们的方法和基于锚框的一阶段方法非常相关。中心点可以被视为一个形状不可知的锚框。然后，这里有几个非常重要的区别。首先，CenterNet仅仅基于位置分配Anchor，而不是基于检测框重叠率。这里不需要人为设置阈值来作为前景背景的判别阈值。第二，每个物体只有一个正锚框，因此也不需要非极大抑制进行后处理。我们仅仅从关键点热图上提取局部极值。第三，CenterNet使用更高的输出分辨率（stride=4）,传统目标检测器通常输出分辨率较低（stride=16）。这就不需要多个锚框了。

**基于关键点估计的目标检测算法** 我们并不是第一个使用关键估计方法进行目标检测的。CornerNet检测两个目标框的对角作为关键点。ExtremeNet则检测目标框的上，左，下，右和中心点。这些方法都是采用了与CenterNet相同的鲁棒的关键点估计网络。然而，这些方法需要组合分组阶段，这会显著降低每个方法的速度。CenterNet简单地提取单个目标的中心点而不需要后处理。

**单目3D目标检测** 3D边界框的估计为自动驾驶应用提供了驱动力。Deep3Dbox使用了慢速RCNN的框架，首先检测2D对象，然后将每个对象输入3D估计网络。3D RCNN给Faster RCNN增加了一个额外的头部，然后增加了3D投影。Deepmanta使用了一个从粗到细的Faster RCNN来训练很多任务。我们的方法类似于Deep3Dbox或者3D RCNN的一阶段版本。因此，CenterNet比其他竞争方法要更简单并且更快。

![（a）基于锚框的检测器和（b）CenterNet](CenterNet-Object%20As%20Points.assets/image-20200529150041553.png)

## 3. 初步调研

$I\in R^{W\times H\times 3}$代表输入图片的宽度为W,高度为H。我们的目标是生成一个关键点热度图$\hat{Y}\in [0,1]^{\frac{W}{R}\times \frac{H}{R}\times C}$, 其中R是输出的跨步，C代表关键点类别的个数。在人体姿态关键点估计问题中，C=17。在目标检测问题中的类别领域，C=80.我们使用默认的stride=4.输出的步长用一个因子R对输出预测进行下采样。预测值$\hat{Y}_{x,y,c}=1$对应的是被检测到的关键点；$\hat{Y}_{x,y,c}=0$代表该位置是背景。我们使用了几个不同的去案卷集，编码解码格式的网络来从图像I中预测$\hat{Y}$：

- Hourglass Network
- 上卷积残差网络ResNet
- 深层聚集网络DLA

对于每个标注的类别$c$的关键点$p\in R^2$，我们计算出来一个低分辨率的等价量$\tilde{p}=\lfloor \frac{p}{R}\rfloor$。然后使用高斯核$Y_{xyz}=exp(-\frac{(x-\tilde{p_x})^2+(y-\tilde{p}_y)^2}{2\sigma^2_p})$将关键点散布到热图$Y\in[0,1]^{\frac{W}{R}\times \frac{H}{R} \times C}$。其中$\sigma_p$是一个目标大小自适应的标准偏差。如果相同的两个类别的高斯分布重合了，那就取其中最大的值作为最终结果。训练的目标就是减少关键点损失使用focal loss进行的像素级逻辑回归。

![](CenterNet-Object%20As%20Points.assets/image-20200529153729273.png)

其中$\alpha, \beta$是focal loss的超参数，N代表图像I中的关键点个数。选择N的标准化来讲所有的正样的focal loss的值设置为1。在这个实验中我们使用$\alpha=2, \beta=4$。为了恢复由于输出不符引起的离散化误差，额外为每个中心点都预测了一个局部偏移值$\hat{O}\in R^{\frac{W}{R}\times \frac{H}{R}\times 2}$。所有的类别C共享同一个偏移，偏移量使用的是L1损失函数进行训练：

$$
L_{off}=\frac{1}{N}\sum_{p}|\hat{O}_{\tilde{p}-(\frac{p}{R}-\tilde{p})}|
$$

监督仅仅作用在关键点坐标$\tilde{p}$，其他所有的位置会被忽视。

在下一节中，我们将展示如何将这个关键点估计器如何扩展到一个通用的对象检测器。

## 4. 目标作为点

令![img](https://img-blog.csdnimg.cn/20190417193433644.png) 是目标 k （其类别为 ![img](https://img-blog.csdnimg.cn/20190417193515283.png) ）的目标框. 其中心位置为 ![img](https://img-blog.csdnimg.cn/20190417193557489.png) ![img](https://img-blog.csdnimg.cn/20190417193622212.png)，我们用 关键点估计 ![img](https://img-blog.csdnimg.cn/20190417193709898.png)来得到所有的中心点，此外，为每个目标 k 回归出目标的尺寸 ![img](https://img-blog.csdnimg.cn/20190417193815238.png) 。为了减少计算负担，我们为每个目标种类使用单一的尺寸预测 ![img](https://img-blog.csdnimg.cn/20190417194110460.png) ，我们在中心点位置添加了 L1 loss:

![image-20200530075108780](CenterNet-Object%20As%20Points.assets/image-20200530075108780.png)

 我们不将scale进行归一化，直接使用原始像素坐标。为了调节该loss的影响，将其乘了个系数，整个训练的目标loss函数为：

![image-20200530075129718](CenterNet-Object%20As%20Points.assets/image-20200530075129718.png)

除非另有说明，否则我们在所有实验中都将$λ_{size}= 0.1$和$λ_{off}$= 1设置为。 我们使用单个网络来预测关键点$\hat{Y}$，偏移$\hat{O}$和大小$\hat{S}$。 网络预测每个位置的总共C + 4个输出。 所有输出共享一个共同的全卷积骨干网络。 对于每个模态，然后通过单独的3×3卷积，ReLU和另一个1×1卷积传递主干的特征。 图4显示了网络输出的概述。 第5节和补充材料包含其他体系结构细节。

**从点到目标框** 在推理的时候，我们分别提取热力图上每个类别的峰值点。我们将热力图上的所有响应点与其连接的8个临近点进行比较，如果该点响应值大于或等于其八个临近点值则保留，最后保留所有满足之前要求的前100个峰值点。令 ![img](https://img-blog.csdnimg.cn/2019041719581535.png) 是检测到的 c 类别的 n 个中心点的集合。![img](https://img-blog.csdnimg.cn/2019041719594952.png) 每个关键点以整型坐标 ![img](https://img-blog.csdnimg.cn/2019041720003485.png)的形式给出。![img](https://img-blog.csdnimg.cn/2019041720015874.png)作为测量得到的检测置信度， 产生如下的目标框:

![img](https://img-blog.csdnimg.cn/20190417200330996.png)

其中![img](https://img-blog.csdnimg.cn/20190417200406600.png)是偏移预测结果；![img](https://img-blog.csdnimg.cn/20190417200429484.png)是尺度预测结果；所有的输出都直接从关键点估计得到，无需基于IOU的NMS或者其他后处理。峰值关键点提取可作为NMS的替代方案，并且可以使用3×3max池化操作在设备上高效实现。

![我们网络用于不同任务的输出：顶部用于对象检测，中间用于3D对象检测，底部：用于姿势估计。 所有的模态都是从一个共同的主干产生的，具有由ReLU分隔的不同的3×3和1×1输出卷积。 括号中的数字表示输出通道。 有关详细信息，请参见第4节。](CenterNet-Object%20As%20Points.assets/image-20200530075642807.png)

### 4.1 3D 目标检测

3D检测是对每个目标进行3维bbox估计，每个中心点需要3个附加信息：depth, 3D dimension， orientation。我们为每个信息分别添加头.    对于每个中心点，深度值depth是一个维度的。然后深度很难直接回归！我们参考Depth map prediction from a single image using a multi-scale deep network对输出做了变换。![img](https://img-blog.csdnimg.cn/20190417201646862.png) 其中![img](https://img-blog.csdnimg.cn/20190417201722277.png)是sigmoid函数，在特征点估计网络上添加了一个深度计算通道 ![img](https://img-blog.csdnimg.cn/20190417201911258.png)， 该通道使用了两个卷积层，然后做ReLU 。我们用L1 loss来训练深度估计器。目标的3D维度是三个标量值。我们直接回归出它们（长宽高）的绝对值，单位为米，用的是一个独立的头: ![img](https://img-blog.csdnimg.cn/20190417202407570.png) 和L1 loss。方向默认是单标量的值，然而其也很难回归。我们参考3d bounding box estimation using deep learning and geometry用两个bins来呈现方向，且i做n-bin回归。特别地，方向用8个标量值来编码的形式，每个bin有4个值。对于一个bin,两个值用作softmax分类，其余两个值回归到在每个bin中的角度。

### 4.2 人体姿态估计

人的姿态估计旨在估计 图像中每个人的k 个2D人的关节点位置（在COCO中，k是17，即每个人有17个关节点）。因此，我们令中心点的姿态是 kx2维的，然后将每个关键点（关节点对应的点）参数化为相对于中心点的偏移。 我们直接回归出关节点的偏移（像素单位） ![img](https://img-blog.csdnimg.cn/2019041720380753.png)，用到了L1 loss；我们通过给loss添加mask方式来无视那些不可见的关键点（关节点）。此处参照了slow-RCNN。为了refine关键点（关节点），我们进一步估计k 个人体关节点热力图 ![img](https://img-blog.csdnimg.cn/20190417204418510.png) ，使用的是标准的bottom-up 多人体姿态估计,我们训练人的关节点热力图使用focal loss和像素偏移量，这块的思路和中心点的训练雷同。我们找到热力图上训练得到的最近的初始预测值，然后将中心偏移作为一个聚类的线索，来为每个关键点（关节点）分配其最近的人。具体来说，令![img](https://img-blog.csdnimg.cn/2019041720505314.png)是检测到的中心点。第一次回归得到的关节点为：![img](https://img-blog.csdnimg.cn/20190417205156515.png)

我们提取到的所有关键点（关节点，此处是类似中心点检测用热力图回归得到的，对于热力图上值小于0.1的直接略去）：![img](https://img-blog.csdnimg.cn/20190417205534789.png) 对于对应的热度图 ![img](https://img-blog.csdnimg.cn/20190417205709453.png)

然后将每个回归（第一次回归，通过偏移方式）位置 ![img](https://img-blog.csdnimg.cn/20190417205857291.png) 与最近的检测关键点（关节点）进行分配![img](https://img-blog.csdnimg.cn/20190417210117229.png) ，考虑到只对检测到的目标框中的关节点进行关联。

## 5. 实现细节

实验了4个结构：ResNet-18, ResNet-101, DLA-34， Hourglass-104. 我们用可变形卷积层来更改ResNets和DLA-34，按照原样使用Hourglass 网络。

**Hourglass** 堆叠的Hourglass网络通过两个连续的hourglass 模块对输入进行了4倍的下采样，每个hourglass 模块是个对称的5层 下和上卷积网络，且带有skip连接。该网络较大，但是关键点估计效果很好。

**ResNet** Xiao等人对标准的ResNet做了3个转置卷机网络来做到更高的分辨率输出（最终stride为4）。为了节省计算量，我们改变这3个转置卷积的输出通道数分别为256,128,64。转置卷积核初始为双线性插值。

**DLA** 即Deep Layer Aggregation (DLA)，是带多级跳跃连接的图像分类网络，我们采用全卷积上采样版的DLA，用可变形卷积来跳跃连接低层和输出层；将原来上采样层的卷积都替换成3x3的可变形卷积。在每个输出head前加了一个3x3x256的卷积，然后做1x1卷积得到期望输出。

![Table 1 在COCO验证集上，要为不同网络权衡速度/准确性。 我们显示的结果没有测试增强（N.A.），翻转测试（F）和多尺度增强（MS）](CenterNet-Object%20As%20Points.assets/image-20200530082052847.png)

**训练** 我们以512×512的输入分辨率进行训练，这对于所有模块都产生128×128的输出分辨率。 我们使用随机翻转，随机缩放（介于0.6到1.3之间），裁剪和颜色抖动作为数据增强，并使用Adam优化整体目标。 当裁切或缩放更改3D测量值时，我们使用增补训练3D估计分支。 对于残差网络和DLA-34，我们以128个批次的规模（在8个GPU上）进行训练，学习率5e-4持续140个时期，学习率分别在90和120个时期下降了10倍 ）。 对于Hourglass-104，我们遵循ExtremeNet，并使用批处理大小为29（在5个GPU上，主批处理为GPU大小为4）和50个epoch的学习率2.5e-4，而学习率下降10倍。 40个epoch。 为了进行检测，我们对ExtremeNet的Hourglass-104进行了微调，以节省计算量。 Resnet-101和DLA-34的下采样层使用ImageNet预训练进行初始化，而上采样层则是随机初始化。 Resnet-101和DLA-34在8个TITAN-V GPU上训练需要2.5天，而Hourglass-104需要5天。

**推理** 我们使用三个级别的测试增强：增强，翻转增强以及翻转和多尺度（0.5、0.75、1、1.25、1.5）。 对于翻转，我们在解码边界框之前对网络输出进行平均。 对于多尺度，我们使用NMS合并结果。 这些增强产生了不同的速度精度折衷，如下一节所示。



## 6. 实验

我们在MS COCO数据集上评估了目标检测性能，该数据集包含118k训练图像（train2017），5k验证图像（val2017）和20k保持测试图像（test-dev）。 我们报告所有IOU阈值（AP），AP在IOU阈值0.5（AP50）和0.75（AP75）时的平均精度。 该补充包含有关PascalVOC的其他实验。

### 6.1 目标检测

表1显示了我们使用不同主干和测试选项进行的COCO验证的结果，而图1将CenterNet与其他实时检测器进行了比较。 运行时间已在我们的本地计算机上进行了测试，并使用Intel Corei7-8086K CPU，Titan Xp GPU，Pytorch 0.4.1，CUDA 9.0和CUDNN 7.1。 我们下载代码和经过预训练的models12在同一台机器上测试每种型号的运行时间。Hourglass-104以相对较高的速度获得了最佳的精度，在7.8FPS中的AP为42.2％。 在此骨干网上，CenterNet在速度和准确性方面均胜过CornerNet 40.6％APin4.1FPS）和ExtremeNet 40.3％AP in3.1FPS）。 运行时间的改善来自更少的输出扬程和更简单的盒解码方案。 更好的精度表示中心点比角点或极端点更容易检测。使用ResNet-101，我们在相同的网络主干网络上优于RetinaNet  我们仅在向上采样层中使用可变形卷积，这不会影响RetinaNet。 在相同的精度下，我们的速度是以前的两倍（45FPS中的CenterNet34.8％AP（输入512×512）与RetinaNet在18 FPS中的34.4％AP（输入500×800）相比）。 我们最快的ResNet-18模型在142FPS时也可实现28.1％的COCO AP的可观性能。DLA-34提供了最佳的速度/精度权衡。 它的运行速度为52FPS，AP为37.4％。 这是YOLOv3 4.4％AP准确度的两倍以上。 通过翻转测试，我们的模型仍然比YOLOv3更快，并达到了Faster-RCNN-FPN的准确性水平（28FPS中的CenterNet39.2％AP与11FPS中的Faster-RCNN39.8％AP）。 

![最新的COCO测试开发比较。 顶部：两级检测器； 底部：一级探测器。 我们展示了大多数一级检测器的单尺度/多尺度测试。 尽可能在同一台机器上测量每秒帧数（FPS）。 斜体FPS突出显示了从原始出版物中复制绩效衡量指标的情况，破折号表示没有代码和模型，也没有公开时间表的方法。](CenterNet-Object%20As%20Points.assets/image-20200530083012674.png)

**最先进的比较** 我们在表2中与COCO test-dev中的其他最先进的检测器进行了比较。通过多尺度评估，配备Hourglass-104的CenterNet的AP达到45.1％，优于所有现有复杂的两级检测器精度更高，但速度也较慢。 对于不同的物体大小或IoU阈值，CenterNet和滑动窗口检测器之间没有显着差异。 CenterNet的行为类似于常规检测器，只是速度更快。

#### 6.1.1 额外实验

在不幸的情况下，如果两个不同的物体完美地排列在一起，它们可能会共享同一个中心。 在这种情况下，CenterNet 只能检测到其中的一个。 我们首先研究这种情况在实践中发生的频率，并将其与竞争方法的缺失检测联系起来。

**中心点碰撞** 在 COCO 训练集中，有614对物体在相同的中心点相撞。 总共有860001件目标，因此由于中心点的碰撞，CenterNet无法预测<0.1％的对象。 这比因区域建议不完善而导致的慢速或快速RCNN的失误要少2％，比因锚点放置不足而导致的基于锚的方法的失误要少（对于Faster- RCNN具有15个锚点，阈值为0.5IOU）。 此外，715对对象的边界框IoU> 0.7，并将它们分配给两个锚点，因此基于中心的分配会减少碰撞。

**非极大抑制算法** 为了验证CenterNet不需要基于IoU的NMS，我们将其作为预测的后处理步骤来运行。 对于DLA-34（翻转测试），AP将从39.2％提高到39.7％。对于Hourglass-104，AP保持在42.2％。 考虑到较小的影响，我们不使用它。接下来，我们消除模型的新超参数。 所有实验均在DLA-34上完成。

**训练和测试分辨率** 在训练过程中，我们将输入分辨率固定为512×512。 在测试过程中，我们遵循CornerNet持原始图像分辨率，并将输入零填充到网络的最大跨度。对于ResNet和DLA，我们将图像填充多达32个像素，对于HourglassNet，我们使用128 像素。 如表所示。 如图3a所示，保持原始分辨率比固定测试分辨率稍好。 较低分辨率（384×384）的训练和测试运行速度快了1.7倍，但下降了3AP。

**回归损失** 我们将原始L1损失与平滑L1进行比较进行大小回归。 我们在Table 3c中进行的实验表明，L1比Smooth L1好得多。COCO evaluation度量标准敏感的精细范围内，它可以产生更好的精度。 在关键点回归中可以独立观察到这一点

![消除COCO验证集上的设计选择。 结果以COCO AP显示，时间以毫秒为单位。](CenterNet-Object%20As%20Points.assets/image-20200530084500849.png)

**边界框大小权重** 我们分析了我们的方法对损失权重λsize的敏感性。 表3b显示了0.1个良好的结果。 对于较大的值，由于损耗的范围从0到输出大小w / R或者h / R，而不是从0到1，因此AP显着降低。 但是，对于较低的权重，该值不会显着降低。

**训练策略** 默认情况下，我们将关键点估计网络训练140个纪元，而学习率下降90个纪元。 如果我们在降低学习率之前将训练时间增加一倍，则性能会进一步提高1.1AP（表3d），但需要更长的训练时间表。 为了节省计算资源，我们在消融实验中使用了140个epoch，但与其他方法相比，DLA坚持使用了230个epoch。最后，我们通过回归到多个对象大小来尝试使用多个“锚定”版本的Center-Net。 实验没有取得任何成功。 见补充。

### 6.2 3D检测

我们在KITTI数据集上执行3D边界框估计实验，该数据集包含在驾驶场景中为车辆精心标注的3D边界框。 KITTI包含7841个训练图像，我们遵循文献中的标准训练和验证拆分。 评估指标是在IOU阈值0.5时，在11次召回时汽车的平均精度（0.0到1.0，以0.1为增量，以0.1为增量），如物体检测中所述。 我们基于2D边界框（AP），方向（AOP）和鸟瞰边界框（BEV AP）评估IOU。 为了训练和测试，我们将原始图像分辨率和填充保持为1280×384。训练收敛于70个时期，学习率分别下降了45和60个epoch。 我们使用DLA-34主干并将深度，方向和尺寸的损失权重设置为1。 所有其他超参数与检测实验相同。

训练在70个epoch收敛，学习率分别下降了45和60个epoch。 我们使用DLA-34主干并将深度，方向和尺寸的损失权重设置为1。 所有其他超参数都与检测实验相同。由于召回阈值的数量非常小，因此验证AP的波动幅度最大为10％AP。 因此，我们训练了5个模型并报告了具有标准偏差的平均值。我们将它们基于特定验证的分割与基于慢RCNN的Deep3DBox和基于Faster-RCNN的方法Mono3D进行了比较。 如表4所示，我们的方法在AP和AOS中的表现与其他方法相当，而在BEV方面则略胜一筹。 我们的CenterNet比这两种方法快两个数量级。

### 6.3 姿态估计

最后，我们在MS COCO数据集中评估CenterNet的人体姿势估计[34]。 我们评估keypointAP，它类似于边界框AP，但用对象关键点相似度替换边界框IoU。 我们在COCO test-dev上测试并与其他方法进行比较。我们对DLA-34和Hourglass-104进行了实验，两者均从中心点检测进行了微调。 DLA-34收敛于320个epoch（在8个GPU上大约3天），而Hourglass-104收敛于150个epoch（在5个GPU上8天）。 所有其他损失权重都设置为1。 所有其他超参数与对象检测相同。结果如表5所示。对关键点的直接回归性能合理，但不是最先进的。 特别是在高IoU体制下，这种斗争尤其艰巨。 将我们的输出投影到最接近的关节检测可以改善整个结果，并与最新的多人姿势估计器[4、21、39、41]竞争。 这证明CenterNet是通用的，易于适应新任务。图5显示了所有任务的定性示例。

![KITTI评估。 我们在不同的验证分割上显示了2D边界框AP，平均方向得分（AOS）和鸟瞰（BEV）AP。 越高越好。](CenterNet-Object%20As%20Points.assets/image-20200530085832048.png)

![定性结果。 在不考虑算法性能的情况下，对所有图像进行了主题选择。第一行：COCO验证中的目标检测。 对于每一对，我们显示中心偏移回归（左）和热图匹配（右）的结果。第四行和第五行：基于KITTI验证的3D边界框估计。 我们显示了投影的边界框（左）和鸟瞰图（右）。 地面真相检测显示在红色实心方框中。 中心热图和3D框显示为覆盖在原始图像上。](CenterNet-Object%20As%20Points.assets/image-20200530085856326.png)

![COCO test-dev上的关键点检测。 -reg / -jd分别用于直接中心偏心偏移回归和与最接近的关节检测匹配的回归。 结果显示在COCO关键点AP中。 越高越好。](CenterNet-Object%20As%20Points.assets/image-20200530085923884.png)

## 7. 结论

总之，我们提出了一种新的对象表示形式：点。 我们的CenterNet对象检测器建立在成功的关键点估计网络上，找到对象中心，并逐步缩小其大小。 该算法简单，快速，准确且端到端可区分，无需任何NMS后处理。 除了简单的二维检测，这种想法是通用的，并且具有广泛的应用。 CenterNet可以通过一次向前传递来估计一系列其他对象属性，例如姿势，3D方向，深度和范围。 我们的初步实验令人鼓舞，并为实时对象识别和相关任务开辟了新的方向。







