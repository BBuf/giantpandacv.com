【GaintPanda导语】这是关于GiraffeDet的论文详读，该论文提出以S2D Chain为组合模块，构建light backbone，再以Queen Fuse和Skip Connect构建GFPN作为颈部模块，与以往检测器的backbone>neck（FLOPS）的构建方式不同，GiraffeDet的neck在参数量和计算量上远超backbone。

论文名称《GiraffeDet: A Heavy-Neck Paradigm for Object Detection》

论文地址：https://arxiv.org/pdf/2202.04256.pdf

## **摘要** 

在传统的目标检测框架中，模型从骨干提取深层潜在特征，然后由颈部模块融合这些潜在特征，捕获不同尺度的信息。由于目标检测的对分辨率的要求比图像识别的要大得多，因此骨干网的计算成本往往占据了大部分推理成本。这种重骨干的设计范式在传统图像识别往目标检测发展时遗留了下来，但这种范式并不是针对目标检测的端到端优化设计。在这项工作中，我们证明了这种范式确实只能产生次优的目标检测模型。为此，我们提出了一种新的重颈设计范式，GiraffeDet，一种类似长颈鹿的有效物体检测网络。GiraffeDet 使用了一个非常轻的主干和一个非常深和大的颈部模块，这种结构可进行不同空间尺度以及不同级别潜在语义的密集信息交换。这种设计范式帮助检测器在网络的早期阶段以相同的优先级处理高级语义信息和低级空间信息，使其在检测任务中更有效。多个流行的检测基准测试评估表明，它始终优于以前的 SOTA 模型。 

## **介绍** 

在过去的几年中，基于深度学习的目标检测方法取得了显著的进展。尽管目标检测网络在架构设计、训练策略等方面变得越加强大，但检测对于large-scale变化的目标并没有改变。为此，我们通过设计一个有效稳健的方法来解决这个问题。为了缓解由large-scale变化引起的问题，一种直观的方法是使用多尺度金字塔策略来进行训练和测试。虽然这种方法提高了大多数现有cnn检测器的检测性能，但它并不实用，因为图像金字塔方法需要处理每个不同比例的图像，计算比较昂贵。后来，提出了特征金字塔网络，近似图像金字塔的方式但成本更低。**近期的研究仍然依赖于优越的主干设计**，但这会使得高级特征与低级特征之间的信息交换不足。

根据上述挑战，在本任务中提出了以下两个问题： 

*图像分类任务的主干在检测模型中是不可缺少吗？* 

*哪些类型的多尺度表达对检测任务有效？* 

这两个问题促使我们设计了一个新的框架，其中包括两个子任务，即有效的特征降采样和充分的多尺度融合。首先，用于提取特征的传统骨干计算成本昂贵，并且存在domain-shift的问题。其次，检测器在高级语义和低级空间特征之间的信息融合至关重要。根据上述存在的现象，我们设计了一个类似长颈鹿的网络，名为GiraffeDet，具有以下特点：

（1）一种新的轻量级骨干可以提取多尺度特征，而无需较大的计算成本。 

（2）足够的交叉尺度连接--皇后融合，就像国际象棋中的皇后路径，来处理不同层次的特征融合。

（3）根据设计的轻量级骨干和灵活的FPN，我们列出了每个GiraffeDet系列类型的FLOPs，实验结果表明，我们的 GiraffeDet 系列在每个 FLOPs上都取得了更高的准确性和更好的效率。 

综上所述，我们工作的主要贡献如下： 

• 据我们所知，我们是第一个提出轻量级替代骨干和灵活FPN的组合作为检测器的团队。提出的GiraffeDet家族由轻量级**S2D-chain**和**Generalized-FPN**组成，展示了最先进的性能。 

• 我们设计了轻量级的空间深度链 (**S2D-Chain**)，而不是基于传统的CNN主链，实验表明，在目标检测模式下，FPN的作用比传统的骨干更重要。 

• 基于前面我们提出的**Generalized-FPN(GFPN)**，提出了一种新的皇后融合作为我们的跨尺度连接，它融合了前层和当层的层次特征，以及n个跳跃层链路来提供更有效的信息传输，这种方式可以扩展到更深的结构。基于轻主干和重颈部的设计范式，GiraffeDet 家族模型在FLOPs-性能权衡中表现良好。GiraffeDet-D29 在 COCO 数据集上达到了 54.1%的 mAP，并且优于其他SOTA模型。

## **相关工作** 

通过学习尺度特征来识别目标是定位目标的关键。large-scale问题的传统解决方案主要还是基于改进的CNN网络。基于CNN的目标探测器主要分为两级探测器和一级探测器。近年来，主要的研究路线是利用金字塔策略，包括图像金字塔和特征金字塔。图像金字塔策略通过缩放图像来检测实例。例如，Singhetal在2018年提出了一种快速的多尺度训练方法，该方法对真实物体周围的前景区域和背景区域进行采样，进行不同尺度的训练。与图像金字塔方法不同，特征金字塔方法融合了不同尺度和不同语义信息层的金字塔表达。例如，PANet通过额外的自下而上的路径来增强特征金字塔网络顶部的特征层次结构。此外，NAS-FPN利用神经结构自动搜索来探索特征金字塔网络拓扑。我们的工作重点是特征金字塔策略，并提出了一种高级语义和低层次空间信息融合方法。一些研究人员开始设计新的CNN架构来解决large-scale的问题，FishNet通过设计跳跃连接的编码器-解码器架构来融合多尺度特征。SpineNet被设计为一个主干+具有尺度排列的中间特征+跨尺度连接的方式，通过神经结构搜索进行学习。我们的工作受到了这些方法的启发，因此提出了一个轻量级的空间深度骨干，我们的网络设计**轻骨干重颈头**的体系结构，在检测任务中被证明是有效的。

## **3、THE GIRAFFEDET**

large-scale仍是一个挑战，为了充分有效地进行多尺度信息交换，我们提出了用于目标检测的GiraffeDet，整个框架如图 1 所示，它大体上遵循了一阶段检测器的范式。

![](https://img-blog.csdnimg.cn/a856835d6cfb4c8aaa348d4a882ad5d1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

#### **3.1 LIGHTWEIGHT SPACE-TO-DEPTH CHAIN（S2D chain）**

大多数特征金字塔网络采用传统的CNN网络作为骨干，提取多尺度特征图，从而进行信息交换。然而，随着 CNN 的发展，骨干变得更加沉重，使用它们的计算成本是昂贵的。此外，骨干主要是在分类数据集上进行预训练的，例如，在 ImageNet 上进行预训练的ResNet50，我们认为这些预训练的骨干在检测任务中不合适，仍然是domain-shift的问题。相比之下，FPN更注重高级的语义交换和低层次的空间信息交换。因此，我们假设FPN在目标检测模型中比传统骨干更加重要。

我们提出了空间深度链（S2D 链）作为我们的轻量级骨干，其中包括两个 3x3 卷积网络和堆叠的 S2D Block。 具体来说，3x3 卷积用于初始下采样并引入更多非线性变换。 每个 S2D Block由一个 S2D 层和一个 1x1 卷积组成。 S2D层通过均匀采样和重组将空间维度信息移动到更深维度，**在没有额外参数的情况下对特征进行下采样**。 然后使用 1x1 卷积来提供通道池化以生成固定维度的特征图。

![](https://img-blog.csdnimg.cn/1467844edd91484591186ebc7a86a2b6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/694ccd03d55240dea98533687a008fa5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

为了验证我们的假设，我们在第 4 节中对相同的目标进行检测，对不同的主干和颈部进行了控制实验。结果表明，颈部在目标检测任务中比传统的骨干更重要。 

#### **3.2 GENERALIZED-FPN**

在特征金字塔网络中，多尺度特征融合的目的是聚合从骨干网络中提取的不同feature map。图3显示了特征金字塔网络设计的演变过程。传统的 FPN引入了一个自上而下的路径，将从第 3 级到第 7 级的多尺度特征进行融合。考虑到单向信息流的局限性，PANet增加了一个额外的自下而上的路径聚合网络，但计算成本更大。此外，BiFPN删除了只有一条输入边的节点，并在同一级别上从原始输入中添加了额外的边。然而，我们观察到，以前的方法只关注特征融合，而缺乏内部块连接。因此，我们设计了一种新的路径融合，包括跳跃层和交叉尺度连接，如图 3(d).所示。

![](https://img-blog.csdnimg.cn/ae043928be654bfd82054f62b399c882.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

**跳跃层连接**.与其他连接方法相比，在反向传播过程中，跳跃连接在特征层之间的距离较短。为了减少heavy-neck的梯度消失，我们提出了两种特征链接方法： $dense-link$和$log_2n-link$，如图4所示：

![](https://img-blog.csdnimg.cn/578bcc13e1294bf89eea43e794bdfb29.png)

**dense-link：**受 DenseNet的启发，对于k层的每个尺度特征$P_k^l$，第$l$层接收前面所有图层的特征信息：

![](https://img-blog.csdnimg.cn/007393caa1cb47078e62794809cc3f28.png)

其中 $Concat()$指的是在前面所有层中产生的特征映射的连接，$Conv()$表示 3x3 卷积。

$log_2n-link$：具体地说，对于$k$层结构，第$l$层最多接收$log_2l + 1$ 的特征信息，这些输入层与深度 $i$ 成指数分离关系，基数为2：

![](https://img-blog.csdnimg.cn/cb7e380506834b3d890a0c72430a76a8.png)

其中 $l-2^n≥0$、$Concat()$和 $Conv()$也分别表示连接和 3x3 卷积。与深度 $l$ 处的dense-link相比，复杂度只花费了 $O(l·log_2l)$，而不是 $O(l)$。此外，在反向传播过程中，$log_2n-link$将层间距离从 1增加到 $1+log_2l.$。 $1+log_2l.$可以扩展到更深层次的网络。

**跨尺度连接：**基于我们的假设，我们设计的信息交换模块不仅应该包含跳跃层连接，还应该包含跨尺度连接，以克服多尺度的变化。因此，我们提出了一种新的跨尺度融合，称为皇后融合，即考虑如图 3(d)所示的同层和邻层的特征。如图 5(b)所示的一个例子，皇后融合的连接包括前一层的下采样，本研究中，我们分别采用双线性插值和最大池化作为上采样和下采样函数。因此，在极端尺度变化的情况下，该模型需要具有足够的高、低层次的信息交换。基于我们的跳跃层和跨尺度连接的机制，我们提出的Generalized-FPN可以尽可能地扩展，就像“长颈鹿颈部”一样。有了这样“沉重的脖子和轻质的脊梁，我们的GiraffeDet可以取得更高的精度和更好的效率。”

![](https://img-blog.csdnimg.cn/cb789d4fdadd4464be82a43949eefcae.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

#### **3.3 GIRAFFEDET FAMILY**
根据我们提出的S2D-chain和Generalized-FPN，我们可以开发一系列不同的GiraffeDet模型。以往的工作以低效的方式扩大检测器的性能，如改变更大的骨干网络，如 ResNeXt，或堆叠 FPN 块。特别地，EffificientDet开始使用共同复合系数*$φ$*来扩大主干的所有维度。与EffificientDet不同的是，我们只关注 GFPN 层的缩放，而不是包括轻量级骨干的整个框架。具体地说，我们应用了两个系数*$φ_d$* 和 $φ_w$ 可以灵活地调整 GFPN的深度和宽度。

![](https://img-blog.csdnimg.cn/07868975dbf5465e9af81b0331efe21c.png)

遵循上述等式。我们开发了六种 GiraffeDet 架构，如表 1 所示。D7、D11、D14、D16 与 resnet 系列模型具有相持的水平，我们将在下一节中对GiraffeDet家族与 SOTA 模型的性能进行比较。请注意，GFPN 的图层与其他FPN 设计不同，如图 3 所示。在我们提出的 GFPN 中，每一层代表一个深度，而 PANet 和 BiFPN 一层包含两个深度。

![在这里插入图片描述](https://img-blog.csdnimg.cn/831a5dfc68fc430a832885cc37255580.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_12,color_FFFFFF,t_70,g_se,x_16)

在本节中，我们首先介绍实现细节，并给出我们在 COCO 数据集上的实验结果。然后将我们提出的GiraffeDet家族与其他最先进的方法进行比较，最后提供一个深入全面的分析，以更好地理解我们的网络。 

**4.1 数据集和实施细节** 

为了进行公平的比较，所有的结果都是在mmdetection框架和标准的 coco 式评估方案下进行。所有模型都从头开始进行训练，以减少骨干对ImageNet 的影响。输入图像的短边被调整到 800，最大尺寸被限制在 1333 范围内。为了提高训练的稳定性，我们对所有模型采用多尺度训练，包括：在 R2-101-DCN 主干实验中使用 2x imagenet-pretrained (p-2x) 训练方案（24 epoch，在 16 和 22 epoch 衰减），3x  scratch（s-3x）训练方案（36 epoch，在 28 和 33衰减）和目前SOTA网络比较中的 6x Scratch (s-6x) 的训练方案（72 epochs，在 65 和 71 epochs 衰减）。 

![在这里插入图片描述](https://img-blog.csdnimg.cn/59c3222a8e834506b586f4a8792a6725.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

#### **4.2  COCO 数据集的评价**

为了公平的比较，我们还使用了RetinaNet、FCOS、HRNet、GFLV2等模型，进行了 6 次训练，记为七大方差。根据图 6 的性能，我们可以观察到我们提出的 GiraffeDet 在每个像素尺度范围内都取得了最好的性能，这表明轻主干和重颈部的设计范式以及我们提出的GFPN可以有效地解决大尺度方差问题。此外，在跳跃层和跨尺度连接下，可以实现高级语义信息和低级空间信息的充分交换。许多实例小于图像面积的1%，这使得很难被检出，但我们的方法在像素 0-32 范围内仍然比RetinaNet高5.7个map，在中间像素80-144范围内具有相同的 map。值得注意的是，在像素为 192-256 的 范 围 内 ， 所提出的GiraffeDet性能优于其他方法，这证明了我们的设计可以有效地学习对不同尺度的特征。

![](https://img-blog.csdnimg.cn/ba4862d7ca474c0f984562e34d2d023d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_17,color_FFFFFF,t_70,g_se,x_16)


从表 2 可以看出，我们的 GiraffeDet 对比每个相同级别的探测器具有更好的性能，这表明我们的方法可以有效地检测目标。

1)与基于 resnet 的低水平 FLOPs 尺度上的方法相比，我们发现，即使总体性能没有明显提高太多，但我们的方法在检测小对象情况和大对象情况方面都有显著的性能。这表明我们的方法在大尺度变化的数据集上表现更好。

2)与基于 ResNexts 的方法相比，我们发现 GiraffeDet 比低水平 FLOPs 具有更高的性能，**这表明良好的 FPN 设计比主干更重要。**

3)与其他方法相比，所提出的GiraffeDet也具有SOTA性能，证明了我们的设计在每个FLOPs 水平上都获得了更高的精度和更高的效率。此外，基于NAS的方法在训练过程中消耗了大量的计算资源，因此我们不考虑与我们的方法进行比较。最后，**通过多尺度测试方案，我们的 GiraffeDet 达到了 54.1%的 mAP，特别是 $APs$ 增加了2.8%， $APl$增长 2.3%，远远超过$APm$增加了1.9%。**

![](https://img-blog.csdnimg.cn/7a37a64ac71947839c4d51abd4c53f59.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

**深度和宽度的影响**.为了进一步与不同的neck进行公平比较，我们在同一 FLOPs 水平上对FPN、PANet 和 BiFPN 进行了两组实验比较，以分析我们提出的**Generalized-FPN**中深度和宽度的有效性。请注意，如图 3 所示，我们的GFPN每一层都包含一个深度，而 PANet和 BiFPN 的每一层都包含两个深度。

![](https://img-blog.csdnimg.cn/a9bd558c32494a659f175337536ff0ea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)


如表 4 所示，我们观察到我们提出的 GFPN 在各种深度和宽度上都优于其他FPN，这也表明 $log_2n-link$ 和皇后融合可以有效地提供信息传输和交换。此外，我们提出的 GFPN 可以在更小的设计中实现更高的性能。

**骨干效应。**图 7 显示了不同neck深度和不同backbones在同一 FLOPs 水平上的性能。结果表明，S2D-chain 和 GFPN 的组合优 于其他骨干模型，这可以验证我们的假设，**即 FPN 更关键，传统的骨干不会随着深度的增加而提高性能。**特别是，性能甚至随着主干的变重而下降。这可能是因为*domain-shift*问题在一个大主干中变得更严重。

![](https://img-blog.csdnimg.cn/d0a5ea1dcda34b659f7205bc268fcee8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_19,color_FFFFFF,t_70,g_se,x_16)

**添加DCN的结果**

表 5：GiraffeDet-D11应用可变形卷积网络的结果（val 2017）。 *‡*表示使用多 gpu 训练的 SyBN GFPN。

![](https://img-blog.csdnimg.cn/94903db9ce0547ecacb8e886235f6ea2.png)

在我们的 GiraffeDet 中引入可变形卷积网络(DCN)，该网络近期被广泛用于提高检测性能。如表 5 所示，我们观察到 DCN 可以显著提高GiraffeDet的性能。特别是根据表2，GiraffeDet-D11比GiraffeDet-D16 具有更好的性能。在可接受的推理时间下，我们观察到具有DCN主干和浅层GFPNTiny可以提高性能，并且性能随着GFPN 深度的增长而大幅提高，如表 6 所示。

表 6：具有多个 GFPN 的 Res2Net-101-DCN(R2-101-DCN)骨干的结果（val-2017）。GFPNtiny 指深度为8，宽度为 122(与 FPN 的 FLOPs 级别相同)的GFPN。

![](https://img-blog.csdnimg.cn/41a6c2f9d8c14aa8b2bee4cbdb8baac4.png)

**5 结论** 

在本文中，我们提出了一个新的设计范式，GiraffeDet，一个类似长颈鹿的网络，以解决large-scale变化的问题。特别是，GiraffeDet使用一个轻量级的空间深度链作为骨干，Generalized-FPN作为neck。采用轻量级空间深度链提取多尺度图像特征，GFPN来处理高级语义信息和低层次空间信息交换。大量的结果表明，所提出的GiraffeDet实现了更高的精度和更好的 效率，特别是检测小和大的对象。
![在这里插入图片描述](https://img-blog.csdnimg.cn/59819a412842458590427c16dc81b2e0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

![在这里插入图片描述](https://img-blog.csdnimg.cn/0dc19044cd534a419c37774184acb656.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

随机测试效果如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/0c18bd1b44b34794adf5d5d60ad9a907.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAcG9nZ18=,size_20,color_FFFFFF,t_70,g_se,x_16)

