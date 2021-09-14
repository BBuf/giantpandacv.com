# 前言
今天开始分享一下YOLO系列的目标检测算法，前面介绍了SSD算法和Faster-RCNN，现在公司用Faster-RCNN的似乎不是很多，主要集中在YOLO，SSD以及CenterNet等。我们的检测和宇宙和分割宇宙刚刚开始，之后会更新一些这些算法的代码实战等，敬请期待吧。
# 创新点
- 将整张图作为网络的输入，直接在输出层回归bounding box的位置和所属类别。
- 速度快，One-Stage检测算法开山之作。

# 介绍
回顾YOLO之前的目标检测算法，都是基于产生大量可能包含物体的先验框，然后用分类器判断每个先验框对应的边界框里是否包含待检测物体，以及物体所属类别的概率或者置信度，同时需要后处理修正边界框，最后基于一些准则过滤掉置信度不高和重叠度较高的边界框，进而得到检测结果。这种基于先产生候选区域再进行检测的方法虽然有较高的精度，但速度非常慢。YOLO直接将目标检测堪称一个回归问题进行处理，将候选区和检测两个阶段合二为一。YOLO的检测过程如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122144238960.png)

		Fig1
事实上，YOLO并没有真正的去掉候选区，而是直接将输入图片划分成7x7=49个网格，每个网格预测两个边界框，一共预测49x2=98个边界框。可以近似理解为在输入图片上粗略的选取98个候选区，这98个候选区覆盖了图片的整个区域，进而用回归预测这98个候选框对应的边界框。


# 原理介绍
YOLO将输入图像划分为S*S的栅格，每个栅格负责检测中心落在该栅格中的物体。每一个栅格预测B个bounding boxes，以及这些bounding boxes的confidence scores。这个 confidence scores反映了模型对于这个栅格的预测：该栅格是否含有物体，以及这个box的坐标预测的有多准。公式定义如下： 

$confidence = Pr(Object)*IOU_{pred}^{truth}$

如果这个栅格中不存在一个object，则confidence score应该为0。相反，confidence score则为预测框与真实框框之间的交并比。

YOLO对每个bounding box有5个predictions：x, y, w, h,和 confidence。坐标x,y代表了预测的bounding box的中心与栅格边界的相对值。

坐标w,h代表了预测的bounding box的width、height相对于整幅图像width,height的比例。
confidence就是预测的bounding box和ground truth box的IOU值。 每一个栅格还要预测C个conditional class probability（条件类别概率）：Pr(Classi|Object)。即在一个栅格包含一个Object的前提下，它属于某个类的概率。我们只为每个栅格预测一组（C个）类概率，而不考虑框B的数量。如Fig2所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122181032745.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70) 

		Fig2 
YOLO将检测模型化为回归问题。 它将图像划分为S×S网格，并且每个网格单元预测B个边界框，对这些框的置信度以及C类概率。 这些预测值被编码为S×S×（B * 5 + C）张量。为了评估PASCAL VOC上的YOLO，我们使用S = 7，B = 2。PASCAL VOC有20个标记类，因此C = 20。我们的最终预测是7×7×30张量。

# 网络结构
我们将此模型作为卷积神经网络实施并在PASCAL VOC检测数据集上进行评估。 网络的初始卷积层从图像中提取特征，而全连接的层预测输出概率和坐标。

YOLO网络借鉴了GoogLeNet分类网络结构。不同的是，YOLO未使用inception module，而是使用1x1卷积层（此处1x1卷积层的存在是为了跨通道信息整合）+3x3卷积层简单替代。完整的网络结构如Fig3所示，最终的输出结果是一个7*7*30的张量。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122181235694.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 训练

首先利用ImageNet 1000-class的分类任务数据集Pretrain卷积层。使用上述网络中的前20 个卷积层，加上一个 average-pooling layer，最后加一个全连接层，作为 Pretrain 的网络。训练大约一周的时间，使得在ImageNet 2012的验证数据集Top-5的精度达到 88%，这个结果跟 GoogleNet 的效果相当。 

将Pretrain的结果的前20层卷积层应用到Detection中，并加入剩下的4个卷积层及2个全连接。同时为了获取更精细化的结果，将输入图像的分辨率由 224* 224 提升到 448* 448。 
将所有的预测结果都归一化到 0~1, 使用 Leaky RELU 作为激活函数。 Leaky RELU的公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122183122683.png)

Leaky RELU可以解决RELU的梯度消失问题。

损失函数的设计目标就是让坐标（x,y,w,h），confidence，classification 这个三个方面达到很好的平衡。

简单的全部采用了sum-squared error loss来做这件事会有以下不足：
a) 8维的localization error和20维的classification error同等重要显然是不合理的。
b) 如果一些栅格中没有object（一幅图中这种栅格很多），那么就会将这些栅格中的bounding box的confidence 置为0，相比于较少的有object的栅格，这些不包含物体的栅格对梯度更新的贡献会远大于包含物体的栅格对梯度更新的贡献，这会导致网络不稳定甚至发散。 为了解决这些问题，YOLO的损失函数的定义如下：

![这里写图片描述](https://img-blog.csdn.net/20180613153803701?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

网上找到一张详细的损失函数解释图：

![这里写图片描述](https://img-blog.csdn.net/20180613153914440?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

YOLO的损失函数更重视8维的坐标预测，给这些损失前面赋予更大的loss weight, 记为 λcoord ,在pascal VOC训练中取5。（上图蓝色框）
对没有object的bbox的confidence loss，赋予小的loss weight，记为 λnoobj ，在pascal VOC训练中取0.5。（上图橙色框）
有object的bbox的confidence loss (上图红色框) 和类别的loss （上图紫色框）的loss weight正常取1。
对不同大小的bbox预测中，相比于大bbox预测偏一点，小box预测偏相同的尺寸对IOU的影响更大。而sum-square error loss中对同样的偏移loss是一样。
为了缓和这个问题，作者用了一个巧妙的办法，就是将box的width和height取平方根代替原本的height和width。 如下图：small bbox的横轴值较小，发生偏移时，反应到y轴上的loss（下图绿色）比big box(下图红色)要大。

![这里写图片描述](https://img-blog.csdn.net/20170420214304420?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHJzc3R1ZHk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

在 YOLO中，每个栅格预测多个bounding box，但在网络模型的训练中，希望每一个物体最后由一个bounding box predictor来负责预测。
因此，当前哪一个predictor预测的bounding box与ground truth box的IOU最大，这个 predictor就负责 predict object。
这会使得每个predictor可以专门的负责特定的物体检测。随着训练的进行，每一个 predictor对特定的物体尺寸、长宽比的物体的类别的预测会越来越好。

# 测试
测试的时候，每个网格预测的class信息$( Pr(Class_i | Object) )$和bounding box预测的confidence信息$( Pr(Object)*IOU^{truth}_{pred} )$ 相乘，就得到每个bounding box的class-specific confidence score。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122184433874.png)

- 等式左边第一项就是每个网格预测的类别信息，第二三项就是每个bounding box预测的confidence。这个乘积即encode了预测的box属于某一类的概率，也有该box准确度的信息。

![这里写图片描述](https://img-blog.csdn.net/20180613161242186?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

- 对每一个网格的每一个bbox执行同样操作： 7x7x2 = 98 bbox （每个bbox既有对应的class信息又有坐标信息）

![这里写图片描述](https://img-blog.csdn.net/20180613161358348?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

- 得到每个bbox的class-specific confidence score以后，设置阈值，滤掉得分低的boxes，对保留的boxes进行NMS处理，就得到最终的检测结果。 

![这里写图片描述](https://img-blog.csdn.net/2018061316143925?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

NMS的过程如下：

![这里写图片描述](https://img-blog.csdn.net/20170420214347813?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvaHJzc3R1ZHk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

# 算法优缺点
### 优点

- 就像在训练中一样，图像的检测只需要一个网络评估。 在PASCAL VOC上，网络预测每个图像的98个边界框和每个框的类概率。 YOLO在测试时间速度非常快，因为它只需要一个网络预测，而不像基于分类器的方法，所以速度很快。
- 速度快，YOLO将物体检测作为回归问题进行求解，整个检测网络pipeline简单。在titan x GPU上，在保证检测准确率的前提下（63.4% mAP，VOC 2007 test set），可以达到45fps的检测速度。
- 背景误检率低。YOLO在训练和推理过程中能‘看到’整张图像的整体信息，而基于region proposal的物体检测方法（如rcnn/fast rcnn），在检测过程中，只‘看到’候选框内的局部图像信息。因此，若当图像背景（非物体）中的部分数据被包含在候选框中送入检测网络进行检测时，容易被误检测成物体。测试证明，YOLO对于背景图像的误检率低于fast rcnn误检率的一半。
- 通用性强。YOLO对于艺术类作品中的物体检测同样适用。它对非自然图像物体的检测率远远高于DPM和RCNN系列检测方法。
### 缺点
- 每个 grid cell 只预测一个 类别的 Bounding Boxes，而且最后只取置信度最大的那个 Box。这就导致如果多个不同物体(或者同类物体的不同实体)的中心落在同一个网格中，会造成漏检。

- 预测的 Box 对于尺度的变化比较敏感，在尺度上的泛化能力比较差。

- 识别物体位置精准性差。

- 召回率低。

### 和其它算法对比
Table1给出了YOLO与其他物体检测方法，在检测速度和准确性方面的比较结果（使用VOC 2007数据集）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019112218372369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

论文中，作者还给出了YOLO与Fast RCNN在各方面的识别误差比例，如Table4所示。YOLO对背景内容的误判率（4.75%）比Fast RCNN的误判率（13.6%）低很多。但是YOLO的定位准确率较差，占总误差比例的19.0%，而Fast RCNN仅为8.6%。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191122183753120.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 总结

- YOLOv1最大的开创性贡献在于将物体检测作为一个回归问题进行求解，输入图像经过一次inference，便能得到图像中所有物体的位置和其所属类别及相应的置信概率。而rcnn/fast rcnn/faster rcnn将检测结果分为两部分求解：物体类别（分类问题），物体位置即bounding box（回归问题），所以YOLO的目标检测速度很快。

- YOLO仍然是一个速度换精度的算法，目标检测的精度不如RCNN

# 参考
https://zhuanlan.zhihu.com/p/25236464
https://www.bilibili.com/video/av23354360?from=search&seid=14097781066157427376

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPadaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS01M2E3NWVmOTQ2YjA0OTE3LnBuZw?x-oss-process=image/format,png)