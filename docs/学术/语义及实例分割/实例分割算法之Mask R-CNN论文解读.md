# 前言
Anchor-Based的目标检测算法我们已经讲了比较多了，另外Anchor-Free的目标检测我们也已经简单解读了一下DenseBox开了个头，而今天我们要来说说另外一个方向即实例分割。而实例分割首当其冲需要介绍的就是2017年He Kaiming大神的力作Mask-RCNN，其在进行目标检测的同时进行实例分割，取得了出色的效果，并获得了2016年COCO实例分割比赛的冠军。

# 总览
Mask-RCNN是一个实例分割（Instance segmentation）框架，通过增加不同的分支可以完成目标分类，目标检测，语义分割，实例分割，人体姿态估计等多种任务。对于实例分割来讲，就是在Faster-RCNN的基础上(分类+回归分支)增加了一个分支用于语义分割，其抽象结构如Figure1所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111182030337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

稍微描述一下这个结构：

- 输入预处理后的原始图片。
- 将输入图片送入到特征提取网络得到特征图。
- 然后对特征图的每一个像素位置设定固定个数的ROI（也可以叫Anchor），然后将ROI区域送入RPN网络进行二分类(前景和背景)以及坐标回归，以获得精炼后的ROI区域。
- 对上个步骤中获得的ROI区域执行论文提出的ROIAlign操作，即先将原图和feature map的pixel对应起来，然后将feature map和固定的feature对应起来。
- 最后对这些ROI区域进行多类别分类，候选框回归和引入FCN生成Mask，完成分割任务。

下图更清晰的展示了Mask-RCNN的整体框架，来自知乎用户`vision`：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111184333800.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 方法
## 原始ROI Pooling的问题
在Faster-RCNN中ROIPooling的过程如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111203147569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

输入图片的大小为$800\times 800$，其中狗这个目标框的大小为$665\times 665$，经过VGG16网络之后获得的特征图尺寸为$800/32 \times 800/32=25\times 25$，其中$32$代表VGG16中的$5$次下采样（步长为2）操作。同样，对于狗这个目标，我们将其对应到特征图上得到的结果是$665/32 \times 665/32=20.78\times 20.78=20\times 20$，因为坐标要保留整数所以这里引入了第一个量化误差即舍弃了目标框在特征图上对应长宽的浮点数部分。

然后我们需要将这个$20\times 20$的ROI区域映射为$7\times 7$的ROI特征图，根据ROI Pooling的计算方式，其结果就是$20/7 \times 20/7=2.86\times 2.86$，同样执行取整操作操作后ROI特征区域的尺寸为$2\times 2$，这里引入了第二次量化误差。

从上面的分析可以看出，这两次量化误差会导致原始图像中的像素和特征图中的像素进行对应时出现偏差，例如上面将$2.86$量化为$2$的时候就引入了$0.86$的偏差，这个偏差映射回原图就是$0.86\times 32=27.52$，可以看到这个像素偏差是很大的。

## ROIAlign
为了缓解ROI Pooling量化误差过大的缺点，本论文提出了ROIAlign，ROIAligin没有使用量化操作，而是使用了双线性插值。它充分的利用原图中的虚拟像素值如$27.52$四周的四个真实存在的像素值来共同决定目标图中的一个像素值，即可以将和$27.52$类似的非整数坐标值像素对应的输出像素值估计出来。这一过程如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111205721560.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中feat. map就是VGG16或者其他Backbone网络获得的特征图，黑色实线表示的是ROI划分方式，最后输出的特征图大小为$2\times 2$，然后就使用双线性插值的方式来估计这些蓝色点的像素值，最后得到输出，然后再在橘红色的区域中执行Pooling操作最后得到$2\times 2$的输出特征图。可以看到，这个过程相比于ROI Pooling没有引入任何量化操作，即原图中的像素和特征图中的像素是完全对齐的，没有偏差，这不仅会提高检测的精度，同时也会有利于实例分割。

## 网络结构
为了证明次网络的通用性，论文构造了多种不同结构的Mask R-CNN，具体为使用Backbone网络以及**是否**将用于边框识别和Mask预测的**上层**网络分别应用于每个ROI。对于Backbone网络，Mask R-CNN基本使用了之前提出的架构，同时添加了一个全卷积的Mask(掩膜)预测分支。Figure3展示了两种典型的Mask R-CNN网络结构，左边的是采用$ResNet$或者$ResNeXt$做网络的backbone提取特征，右边的网络采用FPN网络做Backbone提取特征，这两个网络的介绍均在公众号的往期文章中可以找到，最终作者发现**使用ResNet-FPN作为特征提取的backbone具有更高的精度和更快的运行速度**，所以实际工作时大多采用右图的完全并行的mask/分类回归。


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111210616101.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 损失函数
Mask分支针对每个ROI区域产生一个$K\times m\times m$的输出特征图，即$K$个$m\times m$的二值掩膜图像，其中$K$代表目标种类数。Mask-RCNN在Faster-RCNN的基础上多了一个ROIAligin和Mask预测分支，因此Mask R-CNN的损失也是多任务损失，可以表示为如下公式：
$L=L_{cls}+L_{box}+L_{mask}$
其中$L_{cls}$表示预测框的分类损失，$L_{box}$表示预测框的回归损失，$L_{mask}$表示Mask部分的损失。
对于预测的二值掩膜输出，论文对每一个像素点应用`sigmoid`函数，整体损失定义为平均二值交叉损失熵。引入预测K个输出的机制，允许每个类都生成独立的掩膜，避免类间竞争。这样做解耦了掩膜和种类预测。不像FCN的做法，在每个像素点上应用`softmax`函数，整体采用的多任务交叉熵，这样会导致类间竞争，最终导致分割效果差。

下图更清晰的展示了Mask-RCNN的Mask预测部分的损失计算，来自知乎用户`vision`：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111213745379.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 训练
在Faster-RCNN中，如果ROI区域和GT框的`IOU>0.5`，则ROI是正样本，否则为负样本。$L_{mask}$只在正样本上定义，而Mask的标签是ROI和它对应的Ground Truth Mask的交集。其他的一些训练细节如下：

- 采用image-centric方式训练，将图片的长宽较小的一边缩放到800像素。
- 每个GPU的`mini-batch=2`，每张图片有$N$个采样ROIs，其中正负样本比例为`1:3`。
- 在8个gpu上进行训练，`batch_size=2`，迭代`160k`次，初始学习率`0.02`，在第`120k`次迭代时衰减10倍，`weight_decay=0.0001`,`momentum=0.9`。
# 测试

测试阶段，采用的`proposals`的数量分别为$300$（Faster-RCNN）和1000(FPN)。在这些`proposals`上，使用`bbox`预测分支配合后处理`nms`来预测`box`。然后使用Mask预测分支对最高`score`的100个检测框进行处理。可以看到这里和训练时Mask预测并行处理的方式不同，这里主要是为了加速推断效率。然后，Mask网络分支对每个ROI预测$K$个掩膜图像，但这里只需要使用其中类别概率最大的那个掩膜图像就可以了，并将这个掩膜图像`resize`回ROI大小，并以`0.5`的阈值进行二值化。

# 实验
## 总览
非常的SOTA，Mask R-CNN打败了上界冠军FCIS（其使用了multi-scale训练，水平翻转测试，OHEM等），具体结果如Table1所示：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111220352932.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

再来一些可视化结果看看，如Figure5所示。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111220503232.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
## 消融实验
Table2展示了Mask-RCNN的消融实验，`(a)`显示网络越深，效果越好。并且FPN效果要好一些。而`(b`)显示`sigmoid`要比`softmax`效果好一些。`(c)`和`(d)`显示ROIAligin效果有提升，特别是AP75提升最明显，说明对精度提升很有用。`(e)`显示mask banch采用FCN效果较好（因为FCN没有破坏空间关系）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/202001112208292.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
### 目标检测结果对比
从Table3可以看出，在预测的时候即使不使用Mask分支，结果精度也是很高的。可以看到ROIAligin直接使用到Faster-RCNN上相对于ROI Pooling提高了0.9个点，但比Mask-RCNN低0.9个点。**作者将其归结为多任务训练的提升，由于加入了mask分支，带来的loss改变，间接影响了主干网络的效果。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111221135989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
### 关键点检测
和Mask-RCNN相比，关键点检测就是将Mask分支变成`heatmap`回归分支，需要注意的是最后的输出是$m\times m$形式的`softmax`, 不再是`sigmoid`，论文提到这有利于单独一个点的检测，并且最后的Mask分辨率是$56\times 56$，不再是$28\times 28$。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200111221610989.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 后记
后面我会更新Mask R-CNN的代码详细解析，从代码角度详细分析Mask R-CNN的细节，论文解析暂时就讲到这里了。

# 附录
- 论文原文：https://arxiv.org/pdf/1703.06870.pdf
- 参考资料：https://blog.csdn.net/chunfengyanyulove/article/details/83545784

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)