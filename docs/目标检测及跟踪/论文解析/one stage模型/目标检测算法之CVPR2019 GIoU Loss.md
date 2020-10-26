![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214132653222.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 前言
上周介绍了Gaussian YOLOv3以及GHM Loss，这周我们来看看斯坦福大学和澳大利亚阿德莱德大学在CVPR2019发表的《Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression》，论文的核心就是提出了一个新的Loss，叫做GIOU Loss，论文原文地址见附录。

# 背景
前面介绍了很多Anchor-Based的目标检测算法，它们的Loss普遍使用bbox和ground truth bbox的L1范数，L2范数来计算位置回归Loss。但在评测的时候却使用IOU(交并比)去判断是否检测到目标。显然这两者并不是完全等价的，论文中举了Figure 1作为例子：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214132621872.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)图中第一行，所有目标的L1 Loss都一样，但是第三个的IOU显然是要大于第一个，并且第3个的检测结果似乎也是好于第一个的。第二行类似，所有目标的L1 Loss也都一样，但IOU却存在差异。因此使用bbox和ground truth bbox的L1范数，L2范数来计算位置回归Loss以及在评测的时候却使用IOU(交并比)去判断是否检测到目标是有一个界限的，这两者并不等价。因此，一种直观的想法就是直接将IOU作为Loss来优化任务，但这存在两个问题：
- 预测框bbox和ground truth bbox如果没有重叠，IOU就始终为0并且无法优化。也就是说损失函数失去了可导的性质。
- IOU无法分辨不同方式的对齐，例如方向不一致等，如下图所示，可以看到三种方式拥有相同的IOU值，但空间却完全不同。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019121413345421.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# GIoU
因此为了解决上面两个问题，这篇论文提出了GIOU。GIOU指的是假设现在有两个任意的bbox A和B，我们要找到一个最小的封闭形状C，让C可以将A和B包围在里面，然后我们计算C中没有覆盖A和B的面积占C总面积的比例，然后用A和B的IOU值减去这个比值，具体过程用下面的算法1来表示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214133926193.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)GIoU相比于IoU有如下性质：
- 和IoU类似，GIoU也可以作为一个距离，然后损失可以用下面的公式来计算： $L_{GIoU}=1-GIoU$
- 和原始的IoU类似，GIoU对物体的尺度大小不敏感(因为比值的原因)，并且$GIoU<=IoU$，而$0<=IoU<=1$，所以$-1<=GIoU<=1$，当预测bbox A和ground truth bbox B完全重合时$IoU=GIoU=1$。
- 由于GIoU引入了包含A，B两个框的C，所以当A，B不重合时也同样可以计算。

# GIoU Loss
针对二维图像的目标检测，具体如何计算GIoU Loss呢？假设现在预测的bbox和ground truth bbox的坐标分别表示为：
![$B^$](https://img-blog.csdnimg.cn/20191214134726478.png)
其中，$x_2^p>x_1^p，y_2^p>y_1^p$。然后计算Loss的具体过程可以表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214135008303.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214135017761.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214135811584.png)
# 实验
论文给出了一些实验结果，（针对分割任务和分类任务有一定 loss 的调整设计，不过论文中没有详细给出）结果是 IoU loss 可以轻微提升使用 MSE 作为 loss 的表现，而 GIoU 的提升幅度更大，这个结论在 YOLO 算法和 faster R-CNN 系列上都是成立的。具体的实验结果如下所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214140024308.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)使用YOLOv3在PASCAL VOC 2007上的测试结果。AP值大概涨了近2个百分点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214140204360.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)使用YOLOv3在MS COCO上测试，关注分类损失和准确率（平均IoU值），可以看到GIoU loss提升了准确率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214140933467.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
使用Faster-RCNN在PASCAL VOC 2007上的测试结果，坐标回归损失使用的L1损失。AP值大概涨了1个百分点。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191214141011299.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)使用Mask-RCNN在MS COCO 2018上的测试结果，坐标回归损失使用的L1损失。AP值大概涨了小于1个百分点。

# 附录
论文原文：https://arxiv.org/abs/1902.09630

开源代码：https://github.com/generalized-iou/g-darknet

AlexAB版本Darknet支持了GIOU Loss：https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov3.coco-giou-12.cfg

参考文章：https://zhuanlan.zhihu.com/p/63389116
# 结论
这篇论文用IOU作文章提出了GIoU Loss，最终YOLO V3涨了2个点，Faster RCNN，MaskRCNN这种涨点少了些。我猜测原因在于Faster RCNN，MaskRCNN本身的Anchor很多，出现完全无重合的情况比较少，这样GIOU和IOU Loss就无明显差别，所以提升不是太明显。

# 后记
今天就讲到这里了，有什么问题欢迎留言交流哦。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)