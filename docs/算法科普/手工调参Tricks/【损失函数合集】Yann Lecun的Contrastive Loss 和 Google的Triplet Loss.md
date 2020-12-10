# 前言
昨天在介绍Center Loss的时候提到了这两个损失函数，今天就来介绍一下。Contrastive Loss是来自Yann LeCun的论文`Dimensionality Reduction by Learning an Invariant Mapping`，目的是增大分类器的类间差异。而Triplet Loss是在FaceNet论文中的提出来的，原文名字为：`FaceNet: A Unified Embedding for Face Recognition and Clustering`，是对Contrastive Loss的改进。接下来就一起来看看这两个损失函数。论文原文均见附录。

# 问题引入
假设我们现在有2张人脸图片，我们要进行一个简单的对比任务，就是判断这两张人脸图片是不是对应同一个人，那么我们一般会如何解决？一种简单直接的思路就是提取图片的特征向量，然后去对比两个向量的相似度。但这种简单的做法存在一个明显的问题，那就是CNN提取的特征“类间”区分性真的有那么好吗？昨天我们了解到用SoftMax损失函数训练出的分类模型在Mnist测试集上就表现出“类间”区分边界不大的问题了，使得遭受对抗样本攻击的时候很容易就分类失败。况且人脸识别需要考虑到样本的类别以及数量都是非常多的，这无疑使得直接用特征向量来对比更加困难。

# Contrastive Loss
针对上面这个问题，孪生网络被提出，大致结构如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117140619184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后孪生网络一般就使用这里要介绍的Contrastive Loss作为损失函数，这种损失函数可以有效的处理这种网络中的成对数据的关系。

Contrastive Loss的公式如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117132205211.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中$W$是网络权重，$Y$是成对标签，如果$X_1$，$X_2$这对样本属于同一个类，则$Y=0$，属于不同类则$Y=1$。$D_W$是$X_1$与$X_2$在潜变量空间的欧几里德距离。当$Y=0$，调整参数最小化$X_1$与$X_2$之间的距离。当$Y=1$，当$X_1$与$X_2$之间距离大于$m$，则不做优化（省时省力）当$X1$与 X2 之间的距离小于$m$, 则增大两者距离到m。下面的公式（4）是将上面的$L$展开写了一下，如下所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011713220570.png)

而下面的Figure1展示的就是损失函数$L$和样本特征的欧氏距离之间的关系，其中红色虚线表示相似样本的损失值，而蓝色实线表示的是不相似样本的损失值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117134404969.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

在LeCun的论文中他用弹簧在收缩到一定程度的时候因为受到斥力的原因会恢复到原始长度来形象解释了这个损失函数，如下图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117135948640.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

代码实现：

```
# Custom Contrastive Loss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive
```


# Triplet Loss
## 原理
Triplet Loss是Google在2015年发表的FaceNet论文中提出的，论文原文见附录。Triplet Loss即三元组损失，我们详细来介绍一下。

- Triplet Loss定义：最小化锚点和具有相同身份的正样本之间的距离，最小化锚点和具有不同身份的负样本之间的距离。
- Triplet Loss的目标：Triplet Loss的目标是使得相同标签的特征在空间位置上尽量靠近，同时不同标签的特征在空间位置上尽量远离，同时为了不让样本的特征聚合到一个非常小的空间中要求对于同一类的两个正例和一个负例，负例应该比正例的距离至少远`margin`。如下图所示：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117203441618.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

因为我们期望的是下式成立，即：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011720351165.png)

其中$\alpha$就是上面提到的`margin`，$T$就是样本容量为$N$的数据集的各种三元组。然后根据上式，Triplet Loss可以写成：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117204030299.png)

对应的针对三个样本的梯度计算公式为：

$\frac{\partial L}{\partial f(x_i^a)}=2*(f(x_i^a)-f(x_i^p))-2*(f(x_i^a)-f(x_i^n))=2*(f(x_i^n)-f(x_i^p))$
$\frac{\partial L}{\partial f(x_i^p)}=2*(f(x_i^a)-f(x_i^p))*(-1)=2*(f(x_i^p)-f(x_i^a))$
$\frac{\partial L}{\partial f(x_i^n)}=2*(f(x_i^a)-f(x_i^n))*(-1)=2*(f(x_i^a)-f(x_i^p))$

## FaceNet

我们将三元组重新描述为$(a,p,n)$，那么最小化上面的损失就是就是让锚点$a$和正样本的距离$d(a,p)->0$，并使得锚点$a$和负样本的距离大于$d(a,p)+margin$，即$d(a,n)>d(a,p)+margin$。那么三元组的总体距离可以表示为：
$L=max(d(a,p)-d(a,n)+margin,0)$。

FaceNet网络可以更加形象的表示为下图：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117205140780.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

所以网络的最终的优化目标是要让`a,p`的距离近，而`a，n`的距离远。下面定义一下三种不同优化难度的三元组样本。

- `easy triplets`：代表$L=0$，即$d(a,p)+margin<d(a,n)$，无需优化，$a，p$初始距离就很近，$a，n$初始的距离很远。
- `hard triplets`：代表$d(a,n)<d(a,p)$，即$a，p$的初始距离很远。
- `semi-hard triplets`：代表$d(a,p)<d(a,n)<d(a,p)+margin$，即$a,n$的距离靠得比较近，但是有一个`mergin`。
这三种不同的`triplets`可以用下图来表示：


![在这里插入图片描述](https://img-blog.csdnimg.cn/2020011721215298.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后FaceNet的训练策略是随机选取`semi-hard triplets`来进行训练的，当然也可以选择`hard triplets`或者两者结合来训练。关于FaceNet更多的训练细节我们就不再介绍了，这一节的目的是介绍Triplet Loss，之后在人脸识别专栏再单独写一篇介绍FaceNet训练测试以及网络参数细节的。


## 代码实现
简单提供一个Triplet Loss训练Mnist数据集的Keras代码，工程完整地址见附录：

```cpp
def triplet_loss(y_true, y_pred):
        """
        Triplet Loss的损失函数
        """

        anc, pos, neg = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:]

        # 欧式距离
        pos_dist = K.sum(K.square(anc - pos), axis=-1, keepdims=True)
        neg_dist = K.sum(K.square(anc - neg), axis=-1, keepdims=True)
        basic_loss = pos_dist - neg_dist + TripletModel.MARGIN

        loss = K.maximum(basic_loss, 0.0)

        print "[INFO] model - triplet_loss shape: %s" % str(loss.shape)
        return loss

```

可以来感受一下Triplet Loss训练Mnist时Loss下降效果，几乎是线性下降：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117213315416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

另外看一下作者只跑了2个Epoch之后的降维可视化结果，截图如下，可以看到已经类别已经聚集得特别好了：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20200117213427137.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 附录

- Contrasstive Loss原文：http://www.cs.toronto.edu/~hinton/csc2535/readings/hadsell-chopra-lecun-06-1.pdf
- Triplet Loss原文：https://arxiv.org/abs/1503.03832
- 参考：https://zhuanlan.zhihu.com/p/76515370
- Triplet Loss代码实现：https://github.com/SpikeKing/triplet-loss-mnist


# 推荐阅读

- [【损失函数合集】ECCV2016 Center Loss](https://mp.weixin.qq.com/s/aYrpdwd4J501hKyHozJZBw)
- [目标检测算法之RetinaNet（引入Focal Loss）](https://mp.weixin.qq.com/s/2VZ_RC0iDvL-UcToEi93og)
- [目标检测算法之AAAI2019 Oral论文GHM Loss](https://mp.weixin.qq.com/s/mHOo148aUIuK7fewTD1IyQ)
- [目标检测算法之CVPR2019 GIoU Loss](https://mp.weixin.qq.com/s/CNVgrIkv8hVyLRhMuQ40EA)
- [目标检测算法之AAAI 2020 DIoU Loss 已开源(YOLOV3涨近3个点)](https://mp.weixin.qq.com/s/u41W31IEg5xuX9jtRyVGmQ)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)