# 前言
最近准备开始认真的梳理一下目标检测的相关算法，组合成一个目标检测算法系列。之前看到了一张特别好的目标检测算法分类的甘特图，但忘记是哪里的了，要是原始出处请提醒我标注。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191110120920893.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
我也会按照这个图来讲解目标检测算法。

# 背景介绍

## 什么是目标检测
所谓目标检测就是在一张图像中找到我们关注的目标，并确定它的类别和位置，这是计算机视觉领域最核心的问题之一。由于各类目标不同的外观，颜色，大小以及在成像时光照，遮挡等具有挑战性的问题，目标检测一直处于不断的优化和研究中。
## 目标检测算法分类
上面那张甘特图已经说明了目标检测算法主要分为两类，即：
- Two Stage目标检测算法。这类算法都是先进行区域候选框生成，就是找到一个可能包含物体的预选框，再通过卷积神经网络进行分类和回归修正，常见算法有R-CNN，SPP-Net，Fast-RCNN，Faster-RCNN和R-FCN等。
- One Stage目标检测算法。这类算法不使用候选框生成，直接在网络中提取特征来预测物体的分类和位置。常见的One-Stage算法有：YOLO系列，SSD，RetinaNet。

# RCNN算法
## 贡献
RCNN是第一个使用卷积神经网络来对目标候选框提取特征的目标检测算法。同时，RCNN使用了微调(finetune)的技术，使用大数据集上训练好的分类模型的前几层做backbone，进行更有效的特征提取。

## RCNN总览
看下图：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191110181615124.png) 首先，R-CNN是将传统图像算法和深度学习技术结合起来的结构，第一部分是需要候选框区域建议，这里一般使用Selective Search的方法提取出候选框，然后再传入CNN做特征提取及分类，后面还借助了机器学习算法做回归修正。

## RCNN算法步骤
- 选择一个预训练 （pre-trained）神经网络（如AlexNet、VGG）。
- 重新训练全连接层。使用需要检测的目标重新训练（re-train）最后全连接层（connected layer）。即是fintune技术的应用。
- 生成候选框。利用Selective Search算法提取所有的Proposals，一张图片大概产生2000张，然后将图片规整化固定大小，使得其满足CNN的输入要求，最后将feature map存到磁盘(是的，你没有看错，RCNN要把提取到的特征存储到磁盘)，这个过程可以用下图表示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019111018214310.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- 利用feature map训练SVM来对目标和背景进行分类，这里每一个类一个二元SVM。
- 训练线性回归器修正目标的位置，如下图所示：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191110182347369.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 实验结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191110182427652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)RCNN的出世成为当时目标检测领域的SOAT算法，虽然现在很少有人使用到了，但论文的思想我们仍可以借鉴。任何事情都要经历一个从无到有的过程。

# 源码
rgbirshick大神，也就是RCNN作者，提供了源码，链接如下：
https://github.com/rbgirshick/rcnn