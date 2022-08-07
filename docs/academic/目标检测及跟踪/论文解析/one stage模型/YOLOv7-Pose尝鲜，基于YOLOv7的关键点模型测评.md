### 【前言】
目前人体姿态估计总体分为Top-down和Bottom-up两种，与目标检测不同，无论是基于热力图或是基于检测器处理的关键点检测算法，都较为依赖计算资源，推理耗时略长，今年出现了以YOLO为基线的关键点检测器。玩过目标检测的童鞋都知道YOLO以及各种变种目前算是工业落地较多的一类检测器，其简单的设计思想，长期活跃的社区生态，使其始终占据着较高的话题度。

### 【演变】
在ECCV 2022和CVPRW 2022会议上，YoLo-Pose和KaPao（下称为yolo-like-pose）都基于流行的YOLO目标检测框架提出一种新颖的无热力图的方法，类似于很久以前谷歌使用回归计算关键点的思想，yolo-like-pose一不使用检测器进行二阶处理，二部使用热力图拼接，虽然是一种暴力回归关键点的检测算法，但在处理速度上具有一定优势。
![](https://img-blog.csdnimg.cn/img_convert/3f70f3ab88c3ce802c96f341e34f05d1.png)
#### kapao
去年11月，滑铁卢大学率先提出了 KaPao：Rethinking Keypoint Representations: Modeling Keypoints and Poses as Objects for Multi-Person Human Pose Estimation，基于YOLOv5进行关键点检测，该文章目前已被ECCV 2022接收，该算法所取得的性能如下：
![](https://img-blog.csdnimg.cn/img_convert/c358ec0aaae901eb580d2bd4816ff303.png)
paper：https://arxiv.org/abs/2111.08557
code：https://github.com/wmcnally/kapao

#### yolov5-pose
今年4月，yolo-pose也挂在了arvix，在论文中，通过调研发现 HeatMap 的方式普遍使用L1 Loss。然而，L1损失并不一定适合获得最佳的OKS。且由于HeatMap是概率图，因此在基于纯HeatMap的方法中不可能使用OKS作为loss，只有当回归到关键点位置时，OKS才能被用作损失函数。
因此，yolo-pose使用oks loss作为关键点的损失
![](https://img-blog.csdnimg.cn/img_convert/36b2393d3196e9cd9ad557996e62061f.png)
相关代码在https://github.com/TexasInstruments/edgeai-yolov5/blob/yolo-pose/utils/loss.py也可见到：
```python
				if self.kpt_label:
                    #Direct kpt prediction
                    pkpt_x = ps[:, 6::3] * 2. - 0.5
                    pkpt_y = ps[:, 7::3] * 2. - 0.5
                    pkpt_score = ps[:, 8::3]
                    #mask
                    kpt_mask = (tkpt[i][:, 0::2] != 0)
                    lkptv += self.BCEcls(pkpt_score, kpt_mask.float()) 
                    #l2 distance based loss
                    #lkpt += (((pkpt-tkpt[i])*kpt_mask)**2).mean()  #Try to make this loss based on distance instead of ordinary difference
                    #oks based loss
                    d = (pkpt_x-tkpt[i][:,0::2])**2 + (pkpt_y-tkpt[i][:,1::2])**2
                    s = torch.prod(tbox[i][:,-2:], dim=1, keepdim=True)
                    kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/torch.sum(kpt_mask != 0)
                    lkpt += kpt_loss_factor*((1 - torch.exp(-d/(s*(4*sigmas**2)+1e-9)))*kpt_mask).mean()
```
相关性能如下：
![](https://img-blog.csdnimg.cn/img_convert/b63221a3e04e86ee1b93e1133523946a.png)
#### yolov7-pose
上个星期，YOLOv7的作者也放出了关于人体关键点检测的模型，该模型基于YOLOv7-w6，
![](https://img-blog.csdnimg.cn/img_convert/9647d78bd3aefcc55f56f76d926520c5.png)
目前作者提供了.pt文件和推理测试的脚本，有兴趣的童靴可以去看看，本文的重点更偏向于对yolov7-pose.pt进行onnx文件的抽取和推理。

### 【yolov7-pose + onnxruntime】
首先下载好官方的预训练模型，使用提供的脚本进行推理：

```python
% weigths = torch.load('weights/yolov7-w6-pose.pt')
% image = cv2.imread('sample/pose.jpeg')
!python pose.py 
```
![](https://img-blog.csdnimg.cn/img_convert/3d8927e32af2dfdfbcc78a8e9595ac42.png)

**一、yolov7-w6 VS yolov7-w6-pose**：

1. 首先看下yolov7-w6使用的检测头
![](https://img-blog.csdnimg.cn/img_convert/8014261b5a208f88792a05c0dedc4c38.png)
 - $f$ 表示一共有四组不同尺度的检测头，分别为15×15,30×30,60×60,120×120，对应输出的节点为114,115,116,117
 -  nc对应coco的80个类别
 -  no表示$class_.num+obj+reg = 80+1+4=85$

2. 再看看yolov7-w6-pose使用的检测头：
![](https://img-blog.csdnimg.cn/img_convert/b570ca055f44007c34aeeb13f81276b9.png)
上述重复的地方不累述，讲几个点：

 - $nc=1$ 代表person一个类别
 - nkpt表示人体的17个关键点
 - $no=17*3=nkpt*(x+y+obj)=57$

**二、修改export脚本**

如果直接使用export脚本进行onnx的抽取一定报错，在上一节我们已经看到pose.pt模型使用的检测头为IKeypoint，那么脚本需要进行相应更改：
在export.py的这个位置插入：

```python
    # 原代码:
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
     model.model[-1].export = not opt.grid  # set Detect() layer grid export
                
    # 修改代码:
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, models.yolo.IKeypoint):
            m.forward = m.forward_keypoint  # assign forward (optional)
            # 此处切换检测头
    model.model[-1].export = not opt.grid  # set Detect() layer grid export
```
forward_keypoint在原始的yolov7 repo源码中有，作者已经封装好，但估计是还没打算开放使用。

使用以下命令进行抽取：

```python
python export.py --weights 'weights/yolov7-w6-pose.pt' --img-size 960 --simplify True
```
抽取后的onnx检测头：

![](https://img-blog.csdnimg.cn/img_convert/1d50396302574a161e3b4f5995cd3325.png)
**三、onnxruntime推理**

onnxruntime推理代码：

```python
import onnxruntime
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint, plot_skeleton_kpts

device = torch.device("cpu")

image = cv2.imread('sample/pose.jpeg')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

print(image.shape)
sess = onnxruntime.InferenceSession('weights/yolov7-w6-pose.onnx')
out = sess.run(['output'], {'images': image.numpy()})[0]
out = torch.from_numpy(out)

output = non_max_suppression_kpt(out, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

# matplotlib inline
plt.figure(figsize=(8, 8))
plt.axis('off')
plt.imshow(nimg)
plt.show()
plt.savefig("tmp")

```

![在这里插入图片描述](https://img-blog.csdnimg.cn/05e02d4d0776483cacd4eb0f3af9d4d9.png)

推理效果几乎无损，但耗时会缩短一倍左右，另外有几个点：

 - image = letterbox(image, 960, stride=64, auto=True)[0] 中stride指的是最大步长，yolov7-w6和yolov5s下采样多了一步，导致在8，16，32的基础上多了64的下采样步长
 - output = non_max_suppression_kpt(out, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True) ，nc 和 kpt_label 等信息在netron打印模型文件时可以看到
 - 所得到的onnx相比原半精度模型大了将近三倍，后续排查原因
 - yolov7-w6-pose极度吃显存，推理一张960×960的图像，需要2-4G的显存，训练更难以想象

