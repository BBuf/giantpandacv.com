#  SMOKE window 版
【导语】修改开源的 3D 检测算法，使用了 pytorch 自带的 DConv，省去 linux 下编译 DConv 的 cuda 代码，可以直接在 window 下训练和测试。在源码基础上增加了 finetune 和 resume 等功能，并提供了重新训练的模型。

代码： https://github.com/Huangdebo/SMOKE-window

- [x] 模型效果

![](https://img-blog.csdnimg.cn/30443f3668a94ac7baf57f4341262bed.gif)

## 1 网络结构

![网路结构](https://img-blog.csdnimg.cn/cc1bb792dedc4cf9bd6b5a6435bcda3b.png#pic_center)

smoke 的网络机构比较简单粗暴，一个 DLA 直接出 w/4, h/4 的特征图，然后分成两路：

（1） 关键点检测，用以得出每个类别的目标的 3D 中心点在 2D 图像中的投影

（2） 3D 信息的回归， depth_offset(1), keypoint_offset(2), dimension_offset(3), orientation(2)，共 8 个channel。

## 2  3D 检测
### 2.1 关键点分支
类似 centernet，把目标看成点来检测，但不同的是，smoke 利用的是 3D box 的中心投影点，而不是 2D box 的中心点。

![3D和2D中心点投影](https://img-blog.csdnimg.cn/dbe1ed998a7c44f6adfe3dd1b8bc56ca.png#pic_center)
### 2.2 回归分支
首先是 location，网络回归的都是偏差，对于 Z 方向的值，通过预设的偏移量和尺度因子来计算：

![在这里插入图片描述](https://img-blog.csdnimg.cn/04b8458d328a46189382d48e3e46bf7c.png#pic_center)

对于 x,y，则是通过关键点分支得到的投影中心点和回归的偏移量计算得出，然后在通过反投影计算出 location：

![公式3](https://img-blog.csdnimg.cn/5ec27969712c43318c188aa6ba29ae6a.png#pic_center)

dimension 也是通过统计预设值和回归值来计算：

![公式4](https://img-blog.csdnimg.cn/b5b6b0b63d5841f8845fac0aef2200f7.png#pic_center)

对于航向角的计算比较绕，smoke 并不是直接回归航向角，而是通过间接计算：

![航向角表示](https://img-blog.csdnimg.cn/88259b6732f74d8898d42e5bff469da7.png#pic_center)

在下图中将小车沿着y轴顺时针旋转，待小车和camera连线与相机坐标系的z轴重合时停止，那么紫色的角是没有发生变化的。于是有

![公式5.1](https://img-blog.csdnimg.cn/ecf52a53d1564f88ac7155885ab77439.png#pic_center)

r_y 和 alpha 就是 KITTI label 中的 rotation_y 和 alpha，β 就是论文中的 αz。但 smoke 也不是直接回归  αz，而且转化成另外一个变量  αx，αx 也是编码成 [sin(α), cos(α)]，进行归一化处理

![航向角回归](https://img-blog.csdnimg.cn/82e924e1206c449fac02218b597f66d1.png#pic_center)

基于上面的计算值，就可以 3D box 的8个顶点坐标了

![公式6](https://img-blog.csdnimg.cn/689091adfb8c49f39c17ec252118b86c.png#pic_center)

## 3 loss 计算
### 3.1 关键点分类 loss
与 centernet 类似，在 focal loss 的基础上加权，对中心点附近的位置降低loss权重，并通过巧妙的设计来抑制中心附近的回归值，让只有中心一个尖峰：

![公式7](https://img-blog.csdnimg.cn/b02144ac55854e75b05ebc181ae2cbe9.png#pic_center)

![公式7.1](https://img-blog.csdnimg.cn/8d32276c91d24034b2a8dc5d326ad770.png#pic_center)

### 3.2 回归 loss
smoke 使用了 disentangling loss 形式来计算回归分支的 loss，计算 3D box 需要 rotation_y，dimension 和 location 三个变量，disentangling 则是用一个预测值 + 两个 groundtruth 值来计算，从而得到三个 3D box，在进行 L1 loss 计算。

![公式8](https://img-blog.csdnimg.cn/51fde4b8031347efac8ab16c3295b7ee.png#pic_center)

所以总的 loss 就是 分类 loss 和 3 个不同的 L1 回归 loss

![公式9](https://img-blog.csdnimg.cn/6c474493b2e84fb1998f9e50282131cc.png#pic_center)

## 结论：
SMOKE 的backbone 使用了 DLA，而且依赖 Dcnv，所以模型比较大，对落地部署不是很友好。而且是依赖镜头的内参的，不清楚多种数据训和训练效果如何。3D框的回归比较稳定，但存在漏检，尤其是小目标和只出现一部分的目标。如果能参考主流 2D 检测的多层检测，和优化关键点分支的策略，也许能缓解这些问题。


