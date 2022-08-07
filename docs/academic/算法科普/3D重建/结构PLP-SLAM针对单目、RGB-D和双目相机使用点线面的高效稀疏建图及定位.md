# 结构PLP-SLAM:针对单目、RGB-D和双目相机使用点线面的高效稀疏建图及定位

## 0. 引言

在低纹理环境、变化光线和运动模糊条件下，基于特征点法的SLAM系统会出现各种各样的问题，而结构化环境中的线段和平面有助于帮助SLAM系统提高定位精度，在论文"Structure PLP-SLAM: Efficient Sparse Mapping and Localization using Point, Line and Plane for Monocular, RGB-D and Stereo Cameras"中，作者提出了一个紧密结合语义特征的通用框架PLP-SLAM，该算法具有较好的鲁棒性和精度，目前已经开源。

## 1. 论文信息

标题：Structure PLP-SLAM: Efficient Sparse Mapping and Localization using Point, Line and Plane for Monocular, RGB-D and Stereo Cameras

作者：Fangwen Shu, Jiaxuan Wang, Alain Pagani, Didier Stricker

原文链接：https://arxiv.org/abs/2207.06058

代码链接：https://github.com/PeterFWS/Structure-PLP-SLAM

## 2. 摘要

本文展示了一个视觉SLAM系统，该系统利用点云和线云进行鲁棒的相机定位，同时具有嵌入式分段平面重建(PPR)模块，该模块提供了一个结构图。建立并行跟踪的尺度同步建图，例如采用单个相机带来了重建具有比例模糊性的几何图元的挑战，并且进一步引入了光束法平差(BA)的图形优化的困难。我们通过对重构的线和平面提出几种运行时优化来解决这些问题。然后，该系统基于单目框架的设计扩展了深度和双目传感器。结果表明，我们提出的SLAM紧密结合了语义特征，以促进前端跟踪和后端优化。我们在各种数据集上详尽地评估了我们的系统，并为社区开源了我们的代码。

## 3. 算法分析

如图1所示是作者提出的SLAM系统，该框架基于OpenVSLAM构建，利用分段平面重建进行语义建图。

![](https://img-blog.csdnimg.cn/b997b9f8a32a44319aea6a88154b9d92.png)

图1 利用点和线云进行鲁棒的相机定位

作者的主要贡献总结如下：

\(1\) 设计了一个模块化的单目SLAM系统，除了标准特征点之外，该系统还利用了线跟踪和建图、实时分段平面重建和联合图优化。同时证明了闭环可以通过校正线地图和基于预先构建的点线地图的重新定位模块来完成。

\(2\) 使用RGB-D和双目相机扩展了项目的应用。

\(3\) 考虑到添加的特征线引入了更多的不确定性并减慢了跟踪和优化，它被设计为对噪声输入是鲁棒的，例如训练的CNN可能在看不见的图像上产生噪声预测。由相机观察到的环境实际上是非常多样的，因此当点仍然是基本的几何图元时，线和平面被重建和优化，且不需要任何强假设，例如MW，因此该系统不限于小比例的场景。

\(4\) 在室内数据集TUM RGB-D、ICL-NUIM和EuRoC上进行了详尽的基准测试。与其他最先进的SLAM系统相比，给出了竞争结果以及一些更好的定量结果。

### 3.1 基于结构的单目SLAM

在单目SLAM中，定义平面的重投影误差较为困难，因此作者增加了三维点和三维平面之间的约束。

\(1\) 利用线段分割：二维线段由LSD提取，并通过LBD描述符跨帧匹配。如表1所示，通过隐藏参数调整和长度拒绝策略，对LSD的参数进行调整，可以比OpenCV的原始实现快3倍。如图2(a)是两个线段终点的简单表示，这也是PL-VINS的方法，但由于在直线方向上移动模糊，容易被遮挡或误检。如图2(b)是普朗克坐标表示，它是一个无限的三维线表示，其中m和d不需要是单位向量。这样就可以通过类似于三维点的变换矩阵，将三维线从世界坐标转换到相机坐标，然后投影到图像平面上。其误差定义为：

<img src="https://img-blog.csdnimg.cn/9eff863d6a7346278a6dd8e069e74dc0.png" style="zoom: 67%;" />

这个误差将用于BA中优化3三维线和相机姿态。然而普朗克坐标仍然过度参数化，因为它在满足克莱因二次约束的齐次坐标中呈现了5-DOF的三维线。线特征的三角化是通过从两个图像视图中前向投影匹配的2D线段l1和l2，得到两个三维平面，并与这两个平面相交，可以重建一条三维线Lw。然而，三角化的三维线是一条来自两个相交的三维平面的无限线(图2(c))，其三维端点必须被估计才能可视化。使用端点修剪方法(图2(d))可以进一步利用局部BA内的端点修剪来进行离群值拒绝，以及对重投影误差的卡方分布测试。

表1 使用LSD提取二维线段的隐藏参数，可以比OpenCV原始实现快3倍

<img src="https://img-blog.csdnimg.cn/08bd0a322e9941b2aa51b086a442a141.png" style="zoom: 67%;" />

<img src="https://img-blog.csdnimg.cn/8db855f2059f43fe9518971674065124.png" style="zoom: 67%;" />

图2 3D线条表示

\(2\) 利用三维平面结构：如图3(b)所示，通过实时拟合一组稀疏且有噪声的三维点，可以重建三维平面实例，并进行实例平面分割初始化。然而，为了应对神经网络可能出现的错误分类，重建借助RANSAC实现，并结合图切的内部局部优化。这样，可以局部优化三维空间的空间相干平面，如图3(a)所示。因此，平面拟合问题被表述为一个具有能量项的最优二值标记问题：

<img src="https://img-blog.csdnimg.cn/90d436f39f204a76bccba1ff62737ae0.png" style="zoom:80%;" />

<img src="https://img-blog.csdnimg.cn/43dade513ffd4581b35a99ecce586436.png" style="zoom: 67%;" />

图3 3D平面的解释:(a)利用空间相干性优化3D平面拟合的例子 (b)三维点的可能分布

### 3.2 仅运动BA和局部BA

在仅运动BA和局部BA中，点和线的重投影误差最小，因此总体代价函数为：

<img src="https://img-blog.csdnimg.cn/54807841e33440ee91f567200768b0fc.png" style="zoom: 60%;" />

其中第一项表示特征点的标准重投影误差，第二项表示线重投影误差，线的雅可比可以用链规则解析计算。

在回环和全局BA阶段，作者没有像PL-SLAM那样基于线特征重新构建词袋，而是使用ORB系统自带的词袋库，然后计算三维线相似变换：

<img src="https://img-blog.csdnimg.cn/e606edb3a0b1475eb0e2a41b5a2821fb.png" style="zoom: 60%;" />

之后，作者使用ORB系统的优化算法进行BA运算，通过相似变换对本质图进行优化，均摊回环误差，并得到正确的尺度漂移。然后根据观察到的参考关键帧的修正，对三维地图点和线进行变换，此过程如图1所示。为了获得最优解，作者在ORB-SLAM2的基础上进行一个全局BA。同时，优化后利用端点裁剪重新估计三维线端点，并在BA过程中作为地图剔除方法。

### 3.3 重定位

现有的基于特征点的SLAM方法利用BoW的全局描述符进行图像检索，然后使用EPnP的O(n)封闭解来初始化迭代。因此，像PL-SLAM那样简单地用EPnPL替换EPnP，并没有对精度有明显改善。在PLP-SLAM中，作者利用BA与点和线重投影误差，提供更好的相机姿态优化。注意，作者优化了线的标准正交表示，而不是像EPnPL那样的强制端点对应，这使得PLP-SLAM方法更有效，并避免了线的移动歧义。

## 4. 实验

作者在TUM RGB-D、ICL-NUIM单目和RGB-D上进行了定量实验。并在EuRoC MAV进行定性评估，其结果如图4所示。

<img src="https://img-blog.csdnimg.cn/10c9c1647ad7496ca9714f5553482a9b.png" style="zoom:80%;" />

图4 不同传感器设置下的重建图的定性结果

### 4.1 VSLAM系统性能

如表2和表3所示是PLP-SLAM的轨迹误差，所评估的数据集ICL-NUIM提供了低对比度和低纹理的合成室内图像序列，而TUM RGB-D数据集提供了不同纹理和结构条件下的真实世界室内序列。

表2 ICL-NUIM上评估的单目SLAM，每个结果都是5次执行的平均值

<img src="https://img-blog.csdnimg.cn/62268824831d47b89010a69c2293075e.png" style="zoom:80%;" />

表3 在数据集TUM RGB-D上评估的单目SLAM

<img src="https://img-blog.csdnimg.cn/3e93132b03284f749dca4534fcbc18fe.png" style="zoom:80%;" />

有趣的是，在表3中可以观察到，PLP-SLAM的隐式约束使3D点和3D平面之间的距离最小化，进而使平面场景的性能得到提高，如fr1地板，fr3结构纹理附近。此外在表4中，可以发现当深度传感器可用时，产生了最好的平均性能（点+线+平面）。该点面距离约束在理论上可以集成到图优化中，其中一条一元边连接到三维点顶点，而平面的位置从RANSAC和SVD中被认为是统计最优，因此在优化过程中是固定的。

表4 在数据集ICL- NUIM上评估的RGB-D SLAM

<img src="https://img-blog.csdnimg.cn/8d87bd639a7d4dd19f77f9c18c67c990.png" style="zoom:80%;" />

### 4.2 稀疏语义建图

在图1和图4中，作者选择性地示出了重建的点线图和线面图。该地图表示被设计为一个轻量级的稀疏地图，它直观地显示了场景结构，且没有大量的内存消耗。通过使用以相关联的3D点为中心的矩形平面片来可视化3D平面结构，可以利用非结构点云来有效地可视化结构平面，而无需附加计算。其中，3D线条是可以自然观察到的。它可以说明场景的大部分边缘特征。

在PLP-SLAM中，当实例平面分割噪声太大时，平面建图可能会失败。为此，作者利用RANSAC中的图形结构优化来拟合平面。并通过采用自适应参数设置策略来解决参数难以微调的问题。最后，PLP-SLAM的平面地图表示的缺点是它强烈地依赖于地图点，这带来了当没有足够的点标志时不能拟合可靠平面的限制。PLP-SLAM的优化器只保留高质量的平面，并忽略了较小的平面，这也会导致无纹理场景中的建图失败，结果如表5所示。

表5 在数据集TUM RGB-D上评估的RGB-D SLAM

<img src="https://img-blog.csdnimg.cn/b2bd7f1e78aa445a846fda89c18bf177.png" style="zoom:75%;" />

### 4.3 重定位模块

重定位模块在不使用除地图之外的任何先验信息的情况下估计相机姿态，作者通过进行基于单目地图的定位来评估重定位模块。具体实验方法为：基于RGB-D SLAM系统预先构建的地图，给定该数据序列中的每一幅图像，通过图像检索和EPnP估计初始摄像机姿态，然后使用3D-2D点和线对应的运动BA进行优化。绝对相机位置误差(APEs)如图5所示，蓝线表示使用点和线进行姿态优化(基于预先构建的RGB-D点线图)，与仅使用点(基于预先构建的RGB-D点云图)相比，显示出明显的较小误差。

<img src="https://img-blog.csdnimg.cn/61fbb4dc339c4d1596101af8234cb0f8.png" style="zoom:80%;" />

图5 重定位模块的评估结果

### 4.4 其他定性及定量结果

在表6和表7，作者在EuRoC MAV上测试了单目和双目的性能，包含两个室内房间和一个工业场景。需要注意的是，由于实例平面分割CNN的失败，在工厂图像序列MH01-05上没有利用平面重建的结果。线段的利用几乎没有提高数据集EuRoC MAV的精度，在双目PL-SLAM中也提到了类似的结果。尽管如此，与单点方法相比，PLP-SLAM系统在大多数序列上显示出略优的性能，特别是重建的地图更直观。

表6 在数据集EuRoC MAV上评估的单目SLAM

<img src="https://img-blog.csdnimg.cn/607f8c5af8ed4e6a8b1bad691de174e6.png" style="zoom:80%;" />

表7 在数据集EuRoC MAV上评估的双目SLAM

<img src="https://img-blog.csdnimg.cn/7cbc207c62a1468c94a9949ad39c5052.png" style="zoom:80%;" />

在ICL-NUIM数据集上的单目SLAM的更多测试案例如图6所示，图7给出了TUM上RGB-D SLAM的另一个例子，作者还在图8中提供了PLP-SLAM和PL-VINS之间的比较，定性地表明PLP-SLAM的单目和双目系统都提供了高度精确的点和线图，图9给出了另一个具有平面重建的例子。

<img src="https://img-blog.csdnimg.cn/412f9b860ca84adabc37289388ea9211.png" style="zoom:100%;" />

图6 在ICL-NUIM数据集上单目PLP-SLAM重建地图的定性结果

<img src="https://img-blog.csdnimg.cn/d47f46bde6a84fe499314571f8437363.png" style="zoom:80%;" />

图7 在TUM RGBD数据集上重建的点线面图的定性结果

<img src="https://img-blog.csdnimg.cn/9371f099e6a243faafa232a8f307f6a9.png"  />

图8 PLP-SLAM的双目系统和PL-VINS之间的定性比较

<img src="https://img-blog.csdnimg.cn/841e78f7564c4fbcb3365072a61b416d.png"  />

图9 在EuRoC V1 03困难序列的双目SLAM的定性结果

## 5. 结论

在论文"Structure PLP-SLAM: Efficient Sparse Mapping and Localization using Point, Line and Plane for Monocular, RGB-D and Stereo Cameras"中，作者提出了一种稀疏视觉SLAM，它利用点和线段进行鲁棒的相机定位，同时利用平面直观地说明场景结构，并综合评价给出了各种重建地图的定性结果。作者指出，一个重要的研究方向是将其推广到更具挑战性的场景，例如无纹理场景。
