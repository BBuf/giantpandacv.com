[Side Window Filtering](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1905.07177.pdf)

刚开始看到这篇论文的时候，我就很感兴趣想去复现一把看看效果。这篇论文是今年 CVPR oral
且不是深度学习方向的，其核心贡献点就是：不管原来的滤波器保不保边，运用了side-window思想之后，都可以让它变成保边滤波！

于是利用业余时间，参考作者开源的matlab代码，我用C++实现了一下Side-window
盒子滤波，其他滤波器有时间再试下，下面是github的链接，读者可以去跑下代码看看效果玩下，从实验结果上看我觉得算是复现了论文的效果：

side window 盒子滤波C++复现代码：

[Ldpe2G/ArmNeonOptimization](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/tree/master/sideWindowBoxFilter)

side window 中值滤波C++复现代码：

[Ldpe2G/ArmNeonOptimization](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/tree/master/sideWindowMedianFilter)

我们来看下复现论文的效果，对于一张普通图片，经典的盒子滤波和side-window 盒子滤波的效果对比：

![](https://pic2.zhimg.com/v2-0405f16808d74467e3ba0f15e0b27dfd_b.jpg)

从滤波结果对比上可以看到，经典的盒子滤波随着对同一张图片反复应用盒子滤波的迭代次数的增加，视觉效果是越来越模糊，到了30次迭代的时候已经糊的没法看了，但是Side-window盒子滤波即使迭代了30次，对于边缘的保持还很好，和原图基本看不出大的区别，就是边缘细节有些丢失。

然后对原图加上椒盐噪声，再对比下滤波效果：

![](https://pic1.zhimg.com/v2-23553580493918ce0a45d07ea5db44b0_b.jpg)

从滤波结果对比上可以看到，经典的盒子滤波到了10次迭代的时候，虽然椒盐噪声已经很好的消除了，但是图片也变得很模糊，边缘都细节都丢失了，但是Side-
window盒子滤波却能很好的消除椒盐噪声的同时，对于边缘的保持还很好，基本上算是还原了原图。

下面从我的理解上去简单解读下这篇论文的核心思想，还有我在复现过程中的一些实现细节介绍。

目前的经典滤波算法基本都是，以某个像素点为中心，按照滤波半径，把这个包括像素点和其邻域加权线性组合得到输出，一般公式如下：

![](https://pic2.zhimg.com/v2-5a1bb3fcf5c0c92b34f1b6f34ba4f489_b.png)

Ω是以像素点 i
为中心的滤波窗口，w是滤波权值，q是原图像素值，I'是输出结果。但是这样以一个像素为中心去滤波会导致的问题是，如果一个像素点处在边缘位置（这里的边缘不是指图片的大小边界，而是指图像中物体的边缘）的话，以像素为中心去滤波会导致滤波结果的边缘部分变模糊。具体是为什么，论文中给出了分析过程。

首先来看下，论文中的一张图：

![](https://pic2.zhimg.com/v2-0bcbdb02a3cb983d94f4e3a3b8fa5ee9_b.jpg)

文中提到为了分析方便只讨论3种典型的边缘，分别是图中的 (a)阶梯状边缘、(b)斜坡状边缘和(c)屋顶状边缘。论文中也给出了这3三种边缘的形象展示：

![](https://pic2.zhimg.com/v2-9a676b73faffd75d6867fd816faa9339_b.jpg)

然后文中采用了泰勒展开去分析，首先假定，图像上(x, y)坐标点的像素值为g(x, y)，对于图中展示的情况来看，函数 g(x,
y)是连续但不可导的。对于(a)阶梯状边缘的 'a' (蓝色方框那个点)点来说，文中定义 'a-' 和 'a+' 来分别表示 'a' 点左极限 (x -
ε, y)，和右极限 (x + ε, y)，且 ε > 0。 很明显从图中可以看出来 g(x - ε, y) ≠ g(x + ε, y)
且/或（文中的用词是"and (or)"）g'(x - ε, y) ≠ g'(x + ε,
y)，导数也不等是由于边缘部分的跳跃。因此对于这两块区域的泰勒展开也是不一样的，首先来看下泰勒展开的一般公式：

![](https://pic3.zhimg.com/v2-b6dad563fbda58a5a25fafc54ff88b6a_b.png)

“泰勒公式是将一个在 x=x0 处具有n阶导数的函数 f(x) 利用关于 (x - x0) 的n次多项式来逼近函数的方法。”----百度百科

根据文中的分析，这里设定 f(x) = g(x - 2ε, y)，x0 = x - ε，则根据泰勒展开公式：

g(x - 2ε, y) ≈ f(x0) + f'(x0)(x - x0)

= g(x - ε, y) + g'(x - ε, y)(x - 2ε - (x - ε))

= g(x - ε, y) + g'(x - ε, y)(- ε)

同理，设 f(x) = g(x + 2ε, y)，x0 = x + ε，则泰勒展开得：

g(x + 2ε, y) ≈ f(x0) + f'(x0)(x - x0)

= g(x + ε, y) + g'(x + ε, y)(x + 2ε - (x + ε))

= g(x + ε, y) + g'(x + ε, y)ε

所以从两边的泰勒展开式可以得出结论，对于 'a-' 区域的滤波估计肯定是来自区域 'a' 的左边，而对于 'a+' 估计是来自于 'a'
的右边，然后类比分析区域 'b'，'c' 和 'd' 都可以得到类似的结论。

因此分析得到的结论是，如果一个像素点处于图像中的边缘位置，那么滤波的时候就应该把滤波器的边缘和该像素点对齐，而不是把滤波器的中心和该像素点对齐。受该发现的启发，文中提出了一个新的保边滤波策略，就是把每个滤波像素点都当成是潜在的边缘点，然后对于每个待滤波的像素点，生成几种不同的滤波子窗口，然后把这些滤波窗口的边缘或者角点位置和该像素点对齐，然后滤波得到结果，最后根据把这些子窗口的滤波之后的最佳重构结果作为最终的滤波结果。以上就是side window 滤波的思想。

然后文中提出了8个方向的滤波窗口，分别是上、下，左、右、左上、右上、左下和右下，最后就得到了 side window filter 的核心算法流程：

![](https://pic4.zhimg.com/v2-dbb29b0a68b174354595e75b2f95f1ff_b.jpg)

其实从核心算法逻辑来看，对原来滤波算法的改动其实不大，就是滤波的窗口位置和大小需要改动下，然后把8次的结果每个位置取重构误差最小的。然后论文中又详细分析了 box filter 和 side window box filter 对于上面提到的三种经典边缘的滤波之后的保留情况。文中给出分析的图表如下：

![](https://pic4.zhimg.com/v2-ed62ad64402b511a0016c831b2d9d397_b.jpg)

总的来说结论就是 side window box filter 对于阶梯和斜坡状的边缘都能完整的保留，而对于屋顶状边缘虽然不能完整的保留边缘，但是也比经典的盒子滤波要好很多。

在复现过程中，本来一开始是想对文中提到的8种side window去分别写对应的盒子滤波的，因为盒子滤波有个经典的优化思路，可以让运行时间不受滤波半径的影响，具体可以参考我之前写得一篇博客：

[梁德澎：移动端arm cpu优化学习笔记----一步步优化盒子滤波（Box Filter）](https://zhuanlan.zhihu.com/p/64522357)

后来仔细想了下，这8个side window其实也就是边界处理不同，核心运算逻辑都是一致的，最后就是抽象成一个函数，对于不同的side window传不同的边界参数，就不需要每个窗口写一个函数了，具体可以看看github上的代码。

然后在实现side window中值滤波的时候针对移动端想了一个加速方案，因为求中值需要对窗口内元素排序，这里直观上感觉是没什么办法加速的，不过我尝试了一个方案，利用neon指令相对原来提速了不少，有空的话可以写一篇小博客去解释，这里先埋一个彩蛋。

最后看看几组对比结果，看看在迭代10次的情况下，经典box filter 和 side window box filter的结果对比：

![](https://pic4.zhimg.com/v2-bd0a4b551ee9c660a64526758dfbb1e3_b.jpg)

​                                                                                                                               熊猫宝宝原图

![](https://pic1.zhimg.com/v2-8b093a37c961f2f2a13e27342dae80c4_b.jpg)

​                                                                                                           滤波结果, box filter, iteration = 10

![](https://pic3.zhimg.com/v2-4b9e11ff93e7a06d000cc4da629577b2_b.jpg)

​                                                                                           滤波结果, side window box filter, iteration = 10

![](https://pic3.zhimg.com/v2-c316f57022121b528a5bdbc8aca8ced6_b.jpg)

​                                                                                                                     熊猫宝宝原图+椒盐噪声

![](https://pic1.zhimg.com/v2-836872c33e07b85d37afa443f806bd4c_b.jpg)

​                                                                                                          去噪结果, box filter, iteration = 10

![](https://pic2.zhimg.com/v2-66498deaca14923ce51ed09312c2ac79_b.jpg)

​                                                                                                 去噪结果, side window box filter, iteration = 10

然后在迭代10次的情况下，经典中值 filter 和 side window 中值滤波的结果对比：

![](https://pic2.zhimg.com/v2-09f9da641ce1e769b2dbd670ff18d511_b.jpg)

​                                                                                                                                 歼20+椒盐噪声

![](https://pic4.zhimg.com/v2-f47e4404a0b00453ee0ef6cb5c7db3b7_b.jpg)

​                                                                                                           去噪结果, median filter, iteration = 10

![](https://pic2.zhimg.com/v2-cdcf4a770e4681216bdf9549c92bf6e1_b.jpg)

​                                                                                                  去噪结果, side window median filter, iteration = 10

 **相关资料：**

[AI鸡蛋：CVPR2019 Oral论文 #5176 Side Window
Filtering介绍](https://zhuanlan.zhihu.com/p/58326095)[AI鸡蛋：Sub-window Box
Filter论文介绍](https://zhuanlan.zhihu.com/p/64647829)

