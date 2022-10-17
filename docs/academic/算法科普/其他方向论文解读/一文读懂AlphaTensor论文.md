
## 前言

这篇文章的主要内容是，解读 AlphaTensor 这篇论文的主要思想，如何通过强化学习来探索发现更高效的矩阵乘算法。

## 1、二进制加法和乘法

这一节简单介绍一下计算机是怎么实现加法和乘法的。

以 `2 + 5` 和 `2 * 5` 为例。

我们知道数字在计算机中是以二进制形式表示的。

整数2的二进制表示为： `0010`

整数5的二进制表示为： `0101`

### 1.1、二进制加法
二进制加法很简单，也就是两个二进制数按位相加，如下图所示：


![](https://files.mdnice.com/user/7704/6b33b35d-7183-4419-9c75-4c297179e1e2.png)


当然具体到硬件实现其实是包含了`异或`运算和`与`运算，具体细节可以阅读文末参考的资料。


### 1.2、二进制乘法
二进制乘法其实也是通过二进制加法来实现的，如下图所示：


![](https://files.mdnice.com/user/7704/a07fbc23-3626-4380-9732-1c4d416431a9.png)


乘法在硬件上的实现本质是`移位相加`。

对于二进制数来说乘数和被乘数的每一位非`0`即`1`。

所以相当于乘数中的每一位从低位到高位，分别和被乘数的每一位进行与运算并产生其相应的局部乘积，再将这些局部乘积左移一位与上次的和相加。

从乘数的最低位开始：

若为1，则复制被乘数，并左移一位与上一次的和相加;

若为0，则直接将0左移一位与上一次的和相加；

如此循环至乘数的最高位。

从二进制乘法的实现也可以看出来，加法比乘法操作要快。

### 1.3、用加法替换乘法的简单例子


![](https://files.mdnice.com/user/7704/4dd99862-6e51-47f2-9c5a-fb17808c58ec.png)



上面这个公式相信大家都很熟悉了，式子两边是等价的

左边包含了2次乘法和1次加法（减法也可以看成加法）

右边则包含了1次乘法和2次加法

可以看到通过数学上的等价变换，增加了加法的次数同时减少了乘法的次数。

## 2、矩阵乘算法

对于两个大小分别为 `Q x R` 和 `R x P` 的矩阵相乘，通用的实现就需要 `Q * P * R` 次乘法操作（输出矩阵大小 `Q x P`，总共 `Q * P` 个元素，每个元素计算需要 `R` 次乘法操作）。

根据前面 1.2内容可知，乘法比加法慢，所以如果能减少的乘法次数就能有效加速矩阵乘的运算。

### 2.1、通用矩阵乘算法

首先来看一下通用的矩阵乘算法:

![](https://files.mdnice.com/user/7704/21b174d9-dda3-4dca-bb9c-3da25858235a.png)

如上图所示，两个大小为`2x2`矩阵做乘法，总共需要`8`次乘法和`4`次加法。

### 2.2、Strassen 矩阵乘算法


![](https://files.mdnice.com/user/7704/de6be791-02a8-4ec2-8b52-f180a0f4d2d3.png)

上图所示即为 Strassen 矩阵乘算法，和通用矩阵乘算法不一样的地方是，引入了7个中间变量 `m`，只有在计算这7个中间变量才会用到乘法。

简单用 `c1` 验证一下：


![](https://files.mdnice.com/user/7704/14f19c10-8f1d-4cc1-b18e-f50afdf2e435.png)


可以看到 Strassen 算法总共包含`7`次乘法和`18`次加法，通过数学上的等价变换减少了`1`次乘法同时增加了`14`次加法。

## 3、AlphaTensor 核心思想解读

### 3.1、将矩阵乘表示为3维张量

首先来看下论文中的一张图

![](https://files.mdnice.com/user/7704/92a59600-392c-4dea-9ae8-385164685fc8.png)

图中下方是3维张量，每个立方体表示3维张量一个坐标点。

其中张量每个位置的值只能是 `0` 或者 `1`，透明的立方体表示 `0`，紫色的立方体表示 `1`。

现在将图简化一下，以`[a,b,c]`这样的维度顺序，将张量以维度`a`平摊开，这样更容易理解：


![](https://files.mdnice.com/user/7704/c657e6ad-cac0-4140-ae3a-767a9348d3b1.png)



这个3维张量怎么理解呢？

比如对于 `c1`，我们知道 `c1` 的计算需要用到 `a1,a2,b1,b3`，对应到3维张量就是：


![](https://files.mdnice.com/user/7704/b395ea94-dfc0-4e08-9447-74bfbfd665ba.png)


而从上图可知，对于两个 `2 x 2` 的矩阵相乘，3维张量大小为 `4 x 4 x 4`。

一般的，对于两个 `n x n` 的矩阵相乘，3维张量大小为 `n^2 x n^2 x n^2`。

更一般的，对于两个 `n x m` 和 `m x p` 的矩阵相乘，3维张量大小为 `n*m x m*p x n*p`。

然后论文中为了简化理解，都是以 `n x n` 矩阵乘来讲解的，论文中以 

![](https://files.mdnice.com/user/7704/8024caeb-8d0c-46e8-ae3b-6b757cfa9000.png)


表示 `n x n` 矩阵乘的3维张量，下文中为了方便写作以 `Tn` 来表示。

### 3.2、3维张量分解

然后论文中提出了一个假设：

如果能将3维张量 `Tn` 分解为 `R` 个秩1的3维张量（R rank-one terms）的和的话，那么对于任意的 `n x n` 矩阵乘计算就只需要 `R` 次乘法。

![](https://files.mdnice.com/user/7704/281f3908-6e64-47b3-b5b0-81f66109843b.png)

如上图公式所示，就是表示的这个分解，其中的


![](https://files.mdnice.com/user/7704/68d1324e-dee2-4bf3-865f-5fa71fa12c9e.png)

就表示的一个秩1的3维张量，是由 `u^(r)` 、 `v^(r)` 和  `w^(r)` 这3个一维向量做外积得到的。

这具体怎么什么理解呢？我们回去看一下 Strassen 矩阵乘算法：


![](https://files.mdnice.com/user/7704/1f8c35ef-1783-4033-a9af-2618bce585b5.png)


上图左边就是 Strassen 矩阵乘算法的计算过程，右边的 `U`，`V` 和 `W` 3个矩阵，各自分别对应左边 `U -> a`， `V -> b` 和 `W -> m`。

具体又怎么理解这三个矩阵呢？


![](https://files.mdnice.com/user/7704/27a8fba3-a7d6-488e-873f-3aa5998d201c.png)

我们在图上加一些标注来解释，其中 `U` ， `V` 和 `W` 矩阵每一列从左到右按顺序，就对应上文提到的，`u^(r)` 、 `v^(r)` 和  `w^(r)` 这3个一维向量。

然后矩阵 `U` 每一列和 `[a1,a2,a3,a4]` 做内积，矩阵 `V` 每一列和 `[b1,b2,b3,b4]` 做内积，然后内积结果相乘就得到 `[m1,m2,m3,m4,m5,m6,m7]`了。

最后矩阵 `W` 每一行和 `[m1,m2,m3,m4,m5,m6,m7]` 做内积就得到 `[c1,c2,c3,c4]`。

接着再看一下的 `U`，`V` 和 `W` 这三个矩阵第一列的外积结果


![](https://files.mdnice.com/user/7704/28aa2e4a-413a-4ab2-8c49-73f1908db6c7.png)

如下图所示：


![](https://files.mdnice.com/user/7704/40db9598-1c0d-4a90-9f00-b79f5a8c870a.png)


可以看到 `U`，`V` 和 `W` 三个矩阵每一列对应的外积的结果就是一个3维张量，那么这些3维张量全部加起来就会得到 `Tn` 么？下面我们来验证一下：


![](https://files.mdnice.com/user/7704/83c1a483-115a-4357-9a82-8924a1ab0486.png)


![](https://files.mdnice.com/user/7704/5cd6ebd6-75fd-40f3-889b-0f89b0020c99.png)

可以看到这些外积的结果全部加起来就恰好等于 `Tn`：

![](https://files.mdnice.com/user/7704/c657e6ad-cac0-4140-ae3a-767a9348d3b1.png)

所以也就证实了开头的假设:

如果能将表示矩阵乘的3维张量 `Tn` 分解为 `R` 个秩1的3维张量（R rank-one terms）的和，那么对于任意的 `n x n` 矩阵乘计算就只需要 `R` 次乘法。

![](https://files.mdnice.com/user/7704/281f3908-6e64-47b3-b5b0-81f66109843b.png)

因此也就很自然的可以想到，如果能找到更优的张量分解，也就是让 `R` 更小的话，那么就相当于找到乘法次数更小的矩阵乘算法了。

## 通过强化学习探索更优的3维张量分解

### 将探索3维张量分解过程变成游戏

论文中是采用了强化学习这个框架，来探索对3维张量`Tn`的更优的分解。强化学习的环境是一个单玩家的游戏（a single-player game, TensorGame）。

首先定义这个游戏进行 `t` 步之后的状态为 `St`：



![](https://files.mdnice.com/user/7704/72f61237-64c7-440c-84d4-b07054a315c5.png)


然后初始状态 `S0` 就设置为要分解的3维张量 `Tn`：



![](https://files.mdnice.com/user/7704/c3929be9-e015-4eed-9fbf-77147d57c676.png)

对于游戏中的每一步`t`，玩家（就是本论文提出的 `AlphaTensor`）会根据当前的状态选择下一步的行动，也就是通过生成新的三个一维向量从而得到新的秩1张量：


![](https://files.mdnice.com/user/7704/b8160b7e-ff07-486b-8f8a-2e50df53510d.png)

接着更新状态 `St`减去这个秩1张量：

![](https://files.mdnice.com/user/7704/d5ad17d0-dd42-4be9-bf56-433ead04ed67.png)

玩家的目标就是，让最终状态 `St=0`同时尽量的减少游戏的步数。

当到达最终状态 `St=0` 之后，也就找到了3维张量`Tn`的一个分解了：

![](https://files.mdnice.com/user/7704/a25b7a8d-1a85-4b79-a64d-3753cf18a388.png)

还有些细节是，对于玩家每一步的选择都是给一个 `-1` 的分数奖励，其实也很容易理解，也就是玩的步数越多，奖励越低，从而鼓励玩家用更少的步数完成游戏。

而且对于一维向量的生成，也做了限制


![](https://files.mdnice.com/user/7704/b8160b7e-ff07-486b-8f8a-2e50df53510d.png)

就是生成这些一维向量的值，只限定在比如 `[−2, −1, 0, 1, 2]` 这5个离散值之内。

### AlphaTensor 简要解读

论文中是怎么说的，在游戏过程中玩家 `AlphaTensor` 是通过一个深度神经网络来指导蒙特卡洛树搜索（MonteCarlo tree search）。关于这个蒙特卡洛树搜索，我不是很了解这里就不做解读了，有兴趣的读者可以阅读文末参考资料。

首先看下深渡神经网络部分：

![](https://files.mdnice.com/user/7704/6e9580d2-75ee-426b-b8a3-a9b29b44f473.png)

深度神经网络的输入是当前的状态 `St`也就是需要分解的张量（上图中的最右边的粉红色立方体）。输出包含两个部分，分别是 `Policy head` 和 `Value head`。

其中 `Policy head` 的输出是对于当前状态可以采取的潜在下一步行动，也就是一维向量`(u(t), v(t), w(t))` 的候选分布，然后通过采样得到下一步的行动。

然后 `Value head` 应该是对于给定的当前的状态 `St` ，估计游戏完成之后的最终奖励分数的分布。

接下来简要解读一下整个游戏的流程，还有深度神经网络是如何训练的：

![](https://files.mdnice.com/user/7704/efec8137-07b2-4c01-99e8-0b8daa885e6b.png)

先看流程图的上方 `Acting` 那个方框内，表示的是用训练好的网络做推理玩游戏的过程。

可以看到最左边绿色的立方体，也就是待分解的3维张量 `Tn`变换到粉红色立方体，论文中提到是作了基的变换，但是这块感觉如果不是去复现就不用了解的那么深入，而且我也没去细看这块就跳过吧。

然后从最初待分解的 `Tn` 开始，输入到神经网络，通过蒙特卡洛树搜索得到秩1张量，然后减去该张量之后，继续将相减的结果输入到网路中，继续这个过程直到张量相减的结果为0。

将游戏过程记录下来，就是流程图最右边的 `Played game`。

然后流程图下方的 `Learning` 方框表示的就是训练过程，训练数据有两个部分，一个是已经玩过的游戏记录 `Played games buffer` 还有就是通过人工生成的数据。

人工怎么生成训练数据呢？

论文中提到，尽管张量分解是个 `NP-hard` 的问题，给定一个 `Tn` 要找其分解很难。但是我们可以反过来用秩1张量来构造出一个待分解的张量嘛！简单来说就是采样`R`个秩1张量，然后加起来就能的到分解的张量了。

因为对于强化学习这块我不是了解的并不深入，所以也就只能作粗浅的解读。

### 实验结果

最后看一下实验结果
![](https://files.mdnice.com/user/7704/c7808e74-2176-496e-bda7-ce6d568d0403.png)

表格最左边一列表示矩阵乘的规模，最右边三列表示矩阵乘算法乘法次数。

第一列表示目前为止，数学家找到的最优乘法次数。

第2和3列就是 `AlphaTensor` 找到的最优乘法次数。

可以看到其中有5个规模，`AlphaTensor` 能找到更优的乘法次数（标红的部分）：

两个 `4 x 4` 和 `4 x 4` 的矩阵乘，`AlphaTensor` 搜索出`47`次乘法；

两个 `5 x 5` 和 `5 x 5` 的矩阵乘，`AlphaTensor` 搜索出`96`次乘法；

两个 `3 x 4` 和 `4 x 5` 的矩阵乘，`AlphaTensor` 搜索出`47`次乘法；

两个 `4 x 4` 和 `4 x 5` 的矩阵乘，`AlphaTensor` 搜索出`63`次乘法；

两个 `4 x 5` 和 `5 x 5` 的矩阵乘，`AlphaTensor` 搜索出`76`次乘法；


## 参考资料

- https://www.nature.com/articles/s41586-022-05172-4
- https://www.youtube.com/watch?v=3N3Bl5AA5QU&ab_channel=YannicKilcher
- https://www.youtube.com/watch?v=gpYnDls4PdQ&ab_channel=HarvardMedicalAI%7CRajpurkarLab
- https://www.jobilize.com/course/section/hardware-for-addition-and-subtraction-by-openstax
- https://www.eet-china.com/mp/a94582.html
- https://baike.baidu.com/item/%E7%A1%AC%E4%BB%B6%E4%B9%98%E6%B3%95%E5%99%A8/4865151
- https://blog.csdn.net/SunnyYoona/article/details/43570853
- https://nikcheerla.github.io/deeplearningschool/2018/01/01/AlphaZero-Explained/
- https://www.youtube.com/watch?v=hmQogtp6-fs&ab_channel=GauravSen
- https://www.youtube.com/watch?v=62nq4Zsn8vc&ab_channel=JoshVarty
- https://www.youtube.com/watch?v=J3I3WaJei_E&ab_channel=%E8%B5%B0%E6%AD%AA%E7%9A%84%E5%B7%A5%E7%A8%8B%E5%B8%ABJames