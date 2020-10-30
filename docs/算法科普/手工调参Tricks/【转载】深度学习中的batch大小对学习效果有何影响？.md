> 作者：程引
> https://www.zhihu.com/question/32673260/answer/71137399
> 本文已由原作者授权，未经允许，不得二次转载。

**谈谈深度学习中的Batch_Size**

Batch_Size（批尺寸）是机器学习中一个重要参数，涉及诸多矛盾，下面逐一展开。

**首先，为什么需要有 Batch_Size 这个参数？**

Batch 的选择，**首先决定的是下降的方向**。如果数据集比较小，完全可以采用**全数据集 （ Full Batch Learning ）**的形式，这样做至少有 2 个好处：其一，由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。其二，由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用**Rprop** 只基于梯度符号并且针对性单独更新各权值。

对于更大的数据集，以上 2 个好处又变成了 2 个坏处：其一，随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。其二，以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来 **RMSProp** 的妥协方案。

**既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？**

所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。这就是**在线学习（Online Learning）**。线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，**难以达到收敛**。

![优化示意图](https://img-blog.csdnimg.cn/20200302205726871.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**可不可以选择一个适中的 Batch_Size 值呢？**

当然可以，这就是**批梯度下降法（Mini-batches Learning）**。因为如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的。

**在合理范围内，增大 Batch_Size 有何好处？**

- 内存利用率提高了，大矩阵乘法的并行化效率提高。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，对于相同数据量的处理速度进一步加快。
- 在一定范围内，一般来说 Batch_Size 越大，其确定的下降方向越准，引起训练震荡越小。

**盲目增大 Batch_Size 有何坏处？**
- 内存利用率提高了，但是内存容量可能撑不住了。
- 跑完一次 epoch（全数据集）所需的迭代次数减少，要想达到相同的精度，其所花费的时间大大增加了，从而对参数的修正也就显得更加缓慢。
- Batch_Size 增大到一定程度，其确定的下降方向已经基本不再变化。

**调节 Batch_Size 对训练效果影响到底如何？**

这里跑一个 LeNet 在 MNIST 数据集上的效果。MNIST 是一个手写体标准库，我使用的是 Theano 框架。这是一个 Python 的深度学习库。安装方便（几行命令而已），调试简单（自带 Profile），GPU / CPU 通吃，官方教程相当完备，支持模块十分丰富Profile），GPU / CPU 通吃，官方教程相当完备，支持模块十分丰富（除了 CNNs，更是支持 RBM / DBN / LSTM / RBM-RNN / SdA / MLPs）。在其上层有 Keras 封装，支持 GRU / JZS1, JZS2, JZS3 等较新结构，支持 Adagrad / Adadelta / RMSprop / Adam 等优化算法。

![实验用到的网络](https://img-blog.csdnimg.cn/20200302210218328.png)
![实验结果](https://img-blog.csdnimg.cn/20200302210232682.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
运行结果如上图所示，其中绝对时间做了标幺化处理。运行结果与上文分析相印证：
- Batch_Size 太小，算法在 200 epoches 内不收敛。
- 随着 Batch_Size 增大，处理相同数据量的速度越快。
- 随着 Batch_Size 增大，达到相同精度所需要的 epoch 数量越来越多。
- 由于上述两种因素的矛盾， Batch_Size 增大到某个时候，达到**时间上**的最优。
- 由于最终收敛精度会陷入不同的局部极值，因此 Batch_Size 增大到某些时候，达到最终收敛**精度上**的最优。

**文章相关超链接**
- **为什么 feature scaling 会使 gradient descent 的收敛更好?** https://www.zhihu.com/question/37129350/answer/70964527#
- **theano安装**：http://deeplearning.net/software/theano/install.html#install
- **theano教程**：http://deeplearning.net/tutorial/contents.html