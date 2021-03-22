## Int 4量化用于目标检测

【GiantPandaCV】文章2019 CVPR，讲的是Int 4量化用于目标检测，主要是工程化的一些trick。

文章介绍：文章是2019 CVPR 《Fully Quantized Network for Object Detection》，没有开源代码。
感受：这篇文章主要是做了实践工作，可以看作是低bit量化(**Int 4**)用于目标检测的一些trick。

《Quantization and training of neural networks for efficient integer-arithmetic-only inference》简称**IAO**

量化用于目标检测有以下困难：
 **1、Hardware-friendly end-to-end quantization**
 现有的一些量化算法（Dorefa-net、《Quantized neural networks:Training neural networks with low precision weights andactivations》）只是在部分操作做量化，比如卷积操作和矩阵乘法，还有些操作是全精度的。

这个会带来两个问题：
 (1)、一些操作在训练中没办法进行量化，比如batch normalization（Dorefa-net），这会造成训练和验证之间的不匹配(mismatch)和难以收敛；
 (2)、在推理时候，还是有浮点数操作，这让数据在int arithmetic和float arithmetic之间转化，影响推理速度。

**2、Low bitwidth quantization on complex tasks**
 超低bit的量化如binary NN和Ternary NN精度上难以满足，8-bit的量化算法比较成熟，再低bit的Int算法即**Int-4**。在IAO中，低于8-bit的quantization-aware finetune会不稳定且难以收敛。

作者发现，糟糕的精度和收敛是量化模型的一些敏感操作的不稳定造成的。
1、在batch normalization中，非常小的batch在做finetune时候，会导致统计量的估算值不准。
2、bn之后的activation会包含离群值，这个也会导致精度的损失。
3、不同通道的模型的权值会有不同的值域，因此直接layer-wise的bn会不准确。

针对上述的问题，采用下面的**trick**来提升quantization-aware finetune：
 1、在做quantization-aware finetune时候，固定bn层的参数；然后归一化activation，归一化的参数是模型训练完的时候的均值和方差。
 2、用小的训练数据集去调整(activation)激活函数的阈值(要clip的阈值)，用百分比的方法丢弃离群值和截断量化的激活函数值和梯度。
 3、对所有的参数，采用channel-wise的量化。

**量化神经网络的过程：**
 1、全精度训练
 2、quantization-aware finetune（只在前向传播）
 3、fully-quantized inference，这里要fold BN，激活函数和模型参数都要量化到低bit，且没有浮点数的操作

**一些细节：**
 1、mapping scheme采用均匀分布的量化策略，且是非对称量化，就是有zero-point。
 2、weight quantization：采用channel-wise的方式
 3、激活函数值的量化：量化所有的activation，从input，到送进anchor回归和NMS的最后的activation都进行量化。激活函数截断的阈值采用EMA，与IAO文章中的方法一样。为了让4-bit更加稳定，首先从训练集中随机采样n个batches数据做校准(calibration)，在这个采样的数据中做验证(跑一次训练中的evaluation)，记录下每层的activation，并且让activation在[$1-\gamma$，$\gamma$] (0 < $\gamma$ < 1) 内，文章设置n=20，$\gamma$=0.999.
 4、折叠bn。在quantization-aware finetune中，固定bn的参数，不更新bn的均值和方差。
 5、对齐zero-point：zero-point主要用于zero-padding
 6、上采样和element-wise的操作（针对FPN）：上采样都采用最近插值( nearest interpolation); element-wise的加法跟IAO的一样，就是为了能有bit-shit的优化



下面是一些实验数据和表格：

![Int4量化与全精度对比](https://img-blog.csdnimg.cn/20210309230414439.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)


![消融实验](https://img-blog.csdnimg.cn/20210309230041782.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

![FreezeBN与不同bit的对比](https://img-blog.csdnimg.cn/20210130231503929.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

![截断激活函数阈值的百分比](https://img-blog.csdnimg.cn/20210130231528212.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

![与其他量化方法的对比图](https://img-blog.csdnimg.cn/20210130231610716.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)