
## 【导语】Diffusion Models 是近期人工智能领域最火的方向，也衍生了很多实用的技术。最近开始整理了几篇相关的经典论文，加上一些自己的理解和公式推导，分享出来和大家一起学习。
### 第 0 篇：《Deep Unsupervised Learning using Nonequilibrium Thermodynamics》

@[toc]
## 摘要

之前想拟合数据分布的时候，基本都是想一步到位，使用一个复杂的函数来拟合，但往往这个函数不可解析计算或者计算太复杂。而一些可解析的分布函数又难以表征比较复杂的数据分布。这篇文章从热力学扩散中得到灵感，提出一种扩散模型（diffusion model），把原始信息逐步扩散到一个简单明了并能解析计算的分布（比如正态分布），然后学习这个扩散（diffusion）过程，最后在进行反传（reverse diffusion），从一个纯噪声逐步恢复出原始信息。

## 1、 扩散（diffusion）

扩散模型巧妙地把学习数据分布的过程转化成学习数据分布的扩散过程，因为每一步的扩散都是很小的扰动，所以很好学习，而且能表征任意形式的数据分布。作者使用马尔科夫链来实现这个扩散过程，每一步的分布都是在上一步的分布上加上一定的噪声得到：

![，扩散过程](https://img-blog.csdnimg.cn/067a54ace1e34870ac1e43f01c85a872.png#pic_center)

（1） 每一步的新数据都是由上这次的扩散核作用在上一次数据上产生的
（2）扩散核，也就是扩散过程的规则，
（3）q(x0) 表示x0的分布概率，q(xt|xt-1)表示马尔科夫链中的扩散核，这个式子表示一个马尔科夫过程：q(x0...T)是初始状态为q(x0)的马尔科夫链的所有状态的联合分布概率=q(x0)*q(x1)*...*q(xT)，后面的可以使用累乘表示。

## 2、 反向传播
![反向传播](https://img-blog.csdnimg.cn/7e34d349aaa54d5c8bdfecd861dee12a.png#pic_center)
在扩散率 β 很小的情况下，可以把正向扩散和反向传播的每一步都可以看成是同一种形式（比如高斯扩散）。而这个反向传播中高斯扩散的均值和方差便可以作为参数进行训练。

## 3、 训练
### 3.1 优化目标
此模型采用了最大化对数似然（交叉熵）来进行优化训练：
![交叉熵](https://img-blog.csdnimg.cn/0fa75b5c32864979b63c51742515ac65.png#pic_center)
经过一些骚作后得到 L 的下限，最大化 L 其实就是最大化其下限 K：
![K的下限](https://img-blog.csdnimg.cn/4e5d874039434f63b0663bb9b65def14.png#pic_center)
![K](https://img-blog.csdnimg.cn/3a1be3c1a8b249888ff249af12e76636.png#pic_center)![Hp-Hq](https://img-blog.csdnimg.cn/3a77c4588d6c492e85cecdab4568196d.png#pic_center)
那如何来最大化 K ，这就需要把 q 和 p 的解释式显式化，接下来详细讲解。

### 3.2 扩散特性
扩散过程是一个马尔科夫高斯扩散，每个时刻的分布只和前一个分布和扩散核有关：
![q(x|x-1)](https://img-blog.csdnimg.cn/8f85cfbd54b344e8a67903e89c03235f.png#pic_center)
设 αt = 1 - βt，则：
![x/x-1](https://img-blog.csdnimg.cn/ec75e1a810894b78b828a22cba77f7fc.png#pic_center)
将以上两个式子联立，带入，可以得到：
![xt](https://img-blog.csdnimg.cn/40ae9a76270c4a238012245ec2cf3438.png#pic_center)
两个独立同分布的高斯分布相加之后有以下公式：
![N](https://img-blog.csdnimg.cn/90a419a1bccd4526aac045e8ba32fe99.png#pic_center)
最终可以简化并推导可得：
![简化结果](https://img-blog.csdnimg.cn/678f6f3f52484f5c92d121315165d061.png#pic_center)
### 3.3 逆扩散过程q(xt|xt-1) 
q(xt|xt-1) 为扩散的过程，则 q(xt-1|xt) 为理想的逆扩散过程，利用贝叶斯可求解，而且当 β 足够小时候，逆扩散扩展可以看成和扩散过程一样的高斯扩散形式：
![逆扩散](https://img-blog.csdnimg.cn/66aac2acbc614320ace80f970d2f8ba2.png#pic_center)标准正态分布的展开为：
![标准高斯形式](https://img-blog.csdnimg.cn/f586e4fd5e444e6da171a6e5dcef44e3.png#pic_center)
对比上面两式子可得：
![13](https://img-blog.csdnimg.cn/1fb482e9d81c4dda87cc70e972d76987.png#pic_center)
根据 3.2 的扩散特性可消掉 x0，得
![消掉x0](https://img-blog.csdnimg.cn/3acd191097fd46fd8bb6ec509d7c82d4.png#pic_center)
最终得到 q(xt|xt-1)  的表达形式（其中的噪声则是正向扩散时添加的噪声）：
![q的最终表达式](https://img-blog.csdnimg.cn/8347d5e65216489b908fa0bac2f5e48d.png#pic_center)
### 3.4 生成过程 p(xt|xt-1) 
目前在 K 的表达式中还需要计算 p(xt|xt-1) 的解析式，设生成过程和扩散过程一样，是使用高斯核的马尔可过程，其高斯核的均值和方差则有多层网络根据该步生成的 xt 的回归通过一些计算间接得出：
![网络结构图](https://img-blog.csdnimg.cn/4cb0733ae2894e34a7f6cd7d05b32a8c.png#pic_center)
首先输入 Xt，由多层网络输出一个多通道 y，然后在通道维度分成两部分，一部分用以计算均值，另一部分计算方差。
![均值和方差](https://img-blog.csdnimg.cn/6959a7906da142ed8b465968c454d728.png#pic_center)
### 3.5训练目标
经过上面几个步骤，q(xt|xt-1) 、q(xt-1|xt) 和 p(xt-1|xt) 的解析表达式都得到，这样就可以带入3.1 中的 K 中，使用梯度下降法来最大化 K 了。训练完毕后，便可以使用训练好的网络来计算 p(xt-1|xt) 中的均值和方差，便可以一步一步地从 π（xt）中生成样本。
![训练目标](https://img-blog.csdnimg.cn/3ca7d7381c8644948179662d29910ddd.png#pic_center)
## 4、总结
文章从热力学扩散中得到灵感提出了一个很新颖的扩散模型来表征数据的分布，把学习数据分布的过程转化成学习数据分布的扩散过程。通过马尔科夫扩散链，逐步把原始分布扩散到一个纯噪声分布，相当于学习到了原始分布到噪声分布的映射。因为每一个马尔科夫过程都是很小的扰动，所以比较容易学习，并且能灵活地表征任意数据的分布。
