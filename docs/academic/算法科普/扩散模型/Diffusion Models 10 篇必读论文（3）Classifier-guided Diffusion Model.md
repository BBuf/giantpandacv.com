## 【导语】Diffusion Models 是近期人工智能领域最火的方向，也衍生了很多实用的技术。最近开始整理了几篇相关的经典论文，加上一些自己的理解和公式推导，分享出来和大家一起学习，欢迎讨论：702864842(QQ)，https://github.com/Huangdebo。
## 第 3 篇：《Diffusion Models Beat GANs on Image Synthesis》

## 1、摘要
目前生成模型有好几种，包括 GANs 和 likelihood-based models 等，目前在生成任务上，依然是 GANs 取得最好的效果，但 GANs 难以训练和扩展，限制了其应用。虽然 diffusion model 近几年有了大的发展，但在生成任务上，比较 GANs 还是略逊一筹。作者认为 diffusion model 在目前还没有被深度研究优化，于是对目前的 diffusion model 进行大量的消融优化，并借鉴 conditional GANs 来训练 conditional diffusion model，并使用分类信息来引导生成过程，大幅度提到了 diffusion model 的性能，并超越了 GANs。

## 2、背景
### 2.1 diffusion model 的发展
diffusion model 是通过一个逆加噪过程来生成样本，比如从一个纯噪声分布 xT，逐步地去噪，生成 xT-1, xT-2....。直到生成高质量的图像 x0。在生成个过程中，每个 xt 都可以看做是 x0 和某个分布的噪声的叠加，而 diffusion model 的训练过程则就是学习逐步去掉这个噪声。在 DPM 和 DDPM 中，把正向加噪和去噪的过程都看成一个马尔科夫链，因而需要比较多的步骤来生成样本。后来 DDIM 把去噪的过程设计成非马尔科夫过程，使得模型能利用少量的去噪步骤便可生成高质量图像，而且把噪声的方差设成 0 后，生成过程便具有了确定性（一致性）。
### 2.2 生成样本的质量指标
目前在模拟人类对生成的图像质量的评价上，主要有两种指标。一个是 Inception Score (IS)，主要是评价一个生成模型对 Imagenet 数据集的分布的学习和生成图像的真实性，缺点是并没有对生成图像的多样性有很好的评价。另一个指标是 Fréchet Inception
 Distance(FID)，主要评价生成的图像在 Inception-V3 的特征空间上的差异。可以看成 IS 偏向评价生成样本的真实性，FID 则是偏向评价生成样本的多样性。
 
##  3、结构优化
在 DDMP 中作者使用了 Unet 作为 diffusion model 的主干结构，该 Unet 使用了多节残差结构，而且先经过降采样然后上采样处理，并且在相同的空间分辨率的层中加上跳跃连接。额外地，作者还使用了多个全局注意力层，并在在每个残差块中加入了时间的 embedding。一些研究发现，对 Unet 上的优化能提高模型在一些数据集上的表现，于是作者在diffusion model 的结构上做了大量的消融实验，以达到提升模型性能的效果：

 - 增加网络的深度，减小宽度，保持整个模型的大小一致
 - 增加 attention 的头
 - 使用 32x32, 16x16, 8x8 多种分辨率的 attention，而不是一个 16x16的分辨率
 - 使用 BigGAN 中的残差组块来做上下采样
 - 使用 1/(根号2) 来作为残差连接的系数
 
加深 Unet 能直接提高模型的性能，但也明显增加了模型的推理时间
![不同结构的 Unet 对模型性能的影响](https://img-blog.csdnimg.cn/93d847946f1849769d050de17f48b041.png#pic_center)
更多的注意力头和少的通道也能提升模型的性能，在残差块中使用了 AdaGN 也能显著地改善 FID 指标。
![注意力头层的头和通道数以及归一化层对模型性能的影响](https://img-blog.csdnimg.cn/738816b8b08d43c2bb8778a1a5cf1698.png#pic_center)
## 4、分类器引导
### 4.1 生成过程
借鉴 GANs，作者也在 diffusion model 中引入了分类器引导，在归一化层中加入了类别信息，并且使用分类器产生的梯度来引导模型的生成过程（其中的分类器是针对不同时刻的 xt 进行训练得来）：
![加入分类信息的生成过程](https://img-blog.csdnimg.cn/d5a54f6d1c5e4b96b883f34ab1b684d4.png#pic_center)

现在回顾一下 diffusion model 的推理过程 pθ(xt|xt+1)，是一个高斯分布：
![diffusion model的推理过程](https://img-blog.csdnimg.cn/46d5cbc078be4fdf9fa633be9fb27d47.png#pic_center)
假设 p(y|xt) 在 μ 附近的曲率比较低，所以可以使用低阶的泰勒展开得：
![分类信息泰勒展开](https://img-blog.csdnimg.cn/02493596a48d43fa9448d56771436d5b.png#pic_center)
与式子（4）结合后得到：
![加入分类信息的生成过程](https://img-blog.csdnimg.cn/2370f6d05a54486b965e429a0a54483e.png#pic_center)
这样便可利用式子（2），在目标函数中加入分类器的梯度信息（目标函数的具体情况可以看 DDPM 的介绍）。
### 4.2 DDIM 中引入条件信息
对于 DDIM，由于其生成的过程是非马尔科夫过程，所以情况和 DDMP 不一样，其 score function 为
![DDIM 的 score function](https://img-blog.csdnimg.cn/b6476545d69c4cfabcd6c08ad693192f.png#pic_center)
添加了分类梯度信息后：
![加入分类信息后的  score function](https://img-blog.csdnimg.cn/2763db9c7b944589a78a8a5dff6c143f.png#pic_center)
最后可以得出生成中模型预测的噪声转化成：
![最终的噪声预测](https://img-blog.csdnimg.cn/d9d4e54f50f44966a0f27922d454d419.png#pic_center)
于是引入分类器引导之后，DDPM 和 DDIM 的生成过程转变为：
![加入分类信息的 DDPM 和 DDIM](https://img-blog.csdnimg.cn/28ad280f5fc545319852475925dfa7d3.png#pic_center)
训练conditional diffusion model 的时候，利用带有噪声的xt来训练一个分类器 p(y|xt,t)，然后通过式子（2）加上 p(y|x)的信息，并且在normal 层添加了class embedding。在生成的时候再加上p(y|x)的信息指导整个过程。

### 4.3 分类器引导的权重
作者使用了不同的权重来对比试验，发现在unconditional diffusion model 和 conditional diffusion model中，分类器引导都能大幅度地提升模型的性能，即使是在 unconditional diffusion model中使用分类器引导，只要权重足够大，也能接近没有使用分类器引导的 conditional diffusion model 的性能。
![conditional 和 guidance 对模型性能的影响](https://img-blog.csdnimg.cn/890d017a01114354b4d6965b57c5dbd6.png#pic_center)
试验结果还发现，提高 gradient scale 可以提高生成样本的精度（真实度），也就是想生成猫就是猫，想生成狗就是狗；但会降低生成样本的多样性。所以调整 gradient scale 可以在生成结果的多样性(FID）和真实度(IS)中权衡。
![gradient scale 对生成样本的IS 和 FID 的影响](https://img-blog.csdnimg.cn/3b5518d6bb5e439591bdcb8dd6354857.png#pic_center)
## 5、总结
作者对目前的 diffusion model 进行大量的消融优化，优化了模型的主干网络结构，大幅度提到了 diffusion model 的性能，并超越了 GANs。之外，作者引入了分类器指导技术，可以使得 diffusion model 也能完成类别指定的任务，而且通过调节分类器指导的权重，可以调节生成的数据的真实性和多样性。最后，配合上上采样技术，还可以生成更高质量的高分辨率图像。









