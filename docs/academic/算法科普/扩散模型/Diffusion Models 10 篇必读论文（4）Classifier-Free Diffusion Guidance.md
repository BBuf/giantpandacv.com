## 【导语】Diffusion Models 是近期人工智能领域最火的方向，也衍生了很多实用的技术。最近开始整理了几篇相关的经典论文，加上一些自己的理解和公式推导，分享出来和大家一起学习，欢迎讨论：702864842(QQ)，https://github.com/Huangdebo。
## 第 3 篇：《Classifier-Free Diffusion Guidance》


## 1、摘要
经过 DDPM 和 DDIP 和 classifier-guided diffusion model 等技术的发展，diffusion model 生成的效果已经可以超越 GANs，称为一种生成模型的直流。尤其是 classifier-guided diffusion model 可以让生成图像的效果在多样性(FID）和真实度(IS)中权衡取舍。但 classifier-guided diffusion model  需要额外训练一个分类器，而且是使用带噪声的图像来训练的，所以就不能用之前训练好的一般分类器，而且从形式上看，classifier-guided diffusion model 加了分类器后，有点类似 GANs 一样，加入了分类器梯度的对抗的机制，而 GAN 在这些数据上的 IS 和 FID 评价都不错，故不清楚是生成时分类器的引导起了作用还是训练时的对抗机制起了作用是利用生成过程和分类器梯度的对抗。
针对这些问题，作者提出了一个 classifier-free guidance model，在一个模型中整合了 conditional 和 unconditional model，并通过调整两种模型的 score 的权重来在多样性(FID）和真实度(IS)中权衡取舍。

## 2、背景
回顾一地下扩散模型，其前传过程是一个马尔科夫过程：

![前传马尔科夫过程](https://img-blog.csdnimg.cn/direct/4cdfd907e43d460499b8d9f6b525e734.png#pic_center)

其中的 λ 可以看成是每一步生成过程量 Zλ 的信噪比。针对初始信息 x 来说，前传则可以表达成扩散过程的逆过程：

![扩散逆过程](https://img-blog.csdnimg.cn/direct/1e523c9860cd4f029cd94080aed0826a.png#pic_center)

其中的均值 μ 为：

![均值](https://img-blog.csdnimg.cn/direct/f97879e887f545e3a2ffb482c517355d.png#pic_center)

生成过程是由一个纯噪声X~N（0,1），开始逐步去噪：

![去噪过程](https://img-blog.csdnimg.cn/direct/062d9de67a4b428d99d57bc5416dab86.png#pic_center)

模型的训练目标则为：

![学习目标](https://img-blog.csdnimg.cn/direct/74c8749c2aeb4af4a231815cc5b7ee20.png#pic_center)

## 3、生成过程指导
### 3.1 classifier guidance
在《Diffusion Models Beat GANs on Image Synthesis》中，通过在生成过程中的近似噪声中加入分类器梯度信息来进行指导：

![classifier-guidance 的知道](https://img-blog.csdnimg.cn/direct/333fbe7456f54d68ae9f058747f127a8.png#pic_center)

右式子看出，其实就是形成另一种近似的数据分布：

![加了指导信息后的新分布](https://img-blog.csdnimg.cn/direct/0231642424f145ef9b480929906bc207.png#pic_center)

从可视化上解释这个新的分布的特性：比如有三个类别的数据，每个类别的分布p(z|c)都是一个高斯分布（中心不一样）,乘上 p(c|z)w 之后，结果就把各自交叉部分区域的概率降低了，w 值越大效果越明显。相当于把不同类别的类分得更开了。

![新分布可视化](https://img-blog.csdnimg.cn/direct/18d1f50f4d1f4024836cce5486a11bb4.png#pic_center)

### 3.2、classifier-free guidance
在 classifier-free guidance model 中，没有利用 classifier，而是同时训练了condition model 和 unconditional model，而且使用同一个网络来实现，只需要需要输入信息中的类别信息即可，在生成过程中，则通过调整两种模型的 score 的权重来在多样性(FID）和真实度(IS)中权衡取舍。
#### 3.2.1 训练

![训练](https://img-blog.csdnimg.cn/direct/062ab96d0b844fe48aaeee9d96c08643.png#pic_center)

训练的时候，对于 conditional model，c 的取值则为图像label +1，对于 unconditional model，c 的取值则为0（因为此时对于所有类别的图像，类型c都为0，也相当于没有类别信息），代码的实现是把 c 转化成 embedding 加入到 unet中。
#### 3.2.2 生成

![生成过程](https://img-blog.csdnimg.cn/direct/d07c047ed67f4f9cbe97c58c1ce572d1.png#pic_center)

在生成过程中，则是同时使用了 conditional model 和 unconditional model 进行计算，在使用一个 w 权重来调整两者的占比，w越大的时候，conditional model 作用越大，则生成的图像越真实，也可以指定生成哪一类图像。

## 4、实验
作者的实验只是为了证明 classifier-free guidance 也可以在和 classifier guidance 一样在 IS 和 FID（即是生成样本的真实度和多样性）之间权衡取舍，而不需要额外训练一个分类器来知道生成过程。所以实验中使用了 classifier-guided model 一样的网络结构和超参数。即是这些参数对于 classifier-free guidance 来说还不是最优的，但依然可以取得了更有竞争力的性能。
### 4.1 变化 classifier-free guidance 的强度
作者在 64x64 和 128x128 的分辨率下，在 Imagenet 中训练了 classifier-free guidaned 模型，证明在没有分类器的指导下，该模型也能和 classifier guidance 或 GAN 训练截断一样，权衡取舍 IS 和 FID，甚至在某些参数下能取得很好的性能。

![通过权重调整 IS 和 FID](https://img-blog.csdnimg.cn/direct/1c5ddac376b24c3eb99ff54dc23a6077.png#pic_center)

### 4.2 变化非条件训练的概率
classifier-free guidaned 模型在训练的时候，会根据一个概率来在 conditional 和 unconditional 之间切换训练，相当于调节 unconditional 训练在模型学习中的比重。实验发现，这个参数影响比较小，也说明在模型学习中，只有一小部分能力需要用 unconditional 训练来获取，过多的 unconditional 训练反而降低模型的性能，这个和在 classifier-guided model 中发现的一致。

![unconditional 训练的影响](https://img-blog.csdnimg.cn/direct/0a3f3fd5aeea45b7a2c83e151f3f1258.png#pic_center)

### 4.3 变化采样步骤的数量
生成样本中的步骤数量是一个很重要的参数，一般来说，步骤越多，生成的样本的质量就越高。作者在 128x128 的分辨率下，使用 Imagenet 数据集来训练模型，然后分别在 128; 256; 1024 三种步骤数量下最模型进行评价，发现在步骤数量为 256 时能比较好的在样本质量和耗时时间平衡取舍。

![采样步数的影响](https://img-blog.csdnimg.cn/direct/0fb20cca445b4884820006139bd3ca7e.png#pic_center)

## 5、讨论和总结
Classifier-free guided model 可以不用重新训练一个分类器，也不需要在生成的时候再加上分类信息，只需要再训练的时候使用一个概率来随机控制 conditional 训练 和 unconditional 训练，便可实现生成的样本在真实度和多样性中权衡取舍，而且在代码实现上也是只需要在原本 diffusion model 上加若干修改便可。

## 6、代码解析

非官方代码：[classifier-free-diffusion-guidance-Pytorch](https://github.com/coderpiaobozhe/classifier-free-diffusion-guidance-Pytorch)
- unet 使用 resblock，每个block 都加入了类别 c 和时间 t 的 embedding，方式是直接与输入经过一次卷积后相加：

```python
# unet.py L110
def forward(self, x:torch.Tensor, temb:torch.Tensor, cemb:torch.Tensor) -> torch.Tensor:
    latent = self.block_1(x)
    latent += self.temb_proj(temb)[:, :, None, None]
    latent += self.cemb_proj(cemb)[:, :, None, None]
    latent = self.block_2(latent)

    latent += self.residual(x)
    return latent
```
- 每个 resblock 后面都跟着一个 self-attention

```python
# unet.py L206
layers = [
    ResBlock(now_ch, nxt_ch, tdim, tdim, self.droprate),
    AttnBlock(nxt_ch)
]
```

- conditional model 的 类别信息 c 取值为 label+1；unconditional model 的 c=0

```python
# train.py L107
for img, lab in tqdmDataLoader:
    b = img.shape[0]
    optimizer.zero_grad()
    x_0 = img.to(device)
    lab = lab.to(device)
    cemb = cemblayer(lab)
    cemb[np.where(np.random.rand(b)<params.threshold)] = 0  # unconditional model 则 c=0
    loss = diffusion.trainloss(x_0, cemb = cemb)
    loss.backward()
    optimizer.step()
```









