推荐阅读：
- [一文弄懂 Diffusion Model](https://mp.weixin.qq.com/s/G50p0SDQLSghTnMAOK6BMA)
- [35张图，直观理解Stable Diffusion](https://mp.weixin.qq.com/s/8C2RqYrHZTpFFzaHIbPhRw)
- [如何简单高效地定制自己的文本作画模型？](https://mp.weixin.qq.com/s/vFbUcnlaW-JRPZmGCHTsfQ)
- [Diffusion Model的演进 NeurIPS 2022最佳论文：Imagen](https://mp.weixin.qq.com/s/kopZJpccN3sFPym7yp7Cuw)
- [不需要训练来纠正文本生成图像模型的错误匹配](https://mp.weixin.qq.com/s/AyZkledU3mUSktWBPR0bFw)
- [Google Brain提出基于Diffusion的新全景分割算法](https://mp.weixin.qq.com/s/CXMzZd0JP0XBJzEPhPmLvA)

# Scalable Diffusion Models with Transformers

### 1. 论文信息

标题：Scalable Diffusion Models with Transformers

作者：William Peebles, Saining Xie

原文链接：https://arxiv.org/abs/2212.09748

代码链接：https://www.wpeebles.com/DiT

### 2. 引言



机器学习正在经历一场由基于self-attention的Transformer的革命。 在过去的五年中，用于自然语言处理的神经架构、基础视觉模型和其他几个重要的领域在很大程度上都被 transformers 纳入了统一的大框架中。在统一大模型上，ViT似乎已经逐渐取代了CNN原有的江湖地位，开始成为主流视觉架构的基本模型，但U-Net在扩散模型领域仍然一枝独秀，无论是带领text-to-image出圈的DALL·E2，还是真正引发人们对图像生成文本模型讨论的Stable Diffusion，都采用了U-Net的结构进行编码，而没有使用Transformer作为图像生成架构。但是self-attention的思想显然能带给模型选择更多的灵活性，那么能否将Transformer与diffusion的生成过程进行结合呢？kaiming在Meta重要的合作者Saining Xie大神（但现在应该跳槽去了New York University）给出了肯定的回答。

![](https://img-blog.csdnimg.cn/02ecd552d0f241029294db3f2ec70293.png)

DDPM可以说是奠定了diffusion的基础，属于开山之作级别的工作，DDPM首次引入了用于扩散模型的 U-Net 主干。 这个backbone是基于一个自回归生成模型，采用了卷积的结构，主要由 ResNet的残差块组成。 与标准 U-Net相比，额外的空间self-attention block（Visual Transformer 中的重要组成部分）带来了以较低的resolution。 而后续Dhariwal等人的工作ADM消融了 U-Net 的几种架构选择，例如使用自适应normalization为卷积层注入条件信息和通道数。 然而，主流的U-Net 的结构设计在很大程度上保持不变。

通过这项工作，我们旨在揭开扩散模型中架构选择的重要性的神秘面纱，并为未来的生成模型的建模研究提供一个重要的新baseline。 我们表明 U-Net 学习到的inductive bias对diffusion model的性能并不是至关重要的，并且可以很容易地用主流的视觉架构来完成相应的替换。 因此，扩散模型已准备好从最近的架构统一趋势中受益——例如，通过继承其他领域的最佳的基础模型训练方法，以及保留可扩展性、稳健性和效率等。 标准化的视觉基础模型也将为跨领域研究开辟新的可能性。在本文中，作者关注基于 Transformer 的新型扩散模型。 我们称它们为 Diffusion Transformers，或简称为 DiTs。 

![](https://img-blog.csdnimg.cn/a564f057998c414c84cb587565bb0191.png)

DiTs 遵循 Vision Transformers (ViTs)的设计原则，与传统的卷积网络（ResNet等）相比，它能提供更强的灵活性。根据上图我们可以发现，在基于ImageNet这一数据集的模式下，论文提出的 DiT 模型在 400K 次训练迭代时的 FID-50K更优。 随着模型shot的增加，FID 的性能稳步提高，而且根据右图发现论文提出的最好模型 DiT-XL/2 的计算效率也有明显优势，优于所有先前基于 U-Net 的扩散模型，如 ADM 和 LDM。

### 3. 方法

下面来具体介绍 Diffusion Transformers (DiTs) 这种新的扩散模型架构。 首先明确，本文提出方法的目标是尽可能忠实于标准的Transformer架构，以保留其缩放属性，从而完成对基础模型的统一。DiTs方法主要框架如下：

![](https://img-blog.csdnimg.cn/0e8a6982d9f344ecb647c3b13df03fe1.png) 由于论文的重点是训练基于视觉的 DDPM，因此 DiT 基于ViT的基本架构，该架构对空间上的序列完成建模。 DiT 保留了 ViTs 的许多最佳实践。 在本节中，我们描述了 DiT 的前向传播，以及 DiT 类设计空间的组成部分。

- DiT  blocks的设计： 在 patchify 之后，输入 tokens 由一系列Transformer Block处理。 除了噪声图像输入之外，扩散模型有时还会处理额外的条件信息，例如噪声时间步长 $t$、类别标签 $c$、自然语言等。论文探索了四种以不同方式处理condition输入的Transformer变体。 这些设计对标准 ViT 块设计进行了微小但重要的修改。 

- In-context conditioning：我们只是将 $t$ 和 $c$ 的向量嵌入附加为输入序列中的两个附加标记，处理这些和图像标记的方式一致。在最后一个块之后，该方法仅仅从序列中删除条件标记。 这种方法引入的新 GFLOPs （计算量）可以忽略不记。

- Cross-attention block：可以将 $t$ 和 $c$ 的 embedding 连接成一个长度为2的序列，与图像的token序列分开。 Transformer 块被修改为在多头自注意块之后包含一个额外的多头交叉注意层，也类似于 LDM 用于 调节类标签。Cross-attention block为模型增加了最多的 GFLOPs，大约有 15% 的开销。

- Adaptive layer norm (adaLN) block： 随着Adaptive layer norm 在 GANs中已经成功应用了 ，所以本文探索用Adaptive layer norm (adaLN) 来替换 transformer 块中的标准层的正则化方式。 我们不是直接学习按维度缩放和移动参数 $\gamma$ 和 $\beta$，而是从 $t$ 和 $c$ 的嵌入向量的总和中回归它们。 在我们探索的三个模块设计中，adaLN 添加的 GFLOPs 最少，因此计算效率最高。 

  ![](https://img-blog.csdnimg.cn/6fd2f05664e44c84875810e9ab5b3a8d.png)

- adaLN-Zero block： ResNets 的先前工作发现将每个残差块初始化为恒等函数是有益的。 而基于U-Net的diffusion model也使用类似的初始化策略。 论文探索了 adaLN DiT 块的修改，它也完成恒等映射。 除了回归 $\gamma$ 和 $\beta$ 之外，这个模块还回归维度缩放参数 $\alpha$，这些参数紧接在 DiT 块内的任何剩余连接之前应用。 然后直接初始化 MLP 以输出所有 $\alpha$ 的零向量； 这会将完整的 DiT 块初始化为恒等函数。 与普通的 adaLN 块一样，adaLN-Zero 向模型添加的 GFLOPs 可以忽略不计。

更多细节请查看原论文。

### 4. 实验

为了验证DiTs的最终效果，研究者将DiTs沿“模型大小”和“输入 tokens 数量”两个轴进行了缩放。

具体来说，他们尝试了四种不同模型深度和宽度的配置：DiT-S、DiT-B、DiT-L和DiT-XL，在此基础上又分别训练了3个潜块大小为8、4和2的模型，总共是12个模型。

![](https://img-blog.csdnimg.cn/0e60e8fc0d2543deb911ec0ef19d636c.png)

从FID测量结果可以看出，就像其他领域一样，增加模型大小和减少输入 tokens 数量可以大大提高DiT的性能。FID是计算真实图像和生成图像的特征向量之间距离的一种度量，越小越好。

这也就意味着，较大的DiTs模型相对于较小的模型是计算效率高的，而且较大的模型比较小的模型需要更少的训练计算来达到给定的FID。其中，Gflop最高的模型是DiT-XL/2，它使用最大的XL配置，patch大小为2，当训练时间足够长时，DiT-XL/2就是里面的最佳模型。

![](https://img-blog.csdnimg.cn/6ed8bdda447949c5bbec7e97b1e84e64.png)

于是在接下来，研究人员就专注于DiT-XL/2，他们在ImageNet上训练了两个版本的DiT-XL/2，分辨率分别为256×256和512×512，步骤分别为7M和3M。

当使用无分类器指导时，DiT-XL/2比之前的扩散模型数据都要更好，取得SOTA效果：在256×256分辨率下，DiT-XL/2将之前由LDM实现的最佳FID-50K从3.60降至了2.27。并且与基线相比，DiTs模型本身的计算效率也很高：DiT-XL/2的计算效率为119 GFLOPs，相比而言LDM-4是103 GFLOPs，ADM-U则是742 GFLOPs。

### 5. 结论

论文提出了 Diffusion Transformers (DiTs)，这是一种简单的基于 transformer 的扩散模型backbone，它优于先前的 U-Net 模型，并继承了 transformer 模型类的出色的特性。 鉴于本文中有希望的扩展结果，未来的工作应该继续将 DiT 扩展到更大的模型和tokens 数量。 DiT 也可以作为 DALL E 2 和 Stable Diffusion 等文本到图像模型的嵌入式主干进行探索。

