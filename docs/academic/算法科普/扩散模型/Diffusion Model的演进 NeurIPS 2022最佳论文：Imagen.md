# Imagen： Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

### 1. 论文信息

标题：Imagen： Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding

作者：Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour, Burcu Karagol Ayan, S. Sara Mahdavi, Rapha Gontijo Lopes, Tim Salimans, Jonathan Ho, David J Fleet, Mohammad Norouzi

原文链接：http://export.arxiv.org/abs/2205.11487

代码链接：https://github.com/Alpha-VL/ConvMAE

### 2. 引言

NeurIPS 2022前段时间颁布了outstanding paper的入围名单，以往NeurIPS往往偏好理论分析的论文，而对一些看起来很有趣效果很惊艳的工作貌似不怎么感冒。而意料之外，但又在情理之中地，Imagen入围了这次outstanding paper的名单。其实我个人认为是Imagen其实重塑了大家对于diffusion model的认知，开辟了DALL-E 2以外的Text-to-Image的新范式。效果还是非常强的（如下图）。

![](https://img-blog.csdnimg.cn/76c2820ae1b44721ab4cedb1477647c9.png)

Multimodal learning 最近比较火，有text-to-image synthesis（DALL-E, Stable Diffusion）和image-text contrastive learning （CLIP），都成为了最近关注度超级高的领域。这些模型改变了研究界思考的模式：语言信号在视觉表征学习中，究竟可以充当什么样的角色？同时并通过创意图像生成和编辑图像的这些工作，使得AI帮助人类进行艺术创作提供了可能，造成了很强的出圈效果，引起了广泛的公众关注。为了进一步追求这个研究方向，Google引入了Imagen，一种结合了 Transformer 语言模型 (LM) 的强大功能的文本到图像扩散模型和high- resolution的Diffusion Models，在文本到图像合成中提供前所未有的照片级真实感和深度语言理解。
与仅使用图像文本数据进行模型训练的先前工作相比，Imagen背后的关键发现是来自大型 LMs 的文本embedding在纯文本语料库上预训练，对于文本到图像的合成非常有效。这其实重塑了大家对视觉语言学习的认知。因为大家往往会觉得视觉-语言里的语言信息，和纯文本语料里的语言信息，是存在domain的gap的，但这份基于bert的工作给大家的研究提供了全新的思路。


具体来讲，Imagen包含一个冻结的 T5-XXL的文本编码器，用于将输入文本映射到一系列嵌入和一个 $64\!\times\!64$ 图像扩散模型，然后是两个超分辨率扩散 生成 $256\!\times\!256$ 和 $1024\!\times\!1024$ 图像的模型。 所有扩散模型都以文本嵌入序列为条件，摆脱了classifier的guidance。 Imagen依赖于新的采样技术，允许使用大的引导权重，所以不会像原有工作一样带来样本质量下降，从而使得图像具有比以前更高的保真度，并且更好地完成图像-文本对齐。虽然概念上简单且易于训练，但Imagen 产生了令人惊讶的强大结果。此外，人主观的评价可以看出，从 Imagen生成的样本在图像文本对齐方面与 COCO 字幕上的参考图像相当。

同时，论文还引入了DrawBench，这是一套用于文本到图像评估的新结构化文本提示。DrawBench通过文本到图像的多维评估实现更深入的洞察模型，带有旨在探测模型不同语义属性的文本提示。 这些包括组合性、基数、空间关系、处理复杂文本提示或带有罕见词的提示的能力，并且它们包括创造性提示，这些提示将模型的能力极限推向训练数据范围之外生成高度难以置信的场景的能力。使用DrawBench，经过广泛的人类主观评估，可以得出结论：Imagen的性能明显优于其他同期工作。

该论文的主要贡献包括：

- 我们发现，仅在文本数据上训练的大型冻结语言模型对于文本到图像的生成是非常有效的文本编码器，并且缩放冻结文本编码器的大小比缩放图像扩散模型的大小更能显着提高样本质量 .
- 我们引入dynamic thresholding（动态阈值），一个新的扩散采样技术可利用高引导权重并生成比以前更逼真和更详细的图像。
- 我们强调了几个重要的扩散架构设计选择，并提出了Efficient U-Net，这是一种新的架构变体，它更简单、收敛速度更快并且内存效率更高。
-  我们引入了DrawBench，这是一个针对文本到图像任务的新的综合且具有挑战性的评估基准。 在DrawBench（图像生成领域一个重要的benchmark，涉及到人自主的评估）评估中，我们发现Imagen优于所有其他工作，包括OpenAI引起巨大轰动的DALL-E 2。

### 3. 方法

Imagen由将文本映射到一系列embedding的文本编码器和将这些embedding映射到分辨率不断增加的图像的级联的扩散模型组成（参见 \cref{fig:main_diagram}）。 在以下小节中，我们将详细描述这些组件中的每一个。

![](https://img-blog.csdnimg.cn/ef761655ea6a4a30bd47ecf7f87960e4.png)

#### 3.1 文本编码

由于自然语言的文本输入具有一定的复杂性，所以text到image模型需要强大的语义文本编码器。 而大型语言模型的最新进展（例如，BERT、GPT、T5）带来了文本理解和生成能力的飞跃。 语言模型是在比成对的图像文本数据大得多的纯文本语料库上训练的，因此可以接触到非常丰富和广泛分布的文本。 这些模型也普遍比当前图文模型（CLIP）中的文本编码器大很多（例如PaLM 有540B个参数，而CoCa 1B ）。因此，为文本到图像任务探索这两个文本编码器系列变得很自然。 Imagen采用了最经典的BERT、T5 和以及OpenAI的CLIP。 为简单起见，我们冻结了这些文本编码器的权重。 冻结这部分的参数可以方便embedding的离线计算，导致文本到图像模型训练期间的计算或内存占用可以忽略不计。 同时，作者发现缩放文本编码器的大小可以提高文本到图像生成的质量。

#### 3.2 无分类器引导的diffusion model

扩散模型大家都很熟悉了，实际上就是进行去噪：

![](https://img-blog.csdnimg.cn/a6ec395ee3a24698b7ea44278a38a49e.png)



Classifier-free guidance 是一种提高样本质量的方法。在采样期间使用来自预训练模型 $p(c|z_t)$ 的梯度来减少条件扩散模型的多样性，具体来说就是希望生成的图像能顺利被分类。而Classifier-free guidance 是一种替代技术，它通过在训练期间随机丢弃相应的信息$c$ 来预测最终的联合条件概率，完成无条件目标的单个扩散模型训练，从而避免来自预训练模型的不稳定性。 使用调整后的 $x$-预测 $(z_t - \sigma \tilde \epsilon_ \theta)/\alpha_t$ 执行采样，其中：
$$
\tilde{\epsilon}_\theta(z_t, c) = w\epsilon_\theta(z_t, c) + (1-w)\epsilon_{\theta}(z_t)。
$$
这里，$\epsilon_\theta(z_t, c)$ 和 $\epsilon_{\theta}(z_t)$ 是有条件和无条件的 $\epsilon$ 预测，由 $\epsilon_\theta eq (z_t - \alpha_t\hat{x}_\theta)/\sigma_t$，$w$ 是 guidance的权重。 设置 $w = 1$ 禁用无分类器指导，而增加 $w > 1$ 加强指导的效果。

#### 3.3 级联的diffusion model

Imagen对两个超分辨率模型都使用噪声调节增强，而作者发现这对于生成高保真图像至关重要。在给定条件，对低分辨率图像进行增，并在增加的level上调节扩散模型。 在训练过程中，为了增加随机性，这个level是随机选择的，而在推理过程中，每一个level都生成，然后最后挑置信度最高的（用CLIP算matching score）。 其实这样做的好处非常明显，就是由于DM生成的图像分辨率很低，因此超分DM是带噪声条件增强的，使得超分模型能感知到噪声量和噪声层次，这样超分的结果会对低分图像的噪声更加鲁棒。

原文还有很多网络有关的细节，这里就不做过多介绍。

### 4. 实验

首先看可视化还是非常厉害的，基本上可以做到高清同时保真：

![](https://img-blog.csdnimg.cn/41493366250a4a668ddbed516df9b193.png)

![](https://img-blog.csdnimg.cn/f6a472bcd52d485b9404f95a51f8d7eb.png)

可以发现相较于DALL-E 2,Imagen显然生成的图像更符合文本的描述。

在MS-COCO上，客观指标也非常有竞争力：

![](https://img-blog.csdnimg.cn/1e100c24d1c94c1588f4620d76cba67e.png)

在新提出的benchmark上，相较其他方法也有较为明显的优势。

![](https://img-blog.csdnimg.cn/bb7716e2101a4b99b9803f9f8947f909.png)

### 5. 结论

本文提出了 Imagen，它使用大型变换器语言模型和扩散模型来生成文本到图像。主要发现是使用仅在文本数据上预训练的大型语言模型作为文本编码器是有效的。提出了动态阈值化和高效 U-Net 架构，以提高扩散模型的训练效果和效率。