# 一文带你了解NeurIPS2020的模型剪枝研究

本篇文章主要对NeurlPS 2020会议中模型剪枝工作进行简单的介绍和总结，希望能够对想了解模型剪枝最新工作的朋友有所帮助。

### 前置知识

为了大家更好理解nips2020的最新剪枝工作，这里先介绍一下其中涉及的基础知识。如果您对模型剪枝比较熟悉，可以直接跳过本部分。

- 核心思想：剔除模型中“不重要”的权重，使模型减少参数量和计算量，同时尽量保证模型的性能不受影响。

接下来我将从几个角度切入，对不同的剪枝方法进行分类和特点总结，方便读者尽快理解相关概念。

- 结构化剪枝 VS 非结构化剪枝

  结构化剪枝和非结构化剪枝的主要区别在于剪枝权重的粒度。如下图所示，结构化剪枝的粒度较大，主要是在卷积核的channel和Filter维度进行裁剪，而非结构化剪枝主要是对单个权重进行裁剪。两类剪枝方法各有优势。其中，非结构化剪枝能够实现更高的压缩率，同时保持较高的模型性能。然而其稀疏结构对于硬件并不友好，实际加速效果并不明显，而结构化剪枝恰恰相反。
  

![不同类型剪枝的差别](https://img-blog.csdnimg.cn/20201220011322385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZvdXJ0ZWVuX3poYW5n,size_16,color_FFFFFF,t_70)

<center>图1 不同类型剪枝的差别</center>

- 静态剪枝 VS 动态剪枝

  静态剪枝方法是根据整个训练集训练后的结果，评估权重的重要程度，**永久性**裁剪掉重要程度小的权重参数。然而，动态剪枝方法则是**保留**所有的权重参数，根据每次输入的数据不同衡量权重参数的重要程度，将不重要权重参数忽略计算，从而实现动态剪枝。动态剪枝方法能使CNN网络具备更多的表示形式，因此其通常能够有更好性能表现。

- Retraining/finetuning

  当前模型剪枝的经典工作流程“预训练-剪枝-精度恢复”是由韩松在2015年提出来的。虽然目前这个工作范式正在受到一些新的理论/现象挑战，例如：彩票理论等。但其目前仍是比较有效的剪枝工作流。其中，Retraining/finetuning都是精度恢复过程的一种方法，主要是将剪枝后的模型继续在训练集上进行训练，提升剪枝后模型的表现。

- 彩票假说

  [彩票假说](https://arxiv.org/abs/1803.03635)是ICLR2019会议的best paper，其假说主要是**随机初始化的密集神经网络包含一个初始化的子网，当经过隔离训练时，它可以匹配训练后最多相同迭代次数的原始网络的测试精度**。其实彩票假说的提出，是对传统“预训练-剪枝-精度恢复”工作流的挑战。因为之所以在模型预训练之后进行裁剪，是认为预训练模型的权重对于压缩后模型的精度恢复是十分重要的，然而彩票理论却认为通过随机初始化权重的子网络仍可以达到原始网络的精度。关于彩票理论的理解以及进展，以后会进行更深层次的探讨。（挖个坑给自己hhh）

### 正文

- [Pruning Filter in Filter](https://arxiv.org/abs/2009.14410) 	/ [Code](https://github.com/fxmeng/Pruning-Filter-in-Filter) 

  目前剪枝技术可以分成两大类：结构化剪枝(filter,channel维度等等)和非结构化剪枝(单个权重)。这两种方法的主要区别是裁剪权重的粒度。其中，结构化剪枝方法裁剪权重的粒度较大，裁剪后的网络更符合硬件设备的计算特点，但是却无法达到很高的压缩率。非结构化剪枝方法裁剪权重的粒度较小，裁剪后的能够实现更高的压缩率，但是并不适合硬件设备计算。为了结合这两种方法的优势，这篇文章提出了一种 SWP (Stripe-Wise Pruning)的方法，将维度为C * K * K的卷积核分成K * K个 1 * 1 * C的卷积核，以1 * 1 * C为单位对卷积核进行裁剪。其中，引入可学习的参数“Filter skeleton”表示裁剪后卷积核的shape。实验结果表明，SWP剪枝方法比之前Filter剪枝方法更加有效，并且在CIFAR-10和ImageNet数据集上达到SOTA效果。

- [Pruning neural networks without any data by conserving synaptic flow](https://arxiv.org/abs/2006.05467)	/ [Code](https://github.com/ganguli-lab/Synaptic-Flow)

  即彩票假说被提出后，人们进行了大量的“训练-剪枝”循环试验，证实了在网络初始化时确实存在“winning ticket”能够达到原网络相似精度的现象。但是随后人们又提出一些问题：在不对子网络进行训练甚至不研究数据集的前提下，能够在网络权重初始化时直接识别出来“winning ticket”吗？该篇文章对于这个问题的回答是肯定的。之前的基于Gradient的初始化剪枝算法会存在网络层崩塌，过早的剪枝会使网络层变得不可训练等问题。该篇文章提出了Iterative Synaptic Flow Pruning (SynFlow)方法，解决了这些问题。另外，值得一提的是，这种方法并不需要训练数据集，就能够和目前存在的SOTA剪枝效果持平甚至超过。

- [Greedy Optimization Provably Wins the Lottery: Logarithmic Number of Winning Tickets is Enough](https://arxiv.org/abs/2010.15969)	/ [Code](https://github.com/lushleaf/Network-Pruning-Greedy-Forward-Selection)

  神经网络模型往往都有大量的冗余参数，可以被剪枝算法删除和压缩。但是存在一个问题：允许模型性能减少某一程度下，我们到底能够剪枝掉多少参数？该篇文章给出了一个回答。其提出一种基于剪枝算法的贪心优化方法。不同于之前工作对网络的结构和参数量的要求，该工作不对原始网络冗余参数量有较高要求，其算法更能揭示实际使用的网络的情况。

- [HYDRA: Pruning Adversarially Robust Neural Networks](https://proceedings.neurips.cc/paper/2020/file/e3a72c791a69f87b05ea7742e04430ed-Paper.pdf)	/ [Code](https://github.com/inspire-group/hydra)

  在安全要求苛刻和计算资源受限的场景下，深度学习面临缺少对抗攻击的鲁棒性和模型庞大的参数等挑战。绝大多数研究只关注其中一个方面。该篇文章让模型剪枝考虑对抗训练的因素，以及让对抗训练目标指导哪些参数应该进行裁剪。他们将模型剪枝看成一个经验风险最小化问题，并提出一种叫做HYDRA的方法，使得被压缩网络能够同时达到benign和robust精度的SOTA效果。

- [Logarithmic Pruning is All You Need](https://arxiv.org/abs/2006.12156)

  自从彩票假说被提出后，一些更强的假说也被提出来，即每个拥有充分冗余参数的网络，都有一个权重随机初始化的子网络**不需要训练**也能达到和原网络相似准确率。但是这一假说依赖于很多的假设条件。该篇文章移除了绝大多数这些假设，提出唯一假设条件：子网络logarithmic factor系数来约束参数冗余的原网络。

- [Sanity-Checking Pruning Methods: Random Tickets can Win the Jackpot](https://arxiv.org/abs/2009.11094)	/ [Code](https://github.com/JingtongSu/sanity-checking-pruning)

  文章对于之前剪枝工作进行两点总结：（1）剪枝算法是通过从训练数据集中获取信息，来寻找好的子网络结构。（2）好的网络结构对于模型性能是至关重要的。该文章通过设计sanity-checking实验对上面两点结论进行检验。结果发现：（1）剪枝算法在寻找好的网络结构时，很难从训练集获取有效信息。（2）对于被剪枝网络结构，改变每层权重连接位置但是保持总体权重数量和数值不变，模型精度不变。这一结果反驳了上述总结第二点。由此，得出结论只有网络层的连接数或者压缩率才是真正影响模型性能的因素。这些发现促使作者，利用与数据集无关的每一层剪枝比例，在每一层去随机减去模型权重，获得“intial ticket”与上述方法获得“intial ticket”效果类似。

- [Scientific Control for Reliable Neural Network Pruning](https://papers.nips.cc/paper/2020/hash/7bcdf75ad237b8e02e301f4091fb6bc8-Abstract.html)

  该文章致力于通过科学控制，提出一种可靠的神经网络剪枝算法。他们生成与原数据集相同分布的knockoff数据，并且将其与原始数据集进行混合。对网络输入knockoff数据和中间计算生成的knockoff特征图帮助发现冗余的卷积核，进行剪枝。

- [Neuron-level StructuredPruning using Polarization Regularizer](https://papers.nips.cc/paper/2020/file/703957b6dd9e3a7980e040bee50ded65-Paper.pdf)	/ [Code](https://github.com/polarizationpruning/PolarizationPruning)

  该篇文章是2017年[network slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)的改进工作。在[network slimming](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Learning_Efficient_Convolutional_ICCV_2017_paper.html)的工作中，利用L1正则化技术让BN层的scale系数趋近于0，然后裁剪“不重要”的channel。然而，这篇文章认为这种做法存在一定问题。L1正则化技术会让所有的scale系数都趋近于0，更理想的做法应该是只减小“不重要”channel的scale系数，保持其他系数的仍处于较大状态。为了实现这一想法，该篇文章提出了polarization正则化技术，使scale系数两极化。
  
  ![L1正则化和Polarization正则化下scale factor的对比](https://img-blog.csdnimg.cn/20201220011646419.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2ZvdXJ0ZWVuX3poYW5n,size_16,color_FFFFFF,t_70#pic_center)
  
  <center>图2 L1正则化和Polarization正则化下scale factor的对比</center>
  
- 如图所示，可以看出L1正则化的scale系数分布和polarization正则化的scale系数分布。Polarization正则化技术能够更准确的确定裁剪阈值。该篇文章通过实验证明该剪枝方法在CIFAR和ImageNet数据集上达到SOTA水平。

- [Directional Pruning of Deep Neural Networks](https://arxiv.org/abs/2006.09358)	/ [Code](https://github.com/donlan2710/gRDA-Optimizer/tree/master/directional_pruning)

  利用SGD进行网络优化时，容易使网络损失函数陷入“平坦的山谷”，导致网络优化困难。然而，这一特性可以被当作方向剪枝的理论基础。当网络参数处于“平坦的山谷”中，其梯度趋近于0和二阶梯度存在不少0值。对于这类型参数进行裁剪，其对于training loss的影响微乎其微。因此，本文提出一种新颖的方向剪枝方法 (gRDA)，其不需要retraining和设置先验稀疏度。同时为了减少计算量，其使用tuned l1 proximal gradient算法实现方向剪枝。

- [Storage Efficient and Dynamic Flexible Runtime Channel Pruning via Deep Reinforcement Learning](https://proceedings.neurips.cc/paper/2020/hash/a914ecef9c12ffdb9bede64bb703d877-Abstract.html)	/ [Code](https://github.com/jianda-chen/static_dynamic_rl_pruning)

  该文章利用深度强化学习技术，搜索最优的剪枝方案（即剪多少权重和剪哪些权重），实现自动化剪枝。不同于传统的static pruning，该文章主要依据采用dynamic pruning，根据输入数据的不同，对不同的权重进行剪枝。另外，之前dynamic pruning方法为了实现高灵活性，需要存储所有的权重参数，需要较大的存储空间。本文通过结合static pruning信息和dynamic pruning信息，减少了dynamic pruning的存储空间需求。并且，通过实验证明，在dynamic pruning的方法中，其提出的框架能够有效的平衡动态灵活性和存储空间之间矛盾。

- [Movement Pruning: Adaptive Sparsity by Fine-Tuning](https://arxiv.org/abs/2005.07683)	/ [Code](https://github.com/huggingface/transformers/tree/master/examples/movement-pruning)

  模型剪枝技术在监督学习中十分常见，但是在迁移学习的领域比较少应用。然而迁移学习已经成为自然语言处理领域的“范式”。该篇文章提出了新的剪枝算法 movement pruning，针对BERT等预训练模型进行剪枝。相比关注权重的L1范数，movement pruning剪枝方法更关注训练过程中逐渐远离“0”值的权重。在文章中，作者给出这种方法的数学证明并与主流的剪枝算法进行对比。实验结果显示，movement pruning方法对于大型预训练语言模型的压缩有较大提升。

- [The Generalization-Stability Tradeoff In Neural Network Pruning](https://arxiv.org/abs/1906.03728)	 / [Code](https://github.com/bbartoldson/GeneralizationStabilityTradeoff)

  在网络剪枝的过程中，经常出现去掉一些网络参数后，经过retraining的模型在测试集的表现会更好的情况。为了更好的理解这种现象，该篇文章设计相关实验，认为网络泛化能力的提升来自于剪枝的不确定性。这里的不确定性指的是：相对于原模型精度，剪枝后不经过retraining的模型精度损失。并且提出了“generalization-stability tradeoff”观点。其认为剪枝带来的不确定性越大，剪枝后模型泛化能力增强的越多。该篇文章通过实验支撑了其观点和假设。

## 总结

这些文章看下来，感觉还是比较开心的。因为绝大多数文章都或多或少提到了彩票理论，开始去推动模型剪枝的新范式的发展，思索模型剪枝的边界和真正意义。在我看来，这些工作比单纯的刷榜/冲分更有意义。后面如果有机会，希望还能和大家继续分享我对于模型剪枝的理解，一起交流和学习！


