# 0x0. 前言
这篇论文对应的链接为：https://openreview.net/pdf?id=tuzTN0eIO5 ，最近被ICLR 2024接收，但不少AI Infra的同行已经发现了这个工作的价值，并且已经开源在 https://github.com/sail-sg/zero-bubble-pipeline-parallelism ，在一些AI Infra相关的地方也存在一些讨论和介绍。比如 https://www.zhihu.com/question/637480969/answer/3354692418 

![](https://img-blog.csdnimg.cn/direct/9aef60faad6f45cf90d3bbf1d670c5eb.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/59d4c87955f24958a4fb52eaf1d6dfa0.png)

所以来解读下这篇论文，此外作者的代码也可以很方便的在Megatron-LM中嵌入，总的来说是一个非常实用的Infra工作。后面论文解读完毕之后也会对ZB-H1代码实现进行解析。

# 0x1. 番外
这里简单对Megatron-LM提供的Pipline并行调度模式做一个理论讲解，这是读懂这篇文章的基础，由于整体的代码实现还是偏向复杂，所以这里偏向于理论讲解。

Pipline并行两篇比较关键的paper应该是GPipe和PipeDream，后面Meagtron在他们的基础上还做了工程优化比如vpp来减少bubble，我们只需要看懂 https://arxiv.org/pdf/2104.04473.pdf 里面的这两张图就可以了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5138a08bac51445886e7c1f85a3cc45e.png)

翻译下Figure3的描述：GPipe流水线schedule，所有micro-batch（以数字表示）均为前向传播（蓝色），然后为后向传播（绿色）。灰色区域表示流水线气泡。为简单起见，我们假设前向传播的时间是后向传播的两倍。流水线计划的schedule不取决于此时间因素。本例中的每个batch由8个micro-batch组成，每个蓝色或绿色框中的数字是给相应micro-batch的唯一标识符（比如，第一个batch由1− 8个micro-batch组成，第二个batch由micro-batch 9− 16组成等）。优化器在流水线刷新时进行步进（step）并更新权重参数，以确保严格的优化器语义。

> 这里说的严格的 Optimizer 语义是指，一个 batch 内的所有 micro-batch 的数据遇到的模型都是同一个版本的模型。为了达到这个效果，Megatron-LM 在一个 batch 的结尾引入了流水线 flush，即做一次跨设备的同步，然后再用 Optimizer 更新参数。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/15a1af7cd0424dd39ecff53fe2ca4673.png)翻译下Figure4的描述：默认和交错的1F1B Pipline Schedule。Figure4的上半部分显示了默认的非交错1F1B Schedule。Figure4的下半部分显示了交错的1F1B Schedule，其中每个设备被分配了多个chunks（在这个例子中是2个）。深色显示第一个chunk，浅色显示第二个chunk。Pipline并行的气泡的大小更小（在交错的1F1B Schedule时间线中，pipline flush更早发生）。这里交错的1F1B Schedule就是Meagtron-LM里面的VPP优化。


接着，我们首先来看上面的Figure3，为了看得更清楚把图放大：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/22d1ad8a6b3e4eb0a16d43b5f6124155.png)

从这里可以看到GPipe的schedule是首先执行一个batch中所有micro-batch的前向传播，然后执行所有micro-batch的反向传播，这里假设micro-batch的大小为m（这里就是8），流水线的深度为d（这里就是4），一个 micro batch 的整个前向、整个后向的执行时间分别为 $t_{f}$ 和 $t_{b}$. 则上图中在前向存在 $p-1$ 个 $t_{f}$ 的气泡，在后向存在 $p-1$ 个 $t_{b}$ 的 气泡，所以一个迭代的气泡 $t_{pb}$:

$t_{pb}=(p-1)*(t_f+t_b)$

注意，这里的$t_f$和$t_b$的关系是假设$t_b$是$t_f$的两倍，也就是反向的计算时间是前向的两倍，所以$(p-1)*(t_f+t_b)=3\times 3\times t_f = 9 \times t_f$，也就是说每个设备上有9个格子的气泡，我们可以数一下，确实如此。例如第一个stage上的红色框部分就是9个气泡。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8f66037d78864d6fa9cde3da01a5a193.png)

而一个迭代理想的处理时间，即没有气泡的处理时间 $t_{id}$:

$t_{id} = m\cdot(t_{f}+t_{b})$

这样气泡占比 Bubble time fraction:

$\frac{t_{pb}}{t_{id}} = \frac{(p-1)}{m}$

这样为了降低气泡量，就需要 $m \gg p$. 但是每个 micro bath 前向的 Activation 都需要暂存直到依赖它的后向计算完成，这样 micro batch 数量过多，会导致显存占用增加过多。后面Pipline-Flush（https://arxiv.org/abs/2006.09503）论文里提供了一种改进策略：1F1B。核心部分就是下图：


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5bd5d4060524458e88d3de5c8a3ead11.png)

解释一下这个图。对于GPipe来说流水线中最长驻留了 $m$ 个未完成的 micro batch（上半部分图）. 而 1F1B 则限制其最多驻留流水线深度 $p$ 个未完成的 micro batch，如此形成了上图中的下半部分的流水线。这个流水线的特点是一个迭代的时间没有变化，但是  $p \ll m$ ，所以驻留的未完成的 micro batch极大减少，减少了显存峰值。（重点是减少了显存的峰值，但是气泡还是不变）

由于1F1B没有减少气泡大小，只是降低了显存占用峰值，所以后续Megatron-LM里在1F1B的基础上做了Interleaved 1F1B的优化，减少了流水线气泡，也就是VPP。

VPP的idea是于让 micro batch （micro batch size 更小）更多来减少气泡。方法是让一个 device 虚拟成 $v$ 个 device，从计算 1个连续的 layer 段（有 $x$ 个 layer）变成计算 $v$ 个不连续的 layer 段（每段 layer 数量为 $x/v$）。比如之前 1F1B 时 device 1 负责 layer 1~4，device 2 负责 5~8，在 Interleaved 1F1B 下 device 1 负责 layer 1~2 和 9~10，device 2 负责 3~4 和 11~12，这样可以让流水线中每个 stage 更小，因而下个 stage 的等待时间更短，气泡更小。需要注意的是， $m$ 需要是 $p$ 的整数倍。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1dfd3aa067a947b88a3622f2630566a7.png)


此时完成 $v$ 个 layer 段中一个的前向、后向时间分别为$t_{f}/v$ 和 $t_{b}/v$, 流水线的气泡 $t_{pb}^{int.}$:

$t_{pb}^{int.}=\frac{(p-1)\cdot(t_f + t_b)}{v}$

然后可以计算出气泡占比：

$\frac{t_{pb}^{int.}}{t_{id}} = \frac{1}{v}\cdot\frac{(p-1)}{m}$

可以看到，相比于1FB现在的气泡占比就减少到了 $1/v$。但是流水线之间的通信量也增加了 $v$ 倍。对于一个 pipeline stage，里面包括多个 Transformer layer，一个 microbatch 在流水并行的多个 pipeline stage 间的通信量是 $2bsh$ （考虑前向、后向各一次），采用 point-to-point 通信，且和  Transformer layer 数量无关 。所以现在相当于流水线的stage增加了，通信量也会增加。特别是当global的batch越来越大的时候，这个通信开销就会更显著。

所以，VPP似乎也不是一个鱼和熊掌兼得的方案，但我感觉这篇文章要解读的Paper在一定程度上是一个更完美的方案，相信完全有取代vpp的潜力。

这一节只是说Pipline的几个有影响力工作的核心idea，实际上在工程实现还有很多技巧例如通信加速之类的，之后有机会我们再分析。
# 0x2. 摘要

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/36d5776d247f4849942447df2798bab9.png)

大概就是说这个paper提出了一个新的流水线调度算法，实现了流水线并行同步训练时的的零气泡。我理解这里的同步训练指的就是1F1B中一个 batch 内的所有 micro-batch 的数据遇到的模型都是同一个版本的模型。然后这个改进基于一个关键的观察就是反向计算可以分成两部分，一部分计算输入的梯度，另一部分计算参数的梯度。此外paper还提到，他们开发了一个算法可以根据特定模型配置和内存限制自动找到最佳调度。另外，为了实现真正的零气泡，作者引入了一种新技术来绕过优化器步骤中的同步。实验评估结果显示，这种调度算法在类似的内存限制下，吞吐量比1F1B调度高出至多15%。当内存限制放宽时，这个数字可以进一步提高到30%。


# 0x3. 介绍
第，1，2，3段可以不看，就是番外介绍到的知识。从第4段开始看下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6bf1694fd7e1446786adf6d23ceaf8b8.png)

重点就是说目前1F1B仍然存在流水线气泡的问题，然后作者发现过以更细的粒度表示和调度计算图，可以进一步优化Pipline并行中的气泡。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/70ae002d9c46430082eb502654de85e6.png)

一般神经网络为组织为一些堆叠的层。每一层有前向传播和反向传播。前向传播中，输入$x$通过$f(x, W)$映射到输出$y$。反向传播对于训练至关重要，涉及两个计算：$\nabla_x f(x, W)^\top \frac{d\ell}{dy}$ 和 $\nabla_W f(x, W)^\top \frac{d\ell}{dy}$ ，它们计算相对于输入$x$和层的参数$W$的梯度。为方便起见，我们使用单个字母$B$和$W$分别表示这两个计算，以及使用$F$表示前向传播，如Figure1所示。

传统上，$B$和$W$被分组并作为单一的后向函数提供。这种设计在概念上对用户友好，并且对于数据并行（DP）来说恰好工作良好，因为在第$i$层的权重梯度的通信可以与第$i-1$层的后向计算重叠。然而，在流水线并行（PP）中，这种设计不必要地增加了顺序依赖的计算，即第$i-1$层的$B$依赖于第$i$层的$W$，这通常对流水线并行的效率不利。基于分割的$B$和$W$，paper提出了新的Pipline调度算法，大大提高了Pipline并行的效率。

paper其它部分的行文方式如下：在第2节中，介绍了基于$F$、$B$和$W$的执行时间相同的理想假设下的手工调度。随后，在第3节中，我们取消了这个假设，并提出了一个在更现实条件下工作的自动调度算法。为了实现零气泡，第4节详细介绍了一种方法，该方法在优化器步骤中绕过了同步的需要，但保留了同步训练语义。通过在不同设置下将paper的方法与基线方法进行实证评估来结束本文。

需要注意，作者的目标不是探索大规模分布式训练的一般混合策略。相反，作者特别致力于提高Pipline并行调度的效率，并通过与基线的比较来证实。作者的方法与数据并行（$DP$）、张量并行（$TP$）和$ZeRO$策略是正交的，它可以作为大规模训练中$PP$部分的并行替代品。


# 0x4. 手工编排的Pipline Schedule
基于将B和W分开可以减少顺序依赖并因此提高效率的关键观察，我们从常用的1F1B调度开始重新设计Pipline并行。如Figure 2所示，1F1B以一个预热阶段开始。在这个阶段，每个workers（GPU）执行不同数量的前向传播，每个stage通常比它后面的stage多执行一次前向传播。预热阶段之后，每个workers过渡到一个稳定状态，在这个状态中，他们交替执行一次前向传播和一次后向传播，确保在各个阶段之间均匀分配工作负载。在最后阶段，每个worker处理未完成的micro batch的后向传播，完成这个batch。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a9e28996362840c0bf5c432dad4b8ab3.png)

在paper改进的版本中，**将反向传播分为B和W两个阶段，仍然必须确保同一个micro batch中的F和B在Pipline stages之间保持顺序依赖。然而，相同阶段的W可以在对应的B之后的任何地方灵活地安排。这允许策略性地放置W来填充Pipline的气泡。** 之前也有一些方法改进了1F1B的调度，以不同的方式在气泡大小和显存占用之间进行权衡。在本节中，paper介绍了两个有趣的手工Pipline调度，以展示更细粒度在减少Pipline气泡方面的巨大潜力（见Figure 3）。为了在让初步设计更易懂，我们假设F、B和W的时间成本是相同的，这一假设也被早期的研究（Narayanan et al., 2021; Huang et al., 2019）所共享。然而，在第paper的第3节中，我们重新评估了这一假设，以在现实场景中优化Pipline调度的效率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5ad8cfaeea664620ba3fff6d1e99f1cd.png)

## 0x4.1 内存高效的Schedule

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/019b013088ca4c6ab4fd7970bb5e3e26.png)

Paper提出的第一个手工调度，名为ZB-H1，确保所有worker的最大峰值内存使用量不超过1F1B的使用量。ZB-H1通常遵循1F1B的调度，但它根据预热micro batch的数量调整W的开始点。这确保所有worker维持相同数量的in-fight micro-batch。因此，如Figure 3（顶部）所示，**气泡大小减少到1F1B大小的三分之一**。这种减少是因为与1F1B相比，所有worker更早地启动B，并且尾端的气泡被较晚启动的W阶段填补。由于W通常比B使用更少的内存（见下面的Table 1），另外第一个worker有最大的峰值内存使用量，这与1F1B一致。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/07e9e1ad3236466195ef0cf256be3e78.png)

## 0x4.2 零气泡Schedule

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7e5bf4bc8f024e5da4e8435cc236e49f.png)

paper指出当允许比1F1B更大的内存占用，并且有足够数量的micro batch时，可以实现一个零气泡调度，我们将其标记为ZB-H2。如上面Figure 3（底部）所示，我们在预热阶段引入更多的F来填补第一个B之前的气泡。我们还在尾部重新排序W，这将Pipline的布局从梯形变为平行四边形，消除了Pipline中的所有气泡。还要强调的是，在这里，优化器步骤之间的同步被移除了，paper的第4节讨论如何安全地完成这一点。

## 0x4.3 量化分析

先做了一些约定，使用$p$来表示流水线stage的数量，用$b$来表示每个micro batch的大小。对于Tranformer架构，用$a$表示注意力头的数量，用$s$表示序列长度，用$h$表示隐藏维度的大小。使用记号$M_B$/$M_W$来表示存储一个$B$/$W$激活所需的内存，以及$T_F/T_B/T_W$来表示一个$F/B/W$的运行时间。为了简单起见，我们仅对Transformer架构进行定量分析，使用类似于GPT-3的经典设置，其中前馈内部的隐藏维度大小为$4h$，每个注意力头的维度大小为$h/a$。

正如Narayanan等人（2021年）所述，paper在计算FLOPs时只考虑矩阵乘法操作，因为它们在Transformer层中贡献了大部分的计算。在前向传播中的每一个矩阵乘法操作，对应的反向传播中有两个具有相同FLOPs的矩阵乘法操作（见上面的Figure 1的B和W）。计算Transformer层FLOPs的近似公式在表1中。我们可以看到$T_W < T_F < T_B$ 且 $T_B + T_W = 2T_F$。paper使用Korthikanti等人（2023年）的方法来估计$B$所需的激活内存。$B$完成后，它释放了一些不再使用的激活，但为$W$保留了一些额外的梯度（Figure1中的$\nabla_z L$）。Table 1中，$W$所需的总内存小于$B$。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/07e9e1ad3236466195ef0cf256be3e78.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/a056f4fafbc042b99c1d6bb4bee3a006.png)

在不预设$T_F = T_B = T_W$的前提下，表2中对ZB-H1和ZB-H2的最大激活内存及流水线中的气泡大小进行了量化分析。特别地，对于ZB-H1而言，第$i$个工作节点的激活内存计算公式为$(p - i + 1)M_B + (i - 1)M_W$；对于ZB-H2，其激活内存的计算公式则为$(2p - 2i + 1)M_B + (2i - 2)M_W$。根据表1的数据，我们知道$W$阶段所需的激活内存小于$B$阶段所需的。因此，对于ZB-H1和ZB-H2，最大激活内存分别是$pM_B$和$(2p - 1)M_B$。

# 0x5. 自动Pipline Schedule

虽然手工调度更直观和易于理解，但在实际应用中它们面临着几个问题。首先，基于$T_F = T_B = T_W$进行调度会引入不希望的气泡，特别是对于这些值相差显著的模型。此外，传统手工调度常常忽略了在stage之间传输activation/gradient所需的通信时间（表示为$T_{comm}$），这导致了流水线流中的明显延迟。最后，当可用内存不足以容纳足够多的micro batch以实现无气泡调度时，在**最小化气泡大小和遵守内存限制之间寻找平衡**变得特别具有挑战性。

为了应对这些挑战并确保在实际场景中的泛化，paper提出了给定Pipline Stage数$p$、micro batch数$m$、激活内存限制$M_{limit}$，以及运行时间估计$T_F、T_B、T_W$和$T_{comm}$下自动搜索最优调度的算法。作者设计了一种启发式策略，当$m$足够大时，总是能生成最优或接近最优的解决方案。作者还将问题系统地形式化为整数线性规划（更多细节见附录G），当问题规模在一定范围内时，可以通过现成的ILP求解器（Forrest & Lougee-Heimer, 2005）来解决。这两种方法可以结合使用：首先，使用启发式解作为初始化，然后用ILP进一步优化。

这个搜索算法的步骤如下：
- 在预热阶段，我们在内存限制内尽可能地安排多个$F$，以最小化第一个$B$之前的气泡。如果没有达到内存限制，生成的调度可能仍然会在第一个$B$之前有一个小气泡（小于$T_F$），此时安排另一个$F$可能会延迟后续的$B$。我们使用一个二进制超参数来控制是否进行此操作。
- 在预热阶段之后，我们遵循1F1B模式，即交替安排一个F和一个B。当出现大于$T_W$的气泡时，我们插入$W$来填补气泡。当出现气泡但大小小于$T_W$时，如果当前气泡使所有stage中累积的最大气泡大小变大，我们仍然插入一个$W$。当达到内存限制时，我们也插入$W$来回收一些内存。通常，我们的启发式策略进入一个遵循$1F-1B-1W$模式的稳定状态。
- 在整个过程中，Pipline Stage $i$总是保证在$F$用完之前，比阶段$i+1$多安排至少一个$F$。当这种差异超过一个时，我们使用另一个二进制超参数来决定是否在Pipline Stage $i$跳过一个$F$，如果这不会导致更多的气泡。我们执行网格搜索来找到最佳的超参数组合。
- 在每个stage，当$F$和$B$用完时，我们依次安排所有剩余的$W$。

# 0x6. 绕过优化器同步

在大多数流水线并行实践中，出于数值稳定性的考虑，通常会在optimizer步骤中对Pipline Stage进行同步。例如，需要计算全局梯度范数以进行梯度范数裁剪（Pascanu等人，2013年）；在混合精度设置中执行对NAN和INF值的全局检查（Micikevicius等人，2017年）；这两者都需要跨所有阶段的全规约（all-reduce）通信。然而，在优化器步骤中进行同步会破坏平行四边形（图3）并使零气泡变得不可能。在本节中，我们提出一种替代机制来绕过这些同步，同时仍然保持同步优化语义。

在现有的实现中，首先启动all-reduce通信来收集全局状态，然后执行基于全局状态的optimizer步骤。然而，我们注意到大多数时候全局状态没有影响，例如，对NAN和INF的全局检查很少触发，因为在一个稳健的设置中大多数迭代不应该有数值问题；从经验上看，梯度裁剪率也相当低，不足以证明每次迭代都同步全局梯度范数的必要性。

基于这些观察，paper提出用post update validation来代替事先的同步。这个想法在Figure 4中说明，在优化器步骤之前的每个Stage，从前一个stage接收到部分reduced的全局状态，与当前stage的局部状态结合，然后传递给下一个stage。每个satge的优化器步骤由部分reduced的状态控制，例如，当发现NAN或部分减少的梯度范数超过裁剪阈值时，跳过更新。在下一次迭代的预热阶段，全量reduced的全局状态从最后一个阶段传回到第一个阶段。在接收到全局状态后，每个阶段执行验证以决定之前的优化器步骤是否合法。如果需要对梯度进行修正，将发出回滚（更多细节见附录C），然后会根据全量reduced的全局状态重新执行优化器步骤。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/d725afeda51e43d3b231b21c63bae05d.png)

这里的目的就是避免在一个时间点执行全局allreduce同步，而是把nan/inf以及clip grad的操作隐藏在流水线中。

# 0x7. 实验
## 0x7.1 实验设置
paper的实现基于开源的Megatron-LM项目（Narayanan等人，2021年），并使用与GPT-3（Brown等人，2020年）类似的模型来评估其性能，如Table 3详细展示的那样。在paper的实验中，首先进行了特定数量的迭代用于分析，收集了$T_F、T_B、T_W$和$T_{comm}$的经验测量值。获取这些值后，将它们输入到paper设计的的自动Pipline调度算法中，以确定最佳调度。值得注意的是，相较于中间stage，初始和最终的Pipline阶段少一个Transformer层。这种设计是为了补偿初始和最终阶段中额外的Embedding和损失计算，以便它们不会成为瓶颈并导致其它stage出现气泡。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f6a32e58bb524755818f23ea2c19bb10.png)对比的方法：

- ZB-1p：自动搜索的schedule，其激活内存限制为$pM_B$，理论上与1F1B具有相同的峰值内存。
- ZB-2p：自动搜索的schedule，其激活内存限制为$2pM_B$，这是实证上接近零气泡所需的最小内存量（见Figure 7）。
- 1F1B和1F1B-I：由Harlap等人（2018）和Narayanan等人（2021）引入的1F1B和交错1F1B方法（VPP），其实现来自Megatron-LM。对于交错1F1B，整个模型被划分为一系列chunk，这些chunk被每个阶段循环取用，形成一个交错的管道。在我们的交错实验中，我们总是使用最大数量的chunk来确保最小的气泡，即每个Transformer层作为一个chunk。

我们的实验使用了多达32张NVIDIA A100 SXM 80G GPU，这些GPU分布在4个节点上，通过RoCE RDMA网络互联。在几次预热迭代后，记录了每次迭代的运行时间。得益于Megatron-LM实现提供的可复现性，我们可以在不将模型运行到收敛的情况下，验证ZB-1p和ZB-2p的正确性。我们使用固定的随机种子来初始化模型，记录ZB-1p、ZB-2p和1F1B每次迭代后的损失，然后验证它们是否是逐位相同的。
## 0x7.2 主要结果

paper在Figure 5中展示了所有方法的吞吐量，并在Table 4中留下了每种设置的额外细节。paper的实验表明，ZB-2p在各种设置下始终优于所有其他方法。值得注意的是，1F1B、1F1B-I和ZB-1p的吞吐量与micro batch数呈强正相关。相比之下，ZB-2p即使在micro batch较少的情况下也保持了效率。这是因为ZB-2p中的气泡率几乎已经达到零，其吞吐量已经接近上界。这里的上界大致通过将1F1B的吞吐量与$\frac{1}{1−1F1B 的气泡率}$相乘来估计（更多细节见第5.3节）。如前所述，ZB-2p相较于1F1B基线提高效率的代价是更高的内存消耗。paper还在附录F中将ZB-2p与相同内存消耗下的1F1B进行了比较，实验结果也显示，与1F1B Baseline相比，即使微micro batch大小减半，ZB-2p也实现了更高的吞吐量。

相比之下，ZB-1p旨在拥有与1F1B基线相同的峰值内存成本。在8个GPU的设置中，它展示了与1F1B-I相当的吞吐量。在多节点设置中，通信带宽更多成为瓶颈时，ZB-1p明显优于1F1B-I，突出了其在减少流水线气泡同时不增加额外通信成本方面的优势。在paper的大多数设置中，我们将micro batch数$m$设置为大于流水线stage数$p$，因为它们是管道并行更常见的使用案例。然而，paper也在附录H中进行的实验列出了$m ≤ p$的情况，显示出在相似的内存消耗下有20%到30%的提升。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0d20cec473d94f72ba5585e248e6e90a.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/dd93cb3c67794848a9cb7636cc3c7dc1.png)

## 0x7.3 自动调度的效率
我们研究了从我们的自动调度算法生成的调度的效率。使用与上面的主要实验相同的设置，但由于这里的目的是研究自动调度算法的效率，这里的数字基于理论计算而非真实实验。为了量化Pipline并行调度的效率，这里引入了气泡率的概念，其计算为$(cost − m(T_F + T_B + T_W ))/cost$。这里的cost定义为所有阶段中最大的执行时间，使用经过profile得到的$T_F、T_B、T_W$和$T_{comm}$值为每个调度计算。$m(T_F + T_B + T_W)$是当所有通信与计算重叠，因此Pipline中无气泡时的最优执行时间。

不同调度的气泡率在下面Table 5中呈现。paper将手工调度的ZB-H1和ZB-H2作为自动搜索调度的基准进行比较。在大多数设置中，ZB-2p产生的气泡率少于1%，这是所有调度中最好的。相比之下，ZB-H2的表现一致地不如ZB-2p。这提供了一个强有力的证据，表明我们的自动调度算法通过使用更准确的$T_F、T_B、T_W$和$T_{comm}$估计，更好地适应现实场景。值得注意的是，我们所有的方法都显著优于1F1B。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/354bbd5d02104967b092b0c3d164d66d.png)

paper还绘制了ZB-2p及其在16个GPU上的实际执行的profile结果，以提供直观的证据，证明它确实是一个零气泡的调度。如下面的Figure 6所示，自动生成的ZB-2p调度几乎没有气泡。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0bb2aebb22ba4f2b935808332423e8c6.png)


## 0x7.4 内存限制

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/543ccd5686d6411392724bd8aa93fd4e.png)

为了更好地理解内存限制的效果，paper研究了气泡率与$M_{limit}$之间的关系。paper使用一系列的$M_{limit}$运行启发式搜索算法，并在Figure 7中绘制它们。最初，随着增加$M_{limit}$的值，气泡率显示出近似线性的下降趋势。理论上，曲线应该在$\frac{(p-1)(T_B+2T_{comm})+pT_F}{T_F}M_B$附近趋于平稳。根据经验，发现当$T_F \approx T_B$且$T_{comm}$相对较小时，$2pM_B$是实现接近零气泡率的一个好阈值。超过拐点后，尽管充足的内存限制确实理论上会导致零气泡率，但通常成本大于收益。更多细节见附录B。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7d101bd56c804e8c98b641fd6cc8078b.png)

$\frac{(p-1)(T_B+2T_{comm})+pT_F}{T_F}M_B$ 这个公式怎么来的？在附录B里面有详细介绍。



# 0x8. paper的结论部分
paper通过在后向计算中分离激活梯度和参数梯度，引入了一种改进Pipline并行效率的新策略，并设计了一个能够在不同内存预算下最小化Pipline气泡率的自动Pipline调度算法。这个算法产生的调度一致地优于1F1B，并且甚至能够实现接近零的气泡率。根据经验，实现零气泡大约需要与1F1B相比两倍的激活内存，这引发了关于内存溢出问题的担忧。根据附录F，paper认为在大型模型训练中用一些内存交换零气泡Pipline调度是值得的。像ZeRO、张量并行这样的策略可以用来满足增加的内存需求。零气泡调度的另一个优势是它可以在较少数量的micro-batch（通常$3\times p$就足够了，$p$表示流水线stage数量）下达到最优效率，这意味着更多的micro-batch可以在数据并行维度上进行划分。这为大型模型的训练带来了更好的可扩展性。


# 0x9. 附录
附录里面也是有一些东西，这里也做一个翻译解读。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/681a78b251d44a4c94f358c21a209e74.png)

这里说的是当考虑到数据并行时，会在optimizer步骤之前启动一个all-reduce通信来收集梯度。通常，这种通信与计算的重叠程度较低，特别是当通信带宽有限时，会导致延迟。如Figure 3所示，通常在迭代的末尾会安排多个W。对于每个W，它由几个独立的计算组成，这些计算为不同的参数计算梯度。如Figure 8所示，我们可以重新排序这些计算，以聚焦那些为相同参数计算梯度的计算，从而实现计算与通信之间的最优重叠。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f14b5b27306a4ade9c1fd1f5b12e5867.png)

这里是对自动调度算法的理论峰值内存进行计算。

初始 stage 中第一个$B$之前的气泡对内存限制与气泡率之间的关系影响很大。对于第一个micro batch，前向需要从初始stage经过到最终stage，而后向则反转这一过程，直到它最终回到初始stage。第一个micro batch从开始到完成的总时间至少需要$p(T_F + T_B) + 2(p - 1)T_{comm}$，并且由于依赖链的存在，这个时间无法被压缩。我们将F的数量表示为$k(≥ 1)$，在初始阶段中第一个B之前的气泡大小表示为$\beta(≥ 0)$。那么我们有：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c389556079554055950983fff9e139ef.png)

$M_{limit}$的下界与$k$成正比（见公式1），而$β$与$k$成反比（见公式2）。当增加$k$并保持$k < \lfloor \frac{(p-1)(T_B + 2T_{comm}) + pT_F}{T_F} \rfloor$时，$β$线性减少，同时$M_{limit}$的下界线性增加。当$k = \lfloor \frac{(p-1)(T_B + 2T_{comm}) + pT_F}{T_F} \rfloor$时，$β$达到其最小值而不延迟$B$，其值小于$T_F$，峰值激活内存至少为$\lfloor \frac{(p-1)(T_B + 2T_{comm}) + pT_F}{T_F} \rfloor M_B$。超过这个点，进一步将Pipline气泡减少到零并不容易。这是因为每个stage都有一个小于$T_F$的小气泡（见Figure 6），并且安排另一个F会延迟B的开始时间，因此对前面阶段中的F提出更多要求。理论上，为了完全消除所有阶段第一个B之前的气泡，初始阶段需要额外的$p - 1$个F（见图9），这也意味着总激活内存使用至少为$\lfloor \frac{(p-1)(T_B + 2T_{comm}) + (2p-1)T_F}{T_F} \rfloor M_B$。


后面还有几个附录介绍了Optimizer回滚，时间线Profile以及一些更细微的实验结果，大家感兴趣可以自行查看。

我个人感觉ZB-H2略微还是有些激进，所以我下面会解读下ZB-H1的代码实现。

# 0x10. ZB-H1在Megatron-LM中的代码实现
虽然paper看起来非常复杂，但作者在代码仓里提供了一个ZB-H1的快速简洁实现，非常容易理解paper的思想。在：https://github.com/sail-sg/zero-bubble-pipeline-parallelism/commit/95212f7000dca3d03dc518759020355cfdae231f 这里。下面对代码进行解析，注意paper的代码是构建在NVIDIA Meagtron-LM基础上的：

首先创建了一个：`megatron/core/weight_grad_store.py` 。


```python
import queue
from megatron import get_args

# 这段代码定义了一个名为WeightGradStore的类，用来管理和优化分布式训练中的权重梯度计算。
class WeightGradStore:
		
		# cache 用于暂存权重梯度计算过程中的参数，weight_grad_queue是一个队列
		# ，用于存储所有待执行的权重梯度计算任务，split_bw是一个布尔值，指示是否分割后向传播。
    cache = []
    weight_grad_queue = queue.Queue()
    split_bw = True
		
		# 这个类方法检查当前的训练配置是否支持权重梯度存储和优化机制。
		# 这包括检查PP并行大小、VPP大小、overlap_grad_reduce、
		# 是否为transformer_engine实现和sequence_parallel等条件。
		# 如果配置不满足特定条件，则不支持此优化，会回退到原始调度。
    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        args = get_args()
        if args.pipeline_model_parallel_size <= 1:
            return False
        if args.virtual_pipeline_model_parallel_size is not None:
            return False
        if args.overlap_grad_reduce:
            # the logic of overlapping grad reduce should be changed
            return False
        if args.transformer_impl == 'transformer_engine':
            # hard to capture weight gradient computation for transformer_engine
            return False
        if args.sequence_parallel:
            # not supported in this commit
            return False
        return True

    # 该方法负责将权重梯度计算的输入参数暂存到cache中。
    # 如果不分割后向传播或当前配置不支持优化，它会立即执行权重梯度计算函数func。
    @classmethod
    def put(cls, total_input, grad_output, weight, func):
        if not cls.split_bw or not cls.is_supported():
            func(total_input, grad_output, weight.main_grad)
            return
        # Store the weight gradient computation of linear layers.
        cls.cache.append((total_input, grad_output, weight, func))

    # 在后向传播的某个阶段，flush方法被调用来将cache中暂存的所有权重梯度
    # 计算任务放入weight_grad_queue队列，并清空cache。
    @classmethod
    def flush(cls):
        if not cls.is_supported():
            return
        # Collect all stored computations during backward as a W.
        cls.weight_grad_queue.put(cls.cache)
        cls.cache = []

    # 执行weight_grad_queue队列中的一个权重梯度计算任务。这包括取出一个存储的梯度计算任务，并执行它。
    @classmethod
    def pop(cls):
        if not cls.is_supported():
            return
        # Execute a single W.
        assert cls.weight_grad_queue.qsize() > 0
        stored_grads = cls.weight_grad_queue.get()
        for total_input, grad_output, weight, func in stored_grads:
            func(total_input, grad_output, weight.main_grad)

    # 执行weight_grad_queue队列中剩余的所有权重梯度计算任务，直到队列为空。
    @classmethod
    def pop_all(cls):
        # Execute all remaining W.
        remaining_qsize = cls.weight_grad_queue.qsize()
        for _ in range(remaining_qsize):
            cls.pop()
```

这个数据结构其实就是paper里面的解耦Linear后向中的W和B pass，接下来在`megatron/core/tensor_parallel/layers.py`中对weight grad进行梯度累积的地方进行修改，把这个操作放到WeightGradStore里面cache起来，不是立即计算。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2d048961c79d497fa2f70c2024b364e0.png)

最后，在Megatron-LM的pipline模块的schedules.py中进行如下修改：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/48025f04769e4cf386204c9ea86354ab.png)

注意其中的这段代码是在1F1B阶段：

```python
# Run 1F1B in steady state.
for i in range(num_microbatches_remaining):
	# For BWF pattern or in rank 0, we don't split W and B for reasons below.
	#   1. to leverage batched p2p op (send_backward_recv_forward)
	#   2. to overlap grad all-reduce for tensor parallel
	#   3. to avoid redoing grad all-gather for sequence parallel
	# Note that the order of grad accumulation is changed by this behavior,
	# thus causing a minor precision error compared to 1F1B even it's mathematically correct.
	WeightGradStore.split_bw = (i < rank or last_iteration) and rank > 0
	input_tensor_grad = backward_step(
	    input_tensor, output_tensor, output_tensor_grad, model_type, config
	)
	if WeightGradStore.split_bw:
	    WeightGradStore.flush()
	
	if last_iteration:
         input_tensor = None
         send_backward(input_tensor_grad, recv_tensor_shapes, config)
         if i >= rank > 0:  # delay W by rank
             WeightGradStore.pop()  # W
     else:
         input_tensor = send_backward_recv_forward(
             input_tensor_grad, recv_tensor_shapes, config
```
这里的意思是如果当前处于BWF的模式或者在rank0或者当前的迭代处于最后一个micro batch，就不分割B和W，这里的BWF模式应该就是表示ZB-H1的1F1B1W稳定状态，这是通过当前micro batch的index和rank的大小关系来判断的，不是很确定这样是不是完全正确，但从ZB-H1的图来看这样判断是没问题的。


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/1f30f829ff7f4f77bec6bde00a179895.png)

比如我们看流水线的第4个stage，只有当micro batch的index为4时才出现了B和W分离调度。`WeightGradStore.flush()`表示当前stage的每一个micro batch执行完之后我们要把所有的W Cache起来，然后`if i >= rank > 0:  # delay W by rank`这个操作发生在最后一个micro batch，这个时候我们就需要更新W了。这里可能那个有个必须了解Meagtron Pipline并行才可以理解的问题，在上面的图里，这里的最后一个micro batch表示的是8吗？不是，是4，为什么？因为在Megatron-LM的pipline里面分成了warmp和1f1b和cooldow阶段，然后这里的最后一个micro batch指的是在1F1B的最后一个micro batch，而总的micro batch是8，warmup的micro batch是4，那么1F1B阶段的micro batch就是4了。

最后在cooldown阶段，代码的修改是这样：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/20fd3efc05734312bbed43f8f9d7c310.png)

这里的1295行对应的就是下面红色的部分：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6196cb6bf4294da09e4fa30d51beaa49.png)

而1292行这个判断对应cooldown阶段的1F1B1W。应该对应这部分：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/18006954040b440e9c6af33257233a62.png)



感兴趣的读者可以关注他们的开源仓库：https://github.com/sail-sg/zero-bubble-pipeline-parallelism
# 0x11. Pipline并行 PlayGround

作者团队提供了一个令人眼前一亮的流水线Schedule PlayGround，访问地址如下：https://huggingface.co/spaces/sail/zero-bubble-pipeline-parallellism

给定流水线并行的一些关键参数之后就可以对流水线Schedule自动作图：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9d6ca16f75564652bcb5f1a1876a4764.png)

个人感觉paper的学术和工程价值都不错，有完全取代VPP的势头，推荐同行阅读。


# 0x12. 参考
- https://strint.notion.site/Megatron-LM-86381cfe51184b9c888be10ee82f3812









