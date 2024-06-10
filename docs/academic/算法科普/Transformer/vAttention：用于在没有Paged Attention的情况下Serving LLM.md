
# 0x0. 前言（太长不看版）

paper链接：https://arxiv.org/pdf/2405.04437v1

之前浏览 vllm 的时候看到一篇尝试去掉 vLLM 的 PagedAttention，利用 CUDA 底层的虚拟内存和物理内存分配 API 直接分配连续的虚拟内存以及做物理内存映射从而避免 PagedAttention 由于要手动维护 Block Table 和物理内存分配带来的一系列工程麻烦以及高 Overhead 。另外对于新的 Attention 架构，想用上 Paged Attention，不得不从GPU Kernel的角度去适配Paged Attention，非常困难，而使用vAttention实现则无需对底层的GPU Kernel进行任何改动。从 Paper 的结果来看，从 PagedAttention 切换到 Paper 提出的 vAttention 时，无论是在首 Token 延迟，decode吞吐，Overhead 都明显优于 vLLM 框架。最重要的是，它对新 Attention架构的适配会比 vLLM 更加简单，因为不用 Paged kernel。

在 vllm github 仓库 issue 中有人询问什么时候支持 vAttention ，paper 的作者回答会在最近开源 vAttention，同时会提 PR 将其作为 vLLM 管理逻辑内存和物理内存的另外一种选择，期待后续进展。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5c0693d662594ef483bbfe95d8163700.png)


> 注：下面是Paper的详细阅读，然后在每一节也穿插了一些自己的总结，博客里的 page 和 页 都是同一个意思，感兴趣的读者可以继续阅读本文。此外，省掉了Paper里面一些没有干货的章节，主要是还是围绕了vAttention的架构进行阅读，图表比较多，所以看起来有点长。
# 0x1. 摘要

高效利用GPU内存对于高吞吐量的LLM推理至关重要。以前的系统提前为 KV Cache 保留内存，导致由于内部碎片而浪费容量。受操作系统基于虚拟内存系统的启发，vLLM 提出了 PagedAttention，以实现 KV Cache 的动态内存分配。这种方法消除了碎片问题，使得能够在更大批量的情况下高吞吐量地服务 LLM。然而，为了能够动态分配物理内存，**PagedAttention 将 KV Cache 的布局从连续的虚拟内存更改为非连续的虚拟内存** 。这一变化要求重写注意力 kernel 以支持分页，并且服务框架需要实现一个内存管理器。因此，PagedAttention 模型导致了软件复杂性、可移植性问题、冗余和低效率。

在本文中，提出了 vAttention 用于动态 KV Cache 内存管理。**与 PagedAttention 不同，vAttention 在连续虚拟内存中保留 KV Cache，并利用已有系统支持的 low-level 按需分页来实现按需物理内存分配**。因此，vAttention解除了注意力 kernel 开发人员必须显式支持分页的负担，并避免了在服务框架中重新实现内存管理。paper证明了 vAttention 能够无缝动态地管理各种未修改的注意力 kernels 的内存。vAttention 生成 tokens的速度比 vLLM 快达 1.97 倍，而处理输入 prompts 的速度比 PagedAttention 版本的 FlashAttention 和FlashInfer 分别快达3.92倍和1.45倍。

# 0x2. 介绍&背景
> paper介绍和背景酱的东西都是一回事，这里就直接用一节进行描述。

大型语言模型（LLM）被部署在广泛的应用中，例如聊天机器人、搜索引擎和编码助手。因此，优化LLM推理变得非常重要。提高LLM服务吞吐量的关键技术之一是批处理。在LLM推理的两个阶段——prefill 和 decode 中，decode 阶段是内存受限的，因为它每次请求处理一个单独的 token 。批处理通过均摊从 GPU 内存中获取模型权重的成本并提高内存带宽利用率，从而提高了吞吐量。

高效的推理还需要谨慎分配GPU内存。对于每个请求，LLM维护一个称为 KV Cache 的内存状态，并在请求的整个生命周期内每次迭代都重新使用它。在推理过程中实现高内存容量利用率具有挑战性，原因有两个：1）每个请求的KV Cache增长缓慢，即每次迭代一个token（几十毫秒），2）生成的token数量以及请求的KV Cache总大小通常事先未知。

以前的系统如 Orca 和 FasterTransformer 为 KV Cache 分配了一块连续的虚拟内存（由预先分配的物理内存支持）。分配的大小对应于模型支持的最大序列长度，例如32K。由于模型生成的 token 通常远少于最大限制，因此由于内部碎片而浪费了大量GPU内存。因此，这些系统表现出较差的吞吐量，因为它们无法支持较大的批处理大小。

受操作系统基于虚拟内存系统的需求分页启发，vLLM引入了PagedAttention，以减轻与KV Cache相关的内存碎片问题。vLLM并不预先保留KV Cache内存的最大序列长度，而是在需要时按需分配小块的虚拟内存（由物理内存支持），即当先前分配的块已完全使用且模型继续生成更多token时进行分配。**然而，动态分配的块在虚拟内存中不一定是连续的（系统可能已将这些块分配给其它请求）。因此，PagedAttention接受KV Cache分配是非连续的，并实现了一个块表以将这些非连续的分配拼接在一起。**

如今，PagedAttention已经成为LLM服务系统中动态内存分配的事实标准，例如在TensorRT-LLM、HuggingFace TGI、FlashInfer、LightLLM等系统中。PagedAttention方法最显著的方面是将KV Cache存储在非连续的虚拟内存中，以便能够动态分配物理内存。尽管这种方法为KV Cache碎片问题提供了一个合适的解决方案，但paper认为它存在几个**缺陷**（参见Table 1中的实证证据和实际经验）：

**1. 需要重写注意力kernel（GPU代码）**。已知可以使用基于索引的查找简单而高效地访问虚拟连续对象的元素。而PagedAttention的 KV Cache 存储在非连续的虚拟内存中，所以 PagedAttention 要求重写 GPU 代码，以便注意力 kernel 能够取消引用 KV Cache 的所有元素。重写代码的需求是将新 Attention 优化用于生产环境的主要障碍。

**2. 增加了软件复杂性和冗余（CPU代码）**。PagedAttention 还迫使开发者在服务框架内实现一个内存管理器，使其负责（取消）分配 KV Cache 和跟踪动态分配的 KV Cache 块的位置。这种方法实际上相当于在用户代码中重新实现了分页需求——这是操作系统的功能。

**3. 引入性能开销**。PagedAttention 可能会通过两种方式在执行的关键路径中增加运行时开销。首先，它要求 GPU kernel 执行与从非连续内存块中获取 KV Cache 相关的额外代码。paper 发现，这在许多情况下会使注意力计算速度减慢超过 10%。其次，用户空间内存管理器可能会增加 CPU 开销，导致额外的 10% 的成本。

在 paper 中，作者认为**保持 KV Cache 的虚拟内存连续性**对于减少 LLM 部署中的软件复杂性和冗余性至关重要。**作者主张，不应在用户级别重新实现分页，而应重新利用操作系统中现有的虚拟内存抽象来进行动态 KV Cache 内存管理，从而简化部署并提高性能。**

为此，paper提出了 vAttention —— 一个在不提前分配物理内存的情况下将 KV Cache 存储在连续虚拟内存中的系统。paper通过利用CUDA支持的 Low-Level 虚拟内存API实现了这一点，这些 API 提供了分配虚拟和物理内存的不同接口。vAttention 提供了一组简单的 API，通过这些 API，服务框架可以为 KV Cache 保留连续的虚拟空间，并按需分配物理内存。这种方法带来了多项好处，如表2所示。vAttention 还通过实现现有 GPU kernel 的无缝重用，消除了在服务系统中实现内存管理器的需求，从而提高了可移植性。

**挑战和优化**：vAttention 解决了在没有 PagedAttention 的情况下实现高效动态内存管理的两个关键挑战。首先，CUDA API 支持的最小物理内存分配粒度为 2MB。根据模型和工作负载的特性，这种大小可能导致显著的容量浪费。为了解决这个问题，paper 修改了开源的 CUDA 统一虚拟内存驱动程序，以支持 64KB 到 256KB的更细粒度的物理内存分配。其次，使用 CUDA API 进行内存分配会产生高延迟，因为每次分配都涉及到kernel的往返。为了将内存分配的延迟对终端用户的影响隐藏起来，引入了几种 LLM 特定的优化措施，例如将内存分配与计算重叠，提前执行一些操作和延迟内存回收。最后展示了这些优化使 vAttention 成为一个高效的KV Cache 内存管理器。

总体而言，作者在 Paper 中做出了以下贡献：

- 提出了 vAttention，一个在保持 KV Cache 虚拟内存连续性的同时实现物理内存动态分配的系统。
- 展示了 vAttention 能够无缝地为未修改的 FlashAttention 和 FlashInfer 的注意力 kernel 添加动态内存管理支持，同时具有高性能。
- 在 1-2个A100 GPU 上评估了 Yi-6B、Llama-3-8B 和 Yi-34B，并展示了使用 FlashAttention 原始 kernel 时，vAttention 的性能优于 vLLM 高达 1.97 倍，而相对于 PagedAttention 版本的 FlashInfer，首 token 生成时间（TTFT）减少了高达 1.45 倍。


在背景部分仍然是对大语言模型，KV Cache，vLLM这些概念反复进行描述，就不赘述了，最后看一下Figure1吧。

![图1. 由于内部碎片导致的内存浪费示意图。Orca（顶部）提前为 KV Cache 保留内存。vLLM（底部）通过动态内存分配来减少碎片。阴影框表示被KV token占用的内存，而白色框表示已分配但未使用的内存。](https://img-blog.csdnimg.cn/direct/bc2a0c74fb6c4c86a86417fe50d26d80.png)


# 0x3. 使用PagedAttention模型的问题
尽管受到需求分页的启发，PagedAttention采用了一种不同于传统需求分页的方法：它要求修改应用程序代码以适应动态分配的物理内存，而传统的需求分页对应用程序是透明的。本节详细说明了这种方法带来的一些问题。

## 0x3.1 需要重写注意力kernel

PagedAttention需要重写注意力kernel。这是因为传统实现的注意力操作符假设两个输入张量K和V存储在连续的内存中。由于偏离了传统的内存布局，PagedAttention要求修改注意力算子的实现，以便在非连续的KV Cache块上计算注意力得分。编写正确且高效的GPU kernel对大多数程序员来说是具有挑战性的。

作为Transformer架构的基本构建块，注意力算子在系统和机器学习社区中见证了大量的性能优化创新，这一趋势可能会继续。在PagedAttention模型中，跟上新研究需要持续努力将新的优化移植到支持PagedAttention的实现中。因此，生产系统很容易落后于研究，从而可能失去性能和竞争优势。举例来说，图9b显示，vLLM的分页kernel在GQA情况下已经比FlashAttention对应的kernel慢多达2.85倍。

## 0x3.2 在服务框架中增加冗余

PagedAttention使LLM服务系统负责管理KV Cache与动态分配的内存块之间的映射。例如，考虑一个请求，它随着时间的推移分配了四个KV Cache块（图2左半部分）。这些块在虚拟内存中通常是非连续的。在计算注意力矩阵的过程中，PagedAttention Kernel需要访问这四个KV Cache块的所有元素。为了实现这一点，服务系统需要跟踪KV Cache块的虚拟内存地址并在运行时将它们传递给注意力kernel。这种方法实际上需要复制操作系统已经为实现虚拟到物理地址转换所做的工作（图2右半部分）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e87d11b457114d58bb2edfb8917906bb.png)


## 0x3.3 性能开销

PagedAttention在GPU和CPU上都会导致潜在的性能问题。vAttention分别对此进行了研究。

### 0x3.3.1 GPU上的运行时开销

PagedAttention通过在关键路径中添加额外代码，减慢了注意力计算。例如，vLLM承认其基于PagedAttention的实现比原始的FasterTransformer kernel慢20-26%，主要是由于查找块表和执行额外分支的开销。图3显示了FlashAttention中的分页解码kernel也比普通kernel慢。进一步分析表明，PagedAttention中执行的指令数量比普通kernel高出13%。vAttention还发现，分页的开销在大批处理大小或长上下文长度时减少。这是因为对于解码来说，计算注意力是内存受限的，当KV Cache大小较大时，内存墙隐藏了指令开销。

![方块代表batch size不同](https://img-blog.csdnimg.cn/direct/f67423197122406e8500a4d10bcdb0bb.png)


然而，在计算复杂度为$N^2$的 prefill 注意力kernel中，隐藏指令开销更具挑战性。例如，图4显示了FlashInfer的prefill kernel分页版本比普通kernel慢多达14%。分页kernel也缺少一些众所周知的优化。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0543498a1f2e43caa014118c47aabd30.png)


另一个例子显示编写高效注意力kernel的难度，图5表明vLLM的分页解码kernel在块大小为64和128时的性能显著较差。分析显示，这可能是由于L1缓存效率：较小的块由于L1缓存命中率较高而具有更高的内存带宽利用率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c37ea17e088d450c9833b299888f2718.png)


### 0x3.3.2 CPU上的运行时开销

实现一个额外的内存管理器会在服务系统的CPU运行时中增加性能问题。作者引用了一些实际案例和对vLLM的观察来证明这一观点。

为了启用PagedAttention，服务系统需要向注意力kernel提供块表。在vLLM中，准备块表的延迟取决于批处理组成，并且随着`max_num_blocks × batch_size`成比例增长，其中max_num_blocks指的是批处理中最长请求的KV Cache块数量。这是因为vLLM将块表管理为一个2D张量，并通过用零填充未占用的槽来对齐每个请求中的KV Cache块数量。如果一个批处理中包含一些长请求和许多短请求，这种填充会导致显著的开销。在作者早期的实验中，作者观察到vLLM中块表准备在解码迭代中贡献了30%的延迟。虽然最近的修复减轻了部分开销，但作者发现它仍然可能高达10%。在TensorRT-LLM中也发现了PagedAttention的高开销，吞吐量下降了11%，从412 tokens/s降至365 tokens/s。这个问题归因于TensorRT-LLM的Python运行时，转移到C++运行时可以减轻CPU开销。

总体而言，本节显示了PagedAttention模型增加了显著的编程负担，同时也效率低下。vAttention通过利用现有的需求分页系统支持，引入了一种更系统的方法来动态管理KV Cache内存。然而，在深入讨论vAttention之前，vAttention首先强调LLM服务工作负载在内存管理方面的一些基本特征。

# 0x4. 对LLM服务系统的洞察

为了突出LLM服务系统的内存分配模式，vAttention对Yi-6B在单个NVIDIA A100 GPU上运行，Llama-3-8B和Yi-34B在两个A100 GPU上以张量并行方式运行进行了实验。vAttention将每个请求的初始上下文长度设置为1K tokens，批量大小从1到320变化，测量了decode阶段的延迟、吞吐量和内存需求（有关prefill阶段的讨论和优化，见第6节）。

**观察1**：每次迭代的KV Cache内存需求是事先已知的。这是因为自回归解码每次请求生成一个 token。因此，每次迭代中，请求的 KV Cache 内存占用都会均匀增长一个token，直到请求完成。

**观察2**：KV Cache不需要高内存分配带宽。单个 token 在所有层中的内存占用通常是几十到几百KB。例如，Yi-6B、Llama-3-8B 和 Yi-34B 的每个 token 内存占用分别为 64KB、128KB 和 240KB。此外，每次迭代运行10到100毫秒（图6b），这意味着每个请求每秒最多需要几MB的内存。尽管批处理提高了系统吞吐量，但每秒生成的 token 数量在某个批量大小之后趋于平稳（图6a）。这意味着内存分配带宽需求在大批量大小时也趋于饱和（例如，对于Yi-34B，在128时）。对于vAttention研究的所有模型，观察到最高内存分配速率最多为600MB每秒（图6c）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/55a4964931184320849573fa3f62bbba.png)


在vAttention中，利用上面这些观察来实现一个高效的KV Cache动态内存管理系统。在下一节中，作者首先提供vAttention的高层设计描述（§5.1），然后讨论vAttention如何用于 Serving LLM（§5.3），最后描述的优化方法（§6）。

# 0x5. vAttention的系统设计
目标是通过为现有的kernel添加动态内存分配支持来提高效率和可移植性。为了实现这一目标，vAttention利用系统支持的动态内存分配，而不是在用户空间中实现分页。

## 0x5.1 设计概述
vAttention基于独立分配虚拟内存和物理内存的能力。具体来说，vAttention在虚拟内存中预先分配一个用于KV Cache的大块连续Buffer（类似于eservation-based的分配器），而将物理内存的分配推迟到运行时，即仅在需要时分配物理内存（类似于PagedAttention）。这样，vAttention在不浪费物理内存的情况下保留了KV Cache的虚拟连续性。这种方法是可行的，因为内存容量和碎片化仅对物理内存构成限制，而虚拟内存是充裕的，例如，现代64位系统为每个进程提供128TB的用户管理虚拟地址空间。

### 0x5.1.1 预保留虚拟内存
由于虚拟内存充裕，vAttention预先分配足够大的虚拟内存空间，以容纳最大批量大小（可配置）的KV Cache。

**虚拟内存buffers数量**：LLM中的每一层都维护自己的K和V张量：vAttention分别称之为K缓存和V缓存。vAttention为K缓存和V缓存分配单独的虚拟内存缓冲区。对于单个GPU任务，这需要预保留$2 \times N$个缓冲区，其中$N$是模型中的层数。在多GPU任务中，每个worker预留$2 \times N^{'}$个缓冲区，其中$N^{'}$是该worker管理的层数（在张量并行下$N^{'}=N$，而在流水线并行下$N^{'}<N$）。

**虚拟内存buffers的大小**：buffers的最大大小为𝐵𝑆 = 𝐵 × 𝐿 × 𝑆，其中𝐵是最大批量大小，𝐿是模型支持的最大上下文长度，𝑆是单个token在worker上的每层K缓存（或V缓存）的大小。此外，𝑆 = 𝐻 × 𝐷 × 𝑃，其中𝐻是worker上的KV头数，𝐷是每个KV头的维度，𝑃是基于模型精度的字节数（例如，FP16/BF16时P=2）。注意，对于给定的模型配置，𝑆是恒定的。

以Yi-34B和FP16及两路张量并行（TP-2）为例。在这种情况下，𝑁 = 60，𝐻 = 4，𝐷 = 128，𝑃 = 2（Yi-34B的8个KV头在两个GPU上平均分配），最大支持的上下文长度𝐿 = 200K。对于这个模型，每个worker每层的K缓存（或V缓存）的最大大小为𝑆 = 200MB（200K ∗ 4 ∗ 128 ∗ 2）。假设𝐵 = 500，每个worker的每个buffer的最大大小为𝐵𝑆 = 100GB（500 × 200MB）。因此，Yi-34B的60层总虚拟内存需求为每个100GB的120个buffers（总计12TB）。注意，虚拟地址空间的数量随着GPU数量的增加而增加，例如，对于两个TP worker，可用的虚拟地址空间为256TB。因此，虚拟内存分配可以轻松满足。

### 0x5.1.2 按需分配物理内存
vAttention优先每次分配一页物理内存，并且仅在请求用完所有先前分配的物理内存页时才进行分配。为了说明其工作原理，vAttention参考图7中的一个简单示例。该示例展示了vAttention在模型的某一层如何管理K缓存（或V缓存），假设最大批量大小为2。其余的K缓存和V缓存缓冲区在所有层中以类似方式管理。


![图7. vAttention中单个K缓存（或V缓存）张量的动态内存管理。（a）显示了一个包含两个请求批次的虚拟张量，还没有进行物理内存分配。（b）R1被分配了一页物理内存。（c）R1被分配了两页内存，R2被分配了一页内存。（d）R1已完成，但vAttention不会回收其内存（延迟回收）。（e）当R3到达时，vAttention将R1的张量分配给它，因为它已经有物理内存支持。](https://img-blog.csdnimg.cn/direct/947a77764163416ebe1b62073030fe2d.png)
再解释一下：

这个图展示的是vAttention如何动态管理单个K缓存（或V缓存）张量的内存，具体分为五个步骤：

1. `(a)`：虚拟内存中包含了两个请求（R1和R2）的虚拟张量，但还没有进行任何物理内存分配。浅色阴影区域代表了没有任何物理页支持的虚拟张量部分。

2. `(b)`：R1被分配了一页物理内存。这个分配使得虚拟张量的一部分（浅色阴影区域）映射到了实际的物理内存上（深色阴影区域）。

3. `(c)`：R1被分配了两页物理内存，而R2被分配了一页物理内存。此时，R1的虚拟张量有两部分映射到了物理内存上，而R2的虚拟张量有一部分映射到了物理内存上。

4. `(d)`：R1已经完成了它的任务，但vAttention没有立即回收其内存（延迟回收）。因此，R1的虚拟张量依旧占据物理内存。

5. `(e)`：当新的请求R3到达时，vAttention将R1的张量分配给R3，因为这些张量已经有物理内存支持。

这个图展示了vAttention如何通过按需分配物理内存来管理 KV Cache 的内存，同时通过延迟回收策略优化内存使用，以便在新的请求到达时可以高效地重新利用已经分配的物理内存。


## 0x5.2 利用low-level的CUDA支持
标准的GPU内存分配接口cudaMalloc不支持需求分页，即它同时分配虚拟内存和物理内存。然而，最近的CUDA版本为程序员提供了对虚拟内存和物理内存的细粒度控制。在vAttention中利用了这些low-level的API。

### 0x5.2.1 CUDA虚拟内存API

表3提供了CUDA API的高级概述，这些API允许将虚拟内存的分配与物理内存的分配分开（见最左列）。分配粒度取决于GPU使用的页大小，并且虚拟内存buffer或物理内存句柄的大小必须是分配粒度的倍数。虚拟内存buffer的不同子区域可以独立于缓冲区中的其他子区域由物理内存支持（参见图7c的示例）。为了简单起见，vAttention将物理内存分配的粒度称为页大小。


表3. 虚拟内存管理的Low-LevelAPI及其在不同分配大小下的延迟。*表示vAttention在实例化或终止服务框架时使用的API。其余的API用于在运行时（取消）映射物理内存页。CUDA API（以cu为前缀）仅支持2MB的分配大小，而vAttention的CUDA扩展API（以v为前缀）支持细粒度的分配。
### 0x5.2.2 扩展PyTorch Caching Allocator
KV Cache是张量的集合。在当前的深度学习框架如PyTorch中，通过API如torch.empty分配的张量带有预先分配的物理内存。这是因为PyTorch Caching Allocator 使用cudaMalloc接口来分配GPU内存（包括虚拟和物理内存）。依赖CUDA的Low-Level API支持，vAttention扩展了PyTorch Cache Allocator，允许应用程序为张量保留虚拟内存缓冲区而不提前分配物理内存。vAttention称通过这些API分配的张量为虚拟张量。

### 0x5.2.3 请求级KV Cache索引
注意，每个虚拟张量表示一层的K缓存（或V缓存），用于最大批量大小B。在这些张量中，不同请求占据不同的不重叠子区域（称为子张量）。vAttention通过唯一的整数标识符$reqId$定位请求的子张量，该标识符在$0$到$B-1$范围内（注意最多有$B$个请求同时运行）。请求的子张量在整个批次的虚拟张量中的K缓存（或V缓存）偏移量为$reqId \times S$，其中$S$是worker上一个请求的最大K缓存（或V缓存）大小。请求标识符$reqId$由vAttention分配。

## 0x5.3 使用vAttention服务LLM
作者将vAttention构建为一个Python库，该库内部使用CUDA/C++扩展与CUDA驱动程序交互。这个库向服务框架公开了一组简单的API，列在表4中。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b1f6a6ba54534b26aa3b5da86dc15686.png)

### 0x5.3.1 初始设置
当服务框架启动时，每个模型workers加载vAttention库，并通过init API用模型参数`𝑁 ′、𝐻、𝐷、𝑃、𝐵`和首选页大小对其进行配置。在内部，vAttention为每个 worker 的KV Cache 保留`2 × 𝑁 ′`个虚拟张量（使用作者修改的PyTorch Caching Allocator）。这些虚拟张量在服务应用程序的整个生命周期内保留。此外，vAttention还在初始化期间预先分配物理内存页。然而，这些页尚未映射到KV Cache中。

### 0x5.3.2 调度新请求
当首次调度新请求时，服务框架通过`alloc_reqid`从vAttention获取一个新的`reqId`。请求的所有后续内存管理操作都 token 为该`reqId`。

### 0x5.3.3 模型执行
在调度批处理执行之前，框架需要确保每个活跃请求的KV Cache子张量由物理内存支持。为此，在将迭代的第一个kernel分派给GPU之前，框架调用step API，指定每个请求的当前上下文长度（每个活跃`reqId`的上下文长度设置为0）。在内部，vAttention确保在将执行返回给框架之前为每个活跃的`reqId`映射足够的物理页。如果vAttention无法满足内存需求，它将返回失败，服务框架可以通过抢占一个或多个请求来继续进行（这类似于vLLM的默认行为）。vAttention将KV Cache交换到CPU内存等更复杂的策略留作未来工作。

根据请求是处于prefill阶段还是decode阶段，给定迭代可能需要映射不同数量的物理内存页。prefill阶段并行处理给定提示的输入token，并在模型每一层的请求的K缓存（和V缓存）中填充一个槽。因此，所需映射的页数取决于调度的 prompt tokens 数量。如果模型某一层所有 prompt tokens 的总K缓存大小为$s$，页大小为$$t$，则每个worker需要确保在给定reqId的`2 × 𝑁 ′`个KV Cache子张量中的每个子张量中至少映射`(𝑠 + 𝑡 − 1)/𝑡`个物理内存页。对于处于decode阶段的请求，每个请求最多需要一个新页。这是因为每次迭代只为一个请求生成一个输出token。vAttention在内部跟踪每个请求映射的页数量，并且只有当分配给该请求的最后一页已完全使用时才映射新页。

### 0x5.3.4 请求完成
当达到用户指定的上下文长度或模型支持的最大上下文长度，或者当模型生成特殊的序列结束 token 时，请求终止。框架通过`free_reqid`通知vAttention请求的完成。在内部，vAttention可能会取消映射已完成请求的页，或推迟释放这些页。


> 这一节是对vAttention的系统设计进行概述。首先会给每个请求分配一个特殊的`reqId`，和其它请求区分开。在初始化阶段，vAttention通过利用low-level的cuda内存分配api修改了PyTorch的Cache Allocator，可以为请求分配足够且连续的虚拟内存buffer。在运行时针对活跃的请求完成具体的物理内存页的映射让请求完成计算，如果物理内存不够了可能会触发请求抢占，也就是某些活跃的请求可能会变成不活跃。最后在请求处理完成之后，需要把这个请求的`reqId`清空。更多的细节，可能还是得等最近作者开源代码才能看到。

# 0x6. vAttention的优化
使用CUDA的虚拟内存支持来服务LLM时面临两个主要挑战。首先，cuMemCreate当前分配的最小物理内存页大小为2MB。大的页会因内部碎片而浪费物理内存。其次，调用CUDA API会带来高延迟。本节详细介绍了一组简单但有效的优化措施，以克服这些限制。

## 0x6.1 减少内部碎片
vAttention通过减少物理内存分配的粒度来减轻内部碎片。NVIDIA GPU本身支持至少三种页大小：4KB、64KB和2MB。因此，原则上，可以以4KB的倍数分配物理内存。实现这一目标的最简单方法是扩展现有的CUDA虚拟内存API（列在表3中），以支持分配更小的页（类似于Linux中的`mmap`支持多种页大小）。不幸的是，CUDA API在闭源的NVIDIA驱动程序中实现，这使得vAttention无法修改它们的实现。

幸运的是，NVIDIA驱动程序的一部分（特别是与统一内存管理相关的部分）是开源的。因此，vAttention在开源的NVIDIA驱动程序中实现了一组新的API，以模拟现有CUDA API提供的相同功能，但支持多种页大小。表3的第二列显示了vAttention的新API：vAttention的大多数API与现有CUDA API一一对应，除了vMemMap（结合了cuMemMap和cuMemSetAccess的功能）和vMemRelease（结合了cuMemUnmap和cuMemRelease的功能）以简化操作。与CUDA API不同，vAttention的API可以分配64KB、128KB和256KB大小的内存。服务框架可以在初始化vAttention时配置所需的页大小：vAttention建议默认使用256KB页。表3的最后一组列显示了每个API在不同页大小下的延迟。

![表3. 虚拟内存管理的Low-LevelAPI及其在不同分配大小下的延迟。*表示vAttention在实例化或终止服务框架时使用的API。其余的API用于在运行时（取消）映射物理内存页。CUDA API（以cu为前缀）仅支持2MB的分配大小，而vAttention的CUDA扩展API（以v为前缀）支持细粒度的分配。](https://img-blog.csdnimg.cn/direct/7cc2b700a1ea45b1895174dccddf548c.png)

## 0x6.2 隐藏内存分配延迟
服务框架在每次迭代中调用step API。step的延迟取决于需要在KV Cache的虚拟张量中映射多少新页。例如，假设Yi-34B的一个请求的KV Cache需要扩展，这个模型有60层。这需要120次调用vMemMap，每次大约需要9微秒。因此，将一个请求的KV Cache增加一页将为相应的迭代增加大约1毫秒的延迟，并且会随着需要映射的物理内存量的增加而增长。vAttention提出了以下优化措施来隐藏分配延迟：

### 0x6.2.1 内存分配与计算重叠
利用内存需求的可预测性来将内存分配与计算重叠。特别地，每次迭代为每个解码请求生成一个输出token。因此，解码迭代的内存需求是提前知道的。此外，在decode阶段，每个请求最多需要一个新页。vAttention跟踪当前上下文长度以及每个请求已映射的物理内存页数。利用这些信息，它可以确定请求何时需要新页，并在前一迭代执行时使用后台线程分配新页。例如，假设请求$R1$在迭代$i$中需要一个新页。当服务框架在迭代$i-1$中调用step API时，vAttention启动一个后台线程，为迭代$i$映射物理内存页。由于迭代延迟通常在10到100毫秒范围内，后台线程有足够的时间在迭代开始执行之前准备物理内存映射。这样，vAttention通过在关键路径之外的 KV Cache 张量中映射物理页来隐藏CUDA API的延迟。注意，在每次迭代中，`step` API仍然需要确保当前迭代所需的物理页实际映射。如果没有，则同步映射所需的页。
### 0x6.2.2 延迟回收 + 预先分配
我们观察到，在许多情况下，可以避免为新请求分配物理内存。例如，假设请求$R1$在迭代$i$中完成，而新请求$R2$在迭代$i+1$中加入运行批次。为了避免从头开始为$R2$分配新页，vAttention简单地延迟回收$R1$的页，并将$R1$的$reqId$分配给$R2$。这样，$R2$使用的是$R1$已经使用的由物理页支持的相同张量。因此，仅当$R2$的上下文长度大于$R1$时，才需要为$R2$分配新页。

vAttention进一步通过在需要之前主动映射物理页来优化内存分配。通过使用一个非活跃的$reqId$的KV Cache来实现这一点。当有新请求到达时，我们可以分配这个$reqId$而不映射任何物理页。然后，我们选择一个将要分配的新$reqId$，并为其映射物理页。在大多数情况下，这些主动优化消除了甚至为新请求的prefill阶段分配新物理页的需要。最后，我们仅在 vAttention 中缓存的物理内存页低于某个阈值（例如，少于GPU内存的10%）时触发内存回收。我们将延迟回收和主动分配都交给`step` API spawns（感觉翻译成启动好一点）的后台线程处理。


> 这一节讲的是使用CUDA虚拟内存支持LLM时存在两个主要挑战：大的物理内存页导致内部碎片和高延迟。然后vAttention通过减少物理内存分配粒度和重叠内存分配与计算来解决这些问题，支持更小的页大小，并通过延迟回收和预先分配优化内存管理。这些优化措施提高了内存利用效率并降低了分配延迟。

# 0x7. 评测
论文的评估旨在回答以下问题：

- vAttention在LLM推理的prefill和decode阶段表现如何？vAttention的可移植性和性能优势是什么？
- vAttention能多高效地为LLM服务工作负载分配GPU内存，以及它能多有效地处理KV Cache碎片？

**模型和硬件**：我们评估了三个模型Yi-6B、Llama-3-8B和Yi-34B，使用一台NVIDIA A100 GPU评估Yi-6B，使用两台通过NVLink连接的A100 GPU评估Llama-3-8B和Yi-34B（见表5）。每个GPU有80GB物理内存。我们对Llama-3-8B和Yi-34B都使用了两路张量并行（TP-2）。所有三个模型都使用了GQA，这是最近LLM中最常用的注意力机制。

**评估方法**：prefill 和decode阶段的计算和内存分配模式有显著不同。这两个阶段使用的注意力 kernel 也不同，因此我们分别评估它们。prefill阶段需要一次性内存分配，可能跨越多个页。相比之下，decode 阶段需要在请求的整个生命周期内进行增量内存分配。我们以每秒处理（或生成）的 token 数来衡量这些阶段的吞吐量。

## 0x7.1 prefill阶段的可移植性和性能
为了评估prefill阶段，我们重点关注FlashAttention v2.5.6和FlashInfer v0.0.3提供的注意力kernel。我们没有在这些实验中包括vLLM，因为它没有自己的prefill内核，而是使用FlashAttention的kernel。此外，由于FlashInfer kernel不支持Yi-34B的KV group size大小为7，我们也无法评估Yi-34B。

FlashInfer是一个最近推出的库，提供了一组针对不同场景优化的注意力kernel，例如用于分块 prefill 的 kernel——这是 Sarathi 提出并后来在各种系统中采用的优化。Sarathi 将 prompt 的输入 tokens 拆分为多个较小的块，并一次调度一个块，从而使服务系统可以在不中断正在进行的解码的情况下添加新请求。这有助于在不增加延迟的情况下提高吞吐量。FlashAttention 和 FlashInfer 都提供了计算分块 prefill 注意力得分的 kernel (支持和不支持 PagedAttention) 。我们将它们集成到vLLM中，使用 2048 个 tokens 的chunk size，测量以下配置的首  token 生成时间（TTFT）：

**FA_Paged**：使用 FlashAttention 的 flash_attn_with_kv_cache kernel API。
**FI_Paged**：使用 FlashInfer 的 PagedAttention kernel，代表了 prefill 阶段基于 PagedAttention 的最新kernel。
**FA_vAttention**：通过 `flash_attn_func` API使用 FlashAttention 原始的 prefill kernel。(应用vAttention)
**FI_vAttention**：通过 `single_prefill_with_kv_cache` API 使用 FlashInfer 的普通 prefill kernel。(应用vAttention)

因此，vAttention的两个配置都使用支持分块 prefill 的 kernel，且在虚拟连续的KV Cache上进行。我们为它们添加了动态内存分配支持，而无需修改它们的代码。

图8显示了四种配置在 Yi-6B 和 Llama-3-8B 上的 prefill 吞吐量。在所有情况下，vAttention 都提供了更高的吞吐量，超过 FA_Paged 的 1.60-3.92 倍和 FI_Paged 的 1.03-1.45 倍。对于相同的实验，表6显示了不同上下文长度下的 TTFT。由于 TTFT 直接依赖于 prefill 吞吐量，与使用普通 kernel 的 vAttention 相比，FA_Paged 和FI_Paged 的 TTFT 分别增加了最多3.92倍（Yi-6B，上下文长度192K）和1.45倍（Llama-3-8B，上下文长度192K）。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/376bb4095f764177951e727dcad592af.png)


在这些情况下，vAttention性能提升的原因有两个。首先，虽然 FlashAttention 的 Paged kernel 未针对 prefill 优化（它针对 decode 进行了优化），FlashInfer的 Paged kernel 专门设计支持分块 prefill，但是非Paged的 kernel 在 FlashAttention 和 FlashInfer 中都比Paged kernel 更快。这一例子说明了实现性能关键优化的复杂性，即使这些实现是由同一团队编写的。第二个性能提升来源是 vAttention 较低的CPU开销。例如，将新的 K 或 V 张量追加到 KV Cache 中在 vAttention 中只需要一次张量复制操作，而在分页实现中则需要一次一个块地追加。此外，FlashInfer 在每次迭代中涉及创建和删除一些对象以管理其压缩的块表。vAttention避免了这种开销，因为它保持了 KV Cache 的虚拟连续性，因此不需要块表。


## 0x7.2 decode阶段的可移植性和性能

为了评估解码性能，我们重点关注长上下文场景（16K），因为只有在长上下文情况下注意力kernel的延迟才变得显著。我们评估了以下配置：

- **vLLM**：我们使用vLLM v0.2.7作为主要基准。vLLM开创了PagedAttention，并使用从FasterTransformer派生的自定义Paged kernel进行解码。
- **FA_Paged**：作为第二个基准，我们将 FlashAttention kernel 集成到vLLM的服务栈中。这代表了最新的PagedAttention kernel，包括序列并行和将新的 key 和 value 向量就地复制到 KV Cache 中的优化。我们使用两种不同的块大小（16和128）评估 vLLM 和 FlashAttention 的 Paged kernel，以捕捉块大小对性能的影响。
- **FA_vAttention**：对于 vAttention，我们将 FlashAttention 的普通 kernel 集成到vLLM的服务栈中。该 kernel 与虚拟连续的 KV Cache 一起工作，我们使用 2MB 页动态分配物理内存。

图9a显示了 Yi-6B、Llama-3-8B 和 Yi-34B 在不同批量大小下的解码吞吐量，其中每个请求的初始上下文长度为 16K tokens，并为每个请求生成 256 个 tokens。我们基于 256 次解码迭代的平均延迟计算解码吞吐量。以下是主要结论：

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/160bcf70033a4638bae81dd3b07d0b2b.png)

首先，vAttention在吞吐量上优于vLLM（两种块大小）和FA_Paged（块大小16），并大致匹配FA_Paged的最佳配置（块大小128）。与vLLM相比，vAttention在Yi-6B上的最大提升为1.97倍，在Llama-3-8B上为1.3倍，在Yi-34B上为1.6倍。随着批量大小的增加，相对于vLLM的相对增益也增加。例如，对于Yi-6B，当批量大小从1增加到8时，增益从大约1.1倍增加到1.97倍。这是因为注意力计算的延迟与批量中的 tokens 总数成正比增长（见图9b），而线性操作的成本保持大致不变。因此，注意力 kernel 在总体延迟中的贡献——以及更高效 kernel 带来的增益——随着批量大小的增加而增加。虽然FA_Paged（块大小128）提供了与 vAttention 相似的增益，但请注意，FA_Paged 需要重新实现 GPU kernel，而 vAttention 则简单地利用了 FlashAttention 的普通kernel。

其次，图9b确认了vLLM和FA_Paged/vAttention之间的性能差异确实是由于注意力kernel。在最坏情况下，vLLM的最佳PagedAttention kernel（块大小16）的延迟比FlashAttention kernel高出最多2.85倍（Yi-6B），1.45倍（Llama-3-8B）和2.62倍（Yi-34B）。

最后，即使在内存容量不是约束条件的情况下，吞吐量也可能对块大小敏感。例如，如paper第3.3节所述，vLLM的注意力 kernel 在块大小128时的延迟显著高于块大小16（见图9b）。在最坏情况下，块大小128会使vLLM的吞吐量降低36%。虽然块大小对FlashAttention的影响较小，但使用小块大小仍然会由于 CPU 开销而影响吞吐量，特别是每次迭代创建块表的开销（§3.3）。例如，Llama-3-8B使用块大小128的 FlashAttention 比块大小 16 的吞吐量高出7%（批量大小为32时，每秒531个 tokens 对比494个 tokens）。


## 0x7.3 物理内存分配的有效性
PyTorch的Caching Allocator在分配内存对象（如张量）时不需要往返kernel。而 vAttention在 请求的KV Cache中映射新物理页时需要调用CUDA的内核驱动程序。本节展示了通过我们的优化，vAttention可以有效满足LLM服务系统中 prefill 和 decode 阶段的要求。

表7显示，即使使用我们最小的页大小 64KB ，vAttention 每秒每个 GPU 可以分配高达 7.6GB 的内存。这比解码的最大内存分配速率600MB每秒（图6）高一个数量级以上。更大的页大小和更高的TP维度会成比例地增加内存分配速率。这表明vAttention完全能够满足解码的内存分配带宽需求。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6e5d53f795804ae0a546790214c9367a.png)


此外，图10显示了我们通过将内存分配与模型执行重叠的优化也隐藏了调用 CUDA API 的延迟影响。这个示例显示了Llama-3-8B在TP-1和批量大小为4时连续解码迭代的延迟。如果不将内存分配与计算重叠，新页同步分配给 KV Cache 会使某些迭代的延迟从25毫秒增加到41毫秒（≈4毫秒的延迟是由于为单个请求分配内存）。注意，这些延迟峰值每1024次迭代后发生一次，因为我们在这些实验中使用了2MB的页大小，并且每个 2MB 页包含 1024 个tokens。将内存分配与前一次解码迭代的模型执行重叠时，延迟效应完全被隐藏。

![图10. 解码迭代中有无与计算重叠内存分配的延迟（批量大小=4，上下文长度=32K）。峰值显示了同步内存分配的延迟影响。](https://img-blog.csdnimg.cn/direct/9065033d0058402ca8431e3afa1de8b4.png)


最后，由于 prefill 可能需要为 KV Cache 分配多个页，我们还研究了不同的内存分配方案如何影响 prefill 的延迟。图11显示，当按需同步分配物理内存（当我们的后台线程、延迟回收和预先分配优化都被禁用时）可能会增加多达15%的开销，使用64KB的页大小时尤为明显。更大的页大小会摊销分配成本，将开销降低到最低3%（使用256KB和2MB页大小时）。vAttention通过延迟回收和预先分配进一步减少分配成本，同时将内存分配与计算重叠。在大多数情况下，这些优化确保新到达的请求可以简单地重用先前请求分配的物理内存页。因此，vAttention几乎没有开销，其 prefill 性能与vLLM一样出色。

![图11. 计算一个包含16K token的单个提示的 prefill 阶段所需的时间。由于vAttention在关键路径之外分配内存，其性能与vLLM一样出色。为了说明不同页大小的结果，图中展示了在请求的 prefill 阶段开始之前同步分配物理内存的开销。](https://img-blog.csdnimg.cn/direct/798b455e2db148a6af2023d19595ea70.png)

## 0x7.4 内存碎片分析

表8显示了块大小（定义为页中最小的 token 数）以及在最坏情况下因过度分配而可能浪费的物理内存量。最坏情况发生在分配了一个新页但完全未使用的情况下。此外，我们展示了每个模型在两种TP配置（TP-1和TP-2）下的表现，以突出TP维度对块大小的影响。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e9ff9346a51c4e409f0f8bd2fc1aabb2.png)


vAttention在每个 TP woker 上分配相当于页大小的物理内存，而随着TP维度的增加，worker 每个 token 的物理内存需求下降（因为 KV 头在 TP worker 间分割）。因此，块大小与TP维度成比例增加。表8显示，这导致最小的块大小从32（Yi-34B TP-1）到128（Yi-6B TP-2）不等。在物理内存量方面，64KB页大小的最坏情况下每个请求最多浪费4-15MB，而 256KB 页大小则增加到16-60MB。总体而言，重要的是通过控制物理内存分配的粒度，vAttention 使内存碎片变得无关紧要。请记住，所有模型的服务吞吐量在批处理大小约为 200 时达到饱和（图6）。因此，即使在如此大的批处理大小下，最坏情况下的内存浪费也仅为几GB。因此，类似于vLLM，vAttention 在减少碎片方面非常有效，并且允许使用大批量大小进行服务。如果需要，页大小还可以进一步减小到4KB，这是包括NVIDIA GPU在内的几乎所有架构今天支持的最小页大小。

### 实现工作量

vAttention的主要优势是可移植性：它允许无缝集成新的注意力 kernel，而无需编写其分页版本或更改服务框架。例如，切换 FlashAttention 和 FlashInfer 的 prefill 或 decode kernel 只需要几行代码的更改，如图12所示。相比之下，在PagedAttention中，开发人员首先需要编写一个分页的注意力 kernel，然后在服务框架中进行重大更改。例如，将FlashInfer decode kernel 集成到 vLLM 中需要在15个文件中进行超过600行代码的更改。在FlashAttention GPU kernel 中实现初始分页支持也需要大约280行代码更改以及额外的工作以支持更小的块大小。鉴于LLM的快速创新，我们认为减少编程负担很重要：生产系统应该能够利用注意力操作符的新优化而无需重写代码——类似于深度学习框架在不需程序员干预的情况下利用优化的GEMM实现。

![图12. 代码变更示意图：将FlashAttention的 prefill 注意力 kernel 替换为 FlashInfer 的 prefill 注意力kernel。vAttention 支持透明的动态内存分配，使得 kernel 之间的切换变得容易。](https://img-blog.csdnimg.cn/direct/9d1ae6d6ff18446bbcfc35565cb80bef.png)

> 这部分主要是展示一些实验结果，读者可以选择性阅读或者忽略。


# 0x8. 总结
PagedAttention 已成为 LLM 推理中动态内存分配的事实标准。PagedAttention 消除了提前预留 GPU 内存的需求，从而通过在 GPU 内存中容纳更大的批量大小来提升服务吞吐量。虽然 PagedAttention 有效地解决了内存碎片问题，但作者认为其将  KV  Cache 存储在非连续虚拟内存中的方法引入了软件复杂性、可移植性和效率挑战。相反，paper展示了使用 Low-Level 系统支持的需求分页可以避免 PagedAttention 的缺点。paper 提出的系统 vAttention 为现有的注意力 kernel 添加了动态物理内存分配支持，无需重写 GPU 代码或在服务框架中编写内存管理器。并证明了 vAttention 在减少软件复杂性的同时，提高了可移植性和性能。




