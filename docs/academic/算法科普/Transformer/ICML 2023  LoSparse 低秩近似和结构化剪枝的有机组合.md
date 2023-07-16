# 标题：ICML 2023 | LoSparse：低秩近似和结构化剪枝的有机组合

收录于合集 #低秩近似 #ICML 2023 #结构化剪枝

## 1. 论文信息

![Untitled](https://img-blog.csdnimg.cn/54926d0512f848389d0a83e3a362882f.png#pic_center)

标题：LoSparse: Structured Compression of Large Language Models based on Low-Rank and Sparse Approximation

原文链接：[https://arxiv.org/pdf/2306.11222.pdf](https://arxiv.org/pdf/2306.11222.pdf)

代码链接：[https://github.com/yxli2123/LoSparse](https://github.com/yxli2123/LoSparse)

## 2. 动机&背景

Transformer 模型在各种自然语言任务中取得了显著的成果，但内存和计算资源的瓶颈阻碍了其实用化部署。低秩近似和结构化剪枝是缓解这一瓶颈的主流方法。然而，作者通过分析发现，结构化剪枝在高稀疏率时往往不可避免地删除表达神经元，这将导致模型性能严重降低。低秩近似则旨在压缩表达神经元，它对于压缩神经元中的相干部分十分有效，其本质就是提取神经元共享相干子空间的公共基，该方法在 Transformer 结构上也遇到了困难，不同于 CNN，Transformer 模型的权重矩阵往往是满秩的，这导致低秩近似会破坏神经元的多样性，从而影响模型的表达能力。

为了解决结构化剪枝和低秩近似的局限性和困难，本文提出了一种新的模型压缩技术 LoSparse（Low-Rank and Sparse approximation），该技术通过低秩矩阵和稀疏矩阵的和来近似权重矩阵。这种复合近似将相干部分与神经元的非相干部分解耦。**低秩近似压缩神经元中的连贯和表达部分，而修剪去除神经元中的不连贯和非表达部分**。从这个意义上说，低秩近似可以防止剪枝过度去除表达神经元，而稀疏近似增强了低秩近似的多样性。

## 3. 方法：LoSparse

本文提出了一种 Transformer 模型的压缩方法——LoSparse。具体来说，LoSparse 通过低秩矩阵和稀疏矩阵的和来近似权重矩阵（如图 1 所示）。这两个近似的组合使得压缩方法更有效和稳定。

![图 1. LoSparse 在单个线性投影矩阵的示意图（两部分并行进行前向传递）](https://img-blog.csdnimg.cn/f6635f49023c43e493e5b6b76c8ab532.png#pic_center)

图 1. LoSparse 在单个线性投影矩阵的示意图（两部分并行进行前向传递）

### 3.1  低秩矩阵和稀疏矩阵的近似

给定一个权重矩阵 $W \in \mathbb{R}^{d_1 \times d_2}$，通常采用结构化剪枝稀疏矩阵 $S \in \mathbb{R}^{d_1 \times d_2}$ 来近似 $W$ 以进行压缩。然而，稀疏矩阵近似导致性能不佳，尤其是当压缩比率较高时。因此，本文引入了一个低秩矩阵来改进近似。具体来说，权重矩阵可以表示为：

$$
\begin{equation}W=U V+S \text {, }\end{equation}
$$

其中 $U \in \mathbb{R}^{d_1 \times r}$ 和 $V \in \mathbb{R}^{r \times d_2}$ 的乘积表示秩为 $r$ 的低秩矩阵。

![图 2. 语言模型的奇异值](https://img-blog.csdnimg.cn/7071eba888fd42b6b1216aadc7947917.png#pic_center)

图 2. 语言模型的奇异值

**为什么需要低秩矩阵？**首先，它可以有效地逼近神经元的相干部分。如图 2 所示，我们可以看到语言模型中权重矩阵的频谱在开始时迅速下降。这表明权重矩阵中的神经元有一个共同的子空间，可以看作是这些神经元的连贯部分。此外，公共子空间可以通过顶部奇异值的奇异向量来恢复。其次，低秩矩阵和稀疏矩阵的解耦使得剪枝变得容易。图 2 中的尾谱表示每个神经元跨越它们的单个子空间，可以表示这些神经元的非相干部分。由于这些子空间不共享，因此低秩近似无法捕获非相干部分。幸运的是，低秩矩阵能够将相干部分与神经元的非相干部分解耦。这使我们能够通过添加一个新的矩阵 $S$  来近似剩余的不连贯部分，然后修剪非表达不连贯的部分。图 3 表明，大多数不连贯的部分在解耦后具有较低的重要性分数，这有助于剪枝删除这些冗余参数。

![图3. 线性投影的神经元的重要性得分分布情况（ITP vs LoSparse）](https://img-blog.csdnimg.cn/b519d5f5951146f4811e864dd45a0954.png#pic_center)

图3. 线性投影的神经元的重要性得分分布情况（ITP vs LoSparse）

### 3.2 算法

给定一个预训练的权重矩阵 $W^{(0)}$，我们首先基于 $W^{(0)}$ 的奇异值分解（SVD）初始化秩 $r$  的低秩矩阵。具体来说，本文选择：

$$
\begin{equation}\begin{aligned}& U^{(0)}=\left[\sqrt{\sigma_1} u_1 ; \sqrt{\sigma_2} u_2 ; \ldots ; \sqrt{\sigma_r} u_r\right], \\& V^{(0)}=\left[\sqrt{\sigma_1} v_1 ; \sqrt{\sigma_2} v_2 ; \ldots ; \sqrt{\sigma_r} v_r\right]^{\top},\end{aligned}\end{equation}
$$

在此基础上，我们可以初始化 $S^{(0)}$ 为：

$$
\begin{equation}S^{(0)}=W^{(0)}-U^{(0)} V^{(0)} \end{equation}
$$

原始的前向传递（$Y=X W$）可替换为更高效的形式：

$$
\begin{equation}Y=(X U) V+X S .\end{equation}
$$

LoSparse 对模型的每个权重矩阵应用这样的分解，并将 $\mathcal{S}=\left\{S_m\right\}_{m=1}^M$ 表示为所有稀疏矩阵的集合。初始化后，本文对 $S$ 进行迭代结构化剪枝。具体来说，在第 $t$ 次迭代时，我们首先采用随机梯度下降更新 $U^{(t)}$、$V^{(t)}$ 和 $S^{(t)}$。重要性得分和迭代更新策略均采用标准设置（一阶泰勒评估重要性+三次时间表的迭代衰减策略）。具体算法见算法 1。

![Untitled](https://img-blog.csdnimg.cn/9dce76a0380f4b7490922042cbb70c12.png#pic_center)

## 4. 实验

1. **自然语言理解**：表 1 和 表 2 分别展示了 DeBERTaV3-base 和 BERT-base 模型上各个压缩方法在 GLUE 上的表现。LoSparse 表现出了远超其他方法的性能，与此同时，它还比其他方法更稳定，这是因为 LoSparse 方法中每个权重矩阵至少有一个低秩矩阵来保证连贯和表达神经元信息的不过分丢失。

![表 1. GLUE 验证集上 DeBERTaV3-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）](https://img-blog.csdnimg.cn/5f4f2f0770e34e2dad568a95895e842f.png#pic_center)

表 1. GLUE 验证集上 DeBERTaV3-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）

![表 2. GLUE 验证集上 BERT-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）](https://img-blog.csdnimg.cn/e5e0b86aa25f435abf2daf56d8f0a6ab.png#pic_center)

表 2. GLUE 验证集上 BERT-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）

2. **问答任务**：表 3 对比了 LoSparse 方法在 SQuAD v1.1 上的表现。在所有压缩比率下，LoSparse 都优于其他压缩方法，尤其是在更高压缩比的情况下。

![表 3. SQuAD v1.1 上 DeBERTaV3-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）](https://img-blog.csdnimg.cn/055bc8b584e2406dbef52d8d57e26624.png#pic_center)

表 3. SQuAD v1.1 上 DeBERTaV3-base 的压缩结果（Ratio 表示剩余权重比例，N.A.表示模型不收敛，最佳结果以粗体显示）

3. **自然语言生成**：表 4 说明在自然语言生成任务上，LoSparse 仍然表现优异，在各个压缩比下优于现有方法。值得注意的是，LoSparse 在更困难的摘要任务上表现更好。

![表 4. XSum 上 BART-Large 的压缩结果（Ratio表示剩余权重比例，最佳结果以粗体显示）](https://img-blog.csdnimg.cn/bbe9bd9ca0854ca2b843358a598a8374.png#pic_center)

表 4. XSum 上 BART-Large 的压缩结果（Ratio表示剩余权重比例，最佳结果以粗体显示）

4. **消融实验**：论文分析了稀疏近似的有效性和稀疏分配的影响（低秩矩阵和稀疏矩阵的权重占比），实验表明本文提出的稀疏近似对于性能有很大正贡献，且 LoSparse 对稀疏分配策略相对鲁棒，具体细节可见原文。
