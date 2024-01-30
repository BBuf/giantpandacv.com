> 我们推出了一个新的系列，对PytorchConference2023 的博客进行中文编译，会陆续在公众号发表。也可以访问下面的地址 https://www.aispacewalk.cn/docs/ai/framework/pytorch/PytorchConference2023/%E4%BD%BF%E7%94%A8TorchBench%20for%20PyTorch%E6%A0%87%E5%87%86%E5%8C%96CPU%E5%9F%BA%E5%87%86%E6%B5%8B%E8%AF%95 阅读。

# 大纲

1. 讲座背景与目的
2. Torchbench简介
3. 支持的PyTorch CPU特性
   - Channels last (NHWC)支持
   - INT8量化模型
   - 自动混合精度(torch.amp)
4. 增加模型覆盖范围
5. 实现CPU用户基准测试
   - 测试架构兼容性
   - 运行时配置选项与性能指标
6. 未来计划与展望

## 详细要点

### 1. 讲座背景与目的

- 明飞代表同事王传奇和姜彦斌介绍基于 Torchbench(https://github.com/pytorch/benchmark)的CPU基准测试标准化工作。
- Meta工程师赵旭提供了关键协助。

### 2. Torchbench简介

- TorchBench是一个开源的PyTorch性能评估工具包，旨在创建并维护一个用于CPU的标准化基准测试套件。
- 其目标包括：
  - 监测性能退化情况
  - 验证新优化策略的效果
  - 提供可复现的基准测试环境

### 3. 支持的PyTorch CPU特性

#### Channels last (NHWC)支持
- 对于CNN模型（如ResNet-50），提供训练和推理阶段的channel_last内存格式支持。
- 结合Intel PyTorch扩展，默认选择channel_last以适应特定场景。

#### INT8量化模型
- 利用torch.fx前端实现INT8量化模型。

#### 自动混合精度(torch.amp)
- 在CPU上使用FP32与BF16混合精度，针对Xeon硬件优化，未来将涵盖float16支持。

### 4. 增加模型覆盖范围

- 添加了GCN、GIN、SAGE、EdgeCNN等典型GNN工作负载至TorchBench中。
- 确保现有CNN、Transformers模型在CPU后端得到良好支持。

### 5. 实现CPU用户基准测试

- 支持x86和ARM架构下的全面基准测试。
- 创建"cpu"文件夹，在userbenchmark下添加适用于典型CPU用户场景的基准测试。
- 提供CPU运行时配置选项，如OpenMP线程数、CPU亲和性和Neumark控制等。
- 支持吞吐量、延迟等性能度量指标。

### 6. 未来计划与展望

- 持续扩大模型覆盖范围，增加大型语言模型等新模型。
- 集成新的CPU特性，例如电感器量化技术。
- 不断增强CPU用户基准测试，并将其整合到PyTorch常规测试流程和HUD中。
- 考虑引入编译器后端的新特性，如AOT Inductor，并与其他工具进行比较。

大家好，我是来自英特尔的明飞。今天的主题是关于使用[Torchbench](https://github.com/pytorch/benchmark)对PyTorch社区进行CPU基准测试的标准化。实际上，这是我同事王传奇和姜彦斌的一项工作，但不幸的是他们有一些签证问题，无法亲自来参加，所以我代替他们进行演讲。特别感谢来自Meta的工程师赵旭，在这项工作中给予了很多帮助。首先，我们来看一下为什么做这个？

```
TorchBench is a collection of open-source benchmarks used to evaluate PyTorch performance. 

It aims to create and maintain a standard CPU benchmarking suite for PyTorch, which can be used for:

Track performance regressions
Prove performance benefit of new optimizations
Easy to reproduce
```


TorchBench是一个开源基准测试集合，用于计算PyTorch项目的性能。它包含了几个非常流行的模型，例如传统的基于卷积神经网络的图像分类模型和transformers等等。一个问题是，它主要面向GPU(CUDA)，所以我们想要增加对CPU性能测试的覆盖范围。我们在这里做的是在TorchBench中创建和维护一个标准化的CPU基准测试。它有三个用途，首先，我们可以使用它来跟踪性能状况。通常情况下，当我们提交性能优化的PR时，它可能在某些情况下加速，但在其他情况下表现不佳，甚至可能引入一些降低。借助TorchBench的帮助，我们可以很容易地排除引入性能下降的PR。此外，我们还可以使用TorchBench来证明新优化和硬件特性的性能优势。由于TorchBench是一个标准化的测试套件，我们更容易重现测量得到的性能值。

```
Supported benchmarking for PyTorch CPU features includes:

Channels last (NHWC) support for both inference and training.
Quantized model using torch.fx with INT8.
Automatic mixed precision (torch.amp) with BF16.
```
我们在TorchBench套件中提供了几个已启用的Torch在CPU端的特性。首先是 channel_last 支持。channel last 有时被称为 NHWC。它适用于基于CNN的图像分类模型，如ResNet-50。通常，channel_last 比 channel_first 运行得更快。我们还有一个项目，名为Intel PyTorch扩展。默认选择channel_last作为内存格式。在这里，我们想要做的不单是比较这两种的性能。因为在某些场合，channel_last可能没有那么快。而Torchbench可以帮助我们选择特定的情况。其次，我们还使用Torch FX的前端启用了量化模型的INT8。且接下来的工作是启用新的量化后端。最后一部分是我们还启用了自动混合精度，即torch AMP。在CPU端，torch AMP指的是FP32和BF16。这是因为在当前的Xeon硬件上，BF16更适合于实现加速。我们只有矩阵乘法、INT8和BF16硬件加速器，没有float16。在下一代中，我们将覆盖float16。因此，未来torch AMP的行为将在CPU和GPU之间一视同仁。

```
Increase Model Coverage

Added typical GNN workloads such as GCN, GIN, SAGE, EdgecNN, etc.
Enabled & fixed CPU support in existing models in CNN, Transformers, etc.
```
另外，我们还在Torchbench中增加了一些GNN workload，如GCN、GIN、SAGE和EdgeCNN等。集成GNN模型到Torchbench中遇到的最大挑战之一是一些GNN workload具有非常大的数据集。因此，如果想运行整个数据集，将需要相当多的时间。此外，使用随机生成的数字并不合理，因为输入中的稀疏模式实际上是有意义的，它代表了源和目标之间的连接。为了解决这个问题，我们选择了从整个数据集中选择一个子集，并将其输入到Torchbench中。
```
Implement a user benchmark for CPU.
- Cover all benchmarks in X86 and ARM.
Added "cpu" folder under userbenchmark.
- Fit for typical CPU user scenarios.
Added CPU runtime configuration options.
Supported metrics (Throughput/Latency/...).
```

我们所做的另一件事是启用和修复现有模型（如CNN和transformers）中的一些CPU支持，以确保它们在CPU后端上自由运行。此外，我们还增强了CPU后端的测试框架能力。我们为CPU实现了一个用户基准测试。它适用于x86和ARM架构。另外，如果您想测量CPU性能，设置正确的运行时配置非常重要，例如OpenMP线程数、CPU亲和性，以及在最新一代Xeon上还需要设置Neumark控制。因此，我们有几个API用于对CPU后端进行运行时调整。我们还启用了一些度量指标来衡量性能，例如吞吐量和延迟。

```
- Continuously improve model coverage.
- Integrate new CPU features, e.g., inductor quantization and so on.
- Enhance CPU user benchmark and promote it into PyTorch regular test workflows and HUD.
```

在未来，我们将不断改进模型覆盖范围，添加更多新的模型，如大型语言模型等。我们还将集成新的特性，如inductor、量化等。在本次会议中，我们听到了很多关于编译器后端的新特性，比如AOT inductor。我们肯定会将新技术加入到Torchbench中，以进行交叉比较。所以人们可能会想知道，我能否用不同的工具实现相同类型的目标。哪个是最好的呢？我们将继续提升CPU用户基准测试，并将其推广为PyTorch的常规测试。

