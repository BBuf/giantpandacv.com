> 我们推出了一个新的系列，对PytorchConference2023 的博客进行中文编译，会陆续在公众号发表。也可以访问下面的地址 https://www.aispacewalk.cn/docs/ai/framework/pytorch/PytorchConference2023/torchfix 阅读。

## 大纲

1. TorchFix简介

2. TorchFix试图解决的几个问题实例

3. TorchFix的工作原理

4. TorchFix的两种使用模式

5. TorchFix的获取及启用规则

6. TorchFix下一步工作

## 详细要点

### 1. TorchFix简介

- 旨在帮助维护PyTorch代码质量,遵循最佳实践

### 2. 几个问题实例

- PyTorch API变更带来的兼容性问题

- requires_grad拼写错误难以排查

- 数据加载器性能问题

- TorchVision API变更频繁

### 3. 工作原理

- 使用LibAST解析和修改语法树

### 4. 两种使用模式

- Flake8插件 mode: 方便集成,无自动修复

- 独立运行模式:提供自动修复

### 5. 获取及启用规则

- 提供了查找和修复上述问题实例的规则

- 规则默认不全部启用,可以通过参数选择

### 6. 下一步工作

- 支持更多类别的规则

- 提供更多配置选项

- 与PyTorch生态更好地集成





你好，我叫塞尔吉。我在Meta公司负责PyTorch的开发者体验。今天我想要谈谈TorchFix。

TorchFix是我们最近开发的一个新工具，旨在帮助PyTorch用户维护健康的代码库并遵循PyTorch的最佳实践。首先，我想要展示一些我们努力解决的问题的示例。

```Plain Text
For Cholesky decomposition
- torch.cholesky deprecated in favor of torch.linalg.cholesky
- Replace torch.cholesky(A) with torch.linalg.cholesky(A)
- Replace torch.cholesky(A,upper=True）with torch.linalg.ch
```

首先是第一个示例。最近，PyTorch的API中计算Cholesky分解的函数发生了改变。将该函数从torch.shalesky移动到torch.linauk.shalesky，并且参数也进行了变更。在旧的API中，您可以提供upper equals true参数，但在新的API中，您只需计算一个联合。我们希望更新我们的代码以使用这个新的API。但是手动操作这个过程非常繁琐。

```Plain Text
- Bad: param.require grad = False
- Good: param.requires grad = False (notice 'requires')
The bad code doesn't cause any explicit errors but doesn't do what it's supposed to do 
```

有时，出于性能原因，您不想为参数计算梯度。要告诉PyTorch您不需要梯度，只需将requires grad属性设置为false。不幸的是，人们会经常会输入require gradient， requiregrad false。又因为这个是python，属性动态创建没有错误，你的程序继续工作但没有执行预期的操作，这可能会导致性能下降。这实际上很难注意到，我们在多个流行的大型开源库发现了这个问题。

```Plain Text
- Synchronized dataloader
torch.utils.data.DataLoader(dataset,batch_size=10)
- For efficiency in production
torch.utils.data.DataLoader(dataset,batch_size=10,num_workers=n）
```

关于数据加载器的另一个问题是，如果你没有为数据加载器提供"numWorkers"参数，那么默认值为零。这意味着数据加载将在与计算相同的进程中进行。数据加载可能会阻塞计算。因此，出于效率原因，您希望在生产环境中提供"numWorkers"参数，并将其设置为大于零的值。具体的数字可能取决于您拥有的CPU数量或其他因素。但这个问题不一定是一个错误。根据您的目标和代码的运行方式，默认值0可能是完全有效的。但是，我们仍然希望向用户标记此问题，以便用户可以检查和理解它是否对他们造成了实际问题。

```Plain Text
TorchVision introduced new
Multi-weight support API
Replace
models.resnet101(pretrained=True)
with
models.resnet101(weights=models.ResNetl01_Weights.IMAGENET1K_V1)
```

这个例子与核心PyTorch无关，而是与一种流行的领域库TorchVision有关。

最近，TorchVision中加载预训练权重的API发生了变化。所以以前你提供的是Pretend等于true或等于false。但是使用新的API，你需要提供weight参数，并明确指定要加载的权重。这个新的API更加灵活，我们希望更新我们的代码来使用它。实际上，我们希望全世界的代码都能更新使用这个新的API。在那之后，TorchVision可以完全停止支持旧的API。再次强调，手动做这个过程非常繁琐。特别是考虑到TorchVIsion不只有一个模型，TorchVision有许多模型和许多权重，这个API的变化适用于所有模型。

```Plain Text
: A specialized static analysis tool for Pythoncode
Uses LibcsT
- A concrete syntax tree parser and serializer library for Python
- Similar to standard Python's ast, but.preserves things like comments and formatting
- https://github.com/Instagram/LibCST
```

如果有什么是解决所有这些问题的方法，它就是TorchFix。TorchFix是一个专门为PyTorch设计的静态分析工具。他们使用了Lipcea ST这个流行的库。Lipcea ST允许TorchFix加载、获取语法树、更新语法树，然后将修改后的语法树写回。关于如何运行TorchFix，有两种模式。一种是作为Flake8插件，另一种是独立模式。在Flake 8插件模式中，你只需要安装Torchfix，然后基本上使用Flake 8. 如果你的项目中已经使用了Flake 8，这种模式非常方便。如果你的CI正在运行Flake 8，你只需要安装并指定你想要处理的额外警告。但是在这种模式下，没有自动修复，只有代码检查和错误提示。

```Plain Text
Two modes: flake8-plugin and standalone
flake8-plugin: linting only
- flake8 --select=TOR0,TOR1,TOR2
standalone: linting and autofixing
- torchfix
- torchfix . --fix
- torchfix --select=ALL
```

另一种模式是独立运行，你可以将TorchFix作为脚本运行，并提供相应的参数。这张幻灯片的最后一行显示，并不是所有规则都默认启用。这是因为有些规则太过繁杂，不能默认启用。若要查看所有规则和结果，你可以提供SELECT等命令

```Plain Text
Get a release from PyPI
pip install torchfix

Latest code from GitHub Clone
https://github.com/pytorch/test-infra/tree/main/tools/torchfix
pip install . 
```

获取Torchfix也很简单，你只需要从PyPy安装Torchfix的最新版本，或者从GitHub克隆仓库并进行安装。目前的阶段是早期测试版（early beta），但已经非常有用。TorchFix已经拥有查找和修复我之前提到的所有示例的规则。它在Meta内部和开源项目中已经被使用来查找问题并更新代码。

```Plain Text
Beta version stage, already useful
- Rules for all the mentioned examples problems and more
- Was used to find issues and update code of multiple projects both internally at Meta and in open source
- Running in CI of several projects on GitHub
```

并且我们已经在几个GitHub上的元开源项目的CI中运行了它。将来我们希望为更多类别的问题添加更多的规则。而这项工作将根据我们在真实代码库中发现的实际问题进行引导。此外，我们还希望增加更多的配置选项。

目前TorchFix假设您使用最新版本的PyTorch，但实际上这未必是正确的。我们还希望将其与PyTorch CI和PyTorch文档生成集成。因此，例如，当您在PyTorch中弃用一个函数时，我们希望能够检查是否存在TorchFix的规则，以标记并更新弃用的函数...当然，我们还希望看到TorchFix在更多项目的CI中使用。希望这是有机地发生的，当人们尝试TorchFix并发现它很有用时。以及如何参与进来。首先，只需尝试在您的代码库上运行它。如果它发现任何问题，或者您可以发现TorchFix本身出现了一些问题。可以在github反馈。

```Plain Text
https://github.com/pytorch/test-infra/tree/main/tools/torchfix
- Bug reports, feature requests, and code contributions are very welcome
- There are open "good first issues" -searchfor[TorchFix]
```



