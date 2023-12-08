> 我们推出了一个新的系列，对PytorchConference2023 的博客进行中文编译，会陆续在公众号发表。也可以访问下面的地址 https://www.aispacewalk.cn/docs/ai/framework/pytorch/PytorchConference2023/State_of_PyTorch 阅读。

## 大纲

1. PyTorch 2.0版本发布情况介绍

2. 2022年PyTorch三大重要事件

- 春季代码清理活动

- PyTorch基金会成立

- 四个新成员加入

1. PyTorch代码提交量及贡献者情况

- 提交次数及提交者

- 使用PyTorch的项目数量

- 研究文章数量

1. PyTorch社区现状

- Discuz论坛情况

1. 如何参与贡献PyTorch

## 详细要点

### 1. PyTorch 2.0发布情况

- 2千万次下载量

- 添加了多种新功能,如MPS、TorchScript、set_default_device等

### 2. 2022年三大事件

- 春季代码清理活动:27名参与者,合并45+ PRs

- 基金会成立第一年:6家新成员加入,覆盖范围更广

### 3. 代码提交&贡献者情况

- 12,000次提交,8%来自核心维护者之外

- 1,128名贡献者,较2021年增长10%

- 使用PyTorch的GitHub项目达60万个,增长20%

- 7,000多篇基于PyTorch的研究论文发表

### 4. PyTorch社区现状

- Discuz论坛每月2,000帖,40万浏览量

### 5. 如何参与贡献

- 回答问题、提建议

- 帮助调试复现问题

- 提交代码、进行Code Review

- 完善文档

- 参加文档马拉松活动



## 全文

我的名字是albin，今天在这个快速的闪电演讲中，我要给大家介绍一下我的torch的现状。我之前参加过这个会议，也许你们见过我做这个演讲的多个版本。我们每年都会进行这个演讲，我去年做过一次。我在PyTorch的核心库维护方面做了很多工作。今天我想谈论的是三个重要的PyTorch里程碑以及今年发生的事情中的三个重要事件。还有一些有趣的数字，Joe之前已经给大家展示过其中的一些，但能亲眼看到这些数据总是很有趣的。最后，我会给大家介绍一下如何参与pytorch以及如何帮助我们build pytorch。

```Plain Text
PyTorch 2.0
- 20M+ downloads
- Adds:
  torch.compile
  MPS backend for Apple M1/M2
  torch.func
  set default device
  Better transformer
```

第一个重要的PyTorch里程碑是今年早些时候发布的PyTorch 2.0版本。在所有平台上下载超过2000万次，所以对我们来说是一个相当重大的发布。下载数量仍在增加。它添加了一系列非常重要的功能。其中一个重要的功能是MPS后端（apple等）。现在处于测试阶段，在覆盖范围和稳定性上有了很大的提升，支持新功能。Torch.func，对于了解这一点的人来说，这是个非常实用的torch。它添加了一个函数式API。举个例子，Jax已经有了。它在PyTorch世界中保持一致，并与其他所有功能一起工作。还有一个set default device功能，你们中不知道的人可能不熟悉它，它可以改变构建PyTorch模型时使用的默认设备，例如通过直接在设备上进行初始化来显著加速初始化，或者如你在一些主题演讲例子中看到的那样，可以快速地创建模型。

你可以使用元设备，它是一个没有数据的虚拟设备，可以跳过模型的初始化过程，直接加载权重。这样就可以避免占用太多内存，进行不必要的初始化等，所有这些都会出现。最后是更好的transformer模型，这是许多人的共同努力的结果。PyTorch团队和许多维护者都在改进PyTorch中的transformer模型。无论是高级的api核心功能，还是低级的实现，都尽量使用最佳的实现。

另一个值得一提的重要里程碑是今年发生的春季代码清理活动，我们有27个参与者，合并了45多个pull request，关闭了53个问题。这对改进我们的教程仓库非常有帮助，提高质量增加了新的教程，并确保我们拥有最新的教程。最后，正如乔和易卜拉欣之前提到的，这是基金会成立的第一年。


![](https://files.mdnice.com/user/59/7ab83496-c1c8-4e27-8cc2-8be83ba8cb46.png)


在今年六月份，该组织对新成员开放了。所以能够看到我们已经有了这个事实，真的让人非常激动。在我写这篇演讲的时候，我们有四个新成员，现在又增加了两个，所以它的增长速度比我做幻灯片还快。我要说我对我们正在吸引的成员的多样性非常兴奋。正如IBM和Intel，你可以看到，底层组件、新的后端，但Graphcore的Hugging-facing则更高级和面向终端用户。看到我们在所有方面都在增长，对于基金会和生态系统来说，这是非常令人兴奋的。

现在来说几个数字。对于那些关注代码库的人来说，过去一年我们有12000次提交。我们非常高兴看到开源贡献增长了8%，我们指的是来自于通常核心维护人员以外的开源贡献者。


![](https://files.mdnice.com/user/59/68a80a5e-48e8-47b6-bf49-ecdcc628ab7b.png)


图表展示了每年我们在一个代码库中有多少次提交，如你所见，它正在不断增长，发生了越来越多的事情。

```Plain Text
The pytorch/pytorch top contributors
- 1128 Contributors this year (+10% more than last year
- Top oss Contributors
@peterbell10
@nkaretnikov
@XiaobingSuper
@cyyever
@lezcano
```

所有这些都是我们所有贡献者共同努力的结果。今年我们有1,128名贡献者参与其中。他们为该代码库做出了贡献。相比去年增长了10%，看到这么多人致力于这项工作真的非常令人兴奋。我在这里列出了一些做出了最多提交的人的名字，他们非常重要，因为有很多人只提交了少量的代码，但对于我们的贡献和代码改进却非常重要。所有这一切的努力使得我们在GitHub上有600,000多个使用PyTorch的项目，相比去年增长了20%。因此，看到越来越多的人实际使用PyTorch，并发布利用PyTorch的代码，真的非常令人兴奋。其中大部分来自于研究领域，PyTorch仍然推动着很多最新技术的发展。今年已经发表了7,000多个与GitHub代码库相关的研究论文，仍然有60%的人在使用PyTorch来进行AI研究实现。因此，看到一个非常开放和繁荣的研究生态系统真的非常令人兴奋。



![](https://files.mdnice.com/user/59/3911e67d-4834-45fe-a497-e2eccab36a73.png)


如今，PyTorch在行业方面也取得了显著的进展，之前一直滞后的情况正在迎头赶上。例如，根据LinkedIn的统计数据，我们看到每年有50%的人表示PyTorch是一项核心技能，他们希望学习更多，并且越来越多的工作也需要掌握PyTorch。这一切都得益于我们庞大的社区。我们在Disqus论坛上也有相关数据，这是用户、开发者和所有人互动的主要平台。每月约有400名新成员加入，并且浏览量超过200万次。这是一个非常活跃的网站，每月有约2000篇帖子。我们需要很多人来回答所有这些问题，这就是我呼吁大家行动起来的原因。加入我们吧！PyTorch是一个开源项目，这是我们所战斗的。有了这个基础，我们形成了一个庞大的社区。



![](https://files.mdnice.com/user/59/4c7b6245-9faf-4042-aae3-173116c2a722.png)


我们有了这个新的页面，它是PyTorch贡献的终极指南。

[https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)

请去查看一下。我会给你们讲一下如何参与其中。有很多非代码贡献。所以对PyTorch的贡献不仅仅是编写代码，还有很多其他方面的工作。


![](https://files.mdnice.com/user/59/f696253d-87d7-42b7-bb59-c5e47f511846.png)


在左边你们可以看到我之前提到的论坛。我们有一个DevDiscuss，还有一个Discuss论坛。发表问题，回答问题，并与社区讨论是非常重要的工作，任何人都可以参与。另外，我们正在为社区中的每个人构建PyTorch，所以报告问题、提出新功能的建议等都是你可以贡献的重要内容。比如说，我们正在寻找更多关于我们正在编写的所有功能的反馈。我相信对Torch Compile你们已经听说过很多了。您今天听到了很多其他功能。请给我们反馈，告诉我们有什么不起作用，什么工作得很好，以及您想看到什么。

```Plain Text
Code contributions 
- Reproduce and investigate issues
- Bugfix/FeaturePullReguests
- Review/maintain
- Documentationch
```

当然，还有代码贡献，正如你们中的许多人可能知道的那样。这里我认为第一点对人们特别有趣，因为...它不是你总是会考虑的事情，但帮助复现和调查问题是非常非常有帮助的一件事情。当我多年前为PyTorch做贡献时，我做的第一件事就是调试问题，并在论坛上回答问题。所以这非常有趣，并且也非常有帮助。因为一旦我们确切知道问题是什么，接下来的阶段就简单得多了，那就是修复错误和功能，发送拉取请求来修复问题。在PyTorch里，发送拉取请求的部分，在很多情况下实际上是最快的部分。找出需要做什么实际上才是最关键的。



第三点是，我鼓励任何感兴趣的人提交代码，同时帮助我们审查代码并维护Python代码库。现在基金会完全开放了，我们有来自不同地方的许多人帮助我们维护库的各个子集。所以我建议并鼓励任何有兴趣的人参与其中。关于文档的更改，我相信许多人都知道我们的文档网站和教程网站。我们正在更加努力地制作许多文档。对于这方面的任何贡献都是受欢迎的。至于这一点，对于那些还没看到的人，我们下个月将要举办一个PyTorch文档马拉松。如果你有兴趣与我们合作，学习新的技能并获得很多认可，去我们的博客文章看看。我们有一个公告博客文章，你可以在那里注册。



```Plain Text
Fall docathon: Nov 1st - 15 
- ImprovePyTorchdocumentation
- CollaboratewithPyTorchmaintainers
- Learn new skills
- Get recognition
Register now!
https://pytorch.org/blog/announcing-docathon-h2-2023/
```

