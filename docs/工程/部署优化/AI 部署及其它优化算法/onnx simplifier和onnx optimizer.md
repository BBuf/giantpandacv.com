常用我的 onnx simplifier（简称 onnxsim） 的小伙伴可能知道，onnxsim 本身只提供 constant folding/propagation（即消除结果恒为常量的算子）的能力，而图变换（即合并 conv 和 bn 等等）的能力是由 onnxsim 调用 onnx optimizer 的各种 pass 实现的。constant folding 和图变换同时使用时，很多隐藏的优化机会会被挖掘出来，这也是 onnxsim 优化效果出色的原因之一。例如 add(add(x, 1), 2) 在变换为 add(x, add(1, 2)) 之后就可以通过 constant folding 变为 add(x, 3)，而 pad(conv(x, w, padding=0), add(1, 1)) 在经过 constant folding 变为 pad(conv(x, w, padding=0), 2) 后，就可以进一步融合成 conv(x, w, padding=2)。

然而，直到不久之前，onnxsim 用户还经常需要使用 --skip-optimization 参数来禁用 onnx optimizer 的图变换，否则就会遇到 segfault。这是因为当时 onnx optimizer 已经很久没有维护，积累了很多 bug。后来我和其他小伙伴一起接手了 onnx optimizer 的维护工作，接手之后它仍然在 onnx 这个 github orgnization 下，但移到了独立的仓库维护。

以往使用 onnx optimizer 的方式是

```python
import onnx
# ...
new_model = onnx.optimizer.optimize(model)
```

现在 onnx optimizer 在独立的仓库维护，有了自己的`onnxoptimizer`包：

```
import onnxoptimizer
# ...
new_model = onnxoptimizer.optimize(model)
```

而原`onnx`包里的`optimizer`部分在下一个版本就会删除掉。

目前`onnxoptimizer`已经修复了所有官方团队维护时期遗留的重要 bug，并且 ci 里已经包含了 torchvision 的 maskrcnn、faster-rcnn、deeplabv3 等等模型的测试，确保 onnx optimizer 之后始终可以正确处理这些经典模型。onnxsim 的 `--skip-optimization`参数已经几乎不再需要了，有了稳定的 onnx optimizer 加持， onnxsim 在很多网络上都可以取得令人满意的效果。例如，借助最新版的 onnx optimizer，onnxsim 可以完美的优化 PyTorch squeeze op 带来的冗余操作。具体来说，一段只包含 squeeze 操作的 PyTorch 代码 

```
class Net(nn.Module):
   def __init__(self):
      super().__init__()

   def forward(self, x):
      return torch.squeeze(x, dim=2)

net = Net()
torch.onnx.export(net, torch.ones(1,3,1,2), 'squeeze.onnx', opset_version=11)
```

导出的 onnx 模型如下图（netron 没有把模型结构显示完全，If node 里还包含了 true 和 false 两个未显示的子图，true 子图是一个 squeeze op，false 子图是一个 identity op）：


![导出onnx模型可视化](https://img-blog.csdnimg.cn/2021021623040820.png)

这个 onnx 模型这么复杂的原因是因为 onnx 的 squeeze op 和 pytorch squeeze op 的行为不完全一样：当 squeezed dim 那一维的长度不为 1 的时候 onnx squeeze op 会抛出错误，而 pytorch 则会让输出等于输入。这个复杂的 onnx 模型就是想和 pytorch 的行为对齐：先通过 Shape、Gather op 得到 dim 那一维的长度，再通过 Equal 和 If op 判断长度是不是 1，如果是 1 才运行 Squeeze op，否则运行 Identity op。

但是实际上对这个模型来说，这一大堆 op 都是没有必要的，因为输入形状是已知的 (1, 3, 1, 2)，squeeze dim 的长度是 1，所以一个普通的 onnx squeeze op 就足够了。

onnxsim 会先用 constant folding 优化掉 Shape、Gather、Equal op，变成下面的样子（图片里 "0" 游离在外面也是因为 netron 的显示问题，它实际上被 If node 里的子图使用，netron 没有显示出来）： 

![没有optimize前的ONNX可视化](https://img-blog.csdnimg.cn/20210216230450497.png)

此外 If node 的输入 cond 此时也已经是一个恒为 true 的常数。 

![If node 的输入 cond 此时也已经是一个恒为 true 的常数](https://img-blog.csdnimg.cn/20210216230529910.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

也就是说 If 一定会走到 true 这个分支。接着 onnxsim 会调用新版 onnx optimizer 里刚刚实现的消除死分支的 pass（相关的 pr 是 https://github.com/onnx/optimizer/pull/32 ），把这个模型里的 If op 删掉，把 true 分支提取出来，让这个复杂的 onnx 模型变成了它该有的样子：

![简化后的ONNX模型](https://img-blog.csdnimg.cn/20210216230552691.png)

如果小伙伴们想亲自尝试 onnxsim 的优化效果，按照 `https://github.com/daquexian/onnx-simplifier#python-version`的方法就可以安装和体验。

说句题外话：PyTorch 的一个简单的 squeeze 操作在 onnx 里却变得这么复杂，这样是好的吗？我觉得不是，这个复杂的模型会让用户一头雾水（https://github.com/pytorch/pytorch/issues/50687 ），而且对部署性能也有负面影响。那什么样是好的呢，给 onnx squeeze op 再额外增加一个和 PyTorch 对齐的模式是好的吗，或者甩开 onnx 另起炉灶是好的吗，我也不知道。