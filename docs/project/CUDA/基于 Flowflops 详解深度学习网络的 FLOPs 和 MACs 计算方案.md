# flowflops: OneFlow 模型的 Flops 计算

用于计算 OneFlow 模型的 FLOPs 和 Parameters 的第三方库。

源码地址(目前是private，我没有权限从private转到public): `https://github.com/Oneflow-Inc/flow-OpCounter`

## 介绍 & 使用

### FLOPs

有许多人分不清楚 FLOPs 和 MACs 之间的关系，如[ptflops中的issue](https://github.com/sovrasov/flops-counter.pytorch/issues/70)

针对该问题，可以查看[thop中的解释](https://github.com/Lyken17/pytorch-OpCounter/blob/master/benchmark/README.md)，翻译如下：

#### MACs, FLOPs, what is the difference?

`FLOPs` 是**浮动算子**(floating operations)的缩写，包括mul / add / div ...等。

`MACs` 代表执行的**乘法累加运算**，例如: `a <- a + (b x c)`。

如文中所示，一`MACs`有一`mul`和一`add`。这就是为什么在许多地方`FLOPs`几乎是两倍`MACs`的原因。

然而，现实世界中的应用要复杂得多。让我们考虑一个矩阵乘法示例。`A`是一个形状为 $m \times n$ 的矩阵，`B`是一个 $n \times 1$ 的向量。

```python
for i in range(m):
    for j in range(n):
        C[i][j] += A[i][j] * B[j] # one mul-add
```     

它会是m\*n个`MACs`和2m\*n个`FLOPs`。但是这样的矩阵乘法实现速度很慢，需要并行化才能运行得更快。

```python
for i in range(m):
    parallelfor j in range(n):
        d[j] = A[i][j] * B[j] # one mul
    C[i][j] = sum(d) # n adds
```

此时`MACs`数值不再是 m\*n 。

在比较 MAC / FLOP 时，我们希望数字与实现无关并且尽可能通用。因此在 (thop)[https://github.com/Lyken17/pytorch-OpCounter] 中，**我们只考虑乘法的次数**，而忽略所有其他操作。

### 安装方法

```shell
pip install flowflops
```

### 使用方法

目前支持两种 FLOPs 计算策略： 在 Eager 模式下计算和在 Graph 模式下计算。

> 在 Graph 模式下计算耗时较长，但结果更加精确

示例:

```python
import oneflow as flow
import flowvision.models as models
from flowflops import get_model_complexity_info


model = models.resnet50() # your own model, nn.Module
dsize = (1, 3, 224, 224)  # B, C, H, W

for mode in ["eager", "graph"]:
    print("====== {} ======".format(mode))
    total_flops, total_params = get_model_complexity_info(
        model, dsize,
        as_strings=False,
        print_per_layer_stat=False,
        mode=mode
    )
    print(total_flops, total_params)
```

输出:

```shell
====== eager ======
4121925096 25557032
====== graph ======
4127444456 25557032
```

可以看到两种计算方式下的输出有一定差别，这是因为在 ResNet 的 forward 代码里存在类似 `out += identity` 的语句，这会造成 FLOPs 额外增加。而在 Eager 模式下我们只关注在 `__init__()` 中定义的网络层，所以这种情况不会在 Eager 模式中被 hook 到。

我们可以计算一下有哪些 add_n 算子在 Eager 模式中被我们忽略了:

```shell
stage-one:   (1, 256, 56, 56)  * 3
stage-two:   (1, 512, 28, 28)  * 4
stage-three: (1, 1024, 14, 14) * 6
stage-four:  (1, 2048, 7, 7)   * 3
```

一共为 5,519,360 ，刚好为两种模式的输出差值 `4127444456 - 4121925096 = 5519360`

在 Eager 模式下也会存在一些小误差，一般认为 ResNet50 的 FLOPs 为 4.09G ，而这里计算得到 4.12G ，是因为一般研究中会忽略类似 ReLU 等算子的 FLOPs 计算，所以与真实数值会有一定误差。有关一般都忽略了哪些算子的计算，可以查看 `fvcore` 的输出。

```shell
Skipped operation aten::batch_norm 53 time(s)
Skipped operation aten::max_pool2d 1 time(s)
Skipped operation aten::add_ 16 time(s)
Skipped operation aten::adaptive_avg_pool2d 1 time(s)
FLOPs: 4089184256
```

> 在`ptflops`包中也存在这样的问题，笔者也有在issue中回复，详见issue: https://github.com/sovrasov/flops-counter.pytorch/issues/94

## Eager & Graph 模式下的 Flops 计算

接下来我们以简单修改后的 ResNet18 中的 BasicBlock 为例介绍一下两种 FLOPs 计算方式，设定网络如下：

> 我们统一假定输入形状为(1, 32, 64, 64)

```python
import oneflow as flow
import oneflow.nn as nn


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=True,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int = 32,
        planes: int = 32,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.fc = nn.Linear(planes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return self.fc(out)
```

### Eager

在 Eager 模式中，我们只关注 `__init__()` 中定义的网络层，也就是

```python
self.conv1 = conv3x3(inplanes, planes, stride)
self.bn1 = norm_layer(planes)
self.relu = nn.ReLU()
self.fc = nn.Linear(planes, planes)
```

#### 二维卷积

卷积的原理在此不再赘述，直接给出计算公式: $2 k^2 \times H_{out} \times W_{out} \times C_{in} \times C_{out} \div Groups$

#### 归一化

`batchnorm` 主要计算了均值、方差，并基于此对特征进行归一化与仿射变换，其 FLOPs 为 $2 \times C \times H \times W$

如果不进行仿射变换，则其 FLOPs 为 $C \times H \times W$

#### 激活函数

`relu` 对输入(1, C, H, W)进行了 `y = x if x > 0 else 0` 操作，也就是其 FLOPs 为 $C \times H \times W$

#### 线性层

线性层输入为 (N, C, H, W)

线性层权重为 (W1, W)

两者相乘的 FLOPs 为 $C \times H \times W1 \times W$

其本质与 `matmul` 计算相当

### Graph

在 Graph 模式中，我们会将 `flow.nn.Module` 编译为 `flow.nn.Graph` ，从 `Graph` 中抽取出每一个算子输入的张量形状后再对网络的 FLOPs 进行计算

上述网络转换后的 Graph:

```shell
GRAPH:MyGraph_0:MyGraph): (
  (CONFIG:config:GraphConfig(training=False, ))
  (INPUT:_MyGraph_0_input.0.0_2:tensor(..., size=(1, 32, 64, 64), dtype=oneflow.float32))
  (MODULE:model:BasicBlock()): (
    (INPUT:_model_input.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
    (MODULE:model.conv1:Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)): (
      (INPUT:_model.conv1_input.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
      (PARAMETER:model.conv1.weight:tensor(..., size=(32, 32, 3, 3), dtype=oneflow.float32, requires_grad=True)): ()
      (OPERATOR: model.conv1.weight() -> (out:sbp=(B), size=(32, 32, 3, 3), dtype=(oneflow.float32)):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.conv1-conv2d-0(_MyGraph_0_input.0.0_2/out:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32)), model.conv1.weight/out:(sbp=(B), size=(32, 32, 3, 3), dtype=(oneflow.float32))) -> (model.conv1-conv2d-0/out_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OUTPUT:_model.conv1_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
    )
    (MODULE:model.bn1:BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)): (
      (INPUT:_model.bn1_input.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
      (PARAMETER:model.bn1.weight:tensor(..., size=(32,), dtype=oneflow.float32, requires_grad=True)): ()
      (PARAMETER:model.bn1.bias:tensor(..., size=(32,), dtype=oneflow.float32, requires_grad=True)): ()
      (BUFFER:model.bn1.running_mean:tensor(..., size=(32,), dtype=oneflow.float32)): ()
      (BUFFER:model.bn1.running_var:tensor(..., size=(32,), dtype=oneflow.float32)): ()
      (BUFFER:model.bn1.num_batches_tracked:tensor(..., size=(), dtype=oneflow.int64)): ()
      (OPERATOR: model.bn1.running_mean() -> (out:sbp=(B), size=(32), dtype=(oneflow.float32)):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.bn1.running_var() -> (out:sbp=(B), size=(32), dtype=(oneflow.float32)):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.bn1.weight() -> (out:sbp=(B), size=(32), dtype=(oneflow.float32)):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.bn1.bias() -> (out:sbp=(B), size=(32), dtype=(oneflow.float32)):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.bn1-normalization-1(model.conv1-conv2d-0/out_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32)), model.bn1.running_mean/out:(sbp=(B), size=(32), dtype=(oneflow.float32)), model.bn1.running_var/out:(sbp=(B), size=(32), dtype=(oneflow.float32)), model.bn1.weight/out:(sbp=(B), size=(32), dtype=(oneflow.float32)), model.bn1.bias/out:(sbp=(B), size=(32), dtype=(oneflow.float32))) -> (model.bn1-normalization-1/y_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OUTPUT:_model.bn1_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
    )
    (MODULE:model.relu:ReLU()): (
      (INPUT:_model.relu_input.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
      (INPUT:_model.relu_input.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
      (OPERATOR: model.relu-relu-2(model.bn1-normalization-1/y_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))) -> (model.relu-relu-2/y_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OPERATOR: model.relu-relu-4(model-add_n-3/out_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))) -> (model.relu-relu-4/y_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))):placement=(oneflow.placement(type="cpu", ranks=[0])))
      (OUTPUT:_model.relu_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
      (OUTPUT:_model.relu_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
    )
    (OPERATOR: model-add_n-3([model.relu-relu-2/y_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32)), _MyGraph_0_input.0.0_2/out:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))]) -> (model-add_n-3/out_0:(sbp=(B), size=(1, 32, 64, 64), dtype=(oneflow.float32))):placement=(oneflow.placement(type="cpu", ranks=[0])))
    (OUTPUT:_model_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
  )
  (OPERATOR: _MyGraph_0_input.0.0_2(...) -> (...):placement=(oneflow.placement(type="cpu", ranks=[0])))
  (OPERATOR: _MyGraph_0_output.0.0_2(...) -> (...):placement=(oneflow.placement(type="cpu", ranks=[0])))
  (OUTPUT:_MyGraph_0_output.0.0_2:tensor(..., is_lazy='True', size=(1, 32, 64, 64), dtype=oneflow.float32))
)
```

`Graph` 中由 `OPERATOR` 开始的层就是我们所需要的信息

#### 卷积

在 `flow.nn.Graph` 中 `conv3x3` 和 `conv1x1` 会被拆解为 `conv2d + bias_add(if bias==True)`

由于我们只关注的卷积层的输入，而在计算 FLOPs 时需要得到卷积层输出的特征尺寸，所以我们需要依据输入计算输出特征的分辨率，方法如下

```python
output_dims = []
for i, in_dim in enumerate(in_dims):
    d = math.ceil((in_dim - kernel_size[i] + 2 * padding[i]) / strides[i]) + 1
    if (in_dim - kernel_size[i] + 2 * padding[i]) % strides[i] != 0:
        d -= 1
    output_dims.append(d)
```

随后即可正常计算 FLOPs

> 至于为什么不直接得到算子输出的形状，因为解析 Graph 需要占用更多的额外时间

#### 归一化

在 `flow.nn.Graph` 中 `norm_layer(bn)` 是一个单独的算子，其计算方法与 Eager 模式中保持一致

> 需要注意的是 InstanceNorm 和 GroupNorm 在 `flow.nn.Graph` 中将被拆解为若干胶水算子，需要逐个计算

#### 激活函数

在 `flow.nn.Graph` 中 `relu` 是一个单独的算子，其 FLOPs 计算方法与 Eager 模式中保持一致

#### 线性层

在 `flow.nn.Graph` 中 `linear` 会被拆解为 `matmul + broadcast_add(if bias==True)`，其 FLOPs 计算公式与 Eager 模式中基本一致

## 目前支持的 Op 与模型

目前该工具支持绝大部分算子、网络层与大多数 CNN ，列表如下

### Eager

```python
# convolutions
nn.Conv1d
nn.Conv2d
nn.Conv3d
# activations
nn.ReLU
nn.PReLU
nn.ELU
nn.LeakyReLU
nn.ReLU6
# poolings
nn.MaxPool1d
nn.AvgPool1d
nn.AvgPool2d
nn.MaxPool2d
nn.MaxPool3d
nn.AvgPool3d
# nn.AdaptiveMaxPool1d
nn.AdaptiveAvgPool1d
# nn.AdaptiveMaxPool2d
nn.AdaptiveAvgPool2d
# nn.AdaptiveMaxPool3d
nn.AdaptiveAvgPool3d
# BNs
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
# INs
nn.InstanceNorm1d
nn.InstanceNorm2d
nn.InstanceNorm3d
# FC
nn.Linear
# Upscale
nn.Upsample
# Deconvolution
nn.ConvTranspose1d
nn.ConvTranspose2d
nn.ConvTranspose3d
# RNN
nn.RNN
nn.GRU
nn.LSTM
nn.RNNCell
nn.LSTMCell
nn.GRUCell
```

### Graph

```
# conv
"conv1d"
"conv2d"
"conv3d"
# pool
"max_pool_1d"
"max_pool_2d"
"max_pool_3d"
"avg_pool_1d"
"avg_pool_2d"
"avg_pool_3d"
"adaptive_max_pool1d"
"adaptive_max_pool2d"
"adaptive_max_pool3d"
"adaptive_avg_pool1d"
"adaptive_avg_pool2d"
"adaptive_avg_pool3d"
# activate
"relu"
"leaky_relu"
"prelu"
"hardtanh"
"elu"
"silu"
"sigmoid"
"sigmoid_v2"
# add
"bias_add"
"add_n"
# matmul
"matmul"
"broadcast_matmul"
# norm
"normalization"
# scalar
"scalar_mul"
"scalar_add"
"scalar_sub"
"scalar_div"
# stats
"var"
# math
"sqrt"
"reduce_sum"
# broadcast
"broadcast_mul"
"broadcast_add"
"broadcast_sub"
"broadcast_div"
# empty
"reshape"
"ones_like"
"zero_like"
"flatten"
"concat"
"transpose"
"slice"
```

### FlowVision 中部分模型的计算结果

```
====== eager ======
+--------------------+----------+-------------+
|       Model        |  Params  |    FLOPs    |
+--------------------+----------+-------------+
|      alexnet       |  61.1 M  | 718.16 MMac |
|       vgg11        | 132.86 M |  7.63 GMac  |
|      vgg11_bn      | 132.87 M |  7.64 GMac  |
|   squeezenet1_0    |  1.25 M  | 830.05 MMac |
|   squeezenet1_1    |  1.24 M  | 355.86 MMac |
|      resnet18      | 11.69 M  |  1.82 GMac  |
|      resnet50      | 25.56 M  |  4.12 GMac  |
|  resnext50_32x4d   | 25.03 M  |  4.27 GMac  |
| shufflenet_v2_x0_5 |  1.37 M  |  43.65 MMac |
|   regnet_x_16gf    | 54.28 M  |  16.01 GMac |
|  efficientnet_b0   |  5.29 M  | 401.67 MMac |
|    densenet121     |  7.98 M  |  2.88 GMac  |
+--------------------+----------+-------------+
====== graph ======
+--------------------+----------+-------------+
|       Model        |  Params  |    FLOPs    |
+--------------------+----------+-------------+
|      alexnet       |  61.1 M  | 718.16 MMac |
|       vgg11        | 132.86 M |  7.63 GMac  |
|      vgg11_bn      | 132.87 M |  7.64 GMac  |
|   squeezenet1_0    |  1.25 M  | 830.05 MMac |
|   squeezenet1_1    |  1.24 M  | 355.86 MMac |
|      resnet18      | 11.69 M  |  1.82 GMac  |
|      resnet50      | 25.56 M  |  4.13 GMac  |
|  resnext50_32x4d   | 25.03 M  |  4.28 GMac  |
| shufflenet_v2_x0_5 |  1.37 M  |  43.7 MMac  |
|   regnet_x_16gf    | 54.28 M  |  16.02 GMac |
|  efficientnet_b0   |  5.29 M  | 410.35 MMac |
|    densenet121     |  7.98 M  |  2.88 GMac  |
+--------------------+----------+-------------+
```

## 总结

简单介绍 OneFlow 模型中如何计算网络 FLOPs
