> GiantPandaCV导语：本文主要介绍笔者是如何在 Oneflow 框架中实现 expand 和 repeat 算子的，也希望通过这篇文章展示 Oneflow 框架的一些特色之处。

## expand 算子

### 用法介绍

```python
oneflow.expand(input, *sizes)
```

`expand` 算子的功能简单来说就是，能够实现将输入张量的沿着大小为 1 的维度进行复制，至于复制多少份由第二个参数决定（下面将该参数称为 `expand_size`）。

下面介绍 `expand_size` 设置的一些约定：

- `expand_size` 维度大小大于等于输入张量，如果大于输入维度则相当于输出会增加维度
- 对于输入张量为 `1` 的维度， `expand_size` 对应维度可以设置为大于等于 `1` 的值

- 对于输入张量不为 `1` 的维度， `expand_size` 对应维度只能设置为等于输入或者 `-1`  

- 新添加的维度只能加在开头且不能设置 `-1`，新增维度也就相当于将整个输入张量进行复制

#### 具体示例

示例1：

```python
input_shape = [4, 3, 1, 2]
exand_size  = [4, 3, 5, 2] 
# 下面这些 expand_size 的设置都是合法的
#             [-1, 3, 5, 2] 
#             [-1, -1, 5, 2] 
#             [-1, -1, 5, -1] 
#             [4, -1, 5, 2]
#             [4, -1, 5, -1]
#             [4, 3, 5, -1]

out_shape   = [4, 3, 5, 2]
```

示例2：

```python
input_shape =       [1, 4, 3, 5]
exand_size  = [2, 1, 2, 4, 3, 5] 
# 下面这些 expand_size 的设置都是合法的
#             [2, 1, 2, -1, 3, 5] 
#             [2, 1, 2, -1, -1, 5] 
#             [2, 1, 2, -1, -1, -1] 
#             [2, 1, 2, 4, -1, 5] 
#             [2, 1, 2, 4, -1, -1] 
#             [2, 1, 2, 4, 3, -1] 

out_shape   = [2, 1, 2, 4, 3, 5]
```

### 单卡视角实现思路

接下来介绍 `expand` 算子单卡视角下的实现思路，也就是先不用考虑分布式的情况。

从上一节的介绍可知 `expand`  算子的输出张量的某个位置的值就是从输入张量的某个位置复制过来的，所以问题就转化成了如何把输出某个位置的索引映射到输入对应位置的索引。

在介绍如何计算索引映射之前，首先来复习一下张量的 `stride ` 属性这个概念。对于内存连续的 `n` 维张量，可以通过其  `stride` 属性，快速定位到该张量任意位置的索引 `(x, y, z, k)` 对应的一维索引。

**举个例子：**

````python
input_shape = [6, 3, 4, 5]
stride      = [60, 20, 5, 1] # 下面会介绍 stide 的计算方法
input[x, y, z, k] == input_flatten[x * 60 + y * 20 + z * 5 + k * 1]
````

`stride` 每一维度的数值表示该维度索引每增加1，对应到内存上应该移动的步长，`stride` 每一维的计算公式如下：

$$
stride[i] = stride[i+1] \times input\_shape[i+1]
$$

**示例代码：**

```python
# 最后一维初始化为1
stride = [1]
# 从后往前生成 stride
for i in range(len(input_shape) - 2, -1, -1):
    # 在 stride 数组开头插入元素
    stride.insert(0, input_stride[0] * input_shape[i + 1])
```

接着来看该如何计算 `expand` 算子的输出索引到输入索引的映射。

我们知道如果输入张量某维度是 1，而 `expanbd_size` 对应的维度大于 1，相当于是将输入张量会沿着该维度进行复制。也就是对于复制的维度来说，不管该输出维度的索引是多少，都对应着输入张量该维度的索引 0。其实就是通过修改输入张量的 `stride` 参数构造一个新的 `output_stride` ，该 `output_stride` 的计算方法就是：

- 如果 `expand_size` 某维度 `i` 值为 `-1`，或者与输入张量的对应维度一致，则 `output_stirde[i] = stride[i]`
- 如果 `expand_size` 某维度 `i` 值大于 `1`，而输入张量对应维度为 1，则 `output_stride[i] = 0`
- 对于 `expand_size` 维度大于输入张量维度的情况，则对于新添加的维度 `i`，`output_stride[i] = 0`

**计算 `output_stirde` 的示例代码：**

```python
output_stride = []
diff = len(expand_size) - len(input_shape)
for i in range(len(expand_size) - 1, -1, -1):
    if i >= diff:
        if expand_size[i] == -1 or expand_size[i] == input_shape[i - diff]:
            output_stride.insert(0, input_stride[i - diff])
        else:
            assert expand_size[i] >= 1 and input_shape[i - diff] == 1
            output_stride.insert(0, 0)
    else:
        assert expand_size[i] >= 1
        output_stride.insert(0, 0)
```

**举个例子：**

```python
input_shape   =       [4, 1, 3, 5]
stride        =     [15, 15, 5, 1]
exand_size    = [2, 1, 4, 4, 3, 5] 
output_stride = [0, 0, 15, 0, 5, 1]
# 输出张量意位置的索引 (x, y, z, k, v, w)
output[x, y, z, k, v, w] = input_flatten[x * 0 + y * 0 + z * 15 + k * 0 + v * 5 + w * 1]
# 反向的计算逻辑
input_grad_flatten[x * 0 + y * 0 + z * 15 + k * 0 + v * 5 + w * 1] += output_grad[x, y, z, k, v, w]
```

前向代码链接: https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/expand_kernel.cu#L30

反向代码链接: https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/expand_kernel.cu#L43

### 多卡一致性视角

接下来介绍 Oneflow 中添加算子与其他框架不一样的地方。除了要正确实现单卡视角下的计算逻辑，还需要考虑多卡一致性视角下的逻辑，包括输出形状推理的逻辑、`sbp` 签名的设置和实际计算的逻辑。

首先简单介绍一致性视角的概念：

`
OneFlow 提出了一致性视角（consistent view）的概念，用于简化分布式训练。
简单而言，在 OneFlow 的一致性视角下，集群被抽象为一台“超级计算设备”。
用户无需关心集群中计算和通信的细节，只需要关心逻辑上的数据与计算。
可以像单机单卡那样去思考要和编程，就能进行分布式训练。
`

然后什么是 sbp:

`sbp 是 OneFlow 发明的概念，描述了在一致性视角下的
数据与集群中真实的物理设备上的数据的映射关系。
它由 split, broadcast, partial 的首字母组合而成。`

- `split` 

  表示真实物理设备上的张量，是将一致性视角的张量切分得到的。切分时需要指定切分的维度，而真实物理设备上的张量经过拼接之后可以还原得到一致性视角的张量。

- `broadcast` 

  表示一致性视角下的张量，在所有的真实物理设备上都有一份完整的复制。

- `partial` 

  表示一致性视角下的张量与物理设备上的张量的形状相同，但是对于物理设备上的值，只是一致性视角下张量的一部分。以 partial_sum 为例，如果我们将集群中所有设备的张量按位置相加，才可以还原得到一致性视角的张量。除了 sum 外，min、max 等操作也适用于 partial。

更多详细内容可以参考：https://docs.oneflow.org/v0.5.0/parallelism/02_sbp.html

所以在 Oneflow 中开发算子，开发者还需要为算子设置其输入和输出支持哪些 `sbp` 签名的组合，这也是需要付出的额外学习成本。 

而在一致性视角下，算子的实现逻辑有可能需要考虑，其在真实物理设备上的计算与逻辑上的计算（也就是一致性视角）不一致的地方。

比如对于 `expand` 算子，在真实物理设备上计算的时候，就可能需要修改用户传入的逻辑上的 `expand_size`。主要原因在于 `expand` 算子的 sbp 签名支持对输入进行 `split`

具体代码链接：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/ops/expand_op.cpp#L62

**举个具体的例子：**

 ```python
 logical input_shape =      [4, 3, 1, 2]
 logical stride =           [6, 2, 2, 1]
 logical expand_size =   [2, 4, 3, 4, 2]
 logical output_stride = [0, 6, 2, 0, 1]
 ```

假设用户设置了输入张量的 sbp 为 `split(3)` ，也就是对最后一维度进行切分。且设置该逻辑张量放置在两张卡上，则每张卡上的真实物理形状为：

```python
physical input_shape = [4, 3, 1, 1]
physical stride      = [3, 1, 1, 1]
```

则对于真实物理设备上的 `expand_size` 和 `output_stride` 都需要做修改：

```python
physical expand_size   =  [2, 4, 3, 4, 1]
physical output_stride =  [0, 3, 1, 0, 1]
```

**为什么 `expand_size` 需要修改呢？**

首先在一致性视角下，每个物理设备上进行实际计算的时候，实际上拿到的输入大小是切分之后的物理形状。

而对于上面的例子，输入的在每个设备上的物理形状变为 `[4, 3, 1, 1]`，而如果 `expand_size` 这时候仍然保持用户设置的逻辑大小 `[2, 4, 3, 4, 2]`，则在每个设备上的输出大小是 `[2, 4, 3, 4, 2]`，则输出对应的逻辑形状则是 `[2, 4, 3, 4, 4]`，则输出结果最后一维就比原来多了。

而由于用户怎么设置 sbp 是运行时才能拿到的信息，所以在物理设备上进行计算之前，都需要根据实际的输入大小，重新计算 `expand_size` 和 `output_stride`。

具体代码链接：https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/user/kernels/expand_kernel.cu#L129

## repeat 算子

### 用法介绍

```python
oneflow.repeat(input, *sizes)
```

`repeat` 算子的功能简单来说就是，能够实现将输入张量的任意维度进行复制，至于复制多少份由第二个参数决定（下面将该参数称为 `repeat_size`）。

下面介绍 `repeat_size` 设置的一些约定：

- `repeat_size` 维度大小大于等于输入张量，如果大于输入维度则相当于输出会增加维度
- `repeat_size` 任意维度的值需要设置为大于等于 `1` ，假设某维度设为 `n`， 则先当于输入张量对应维度复制 `n-1` 份

- 新添加的维度只能加在开头且不能设置为小于1的值，新增维度也就相当于将整个输入张量进行复制，假设新增维度设为 `n`， 则先当于将输入张量复制 `n-1` 份
- `repeat_size` 任意维度的值其实也可以设置为 `0`，但是这里不考虑这种情况

则输出张量每一维的大小计算方式如下：

对于非新增的维度：

$$
output\_shape[i] = input\_shape[i] * repeat\_size[i]
$$

对于新增的维度：

$$
output\_shape[i] = repeat\_size[i]
$$

#### 具体示例

```python
input_shape   =       [4, 1, 3, 5]
repeat_size   = [2, 1, 2, 4, 1, 1] 
output_shape  = [2, 1, 8, 4, 3, 5]
```

### 与 expand 算子的联系

其实仔细思考一下，可以感觉到 `repeat` 算子和 `expand` 算子其实是有联系的，也就是 `repeat` 算子是可以通过 `expand` 算子来实现。

**举些例子：**

例子1：

```python
input_shape   = [5]
repeat_size   = [3] 
output_shape  = [15]

# 等价与以下操作
input_shape            =    [5]
reshaped_input_shape   = [1, 5]
expand_size            = [3, 5] 
output_shape           = [3, 5]
reshaped_output_shape  =   [15]
```

例子2：

```python
input_shape   = [3, 1, 5]
repeat_size   = [5, 3, 1] 
output_shape  = [15, 3, 5]

# 等价于以下操作
input_shape           =    [3, 1, 5]
reshaped_input_shape  = [1, 3, 1, 5]
expand_size           = [5, 3, 3, 5] 
output_shape          = [5, 3, 3, 5]
reshaped_output_shape =   [15, 3, 5]
```

例子3：

```python
input_shape   =    [3, 1, 5]
repeat_size   = [2, 5, 3, 1] 
output_shape  = [2, 15, 3, 5]

# 等价与以下操作
input_shape            =       [3, 1, 5]
reshaped_input_shape   =    [1, 3, 1, 5]
expand_size            = [2, 5, 3, 3, 5] 
output_shape           = [2, 5, 3, 3, 5]
reshaped_output_shape	 =   [2, 15, 3, 5]
```

从上面的例子可以知道， `repeat` 操作可以用 `reshape` + `expand` + `reshape` 来代替，问题就转化成如何根据 `input_shape` 和 `repeat_size` 计算得到输入的 `reshape` 大小，`expand_size` 和输出的 `reshape` 大小。

**计算示例代码：**

```python
input_reshape = []
output_reshape = []
expand_size = []
diff = len(repeat_size) - len(input_shape)
for i in range(len(repeat_size) - 1, -1, -1):
    if i >= diff:
        if repeat_size[i] > 1:
            if input_shape[i - diff] > 1:
                input_reshape.insert(0, input_shape[i - diff])
                input_reshape.insert(0, 1)
                expand_size.insert(0, input_shape[i - diff])
                expand_size.insert(0, repeat_size[i])
                output_reshape.insert(0,
                        input_shape[i - diff] * repeat_size[i])
            else:
                input_reshape.insert(0, input_shape[i - diff])
                expand_size.insert(0, repeat_size[i])
                output_reshape.insert(0, repeat_size[i])
        else:
            input_reshape.insert(0, input_shape[i - diff])
            expand_size.insert(0, input_shape[i - diff])
            output_reshape.insert(0, input_shape[i - diff])
    else: # 新增的维度
        expand_size.insert(0, repeat_size[i])
        output_reshape.insert(0, repeat_size[i])
new_tensor = flow.reshape(input, input_reshape)
tmp_tensor = new_tensor.expand(*expand_size)
out = flow.reshape(tmp_tensor, output_reshape)
```

不过这算是取巧的实现了 `repeat` 算子，因为替换成了`reshape` 和 `expand` 算子来实现，所以也不用考虑 sbp 的问题了，不过后续为了性能还是需要单独写一个算子实现的。

## 参考资料

1. https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=expand#oneflow.expand
2. https://oneflow.readthedocs.io/en/master/oneflow.html?highlight=repeat#oneflow.repeat
3. https://oneflow.readthedocs.io/en/master/tensor.html?highlight=view#oneflow.Tensor.view
4. https://discuss.pytorch.org/t/contigious-vs-non-contigious-tensor/30107/2
5. https://docs.oneflow.org/v0.5.0/parallelism/02_sbp.html
6. https://docs.oneflow.org/v0.5.0/parallelism/01_introduction.html#_3
