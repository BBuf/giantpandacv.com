> 【GiantPandaCV导语】本文介绍了量化感知训练的原理，并基于OneFlow实现了一个量化感知训练Demo，并介绍了在具体实现中的各种细节。希望对想学习量化感知训练的读者有用，本文仅做学习交流。

# 0x0. 前言
这篇文章主要是讲解一下量化感知训练的原理，以及基于OneFlow实现一个Demo级别的手动量化感知训练。

# 0x1. 后量化以及量化感知训练原理
这里说的量化一般都是指的Google TFLite的量化方案，对应的是Google 的论文 `Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference`。虽然TfLite这套量化方案并不是很难，但在实际处理的时候细节还是比较多，一时是很难说清楚的。

所以，这里推荐一系列讲解TFLite后量化和量化感知训练原理的文章，看一下这几篇文章阅读本文就没有任何问题了。

- [神经网络量化入门--基本原理](https://mp.weixin.qq.com/s/Jos9IcIlxsNezt9jZFjHOQ)
- [神经网络量化入门--后训练量化](https://mp.weixin.qq.com/s/sRvpzJjMdAyaxA6Gr6a8HA)
- [神经网络量化入门--量化感知训练](https://mp.weixin.qq.com/s/LiM4A182730ap_aqxbO4CQ)
- [神经网络量化入门--Folding BN ReLU代码实现](https://mp.weixin.qq.com/s/JVnEpErvyzEewfTahyP_RA)

这里我简单的总结一下，无论是TFLite的量化方案还是TensorRT的后量化方案，他们都会基于原始数据和量化数据的数值范围算出一个缩放系数`scale`和零点`zero_point`，这个`zero_point`有可能是0（对应对称量化），也有可能不是0（对应非对称量化）。然后原始数据缩放之后减掉零点就获得了量化后的数据。这里的关键就在于缩放系数`scale`和`zero_point`怎么求，Google的TFLite使用下面的公式：

$S = \frac{r_{max}-r_{min}}{q_{max}-q_{min}}$

$Z = round(q_{max} - \frac{r_{max}}{S})$

其中，$r$表示浮点实数，$q$表示量化后的定点整数，$r_{max}$和$r_{min}$分别是$r$的最大值和最小值，$q_{min}$和$q_{max}$表示$q$的最大值和最小值，如果是有符号8比特量化，那么$q_{min}=-128$，$q_{max}=127$，如果是无符号那么$q_{min}=0$，$q_{max}=255$。$S$就代表scale，$Z$就代表zero_point。

要求取`scale`和`zero_point`关键就是要精确的估计原始浮点实数的最大值和最小值，有了原始浮点实数的最大值和最小值就可以代入上面的公式求出`scale`和`zero_point`了。所以后训练量化以及量化感知训练的目的是要记录各个激活特征图和权重参数的`scale`和`zero_point`。

在后训练量化中，做法一般是使用一部分验证集来对网络做推理，在推理的过程中记录激活特征图以及权重参数的最大和最小值，进而计算`scale`和`zero_point`。而量化感知训练则在训练的过程中记录激活特征图和权重参数的最大和最小值来求取`scale`和`zero_point`。量化感知训练和后训练量化的主要区别在于它会对激活以及权重做模拟量化操作，即FP32->INT8->FP32。这样做的好处是可以模拟量化的实际运行过程，将量化过程中产生的误差也作为一个特征提供给网络学习，一般来说量化感知训练会获得比后训练量化更好的精度。

# 0x2. 组件
在上一节中主要提到了记录激活和权重的`scale`和`zero_point`，以及模拟量化，量化这些操作。这对应着三个量化训练中用到的三个基本组件，即`MinMaxObserver`，`FakeQuantization`，`Quantization`。下面我们分别看一下在OneFlow中这三个组件的实现。

## 组件1. MinMaxObserver

![OneFlow MinMaxObserver文档](https://img-blog.csdnimg.cn/70d3c0db0eec496daaeba7b3fb22b056.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

从这个文档我们可以看到MinMaxObserver操作被封装成`oneflow.nn.MinMaxObserver`这个Module（Module在Pytorch中对应`torch.nn.Module`，然后OneFlow的接口也在靠近Pytorch，也对应有`oneflow.nn.Module`，因此这里将其封装为`oneflow.nn.Module`）。这个Module的参数有：

- `quantization_bit`表示量化Bit数
- `quantization_scheme` 表示量化的方式，有对称量化`symmetric`和非对称量化`affine`两种，区别就是对称量化浮点0和量化空间中的0一致
- `quantization_formula` 表示量化的方案，有Google和Cambricon两种，Cambricon是中科寒武纪的意思
- `per_layer_quantization` 表示对当前的输入Tensor是PerChannel还是PerLayer量化，如果是PerLayer量化设置为True。一般激活特征图的量化都是PerLayer，而权重的量化可以选择PerLayer或者PerChannel。

下面看一下在Python层的用法：

```python
>>> import numpy as np
>>> import oneflow as flow

>>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)

>>> input_tensor = flow.Tensor(
...    weight, dtype=flow.float32
... )

>>> quantization_bit = 8
>>> quantization_scheme = "symmetric"
>>> quantization_formula = "google"
>>> per_layer_quantization = True

>>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)

>>> scale, zero_point = min_max_observer(
...    input_tensor, )
```

在设定好相关量化配置参数后，传入给定Tensor即可统计和计算出该设置下的Tensor的`scale`和`zero_point`。


上面讲的是Python前端的接口和用法，下面看一下在OneFlow中这个Module的具体实现，我们以CPU版本为例（GPU和CPU的Kernel实现是一致的），文件在`oneflow/user/kernels/min_max_observer_kernel.cpp`，核心实现是如下三个函数：

```cpp
// TFLite量化方案，对称量化
template<typename T>
void GenQuantScaleSymmetric(const T* in_ptr, const int32_t quantization_bit,
                            const int64_t num_elements, T* scale, T* zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  in_max = std::max(std::abs(in_max), std::abs(in_min));

  T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;

  *scale = in_max / denominator;
  *zero_point = 0;
}

// TFLite量化方案，非对称量化
template<typename T>
void GenQuantScaleAffine(const T* in_ptr, const int32_t quantization_bit,
                         const int64_t num_elements, T* scale, T* zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;

  *scale = (in_max - in_min) / denominator;
  *zero_point = -std::nearbyint(in_min / (*scale));
}

//寒武纪量化方案
template<typename T>
void GenQuantScaleCambricon(const T* in_ptr, const int32_t quantization_bit,
                            const int64_t num_elements, T* scale, T* zero_point) {
  T in_max = *std::max_element(in_ptr, in_ptr + num_elements);
  T in_min = *std::min_element(in_ptr, in_ptr + num_elements);

  in_max = std::max(std::abs(in_max), std::abs(in_min));

  *scale = std::floor(std::log2(in_max)) - (quantization_bit - 2);
  *zero_point = 0;
}
```

除了这三个函数之外，另外一个关键点就是对`per_layer_quantization`参数的处理了，逻辑如下：

![PerLayer量化或者PerChannel量化](https://img-blog.csdnimg.cn/6a1316932dd54fac992d079cfc9e47d9.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

如果是PerChannel量化则对每个输出通道求一个`scale`和`zero_point`。想了解更多PerLayer量化以及PerChannel量化的知识可以看这篇文章：[神经网络量化--per-channel量化](https://mp.weixin.qq.com/s/Sy6cO9yunv3f8RvbI_ZDvg) 。


## 组件2：FakeQuantization

![OneFlow FakeQuantization文档](https://img-blog.csdnimg.cn/52a51a7b2c554b259e256ea397aaf4f7.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

同样，FakeQuantization也被封装为一个`oneflow.nn.Module`。在上一节提到，量化感知训练和后训练量化的主要区别在于它会对激活以及权重参数做模拟量化操作，即FP32->INT8->FP32。通过这种模拟将量化过程中产生的误差也作为一个特征提供给网络学习，以期在实际量化部署时获得更好的准确率。这个接口有以下参数：

- `scale`：由MinMaxObserver组件算出来的量化`scale`
- `zero_point`：由MinMaxObserver组件算出来的量化`zero_point`
- `quantization_bit`： 量化比特数
- `quantization_scheme` 表示量化的方式，有对称量化`symmetric`和非对称量化`affine`两种，区别就是对称量化浮点0和量化空间中的0一致
- `quantization_formula` 表示量化的方案，有Google和Cambricon两种，Cambricon是中科寒武纪的意思

Python层的示例用法如下：

```python
>>> import numpy as np
>>> import oneflow as flow

>>> weight = (np.random.random((2, 3, 4, 5)) - 0.5).astype(np.float32)

>>> input_tensor = flow.Tensor(
...    weight, dtype=flow.float32
... )

>>> quantization_bit = 8
>>> quantization_scheme = "symmetric"
>>> quantization_formula = "google"
>>> per_layer_quantization = True

>>> min_max_observer = flow.nn.MinMaxObserver(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
... quantization_scheme=quantization_scheme, per_layer_quantization=per_layer_quantization)
>>> fake_quantization = flow.nn.FakeQuantization(quantization_formula=quantization_formula, quantization_bit=quantization_bit,
... quantization_scheme=quantization_scheme)

>>> scale, zero_point = min_max_observer(
...    input_tensor,
... )

>>> output_tensor = fake_quantization(
...    input_tensor,
...    scale,
...    zero_point,
... )
```

在执行FakeQuantizaton必须知道输入Tensor的`scale`和`zero_point`，这是由上面的MinMaxObserver组件获得的。

接下来看一下FakeQuantization组件C++层的实现，仍然有三个核心函数：

```c++
// TFLite量化方案，对称量化
template<typename T>
void FakeQuantizationPerLayerSymmetric(const T* in_ptr, const T scale,
                                       const int32_t quantization_bit, const int64_t num_elements,
                                       T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::nearbyint(in_ptr[i] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = out * scale;
  }
}

// TFLite量化方案，非对称量化
template<typename T>
void FakeQuantizationPerLayerAffine(const T* in_ptr, const T scale, const T zero_point,
                                    const int32_t quantization_bit, const int64_t num_elements,
                                    T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
  T lower_bound = 0;
  uint8_t zero_point_uint8 = static_cast<uint8_t>(std::round(zero_point));
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::nearbyint(in_ptr[i] / scale + zero_point_uint8);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = (out - zero_point_uint8) * scale;
  }
}
// 寒武纪量化方案
template<typename T>
void FakeQuantizationPerLayerCambricon(const T* in_ptr, const T shift,
                                       const int32_t quantization_bit, const int64_t num_elements,
                                       T* out_ptr) {
  T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
  T lower_bound = -upper_bound - 1;
  T scale = static_cast<T>(pow(2.0, static_cast<int32_t>(shift)));
  FOR_RANGE(int64_t, i, 0, num_elements) {
    T out = std::nearbyint(in_ptr[i] / scale);
    out = out > upper_bound ? upper_bound : out;
    out = out < lower_bound ? lower_bound : out;
    out_ptr[i] = out * scale;
  }
}
```

需要注意的一点是由于FakeQuantization要参与训练，所以我们要考虑梯度怎么计算？从上面的三个核心函数实现中我们可以发现里面都用了`std::nearbyint`函数，这个函数其实就对应numpy的`round`操作。而我们知道`round`函数中几乎每一处梯度都是0，所以如果网络中存在这个函数，反向传播的梯度也会变成0。

因此为了解决这个问题，引入了Straight Through Estimator。即直接把卷积层（这里以卷积层为例子，还包含全连接层等需要量化训练的层）的梯度回传到伪量化之前的`weight`上。这样一来，由于卷积中用的`weight`是经过伪量化操作的，因此可以模拟量化误差，把这些误差的梯度回传到原来的 `weight`，又可以更新权重，使其适应量化产生的误差，量化训练也可以正常运行。


具体的实现就非常简单了，直接将`dy`赋值给`dx`，在OneFlow中通过`identity`这个Op即可：

![给FakeQuantization注册梯度，直通估计器](https://img-blog.csdnimg.cn/c41a4360c081453690b6a740d81842bf.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


## 组件三：Quantization
上面的FakeQuantization实现了FP32->INT8->FP32的过程，这里还实现了一个Quantization组件备用。它和FakeQuantization的区别在于它没有INT8->FP32这个过程，直接输出定点的结果。所以这个组件的接口和C++代码实现和FakeQuantization基本完全一样（反向就不需要了），这里不再赘述。之所以要独立这个组件是为了在训练完模型之后可以将神经网络的权重直接以定点的方式存储下来。后面的Demo中将体现这一点。

# 0x3. 基于OneFlow量化感知训练AlexNet
下面以AlexNet为例，基于OneFlow的三个量化组件完成一个量化感知训练Demo。这里先贴一下实验结果：

![基于OneFlow的量化组件完成量化感知训练](https://img-blog.csdnimg.cn/a1b553da0b80448a9d7006ea84c0cb25.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

训练的数据集是ImageNet的一个子集，详细信息可以`https://github.com/Oneflow-Inc/models/pull/78`看到。在8Bit的时候无论是选用Google还是寒武纪，对称还是非对称，PerLayer还是PerChannel，量化感知训练后的模型精度没有明显降低。一旦将量化Bit数从8降到4，在相同的超参配置下精度有了明显下降。

下面分享一下这个基于OneFlow的量化感知训练Demo的做法：

首先代码结构如下：

```markdown
- quantization
	- quantization_ops 伪量化OP实现
	    - q_module.py 实现了Qparam类来管理伪量化参数和操作和QModule基类管理伪量化OP的实现
	    - conv.py 继承QModule基类，实现卷积的伪量化实现
	    - linear.py 继承QModule基类，实现全连接层的伪量化实现
	    - ...
	- models 量化模型实现
	    - q_alexnet.py 量化版AlexNet模型
	- quantization_aware_training.py 量化训练实现
	- quantization_infer.py 量化预测实现
	- train.sh 量化训练脚本
	- infer.sh 量化预测脚本
```

- 由于量化训练时需要先统计样本以及中间层的 `scale`、`zeropoint`，同时也频繁涉及到一些量化、反量化操作，所以实现一个QParam基类封装这些功能。
- 实现了一个量化基类`QModule`，提供了三个成员函数`__init__`，`freeze`。
  - `__init__`函数除了需要i指定`quantization_bit`，`quantization_scheme`，`quantization_formula`，`per_layer_quantization`参数外，还需要指定是否提供量化输入参数(`qi`) 及输出参数 (`qo`)。这是因为不是每一个网络模块都需要统计输入的 `scale`，`zero_point`，大部分中间层都是用上一层的`qo`来作为自己的`qi`，另外有些中间层的激活函数也是直接用上一层的 `qi`来作为自己的`qi`和`qo`。
  - `freeze` 这个函数会在统计完 `scale`，`zero_point` 后发挥作用，这个函数和后训练量化和模型转换有关。如下面的量化公式所示，其中很多项是可以提前计算好的，`freeze` 就是把这些项提前固定下来，同时也将网络的权重由**浮点实数**转化为**定点整数**。
  
  ![后量化公式，定点计算](https://img-blog.csdnimg.cn/7c96711ffbed43898850f63399d5c9e0.png)
  
- 基于这个`QModule`基类定义`QConv2d`，`QReLU`，`QConvBN`等等。

`QConvBN`表示Conv和BN融合后再模拟量化。原理可以看第一节的第4篇参考资料。这里以`QConv2d`为例看看它的实现：

```python
import oneflow as flow
from quantization_ops.q_module import QModule, QParam

__all__ = ["QConv2d"]


class QConv2d(QModule):

    def __init__(self, conv_module, qi=True, qo=True, quantization_bit=8, quantization_scheme='symmetric', quantization_formula='google', per_layer_quantization=True):
        super(QConv2d, self).__init__(qi=qi, qo=qo, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme,
                                      quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.quantization_bit = quantization_bit
        self.quantization_scheme = quantization_scheme
        self.quantization_formula = quantization_formula
        self.per_layer_quantization = per_layer_quantization
        self.conv_module = conv_module
        self.fake_quantization = flow.nn.FakeQuantization(
            quantization_formula=quantization_formula, quantization_bit=quantization_bit, quantization_scheme=quantization_scheme)
        self.qw = QParam(quantization_bit=quantization_bit, quantization_scheme=quantization_scheme,
                         quantization_formula=quantization_formula, per_layer_quantization=per_layer_quantization)
        self.quantization = flow.nn.Quantization(
            quantization_bit=32, quantization_scheme="affine", quantization_formula="google")

    def forward(self, x):
        if hasattr(self, 'qi'):
            self.qi.update(x)
            x = self.qi.fake_quantize_tensor(x)

        self.qw.update(self.conv_module.weight)

        x = flow.F.conv2d(x, self.qw.fake_quantize_tensor(self.conv_module.weight), self.conv_module.bias,
                          stride=self.conv_module.stride,
                          padding=self.conv_module.padding, dilation=self.conv_module.dilation,
                          groups=self.conv_module.groups)

        if hasattr(self, 'qo'):
            self.qo.update(x)
            x = self.qo.fake_quantize_tensor(x)

        return x

    def freeze(self, qi=None, qo=None):

        if hasattr(self, 'qi') and qi is not None:
            raise ValueError('qi has been provided in init function.')
        if not hasattr(self, 'qi') and qi is None:
            raise ValueError('qi is not existed, should be provided.')

        if hasattr(self, 'qo') and qo is not None:
            raise ValueError('qo has been provided in init function.')
        if not hasattr(self, 'qo') and qo is None:
            raise ValueError('qo is not existed, should be provided.')

        if qi is not None:
            self.qi = qi
        if qo is not None:
            self.qo = qo
        self.M = self.qw.scale.numpy() * self.qi.scale.numpy() / self.qo.scale.numpy()

        self.conv_module.weight = flow.nn.Parameter(
            self.qw.quantize_tensor(self.conv_module.weight) - self.qw.zero_point)
        self.conv_module.bias = flow.nn.Parameter(self.quantization(
            self.conv_module.bias, self.qi.scale * self.qw.scale, flow.Tensor([0])))

```


在QConv2d的`__init__.py`中，`conv_module`是原始的FP32的卷积module，其它的参数都是量化配置参数需要在定义模型的时候指定，`forward`函数模拟了FakeQuantization的过程，`freeze`函数则实现了冻结权重参数为定点的功能。其它的量化Module实现类似。

基于这些Module，我们可以定义AlexNet的量化版模型结构`https://github.com/Oneflow-Inc/models/blob/add_quantization_model/quantization/models/q_alexnet.py` ，完成量化感知训练以及模型参数定点固化等。

想完成完整的训练和测试可以直接访问：https://github.com/Oneflow-Inc/models 仓库。

# 0x4. 注意，上面的实现只是Demo级别的
查看了上面的具体实现之后，我们会发现最拉胯的问题是在量化模型的时候需要你手动去调整模型结构。其实不仅OneFlow的这个Demo是这样，在Pytorch1.8.0推出FX的量化方案之前（这里叫第一代量化方案吧）的第一代量化方案也是这样。这里放一段调研报告。

Pytorch第一代量化叫作Eager Mode Quantization，然后从1.8开始推出FX Graph Mode Quantization。Eager Mode Quantization需要用户手动更改模型，并手动指定需要融合的Op。FX Graph Mode Quantization解放了用户，一键自动量化，无需用户修改模型和关心内部操作。这个改动具体可以体现在下面的图中。

![Pytorch两个版本量化的区别](https://img-blog.csdnimg.cn/93a3113194374274965cfbe1578f33fc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


下面以一段代码为例解释一下Pytorch这两种量化方式的区别。

### Eager Mode Quantization

```python
class Net(nn.Module):

    def __init__(self, num_channels=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.fc = nn.Linear(5*5*40, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.reshape(-1, 5*5*40)
        x = self.fc(x)
        return x
```

Pytorch可以在Module的foward里面随意构造网络，可以调用Module，也可以调用Functional，甚至可以在里面写If这种控制逻辑。但这也带来了一个问题，就是比较难获取这个模型的图结构。因为在Eager Mode Quantization中，要量化这个网络必须做**手动**修改：

```python
class NetQuant(nn.Module):

    def __init__(self, num_channels=1):
        super(NetQuant, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 40, 3, 1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(40, 40, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5*5*40, 10)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = x.reshape(-1, 5*5*40)
        x = self.fc(x)
        x = self.dequant(x)
        return x
```

也就是说，除了`Conv`，`Linear`这些含有参数的Module外，`ReLU`，`MaxPool2d`也要在`__init__`中定义，Eager Mode Quantization才可以处理。

除了这一点，由于一些几点是要Fuse之后做量化比如Conv+ReLU，那么还需要手动指定这些层进行折叠，目前支持`ConV + BN、ConV + BN + ReLU、Conv + ReLU、Linear + ReLU、BN + ReLU`的折叠。

```python
model = NetQuant()model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
modules_to_fuse = [['conv1', 'relu1'], ['conv2', 'relu2']]  # 指定合并layer的名字
model_fused = torch.quantization.fuse_modules(model, modules_to_fuse)
model_prepared = torch.quantization.prepare(model_fused)
post_training_quantize(model_prepared, train_loader)   # 这一步是做后训练量化
model_int8 = torch.quantization.convert(model_prepared)
```

整个流程比较逆天，不知道有没有人用。

### FX Graph Mode Quantization

由于 FX 可以自动跟踪 forward 里面的代码，因此它是真正记录了网络里面的每个节点，在 fuse 和动态插入量化节点方面，比 Eager 模式更友好。对于前面那个模型代码，我们不需要对网络做修改，直接让 FX 帮我们自动修改网络即可：

```python
from torch.quantization import get_default_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
model = Net()  
qconfig = get_default_qconfig("fbgemm")
qconfig_dict = {"": qconfig}
model_prepared = prepare_fx(model, qconfig_dict)
post_training_quantize(model_prepared, train_loader)      # 这一步是做后训练量化
model_int8 = convert_fx(model_prepared)
```

### 理解
个人感觉基于OneFlow的Eager接口（OneFlow的Eager接口和Pytorch将完全对齐，用户可零成本迁移算法，并享受OneFlow在多机多卡上的速度红利）做量化感知训练也是要做到完全自动的。Pytorch FX的好处就在于它可以将一个Module通过插入一些Pass转化成一个类似的Module，只要开发者实现了某个Pass，就不需要用户操心了。OneFlow Eager版本的自动量化开发正在进行中（对于Layer版本，我们是支持一键自动量化训练的），敬请期待。打个广告，欢迎关注我司的OneFlow：**https://github.com/Oneflow-Inc/oneflow** 。


# 0x5. 总结
本文分享了笔者最近的一项工作，基于OneFlow Eager版本做量化感知训练，目前手动做量化感知训练对用户没有友好性。但对于想学习量化感知训练的读者来说，通过这个Demo来学习一些技巧还是不错的。另外，本文还调研了Pytorch FX的自动量化方案，它确实比Pytorch的第一代方案更友好，我们的目标也是做出更自动，更友好的量化训练接口。



# 0x6. 参考

- [神经网络量化入门--基本原理](https://mp.weixin.qq.com/s/Jos9IcIlxsNezt9jZFjHOQ)
- [神经网络量化入门--后训练量化](https://mp.weixin.qq.com/s/sRvpzJjMdAyaxA6Gr6a8HA)
- [神经网络量化入门--量化感知训练](https://mp.weixin.qq.com/s/LiM4A182730ap_aqxbO4CQ)
- [神经网络量化入门--Folding BN ReLU代码实现](https://mp.weixin.qq.com/s/JVnEpErvyzEewfTahyP_RA)

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)