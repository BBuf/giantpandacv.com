> 【GiantPandaCV导语】大家好，今天为大家介绍一下如何部署一个人脸106关键点模型到MsnhNet上，涉及到Caffe和Pytorch模型转换以及折叠BN简化网络以及如何编写MsnhNet预测代码等等。
# 1. 前言

之前，MsnhNet主要支持了将Pytorch模型转换为MsnhNet框架可以运行的模型文件（`*.msnhnet`和`*.bin`），并且我们在之前的[Pytorch转Msnhnet模型思路分享](https://mp.weixin.qq.com/s/gSffbAQf8CcOJkusjN09FA)文章中分享了这个转换的思路。

最近尝试了部署一个开源的人脸106点Caffe模型(`https://github.com/dog-qiuqiu/MobileNet-Yolo/tree/master/yoloface50k-landmark106`)到MsnhNet中，所以这篇文章就记录了我是如何将这个Caffe模型转换到MsnhNet并进行部署的。

# 2. 通用的转换思路
由于我们已经在Pytroch2Msnhnet这个过程上花费了比较大的精力，所以最直接的办法就是直接将Caffe模型转为Pytorch模型，然后调用已有的`Pytorch2Msnhnet`工具完成转换，这样是比较快捷省事的。

我参考`https://github.com/UltronAI/pytorch-caffe`这个工程里面的`caffe2pytorch`工具新增了一些上面提到的`yoloface50k-landmark106`关键点模型要用到的OP，如PReLU，nn.BatchNorm1D以及只有2个维度的Scale层等，比如Scale层重写为：

```python
class Scale(nn.Module):
    def __init__(self, channels):
        super(Scale, self).__init__()
        self.weight = Parameter(torch.Tensor(channels))
        self.bias = Parameter(torch.Tensor(channels))
        self.channels = channels
	# Python 有一个内置的函数叫 repr,它能把一个对象用字符串的形式表达出来以便辨认,这就是“字符串表示形式”
    def __repr__(self):
        return 'Scale(channels = %d)' % self.channels

    def forward(self, x):
		# landmark网络最后的全连接层后面接了Scale，所以需要考虑Scale层输入为2维的情况
        if x.dim() == 2:
            nB = x.size(0)
            nC = x.size(1)
            x = x * self.weight.view(1, nC).expand(nB, nC) + \
                self.bias.view(1, nC).expand(nB, nC)
        else:
            nB = x.size(0)
            nC = x.size(1)
            nH = x.size(2)
            nW = x.size(3)
            x = x * self.weight.view(1, nC, 1, 1).expand(nB, nC, nH, nW) + \
                self.bias.view(1, nC, 1, 1).expand(nB, nC, nH, nW)
        return x
```

可以看到这个Scale层Pytorch是不支持的，这是Caffe特有的层，所以这里写一个Scale类继承nn.Module来拼出一个Scale层。除了Scale层还有其它的很多层是这种做法，例如Eletwise层可以这样来拼：

```python
class Eltwise(nn.Module):
    def __init__(self, operation='+'):
        super(Eltwise, self).__init__()
        self.operation = operation

    def __repr__(self):
        return 'Eltwise %s' % self.operation

    def forward(self, *inputs):
        if self.operation == '+' or self.operation == 'SUM':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x + inputs[i]
        elif self.operation == '*' or self.operation == 'MUL':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x * inputs[i]
        elif self.operation == '/' or self.operation == 'DIV':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x = x / inputs[i]
        elif self.operation == 'MAX':
            x = inputs[0]
            for i in range(1,len(inputs)):
                x =torch.max(x, inputs[i])
        else:
            print('forward Eltwise, unknown operator')
        return x
```

介绍了如何在Pytorch中拼凑出Caffe的特有层之后，我们就可以对Caffe模型进行解析，然后利用解析后的层关键信息完成Caffe模型到Pytorch模型的转换了。解析Caffe模型的代码实现在`https://github.com/msnh2012/Msnhnet/blob/master/tools/caffe2Msnhnet/prototxt.py`文件，我们截出一个核心部分说明一下，更多细节读者可以亲自查看。

我们以一个卷积层为例，来理解一下这个Caffe模型中的`prototxt`解析函数：

```python
layer {
  name: "conv1_conv2d"
  type: "Convolution"
  bottom: "data"
  top: "conv1_conv2d"
  convolution_param {
    num_output: 8
    bias_term: false
    group: 1
    stride: 2
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
  }
}
```

解析`prototxt`文件的代码实现如下：

```python
def parse_prototxt(protofile):
	# caffe的每个layer以{}包起来
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        # 使用OrderedDict会根据放入元素的先后顺序进行排序，所以输出的值是排好序的
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                #print line
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if key in  block:
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # 获取块名，以卷积层为例返回[layer, convolution_param]
                key = line.split('{')[0].strip()
                # 递归
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            # 忽略注释
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    counter = 0
    while line:
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if key in  props:
               if type(props[key]) == list:
                   props[key].append(value)
               else:
                   props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1: # 获取块名，以卷积层为例返回[layer, convolution_param]
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props
```

然后解析CaffeModel比较简单，直接调用caffe提供的接口即可，代码实现如下：

```python
def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    print ('Loading caffemodel: '), caffemodel
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model
```


解析完Caffe模型之后，我们拿到了所有层的参数信息和权重，我们只需要将其对应放到Pytorch实现的网络层就可以了，这部分的代码实现就是`https://github.com/msnh2012/Msnhnet/blob/master/tools/caffe2Msnhnet/caffenet.py#L332`这里的`CaffeNet`类，就不需要再次解释了，仅仅是一个构件Pytorch模型并加载权重的过程。执行完这个过程之后我们就可以获得Caffe模型对应的Pytorch模型了。

# 3. 精简网络
为了让Pytorch模型转出来的MsnhNet模型推理更快，我们可以考虑在Caffe转到Pytorch模型时就精简一些网络层，比如常规的Convolution+BN+Scale可以融合为一个层。我们发现这里还存在一个FC+BN+Scale的结构，我们也可以一并融合了。这里可以再简单回顾一下原理。

## 3.1 融合BN原理介绍

**我们知道卷积层的计算可以表示为：**

$Y = W * X + B$

**然后BN层的计算可以表示为：**

$\mu = \frac{1}{m}\sum_{i=1}^mx_i$

$\sigma^2=\frac{1}{m}\sum_{i=1}^m(x_i-\mu)^2$

$x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+ \epsilon}}$

$y_i=\gamma * x_i + \beta$

**我们把二者组合一下，公式如下：**

$Y=\gamma*(\frac{(W*X+B)-\mu}{\sqrt{\sigma^2+\epsilon}})+\beta$

$Y=\frac{\gamma*W}{\sqrt{\sigma^2+\epsilon}}*X+\frac{\gamma*(B-\mu)}{\sqrt{\sigma^2+\epsilon}}+\beta$

然后令$a = \frac{\gamma}{\sqrt{\delta^2+\epsilon}}$

**那么，合并BN层后的卷积层的权重和偏置可以表示为：**

$W_{merged}=W*a$

$B_{merged}=(B-\mu)*a+\beta$


这个公式同样可以用于反卷积，全连接和BN+Scale的组合情况。

## 3.2 融合BN
基于上面的理论，我们可以在转Caffe模型之前就把BN融合掉，这样我们在MsnhNet上推理更快（另外一个原因是目前MsnhNet的图优化工具还在开发中，暂时不支持带BN+Scale层的融合）。融合的代码我放在`https://github.com/msnh2012/Msnhnet/blob/master/tools/caffe2Msnhnet/caffeOptimize/caffeOptimize.py`这里了，简要介绍如下：

![Caffe BN融合工具](https://img-blog.csdnimg.cn/20201117221133180.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


# 4. MsnhNet推理
精简网络之后我们就可以重新将没有BN的Caffe模型转到Pytorch再转到MsnhNet了，这部分的示例如下：


```python
# -*- coding: utf-8
# from pytorch2caffe import plot_graph, pytorch2caffe
import sys
import cv2
import caffe
import numpy as np
import os
from caffenet import *
import argparse
import torch
from PytorchToMsnhnet import *

################################################################################################   
parser = argparse.ArgumentParser(description='Convert Caffe model to MsnhNet model.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--height', type=int, default=None)
parser.add_argument('--width', type=int, default=None)
parser.add_argument('--channels', type=int, default=None)

args = parser.parse_args()

model_def = args.model
model_weights = args.weights
name = model_weights.split('/')[-1].split('.')[0]
width = args.width
height = args.height
channels = args.channels


net = CaffeNet(model_def, width=width, height=height, channels=channels)
net.load_weights(model_weights)
net.to('cpu')
net.eval()

input=torch.ones([1,channels,height,width])

model_name = name + ".msnhnet"

model_bin = name + ".msnhbin"

trans(net, input,model_name,model_bin)
```

获得了MsnhNet的模型文件之后，我们就可以使用MsnhNet进行推理了，推理部分的代码在`https://github.com/msnh2012/Msnhnet/blob/master/examples/landmark106/landmark106.cpp`。


我们来看看效果，随便拿一张人脸图片来测试一下：

![原图](https://img-blog.csdnimg.cn/20201117232402875.png#pic_center)


![结果图](https://img-blog.csdnimg.cn/20201117232336143.png#pic_center)


landmark的结果还是比较正确的，另外我们对比了Caffe/Pytorch/MsnhNet的每层特征值，Float32情况下相似度均为100%，证明我们的转换过程是正确的。


我们在**X86 CPU** `i7 10700F `上测一下速度，结果如下：

| 分辨率  | 线程数 | 时间  |
| ------- | ------ | ----- |
| 112x112 | 1      | 5ms   |
| 112x112 | 2      | 3.5ms |
| 112x112 | 4      | 2.7ms |

速度还是挺快的，由于本框架目前在x86没有太多优化，所以这个速度后面会越来越快的。感兴趣的读者也可以测试在其它平台上这个模型的速度。

# 5. 转换工具支持的OP和用法
## 5.1 介绍

Caffe2msnhnet工具首先将你的Caffe模型转换为Pytorch模型，然后调用Pytorch2msnhnet工具将Caffe模型转为`*.msnhnet`和`*.bin`。

## 5.2 依赖
- Pycaffe
- Pytorch 



## 5.3 计算图优化

- 在调用`caffe2msnhnet.py`之前建议使用caffeOPtimize文件夹中的`caffeOptimize.py`对原始的Caffe模型进行图优化，目前已支持的操作有：

- [x] Conv+BN+Scale 融合到 Conv
- [x] Deconv+BN+Scale 融合到Deconv
- [x] InnerProduct+BN+Scale 融合到InnerProduct

## 5.4 Caffe2Pytorch支持的OP
- [x] Convolution 转为 `nn.Conv2d`
- [x] Deconvolution 转为 `nn.ConvTranspose2d`
- [x] BatchNorm 转为 `nn.BatchNorm2d或者nn.BatchNorm1d`
- [x] Scale 转为 `乘/加`
- [x] ReLU 转为 `nn.ReLU`
- [x] LeakyReLU 转为 `nn.LeakyReLU`
- [x] PReLU 转为 `nn.PReLU`
- [x] Max Pooling 转为 `nn.MaxPool2d`
- [x] AVE Pooling 转为 `nn.AvgPool2d`
- [x] Eltwise 转为 `加/减/乘/除/torch.max`
- [x] InnerProduct 转为`nn.Linear`
- [x] Normalize 转为 `pow/sum/sqrt/加/乘/除`拼接
- [x] Permute 转为`torch.permute`
- [x] Flatten 转为`torch.view`
- [x] Reshape 转为`numpy.reshape/torch.from_numpy`拼接
- [x] Slice 转为`torch.index_select`
- [x] Concat 转为`torch.cat`
- [x] Crop 转为`torch.arange/torch.resize_`拼接
- [x] Softmax 转为`torch.nn.function.softmax`



# 5.5 Pytorch2Msnhnet支持的OP

- [x] conv2d
- [x] max_pool2d
- [x] avg_pool2d
- [x] adaptive_avg_pool2d
- [x] linear
- [x] flatten
- [x] dropout
- [x] batch_norm
- [x] interpolate(nearest, bilinear)
- [x] cat   
- [x] elu
- [x] selu
- [x] relu
- [x] relu6
- [x] leaky_relu
- [x] tanh
- [x] softmax
- [x] sigmoid
- [x] softplus
- [x] abs    
- [x] acos   
- [x] asin   
- [x] atan   
- [x] cos    
- [x] cosh   
- [x] sin    
- [x] sinh   
- [x] tan    
- [x] exp    
- [x] log    
- [x] log10  
- [x] mean
- [x] permute
- [x] view
- [x] contiguous
- [x] sqrt
- [x] pow
- [x] sum
- [x] pad
- [x] +|-|x|/|+=|-=|x=|/=|



## 5.6 使用方法举例
- `python caffe2msnhnet  --model  landmark106.prototxt --weights landmark106.caffemodel --height 112 --width 112 --channels 3 `，执行完之后会在当前目录下生成`lanmark106.msnhnet`和`landmark106.bin`文件。



# 6. 总结
至此，我们完成了`yoloface50k-landmark106`在MsnhNet上的模型转换和部署测试，如果对本框架感兴趣可以尝试部署自己的一个模型试试看，如果转换工具有问题请在github提出issue或者直接联系我们。点击阅读原文可以快速关注MsnhNet，这是我们业余开发的一个轻量级推理框架，如果对模型部署和算法优化感兴趣可以看看，我们也会在GiantPandaCV公众号分享我们的框架开发和算子优化相关的经历。

# 7. 参考
- https://github.com/UltronAI/pytorch-caffe
- https://github.com/msnh2012/Msnhnet

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)