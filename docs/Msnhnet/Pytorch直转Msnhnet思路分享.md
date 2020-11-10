# 0. 前言
这篇文章要为大家介绍一下MsnhNet的模型转换思路，大多数搞CV的小伙伴都知道：珍爱生命，原理模型转换。但是啊，当部署你训练出来的模型时，转换又是必不可少的步骤，这个时候就真的很愁啊，为什么调用xxx脚本转换出来的模型报错了，为什么转换出来的模型推理结果不正确呢？这个时候，我认为拥有一个清晰的分析思路是不可少的，这篇文章就为大家分享了以下最近起步的前向推理框架MsnhNet是如何将原始的Pytorch模型较为优雅的转换过来，希望这里的思路可以对有模型转换需求的同学带来一定启发。

# 1.参数的转换

- 利用pytorch的state_dict字典,可直接进行导出.
  由于msnhnet和pytorch的内存排布是一致的,都为NCHW模式,且对于BN层的参数顺序也相同,都为scale, bias, mean和var.只需将参数进行逐个提取,然后按二进制存储即可。

```python
import torchvision.models as models
import torch
from struct import pack

md = models.resnet18(pretrained = True)
md.to("cpu")
md.eval()
val = []
dd = 0

for name in md.state_dict():
        if "num_batches_tracked" not in name:
                c = md.state_dict()[name].data.flatten().numpy().tolist()
                dd = dd + len(c)
                print(name, ":", len(c))
                val.extend(c)

with open("alexnet.msnhbin","wb") as f:
    for i in val :
        f.write(pack('f',i))
```

注意上面出现了一行`if "num_batches_tracked" not in name:`，这一行是Pytorch的一个坑点，在pytorch 0.4.1及后面的版本里，BatchNorm层新增了num_batches_tracked参数，用来统计训练时的forward过的batch数目，源码如下（pytorch0.4.1）：

```python
  if self.training and self.track_running_stats:
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
```

在调用预训练参数模型时，官方给定的预训练模型是在pytorch0.4之前。因此，调用预训练参数时，需要过滤掉“num_batches_tracked”。

# 2.网络结构的转换

网络结构转换比较复杂,其原因在于涉及到不同的op以及相关的基础操作.

- **思路一**: 利用print的结果进行构建
    - **优点**: 简单易用
    - **缺点**: 大部分网络,print并不能完全展现出其结构.简单网络可用.

- 代码实现:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1   = nn.BatchNorm2d(6,eps=1e-5,momentum=0.1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16,eps=1e-5,momentum=0.1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y) 
        y = self.pool1(y) 
        y = self.conv2(y)
        y = self.bn2(y) 
        y = self.relu2(y) 
        y = self.pool2(y)
        return y

nn = Model()
print(nn)
```

- 结果: 很显然对于用纯nn.Module搭建的网络是可行的

```shell
Model
(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (bn1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

- 如果在forward内添加相关操作,则此方案将无效.
- 代码实现:

```python
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1   = nn.BatchNorm2d(6,eps=1e-5,momentum=0.1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2   = nn.BatchNorm2d(16,eps=1e-5,momentum=0.1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y) 
        y = self.pool1(y) 
        y = self.conv2(y)
        y = self.bn2(y) 
        y = self.relu2(y) 
        y = self.pool2(y)
        y = torch.flatten(y)
        return y

nn = Model()
print(nn)
```

- 结果: 很显然forward内的flatten操作并没有被导出.

```shell
Model(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (bn1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu1): ReLU()
  (pool1): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
  (bn2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu2): ReLU()
  (pool2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

- **思路二**: 通过类似Windows Hook技术.(思路来源[pytorch_to_caffe](https://github.com/wdd0225/pytorch2caffe/blob/master/pytorch_to_caffe.py))
  在pytorch的Op在执行之前,对此Op进行截取,以获取相关信息,从而实现网络构建.
    - 优点: 几乎可以完成所有pytorch的op导出.
    - 缺点: 实现复杂,容易误操作,可能影响pytorch本身结果错误.
- 代码实现: 通过构建Hook类, 重写op, 并替换原op操作,获取op的参数. 层的上下关系,通过tensor的`_cdata`作为唯一识别的ID.

```python
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

logMsg = True
ccc = []
# Hook截取类
class Hook(object):
    hookInited = False
    def __init__(self,raw,replace,**kwargs):
        self.obj=replace # 被截取之后的op
        self.raw=raw # 原op

    def __call__(self,*args,**kwargs):
        if not Hook.hookInited: #在Hook类未初始化之前,该信号原路返回
            return self.raw(*args,**kwargs)
        else:                   #否则,则按截取之后,实现的函数执行
            out=self.obj(self.raw,*args,**kwargs)
            return out

def log(*args):
    if logMsg:
        print(*args)

# 替换原cov2d函数的实现
def _conv2d(raw,inData, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # 对于上下层网络关系,可以使用tensor的_cdata,该参数类似唯一ID
    # 输入tensor的唯一ID
    log( "conv2d-i" , inData._cdata) 
    x=raw(inData,weight,bias,stride,padding,dilation,groups)
    ccc.append(x)                    # 此处将输出保存,防止被inplace操作,导致所有tensor的_cdata丧失唯一性
    # 此处就可以根据conv2d参数就行网络构建
    # msnhnet.buildConv2d(...) 
    # 输出tensor的唯一ID
    log( "conv2d-o" , x._cdata)
    return x

# 被替换OP                  原OP     自定义OP
F.conv2d        =   Hook(F.conv2d,_conv2d)
```

- 完整Demo:

```python
import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.functional as F

logMsg = True
ccc = []
# Hook截取类
class Hook(object):
    hookInited = False
    def __init__(self,raw,replace,**kwargs):
        self.obj=replace # 被截取之后的op
        self.raw=raw # 原op

    def __call__(self,*args,**kwargs):
        if not Hook.hookInited: #在Hook类未初始化之前,该信号原路返回
            return self.raw(*args,**kwargs)
        else:                   #否则,则按截取之后,实现的函数执行
            out=self.obj(self.raw,*args,**kwargs)
            return out

def log(*args):
    if logMsg:
        print(*args)

# 替换原cov2d函数的实现
def _conv2d(raw,inData, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    # 对于上下层网络关系,可以使用tensor的_cdata,该参数类似唯一ID
    # 输入tensor的唯一ID
    log( "conv2d-i" , inData._cdata) 
    x=raw(inData,weight,bias,stride,padding,dilation,groups)
    ccc.append(x)                    # 此处将输出保存,防止被inplace操作,导致所有tensor的_cdata丧失唯一性
    # 此处就可以根据conv2d参数就行网络构建
    # msnhnet.buildConv2d(...) 
    # 输出tensor的唯一ID
    log( "conv2d-o" , x._cdata)
    return x

def _relu(raw, inData, inplace=False):
    log( "relu-i" , inData._cdata)
    x = raw(inData,False)
    ccc.append(x)
    log( "relu-o" , x._cdata)
    return x

def _batch_norm(raw,inData, running_mean, running_var, weight=None, bias=None,training=False, momentum=0.1, eps=1e-5):
    log( "bn-i" , inData._cdata)
    x = raw(inData, running_mean, running_var, weight, bias, training, momentum, eps)
    ccc.append(x)
    log( "bn-o" , x._cdata)
    return x

def _flatten(raw,*args):
    log( "flatten-i" , args[0]._cdata)
    x=raw(*args)
    ccc.append(x)
    log( "flatten-o" , x._cdata)
    return x

# 被替换OP                   原OP       自定义OP
F.conv2d        =   Hook(F.conv2d,_conv2d)

F.batch_norm    =   Hook(F.batch_norm,_batch_norm)
F.relu          =   Hook(F.relu,_relu)
torch.flatten   =   Hook(torch.flatten,_flatten)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1   = nn.BatchNorm2d(6,eps=1e-5,momentum=0.1)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = self.relu1(y) 
        y = torch.flatten(y)
        return y

input_var = torch.autograd.Variable(torch.rand(1, 1, 28, 28))
nn = Model()
nn.eval()
Hook.hookInited = True
res = nn(input_var)
```

- 结果: flatten操作也完成了导出, 且每个op的input的ID都能在前面找到对应op的output的ID.即可知晓上下层之间的关系,由此,即可构建msnhnet.

```shell
conv2d-i 2748363239504
conv2d-o 2748363238224
bn-i 2748363238224
bn-o 2748363242832
relu-i 2748363242832
relu-o 2748363235152
flatten-i 2748363235152
flatten-o 2748363242064
```

# 3.详细转换过程代码编写


这里先截取一下构建MsnhNet的部分代码，完整代码见`https://github.com/msnh2012/Msnhnet/blob/master/tools/pytorch2Msnhnet/PytorchToMsnhnet.py`，如下：

```python
from collections import OrderedDict
import sys

class Msnhnet:
    def __init__(self):
        self.inAddr = ""
        self.net = ""
        self.index = 0
        self.names = []
        self.indexes = []

    def setNameAndIdx(self, name, ids):
        self.names.append(name)
        self.indexes.append(ids)

    def getIndexFromName(self,name):
        ids = self.indexes[self.names.index(name)]
        return ids

    def getLastVal(self):
        return self.indexes[-1]

    def getLastKey(self):
        return self.names[-1]

    def checkInput(self, inAddr,fun):

        if self.index == 0:
            return

        if str(inAddr._cdata) != self.getLastKey():
            try:
                ID = self.getIndexFromName(str(inAddr._cdata))
                self.buildRoute(str(inAddr._cdata),str(ID),False)
            except:
                 raise NotImplementedError("last op is not supported " + fun + str(inAddr._cdata))
            

    def buildConfig(self, inAddr, shape):
        self.inAddr = inAddr
        self.net = self.net + "config:\n"
        self.net = self.net + "  batch: " + str(int(shape[0])) + "\n"
        self.net = self.net + "  channels: " + str(int(shape[1])) + "\n"
        self.net = self.net + "  height: " + str(int(shape[2])) + "\n"
        self.net = self.net + "  width: " + str(int(shape[3])) + "\n"

 
    def buildConv2d(self, name, filters, kSizeX, kSizeY, paddingX, paddingY, strideX, strideY, dilationX, dilationY, groups, useBias):
        self.setNameAndIdx(name,self.index)
        self.net = self.net + "#" + str(self.index) +  "\n"
        self.index = self.index + 1
        self.net = self.net + "conv:\n"
        self.net = self.net + "  filters: " + str(int(filters)) + "\n"
        self.net = self.net + "  kSizeX: " + str(int(kSizeX)) + "\n"
        self.net = self.net + "  kSizeY: " + str(int(kSizeY)) + "\n"
        self.net = self.net + "  paddingX: " + str(int(paddingX)) + "\n"
        self.net = self.net + "  paddingY: " + str(int(paddingY)) + "\n"
        self.net = self.net + "  strideX: " + str(int(strideX)) + "\n"
        self.net = self.net + "  strideY: " + str(int(strideY)) + "\n"
        self.net = self.net + "  dilationX: " + str(int(dilationX)) + "\n"
        self.net = self.net + "  dilationY: " + str(int(dilationY)) + "\n"
        self.net = self.net + "  groups: " + str(int(groups)) + "\n"
        self.net = self.net + "  useBias: " + str(int(useBias)) + "\n"
```


然后Pytorch2MsnhNet就在前向传播的过程中按照我们介绍的Hook技术完成构建Pytorch模型对应的MsnhNet模型结构。

至此，我们就获得了MsnhNet的模型参数文件和权重文件，可以利用MsnhNet加载模型进行推理了。

# 4. 已经支持的OP以及转换实例

Pytorch2MsnhNet已经支持转换的OP如下：

```python
-  conv2d
-  max_pool2d
-  avg_pool2d
-  adaptive_avg_pool2d
-  linear
-  flatten
-  dropout
-  batch_norm
-  interpolate(nearest, bilinear)
-  cat   
-  elu
-  selu
-  relu
-  relu6
-  leaky_relu
-  tanh
-  softmax
-  sigmoid
-  softplus
-  abs    
-  acos   
-  asin   
-  atan   
-  cos    
-  cosh   
-  sin    
-  sinh   
-  tan    
-  exp    
-  log    
-  log10  
-  mean
-  permute
-  view
-  contiguous
-  sqrt
-  pow
-  sum
-  pad
-  +|-|x|/|+=|-=|x=|/=|
```


- ResNet18的转换示例：

```python
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PytorchToMsnhnet import *

resnet18=resnet18(pretrained=True)
resnet18.eval()
input=torch.ones([1,3,224,224])
trans(resnet18, input,"resnet18.msnhnet","resnet18.msnhbin")
```

- DeepLabV3的转换示例：

```python
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from PytorchToMsnhnet import *

deeplabv3=deeplabv3_resnet101(pretrained=False)
ccc = torch.load("C:/Users/msnh/.cache/torch/checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth")
del ccc["aux_classifier.0.weight"]
del ccc["aux_classifier.1.weight"]
del ccc["aux_classifier.1.bias"]
del ccc["aux_classifier.1.running_mean"]
del ccc["aux_classifier.1.running_var"]
del ccc["aux_classifier.1.num_batches_tracked"]
del ccc["aux_classifier.4.weight"]
del ccc["aux_classifier.4.bias"]
deeplabv3.load_state_dict(ccc)
deeplabv3.requires_grad_(False)
deeplabv3.eval()


input=torch.ones([1,3,224,224])

# trans msnhnet and msnhbin file
trans(deeplabv3, input,"deeplabv3.msnhnet","deeplabv3.msnhbin")
```


# 5. MsnhNet介绍
MsnhNet是一款基于纯c++的轻量级推理框架，此框架受到darknet启发，由穆士凝魂主导，并由本公众号作者团队业余协助开发。

项目地址：https://github.com/msnh2012/Msnhnet ，欢迎一键三连。

本框架目前已经支持了X86、Cuda、Arm端的推理（支持的OP有限，正努力开发中），并且可以直接将Pytorch模型（后面也会尝试接入更多框架）转为本框架的模型进行部署，欢迎对前向推理框架感兴趣的同学试用或者加入我们一起维护这个轮子。


最后，欢迎加入Msnhnet开发QQ交流群，有对项目的建议或者个人的需求都可以在群里或者github issue提出。


![交流群图片](https://img-blog.csdnimg.cn/20200909224818556.png#pic_center)

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)