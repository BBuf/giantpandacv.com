【GiantPandaCV导语】本文作为从零开始学深度学习编译器的番外篇，介绍了一下深度学习框架的Data Flow和Control Flow，并基于TensorFlow解释了TensorFlow是如何在静态图中实现Control Flow的。而对于动态图来说，是支持在Python层直接写Control Flow的，最后基于Pytorch介绍了如何将Python层的Control Flow导出到TorchScript模型以及ONNX模型。

# 0x0. 前言
本来是想在讲TVM Relay的时候提一下DataFlow和ControlFlow的，但是担心读者看到解析代码的文章打开就关了，所以这里用一篇简短的文章来介绍一下深度学习框架中的DataFlow和ControlFlow。

# 0x1. DataFlow
我记得我接触的第一个深度学习框架是TensorFlow1.x，本科毕业设计也是基于TensorFlow完成的，因此这里我将以TensorFlow1.x为例介绍一下DataFlow。

假设现在我们要实现一个$(a + b) * c$的逻辑，其中$a$，$b$，$c$都是一个简单的实数，然后我们如果用Python来实现非常简单，大概长这样：


```python
#coding=utf-8

import os

def cal(a, b, c):
    res = (a + b) * c
    print(res)
    return res

print(cal(1.0, 2.0, 3.0))
```

输出结果是9.0。然后我们使用tf1.31.1同样来实现这个过程：

```python
import tensorflow as tf

def cal(a, b, c):
    add_op = a + b
    print(add_op)
    mul_op = add_op * c

    init = tf.global_variables_initializer()
    sess = tf.Session()
    
    sess.run(init)
    mul_op_res = sess.run([mul_op])

    return mul_op_res

a = tf.constant(1.0)
b = tf.constant(2.0)
c = tf.constant(3.0)

print(cal(a, b, c))
```


同样代码的输出是9.0。然后这两个示例是为了解释像TensorFlow这种框架，它的计算图是一个计算流图，是由数据来驱动的。在上面的程序中我们可以发现如果打印`add_op`我们获得的结果是一个`Tensor`：

```python
Tensor("add:0", shape=(), dtype=float32
```

这是因为，TensorFlow1.x实现的这个计算函数首先在内存中构造了一个数据流图，长这样：

![上面tensorflow程序对应的数据流图](https://img-blog.csdnimg.cn/20210503225540217.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

我们回看一下Python的实现，实际上在执行`res = (a + b) * c`这行代码时，已经计算出了`res`的值，因为Python这种过程语言的数学计算是由代码驱动的。而TensorFlow不一样，它首先构造了数据流图，然后对这个计算流图进行绑定数据，让这个数据在这个图里面流起来，这是显示调用`sess.run`来获得输出的。

像TensorFlow这种基于数据流图（DataFlow）进行计算的深度学习框架不少，比如早期的Theano，2020年开源的国内深度学习框架**OneFlow**，PaddlePaddle1.x 初级版本都是基于数据流图的。当然更多人将其称为静态图。

# 0x2. Control Flow
这一节我将结合TensorFlow1.x的Control Flow来为大家解析一下Control Flow的难点以及TensorFlow的一些解决方案。这里的内容理解主要基于这篇博客(https://www.altoros.com/blog/logical-graphs-native-control-flow-operations-in-tensorflow/)，感兴趣的同学可以去查看原文。

在计算机科学中，控制流（Control Flow）定义了独立语句，指令，函数调用等执行或者求值的顺序。举个例子，我们要实现一个本机控制流，即我们需要根据函数A的输出值选择运行函数B或者C中的一个：


![一个Control Flow的例子](https://img-blog.csdnimg.cn/20210503231110291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后要实现这个控制流，最Naive的方式在是Python端写If/Else语句，即Python端的Control Flow，然后在不同条件下使用session.run()来求取不同分支的值。对于TensorFlow来说，大概是这样：


![这里获取A的值只是将其反馈回来](https://img-blog.csdnimg.cn/20210503231551238.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后这个Python层的Control Flow并不会在计算图中被表示出来，即：

![黄色部分在计算图中实际上是被删掉了，因为早期的TensorFlow无法表示这种控制逻辑](https://img-blog.csdnimg.cn/20210503231912396.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

我们可以看到上面的实现是比较烂的，这是因为我们使用`sess.run`对A进行求值之后，没做任何修改又放回了原始的计算图，而TensorFlow 计算图与 Python 交换数据频繁时会严重拖慢运算速度。除了性能的问题，在Python层做Control Flow，你会发现在计算图中并没有表示 Python 逻辑，如果你将 graph 导出，实际上是看不到这些 if/else 语句的，因此网络结构信息会丢失。

这个问题趟过Pytorch导出ONNX的读者应该知道，我们如果想导出一个完整的检测模型，带了NMS后处理那么必须找一张可以正常输出目标的图片作为输入。如果随机输出很可能后处理那部分在导出时就会被丢掉，这就是因为在Pytorch实现检测模型的时候在Python层用了If这种Control Flow。而Pytorch在导出ONNX模型时是根据输入跑一遍模型即tracing（这是以前的版本的做法，新版本的TensorFlow已经支持导出Python层的Control Flow），然后记录这个过程中发生了哪些操作。我们想一下，如果实现模型的过程中有Python层的Control Flow（基于tracing机制），那么必然有一部分节点会被丢弃。

Pytorch官方文档指出当我们导出ONNX的时候如果想导出Python层的控制流到计算图中就需要包一层`@jit.script`

![大概就是如果想在Pytorch里面导出含有Python层控制流的模型时导出ONNX会丢失控制流，如果需要保留建议导出TorchScript模型或者使用基于script模型的导出方式](https://img-blog.csdnimg.cn/20210503234126936.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

像Pytorch这种动态图框架可以方便的使用Python层的Control Flow，但TensorFlow在1.x时代为了解决这个问题是花费了不少努力的，即TensorFlow1.x的原生控制流。

## TensorFlow的原生控制流

TensorFlow提供了几个运算符用于原生控制流，如下：

![TensorFlow提供了几个运算符用于原生控制流](https://img-blog.csdnimg.cn/20210503234502335.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

那么使用这些原生控制流好处是什么呢？

- **高效**。TensorFlow 计算图与 Python 交换数据比较慢，计算图如果是端到端的，才能将数据传输开销降到最低，运行速度更快。
- **灵活**。静态计算图可以使用动态模块加强，计算图逻辑是自包含的。Pytorch目前比TensorFlow更受欢迎的主要原因就是前者为动态计算图，可以在运行时修改计算图。TensorFlow 利用控制流可以在一个静态定义的计算图中实现类似动态计算图的功能。
- **兼容**。通过 TensorBoard 调试和检查计算图，无缝通过 TensorFlow Serving 部署，也可以利用自动微分，队列和流水线机制。


## 控制依赖

TensorFlow会记录每一个运算符的依赖，然后基于依赖进行调度计算。也就是说一个运算符当且仅当它的依赖都完成之后才会执行一次。任何两个完成依赖的运算符可以以任意顺序进行。但这种设定可能会引发**竞争**，比如：

![控制依赖引发竞争](https://img-blog.csdnimg.cn/20210503235250622.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

其中 var 为一个变量，在对 bot 求值时，var 本身自增 2，然后将自增后的值返回。这时 top 语句执行顺序就会对 out 结果产生不同影响，结果不可预知。

为了解决这个问题，开发者可以人为的加入bot和top的依赖关系，让指定运算符先完成，如下图所示：

![人为的加入bot和top的依赖关系，让指定运算符先完成](https://img-blog.csdnimg.cn/20210503235445715.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

这里表示的就是如果我们需要保证读取的值是最新的，就需要新增下图中虚线箭头表示的依赖关系，即下图中上方蓝色圆圈依赖下方蓝色圆圈的运算完成，才能进行计算。

![加入依赖关系之后，计算图长这样](https://img-blog.csdnimg.cn/20210503235543690.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 条件分支

接下来看一下条件分支，即TensorFlow如何处理我们在这一节开头提出来的那个例子？


![TensorFlow提供了两个条件控制OP，即tf.cond和tf.case](https://img-blog.csdnimg.cn/20210504000008433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的代码中，利用了tf.cond实现条件分支，在 a < b 为真时，对 out 求值会执行 tf.add(3, 3)；否则，执行 tf.square(3)。


![使用tf.cond实现条件分支](https://img-blog.csdnimg.cn/20210504000127482.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

上面这段代码等价于：`tf.cond(a < b, lambda: tf.add(3, 3), lambda: tf.sqaure(3))`

然后生成的计算图如下所示：

![带有条件控制流的计算图](https://img-blog.csdnimg.cn/20210504000259314.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

当并列的分支比较多时，我们可以使用tf.case来处理，例如：

![并列的条件分支>2个时，使用tf.case来控制](https://img-blog.csdnimg.cn/20210504000513451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 循环

TensorFlow提供了`tf.while_loop`来构造循环块，感觉和RNN类似的结构有这个需求，例如：

![tf.while_loop可以实现循环控制流解决RNN这种计算图结构的控制逻辑](https://img-blog.csdnimg.cn/20210504000759882.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的代码实现了一个基础的循环例子，即循环100次。

![使用tf.while_loop在静态图中实现循环控制流](https://img-blog.csdnimg.cn/20210504000917874.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

总的来说，TensorFlow应该是首个将Control Flow引入到计算图中的深度学习框架，而不是像动态图框架那样直接在Python层去做Control Flow，这方面必须给予一定的尊重。即使Pytorch目前在学术界已经比TensorFlow更加流行，但基于TensorFlow演化的各种工业级项目仍然发挥着作用。

# 0x3. Pytorch中的Control Flow

在Pytorch这种动态图框架中，支持直接在Python端写Control Flow，并且可以将这些控制逻辑放到计算图中。这里以TorchScript为例，当我们尝试将Pytorch模型转为TorchScript时，有两种方式，一种是trace，另外一种是script。对于trace模式，适合Python层没有Control Flow的计算图，举例如下：


```python
#coding=utf-8
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
       super(MyModule,self).__init__()
       self.conv1 = nn.Conv2d(1,3,3)
    def forward(self,x):
       x = self.conv1(x)
       return x

model = MyModule()  # 实例化模型
trace_module = torch.jit.trace(model,torch.rand(1,1,224,224)) 
print(trace_module.code)  # 查看模型结构
output = trace_module (torch.ones(1, 1, 224, 224)) # 测试
print(output)
# trace_modult('model.pt') 
```

打印`trace_module `的代码可以看到：

```python
def forward(self,
    input: Tensor) -> Tensor:
  return (self.conv1).forward(input, )
```

而script模式则适用于计算图在Python层有Control Flow的情况，比如：

```python
#coding=utf-8
import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self):
        super(MyModule,self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(2,3,3)

    def forward(self,x):
        b,c,h,w = x.shape
        if c ==1:
            x = self.conv1(x)
        else:
            x = self.conv2(x)
        return x

model = MyModule()

# 这样写会报错，因为有控制流
# trace_module = torch.jit.trace(model,torch.rand(1,1,224,224)) 

# 此时应该用script方法
script_module = torch.jit.script(model) 
print(script_module.code)
output = script_module(torch.rand(1,1,224,224))

```

打印`script_module`的代码可以看到TorchScript模型包含了在上面Python层定义的Control Flow：

```python
def forward(self,
    x: Tensor) -> Tensor:
  b, c, h, w, = torch.size(x)
  if torch.eq(c, 1):
    x0 = (self.conv1).forward(x, )
  else:
    x0 = (self.conv2).forward(x, )
  return x0
```

然后我们来实验一下将上面带有Control Flow的Module导出ONNX，这里以Pytorch官方文档提供的一个带循环的Control Flow的示例为例：

```python
import torch

# Trace-based only

class LoopModel(torch.nn.Module):
    def forward(self, x, y):
        for i in range(y):
            x = x + i
        return x

model = LoopModel()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)

torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True)

```

这样就可以成功导出名字为`loop`的ONNX模型，使用Netron可视化软件打开看一下：


![可以看到直接导出Module，Python层的控制逻辑被丢掉（即for循环被完全展开），这是因为Pytorch在导出ONNX的时候默认使用了tracing机制](https://img-blog.csdnimg.cn/2021050423082599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

而当我们使用script模式时，导出的ONNX就会保留Python层的Control Flow并将其转换成ONNX中的Loop OP。示例代码以及Netron可视化结果如下：

```python
import torch
# Mixing tracing and scripting

@torch.jit.script
def loop(x, y):
    for i in range(int(y)):
        x = x + i
    return x

class LoopModel2(torch.nn.Module):
    def forward(self, x, y):
        return loop(x, y)

model = LoopModel2()
dummy_input = torch.ones(2, 3, dtype=torch.long)
loop_count = torch.tensor(5, dtype=torch.long)
torch.onnx.export(model, (dummy_input, loop_count), 'loop.onnx', verbose=True,
                  input_names=['input_data', 'loop_range'])
```


![Pytorch模型中在Python层定义的Control Flow被保留下来了](https://img-blog.csdnimg.cn/20210504231304791.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


# 0x4. 总结
这篇文章介绍了一下深度学习中的Data Flow和Control Flow，然后介绍了一下将Pytorch模型转为TorchScript的两种模式，并探索了要将Pytorch的Python层的Control Flow转换为ONNX应该怎么做。



# 0x5. 参考
- https://blog.csdn.net/lvxingzhe123456/article/details/82597095
- https://www.altoros.com/blog/logical-graphs-native-control-flow-operations-in-tensorflow/
- https://mp.weixin.qq.com/s/6uVeEHcQeaPN_qEhHvcEoA


-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)