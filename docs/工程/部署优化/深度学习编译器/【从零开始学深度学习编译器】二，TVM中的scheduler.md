# 0x0. 前言
在[【从零开始学深度学习编译器】一，深度学习编译器及TVM 介绍](https://mp.weixin.qq.com/s/sZLWjYebbHjCgQ6XAZCiOw)我们已经知道TVM可以将各种深度学习训练框架的模型（计算图）转化为内部的Graph IR（Relay），然后通过TVM提供的指令生成模块将Graph IR翻译成特定硬件可执行的指令或者代码。总的来说的TVM的思想可以总结为表示和调度分离，所谓表示就是IR，调度就是scheduler。同时，在高性能计算方面TVM提供了多种调度源语（**scheduler**），包含了大多数常见的优化手段如算子融合，读写缓存，分块计算，并行计算等等，这些计算方法都可以通过scheduler进行实现。所以这一节，我们就一起来探索一下TVM中的scheduler。


# 0x01. 介绍
我们知道TVM的核心就是自动代码生成技术，而scheduler则是自动代码生成技术的核心概念。scheduler我们可以简单理解为是一系列优化选择的集合，这些选择不会影响整个计算的结果，但对计算的性能却至关重要。一个常见的例子是**矩阵乘法**，给定输入矩阵A和B，维度分别为$[m, k]$和$[k, n]$，然后获得结果矩阵C，维度为$[m, n]$，我在之前的[道阻且长_再探矩阵乘法优化](https://mp.weixin.qq.com/s/w0YCm8TEPxFg0CR6g4A28w) 详细列出了为了加速这个计算所采用的一系列优化方法，注意这里是以Arm端为例。具体如下：

| 文件名                                      | 优化方法                                                     | gFLOPs     | 峰值占比 | 线程数 |
| ------------------------------------------- | ------------------------------------------------------------ | ---------- | -------- | ------ |
| MMult1.h                                    | 无任何优化                                                   | 0.24gflops | 2.1%     | 1      |
| MMult2.h                                    | 一次计算4个元素                                              | 0.24gflops | 2.1%     | 1      |
| MMult_1x4_3.h                               | 一次计算4个元素                                              | 0.24gflops | 2.1%     | 1      |
| MMult_1x4_4.h                               | 一次计算4个元素                                              | 0.24gflops | 2.1%     | 1      |
| MMult_1x4_5.h                               | 一次计算4个元素(将4个循环合并为1个)                          | 0.25gflops | 2.2%     | 1      |
| MMult_1x4_7.h                               | 一次计算4个元素(我们在寄存器中累加C的元素，并对a的元素使用寄存器),用指针来寻址B中的元素 | 0.98gflops | 9.0%     | 1      |
| MMult_1x4_8.h                               | 在MMult_1x4_7的基础上循环展开四个（展开因子的相对任意选择）  | 1.1gflops  | 10%      | 1      |
| MMult_4x4_3.h                               | 一次计算C中的4x4小块                                         | 0.24gflops | 2.1%     | 1      |
| MMult_4x4_4.h                               | 一次计算C中的4x4小块                                         | 0.24gflops | 2.1%     | 1      |
| MMult_4x4_5.h                               | 一次计算C中的4x4小块,将16个循环合并一个                      | 0.25gflops | 2.2%     | 1      |
| MMult_4x4_6.h                               | 一次计算C中的4x4小块(我们在寄存器中累加C的元素，并对a的元素使用寄存器) | 1.75gflops | 16.0%    | 1      |
| MMult_4x4_7.h                               | 在MMult_4x4_6的基础上用指针来寻址B中的元素                   | 1.75gflops | 16.0%    | 1      |
| MMult_4x4_8.h                               | 使用更多的寄存器                                             | 1.75gflops | 16.0%    | 1      |
| MMult_4x4_10.h                              | NEON指令集优化                                               | 2.6gflops  | 23.8%    | 1      |
| MMult_4x4_11.h                              | NEON指令集优化, 并且为了保持较小问题规模所获得的性能，我们分块矩阵C（以及相应的A和B） | 2.6gflops  | 23.8%    | 1      |
| MMult_4x4_13.h                              | NEON指令集优化, 对矩阵A和B进行Pack，这样就可以连续访问内存   | 2.6gflops  | 23.8%    | 1      |
| MMult_4x4_18.h                              | Neon Assembly，Cache优化                                     | 3.0gflops  | 27.5%    | 1      |
| MMult_4x4_19.h                              | MMult_4x4_18基础上+更长的pld+ldd+指令重排                    | 3.8gflops  | 34.59%   | 1      |
| MMult_4x4_20.h                              | MMult_4x4_19基础上更换vldr + 简单调整ping pong               | 4.0gflops  | 36.7%    | 1      |
| conv1x1s1.h（version1）                     | 一次计算多行，neon汇编优化                                   | 3.4gflops  | 31.0%    | 1      |
| conv1x1s1.h（version2）                     | pack，kernel提前做，neon汇编优化，8x4分块                    | 4.9gflops  | 45%      | 1      |
| conv1x1s1.h（version3）                     | pack，kernel提前做，输入NC4HW4，neon汇编优化，8x4分块        | 5.5gflops  | 50.5%    | 1      |
| conv1x1s1.h（version4） idea from megengine | pack，kernel提前做，输入NC4HW4，neon汇编优化，12x4分块       | 5.2gflops  | 47.8%    | 1      |

可以看到虽然这些实现都完成了矩阵乘法这个计算任务，也就是说输入输出都是完全相同的，但在计算过程中却使用了**一系列不同的优化手段**，这些优化算法的集合就可以统称为**scheduler**。

接下来我们明确一下**scheduler**在整个TVM软件栈中的位置，最近一直在找这样一张图，然后OpenMMLab最新放出的介绍Ansor文章里的一张图刚好能完美解释这个问题，这里我就抄过来了。以深度学习中一个常见的MatMul+Add+Relu计算图为例，看一下TVM做代码生成的一个过程。首先TVM将接受的计算图转换为TVM中的领域特定语言Tensor Expression，即图中的黄色部分。接下来用户可以手动指定计算策略即scheduler，然后TVM会自动生成特定后端的代码，注意图中的tiling和binding分别代表拆分和绑定的意思，也是scheduler。我们现在明确了scheduler在TVM软件栈中的位置，也应该清楚TVM能否产生高性能的代码关键就在于scheduler是否指定合理，即优化算法在指定后端是否work and efiicient。


![TVM代码生成过程，图源OpenMMLab](https://img-blog.csdnimg.cn/20210327110934662.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 0x02. 从Tensor Expression开始看TVM是如何生成CUDA代码的

我们以chentianqi大佬在TVM文档中的介绍Tensor Expression例子初步感受一下上面那张图中描述的TVM代码生成过程，这里面也包含了scheduler。这一节之后我们再列举一些其它的例子来感受scheduler的更多用法。现在我们从Tensor Expression开始，看看TVM是如何生成代码的，以及我们具体是如何指定scheduler的。首先导入一堆要用到的包。

```python
import tvm
import tvm.testing
from tvm import te
import numpy as np

# 全局环境定义

tgt_host = "llvm"
# 如果启用了GPU，则将其更改为相应的GPU，例如：cuda、opencl、rocm
tgt = "cuda"
```

然后使用向量加法来演示TVM的工作流程。作为第一步，我们需要描述我们的计算。TVM采用Tensor Expression，每个中间结果表示为一个多维数组。用户需要描述生成张量的计算规则。我们首先定义一个符号变量n来表示形状。然后我们定义两个占位符张量，A和B，具有给定的形状$(n, )$。然后我们用一个计算函数来描述结果张量C。计算函数采用张量的形式，以及描述张量每个位置的计算规则的lambda函数。在这个阶段没有计算发生，因为我们只是声明应该如何进行计算。代码如下：

```python
n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
print(type(C))
```

打印出的信息为：`<class 'tvm.te.tensor.Tensor'>`

接着，虽然上面的几行描述了计算规则，但是我们可以用很多方法来计算C，因为C可以在轴上用数据并行的方式来计算。TVM要求用户提供一个称为schedule的计算描述，即等效于下面的代码：

```sh
for (int i = 0; i < n; ++i) {
  C[i] = A[i] + B[i];
}
```

我们调用`te.create_schedule`来创建scheduler，然后使用split构造来拆分C的轴，这将把原来的一个迭代轴拆分成两个迭代轴的乘积，代码如下：

```python
s = te.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)
```

这等效于下面的代码：

```powershell
for (int bx = 0; bx < ceil(n / 64); ++bx) {
  for (int tx = 0; tx < 64; ++tx) {
    int i = bx * 64 + tx;
    if (i < n) {
      C[i] = A[i] + B[i];
    }
  }
}
```

最后，我们将迭代轴bx和tx绑定到GPU计算grid中的线程。这些是特定于GPU的构造，允许我们生成在GPU上运行的代码。 

```python
if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
```

上面我们已经完成了指定scheduler，接下来我们就可以将上面的所有代码编译成一个TVM的函数了。默认情况下，TVM会将其编译成一个类型擦除函数，可以直接从Python端调用。下面我们使用`tvm,build`来创建一个编译函数，编译函数接收scheduler，函数签名（包含输入输出）以及我们需要编译到的目标语言。编译`fadd`的结果是一个GPU设备函数（如果涉及GPU）以及一个调用GPU函数的host端包装器。`fadd`是生成的host端包装函数，它在内部包含对生成的设备函数的引用。代码如下：

```python
fadd = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name="myadd")
```

编译后的TVM函数生成了一个简洁的C API，可以被任何语言调用。TVM在python中提供了一个最小的array API来帮助快速测试和原型开发。array API基于DLPack（https://github.com/dmlc/dlpack）标准。要运行这个函数，首先需要创建一个GPU context，然后使用`tvm.nd.array`将数据拷贝到GPU，再使用我们编译好的函数`fadd`来执行计算，最后再用`asnumpy()`将GPU端的array拷贝回CPU使用numpy进行计算，最后比较两者计算结果的差距。这部分的代码如下：

```python
ctx = tvm.context(tgt, 0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)
tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
```

到了这里整个计算过程就已经完成了，但是我们相信大家一定对TVM生成的代码长什么样子非常感兴趣，TVM也提供了对应的接口来让用户查看生成的代码。`tvm.build`的结果是一个TVM Module。`fadd`是包含host包装器的模块，同时它也包含了用于CUDA（GPU）设备的功能模块。 我们将使用下面的代码打印生成的代码：

```python
if tgt == "cuda" or tgt == "rocm" or tgt.startswith("opencl"):
    dev_module = fadd.imported_modules[0]
    print("-----GPU code-----")
    print(dev_module.get_source())
else:
    print(fadd.get_source())
```

输出为：

```powershell
-----GPU code-----

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long
  #define uint64_t ulong
#endif
extern "C" __global__ void myadd_kernel0(float* __restrict__ C, float* __restrict__ A, float* __restrict__ B, int n, int stride, int stride1, int stride2) {
  if (((int)blockIdx.x) < (n >> 6)) {
    C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
  } else {
    if (((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) < n) {
      C[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride2))] = (A[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride))] + B[((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) * stride1))]);
    }
  }
}
```

好了，讲到这里，我们就知道如何在TVM中定义scheduler并自动生成计算代码了。


# 0x03. scheduler更详细的例子

## **split**
关于scheduler更详细的例子可以看大神的这篇文章：https://zhuanlan.zhihu.com/p/94846767。我们这里简单列举几个来理解一下，例如在循环优化中，我们以split为例，代码如下：

```python
import tvm
from tvm import te

n = 1024
A = te.placeholder((n,), name='A')
k = te.reduce_axis((0, n), name='k')

B = te.compute((1,), lambda i: te.sum(A[k], axis=k), name='B')

s = te.create_schedule(B.op)

print(tvm.lower(s, [A, B], simple_mode=True))
print("---------cutting line---------")

ko, ki = s[B].split(B.op.reduce_axis[0], factor=32)

print(tvm.lower(s, [A, B], simple_mode=True))
```

生成的函数为：

```powershell
primfn(A_1: handle, B_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B} {
  B_2[0] = 0f32
  for (k: int32, 0, 1024) {
    B_2[0] = ((float32*)B_2[0] + (float32*)A_2[k])
  }
}


---------cutting line---------
primfn(A_1: handle, B_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [1], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024], [])}
  buffer_map = {A_1: A, B_1: B} {
  B_2[0] = 0f32
  for (k.outer: int32, 0, 32) {
    for (k.inner: int32, 0, 32) {
      B_2[0] = ((float32*)B_2[0] + (float32*)A_2[((k.outer*32) + k.inner)])
    }
  }
}
```

可以看到split把iter以factor为间隔分成outer与inner两层迭代，增加循环层数，用于将循环操作分割为更小的子任务。从Cuda的文档中我们可以知道，gridDim和blockDim都可以最多是三维，因此可以通过split可以产生新的维度用于绑定到grid和block上。这个操作在生成CUDA代码中是很常用的。

![threadIdx可以最多是三维](https://img-blog.csdnimg.cn/2021032715082614.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

实验代码可以在https://github.com/BBuf/tvm_learn/blob/main/scheduler 这里找到，我使用的tvm版本为0.8.0-dev。

## reorder
第二个想讲一下的scheduler是reorder，我们贴出实验代码和经TVM生成的代码：

```python
import tvm
from tvm import te

n = 1024
A = te.placeholder((n, n), name='A')
B = te.placeholder((n,n), name='B')
C = te.compute((n, n), lambda i, j: A[i, j] + B[i, j], name='C')

s = te.create_schedule(C.op)

xo, xi = s[C].split(s[C].op.axis[0], factor=32)
yo, yi = s[C].split(s[C].op.axis[1], factor=32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].reorder(xo, yo, yi, xi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

生成的函数为：

```powershell
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer: int32, 0, 32) {
    for (i.inner: int32, 0, 32) {
      for (j.outer: int32, 0, 32) {
        for (j.inner: int32, 0, 32) {
          C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = ((float32*)A_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + (float32*)B_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)])
        }
      }
    }
  }
}


---------cutting line---------
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer: int32, 0, 32) {
    for (j.outer: int32, 0, 32) {
      for (j.inner: int32, 0, 32) {
        for (i.inner: int32, 0, 32) {
          C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = ((float32*)A_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + (float32*)B_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)])
        }
      }
    }
  }
}
```


可以看到reorder 方法重置了循环iter的内外顺序，根据局部性原理，这样可以最大化利用cache中的现有数据，减少数据频繁载入载出的情况，进而提高程序的性能。这也是我们之前探索矩阵乘法时，为什么要将K维放在最外层，而不是将M放在最外层的原因。


## tile

接下来我们再看一下tile这种scheduler，tile可以将stage（理解为一个OP，一个OP对应了一个stage）的两个维度按照各自的factor进行拆分，并以固定顺序返回两个outer和两个inner的iter，从而增加循环层数，形成更小的计算任务。事实上，tile是可以由split和reorder来实现的，tile是矩阵乘法和卷积计算的重要schedule。在这篇文章的第二节介绍部分，我们贴出了在Arm端手写各种优化算法去优化矩阵乘法，里面就多次用到了分块的计算策略，也就是这里的tile scheduler，可以更好的利用缓存和寄存器，获得更高的性能。

```python
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i: int32, 0, 1024) {
    for (j: int32, 0, 1024) {
      C_2[((i*1024) + j)] = 0f32
      for (K: int32, 0, 1024) {
        C_2[((i*1024) + j)] = ((float32*)C_2[((i*1024) + j)] + ((float32*)A_2[((i*1024) + K)]*(float32*)B_2[((K*1024) + j)]))
      }
    }
  }
}


---------cutting line---------
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (i.outer: int32, 0, 32) {
    for (j.outer: int32, 0, 32) {
      for (i.inner: int32, 0, 32) {
        for (j.inner: int32, 0, 32) {
          C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = 0f32
          for (K: int32, 0, 1024) {
            C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] = ((float32*)C_2[((((i.outer*32768) + (i.inner*1024)) + (j.outer*32)) + j.inner)] + ((float32*)A_2[(((i.outer*32768) + (i.inner*1024)) + K)]*(float32*)B_2[(((K*1024) + (j.outer*32)) + j.inner)]))
          }
        }
      }
    }
  }
}
```

## vectorize 

我们最后再介绍一种scheduler，即向量化。这个也就是公众号的【AI PC端算法优化】介绍的一系列优化方法，例如在Intel CPU上使用SSE或者AVX等指令集向量化普通的程序获得更好的性能。现在，我们看一下TVM里面是如何使用的吧。代码如下：

```python
import tvm
import numpy
import timeit
from tvm import te

M = 1024
N = 1024
A = te.placeholder((M, N), name='A')
B = te.placeholder((M, N), name='B')
C = te.compute(
           (M, N),
           lambda x, y: A[x, y] + B[x, y],
           name='C')

s = te.create_schedule(C.op)
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], 32, 32)

print(tvm.lower(s, [A, B, C], simple_mode=True))
print("---------cutting line---------")

s[C].vectorize(yi)

print(tvm.lower(s, [A, B, C], simple_mode=True))
```

生成的函数为：

```powershell
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner: int32, 0, 32) {
        for (y.inner: int32, 0, 32) {
          C_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] = ((float32*)A_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)] + (float32*)B_2[((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)) + y.inner)])
        }
      }
    }
  }
}


---------cutting line---------
primfn(A_1: handle, B_1: handle, C_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {C: Buffer(C_2: Pointer(float32), float32, [1024, 1024], []),
             B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, C_1: C} {
  for (x.outer: int32, 0, 32) {
    for (y.outer: int32, 0, 32) {
      for (x.inner: int32, 0, 32) {
        C_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] = ((float32x32*)A_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)] + (float32x32*)B_2[ramp((((x.outer*32768) + (x.inner*1024)) + (y.outer*32)), 1, 32)])
      }
    }
  }
}
```

我们可以看到vectorize将iter方向上的循环迭代替换成ramp，从而通过SIMD指令实现数据的批量计算，并且只有在数据size为常数、且分割的iter为2的幂（即满足SIMD的计算数量）时才会发生替换，否则vectorize没有效果，这是SIMD计算设备（如Intel CPU、Arm CPU）的常用schedule。

还有很多重要的scheduler介于篇幅原因就不一一列举了，大家可以仔细读这篇文章：https://zhuanlan.zhihu.com/p/94846767。如果要运行最新版本的TVM scheduler实验，可以在https://github.com/BBuf/tvm_learn 这里找到代码。

# 0x04. 小结

这篇文章主要结合了TVM中的一些实例来介绍了scheduler，其实写到这里我们很自然的又会想出一些问题，例如对于一个深度学习模型，我们对于整个计算图要如何应用上面介绍的这些scheduler技巧才可以生成高效的特定后端的代码，这个时候手动指定计算图的scheduler就不现实了。这就和Auto-TVM和Auto-Scheduler(或者叫Ansor)有关了，不得不提的是Ansor是发表在OSDI会议上，目前比Auto-TVM拥有更好的表现，https://zhuanlan.zhihu.com/p/360041136 这篇近期发表的文章很好的介绍了Ansor的工作机制，推荐读者阅读。后面在理清相关概念之后，也会尝试从源码角度走进TVM，希望将前端和调度的具体过程尝试理一下。


# 0x05. 参考资料
- https://tvm.apache.org/docs
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
- https://zhuanlan.zhihu.com/p/94846767

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)