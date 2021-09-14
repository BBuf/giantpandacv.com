最近一段时间做比较多移动端开发相关的工作，感觉移动端优化相关的对我来说挺有趣的，以前都是在PC上写代码，写代码的时候对于代码的性能没有过多的思考和感觉。但是在移动端上写代码明显能察觉到一段代码写的好不好，对于在移动端上运行性能有很大的影响，尤其在一些比较老旧的机型上测试更能有感觉。

然后最近刚好在复现一篇论文，要在[MXNet](https://link.zhihu.com/?target=https%3A//mxnet.apache.org/)中实现类似[盒子滤波（boxfilter）](https://link.zhihu.com/?target=https%3A//en.wikipedia.org/wiki/Box_blur)的操作子，其实就是步长为1的sum pooling，盒子滤波算是很基础和经典的函数，但是在PC上实现的话因为有GPU，借助其强大的算力所以可以很暴力的实现，每个thread计算以某点为中心给定半径下的区域大小的和即可。然后突发奇想想试试在移动端cpu上试试如何写高效的盒子滤波操作。

这篇文章就是把我的实践过程记录下来，首先给出最简单的实现然后如何一步步优化，到最后给出一个性能优化还不错的版本。由于我正式接触移动端优化的时间不长，很多东西理解的不深，所以有哪里论述不正确的地方请读者指出。

本文的代码：

[https://github.com/Ldpe2G/ArmNeonOptimization/tree/master/boxFilter](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/tree/master/boxFilter)

##  1.Boxfilter最简单实现

首先来看下Boxfilter最简单最直观的实现：

```C++
void BoxFilter::filter(float *input, int radius, int height, int width, float *output) {
    for (int h = 0; h < height; ++h) {
        int height_sift = h * width;
        for (int w = 0; w < width; ++w) {
            int start_h = std::max(0, h - radius);
            int end_h = std::min(height - 1, h + radius);
            int start_w = std::max(0, w - radius);
            int end_w = std::min(width - 1, w + radius);
      		float tmp = 0;
          	for (int sh = start_h; sh <= end_h; ++sh) {
                for (int sw = start_w; sw <= end_w; ++ sw) {
                  tmp += input[sh * width + sw];
                }
          	}
          	output[height_sift + w] = tmp;
       	}
	}
}
```


对每个点，计算给定半径下区域的和，需要注意下边界的处理。

其时间复杂度是 O( **height** x **width** x **(radius** x **2** + **1)** x (**radius** x **2** + **1)** )，

这个最简单的实现在输入大小固定的情况下，半径越大耗时越大，有很多重复计算的地方，相邻元素在计算各自区域内和的时候其实是有重叠的。然后第一个优化的思路就是boxfilter的计算是行列可分离的，具体可参考[4]。

## 2.Boxfilter优化第一版

```C++
void BoxFilter::fastFilter(float *input, int radius, int height, int width, float *output) {
	float *cachePtr = &(cache[0]);
    // sum horizonal
    for (int h = 0; h < height; ++h) {
    	int sift = h * width;
        for (int w = 0; w < width; ++w) {
            int start_w = std::max(0, w - radius);
            int end_w = std::min(width - 1, w + radius);
      
      		float tmp = 0;
      		for (int sw = start_w; sw <= end_w; ++ sw) {
        		tmp += input[sift + sw];
     	 	}

      		cachePtr[sift + w] = tmp;
    	}
  }
  // sum vertical
  for (int h = 0; h < height; ++h) {
    int shift = h * width;
    int start_h = std::max(0, h - radius);
    int end_h = std::min(height - 1, h + radius);

    for (int sh = start_h; sh <= end_h; ++sh) {
      int out_shift = sh * width;
      for (int w = 0; w < width; ++w) {
        output[out_shift + w] += cachePtr[shift + w];
      }
    }
  }
}
```


所谓行列可分离就是，把行列分开计算，从代码里可以看到，对每个元素，首先计算行方向上半径内的和，然后再计算列半径内的和，

所以这时候的时间复杂度是O( **height** x **width** x ( **radius** x **2 + 1** ) x **2** )。

可以看到行列分离之后，时间复杂度减少了不少，尤其半径越大减少的越多，但是还是有重复计算的地方，而且在固定输入下时间复杂度还是会随半径的变大而变大。那么有没有方法可以使得计算复杂度不受半径的影响呢，这个优化思路就是比如在算某一行每个点的半径区域内的和时，对于行开头第一个点，首先计算其半径内和，然后对于接下来的点，不需要重新计算其半径区域内和，而是只需要把前一个元素半径内的和，按半径窗口偏移之后减去旧的点和加上新加入的点即可。

## 3.Boxfilter优化第二版

```C++
void BoxFilter::fastFilterV2(float *input, int radius, int height, int width, float *output) {
    float *cachePtr = &(cache[0]);
    // sum horizonal
    for (int h = 0; h < height; ++h) {
        int shift = h * width;    
	
    float tmp = 0;
    for (int w = 0; w < radius; ++w) {
      tmp += input[shift + w];
    }

    for (int w = 0; w <= radius; ++w) {
      tmp += input[shift + w + radius];
      cachePtr[shift + w] = tmp;
    }

    int start = radius + 1;
    int end = width - 1 - radius;
    for (int w = start; w <= end; ++w) {
      tmp += input[shift + w + radius];
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }

    start = width - radius;
    for (int w = start; w < width; ++w) {
      tmp -= input[shift + w - radius - 1];
      cachePtr[shift + w] = tmp;
    }
  }

  float *colSumPtr = &(colSum[0]);
  for (int indexW = 0; indexW < width; ++indexW) {
    colSumPtr[indexW] = 0;
  } 
  // sum vertical
  for (int h = 0; h < radius; ++h) {
    int shift = h * width;
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += cachePtr[shift + w];
    }
  }

  for (int h = 0; h <= radius; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += addPtr[w];
      outPtr[w] = colSumPtr[w];
    }
  }

  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift;
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] += addPtr[w];
      colSumPtr[w] -= subPtr[w];
      outPtr[w] = colSumPtr[w];
    }
  }

  start = height - radius;
  for (int h = start; h < height; ++h) {
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    for (int w = 0; w < width; ++w) {
      colSumPtr[w] -= subPtr[w];
      outPtr[w] = colSumPtr[w];
    }
  }
}
```


这一版时间复杂度大概是O( **height** x **width** x **4**)，不算边界只看中间部分的计算就是一次加法和一次减法，行方向和列方向都一样。这里行方向的部分很好理解，因为边界部分需要特殊处理，比如开始部分只有加，结尾部分只有减法，所以计算分成了3部分。列方向计算的话按照常规思路，那就是按一列列来处理，可是我们知道数据一般是按照行来存储的，这样子跳行取数据，会造成很多次cache miss，这样子性能肯定会受很大的影响，所以这里用了一个大小是width的向量colSum，来存储每一列对应点的半径区域内的和，然后遍历的时候还是按照行来遍历，如果一下子理解不了这个思路的话，可以想象如果width为1的情况，那么应该可以更好的理解。

然后我们来看下实验结果，这三版boxfilter在输入是2000x2000的情况下，在不同半径下的运行耗时，测试手机是华为荣耀4C（CHM-TL00），每个函数运行10次取平均为其耗时：

![](https://pic2.zhimg.com/v2-016e27f32b77248bf93a5ac50e37c10d_b.png)

可以看到第二版优化的耗时在不同半径下的表现都很稳定，基本不受影响。然后接下来的优化思路就是在确定了C++ 的代码之后可以采用[arm Neon
Intrinsics](https://link.zhihu.com/?target=https%3A//developer.arm.com/architectures/instruction-
sets/simd-isas/neon/intrinsics)来加速了，就是利用向量计算指令同时处理多个数据，把独立的运算同时做，比写汇编要容易。

## 4.Boxfilter优化第二版 Neon Intrinsics

```C++
  int n = width >> 2;
  int re = width - (n << 2);  
  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    for (; nn > 0; nn--) {
      float32x4_t _add = vld1q_f32(tmpAddPtr);
      float32x4_t _sub = vld1q_f32(tmpSubPtr);
      float32x4_t _colSum = vld1q_f32(tmpColSumPtr);

      float32x4_t _tmp = vaddq_f32(_colSum, _add);
      _tmp = vsubq_f32(_tmp, _sub);

      vst1q_f32(tmpColSumPtr, _tmp);
      vst1q_f32(tmpOutPtr, _tmp);

      tmpAddPtr += 4;
      tmpSubPtr += 4;
      tmpColSumPtr += 4;
      tmpOutPtr += 4;
    }
#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
```

上面的代码是截取列方向中间计算部分来展示如何使用[arm Neon Intrinsics](https://link.zhihu.com/?target=https%3A//developer.arm.com/architectures/instruction-
sets/simd-isas/neon/intrinsics)函数，完整代码可以看

[https://github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp#L143](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp%23L143)

而行方向是没办法并行的，因为相邻元素有依赖，而列方向则可以，所以在列方向上做改写。以上代码其实挺好理解的，vld1q_f32指令就是加载4个浮点数，然后vaddq_f32，为把两个float32x4_t向量相加，相当于同时计算了4个输出，然后再把结果用vst1q_f32存回去对应的地址，然后所有参与运算的地址都是每次加4，具体可以参考[官网文档](https://link.zhihu.com/?target=http%3A//infocenter.arm.com/help/index.jsp%3Ftopic%3D/com.arm.doc.dui0489c/CIHCCEBB.html)。

然后来看下这版优化的耗时如何：

![](https://pic4.zhimg.com/v2-4853e8a412793116943a70842ad1b853_b.png)

可以看到耗时又少了一点，但是收益已经不大了。然后还想尝试进一步优化把Intrinsics部分改写成内联汇编试试。

## 4.Boxfilter优化第二版 Neon Assembly

```C++
  int n = width >> 2;
  int re = width - (n << 2);
  int start = radius + 1;
  int end = height - 1 - radius;
  for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = n;
    int remain = re;
#if __ARM_NEON
    asm volatile(
      "0:                       \n"
      "vld1.s32 {d0-d1}, [%0]!  \n"
      "vld1.s32 {d2-d3}, [%1]!  \n"
      "vld1.s32 {d4-d5}, [%2]   \n"
      "vadd.f32 q4, q0, q2      \n"
      "vsub.f32 q3, q4, q1      \n"
      "vst1.s32 {d6-d7}, [%3]!  \n"
      "vst1.s32 {d6-d7}, [%2]!  \n"
      "subs %4, #1              \n"
      "bne  0b                  \n"
      : "=r"(tmpAddPtr), //
      "=r"(tmpSubPtr),
      "=r"(tmpColSumPtr),
      "=r"(tmpOutPtr),
      "=r"(nn)
      : "0"(tmpAddPtr),
      "1"(tmpSubPtr),
      "2"(tmpColSumPtr),
      "3"(tmpOutPtr),
      "4"(nn)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4"
    );

#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
```


完整版代码：[https://github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp#L331](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp%23L331)

这里我只对列计算中间部分做了改写，neon汇编下面的"cc"，"memory"之后跟的寄存器，是为了告诉编译器（主要是q开头的，q和d是一样的，q表示128位向量寄存器（16个），d表示64位（32个），q0 =（d0 + d1）），这些寄存器会在汇编内被用到，然后编译器在进入这段代码之前，要缓存这些寄存器的内容，然后在离开这段汇编之后恢复原来的值。一定要记得写上用了哪些向量寄存器。

简单解释一下，指令的意思，"vld1.s32 {d0-d1}, [%0]! \n"，相当等于从tmpAddPtr这个地址连续读取4个浮点数到{d0-d1}也就是q0寄存器，浮点数每个32位，乘以四就是128位。最后的感叹号表示，这个指令完成之后tmpAddPtr地址加4的意思，没有就是不变。"vadd.f32 q4, q0, q2 \n" 就是把 q0和q2相加的结果放到q4

"vsub.f32 q3, q4, q1 \n" 就是把q4减去q1的结果放到q3，和上面的intrinsics指令对应，然后vst1.s32就是把寄存器的内容存到tmpOutPtr和tmpColSumPtr地址指向的内存。最后的subs指令和bne相当于for循环的功能，最后对nn减一然后bne判断是否为0，
不为0则继续循环跳到开头0标记出继续执行。  汇编指令其实和intrinsics函数有对应的具体可参考官方文档

[https://static.docs.arm.com/ddi0406/c/DDI0406C_C_arm_architecture_reference_manual.pdf](https://link.zhihu.com/?target=https%3A//static.docs.arm.com/ddi0406/c/DDI0406C_C_arm_architecture_reference_manual.pdf)

然后我们来看下耗时：

![](https://pic1.zhimg.com/v2-15382832656f927895ccd49a54d07538_b.png)

什么鬼，竟然还慢了，一定是我使用的方式不对。去查了下资料，看到这篇[博客](https://link.zhihu.com/?target=http%3A//armneon.blogspot.com/2013/07/neon-
tutorial-part-1-simple-
function_13.html)里面提到，指令vld和vst都是需要消耗两个时钟周期，其他指令基本都是一个时钟周期，但是却不意味着一个时钟周期之后能立刻得到结果。那么看下来 vsub.f32 指令依赖 vadd.f32 的结果，所以白白浪费了不少时钟周期。而且现代的处理器支持[双发射流水线](https://www.zhihu.com/question/20148756)，也就意味着CPU可以同时拾取两条数据无关指令，那么能否利用这点来更进一步加速呢。

## 5.Boxfilter优化第二版 Neon Assembly 第二版    

```c++
int start = radius + 1;
int end = height - 1 - radius;
for (int h = start; h <= end; ++h) {
    float *addPtr = cachePtr + (h + radius) * width;
    float *subPtr = cachePtr + (h - radius - 1) * width;
    int shift = h * width;
    float *outPtr = output + shift; 
    int indexW = 0;
    float *tmpOutPtr = outPtr;
    float *tmpColSumPtr = colSumPtr;
    float *tmpAddPtr = addPtr;
    float *tmpSubPtr = subPtr;

    int nn = width >> 3;
    int remain = width - (nn << 3);
#if __ARM_NEON
    asm volatile(
      "0:                       \n"
      "pld      [%0, #256]      \n"
      "vld1.s32 {d0-d3}, [%0]!  \n"
      "pld      [%2, #256]      \n"
      "vld1.s32 {d8-d11}, [%2]  \n"

      "vadd.f32 q6, q0, q4      \n"

      "pld      [%1, #256]      \n"
      "vld1.s32 {d4-d7}, [%1]!  \n"
      
      "vadd.f32 q7, q1, q5      \n"
      
      "vsub.f32 q6, q6, q2      \n"
      
      "vsub.f32 q7, q7, q3      \n"
      
      "vst1.s32 {d12-d15}, [%3]!  \n"
      
      // 感谢 @随风漂 指出这里错误，用错了寄存器，输出结果是错的
      // "vst1.s32 {d16-d19}, [%2]!  \n" 

      "vst1.s32 {d12-d15}, [%2]!  \n"

      "subs %4, #1              \n"
      "bne  0b                  \n"
      : "=r"(tmpAddPtr), //
      "=r"(tmpSubPtr),
      "=r"(tmpColSumPtr),
      "=r"(tmpOutPtr),
      "=r"(nn)
      : "0"(tmpAddPtr),
      "1"(tmpSubPtr),
      "2"(tmpColSumPtr),
      "3"(tmpOutPtr),
      "4"(nn)
      : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9"
    );

#endif // __ARM_NEON
    for (; remain > 0; --remain) {
      *tmpColSumPtr += *tmpAddPtr;
      *tmpColSumPtr -= *tmpSubPtr;
      *tmpOutPtr = *tmpColSumPtr;
      tmpAddPtr ++;
      tmpColSumPtr ++;
      tmpOutPtr ++;
      tmpSubPtr ++;
    }
  }
```


完整版代码：[https://github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp#L527](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/blob/master/boxFilter/src/boxFilter.cpp%23L527)

可以看到这里的改进思路就是，把两条 vadd.f32 指令放一起，然后跟两条vsub.f32，然后把加载 vsub.f32 要用到部分数据指令
vld1.s32 放到两个 vadd.f32之间，同时 vld1.s32 指令之前加上 [pld 指令](https://link.zhihu.com/?target=http%3A//infocenter.arm.com/help/index.jsp%3Ftopic%3D/com.arm.doc.dui0489c/CJADCFDC.html)，这个指令为什么能加速我问了下做移动端优化的同事，pld把数据从内存加载到cache然后下一条指令把数据从cache加载到寄存器，如果不用pld，数据若不在cache中，那么就是需要直接从内存加载到寄存器，这里会比前者慢很多。

然后我们来看下最终版的耗时：

![](https://pic3.zhimg.com/v2-a678f6c98960ac9ef8ef2bda7f4f5322_b.png)

看表格最终版的耗时比起最原始的实现至少可以加速6~7倍，肯定是还有更好的优化方式，比如如果能对输入做量化把float类型数据转成8bit整型，那么就可以在单位时间处理更多数据，当然量化到8bit上计算溢出的风险也会增大许多。

有时候炼丹炼久了，学习下优化也挺好玩的，感觉可以很好的锻炼下思维和代码能力，现在深度学习在移动端应用越来越广泛，训出来的模型如果部署到移动端之后运行的效率很低那么也是白费功夫。所以感觉对移动端优化有一定的了解对于如何设计对移动端更友好的模型还是有帮助的。

## 相关资料：

[1] [小鱼干：ARM NEON 优化](https://zhuanlan.zhihu.com/p/24702989)

[2] [https://azeria-labs.com/writing-arm-assembly- part-1/](https://link.zhihu.com/?target=https%3A//azeria-labs.com/writing-arm-
assembly-part-1/)

[3] [http://armneon.blogspot.com/2013/07/neon-tutorial-part-1-simple- function_13.html](https://link.zhihu.com/?target=http%3A//armneon.blogspot.com/2013/07/neon-
tutorial-part-1-simple-function_13.html)

[4] [解析opencv中Box Filter的实现并提出进一步加速的方案（源码共享）。](https://link.zhihu.com/?target=http%3A//www.cnblogs.com/Imageshop/p/5053013.html)

[5] [ARM Information Center](https://link.zhihu.com/?target=http%3A//infocenter.arm.com/help/index.jsp%3Ftopic%3D/com.arm.doc.dui0489c/CIHCCEBB.html)

