### 前言
之前看我司的 [如何实现一个高效的Softmax CUDA kernel？](https://zhuanlan.zhihu.com/p/341059988) 多少还是有些细节没有理解，恰好最近要做一个类似的 Reduce+Scale Kernel，原理机制还是比较相似的，所以翻出来重新理解一下。

### 背景
我们定义这么一个ReduceScale操作：

假设Tensor是(N, C)，首先在C这个维度计算出 absMax 值，我们记作`scale`，然后将每一行除以各自 行的`scale`，并最终输出。

一段朴素的numpy代码是这样：
```python
import numpy as np


N = 1000
C = 128
x = np.random.randn(N, C)
scale = np.expand_dims(np.max(np.abs(x), axis=1), 1)
out = x / scale
print(out.shape)
```
### BaseLine
这里我们BaseLine是直接调用cub库中的 BlockReduce，一个 threadBlock 处理一行数据，计算出AbsMaxVal，然后再缩放，代码如下：
```cpp
#include "cuda.h"
#include "cub/cub.cuh"

constexpr int kReduceBlockSize = 128;

template<typename T>
__device__ T abs_func(const T& a) {
  return abs(a);
}


template<typename T>
__device__ T max_func(const T a, const T b) {
  return a > b ? a : b;
}

template<typename T>
struct AbsMaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max_func(abs_func(a), abs_func(b));
  }
};

template<typename T>
__inline__ __device__ T BlockAllReduceAbsMax(T val) {
  typedef cub::BlockReduce<T, kReduceBlockSize> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ T final_result;
  T result = BlockReduce(temp_storage).Reduce(val, AbsMaxOp<T>());
  if (threadIdx.x == 0) { final_result = result; }
  __syncthreads();
  return final_result;
}

template<typename T, typename IDX>
__global__ void ReduceScaleBlockKernel(T* x, IDX row_size, IDX col_size) {
  for(int32_t row = blockIdx.x, step=gridDim.x; row < row_size; row+= step){
    T thread_scale_factor = 0.0; 
    for(int32_t col=threadIdx.x; col < col_size; col+= blockDim.x){
      IDX idx = row * col_size + col; 
      T x_val = x[idx];
      thread_scale_factor = max_func(thread_scale_factor, abs_func(x_val)); 
    }
    T row_scale_factor = BlockAllReduceAbsMax<T>(thread_scale_factor); 
    for(int32_t col=threadIdx.x; col < col_size; col+=blockDim.x){
      IDX idx = row * col_size + col; 
      x[idx] /= row_scale_factor;
    }
  }
}
```
参数中 x 是输入数据，row_size是行的数量，col_size是列的大小

测试机器是在 A100 40GB，为了让结果区别比较明显，我们将行数设置的比较大，输入形状为(55296*8, 128)，启动的线程块数目根据 [如何设置CUDA Kernel中的grid_size和block_size？](https://zhuanlan.zhihu.com/p/442304996) 这篇文章来指定，这里比较粗暴的设置为(55296, 128)，数据类型为 Float，然后我们看下ncu的结果：

![](https://files.mdnice.com/user/4601/8a0de99d-7252-473b-939f-2d2c23ce35d6.png)

主要有这几个指标，耗时为577.95us，吞吐量为 748.78Gb/s

下面我们就根据 Softmax 优化那篇文章所提及的点来逐步分析：

### 优化1 数据Pack
在之前的 [高效、易用、可拓展我全都要：OneFlow CUDA Elementwise 模板库的设计优化思路](https://zhuanlan.zhihu.com/p/447577193) 里很详细的描述了如何做向量化读写，cuda里最大支持 128bit的读写，那么在数据类型为 Float 时，我们即可以将连续的4个 Float 打包到一起，一次性读写，提升吞吐。

有了解过这方面的读者应该就反应过来，诶 CUDA 里 不是刚好有一个类型叫 float4 就是干这件事的么，没错，但是为了更灵活的支持其他数据类型的向量化，我们利用union共享空间的特性实现了一个 Pack 类：
```cpp
template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};
```
### 优化2 数据缓存
整个算子逻辑是需要读取一遍数据，计算`scale`，然后再读取一遍数据，用`scale`进行缩放。很显然这里**我们读取了两遍数据**，而数据是放在 Global Memory，带宽比较低，会带来读取耗时。

![](https://files.mdnice.com/user/4601/3736d1e9-a2f9-4b69-8592-8c3fdb60563f.png)

![](https://files.mdnice.com/user/4601/77c34aa0-44db-49da-a10a-1d1e69137b1b.png)



一个很自然的想法是缓存到寄存器/Shared Memory中。由于这里我们只实现 WarpReduce 版本，所以我们是缓存到寄存器（其他版本可以参考开头的优化 Softmax 文章）中，减少一次对 Global Memory 的读取。
```cpp
template<typename T, typename IDX, int pack_size, int cols_per_thread>
__global__ void ReduceScaleWarpKernel(T* x, IDX row_size, IDX col_size) {
    // ...
    T buf[cols_per_thread];
    // ...
```

### 优化3 使用Warp处理一行数据
相较 BaseLine，我们这里使用 warp 作为 Reduce 的单位进行操作，首先我们简单看下 WarpReduce 的实现。

```cpp
template<typename T>
struct AbsMaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const {
    return max_func(abs_func(a), abs_func(b));
  }
};

template<typename T>
__inline__ __device__ T WarpAbsMaxAllReduce(T val){
    for(int lane_mask = kWarpSize/2; lane_mask > 0; lane_mask /= 2){
        val = AbsMaxOp<T>()(val, __shfl_xor_sync(0xffffffff, val, lane_mask)); 
    }
    return val; 
}
```
这段代码在别的 BlockReduce 也经常看到，他是借助 `__shfl_xor_sync` 来实现比较，shuffle 指令允许同一线程束的两个线程直接读取对方的寄存器。

```cpp
T __shfl_xor_sync(unsigned mask, T var, int laneMask, int width=warpSize);
```
其中 `mask` 是对线程的一个掩码，我们一般所有线程都要参与计算，所以 `mask` 是 0xffffffff

`var` 则是寄存器值，`laneMask` 则是用来做按位异或的掩码

> 这里引入一个概念叫 Lane，它表示线程束中的第几号线程

示意图如下：

![](https://files.mdnice.com/user/4601/81116856-e2ed-47b2-a7f8-53f8219d624f.png)

当 laneMask = 16 时，其二进制为 0001 0000，然后线程束每个线程与 laneMask 做异或操作

如：

- 0000 0000 xor 0001 0000 = 0001 0000 = 16
- 0000 0001 xor 0001 0000 = 0001 0001 = 17
- 0000 0010 xor 0001 0000 = 0001 0010 = 18

以此类推，最终得到一个 Warp 中的 absmax 值。

接下来我们开始写Kernel，模板参数分别为：

- T 数据类型
- IDX 索引类型
- pack_size pack数，比如float可以pack成4个，那对应pack_size=4
- cols_per_thread 每个线程需要处理的元素个数，比如一行大小是128，而我们一个warp有32个线程，那么这里就是128/32 = 4
```cpp
template<typename T, typename IDX, int pack_size, int cols_per_thread>
__global__ void ReduceScaleWarpKernel(T* x, IDX row_size, IDX col_size) {
    // ...    
}
```

跟BaseLine一样，我们block大小还是设置为128个线程，一个warp是32个线程，所以我们一个block可以组织成(32, 4)，包含4个warp。

![](https://files.mdnice.com/user/4601/5b6143ec-eb92-4c29-a092-bc79bfae26e7.png)

根据这个层级划分，我们可以计算出：
- global_thread_group_id 当前warp的全局index
- num_total_thread_group warp的总数量
- lane_id 线程束内的线程id
- num_packs pack的数目，即每个线程需要处理的元素个数 / pack_size
```cpp
    const int32_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y; 
    const int32_t num_total_thread_group = gridDim.x * blockDim.y; 
    const int32_t lane_id = threadIdx.x; 
    using LoadStoreType = PackType<T, pack_size>;
    using LoadStorePack = Pack<T, pack_size>;
    T buf[cols_per_thread]; 
    constexpr int num_packs = cols_per_thread / pack_size;
```
由于存在启动的warp的数量小于行的数量，所以我们要引入一个 for 循环。

假设我们 cols = 256，那么线程束里的每个线程需要处理 256 /32 = 8个元素，而4个float可以pack到一起，所以我们线程束里的每个线程要处理2个pack，因此也要引入一个关于 num_packs 的 for 循环，以保证整一行都有被读取到：

![](https://files.mdnice.com/user/4601/3775e3bf-3ecd-49e7-87eb-285459aa0812.png)

一次性读取到一个 pack 后，我们再一个个放到寄存器当中缓存起来，并计算线程上的 AbsMaxVal。
```cpp
    for(IDX row_idx = global_thread_group_id; row_idx < row_size; row_idx += num_total_thread_group){
        T thread_abs_max_val = 0.0; 
        for(int pack_idx = 0; pack_idx < num_packs; pack_idx++){
            const int32_t pack_offset = pack_idx * pack_size; 
            const int32_t col_offset = pack_idx * kWarpSize * pack_size + lane_id * pack_size; 
            const int32_t load_offset = (row_idx * col_size + col_offset) / pack_size; 
            LoadStorePack load_pack; 
            load_pack.storage = *(reinterpret_cast<LoadStoreType*>(x)+ load_offset); 
            #pragma unroll 
            for(int i = 0; i < pack_size; i++){
                buf[pack_offset] = load_pack.elem[i]; 
                thread_abs_max_val = max_func(thread_abs_max_val, abs_func(buf[pack_offset]));
            } 
        }
```
接着我们调用 `WarpAbsMaxAllReduce` 进行reduce，获得线程束中的 AbsMaxVal，并对缓存的数据进行数值缩放。
```cpp
        T warp_max_val = WarpAbsMaxAllReduce<T>(thread_abs_max_val); 
        #pragma unroll
        for (int col = 0; col < cols_per_thread; col++) {
            buf[col] = buf[col] / warp_max_val;
        }
```
最后跟一开始读取类似，我们将寄存器里的值再写回去，相关索引的计算逻辑都是一致的：
```cpp
        for(int pack_idx = 0; pack_idx < num_packs; pack_idx++){
            const int32_t pack_offset = pack_idx * pack_size; 
            const int32_t col_offset = pack_idx * pack_size * kWarpSize + lane_id * pack_size; 
            const int32_t store_offset = (row_idx * col_size + col_offset) / pack_size; 
            LoadStorePack store_pack; 
            #pragma unroll 
            for(int i = 0; i < pack_size; i++){
                store_pack.elem[i] = buf[pack_offset + i]; 
            } 
            *(reinterpret_cast<LoadStoreType*>(x)+ store_offset) = store_pack.storage; 
        }
```
完整代码如下：

```cpp
template<typename T>
__inline__ __device__ T WarpAbsMaxAllReduce(T val){
    for(int lane_mask = kWarpSize/2; lane_mask > 0; lane_mask /= 2){
        val = AbsMaxOp<T>()(val, __shfl_xor_sync(0xffffffff, val, lane_mask)); 
    }
    return val; 
}

template<typename T, typename IDX, int pack_size, int cols_per_thread>
__global__ void ReduceScaleWarpKernel(T* x, IDX row_size, IDX col_size) {
    const int32_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y; 
    const int32_t num_total_thread_group = gridDim.x * blockDim.y; 
    const int32_t lane_id = threadIdx.x; 
    using LoadStoreType = PackType<T, pack_size>;
    using LoadStorePack = Pack<T, pack_size>;
    T buf[cols_per_thread]; 
    constexpr int num_packs = cols_per_thread / pack_size;
    for(IDX row_idx = global_thread_group_id; row_idx < row_size; row_idx += num_total_thread_group){
        T thread_abs_max_val = 0.0; 
        for(int pack_idx = 0; pack_idx < num_packs; pack_idx++){
            const int32_t pack_offset = pack_idx * pack_size; 
            const int32_t col_offset = pack_idx * kWarpSize * pack_size + lane_id * pack_size; 
            const int32_t load_offset = (row_idx * col_size + col_offset) / pack_size; 
            LoadStorePack load_pack; 
            load_pack.storage = *(reinterpret_cast<LoadStoreType*>(x)+ load_offset); 
            #pragma unroll 
            for(int i = 0; i < pack_size; i++){
                buf[pack_offset] = load_pack.elem[i]; 
                thread_abs_max_val = max_func(thread_abs_max_val, abs_func(buf[pack_offset]));
            } 
        }
        T warp_max_val = WarpAbsMaxAllReduce<T>(thread_abs_max_val); 
        #pragma unroll
        for (int col = 0; col < cols_per_thread; col++) {
            buf[col] = buf[col] / warp_max_val;
        }
        for(int pack_idx = 0; pack_idx < num_packs; pack_idx++){
            const int32_t pack_offset = pack_idx * pack_size; 
            const int32_t col_offset = pack_idx * pack_size * kWarpSize + lane_id * pack_size; 
            const int32_t store_offset = (row_idx * col_size + col_offset) / pack_size; 
            LoadStorePack store_pack; 
            #pragma unroll 
            for(int i = 0; i < pack_size; i++){
                store_pack.elem[i] = buf[pack_offset + i]; 
            } 
            *(reinterpret_cast<LoadStoreType*>(x)+ store_offset) = store_pack.storage; 
        }
    }
}
```
这里我们方便测试，调用的时候就直接写死一些模板参数
```cpp
constexpr int cols_per_thread = 128 / kWarpSize; 
ReduceScaleWarpKernel<float, int32_t, 4, cols_per_thread><<<55296, block_dim>>>(device_ptr, row_size, col_size);
```
最后我们看一下 ncu 的结果：

![](https://files.mdnice.com/user/4601/1d21c3be-16c1-4fb6-b262-91ee24b2e294.png)

吞吐量达到了1.3T，时间位333us，相比 BaseLine 快了 73 %。

### 总结
还有更多特殊情况可以参考 Softmax 优化的代码，这里仅实现了第一个 Warp 计算方式。我感觉看着还行，真自己写起来理解还是有点困难的，希望这篇博客能帮助读者理解到一些 warp 的使用。