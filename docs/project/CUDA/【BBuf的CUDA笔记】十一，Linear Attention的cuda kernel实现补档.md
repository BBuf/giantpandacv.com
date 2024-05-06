# 0x0. 前言
填一下 [【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw)  留下的坑，阅读本文之前需要先阅读上面这篇文章。这里就不重复介绍背景知识了，只需要知道现在要计算的目标是：$V_j' = (Q_{0:j} * K_{0:j}^T) * V_{0:j}$ ，也就是先计算 K 和 V 的外积（vvt_dot），然后计算 Q 和这个外积的结果（vm_dot），这个过程由 causal_dot_product 这个cuda kernel来实现。

然后，在 https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L661-L689 对应的lmha模板函数的实现中有两种不同的kernel dispatch逻辑：

```cpp
// GO_BACKWARD: 一个布尔类型的模板参数，用于指示是进行前向计算还是反向计算。
template< bool GO_BACKWARD >
int lmha(const Lmha_params<float> &params) {
  int blocks = params.B * params.H; // blocks表示GPU的block数量？
  int res = 1;
  if( blocks < LOW_OCCUPANCY_THRESHOLD ) { 
           if( params.E <=  32 ) {
      res = lmha_low_occupancy_< 32, GO_BACKWARD>(params, blocks);
    } else if( params.E <=  64 ) {
      res = lmha_low_occupancy_< 64, GO_BACKWARD>(params, blocks);
    } else if( params.E <= 128 ) {
      res = lmha_low_occupancy_<128, GO_BACKWARD>(params, blocks);
    } else if( params.E <= 256 ) {
      res = lmha_low_occupancy_<256, GO_BACKWARD>(params, blocks);
    }
  } else {
           if( params.E <=  32 ) {
      res = lmha_< 32, 1, GO_BACKWARD>(params);
    } else if( params.E <=  48 ) {
      res = lmha_< 48, 1, GO_BACKWARD>(params);
    } else if( params.E <=  64 ) {
      res = lmha_< 64, 1, GO_BACKWARD>(params);
    } else if( params.E <= 128 ) {
      res = lmha_<128, 2, GO_BACKWARD>(params);
    } else if( params.E <= 256 ) {
      res = lmha_<256, 4, GO_BACKWARD>(params);
    }
  }
  return res;
}
```

如果blocks（batch 和 注意力头的乘积 即params.B * params.H）小于LOW_OCCUPANCY_THRESHOLD（=40）的时候，走的是`lmha_low_occupancy_`这个kernel的实现，否则就会走到`lmha_`的实现。另外，还会根据 query 的特征维度的大小 E 来设置kernel不同的模板参数。 [【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw) 详细解析了`lmha_`这个kernel的实现，这篇文章就来详解一下`lmha_low_occupancy_`的实现。

# 0x1. lmha_low_occupancy_ kernel实现解析

我们先从理论上来解释一下这个kernel的取名，cuda中occupancy指的是一个SM中实际活跃的warp与理论上可以最高可以活跃的warp的比值，然后如果occupancy太低直接带来的影响就是GPU没有足够多的warp来切换，就无法隐藏数据加载/计算的延时，直接导致了kernel B的算力下降。而触发这个`lmha_low_occupancy_` kernel的条件就是`blocks < LOW_OCCUPANCY_THRESHOLD`，且`LOW_OCCUPANCY_THRESHOLD=40`，我们想一下如果blocks比较小的情况下我们也dispatch到`lmha_`这个kernel会发生什么？

对于`lmha_`这个kernel，它的Block数量就是上面blocks，所以对这个kernel来说，它只能启动少于40个Block，对于V100来说有80个sm，对于A100则有120个sm，如果在这两种显卡上只启动不到40个Block则显然SM只能用上一半不到，会导致GPU存在大量资源浪费的现象。因此，当blocks的数量小于40的时候，Linear Attention的官方实现选择实现了另外一个`lmha_low_occupancy_`来尽量增加Block的数量，减少sm资源浪费。

感觉这里的40并没有设得很好，没有充分考虑到各种GPU的sm数量，感觉可以优化成根据SM的数量来自动选择。


接着来看`lmha_low_occupancy_` kernel的具体逻辑：

```cpp
template< int E, bool GO_BACKWARD >
int lmha_low_occupancy_(const Lmha_params<float> &params, int blocks) {
         if( params.M * blocks >= 8*LOW_OCCUPANCY_THRESHOLD ) {
    return lmha_low_occupancy_<E, GO_BACKWARD,  4>(params);
  } else if( params.M * blocks >= 4*LOW_OCCUPANCY_THRESHOLD ) {
    return lmha_low_occupancy_<E, GO_BACKWARD,  8>(params);
  } else {
    return lmha_low_occupancy_<E, GO_BACKWARD, 16>(params);
  }
  return 1;
}
```

这里还多了一层，会根据query的特征维度E，key的特征维度M以及`LOW_OCCUPANCY_THRESHOLD=40`来决定`lmha_low_occupancy_` kernel的第三个模板参数，也就是warp的数量。继续往下看`lmha_low_occupancy_`真正的kernel启动部分：

```cpp

template< int E, bool GO_BACKWARD, int WARPS >
int lmha_low_occupancy_(const Lmha_params<float> &params) {

  // Make sure we are not going to launch an invalid grid.
  if( params.H > 65535 || params.B > 65535 ) {
    return 1;
  }

  // Prepare the grid and trigger the CUDA kernel.
  dim3 grid;
  grid.x = params.M;
  grid.y = params.H;
  grid.z = params.B;
  lmha_low_occupancy_kernel<E, GO_BACKWARD, WARPS><<<grid, WARPS*THREADS_PER_WARP>>>(params);
  return 0;
}
```

再明确一下params的几个维度，批量大小是 B、头数是 H、序列长度是 L 和query的特征维度 E，**key的特征维度M** 。一般来说E和M是相等的。然后这里有个条件判断，当H和B有一个大于65536的时候就返回1，不执行这个kernel。都则就开一个三维线程网格，大小为（M, H, B），也就是有 `B * H * M`这么多个 Block，而每个Block的线程数量为 `WARPS*THREADS_PER_WARP`。其中WARPS是模板参数表示启动Kernel时用到多少个warp，由query的特征维度E，key的特征维度M以及`LOW_OCCUPANCY_THRESHOLD=40`来共同决定，而`THREADS_PER_WARP=32`表示一个warp有32个线程。

接下来就来到真正的cuda kernel实现了，代码和解释如下，这里假设E和M都是1024，WARPS=4：

```cpp
// 模板参数包括 E（query 的特征维度）、GO_BACKWARD（前向或者反向的布尔值，我们可以忽略这个参数）、WARPS（每个 block 的 warp 数量）以及 COLS_PER_THREAD（每个线程处理的列数，默认值为 4）。
template< int E, bool GO_BACKWARD, int WARPS, int COLS_PER_THREAD = 4 >
__global__ __launch_bounds__(WARPS * THREADS_PER_WARP)
void lmha_low_occupancy_kernel(Lmha_params<float> params) {

  // 这行代码定义了每个 block 的线程总数。它是 warp 数量（WARPS）和
  // 每个 warp 的线程数（THREADS_PER_WARP，在 CUDA 中通常为 32）的乘积。
  constexpr int THREADS_PER_BLOCK = WARPS * THREADS_PER_WARP; // 4*32=128
  // 这定义了每个线程负责的行数。它是特征维度 E 除以每个 warp 的线程数计算得出。
  constexpr int ROWS_PER_THREAD = E / THREADS_PER_WARP; // 1024/32=32
  // 这表示每次迭代处理的列数，由 warp 数量和每个线程的列数确定。
  constexpr int COLS_PER_ITER = WARPS * COLS_PER_THREAD; // 4*4=16

  // Make sure E is a multiple of the warp size.
  static_assert(E % THREADS_PER_WARP == 0, "");

  // ：这两行代码为 V 和 O 分配共享内存。这些数组的大小由 COLS_PER_ITER 决定，即每次迭代处理的总列数。这里为16。
  __shared__ float smem_v[COLS_PER_ITER], smem_o[COLS_PER_ITER];
  // 这行代码为归约操作分配共享内存。内存的大小由特征维度 E 和 warp 数量决定。这里为1024*4=4096。
  __shared__ float smem_reds[E * WARPS]; 

  // 获取当前 block 处理的序列索引，通过 blockIdx.z 获得。
  const int bi = blockIdx.z;
  // 获取当前 block 处理的头索引，通过 blockIdx.y 获得
  const int hi = blockIdx.y;
  // 获取当前 block 处理的在 V（值）/输出缓冲区中的隐藏单元索引，通过 blockIdx.x 获得。
  const int vi = blockIdx.x;

  // 获取当前线程的线性索引。这是线程在其 block 中的唯一标识。
  const int tidx = threadIdx.x;

  // 计算当前线程属于哪个 warp。由于每个 warp 包含 THREADS_PER_WARP 个线程，
  // 所以用线程索引除以每个 warp 的线程数可以得到 warp 索引。
  const int warp = tidx / THREADS_PER_WARP;
  // 计算当前线程在其 warp 内的位置（lane）。这是线程在 warp 内的相对索引。
  const int lane = tidx % THREADS_PER_WARP;

  // 计算查询矩阵 Q 的偏移量。params.q_stride_B 和 params.q_stride_H 分别是 Q 矩阵在序列
  // 和头维度上的步长，通过序列索引 bi、头索引 hi 和 lane 索引来计算具体的偏移量。
  int offset_q = bi*params.q_stride_B + hi*params.q_stride_H + lane;
  // 类似地，为键矩阵 K 计算偏移量。
  int offset_k = bi*params.k_stride_B + hi*params.k_stride_H + lane;

  // If we walk backward, account for the extra offset.
  // 在反向计算中，偏移量需要加上序列长度（L）减一乘以在序列维度上的步长。这样做是为了从序列的末端开始计算。
  if( GO_BACKWARD ) {
    offset_q += (params.L-1)*params.q_stride_L;
    offset_k += (params.L-1)*params.k_stride_L;
  }

  // Position the warp at the beginning of the proper timestep.
  // 这部分代码根据计算方向（正向或反向）调整 warp 的起始位置。
  // 每个 warp 负责 COLS_PER_THREAD=16 列，所以偏移量需要根据这个数值和序列维度上的步长进行调整。
  // 如果是反向计算，则从序列的末端开始向前移动；如果是正向计算，则从序列的开始向后移动。
  if( GO_BACKWARD ) {
    offset_q -= warp*COLS_PER_THREAD*params.q_stride_L;
    offset_k -= warp*COLS_PER_THREAD*params.k_stride_L;
  } else {
    offset_q += warp*COLS_PER_THREAD*params.q_stride_L;
    offset_k += warp*COLS_PER_THREAD*params.k_stride_L;
  }

  // Determine the base pointers for Q and K.
  // 这两行代码通过添加之前计算的偏移量来定位 Q 和 K 的基指针。
  // 这样每个线程都可以访问到正确的数据区域。
  const float *ptr_q = &params.q[offset_q];
  const float *ptr_k = &params.k[offset_k];

  // 这行代码声明了一个数组，用于存储每行是否有效的信息。
  int valid_qk[ROWS_PER_THREAD];
  #pragma unroll
  for( int ii = 0; ii < ROWS_PER_THREAD; ++ii ) {
    // 这个循环通过检查每一行是否在 E 的范围内（即 lane + ii*THREADS_PER_WARP < params.E）
    // 来填充 valid_qk 数组。这样可以确保在处理 Q 和 K 时不会越界访问。
    valid_qk[ii] = lane + ii*THREADS_PER_WARP < params.E;
  }

  // The offset to the position loaded by the thread in V.
  // 这两行代码计算 V 和输出矩阵的初始偏移量。这里的 bi、hi 和 vi 分别表示
  // 当前 block 所处理的序列索引、头索引和隐藏单元索引。
  int offset_v = bi*params.v_stride_B + hi*params.v_stride_H + vi;
  int offset_o = bi*params.o_stride_B + hi*params.o_stride_H + vi;

  // 如果是反向计算（if( GO_BACKWARD ) { ... }），则类似之前对 Q 和 K 的处理，
  // 调整 V 和输出矩阵的偏移量以反映从序列的末尾开始的计算。
  if( GO_BACKWARD ) {
    offset_v += (params.L-1)*params.v_stride_L;
    offset_o += (params.L-1)*params.o_stride_L;
  }

  // We load/store a strided matrix of COLS_PER_ITER x OUTPUTS_PER_BLOCK.
  // 这部分代码根据是正向还是反向计算来调整偏移量，以确保每个线程可以访问到正确的 V 和输出矩阵部分。
  // 如果是反向计算，则减去与线程索引（tidx）相关的偏移量；如果是正向计算，则增加这个偏移量。
  if( GO_BACKWARD ) {
    offset_v -= tidx*params.v_stride_L;
    offset_o -= tidx*params.o_stride_L;
  } else {
    offset_v += tidx*params.v_stride_L;
    offset_o += tidx*params.o_stride_L;
  }

  // 这两行代码分别为 V 矩阵和输出矩阵确定基指针，通过之前计算的偏移量来实现。
  const float *ptr_v = &params.v[offset_v];
  // The output pointer. 
  float *ptr_o = &params.out[offset_o];

  // 声明一个数组来存储每个线程的运行时 KV 值。
  float running_kv[ROWS_PER_THREAD];
  #pragma unroll
  for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
    // 初始化 running KV 数组的每个元素为 0。这是为了在后续的计算中累加值。
    running_kv[ri] = 0.f;
  }

  // 这个循环迭代处理整个序列。params.L 是序列的长度，COLS_PER_ITER 是每次迭代处理的列数。
  // 这意味着在每次迭代中，将会处理 COLS_PER_ITER 个时间步。
  for( int iter = 0; iter < params.L; iter += COLS_PER_ITER ) {

    // 为每个线程声明两个局部数组，用于存储它将加载的 Q 和 K 矩阵的元素。
    // 数组的大小由每个线程负责的行数 (ROWS_PER_THREAD) 和列数 (COLS_PER_THREAD) 决定。
    float q[ROWS_PER_THREAD][COLS_PER_THREAD], k[ROWS_PER_THREAD][COLS_PER_THREAD];

    // 这里使用两层嵌套的循环来加载 Q 和 K 矩阵的元素。
    // 外层循环遍历每个线程负责的列，内层循环遍历每个线程负责的行。
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      #pragma unroll
      for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {

        // For Q/K, each warp loads from various timesteps. 
        // 计算当前 warp 正在处理的时间步索引。这个索引基于迭代的起始点 (iter)，加上 warp 在序列中的位置。
        int ti = iter + warp*COLS_PER_THREAD;
        // 如果是反向计算（GO_BACKWARD 为真），则需要调整时间步索引，以便从序列的末尾开始。
        if( GO_BACKWARD ) {
          ti = params.L - 1 - ti;
        }

        // 在加载 Q 和 K 矩阵的元素之前，需要检查每个访问是否有效，以防止越界访问。
        // 如果是反向计算，检查当前时间步减去列索引是否大于等于 0；如果是正向计算，
        // 检查当前时间步加上列索引是否小于序列长度 (params.L)。
        int valid;
        if( GO_BACKWARD ) {
          valid = valid_qk[ri] && ti - ci >= 0;
        } else {
          valid = valid_qk[ri] && ti + ci < params.L;
        }

        // 这部分代码根据是否进行反向计算（GO_BACKWARD）来调整 Q 和 K 的偏移量。
        // 如果是反向计算，offset_q 和 offset_k 需要减去由当前列索引 (ci) 
        // 和步长 (params.q_stride_L 或 params.k_stride_L) 确定的偏移量。
        // 如果是正向计算，偏移量则相应增加。
        if( GO_BACKWARD ) {
          offset_q = ri*THREADS_PER_WARP - ci*params.q_stride_L;
          offset_k = ri*THREADS_PER_WARP - ci*params.k_stride_L;
        } else {
          offset_q = ri*THREADS_PER_WARP + ci*params.q_stride_L;
          offset_k = ri*THREADS_PER_WARP + ci*params.k_stride_L;
        }

        // 这两行代码负责从 Q 和 K 矩阵中加载元素。
        // 如果当前访问是有效的（由 valid 变量决定），则从 Q 或 K 矩阵的相应位置加载元素。
        // 如果访问无效（例如，超出矩阵边界），则将元素值设为 0。
        q[ri][ci] = valid ? ptr_q[offset_q] : 0.f;
        k[ri][ci] = valid ? ptr_k[offset_k] : 0.f;
      }
    }

    // For the V tensor, we assign contiguous thread to different loads. So, ti is different.
    // 计算当前线程负责的时间步索引。这里，iter 是当前迭代的起始时间步，tidx 是线程的索引。
    int ti = iter + tidx;
    // 如果是反向计算（GO_BACKWARD），则需要调整时间步索引，从序列的末尾开始。
    if( GO_BACKWARD ) {
      ti = params.L - 1 - ti;
    }

    // 首先，检查线程索引是否小于每次迭代处理的列数（COLS_PER_ITER），以确保不会超出预定范围。
    // 接着，根据是正向还是反向计算，检查当前时间步是否在序列长度内。
    int valid_vo = tidx < COLS_PER_ITER;
    if( GO_BACKWARD ) {
      valid_vo &= ti >= 0;
    } else {
      valid_vo &= ti < params.L;
    }

    // 如果访问有效（valid_vo），则从 V 矩阵的相应位置加载元素。如果访问无效，元素值设为 0。
    float ldg_v = valid_vo ? *ptr_v : 0.f;

    // 这部分代码根据是正向还是反向计算来更新 Q、K 和 V 的指针。
    // 如果是反向计算，指针向后移动；如果是正向计算，指针向前移动。
    if( GO_BACKWARD ) {
      ptr_q -= COLS_PER_ITER*params.q_stride_L;
      ptr_k -= COLS_PER_ITER*params.k_stride_L;
      ptr_v -= COLS_PER_ITER*params.v_stride_L;
    } else {
      ptr_q += COLS_PER_ITER*params.q_stride_L;
      ptr_k += COLS_PER_ITER*params.k_stride_L;
      ptr_v += COLS_PER_ITER*params.v_stride_L;
    }

    // 如果线程索引在处理的列数范围内，将加载的 V 值存储到共享内存中。
    // 共享内存用于存储当前 block 内所有线程共享的数据，这有助于减少全局内存访问和提高效率。
    if( tidx < COLS_PER_ITER ) {
      smem_v[tidx] = ldg_v;
    }

    // 确保所有线程都完成了对共享内存的写操作，然后再进行后续计算。这保证了数据的一致性和正确性。
    __syncthreads();

    // Read V from shared memory.
    // 声明一个数组 v，用于存储每个线程将要处理的 V 矩阵列。
    float v[COLS_PER_THREAD];
    // 使用循环从共享内存（smem_v）中加载 V 矩阵的元素到局部数组 v 中。
    // 这里，warp*COLS_PER_THREAD + ci 确保每个 warp 加载不同部分的 V 矩阵。
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      v[ci] = smem_v[warp*COLS_PER_THREAD + ci];
    }

    // Each thread computes local K*V products.
    // 声明一个二维数组 kv，用于存储 K 和 V 的乘积结果。
    // 使用两层嵌套循环将 kv 数组的所有元素初始化为 0。这是为了准备计算 K*V 乘积。
    float kv[ROWS_PER_THREAD][COLS_PER_THREAD];
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] = 0.f;
      }
    }

    // Update the K*V^T product.
    // 内部的两层嵌套循环用于更新 K*V^T 乘积。
    // 这个乘积是注意力机制中的关键部分，它涉及到每个头中键（K）和值（V）的相互作用。
    #pragma unroll
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      #pragma unroll
      for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
        // 这行代码实现了 K*V 乘积的累加。对于每一行 ri 和每一列 ci，
        // 它计算键（K）矩阵的元素和值（V）矩阵的元素之间的乘积，并累加到 kv 数组中。
        kv[ri][ci] += k[ri][ci] * v[ci];
      }
    }

    // We must perform the prefix sums within the thread-block. Start with the thread.
    #pragma unroll
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      #pragma unroll
      for( int ci = 1; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] += kv[ri][ci-1];
      }
    }

    // Store the partial sums to shared memory. Unless we have no inter-warp reduction to perform.
    #pragma unroll
    // 这个循环遍历每个线程负责的行。
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      // 这行代码将每行的部分和（即数组 kv 中每行的最后一个元素）存储到共享内存的 smem_reds 数组中。
      // 这里的索引计算考虑了当前 warp 的索引、行索引和线程在 warp 中的位置（lane）。
      smem_reds[warp*E + ri*THREADS_PER_WARP + lane] = kv[ri][COLS_PER_THREAD-1];
    }

    // 在进行下一步计算之前，调用这个函数来同步线程块中的所有线程。
    // 这确保了所有线程都已经将其部分和写入共享内存。
    __syncthreads();

    // Each thread deals with one or more column(s) of the matrix.
    // 这行代码计算每个线程需要处理的列数，以便进行跨 warp 的规约计算。
    constexpr int SUMS_PER_THREAD = (E + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    #pragma unroll
    // 这个循环遍历每个线程负责的列。
    for( int ii = 0, idx = tidx; ii < SUMS_PER_THREAD; ++ii, idx += THREADS_PER_BLOCK ) {
      // 如果当前线程负责的列索引在 E 的范围内，执行内部的规约计算。
      if( idx < E ) {
        float sum = smem_reds[idx];
        #pragma unroll
        // 这个循环累加同一列中不同 warp 的部分和，并将结果存储回共享内存。
        for( int jj = 1; jj < WARPS; ++jj ) {
          smem_reds[idx + jj*E] = sum += smem_reds[idx + jj*E];
        }
      }
    }

    // 在进行下一步之前再次同步线程，以确保所有的规约计算都已经完成，并且结果已经被写回共享内存。
    __syncthreads();

    // Each thread updates his partial products.
    #pragma unroll
    // 这个循环遍历每个线程负责的行。
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      // 初始化 sum 变量为之前计算的局部 K*V 乘积的累积值。
      float sum = running_kv[ri];
      // 如果当前线程所在的 warp 不是第一个（if( warp > 0 )），则将前一个 warp 的规约结果加到 sum 上。
      if( warp > 0 ) {
        sum += smem_reds[(warp-1)*E + lane + ri*THREADS_PER_WARP];
      }
      // 更新 K*V 乘积的值。这里将每行的累积和加到每个元素上，实现了行内的前缀和累加。
      #pragma unroll
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        kv[ri][ci] += sum;
      }
    }

    // Compute the partial output values for that thread.
    // 声明一个数组 sum，用于存储每个线程的部分输出值。
    float sum[COLS_PER_THREAD];
    #pragma unroll
    // 遍历每个线程负责的列。
    for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
      // 初始化 sum[ci] 为第一行的 Q 和 K*V 乘积。
      sum[ci] = q[0][ci] * kv[0][ci];
      #pragma unroll
      // 累加当前列的其他行的 Q 和 K*V 乘积，以计算该列的总和。
      for( int ri = 1; ri < ROWS_PER_THREAD; ++ri ) {
        sum[ci] += q[ri][ci] * kv[ri][ci];
      }
    }

    // Run the parallel reductions inside the warp.
    // 这个循环控制规约的步骤。mask 变量决定了在每一步中哪些线程会进行通信。
    // 它从线程数的一半开始，每次迭代减半，直到为 1。这是warp规约的经典操作。
    #pragma unroll
    for( int mask = THREADS_PER_WARP / 2; mask >= 1; mask /= 2 ) {
      #pragma unroll
      // 遍历每个线程负责的列。
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        // 这是 CUDA 的 shuffle 指令，用于在一个 warp 内部进行高效的数据交换。
        // 它允许线程读取以 mask 为偏移量的其他线程的 sum[ci] 值，并将这个值加到自己的 sum[ci] 上。
        // 通过这种方式，可以在 warp 内部快速完成规约计算。
        sum[ci] += __shfl_xor_sync(uint32_t(-1), sum[ci], mask);
      }
    }

    // Store the final output to shared memory.
    // 这个条件判断确保只有每个 warp 的第一个线程（lane == 0）执行存储操作。
    if( lane == 0 ) {
      #pragma unroll
      // 遍历每个线程负责的列，并将每列的规约结果存储到共享内存的 smem_o 数组中。
      for( int ci = 0; ci < COLS_PER_THREAD; ++ci ) {
        smem_o[warp*COLS_PER_THREAD + ci] = sum[ci];
      }
    }

    // 在进行下一步计算之前，使用这个函数来同步线程块中的所有线程。
    __syncthreads();

    // 这个条件判断检查当前线程是否有有效的输出要存储。
    // 如果是（即 valid_vo 为真），则将共享内存 smem_o 中对应索引 tidx 的值写入到输出指针 ptr_o 指向的位置。
    if( valid_vo ) {
      *ptr_o = smem_o[tidx];
    }

    // Each thread updates his running kv.
    #pragma unroll
    // 这个循环遍历每个线程负责的行。
    for( int ri = 0; ri < ROWS_PER_THREAD; ++ri ) {
      // 这行代码更新 running_kv 数组的值，加上共享内存 smem_reds 中存储的最后一个 warp 的规约结果。
      // 这种累加操作是为了准备下一个数据批次的处理。
      running_kv[ri] += smem_reds[(WARPS-1)*E + lane + ri*THREADS_PER_WARP];
    }

    // 这部分代码更新输出指针 ptr_o 的位置，以便下一轮迭代时正确地写入输出数据。
    if( GO_BACKWARD ) {
      ptr_o -= COLS_PER_ITER*params.o_stride_L;
    } else {
      ptr_o += COLS_PER_ITER*params.o_stride_L;
    }
  }
}
```

请详细阅读上面的代码和注释，然后这里对这个kernel的核心计算逻辑做一个总结。
- 使用 warp 作为基本的计算单元，每个 warp 处理特定的数据列。通过 THREADS_PER_WARP 和 WARPS 参数来控制每个 block 的线程数量。然后，每个线程加载加载它对应的Q，K，V矩阵部分。并使用 `offset_q`, `offset_k`, 和 `offset_v` 来计算每个线程的数据偏移量。对应： https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L152-L310
- 在每个 warp 内部进行前缀和计算以获得部分 K*V 乘积。这是通过逐行累加来完成的。https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L152-L261 ，这里对V的读取使用了共享内存。
- 使用共享内存进行跨 warp 的规约操作，以计算 K*V^T 的总和。https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L263-L310
- 利用 `__shfl_xor_sync` 指令进行 warp 内部的并行规约，以合并每个 warp 的计算结果。https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L312-L319
- 最终的输出结果被写回到共享内存，然后存储到全局内存中。https://github.com/BBuf/how-to-optim-algorithm-in-cuda/blob/master/linear-attention/causal_product_cuda.cu#L321-L349


这里涉及到的技能主要是使用warp（32个线程）为基本单位来处理这个任务，而不是像[【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw) 中的`lmha_kernel`以单个线程为单位。也就是说对于之前的`lmha_kernel`来说，每个线程都需要完整的算一遍K（`T*M`个元素）和V（T）的外积，再和Q（`T*E`，E一般等于M）进行内积计算，由于这个kernel里面开了比较多的线程`round_up(max(E, M*THREADS_PER_HEAD), 32);`，并且Block数量为B*H一般也是多于SM数量的，这样算是容易打满GPU的。而对于这里的`lmha_low_occupancy_kernel` kernel来说，T维度已经被切开了，所以应当尽量减少线程的数量让每个线程做尽量多的工作来避免频繁的线程切换开销，所以这里使用warp为单位来处理。这种技巧也是应用得比较多了，例如oneflow的softmax和layernorm算子优化中，对于列数比较小的矩阵就是采用一个warp处理一行或者两行这种技巧。

可以把这里介绍的cuda kernel模型抽象成向量外积加内积，我们可以轻易的把它修改为单纯外积，内积用于我们自己的场景。此外，这个`lmha_low_occupancy_kernel`并没有做数据向量化读取，double buffering等等，可能还有进一步优化空间。
# 0x2. 总结
这篇文章和 [【BBuf的CUDA笔记】十，Linear Attention的cuda kernel实现解析](https://mp.weixin.qq.com/s/1EPeU5hsOhB7rNAmmXrZRw)  就是我阅读Linear Attention官方实现的理解。欢迎关注 https://github.com/BBuf/how-to-optim-algorithm-in-cuda 获取更多后续cuda优化相关的知识。

写一个这种cuda kernel难度是挺大的，我在考虑是否要详细分享一段自己在2023年的开源cuda项目开发经历，可以帮助更多的没有很好基础的读者入门cuda kernel开发，如果有这种需要可以在知乎评论区留言。





