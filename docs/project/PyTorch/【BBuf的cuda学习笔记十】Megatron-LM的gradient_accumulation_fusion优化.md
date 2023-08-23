# 0x0. 前言
这篇文章来解析一下Megaton-LM涉及到的一个优化gradient_accumulation_fusion。这里fusion的意思是在gemm接口中会将当前的结果累加到先前计算的梯度上，所有这些都在一个操作中完成，可以避免多次访问global memory提升算子的带宽。下面解析一下这个优化的调度逻辑和cuda实现。

https://github.com/BBuf/how-to-optim-algorithm-in-cuda 这个仓库整理了一些cuda优化相关链接以及大模型训练推理相关的知识链接（large-language-model-note子目录下），欢迎查看。

# 0x1. 调度逻辑解析
gradient_accumulation_fusion的调度逻辑是和`LinearWithGradAccumulationAndAsyncCommunication`这个类的实现有关的，`LinearWithGradAccumulationAndAsyncCommunication` 这个类又被包了一层变成 `linear_with_grad_accumulation_and_async_allreduce` 这个函数，这个函数又给`RowParallelLinear`和`ColumnParallelLinear`这两个实现模型并行的Linear类使用。

下面解析一下`linear_with_grad_accumulation_and_async_allreduce`这个函数(https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L356-L446)：

```python
# 这部分定义了一个函数，名为linear_with_grad_accumulation_and_async_allreduce，
# 它接收七个参数：输入张量、权重张量、一个可选的偏置张量和3个布尔标志。
def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel_enabled: bool,
) -> torch.Tensor:
    """带有反向传播的异步通信和梯度累积融合的线性层实现.

    此函数提供了一个选项，可以将反向传播计算的结果累积到一个现有的梯度缓冲区中，
    从而避免在梯度计算后进行额外的加法核操作。

    此外，输入梯度的张量并行all reduce可以与权重梯度的计算异步进行。

    在使用序列并行的情况下，输入梯度的reduce scatter与权重梯度的计算异步进行。

    使用此模块需要环境变量CUDA_DEVICE_MAX_CONNECTIONS=1。代码中有一些集合操作，
    应该在计算核之前调度，以使通信与计算重叠，这对于加速是必要的，但对于正确性则不是必要的，
    因此调度器不会强制这种排序。将CUDA_DEVICE_MAX_CONNECTIONS设置为1会强制按照它们被调用的顺序调度内核。

    Arguments:

    input (torch.Tensor required): 输入，类似torch.nn.functional.linear

    weight (torch.Tensor required): 权重，类似torch.nn.functional.linear

    bias (torch.Tensor optional): 偏置，类似torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): 执行梯度累积融合，
    需要自定义的CUDA扩展模块fused_weight_gradient_mlp_cuda。
    要使用gradient_accumulation_fusion，你必须使用--cpp_ext和--cuda_ext安装APEX。
    例如："pip install --global-option="--cpp_ext" --global-option="--cuda_ext ." 
    注意，此扩展要求CUDA版本大于或等于11。否则，你必须关闭梯度累积融合。

    async_grad_allreduce (bool required): 异步地与权重梯度的计算进行输入梯度的allreduce。
    如果sequence_parallel_enabled为True，这必须为False，因为不执行allreduce。

    sequence_parallel_enabled (bool required): 表示使用了序列并行，
    因此在前向传播中，输入是add gather后的，在反向传播中，输入梯度是reduce scatter后的。
    """
    # 这部分创建了一个名为args的列表，它基本上是函数输入参数的集合。
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel_enabled,
    ]

    # 这部分检查是否已经发出警告。函数使用一个类级别变量warned来记住是否已经向用户显示了警告。
    if not linear_with_grad_accumulation_and_async_allreduce.warned:
        # 这部分检查环境变量CUDA_DEVICE_MAX_CONNECTIONS是否设置为"1"。
        # 如果没有，并且满足某些条件（sequence_parallel_enabled或async_grad_allreduce），
        # 它会发出警告。然后将warned标志设置为True，以便不会重复发出此警告。
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if sequence_parallel_enabled:
                warnings.warn(
                    "When using sequence parallelism it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

            if async_grad_allreduce:
                warnings.warn(
                    "When using async grad allreduce it is recommended to set the "
                    "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 for "
                    "maximum speedup")
                linear_with_grad_accumulation_and_async_allreduce.warned = True

    # 最后，函数调用另一个名为LinearWithGradAccumulationAndAsyncCommunication的类并返回其结果。
    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)

# 在函数外部，初始化属性warned为False。这用于检查是否已经向用户发出警告。
linear_with_grad_accumulation_and_async_allreduce.warned = False
```

解着解析一下`LinearWithGradAccumulationAndAsyncCommunication`这个类的实现（https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L232）：

```python
# 这定义了一个名为LinearWithGradAccumulationAndAsyncCommunication的类，
# 该类继承自torch.autograd.Function。
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    # 使用两个装饰器标记forward方法。其中@staticmethod表示这是一个静态方法，
    # 而@custom_fwd是一个自定义装饰器，用于特定的前向传播操作。
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
    ):
        # 使用上下文对象ctx保存输入和权重，以便在后向传播中使用。
        ctx.save_for_backward(input, weight)
        # 在上下文对象ctx中存储其他变量和标志。
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        # 如果启用了序列并行，则进行以下操作：
        if sequence_parallel:
            # 获取模型并行的world_size（通常是参与并行处理的GPU数量）。
            world_size = get_tensor_model_parallel_world_size()
            # 更改输入的第一个维度以考虑模型并行的全部大小。
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            # 收集所有GPU上的输入。
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group()
            )
            # 更新total_input为收集的数据。
            total_input = all_gather_buffer
        else:
            # 如果不使用序列并行，则total_input仅仅是传入的输入。
            total_input = input

        # 对total_input和weight的转置进行矩阵乘法以计算输出。
        output = torch.matmul(total_input, weight.t())
        # 如果提供了偏置，则将其添加到输出中
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # 从上下文对象中恢复前向传播保存的张量。
        input, weight = ctx.saved_tensors
        # 从上下文对象中恢复偏置使用的信息。
        use_bias = ctx.use_bias

        # 如果启用了序列并行，要如何获取完整的输入数据。
        # 它通过分布式的_all_gather_base函数来异步地聚集所有输入。
        if ctx.sequence_parallel:
            world_size = get_tensor_model_parallel_world_size()
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            handle = torch.distributed._all_gather_base(
                all_gather_buffer, input, group=get_tensor_model_parallel_group(), async_op=True
            )

            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # gather is scheduled before the input gradient computation
            total_input = all_gather_buffer
        # 如果没有启用序列并行，那么完整的输入就是原始输入。
        else:
            total_input = input
        # 通过矩阵乘法计算关于输入的梯度。
        grad_input = grad_output.matmul(weight)

        # 如果启用了序列并行，则等待所有聚集操作完成。
        if ctx.sequence_parallel:
            handle.wait()

        # Doing gather + slicing during the NeMo forward pass can make this tensor
        # not be contiguous. PyTorch only checks if the tensor is contiguous, and only
        # clones it if it's not contiguous:
        # https://github.com/pytorch/pytorch/blob/c47cf9bc7f9e02f649ab4ed53fe4d35732c92ab6/torch/_refs/__init__.py#L2761
        # 这些是注释，提到在NeMo的前向传递中，执行gather和slicing操作可能会导致grad_output张量
        # 不是连续的。PyTorch只检查张量是否是连续的，并且只在不连续时克隆它。
        grad_output = grad_output.contiguous() # 确保grad_output是连续的
        # Convert the tensor shapes to 2D for execution compatibility
        # 将grad_output张量的形状转化为2D，以确保兼容性。
        grad_output = grad_output.view(
            grad_output.shape[0] * grad_output.shape[1], grad_output.shape[2]
        )
        # 同样地，将total_input张量也转化为2D。
        total_input = total_input.view(
            total_input.shape[0] * total_input.shape[1], total_input.shape[2]
        )

        # 如果启用了异步的梯度all-reduce，执行该操作。这是一个分布式操作，用于聚合所有工作节点上的梯度。
        if ctx.async_grad_allreduce:
            # Asynchronous all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # all-reduce is scheduled before the weight gradient computation

        # 如果启用了序列并行，则不应该在此处启用异步all-reduce（由assert语句确保）。
        # 接着，创建一个新的sub_grad_input张量，并执行一个reduce_scatter操作。
        # 这是一个分布式操作，它会将输入的梯度从所有工作节点上聚合到一个工作节点上。
        if ctx.sequence_parallel:
            assert not ctx.async_grad_allreduce
            dim_size = list(input.size())
            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype, device=torch.cuda.current_device(), requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input, grad_input, group=get_tensor_model_parallel_group(), async_op=True
            )
            # Here we rely on CUDA_DEVICE_MAX_CONNECTIONS=1 to ensure that the
            # reduce scatter is scheduled before the weight gradient computation

        # 根据是否启用了梯度累积融合，使用特定的CUDA操作或标准的矩阵乘法来计算权重的梯度。
        # 这个条件检查是否启用了梯度累积融合。梯度累积通常在小批量训练中用于累积梯度以在较大的有效批量上更新模型。
        if ctx.gradient_accumulation_fusion:
            if weight.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input, grad_output, weight.main_grad
                )
            elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    total_input, grad_output, weight.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            # 在梯度累积融合的情况下，设置grad_weight为None，
            # 这意味着梯度已经在前面的CUDA函数中直接更新了（weight.main_grad），所以在这里没有返回值。
            grad_weight = None
        else:
            grad_weight = grad_output.t().matmul(total_input)
        # 如果使用偏置，则计算关于偏置的梯度。
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # 如果启用了序列并行，等待上述操作完成，并返回计算得到的梯度。
        if ctx.sequence_parallel:
            handle.wait()
            return sub_grad_input, grad_weight, grad_bias, None, None, None

        # 如果启用了异步all-reduce，等待all-reduce操作完成。
        if ctx.async_grad_allreduce:
            handle.wait()

        return grad_input, grad_weight, grad_bias, None, None, None
```


可以看到gradient_accumulation_fusion这个优化作用于Linear层中对weight求梯度的时候，调用了apex库提供的2个fuse cuda kernel原地更新了weight的梯度。

# 0x2. fused_weight_gradient_mlp_cuda 实现
fused_weight_gradient_mlp_cuda接口分别为float32和float16/bfloat16提供了2个cuda kernel实现，我们先看一下上层的接口。（https://github.com/NVIDIA/apex/blob/master/csrc/megatron/fused_weight_gradient_dense.cpp）

```cpp
// 定义了一个名为 wgrad_gemm_accum_fp32_cuda_stub 的函数原型。这是一个CUDA C++函数，
// 用于处理float32数据类型的权重梯度累积。该函数接受三个at::Tensor参数：
// input_2d, d_output_2d, 和 d_weight。
void wgrad_gemm_accum_fp32_cuda_stub(
  at::Tensor &input_2d,
  at::Tensor &d_output_2d,
  at::Tensor &d_weight
);

// 定义了一个名为 wgrad_gemm_accum_fp16_cuda_stub 的函数原型，与上面的函数类似，
// 但它是为float16数据类型设计的。
void wgrad_gemm_accum_fp16_cuda_stub(
  at::Tensor &input_2d,
  at::Tensor &d_output_2d,
  at::Tensor &d_weight
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wgrad_gemm_accum_fp32", &wgrad_gemm_accum_fp32_cuda_stub, "wgrad gemm accum in fp32");
    m.def("wgrad_gemm_accum_fp16", &wgrad_gemm_accum_fp16_cuda_stub, "wgrad gemm accum in fp16");
}
```

接下来解析一下wgrad_gemm_accum_fp32这个kernel，对应 https://github.com/NVIDIA/apex/blob/master/csrc/megatron/fused_weight_gradient_dense_cuda.cu 这个文件。

```cpp

// 这个函数是一个封装了NVIDIA cuBLAS库中的cublasGemmEx函数的C++函数，
// 专门用于执行BFloat16（BF16）的矩阵乘法（GEMM）操作。
// 函数的名称为gemmex_wrapper，它的设计意图是提供一个简单的接口，
// 使得PyTorch可以方便地利用cuBLAS中的高效GEMM操作，特别是当使用BFloat16数据类型时。
// BF16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle, // cuBLAS库的句柄，用于管理cuBLAS调用。
    cublasOperation_t transa, 
    cublasOperation_t transb, // 这两个参数描述了两个输入矩阵A和B是否需要转置。
    // 定义了矩阵A, B和输出矩阵C的维度。具体来说，矩阵A的维度为m x k，
    // 矩阵B的维度为k x n，输出矩阵C的维度为m x n。
    int m,
    int n,
    int k,
    const float* alpha, // 标量系数，用于计算alpha * A * B。
    at::BFloat16* A, // 输入矩阵A，它们都是BFloat16数据类型。
    int lda, //  这个参数是矩阵A的leading dim，通常与矩阵的行数相同。
    at::BFloat16* B,
    int ldb,
    const float* beta, // 标量系数，用于计算beta * C。
    float* C, // 输出矩阵C，它是float数据类型。
    int ldc) { // 矩阵C的leading 维度，通常与矩阵C的行数相同。
  // 使用TORCH_CUDABLAS_CHECK宏调用了cublasGemmEx函数。这是cuBLAS库中用于执行混合精度矩阵乘法的函数。
  // cublasGemmEx函数的参数主要用于描述输入和输出矩阵的属性，以及要执行的具体操作。
  // 在这里，输入矩阵A和B都是BFloat16数据类型，而输出矩阵C是float数据类型。
  // CUDA_R_16BF和CUDA_R_32F是枚举值，用于描述矩阵的数据类型。
  // CUBLAS_GEMM_DEFAULT_TENSOR_OP是一个枚举值，指示cuBLAS使用默认的Tensor Core操作来执行GEMM。
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16BF,
      lda,
      B,
      CUDA_R_16BF,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// 类似上面的函数，用于执行FP16的矩阵乘法
// FP16 Tensor core wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// 类似上面的函数，用于执行FP32的矩阵乘法
// FP32 wrapper around cublas GEMMEx
void gemmex_wrapper(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float *alpha,
    float *A,
    int lda,
    float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_32F,
      lda,
      B,
      CUDA_R_32F,
      ldb,
      beta,
      C,
      CUDA_R_32F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// 这个函数wgrad_gemm_accum_fp32_cuda是一个模板函数，用于在CUDA上执行累加的权重梯度计算（矩阵乘法）。
// 它使用了前面提到的gemmex_wrapper函数，该函数是NVIDIA cuBLAS库中的cublasGemmEx函数的封装，
// 用于执行高效的矩阵乘法。
template <typename T>
void wgrad_gemm_accum_fp32_cuda(T *input, T *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim) {
    // 获取当前CUDA cuBLAS句柄。
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    // 获取CUDA Stream。
    cudaStream_t stream;
    // 从cuBLAS句柄获取当前CUDA流。
    cublasGetStream(handle, &stream);
    // 定义矩阵乘法的标量系数，用于计算alpha * A * B + beta * C。
    const float alpha = 1.0;
    const float beta  = 1.0;

    // 使用CUBLAS_OP_N和CUBLAS_OP_T作为参数，表示输入矩阵不需要转置，但d_output矩阵需要转置。
    // 使用输入矩阵input和输出矩阵的梯度d_output作为输入，将结果存储在权重梯度d_weight中。
    gemmex_wrapper(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        in_dim,
        out_dim,
        hidden_dim,
        &alpha,
        input,
        in_dim,
        d_output,
        out_dim,
        &beta,
        d_weight,
        in_dim);
}

// 这是为数据类型at::Half（即半精度浮点型，也称为FP16）显式实例化的wgrad_gemm_accum_fp32_cuda函数。
// 使用此数据类型的版本，可以进行更快速的计算，尤其是在支持FP16计算的硬件上。
template void wgrad_gemm_accum_fp32_cuda<at::Half>(at::Half *input, at::Half *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp32_cuda<float>(float *input, float *d_output, float *d_weight, int in_dim, int hidden_dim, int out_dim);

// 这个函数名为wgrad_gemm_accum_fp32_cuda_stub，从名字中可以看出这是一个为CUDA定义的存根函数。
// 它处理输入的张量，调整它们的维度，然后调用对应的CUDA模板函数来完成具体的操作。
void wgrad_gemm_accum_fp32_cuda_stub(
  at::Tensor &input,
  at::Tensor &d_output,
  at::Tensor &d_weight
) {
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    // 如果input张量的维度大于2，它将最后一个维度以外的所有维度折叠为第一个维度，
    // 使其成为一个2D张量input_2d。否则，它将使用原始input张量。
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    // 类似地，如果d_output张量的维度大于2，它也会进行同样的维度转换。
    // 否则，它会使用原始的d_output张量。
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    // hidden_dim是input_2d的第一个维度的大小。
    const int hidden_dim = input_2d.size(0);
    // in_dim是input_2d的第二个维度的大小。
    const int in_dim = input_2d.size(1);
    // out_dim是d_weight的第一个维度的大小。
    const int out_dim = d_weight.size(0);

    // 使用DISPATCH_FLOAT_HALF_AND_BFLOAT宏来基于input_2d的数据类型调用相应的函数。
    // 这意味着，根据输入数据的数据类型（浮点、半精度或BFloat16），
    // 它将选择正确的版本的wgrad_gemm_accum_fp32_cuda函数进行调用。
    DISPATCH_FLOAT_HALF_AND_BFLOAT(input_2d.scalar_type(), 0, "wgrad_gemm_accum_fp32",
        wgrad_gemm_accum_fp32_cuda<scalar_t_0>(
            input_2d.data_ptr<scalar_t_0>(),
            d_output_2d.data_ptr<scalar_t_0>(),
            d_weight.data_ptr<float>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}
```

注意，在Kernel中这里会将当前的结果累加到先前计算的梯度上，所有这些都在一个操作中完成，这是fuse的思想，可以避免多次访问global memory提升算子的带宽。
# 0x3. 总结
不需要总结，文本很短。


