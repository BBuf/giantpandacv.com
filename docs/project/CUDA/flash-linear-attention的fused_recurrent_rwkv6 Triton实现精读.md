# 0x0. 前言
继续补 [在GPU上加速RWKV6模型的Linear Attention计算](https://mp.weixin.qq.com/s/YXtvafdxB1rVeoy0qJmjyA) 没有写完的内容，对flash-linear-attention库（https://github.com/sustcsonglin/flash-linear-attention）中的fused_recurrent_rwkv6和chunk_rwkv6的前向实现进行解析，也是对Triton写cuda kernel进行继续学习。这里先解读一下fused_recurrent_rwkv6的实现，chunk_rwkv6的实现后续随缘说。

# 0x1. fused_recurrent_rwkv6 naive python实现
还是从naive的python实现看起，https://github.com/sustcsonglin/flash-linear-attention/blob/main/fla/ops/rwkv6/recurrent_naive.py 。fused_recurrent_rwkv6计算算法对应下面的基础python流程：

```python
def naive_recurrent_rwkv6(
    q,
    k,
    v,
    w,
    u,
    initial_state=None,
    output_final_state=False
):
    # 记录输入张量 q 的原始数据类型。
    orig_dtype = q.dtype
    # 将输入张量转换为 32 位浮点数类型。
    q, k, v, w, u = map(lambda x: x.float(), (q, k, v, w, u))
    # 获取query张量的形状信息。
    batch_size, n_heads, seq_len, d_head_k = q.shape
    # 获取值张量的形状信息。
    _, _, _, d_head_v = v.shape
    # 初始化注意力张量为全零张量，形状为 (B, H, D, D)，在 GPU 上进行计算。
    h = torch.zeros(batch_size, n_heads, d_head_k, d_head_v, dtype=torch.float32, device=q.device)
    # 初始化输出张量为全零张量，形状同值张量 v
    o = torch.zeros_like(v)

    # 如果提供了初始状态 initial_state，则将注意力张量 h 更新为初始状态：
    if initial_state is not None:
        h += initial_state

    # 对序列长度进行迭代，每次迭代处理一个位置的输入：
    for i in range(seq_len):
        q_i = q[:, :, i, :] # 获取当前位置的query张量。shape为[B, H, D]
        k_i = k[:, :, i] # 获取当前位置的key张量。shape为[B, H, D]
        v_i = v[:, :, i, :] # 获取当前位置的value张量。shape为[B, H, D]
        # 获取当前位置的权重张量，并使用指数函数进行处理。shape为[B, H, D]
        w_i = w[:, :, i].exp()
        # 计算当前位置的键值乘积，elementwise操作。
        # shape变化为[B, H, D, 1] * [B, H, D, 1] -> [B, H, D, 1]
        kv_i = k_i[..., None] * v_i[..., None, :] 
        # 计算当前位置的注意力加权输出，都是elementwise操作。
        # h的shape为[B, H, D, D]
        # u[None, ..., None]的shape为[1, H, D, 1]
        # q_i[..., None]的shape为[B, H, D, 1]
        # h + u[None, ..., None] * kv_i 的shape为：
        # [B, H, D, D] + [1, H, D, 1] * [B, H, D, 1] ->
        # [B, H, D, D] + [B, H, D, 1] ->
        # [B, H, D, D]
        o_i = (h + u[None, ..., None] * kv_i) * q_i[..., None] 
        # 将当前位置的输出加入到输出张量中。
        # o[:, :, i]的shape为[B, H, D]，o_i.sum(-2)的shape为[B, H, D]
        o[:, :, i] = o_i.sum(-2)
        # 更新注意力张量 h
        # h的shape为[B, H, D, D]
        # w_i[..., None]的shape为[B, H, D, 1]
        # kv_i的shape为[B, H, D, 1]
        # h * w_i[..., None] 的shape为[B, H, D, D]也是element-wise操作
        h = h * w_i[..., None] + kv_i
    return o.to(orig_dtype)
```

q, k, v, w, u等定义如下：

```python
B = 4 # 批量大小（batch size）为 4。
H = 4 # 头数（number of heads）为 4。
L = 1024 # 序列长度（sequence length）为 1024。
D = 100 # 每个头的维度（dimension）为 100。
dtype = torch.float32 # 定义了张量的数据类型为 32 位浮点数。
# q, k, v 分别是查询（query）、键（key）、值（value）的张量，形状为 (B, H, L, D)，
# 使用随机初始化，并且在 GPU 上进行计算。
q = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
k = (torch.randn(B, H, L, D).cuda().to(dtype)).requires_grad_(True)
v = torch.randn(B, H, L, D).cuda().to(dtype).requires_grad_(True)
# w 是一个权重张量，形状同上，通过 torch.nn.functional.logsigmoid
# 函数处理随机初始化的张量得到，同样在 GPU 上计算。
w = torch.nn.functional.logsigmoid(torch.randn(B, H, L, D)).cuda().to(torch.float32).requires_grad_(True)
# u 是一个权重张量，形状为 (H, D)，也是随机初始化并在 GPU 上计算。
u = (torch.randn(H, D).cuda().to(dtype)).requires_grad_(True)
o = naive_recurrent_rwkv6(q, k, v, w, u)
```

> 这里q，k，v的head dim维度我都设置为了D，和RWKV模型里面保持一致，测试文件里面v的维度是2D。

其中B表示的是Batch，H表示Attention头数量，L表示序列长度，D表示Head dim。

从上面的naive_recurrent_rwkv6中关于在序列长度循环中的每个张量的shape分析以及算子类型分析可以发现所有的操作均是Elemenwise操作，这是一个典型的带宽受限问题。

然后从naive的代码还可以得到的一个信息是它在D维度的计算一直都是一个整体，如果我们在D维度进行切分然后计算最后再做一次reduce sum也是数值等价的，这就是fused_recurrent_rwkv6在D维度进行分块计算的依据。
# 0x2. fused_recurrent_rwkv6 python接口定义
首先来看 fused_recurrent_rwkv6 这个api的定义：

```python
# if scale is None, use d_head_qk ** -0.5 by default. Otherwise specify the scale yourself. e.g. scale = 1.0
# 定义了一个函数 fused_recurrent_rwkv6，它接受多个输入张量和参数，并返回两个张量的元组。
# r, k, v, w, u 这些参数分别表示query、key、value、数据相关衰减和奖励。
# scale为缩放因子，默认值为 -1，如果不提供，则默认为 1 / sqrt(K)。
# initial_state 初始状态，默认为 None。
# output_final_state 是否输出最终状态，默认为 False。
# causal: bool = True：是否使用因果注意力，默认为 True。
def fused_recurrent_rwkv6(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    scale: int = -1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    causal: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Args:
        r (torch.Tensor):
            reception of shape `(B, H, T, K)`. Alias: q, query in linear attention.
        k (torch.Tensor):
            keys of shape `(B, H, T, K)`
        v (torch.Tensor):
            values of shape `(B, H, T, V)`
        w (torch.Tensor):
            data-dependent decays of shape `(B, H, T, K)` in log space! Alias: g.
        u (torch.Tensor):
            bonus of shape `(H, K)`
        scale (Optional[int]):
            Scale factor for the RWKV6 attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `(B, H, K, V)`. Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `(B, H, K, V)`. Default: `False`.
    """
    # 如果没有提供缩放因子，则将其设为 1 / sqrt(K)，其中 K 是接收项的最后一个维度大小。
    if scale == -1:
        scale = r.shape[-1] ** -0.5
    # 如果提供了初始状态，则对其进行detach处理，以避免梯度传播到初始状态。
    if initial_state is not None:
        initial_state = initial_state.detach()
    # 调用自定义的 FusedRecurrentRWKV6Function.apply 函数，传入r、k、v、数据相关衰减、奖励、缩放因子、初始状态和输出最终状态参数，返回输出张量和最终状态。
    o, final_state = FusedRecurrentRWKV6Function.apply(r, k, v, w, u, scale, initial_state, output_final_state)
    return o, final_state
```

fused_recurrent_rwkv6中调用的是FusedRecurrentRWKV6Function这个`autograd.Function`，还需要往里看一层。

```python
# 这段代码定义了一个名为 FusedRecurrentRWKV6Function 的自定义 PyTorch 自动求导函数，
# 并实现了其前向传播过程。该类用于计算融合的循环自注意力机制。
class FusedRecurrentRWKV6Function(torch.autograd.Function):
    @staticmethod
    @contiguous
    @custom_fwd
    # 定义前向传播函数 forward，包含上下文 ctx 和输入参数。
    def forward(ctx, r, k, v, w, u, scale=None, initial_state=None, output_final_state=False, reverse=False):
        # q = r：将接收项 r 别名为 q，在后续代码中使用。
        q = r
        # 获取查询张量 q 的形状信息。
        batch_size, n_heads, seq_len, d_head_qk = q.shape
        # 获取值张量 v 的最后一个维度大小。在RWKV里面，d_head_qk和d_head_v相等
        d_head_v = v.shape[-1]
        # 如果未提供缩放因子，默认使用 1 / sqrt(d_head_qk)。
        if scale is None:
            scale = d_head_qk ** -0.5
			 
			 # 计算 d_head_qk 和 d_head_v 的最接近的 2 的次方，且最大不超过 32。
			 # 根据设定的输入shape，这里计算出来就是32
        BK, BV = min(triton.next_power_of_2(d_head_qk), 32), min(triton.next_power_of_2(d_head_v), 32)
        # 计算 d_head_qk 和 d_head_v 分块后的块数。
        # 根据设定的输入shape，这里算出来都是4
        NK, NV = triton.cdiv(d_head_qk, BK), triton.cdiv(d_head_v, BV)
        # 设定阶段数和 warps 数。
        num_stages = 1
        num_warps = 1

        # 创建一个新的空张量 o，用于存储输出。
        o = q.new_empty(NK, batch_size, n_heads, seq_len,
                        d_head_v, dtype=torch.float32)
			 
			 # 如果需要输出最终状态，初始化最终状态张量。
        if output_final_state:
            final_state = q.new_empty(batch_size, n_heads, d_head_qk, d_head_v)
        else:
            final_state = None

        # 定义计算网格的大小。
        grid = (NV, NK, batch_size * n_heads)
        # 调用 Triton kernel进行前向计算，传入必要的参数和张量。
        fused_recurrent_rwkv6_fwd_kernel[grid](
            q, k, v, w, u, o, initial_state, final_state,
            q.stride(1), q.stride(2), q.stride(3),
            v.stride(1), v.stride(2), v.stride(3),
            batch_size, n_heads, seq_len, scale,
            DK=d_head_qk, DV=d_head_v, BK=BK, BV=BV,
            USE_INITIAL_STATE=initial_state is not None,
            STORE_FINAL_STATE=final_state is not None,
            REVERSE=reverse,
            num_warps=num_warps,
            num_stages=num_stages
        )

        # 在第0维上求和，合并输出张量。
        o = o.sum(0)
        ctx.save_for_backward(q, k, v, w, u, initial_state, o)
        ctx.scale = scale
        ctx.reverse = reverse
        # we do not need the gradient of the final state from the next chunk
        # similiar to Trunctated BPTT
        if final_state is not None:
            final_state = final_state.detach()
        return o.to(q.dtype), final_state
```

# 0x3. 可视化

- 1. 计算块的数量
	- NK = ceil(DK / BK)
	- NV = ceil(DV / BV)

	其中：

	- B = 4
	- H = 4
	- L = 1024
	- DK = 100
	- DV = 100
	- BK = 32
	- BV = 32

	那么：

	- NK = ceil(100 / 32) = 4
	- NV = ceil(100 / 32) = 4

- 2. 每个块的内容

每个块会计算一个 batch 和一个 head 上的整个序列长度（L）。

Grid大小：`grid = (NV, NK, B * H)`

每个 `block (i_v, i_k, i_bh)` 对应的实际计算：`i_v` 对应 DV 维度，`i_k` 对应 DK 维度，`i_bh` 对应 (Batch, Head) 的组合。

- 3. 画一张图展示一下Triton的每个分块在计算什么

	- 横轴：`i_k` 从 0 到 3（共 4 个块）
	- 纵轴：`i_v` 从 0 到 3（共 4 个块）
	- 每个格子内：显示每个 block 计算的 (batch, head) 组合

```bash
(0,0)     (1,0)     (2,0)     (3,0)
+---------+---------+---------+---------+
| (B0,H0) | (B1,H0) | (B2,H0) | (B3,H0) |
| (B0,H1) | (B1,H1) | (B2,H1) | (B3,H1) |
| (B0,H2) | (B1,H2) | (B2,H2) | (B3,H2) |
| (B0,H3) | (B1,H3) | (B2,H3) | (B3,H3) |
+---------+---------+---------+---------+

(0,1)     (1,1)     (2,1)     (3,1)
+---------+---------+---------+---------+
| (B0,H0) | (B1,H0) | (B2,H0) | (B3,H0) |
| (B0,H1) | (B1,H1) | (B2,H1) | (B3,H1) |
| (B0,H2) | (B1,H2) | (B2,H2) | (B3,H2) |
| (B0,H3) | (B1,H3) | (B2,H3) | (B3,H3) |
+---------+---------+---------+---------+

(0,2)     (1,2)     (2,2)     (3,2)
+---------+---------+---------+---------+
| (B0,H0) | (B1,H0) | (B2,H0) | (B3,H0) |
| (B0,H1) | (B1,H1) | (B2,H1) | (B3,H1) |
| (B0,H2) | (B1,H2) | (B2,H2) | (B3,H2) |
| (B0,H3) | (B1,H3) | (B2,H3) | (B3,H3) |
+---------+---------+---------+---------+

(0,3)     (1,3)     (2,3)     (3,3)
+---------+---------+---------+---------+
| (B0,H0) | (B1,H0) | (B2,H0) | (B3,H0) |
| (B0,H1) | (B1,H1) | (B2,H1) | (B3,H1) |
| (B0,H2) | (B1,H2) | (B2,H2) | (B3,H2) |
| (B0,H3) | (B1,H3) | (B2,H3) | (B3,H3) |
+---------+---------+---------+---------+

```

- 每个格子内，展示该块处理的 batch 和 head 组合。所有块都会处理整个序列长度 L。
# 0x4. fused_recurrent_rwkv6 triton实现详解

上面的FusedRecurrentRWKV6Function中给输出张量新增了一个维度NK（也就是qk的维度上的分块数），然后kernel计算出输出之后需要在这个维度进行一次reduce sum。此外，grid的大小设置为了`grid = (NV, NK, batch_size * n_heads)`，也就是说不仅会在d_head_qk的维度上进行分块，也会在d_v的维度上进行分块，现在我们讨论下kernel的详细实现。

> 为了代码更好看，我去掉了其中不会用到的REVERSE相关的判断。

```python
@triton.jit
def fused_recurrent_rwkv6_fwd_kernel(
    # B: batch_size, H: n_heads, T: seq_len, D: d_head
    q,  # query [B, H, L, D_head_K]
    k,  # key [B, H, L, D_head_K]
    v,  # value [B, H, L, D_head_V]
    w,  # log gate [B, H, L, D_head_K]
    u,  # bonus [B, H, D_head_K]
    o,  # output [B, H, L, D_head_V]
    # initial hidden state initialization [B, H, D_head_K, D_head_V]
    initial_state,
    final_state,  # final hidden state [B, H, D_head_K, D_head_V]

    s_qk_h,  # stride size: L * D_head_K
    s_qk_t,  # stride size: D_head_K
    s_qk_d,  # stride size: 1

    s_vo_h,  # stride size: L * D_head_V
    s_vo_t,  # stride size: D_head_V
    s_vo_d,  # stride size: 1

    B,  # batch size
    H,  # n_heads
    T,  # seq_len
    scale,  # D_head_K ** -0.5
    BK: tl.constexpr,  # BLOCK SIZE along the K dimension
    BV: tl.constexpr,  # BLOCK SIZE along the V dimension
    DK: tl.constexpr,  # D_head_K
    DV: tl.constexpr,  # D_head_V
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    REVERSE: tl.constexpr,  # whether to do autoregressive modeling in the reverse direction
):
    # i_v，i_k，i_bh：分别是值、键和batch的program ID。
    i_v, i_k, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # i_h：头的索引。
    i_h = i_bh % H

    # p_q，p_k，p_v，p_o，p_w，p_u：分别是查询、键、值、输出、权重和奖励张量的指针位置。
    # 根据program id以及每个张量的stride就可以确定，以p_q为例子，我们知道
    # q的输入shape为[B, H, L, D]所以i_bh * s_qk_h确定了b和h的维度，
    # 再乘上s_qk_h这个b和h维度上的stride就定位到了i_bh所在的L*D的内存空间的起点，
    # 由于这片q的内存空间会被分成D块来计算，所以使用i_k * BK + tl.arange(0, BK)
    # 来定位当前program所在的q的内存空间位置。
    p_q = q + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_k = k + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_v = v + i_bh * s_vo_h + i_v * BV + tl.arange(0, BV)
    # 这一行见后文详细解释
    p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)
    p_w = w + i_bh * s_qk_h + i_k * BK + tl.arange(0, BK)
    p_u = u + i_h * DK + tl.arange(0, BK) + i_k * BK

    # mask_bk，mask_bv：用于确定当前块是否在query/key和value的头维度范围内。
    mask_bk = (i_k * BK + tl.arange(0, BK)) < DK
    mask_bv = (i_v * BV + tl.arange(0, BV)) < DV

    # 初始化隐藏状态 h 为全零张量。
    h = tl.zeros([BV, BK], dtype=tl.float32)
    
    # 见后文的详细注释
    mask_kv = mask_bk[None, :] & mask_bv[:, None]

    # 如果使用初始状态，加载初始状态值并加到隐藏状态 h。
    if USE_INITIAL_STATE:
        # 注意，这里的p_init_s是二维的
        p_init_s = initial_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        h += tl.load(p_init_s, mask=mask_kv, other=0).to(tl.float32)

    _u = tl.load(p_u, mask=mask_bk, other=0).to(tl.float32)
    for _ in range(0, T):
        _k = tl.load(p_k, mask=mask_bk, other=0).to(tl.float32)
        _v = tl.load(p_v, mask=mask_bv, other=0).to(tl.float32)
        _q = tl.load(p_q, mask=mask_bk, other=0).to(tl.float32) * scale
        _w = tl.load(p_w, mask=mask_bk, other=0).to(tl.float32)
        _w = tl.exp(_w)
        _kv = _k[None, :] * _v[:, None]
        _o = (h + _kv * _u[None, :]) * _q[None, :]
        _o = tl.sum(_o, axis=1)
        h = h * _w[None, :]
        h += _kv
        tl.store(p_o, _o.to(p_o.dtype.element_ty), mask=mask_bv)
        p_q += DK
        p_k += DK
        p_o += DV
        p_v += DV
        p_w += DK

    if STORE_FINAL_STATE:
        p_final_s = final_state + i_bh * DK * DV + \
            (i_k * BK + tl.arange(0, BK)[None, :]) * \
            DV + (i_v * BV + tl.arange(0, BV)[:, None])
        tl.store(p_final_s, h.to(p_final_s.dtype.element_ty), mask=mask_kv)
```

详细解析一下`mask_kv = mask_bk[None, :] & mask_bv[:, None]`：`mask_bk` 是一个一维的掩码，表示每个线程块在查询/键张量的头维度范围内的布尔值。`mask_bv` 也是一个一维的掩码，表示每个线程块在值张量的头维度范围内的布尔值。现在，我们想要创建一个二维的掩码 `mask_kv`，使得它在查询/键和值的头维度范围内的元素为 True，而不在范围内的元素为 False。因此，我们使用广播（broadcasting）来组合这两个一维的掩码，以创建一个二维的掩码矩阵。具体来说：
- `mask_bk[None, :] `将 `mask_bk `变形为一个二维矩阵，其中每行都是 `mask_bk` 的副本。
- `mask_bv[:, None]` 将 `mask_bv` 变形为一个二维矩阵，其中每列都是 `mask_bv` 的副本。
- 通过按位与运算符 & 对这两个二维矩阵进行按位与操作，生成一个新的二维掩码矩阵 `mask_kv`。

另外需要特别注意的是`p_o = o + (i_bh + i_k * B * H) * s_vo_h + i_v * BV + tl.arange(0, BV)`这行代码，在kernel执行阶段输出的shape是`[N_K, B, H, L, D]`，所以这里多了一个`i_k * B * H`来定位输出指针位置，并且计算之后我们会在`N_K`维度做reduce sum以获得最终的结果。


# 0x5. 总结
这就是本片文章介绍的所有内容了，希望讲清楚了这个计算过程，同时我们也可以发现使用Triton实现任务确实很简洁，并且相比于使用CUDA也相对简单。


