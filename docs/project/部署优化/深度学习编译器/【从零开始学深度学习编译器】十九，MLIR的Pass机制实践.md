# 0x0. 前言
这个系列的前面几篇文章对MLIR的组件有了一些粗浅的认识，这篇文章不继续讲MLIR的架构。而是从实践的角度带读者来看一下，MLIR帮助我做了什么，这里仍然以OneFlow Dialect为例。在[MLIR：摩尔定律终结的编译器基础结构 论文解读 ](https://mp.weixin.qq.com/s/SLzMKYugrkhQifqahfdVNw) 这篇文章的评论部分已经简单介绍了OneFlow Dialect相关的组件是如何实现的。在实现了OneFlow Dialect的基础上，我继续来介绍一下MLIR的Pass机制是如何助力OneFlow模型训练和推理加速的。

> 从零开始学深度学习编译器系列的文章以及实验代码均整理在这个仓库：https://github.com/BBuf/tvm_mlir_learn，目前已收获300+ star 。感兴趣可以自行查看，如果能点个star就更好啦。

# 0x1. 背景
当前Transformer架构已经成为做AI的算法开发人员和工程师们不得不谈的基础架构。由Transformer基础架构派生出了一系列超大模型如Bert和GPT-2，在业界都有非常大的影响，并且也引领了大模型的潮流。然而大模型的高昂训练成本让很多人甚至很多公司望而却步，通常只能在预训练的大模型上做一些下游任务，因此如何加速大模型训练是十分重要的。在2019年，英伟达成功地构建并训练了最大的语言模型 GPT-2 8B，这一模型包含 83 亿参数量，是 BERT-Large 模型的 24 倍、GPT-2 的 5.6 倍。英伟达将这一模型称为「Megatron」（威震天），还开源了用来训练这一模型的 pytorch 代码：https://github.com/NVIDIA/Megatron-LM。

这篇论文中提到了很多加速大模型训练的手段，特别的如模型并行训练技术，但本人对分布式训练了解很少这里不做介绍。我这里唯一的关注点是在Megatron论文（`https://arxiv.org/pdf/2104.04473.pdf`）的4.2节中提到的编译优化加速模型训练：

![Megatron 4.2节](https://img-blog.csdnimg.cn/ddf22abeb92144e9bafc3afa6c17b1e6.png)

![Megatron 4.2节](https://img-blog.csdnimg.cn/9dbf38e91f95405294d9f64d7aeb038d.png)


这一节编译优化讲的主要是可以通过PyTorch JIT技术来做一些Op融合，比如将bias_add和gelu融合成一个算子，bias_add+dropout融合成一个算子。做了这些算子融合之后，不仅可以避免GPU重复读写数据减少显存占用，还可以减少cuda kernel launch次数对整个计算过程进行加速。

要实现论文中提到的编译优化，需要两个前置条件。一是框架提供了融合Op的实现，二是基于编译器实现一个优化Pass来自动寻找模型中可以融合的Pattern并将其重写为等价的融合Op，达到对计算图进行运行加速的目的。

# 0x2. BiasAdd Dropout以及融合算子简介
在OneFlow中为了对标Megatron的bias_add和dropout fuse，实现了一个`fused_bias_add_mask_scale`算子，做的事情就是将BiasAdd和Dropout融合成一个算子来加速。这个算子的实现过程这里不展开，重点是如何在模型中基于MLIR自动发现这种Pattern并自动将这种Pattern替换为`fused_bias_add_mask_scale`算子。

为了下一节更好的理解融合Pass的做法，这里对bias_add，dropout以及fused_bias_add_mask_scale这三种Op的参数列表进行简要介绍。

- bias_add 算子：

```cpp
>>> import oneflow as flow
>>> x = flow.randn(2, 3)
>>> y = flow.randn(3)
>>> z = flow._C.bias_add(x, y, axis=1)
```

可以看到这个算子有3个参数，一个是输入Tensor，一个是bias Tensor，还有一个axis属性表示需要把bias Tensor附加到输入Tensor的哪个维度上。在Transformer结构中，带偏置的线性层（`nn.Linear`）就是通过一个矩阵乘法（matmul）算子和一个bias_add实现的。

- nn.Dropout 算子：Dropout算子相信大家非常熟悉，不需要多解释，可以参考下方OneFlow算子文档。

![nn.Dropout文档](https://img-blog.csdnimg.cn/c5cc15ecbe3c470ab020e4219bc77571.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)
例子：

```cpp
>>> import numpy as np
>>> import oneflow as flow

>>> m = flow.nn.Dropout(p=0)
>>> arr = np.array(
...    [
...        [-0.7797, 0.2264, 0.2458, 0.4163],
...        [0.4299, 0.3626, -0.4892, 0.4141],
...        [-1.4115, 1.2183, -0.5503, 0.6520],
...    ]
... )
>>> x = flow.Tensor(arr)
>>> y = m(x)
>>> y 
tensor([[-0.7797,  0.2264,  0.2458,  0.4163],
        [ 0.4299,  0.3626, -0.4892,  0.4141],
        [-1.4115,  1.2183, -0.5503,  0.6520]], dtype=oneflow.float32)
```

- fused_bias_add_mask_scale：fused_bias_add_mask_scale算子需要bias_add算子的输入`a`和`b`（bias），然后还需要一个由输入`a`调用`random_mask_like` Op产生的掩码Tensor `mask`作为它的第三个输入，最后还需要bias_add算子的`axis`属性和Dropout的`p`属性。

这里需要解释一下为什么需要mask。其实Dropout算子在实现的时候也会产生两个输出，一个是输出Tensor，一个是mask。这是因为Dropout会根据`p`和我们输入的随机数种子产生一个mask来决定哪些位置的神经元应该保留，哪些位置的神经元置0，为了正确的反向传播的需要我们必须保留这个mask来求取输入Tensor对应的梯度。因此在fused_bias_add_mask_scale Op中，需要将mask显示的传给这个Op，因为这个Op的输出只有一个，不会再输出一个额外的mask了。而这个mask的生成是利用oneflow内部的`random_mask_like` Op来生成的，这个Op接受一个输入Tensor和`p`以及一个随机数种子来产生一个具有一定概率分布的掩码Tensor mask。

# 0x3. Pattern匹配和重写
在了解了这些Op的操作数，属性以及输出之后，我们就可以基于MLIR来做针对BiasAdd和Dropout的Patten自动匹配和重写了。这个功能实现在：`https://github.com/Oneflow-Inc/oneflow/pull/7709`。

首先，我们需要在`oneflow/ir/include/OneFlow/OneFlowPatterns.td`这个文件中基于MLIR的DRR框架写出自动匹配和重写的模板，实现如下：

```cpp
def GetDefaultSeed :
  NativeCodeCall<"mlir::oneflow::GetDefaultSeed($_builder)">;

def FusedBiasAddMaskScale :
  NativeCodeCall<"mlir::oneflow::CreateFusedBiasAddMaskScale($_builder, $0, $1, $2)">;

def IsAddToOutputNone: Constraint<CPred<"mlir::oneflow::IsAddToOutputNone($0)">, "">;

def FusedBiasAddDropoutPattern : Pattern<
  (
    OneFlow_DropoutOp: $dropout_res
    (
      OneFlow_BiasAddOp: $bias_add_res
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $bias_add_op_axis
    ),
    $_add_to_output,
    $dropout_op_name,
    $dropout_device_tag,
    $dropout_device_name,
    $dropout_scope_symbol_id,
    $dropout_hierarchy,
    $dropout_op_rate
  ),
  [
    (
      FusedBiasAddMaskScale
      $dropout_res__0,
      $bias_add_res,
      (
        OneFlow_RandomMaskLikeOp : $mask
          $a,
          $bias_add_op_name,
          $dropout_device_tag,
          $dropout_device_name,
          $dropout_scope_symbol_id,
          $dropout_hierarchy,
          $dropout_op_rate,
          (GetDefaultSeed)
      )
    ),
    (replaceWithValue $mask)
  ],
  [(IsAddToOutputNone $_add_to_output)]
>;
```

`NativeCodeCall`是一个占位代码，我们可以通过`NativeCodeCall`调用我们在Dialect下手写的C++函数。比如：

```cpp
def GetDefaultSeed :
  NativeCodeCall<"mlir::oneflow::GetDefaultSeed($_builder)">;
```

这里就调用了我们在OneFlow Dialect下手写的`GetDefaultSeed`函数，它返回一个OneFlow的DefaultAutoGenerator类生成的随机种子，这个随机种子在Pattern里面作为RandomMaskLikeOp的一个属性被使用：

```cpp
mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter) {
  const auto gen = CHECK_JUST(::oneflow::one::DefaultAutoGenerator());
  return getSI64IntegerAttr(rewriter, (int64_t)gen->current_seed());
}
```

类似的`CreateFusedBiasAddMaskScale`这个函数就是将匹配上的Pattern（BiasAddOp+DropoutOp）重写为FusedBiasAddMaskScaleOp。代码实现如下：

```cpp
::llvm::SmallVector<::mlir::Value, 4> CreateFusedBiasAddMaskScale(::mlir::PatternRewriter& rewriter,
                                                                  OpResult dropout_result,
                                                                  OpResult bias_add_result,
                                                                  Operation* mask) {
  if (auto dropout_op = llvm::dyn_cast<oneflow::DropoutOp>(dropout_result.getDefiningOp())) {
    if (auto bias_add_op = llvm::dyn_cast<oneflow::BiasAddOp>(bias_add_result.getDefiningOp())) {
      SmallVector<Value, 4> operands;
      operands.push_back(bias_add_op.a());
      operands.push_back(bias_add_op.b());
      operands.push_back(mask->getResults()[0]);
      NamedAttrList fused_bias_add_dropout_attributes = dropout_op->getAttrs();
      fused_bias_add_dropout_attributes.append(llvm::StringRef("axis"), bias_add_op.axisAttr());
      fused_bias_add_dropout_attributes.append(llvm::StringRef("scale"), dropout_op.rateAttr());
      fused_bias_add_dropout_attributes.erase(dropout_op.rateAttrName());
      auto res = rewriter
                     .create<oneflow::FusedBiasAddMaskScaleOp>(
                         dropout_op->getLoc(), dropout_op->getResultTypes().front(), operands,
                         fused_bias_add_dropout_attributes)
                     ->getResults();
      // bias_add and dropout op is expected to be erased if it is not used
      return res;
    }
  }
  return {};
}
```

这个函数接收一个PatternRewriter对象和DropoutOp以及BiasAddOp的输出值，然后从这两个值可以取得定义它们的Op，从Op又可以取得对应的操作数和属性等。然后基于PatternRewriter对象完成创建一个新Op的过程，并在当前DropoutOp的位置完成替换，这样就完成了特定Pattern的重写工作。失效的BiasAddOp和DropoutOp由于是NoSideEffect的，在生成的IR中会自动被删掉。


接下来我们看一下IsAddToOutputNone这个约束，`def IsAddToOutputNone: Constraint<CPred<"mlir::oneflow::IsAddToOutputNone($0)">, "">;` 这里使用`CPred`来自定义了一个约束，这个`CPred`里面可以放一个任何返回bool类型的C++函数。这里的实现为：

```cpp
bool IsAddToOutputNone(ValueRange value) { return (int)value.size() > 0 ? false : true; }
```

即判断Dropout Op的`_add_to_output`这个可选的输入是否存在，如果不存在才可以使用我们实现的这个Pass。

除了上面的常规部分之外，这里需要注意两个特殊的点，我单独列出。

## NativeCodeCall的限制引发的问题
我们可以从MLIR的文档查到NativeCodeCall只能返回一个结果。所以在上面的模板匹配和重写的时候我们给重写的部分设置了2个输出，一个是FusedBiasAddMaskScaleOp的输出（目标输出），一个是使用`(replaceWithValue $mask)`定义的占位输出。原因是因为Dropout Op有2个输出，如果这里没有定义一个新的占位输出那么这里模板匹配重写时就会报输出个数不一样的错误。这里使用`replaceWithValue`的原因是它可以简单直接的完成替换一个值（`mlir::Value`）的功能，比较适合这里的占位作用。


## RandomMaskLikeOp为什么要自定义builder
上面的实现中还有一个依赖就是需要自定义RandomMaskLikeOp的builder，新的RandomMaskLikeOp的定义如下：

```cpp
def OneFlow_RandomMaskLikeOp : OneFlow_BaseOp<"random_mask_like", [NoSideEffect, NoGrad, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    OneFlow_Tensor:$like
  );
  let output = (outs
    OneFlow_Tensor:$out
  );
  let attrs = (ins
    DefaultValuedAttr<F32Attr, "0.">:$rate,
    DefaultValuedAttr<SI64Attr, "0">:$seed
  );
  let builders = [
    OpBuilder<(ins
      "Value":$like,
      "StringRef":$op_name,
      "StringRef":$device_tag,
      "ArrayAttr":$device_name,
      "IntegerAttr":$scope_symbol_id,
      "ArrayAttr":$hierarchy,
      "FloatAttr":$rate,
      "IntegerAttr":$seed
    )>
  ];
  let has_check_fn = 1;
  let has_logical_tensor_desc_infer_fn = 1;
  let has_physical_tensor_desc_infer_fn = 1;
  let has_get_sbp_fn = 1;
  let has_data_type_infer_fn = 1;
}
```

自定义builder之后需要将这个builder使用C++来实现一下：

```cpp
void RandomMaskLikeOp::build(mlir::OpBuilder& odsBuilder, mlir::OperationState& odsState,
                             mlir::Value like, StringRef op_name, StringRef device_tag,
                             ArrayAttr device_name, IntegerAttr scope_symbol_id,
                             ArrayAttr hierarchy, mlir::FloatAttr rate, mlir::IntegerAttr seed) {
  odsState.addOperands(like);
  odsState.addAttribute(op_nameAttrName(odsState.name), odsBuilder.getStringAttr(op_name));
  odsState.addAttribute(device_tagAttrName(odsState.name), odsBuilder.getStringAttr(device_tag));
  odsState.addAttribute(device_nameAttrName(odsState.name), device_name);
  if (scope_symbol_id) {
    odsState.addAttribute(scope_symbol_idAttrName(odsState.name), scope_symbol_id);
  }
  if (hierarchy) { odsState.addAttribute(hierarchyAttrName(odsState.name), hierarchy); }
  odsState.addAttribute(rateAttrName(odsState.name), rate);
  odsState.addAttribute(seedAttrName(odsState.name), seed);
  odsState.addTypes(like.getType());
}
```

这样做的原因是因为，这里使用RandomMaskLikeOp生成的mask不是要替换的Dag的最外层Op，所以MLIR无法推断RandomMaskLikeOp的输出值类型（如果是单个Op的话，这个Op的输出类型就是它将要replace的那个Op的输出类型），所以我们要提供一种特殊的，不需要输出类型的builder。这个builder会做类型推断，在这个例子就是从like直接取得类型。即：`odsState.addTypes(like.getType());` 这行代码。

如果不修改这个就会报类型无法匹配的错误，大概长这样：

```python
python3: /home/xxx/oneflow/build/oneflow/ir/llvm_monorepo-src/mlir/lib/IR/PatternMatch.cpp:328: void mlir::RewriterBase::replaceOpWithResultsOfAnotherOp(mlir::Operation*, mlir::Operation*): Assertion `op->getNumResults() == newOp->getNumResults() && "replacement op doesn't match results of original op"' failed
```


# 0x4. 测试
上面就已经讲完了所有的实现细节，我们可以构造一个OneFlow的程序来验证这个IR融合是否正常工作。测试代码如下：


```python
import unittest
import numpy as np
import os
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = "1"
import oneflow as flow
import oneflow.unittest


def do_bias_add_dropout_graph(test_case, with_cuda, prob):
    x = flow.randn(2, 3, 4, 5)
    bias = flow.randn(5)
    dropout = flow.nn.Dropout(p=prob)
    if with_cuda:
        x = x.cuda()
        bias = bias.to("cuda")
        dropout.to("cuda")

    eager_res = dropout(flow._C.bias_add(x, bias, axis=3))

    class GraphToRun(flow.nn.Graph):
        def __init__(self):
            super().__init__()
            self.dropout = dropout

        def build(self, x, bias):
            return self.dropout(flow._C.bias_add(x, bias, axis=3))

    graph_to_run = GraphToRun()
    lazy_res = graph_to_run(x, bias)
    test_case.assertTrue(np.array_equal(eager_res.numpy(), lazy_res.numpy()))

@flow.unittest.skip_unless_1n1d()
class TestBiasAddDropout(oneflow.unittest.TestCase):
    def test_bias_add_dropout_graph(test_case):
        do_bias_add_dropout_graph(test_case, True, 1.0)


if __name__ == "__main__":
    unittest.main()
```

这里使用了nn.Graph对计算过程进行包装，即使用静态图的模式运行整个程序。nn.Graph被构建之后会生成一个Job（OneFlow的原始计算图表示），然后这个Job会被转换为MLIR表达式（OneFlow Dialect）做上面的Fuse Pass再转回Job（优化后的OneFlow计算图表示）后再做训练或者推理。

我们可以看一下使用MLIR FuseBiasAddDropout Pass前后的IR表示。首先是不使用这个Pass的MLIR表达式：

```python
module {
  oneflow.job @GraphToRun_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x4x5xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %output_0 = "oneflow.input"(%arg1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.1_3", output_lbns = ["_GraphToRun_0_input.0.1_3/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [5 : si64]} : (tensor<5xf32>) -> tensor<5xf32>
    %0 = "oneflow.bias_add"(%output, %output_0) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "bias_add-0", output_lbns = ["bias_add-0/out_0"], scope_symbol_id = 4611686018427420671 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>) -> tensor<2x3x4x5xf32>
    %out, %mask = "oneflow.dropout"(%0) {device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "dropout-dropout-1", output_lbns = ["dropout-dropout-1/out_0", "dropout-dropout-1/mask_0"], rate = 1.000000e+00 : f32, scope_symbol_id = 4611686018427428863 : i64} : (tensor<2x3x4x5xf32>) -> (tensor<2x3x4x5xf32>, tensor<2x3x4x5xi8>)
    %output_1 = "oneflow.output"(%out) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    oneflow.return %output_1 : tensor<2x3x4x5xf32>
  }
}
```

然后是启动这个Pass之后获得的MLIR表达式：

```python
module {
  oneflow.job @GraphToRun_0(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<2x3x4x5xf32> {
    %output = "oneflow.input"(%arg0) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.0_2", output_lbns = ["_GraphToRun_0_input.0.0_2/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %output_0 = "oneflow.input"(%arg1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_input.0.1_3", output_lbns = ["_GraphToRun_0_input.0.1_3/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [5 : si64]} : (tensor<5xf32>) -> tensor<5xf32>
    %0 = "oneflow.random_mask_like"(%output) {device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "bias_add-0", rate = 1.000000e+00 : f32, scope_symbol_id = 4611686018427428863 : i64, seed = 4920936260932536 : si64} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %1 = "oneflow.fused_bias_add_mask_scale"(%output, %output_0, %0) {axis = 3 : si32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], op_name = "dropout-dropout-1", output_lbns = ["dropout-dropout-1/out_0", "dropout-dropout-1/mask_0"], scale = 1.000000e+00 : f32, scope_symbol_id = 4611686018427428863 : i64} : (tensor<2x3x4x5xf32>, tensor<5xf32>, tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    %output_1 = "oneflow.output"(%1) {data_type = 2 : i32, device_name = ["@0:0"], device_tag = "gpu", hierarchy = [1], is_dynamic = false, nd_sbp = ["B"], op_name = "_GraphToRun_0_output.0.0_2", output_lbns = ["_GraphToRun_0_output.0.0_2/out"], scope_symbol_id = 4611686018427420671 : i64, shape = [2 : si64, 3 : si64, 4 : si64, 5 : si64]} : (tensor<2x3x4x5xf32>) -> tensor<2x3x4x5xf32>
    oneflow.return %output_1 : tensor<2x3x4x5xf32>
  }
}
```

可以看到上面实现的FuseBiasAddDropout  Pass成功完成了BiasAdd和Dropout Op的融合。

# 0x5. 总结
这篇文章介绍了MLIR的Pass机制的实践，在OneFlow Dialect中已经实现了很多常用的Fuse Op并且使用MLIR来做Pattern Match和Rewrite，从而在不需要用户修改任何代码的情况下无感加速计算图以及节省显存。如果你对这部分很感兴趣，可以到我们的OneFlow仓库中查看。


# 0x6. 资料
- https://github.com/Oneflow-Inc/oneflow
- https://mlir.llvm.org/docs/DeclarativeRewrites/