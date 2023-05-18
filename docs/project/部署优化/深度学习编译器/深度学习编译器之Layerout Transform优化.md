
> 更多的深度学习编译器知识可以在 https://github.com/BBuf/tvm_mlir_learn 找到。同时也维护了一个cuda学习仓库 https://github.com/BBuf/how-to-optim-algorithm-in-cuda 以及一个如何学习深度学习框架（PyTorch和OneFlow）的学习仓库，https://github.com/BBuf/how-to-learn-deep-learning-framework , 有需要的小伙伴可以**点一点star** 。在https://github.com/BBuf/how-to-optim-algorithm-in-cuda/tree/master/large-language-model-note 这个目录下收集了一系列和LLM训练，推理相关的文章。

> 在本文的描述中，存在一些接口和Interface的混用，这两个是一样的都表示MLIR的Interface。
# 0x0. 背景
继续深度学习编译器的优化工作解读，本篇文章要介绍的是OneFlow系统中如何基于MLIR实现Layerout Transform。在2D卷积神经网络中，除了NCHW数据格式之外一般还存在NHWC的数据格式，对于卷积操作来说使用NHWC格式进行计算可能会获得更好的性能。但深度学习网络的训练一般来说是采用NCHW进行的，我们一般只有在推理时才做NCHW到NHWC的Layerout Transform。这里存在两个问题：首先对于一个算子比如Conv2D，它以NCHW方式训练时保存的权重格式是[out_channels, in_channels, *kernel_size]，但是要以NHWC格式进行推理时我们需要对权重的格式进行转换；然后对于没有权重的算子来说，我们也需要尽量的让算子支持NHWC的运算，来减少因为卷积算子前后插入的Transpose操作带来的额外开销。举个例子，假设有如下的一个小网络 x->conv->relu->conv->relu->out，如果我们要以NHWC格式执行那么我们除了对2个卷积的权重进行改动之外，我们还需要在conv前后插入transpose来修改输入到conv算子的数据格式，也就是**x->transpose(0, 2, 3, 1)->conv->transpose(0, 3, 1, 2) -> relu -> transpose(0, 2, 3, 1)->conv->transpose(0, 3, 1, 2) -> relu->out**。然后细心的读者可以发现，实际上这里存在很多冗余的Transpose，因为ReLU是支持以NHWC格式进行运算的，那么这个网络可以化简为**x->transpose(0, 2, 3, 1)->conv->relu->conv->relu->transpose(0, 3,  1, 2)->out**。这样可以减少一半的Transpose Op开销。

之所以要做transpose的化简是因为transpose算子本身也有运行以及调度的开销，如果我们不尽量减少transpose的个数，那么因为改用NHWC带来的计算加速可能会被 Transpose 的开销掩盖住。我们基于OneFlow实现了上述的Layerout Transform优化，以下给出测试结果。

在V100上对这个优化进行了测试，测试代码见 https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/test/OneFlow/auto_nhwc/test_resnet101_benchmark.py ，性能结果如下：

- 开启nn.Graph的AMP选项。
- 网络选取ResNet101，对其做前向推理。


| batch_size | nchw | auto nhwc |
| ---------- | ---- | --------- |
| 16         | 14s  | 13s       |
| 32         | 24s  | 22s       |
| 64         | 44s  | 38s       |

在BatchSize=64时得到了13.6%的加速，随着BatchSize减少加速比会减小，但始终会保持一些加速。需要注意的是，这里对权重参数部分提前进行了transpose，所以这部分是没有额外开销的。实际上，我们采用了常量折叠的方式来完成，这个下篇文章再讲。

# 0x1. 实现解析
 在实现上主要需要搞定3个问题，第一个是如何确定哪些算子支持NHWC的运算，第二个是插入Transpose算子，第三个是消除多余的Transpose对。

## 0x1.1 基于Interface确定哪些算子支持NHWC运算
在OneFlow中如果我们要让某个Op支持NHWC的计算，只需在Op定义时声明一个NCHWCompatibleInterface。以卷积为例：

```cpp
def OneFlow_Conv2DOp : OneFlow_ConvolutionBaseOp<"conv2d", [NoMemoryEffect, AttrSizedOperandSegments, DeclareOpInterfaceMethods<UserOpCompatibleInterface>, DeclareOpInterfaceMethods<NCHWCompatibleInterface>]> {}
```
这里的 DeclareOpInterfaceMethods<NCHWCompatibleInterface> 表示这个 Operator 实现了 NCHWCompatibleInterface 接口,该接口定义了与 NCHW 格式兼容的 Operator 需要实现的方法。

我们想让其它的任意 Op 支持 NHWC 的运算，只需要定义这个接口并且重写这个接口的成员函数即可，接下来我们看一下NCHWCompatibleInterface 的定义。

```cpp
def NCHWCompatibleInterface : OpInterface<"NCHWCompatible"> {
  let description = [{
    Interface of NCHW compatibility
  }];

  let methods = [
    InterfaceMethod<"",
        "bool", "IsNCHW", (ins)
    >,
    InterfaceMethod<"Create NHWC op and return the new op's results to be transposed",
        "llvm::SmallVector<mlir::Value, 4>", "NchwToNhwc", (ins "llvm::SmallVector<mlir::Value, 4>": $transposed_inputs, "PatternRewriter&": $rewriter)
    >,
    InterfaceMethod<"",
        "llvm::DenseSet<mlir::Value>", "OperandsToTranspose", (ins)
    >,
    InterfaceMethod<"",
        "llvm::DenseSet<mlir::Value>", "ResultsToTranspose", (ins)
    >,
  ];
  let cppNamespace = "::mlir::oneflow";
}
```
这个接口继承自 OpInterface 接口, OpInterface 是 MLIR 框架中描述 Operator Interface 的基类。NCHWCompatibleInterface 表示一个与 NCHW 格式兼容的 Operator Interface。NCHWCompatibleInterface定义了几个方法：
- IsNCHW: 返回一个 bool 值, 表示当前的 Operator 在什么条件下是处理输入为 NCHW 格式的数据。
- NchwToNhwc: 接受 Transpose 后的输入和重写器 (rewriter), 用于从 NCHW 格式转换为 NHWC 格式。
- OperandsToTranspose: 返回需要 Transpose 的输入值集合。
- ResultsToTranspose:返回需要 Transpose 的输出值集合。

接下来我们看一下Conv2D Op对应的 NCHWCompatibleInterface 接口实现：

```cpp
bool Conv2DOp::IsNCHW() { return this->getDataFormat().str() == "channels_first"; }

llvm::DenseSet<Value> Conv2DOp::OperandsToTranspose() {
  if (this->get_addToOutput()) {
    return {this->getIn(), this->getWeight(), this->get_addToOutput()};
  } else {
    return {this->getIn(), this->getWeight()};
  }
}

llvm::DenseSet<Value> Conv2DOp::ResultsToTranspose() { return {this->getOut()}; }

llvm::SmallVector<Value, 4> Conv2DOp::NchwToNhwc(llvm::SmallVector<Value, 4> value,
                                                 PatternRewriter& rewriter) {
  auto conv_op = *this;
  SmallVector<Value, 4> operands;
  operands.push_back(value[0]);
  operands.push_back(value[1]);
  if (conv_op.getBias()) operands.push_back(conv_op.getBias());
  if (this->get_addToOutput()) { operands.push_back(value[2]); }
  NamedAttrList attributes = conv_op->getAttrs();
  attributes.set(conv_op.getDataFormatAttrName(), rewriter.getStringAttr("channels_last"));
  auto res = rewriter
                 .create<oneflow::Conv2DOp>(conv_op.getLoc(), getNHWCResultTypes(conv_op), operands,
                                            attributes)
                 ->getResults();
  llvm::SmallVector<Value, 4> results;
  results.push_back(res[0]);
  return results;
}
```

其中，IsNCHW 方法返回一个 bool 值,表示该 Conv2DOp Operation 是否使用 NCHW 格式。它通过检查 Operation 的data_format 属性来判断。OperandsToTranspose 方法返回需要 Transpose 的输入值集合。对于 Conv2DOp 来说,主要输入包括input、weight、bias(可选) 和 addto_output(可选)，其中bias不需要 Transpose，并且这个addto_output是OneFlow的一个特殊的输出用来做算子融合读者可以忽略。ResultsToTranspose 方法返回需要 Transpose 的输出值集合。对于 Conv2DOp 来说,仅有一个输出, 所以返回输出特征图的值。NchwToNhwc 方法接受 NCHW 格式的输入值和重写器,并返回 NHWC 格式的结果值。它通过创建一个新的 Conv2DOp Operation, 并将 data_format 属性设置为 channels_last, 来实现从 NCHW 到 NHWC 的转换。

## 0x1.2 插入Transpose算子

接下来就是贪心的给网络里的算子插入Transpose算子，这里的思路是我们尽可能的对网络里面的所有算子都前后分别插入一个Transpose，这样的话在消除Transopose对的时候才能获得最优的解。给网络中的算子插入Transpose的逻辑如下面的Pattern代码所述：

```cpp
struct AutoNhwcPattern : public OpInterfaceRewritePattern<NCHWCompatible> {
  explicit AutoNhwcPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<NCHWCompatible>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(NCHWCompatible op, PatternRewriter& rewriter) const override {
    if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
      for (mlir::Value operand : op.OperandsToTranspose()) {
        if (operand.getType().cast<mlir::RankedTensorType>().getShape().size() != 4) {
          return failure();
        }
      }
      const auto device_name = OpTrait::IsOpConfCompatible<void>::getDeviceTag(op)
                                   .cast<mlir::StringAttr>()
                                   .getValue()
                                   .str();
      if (device_name == "cpu") { return failure(); }
    }
    llvm::SmallVector<int32_t> perm = getChannelLastTransposePerm();
    llvm::SmallVector<int32_t> result_perm = getChannelFirstTransposePerm();

    NamedAttrList transpose_attributes;
    if (InitTransposeAttributes(op, transpose_attributes, rewriter).succeeded()) {
      transpose_attributes.append(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, perm));
    } else {
      return failure();
    }
    // when op op has no sense of data_format and pre op is transpose, we greedily insert transpose
    // into this op, seeking more opportunities to eliminate transpose pattern.
    const bool greedily_transpose_flag = !op.IsNCHW() && IsInsertTransposeOpBefore(op, rewriter);

    if (op.IsNCHW() || greedily_transpose_flag) {
      // create transpose op for input operand
      SmallVector<Value, 4> tranposed_operands;
      llvm::DenseSet<Value> operand_transpose = op.OperandsToTranspose();
      int num_transposed_operand = 0;
      for (Value operand : op->getOperands()) {
        if (operand_transpose.find(operand) != operand_transpose.end()) {
          SmallVector<Value, 4> input_res = getInputOperandTransposeOp(
              op, operand, transpose_attributes, num_transposed_operand, rewriter);
          tranposed_operands.push_back(input_res[0]);
          num_transposed_operand += 1;
        }
      }
      // create NHWC op
      SmallVector<Value, 4> created_results = op.NchwToNhwc(tranposed_operands, rewriter);
      // create transpose op for results
      int num_transposed_result = 0;
      transpose_attributes.set(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, result_perm));
      llvm::DenseSet<Value> transpose_result = op.ResultsToTranspose();

      for (Value result : op->getOpResults()) {
        if (transpose_result.find(result) != transpose_result.end()) {
          if (auto result_transpose_op =
                  getResultTransposeOp(op, created_results[num_transposed_result],
                                       transpose_attributes, num_transposed_result, rewriter)) {
            result.replaceAllUsesWith(result_transpose_op);
            num_transposed_result += 1;
          } else {
            return failure();
          }
        }
      }
    }
    return success();
  }
};
```

首先 AutoNhwcPattern 类继承自 OpInterfaceRewritePattern，OpInterfaceRewritePattern 是一个用于重写 Operation 的基类。AutoNhwcPattern 针对实现了 NCHWCompatible Interface 的 Operation 进行重写,以实现 NCHW 到 NHWC 的格式转换。然后，AutoNhwcPattern 重写了 matchAndRewrite 方法。该方法会在遇到 NCHWCompatible Interface 的 Operation 时被调用,来实现从 NCHW 到 NHWC 的转换。接下来，matchAndRewrite 方法首先会检查 Operation 是否满足转换条件,如是否 4 维、是否在 CPU 设备上等。如果不满足则返回 failure。如果满足, matchAndRewrite 方法会获取 NCHW 到NHWC 和 NHWC 到 NCHW 的转换顺序。并初始化 Transpose Operation 的属性。然后对于当前 Op 是 NCHW 格式或者这个 Op 的前一个 Op 是Transpose Op，这里都进行插入 Transpose Op的操作来获得更多的优化机会。

这里还涉及到几个相关的工具函数，我们也解释一下：

```cpp
llvm::SmallVector<int32_t> getChannelLastTransposePerm() { return {0, 2, 3, 1}; }

llvm::SmallVector<int32_t> getChannelFirstTransposePerm() { return {0, 3, 1, 2}; }

llvm::SmallVector<mlir::Value, 4> getInputOperandTransposeOp(NCHWCompatible op, Value val,
                                                             NamedAttrList transpose_attributes,
                                                             int num_transposed_operand,
                                                             PatternRewriter& rewriter) {
  std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                               + "_transpose_input_" + std::to_string(num_transposed_operand);
  transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                           rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> input_operands;
  input_operands.push_back(val);
  auto res = rewriter
                 .create<oneflow::TransposeOp>(op.getLoc(), getNHWCType(val.getType()),
                                               input_operands, transpose_attributes)
                 ->getResults();
  return res;
}

TransposeOp getResultTransposeOp(NCHWCompatible op, Value val, NamedAttrList transpose_attributes,
                                 int num_transposed_result, PatternRewriter& rewriter) {
  std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                               + "_transpose_output_" + std::to_string(num_transposed_result);
  transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                           rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> operands;
  operands.push_back(val);
  TransposeOp transpose_op = rewriter.create<oneflow::TransposeOp>(
      op.getLoc(), getNCHWType(val.getType()), operands, transpose_attributes);
  return transpose_op;
}

bool IsInsertTransposeOpBefore(NCHWCompatible op, PatternRewriter& rewriter) {
  bool insert_transpose_op_flag = false;
  for (mlir::Value operand : op->getOperands()) {
    TransposeOp transposeInputOp = operand.getDefiningOp<TransposeOp>();
    if (!transposeInputOp) continue;
    const auto perm = transposeInputOp.getPermAttr();
    if (perm.size() == 4 && perm[0] == rewriter.getSI32IntegerAttr(0)
        && perm[1] == rewriter.getSI32IntegerAttr(3) && perm[2] == rewriter.getSI32IntegerAttr(1)
        && perm[3] == rewriter.getSI32IntegerAttr(2)) {
      insert_transpose_op_flag = true;
      break;
    }
  }
  return insert_transpose_op_flag;
}
```

其中 getChannelLastTransposePerm 和 getChannelFirstTransposePerm 方法分别返回 NHWC 到 NCHW 和 NCHW 到NHWC 的转换顺序。getInputOperandTransposeOp 方法为 Operation 的输入创建一个Transpose Operation。它使用输入值、Transpose属性 和 重写器创建一个 TransposeOp , 并返回其结果。类似的，getResultTransposeOp 方法为 Operation 的输出创建一个Transpose Operation。它使用输出值、Transpose属性和重写器创建一个TransposeOp,并返回该Operation。 IsInsertTransposeOpBefore方法检查Operation的输入是否已有 Transpose Operation。如果有,并且该 Transpose Operation 将 NHWC 转为 NCHW, 则返回 true, 否则返回false。
## 0x1.3 消除多余的Transpose对
接下来，我们需要把插入Transpose Op的图中所有相邻的Transpose对尽可能的消除，代码实现如下：

```cpp
bool IsRedundantTransposeMatch(ArrayAttr pre, ArrayAttr afe, mlir::PatternRewriter& rewriter) {
  const auto prePerm = pre.getValue().vec();
  const auto afePerm = afe.getValue().vec();
  if (prePerm.size() == 4 && afePerm.size() == 4) {
    // handle nchw->nhwc->nchw: (0, 2, 3, 1) -> (0, 3, 1, 2)
    if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[3] && prePerm[2] == afePerm[1]
        && prePerm[3] == afePerm[2] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
        && prePerm[1] == rewriter.getSI32IntegerAttr(2)
        && prePerm[2] == rewriter.getSI32IntegerAttr(3)
        && prePerm[3] == rewriter.getSI32IntegerAttr(1))
      return true;
    // handle nhwc->nchw->nhwc: (0, 3, 1, 2) -> (0, 2, 3, 1)
    if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[2] && prePerm[2] == afePerm[3]
        && prePerm[3] == afePerm[1] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
        && prePerm[1] == rewriter.getSI32IntegerAttr(3)
        && prePerm[2] == rewriter.getSI32IntegerAttr(1)
        && prePerm[3] == rewriter.getSI32IntegerAttr(2))
      return true;
  }
  return false;
}

struct AutoNhwcEliminateRedundantTransposePattern : public mlir::OpRewritePattern<TransposeOp> {
  explicit AutoNhwcEliminateRedundantTransposePattern(mlir::MLIRContext* context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(TransposeOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    if (!transposeInputOp
        || !IsRedundantTransposeMatch(op.getPermAttr(), transposeInputOp.getPermAttr(), rewriter)) {
      return failure();
    }
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};
```

IsRedundantTransposeMatch 方法检查两个 Transpose Operation的顺序是否会导致冗余。它通过比较两个 Transpose 的 perm 属性来判断。类似 AutoNhwcPattern ，AutoNhwcEliminateRedundantTransposePattern 类继承自 OpRewritePattern 。它对TransposeOp 进行重写以实现 Transpose 消除。如果顺序是 NHWC->NCHW->NHWC 或NCHW->NHWC->NCHW , 则判定为冗余 Transpose 。如果输入也来自TransposeOp且两个 Transpose 顺序导致冗余,matchAndRewrite方法会用TransposeOp的输入替换TransposeOp。实现 Transpose 消除。matchAndRewrite 方法首先获取 TransposeOp 的输入,并检查该输入是否也来自一个 TransposeOp。如果不是, 或两个 Transpose 的顺序不导致冗余, 则返回 failure。最后返回 success 表示成功消除冗余 Transpose 。

最终，上面介绍的2个Pass都被封装到 AutoNhwcPass 中作用在 MLIR 的计算图上完成全局优化。从下面的代码可以看到这个优化只有在打开 ONEFLOW_MLIR_PREFER_NHWC 环境变量时才正常生效。

```cpp
void populateAutoNhwcPatterns(::mlir::RewritePatternSet& patterns) {
  bool enable_nhwc = ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PREFER_NHWC", false);
  if (enable_nhwc) {
    patterns.add<AutoNhwcPattern>(patterns.getContext());
    patterns.add<AutoNhwcEliminateRedundantTransposePattern>(patterns.getContext());
  }
}

class AutoNhwcPass : public AutoNhwcPassBase<AutoNhwcPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    oneflow::populateAutoNhwcPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};
```

## 补充：0x1.4 weight的transpose消除
这里还需要粗略的说明一下对于 weight 的 transpose 是如何处理的。在0x1.2中我们为 weight（常量constant op） 也插入了 Transpose Op，然后我们知道 weight 是常量，所以针对 weight 的 Transpose Op 完全可以在编译期折叠起来。这个过程是在 https://github.com/Oneflow-Inc/oneflow/blob/master/oneflow/ir/oneflow-translate/lib/OneFlow/MLIROneFlowTranslation.cpp#L808-L811  这里完成的，我们后面会单独介绍一下 Constant Folding 的实现。

# 0x2. 结论
本文介绍了一下OneFlow的编译器中的 Layerout Transform，这个技术在后来 OneFlow 版本的 Stable Diffusion 中也发挥了重要作用，提升了推理速度。在 TVM 的 Ansor 中也有类似的优化，通过将不同的 Layerout 设定为 Op 的 strategy 进而影响 Op 的 schedule，在搜索的时候考虑到 Layerout Transform 来获得更大的搜索空间和更好的结果。在处理Transpose 额外开销的方法并不是唯一的，这里只是采用了一种个人认为比较简单的方式，读者们如果有类似需要可以自由发挥。




