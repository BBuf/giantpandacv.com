# 0x0. 前言
这篇笔记是阅读Toy Tutorials的第五章之后总结的，这一节主要讲的是将Toy Dialect Lowering的部分Operation Lowering到Affine Dialect，MemRef Dialect和Standard Dialect，而`toy.print`操作保持不变，所以又被叫作部分Lowering。通过这个Lowering可以将Toy Dialect的Operation更底层的实现逻辑表达出来，以寻求更多的优化机会，得到更好的MLIR表达式。


# 0x1. Dialect转换

MLIR有众多的Dialect，所以MLIR提供了一个统一的`DialectConversion`框架来支持这些Dialect互转。这个框架允许将待转换的非法操作转换为合法的操作。为了使用这个框架，我们需要提供两个条件（还有一个可选条件）。

- **转换目标（Conversation Target）**。明确哪些 Dialect 操作是需要合法转换的，不合法的操作需要重写模式(rewrite patterns )来进行合法化。
- **一组重写模式（Rewrite Pattern）**。这是用于将非法操作转换为零个或多个合法操作的一组模式。
- **类型转换器 （Type Converter）（可选）**。如果提供，则用于转换块参数的类型。这一节将不需要此转换。

下面我们来分步介绍具体是如何将目前的MLIR表达式部分Lowering为新的MLIR表达式，并寻求更多的优化机会。

## 第一步，定义转换目标（Conversion Target）
为了寻求新的优化机会，需要将Toy Dialect中计算密集型操作转换成Affine，MemRef 和 Standard Dialects 的操作组合。代码实现在：`mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp` 。

```cpp
void ToyToAffineLoweringPass::runOnFunction() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  mlir::ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arithmetic`, `MemRef`, and `Standard` dialects.
  target.addLegalDialect<AffineDialect, arith::ArithmeticDialect,
                         memref::MemRefDialect, StandardOpsDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `toy.print`, as *legal*. `toy.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<ToyDialect>();
  target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return type.isa<TensorType>(); });
  });
  ...
}
```

这个函数实现了将Toy Dialect中的部分计算密集型操作（比如矩阵乘法操作）进行Lowering。我们的目标是要将原始的MLIR表达式Lowering到新的表达式，这里是将原来的计算密集型操作变换为更加靠近底层的操作。首先用`mlir::ConversionTarget target(getContext())`来定义转换的目标。然后定义在MLIR表达式Lowering的过程中合法的目标，包括特定的Operation或 Dialect，在这个例子中，将MLIR表达式下降到由Affine、MemRef 和 Standard Dialect 三种Dialect下Operation的组合。所以在代码中使用`target.addLegalDialect`函数将这三个Dialect添加为合法的目标。同时还要定义非法的Dialect，这里定义之前的Toy Dialect为非法的目标，即`target.addIllegalDialect<ToyDialect>()`，这就暗示如果转换结束MLIR表达式中还存在Toy的操作，则Lowering失败。

一个特殊的点是，在此过程中`toy.print`在新的目标Dialect中是不支持的，所以这里需要进行保留，不进行Lowering，即`target.addLegalOp<PrintOp>()`。另外，由于MLIR中单个操作的定义始终优于Dialect的定义，因此上面代码中定义合法，非法以及`PrintOp`的声明顺序是可随意变的。

## 第二步，明确转换模式（Conversion Patterns）

在定义了转换目标之后，我们可以定义如何将非法操作转换为合法操作。 与第 3 章介绍的规范化框架类似，DialectConversion 框架也使用 RewritePatterns 来执行转换逻辑。 这些模式可能是之前看到的 RewritePatterns 或特定于转换框架 ConversionPattern 的新型模式。 **ConversionPatterns** 与传统的 RewritePatterns 不同，因为它们接受一个额外的操作数参数，其中包含已重新映射/替换的操作数。官方文档中给出了Lowering Toy Dialect 中 transpose 操作的例子，代码如下。


```cpp
/// Lower the `toy.transpose` operation to an affine loop nest.
struct TransposeOpLowering : public mlir::ConversionPattern {
  TransposeOpLowering(mlir::MLIRContext *ctx)
      : mlir::ConversionPattern(TransposeOp::getOperationName(), 1, ctx) {}

  /// Match and rewrite the given `toy.transpose` operation, with the given
  /// operands that have been remapped from `tensor<...>` to `memref<...>`.
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op, ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Call to a helper function that will lower the current operation to a set
    // of affine loops. We provide a functor that operates on the remapped
    // operands, as well as the loop induction variables for the inner most
    // loop body.
    lowerOpToLoops(
        op, operands, rewriter,
        [loc](mlir::PatternRewriter &rewriter,
              ArrayRef<mlir::Value> memRefOperands,
              ArrayRef<mlir::Value> loopIvs) {
          // Generate an adaptor for the remapped operands of the TransposeOp.
          // This allows for using the nice named accessors that are generated
          // by the ODS. This adaptor is automatically provided by the ODS
          // framework.
          TransposeOpAdaptor transposeAdaptor(memRefOperands);
          mlir::Value input = transposeAdaptor.input();

          // Transpose the elements by generating a load from the reverse
          // indices.
          SmallVector<mlir::Value, 2> reverseIvs(llvm::reverse(loopIvs));
          return rewriter.create<mlir::AffineLoadOp>(loc, input, reverseIvs);
        });
    return success();
  }
};
```

这段代码就是将`tor.transpose`操作Lowering到Affine loop nest。这里的`lowerOpToLoops`函数以及`TransposeOpAdaptor`的定义是看懂这段代码的关键，感兴趣请查看一下源码。

## 第三步，在Lowering的模式集合中添加上面定义的转换模式
这段代码在`mlir/examples/toy/Ch5/mlir/LowerToAffineLoops.cpp`中：

```cpp
void ToyToAffineLoweringPass::runOnFunction() {
  ...

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<..., TransposeOpLowering>(&getContext());

  ...
```

这里提供了一个模式集合`patterns`来Lowering Toy Dialect的操作，然后在这个集合中添加一系列的Lowering模式，如这里的`TransposeOpLowering`。

## 第四步，执行真正的Lowering过程
明确了Lowering模式之后就可以执行真正的Lowering过程了。`DialectConversion`框架提供了几种不同的Lowering模式，这里使用的是部分Lowering，因为这里不会对`toy.print`操作进行Lowering，因为它不是计算密集形的Operation，只是一个打印操作。下面代码中的`mlir::applyPartialConversion(function, target, patterns)`表示对当前的MLIR表达式中的Operation应用了Lowering。

```cpp
void ToyToAffineLoweringPass::runOnFunction() {
  ...

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our *illegal*
  // operations were not converted successfully.
  auto function = getFunction();
  if (mlir::failed(mlir::applyPartialConversion(function, target, patterns)))
    signalPassFailure();
}

```

## 第五步，部分Lowering的注意事项
在Lowering过程中，我们从值类型 `TensorType` 转换为已分配（类似缓冲区）的类型 `MemRefType`。但是对于`toy.print`操作，这里不想Lowering，因为这里主要是处理一些计算密集型算子并寻求优化机会，`toy.print`只是一个有打印功能的算子。因此这一章节不会Lowering这个算子。但我们知道`toy.print`操作的定义中只支持打印输入类型为`F64Tensor`的输入数据，所以现在为了能将其和MemRef Dialect联系，我们需要为其增加一个`F64MemRef`类型。即修改`mlir/examples/toy/Ch5/mlir/Ops.td`中`toy.print`操作的定义：

```cpp
def PrintOp : Toy_Op<"print"> {
  ...
  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);
  ...
}
```

## 第六步，将上面定义好的部分Lowering功能加到优化pipline里面

这部分代码在`mlir/examples/toy/Ch5/toyc.cpp`中：

```cpp
/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::toy::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

if (isLoweringToAffine) {
	mlir::OpPassManager &optPM = pm.nestmlir::FuncOp();

	// Partially lower the toy dialect with a few cleanups afterwards.
	optPM.addPass(mlir::toy::createLowerToAffinePass());
	optPM.addPass(mlir::createCanonicalizerPass());
	optPM.addPass(mlir::createCSEPass());
	...
}
```

这段代码实现了将`ToyToAffineLoweringPass`加到优化pipline中，当产生MLIR的命令中使用了`-emit=mlir-affine`选项，则`isLoweringToAffine`为真，将执行这个部分Lowering的过程。

引入了部分Lowering之后我们可以观察一下输出的MLIR表达式长什么样子。先看一下原始的MLIR表达式，即`mlir/test/Examples/Toy/Ch5/affine-lowering.mlir`：

```cpp
func @main() {
  %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
  %2 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
  %3 = toy.mul %2, %2 : tensor<3x2xf64>
  toy.print %3 : tensor<3x2xf64>
  toy.return
}
```

然后执行`./toyc-ch5 ../../mlir/test/Examples/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine`之后就可以查看应用了本节的部分Lowering之后的MLIR表达式了。如下所示：


```cpp
func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<3x2xf64>
  %2 = memref.alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %2[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %2[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %2[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %2[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %2[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %2[1, 2] : memref<2x3xf64>

  // Load the transpose value from the input buffer and store it into the
  // next input buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %2[%arg1, %arg0] : memref<2x3xf64>
      affine.store %3, %1[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Multiply and store into the output buffer.
  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      %3 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %4 = affine.load %1[%arg0, %arg1] : memref<3x2xf64>
      %5 = arith.mulf %3, %4 : f64
      affine.store %5, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %2 : memref<2x3xf64>
  memref.dealloc %1 : memref<3x2xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```


对其简单解读一下，首先这里有六个`f64`数据类型的常量，使用`%cst`开头的变量来表示。然后为输入和输出分配缓冲区，如`%0 = memref.alloc() : memref<3x2xf64>`这里就是申请了一个类型为`memref<3x2xf64>`的一段缓冲区。然后将之前声明的六个数据依次存入上述分配的缓冲区中，`affine.store`指明操作是保存数据操作。之后，第一个循环，将加载的输入数据（数据加载操作`affine.load`），保存到另一个数据容器中，最终实现转置操作。接着，第二个循环，加载之前定义在两个数据容器中的数据，相乘并存放到输出的数据容器中。最终使用`toy.print`打印结果，并释放缓冲区。

## 在Affine Dialect中寻求优化机会
使用了Affine Dialect之后，我们可以将Operation更底层的逻辑展示出来，将代码中的冗余更轻易的暴露出来。这里可以优化的地方为两个循环嵌套的循环边界相同，可以进行循环融合。若在同一个循环中进行处理，减少循环的次数，同时减少多余的数据容器的分配，也减少数据加载的耗时，必然会提高程序的运行效率。

这里可以类比TVM的scheduler中各种循环相关的源语。

这里我们要讲上述提到的循环融合以及减少多余的数据容器的分配和加载的优化加入到pipline中，具体在`mlir/examples/toy/Ch5/toyc.cpp`中的如下代码：

```cpp
if (isLoweringToAffine) {
  mlir::OpPassManager &optPM = pm.nestmlir::FuncOp();

  // Partially lower the toy dialect with a few cleanups afterwards.
  optPM.addPass(mlir::toy::createLowerToAffinePass());
  optPM.addPass(mlir::createCanonicalizerPass());
  optPM.addPass(mlir::createCSEPass());
	
  // Add optimizations if enabled.
  if (enableOpt) {
    optPM.addPass(mlir::createLoopFusionPass());
    optPM.addPass(mlir::createMemRefDataFlowOptPass());
  }
}
```


注意到最后一个`if`条件，添加了`createLoopFusionPass`和`createMemRefDataFlowOptPass`，**这两种MLIR自带的Pass分别完成了相同循环边界融合优化以及对于MemRef的数据流优化功能**。我们比较关心多了这两个优化之后上面的MLIR表达式会变成什么样子，具体来说我们只需要在上面那个生成部分Lowering的MLIR表达式的命令中额外加一个`-opt`选项就可以生成加入了这两个优化Pass的新的MLIR表达式了。命令如下：`./toyc-ch5 ../../mlir/test/Examples/Toy/Ch5/affine-lowering.mlir -emit=mlir-affine -opt` 。生成的MLIR表达式为：


```cpp
func @main() {
  %cst = arith.constant 1.000000e+00 : f64
  %cst_0 = arith.constant 2.000000e+00 : f64
  %cst_1 = arith.constant 3.000000e+00 : f64
  %cst_2 = arith.constant 4.000000e+00 : f64
  %cst_3 = arith.constant 5.000000e+00 : f64
  %cst_4 = arith.constant 6.000000e+00 : f64

  // Allocating buffers for the inputs and outputs.
  %0 = memref.alloc() : memref<3x2xf64>
  %1 = memref.alloc() : memref<2x3xf64>

  // Initialize the input buffer with the constant values.
  affine.store %cst, %1[0, 0] : memref<2x3xf64>
  affine.store %cst_0, %1[0, 1] : memref<2x3xf64>
  affine.store %cst_1, %1[0, 2] : memref<2x3xf64>
  affine.store %cst_2, %1[1, 0] : memref<2x3xf64>
  affine.store %cst_3, %1[1, 1] : memref<2x3xf64>
  affine.store %cst_4, %1[1, 2] : memref<2x3xf64>

  affine.for %arg0 = 0 to 3 {
    affine.for %arg1 = 0 to 2 {
      // Load the transpose value from the input buffer.
      %2 = affine.load %1[%arg1, %arg0] : memref<2x3xf64>

      // Multiply and store into the output buffer.
      %3 = arith.mulf %2, %2 : f64
      affine.store %3, %0[%arg0, %arg1] : memref<3x2xf64>
    }
  }

  // Print the value held by the buffer.
  toy.print %0 : memref<3x2xf64>
  memref.dealloc %1 : memref<2x3xf64>
  memref.dealloc %0 : memref<3x2xf64>
  return
}
```

我们可以看到，这里删除了多余的数据缓冲区的分配，两个循环嵌套被融合在一起，另外去除了一些不必要的数据加载操作。

所以我们通过部分Lowering确实寻求到了更多的优化机会，使得MLIR表达式的运行效率更高。


# 参考
- https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/
- https://zhuanlan.zhihu.com/p/362749628