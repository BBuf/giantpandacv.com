# 0x0. 前言
在上一节中，我们将Toy Dialect的部分Operation Lowering到Affine Dialect，MemRef Dialect和Standard Dialect，而`toy.print`操作保持不变，所以又被叫作部分Lowering。通过这个Lowering可以将Toy Dialect的Operation更底层的实现逻辑表达出来，以寻求更多的优化机会，得到更好的MLIR表达式。这一节，我们将在上一节得到的混合型MLIR表达式完全Lowering到LLVM Dialect上，然后生成LLVM IR，并且我们可以使用MLIR的JIT编译引擎来运行最终的MLIR表达式并输出计算结果。


# 0x1. IR下降到LLVM Dialect
这一小节我们将来介绍如何将上一节结束的MLIR表达式完全Lowering为LLVM Dialect，我们还是回顾一下上一节最终的MLIR表达式：

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

我们要将这个三种Dialect混合的MLIR表达式完全Lowering为LLVM Dialect，注意LLVM Dialect是MLIR的一种特殊的Dialect层次的中间表示，**它并不是LLVM IR**。Lowering为LLVM Dialect的整体过程可以分为如下几步：

## 1. Lowering toy.print Operation
之前部分Lowering的时候并没有对`toy.print`操作进行Lowering，所以这里优先将`toy.print`进行Lowering。我们把`toy.print` Lowering到一个非仿射循环嵌套，它为每个元素调用`printf`。Dialect转换框架支持传递Lowering，不需要直接Lowering为LLVM Dialect。通过应用传递Lowering可以应用多种模式来使得操作合法化（合法化的意思在这里指的就是完全Lowering到LLVM Dialect）。 传递Lowering在这里体现为将`toy.print`先Lowering到循环嵌套Dialect里面，而不是直接Lowering为LLVM Dialect。

在Lowering过程中，`printf`的声明在`mlir/examples/toy/Ch6/mlir/LowerToLLVM.cpp`中，代码如下：

```cpp
	/// Return a symbol reference to the printf function, inserting it into the
  /// module if necessary.
  static FlatSymbolRefAttr getOrInsertPrintf(PatternRewriter &rewriter,
                                             ModuleOp module) {
    auto *context = module.getContext();
    if (module.lookupSymbol<LLVM::LLVMFuncOp>("printf"))
      return SymbolRefAttr::get(context, "printf");

    // Create a function declaration for printf, the signature is:
    //   * `i32 (i8*, ...)`
    auto llvmI32Ty = IntegerType::get(context, 32);
    auto llvmI8PtrTy = LLVM::LLVMPointerType::get(IntegerType::get(context, 8));
    auto llvmFnType = LLVM::LLVMFunctionType::get(llvmI32Ty, llvmI8PtrTy,
                                                  /*isVarArg=*/true);

    // Insert the printf function into the body of the parent module.
    PatternRewriter::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    rewriter.create<LLVM::LLVMFuncOp>(module.getLoc(), "printf", llvmFnType);
    return SymbolRefAttr::get(context, "printf");
  }
```

这部分代码返回了`printf`函数的符号引用，必要时将其插入Module。在函数中，为`printf`创建了函数声明，然后将`printf`函数插入到父Module的主体中。

## 2. 确定Lowering过程需要的所有组件

第一个需要确定的是转换目标(ConversionTarget)，对于这个Lowering我们除了顶层的Module将所有的内容都Lowering为LLVM Dialect。这里代码表达的信息和官方文档有一些出入，以最新的代码为准。

```cpp
// The first thing to define is the conversion target. This will define the
// final target for this lowering. For this lowering, we are only targeting
// the LLVM dialect.
LLVMConversionTarget target(getContext());
target.addLegalOp<ModuleOp>();
```

然后需要确定类型转换器(Type Converter)，我们现存的MLIR表达式还有`MemRef`类型，我们需要将其转换为LLVM的类型。为了执行这个转化，我们使用`TypeConverter`作为Lowering的一部分。这个转换器指定一种类型如何映射到另外一种类型。由于现存的操作中已经不存在任何Toy Dialect操作，因此使用MLIR默认的转换器就可以满足需求。定义如下：

```cpp
// During this lowering, we will also be lowering the MemRef types, that are
  // currently being operated on, to a representation in LLVM. To perform this
  // conversion we use a TypeConverter as part of the lowering. This converter
  // details how one type maps to another. This is necessary now that we will be
  // doing more complicated lowerings, involving loop region arguments.
  LLVMTypeConverter typeConverter(&getContext());
```

再然后还需要确定转换模式(Conversion Patterns)。这部分代码为：

```cpp
// Now that the conversion target has been defined, we need to provide the
  // patterns used for lowering. At this point of the compilation process, we
  // have a combination of `toy`, `affine`, and `std` operations. Luckily, there
  // are already exists a set of patterns to transform `affine` and `std`
  // dialects. These patterns lowering in multiple stages, relying on transitive
  // lowerings. Transitive lowering, or A->B->C lowering, is when multiple
  // patterns must be applied to fully transform an illegal operation into a
  // set of legal ones.
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateLoopToStdConversionPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateStdToLLVMConversionPatterns(typeConverter, patterns);

  // The only remaining operation to lower from the `toy` dialect, is the
  // PrintOp.
  patterns.add<PrintOpLowering>(&getContext());
```

上面这段代码展示了为Affine Dialect，Standard Dialect以及遗留的`toy.print`定义匹配重写规则。首先将Affine Dialect下降到Standard Dialect，即`populateAffineToStdConversionPatterns`。然后将Loop(针对的是`toy.print`操作，它已经Lowering到了循环嵌套Dialect)下降到Standard Dialect，即`populateLoopToStdConversionPatterns`。最后，将Standard Dialect转换到LLVM Dialect，即`populateMemRefToLLVMConversionPatterns`。以及不要忘了把`toy.print`的Lowering模式`PrintOpLowering`加到`patterns`里面。

## 3. 完全Lowering
定义了Lowering过程需要的所有组件之后，就可以执行完全Lowering了。使用`applyFullConversion(module, target, std::move(patterns)))` 函数可以保证转换的结果只存在合法的操作，上一篇部分Lowering的笔记调用的是`mlir::applyPartialConversion(function, target, patterns)`可以对比着看一下。

```cpp
// We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
```

## 4. 将上面定义好的完全Lowering的Pass加到Pipline中
这段代码在`mlir/examples/toy/Ch6/toyc.cpp`中：

```cpp
if (isLoweringToLLVM) {
  // Finish lowering the toy IR to the LLVM dialect.
  pm.addPass(mlir::toy::createLowerToLLVMPass());
 }
```

这段代码在优化Pipline中添加了`mlir::toy::createLowerToLLVMPass()`这个完全Lowering的Pass，可以把MLIR 表达式下降为LLVM Dialect表达式。我们运行一下示例程序看下结果：

执行下面的命令：

```bash
cd llvm-project/build/bin
./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/llvm-lowering.mlir -emit=mlir-llvm
```

即获得了完全Lowering之后的MLIR表达式，结果比较长，这里只展示一部分。可以看到目前MLIR表达式已经完全在LLVM Dialect空间下了。


```cpp
llvm.func @free(!llvm<"i8*">)
llvm.func @printf(!llvm<"i8*">, ...) -> i32
llvm.func @malloc(i64) -> !llvm<"i8*">
llvm.func @main() {
  %0 = llvm.mlir.constant(1.000000e+00 : f64) : f64
  %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64

  ...

^bb16:
  %221 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %222 = llvm.mlir.constant(0 : index) : i64
  %223 = llvm.mlir.constant(2 : index) : i64
  %224 = llvm.mul %214, %223 : i64
  %225 = llvm.add %222, %224 : i64
  %226 = llvm.mlir.constant(1 : index) : i64
  %227 = llvm.mul %219, %226 : i64
  %228 = llvm.add %225, %227 : i64
  %229 = llvm.getelementptr %221[%228] : (!llvm."double*">, i64) -> !llvm<"f64*">
  %230 = llvm.load %229 : !llvm<"double*">
  %231 = llvm.call @printf(%207, %230) : (!llvm<"i8*">, f64) -> i32
  %232 = llvm.add %219, %218 : i64
  llvm.br ^bb15(%232 : i64)

  ...

^bb18:
  %235 = llvm.extractvalue %65[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %236 = llvm.bitcast %235 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%236) : (!llvm<"i8*">) -> ()
  %237 = llvm.extractvalue %45[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %238 = llvm.bitcast %237 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%238) : (!llvm<"i8*">) -> ()
  %239 = llvm.extractvalue %25[0 : index] : !llvm<"{ double*, i64, [2 x i64], [2 x i64] }">
  %240 = llvm.bitcast %239 : !llvm<"double*"> to !llvm<"i8*">
  llvm.call @free(%240) : (!llvm<"i8*">) -> ()
  llvm.return
}
```

# 0x2. 代码生成以及Jit执行

我们可以使用JIT编译引擎来运行上面得到的LLVM Dialect IR，获得推理结果。这里我们使用了`mlir::ExecutionEngine`基础架构来运行LLVM Dialect IR。程序位于：`mlir/examples/toy/Ch6/toyc.cpp`。

```cpp
int runJit(mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  auto maybeEngine = mlir::ExecutionEngine::create(
      module, /*llvmModuleBuilder=*/nullptr, optPipeline);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```

> 这里尤其需要注意这行：mlir::registerLLVMDialectTranslation(*module->getContext());。从代码的注释来看这个是将LLVM Dialect表达式翻译成LLVM IR，在JIT编译的时候起到缓存作用，也就是说下次执行的时候不会重复执行上面的各种MLIR表达式变换。

这里创建一个MLIR执行引擎`mlir::ExecutionEngine`来运行表达式中的`main`函数。可以使用下面的命令来输出最终的计算结果：

```cpp
cd llvm-project/build/bin
./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/codegen.toy -emit=jit -opt
```

结果为：

```cpp
1.000000 16.000000 
4.000000 25.000000 
9.000000 36.000000
```

到这里，我们就将原始的MLIR表达式经过一系列Pass进行优化，以及部分Lowering到三种Dialect混合的表达式，和完全Lowering为LLVM Dialect表达式，最后翻译到LLVM IR使用MLIR的Jit执行引擎进行执行，获得了最终结果。

另外，`mlir/examples/toy/Ch6/toyc.cpp`中还提供了一个`dumpLLVMIR`函数，可以将MLIR表达式翻译成LLVM IR表达式。然后再经过LLVM IR的优化处理。使用如下命令可以打印出生成的LLVM IR：

```cpp
$ cd llvm-project/build/bin
$ ./toyc-ch6 ../../mlir/test/Examples/Toy/Ch6/codegen.toy -emit=llvm -opt
```

# 0x3. 总结
这篇文章介绍了如何将部分Lowering之后的MLIR表达式进一步完全Lowering到LLVM Dialect上，然后通过JIT编译引擎来执行代码并获得推理结果，另外还可以输出LLVM Dialect生成的LLVM IR。