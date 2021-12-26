### 前言

最近在同事shenghang的帮助下做了一点OneFlow IR相关的开发，对MLIR执行部分有一些新的感受，所以尝试分享一下。我之前花了不少时间去理解OneFlow IR的整个架构（可以看我的Toy Tutorials系列），但对OneFloiw IR的JIT的执行这部分一直存疑。最近将OneFlow基于Job（OneFlow的作业函数，不考虑设备的话可以理解为一个计算图）接入MLIR工程实现部分重新进行了梳理，并在shenghang的指导下理解了整个流程。所以这篇文档我将介绍一下OneFlow和MLIR是如何结合的，如何在OneFlow IR中新增一个图级别的Pass，OneFlow的Operation是如何自动变成MLIR 的Operation的以及为什么OneFlow IR能利用MLIR为计算带来加速等。我对MLIR的了解不算多，2个月前开始接触，有任何错误请大家批评斧正。本文和 https://github.com/Oneflow-Inc/oneflow & https://github.com/BBuf/tvm_mlir_learn 有关，感兴趣可以star关注一下。

> 本文提到的Op和Operation是一回事，没有严格区分。

### OneFlow是如何和MLIR结合的？

在OneFlow中引入MLIR作为OneFlow的IR有诸多优点，不仅可以取代OneFlow中需要通过C++手写的Operation定义减小开发难度，还可以降低Operation定义中一些容器相关的开销。另外我们还可以通过MLIR维护的基础设施（即多重Dialect）来完成对计算图计算的加速。这里的计算图既可以是Eager的计算图，也可以是Lazy的计算图。由于基于Eager计算图使用MLIR进行加速的工作（即`oneflow.jit.xxx`）还没有正式开放，我这里仍然以Lazy计算图（Job）为例来讲解OneFlow和MLIR的结合过程。

首先我们需要编译好开启MLIR的OneFlow，编译命令如下：

```shell
git clone git@github.com:Oneflow-Inc/oneflow.git
cd oneflow && mkdir build && cd build
cmake-C ../cmake/caches/cn/fast/mlir-cuda-75.cmake -DBUILD_TESTING=ON .. && ninja 
```

然后可以写一个例子进行测试：

```python
os.environ["ONEFLOW_MLIR_ENABLE_ROUND_TRIP"] = '1'
os.environ["ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS"] = '1'

@flow.unittest.skip_unless_1n1d()
class TestFuseBiasAddGeLUCPUMLIR(oneflow.unittest.TestCase):
    def test_fused_bias_add_gelu_graph(test_case):
        data = np.random.randn(1, 2, 3)
        bias_data = np.random.randn(2)
        x = flow.tensor(data, dtype=flow.float32)
        bias = flow.tensor(bias_data, dtype=flow.float32)
        y_eager = flow.gelu(flow._C.bias_add(x, bias, axis=1))

        class FuseBiasAddGeLUGraph(flow.nn.Graph):
            def __init__(self):
                super().__init__()

            def build(self, x):
                return flow.gelu(flow._C.bias_add(x, bias, axis=1))

        bias_add_gelu = FuseBiasAddGeLUGraph()
        y_lazy = bias_add_gelu(x)
        test_case.assertTrue(np.array_equal(y_eager.numpy(), y_lazy.numpy()))
```

运行这个例子之后会在当前运行目录下生成一个log文件，里面有一个`ir_pass` 文件夹记录了经过OneFlow MLIR优化前后的计算图(`.prototxt`) 以及 MLIR的表达式(`*.mlir`)，还有一个`*.mlir.dot`文件可以用`graphviz`打开来可视化MLIR表达式的计算图。需要注意的是如果OneFlow正在执行训练任务，这个log文件夹里不仅包含前向的计算图和MLIR表达式，也会生成后向的计算图和MLIR表达式。所以MLIR在整个神经网络的运行流程中均可以作用，这是区别于前向推理框架的重要一点，即训练也可以加速。

在`oneflow/api/python/ir.cpp` 中有下面两行代码：

```c++
REGISTER_JOB_PASS("IRRoundTripBeforeAD", IRRoundTrip<kBeforeAD>);
REGISTER_JOB_PASS("IRRoundTrip", IRRoundTrip<kAfterAD>);
```

`RoundTrip`即往返的意思，`BeforeAD`可以理解为反向之前，`kAfterAD` 可以理解为反向之后，这里通过将OneFlow Job和MLIR的互转过程注册为OneFlow Job的一个Pass来建立OneFlow计算图和MLIR的联系。在执行OneFlow脚本时，如果想使能MLIR作用于OneFlow计算图，开启`ONEFLOW_MLIR_ENABLE_ROUND_TRIP=1`环境变量即可。

接下来，要将OneFlow的计算图和MLIR建立联系等价于将OneFlow计算图中的Operation和MLIR中的Operation进行一对一的转换。而MLIR的Operation定义在各级Dialect下，按照MLIR的通用接入原则，我们实现了一个OneFlow Dialect并在OneFlow Dialect上实现了OneFlow Operation到OneFlow Dialect下的Operation的一一映射。如何定义OneFlow Dialect和Operation这里就不讲了，可以参考MLIR官方文档的Dialects和ODS一节（https://mlir.llvm.org/docs/OpDefinitions/）或者我之前的文章，它们都是基于TableGen规则来完成的。关于MLIR Operation的定义我之前结合OneFlow Dialect的Op定义总结了一个文档（`https://github.com/BBuf/tvm_mlir_learn` 中） 。除了Dialect和Operation的定义还有一些其它需要定义的东西，比如OneFlow数据类型到MLIR数据类型映射的定义在`oneflow/ir/include/OneFlow/OneFlowEnums.td` ，OneFlow Dialect Operation的一些通用前端接口定义在`oneflow/ir/include/OneFlow/OneFlowEnums.td`。这里我们以Reshape Operation为例子来简单说明一下这个Operation有哪些组成部分：

```c++
def OneFlow_ReshapeOp : OneFlow_BaseOp<"reshape", [NoSideEffect, DeclareOpInterfaceMethods<UserOpCompatibleInterface>]> {
  let input = (ins
    AnyType:$in
  );
  let output = (outs
    AnyType:$out
  );
  let attrs = (ins
    AnyI64ElementsAttr:$shape
  );
}
```

`OneFlow_ReshapeOp` 这个名字下划线之前的是Dialect的名字，后面是这个Dialect下的Operation的名字。然后这个Operation继承了`OneFlow_BaseOp`基类，并声明了约束和前端接口，接下来定义了Operation的输入，输出和属性就结束了。可以发现OneFlow Dialect Operation的定义和OneFlow User Op是完全一致的，这保证了OneFlow和MLIR互转的合法性。OneFlow Reshape Operation的定义如下：

```c++
REGISTER_USER_OP("reshape")
    .Input("in")
    .Output("out")
    .Attr<Shape>("shape")
    ...

```

OneFlow Job和MLIR的互转实现在`oneflow/ir/oneflow-translate`，主要做的事情就是遍历Job的OpGraph，对节点和边分别进行处理最后转换成一个MLIR表达式，同时在计算完成后可以基于MLIR表达式重写Job。这里的整体逻辑偏复杂，因为要处理OneFlow Job OpGraph里面各种类型Operation和边的转化，这里不继续深入讲解，因为它也不是我这篇文章要讨论的点，感兴趣的可以直接阅读代码。

### OneFlow IR如何执行？

在上面Operation定义时是举了一个Reshape的例子，浏览`oneflow/ir/include/OneFlow/OneFlowOps.td`容易发现这里还定义了一个`OneFlow_MlirJitOp`，这个自定义的Op就是用来执行MLIR表达式的，它里面实现了CPU和GPU的Kernel（源码在`oneflow/ir/oneflow-extension/extension.cpp`）用来加载MLIR提供的JIT执行引擎运行最终得到的LLVM IR。那么LLVM IR又是怎么来的呢？这是通过OneFlow MLIR表达式逐级下降之后得来的，具体下降过程如下：

```c++
void AddLowerToLinalgMemRefPasses(PassManager& pm) {
  pm.addPass(createLowerOneFlowToTosaPass());            // lower-oneflow-to-tosa
  pm.addPass(createCSEPass());                           // cse
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalg());  // tosa-to-linalg-on-tensors
  auto p = createLinalgElementwiseOpFusionPass();
  assert(p->initializeOptions("allow-folding-unit-dim-reshapes=true").succeeded());
  pm.addNestedPass<FuncOp>(std::move(p));                     // linalg-fuse-elementwise-ops
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());      // linalg-bufferize
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());      // tensor-bufferize
  pm.addPass(createTensorConstantBufferizePass());            // tensor-constant-bufferize
  pm.addPass(createFuncBufferizePass());                      // func-bufferize
  pm.addPass(createBufferResultsToOutParamsPass());           // buffer-results-to-out-params
  pm.addPass(createCanonicalizerPass());                      // canonicalize
  pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());  // finalizing-bufferize
}

LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module) {
  mlir::PassManager pm(context);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());  // convert-linalg-to-loops
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());            // convert-scf-to-std
  pm.addPass(createConvertLinalgToLLVMPass());                 // convert-linalg-to-llvm
  pm.addPass(createMemRefToLLVMPass());                        // convert-memref-to-llvm
  pm.addPass(createLowerToLLVMPass());                         // convert-std-to-llvm
  pm.addPass(createReconcileUnrealizedCastsPass());
  return pm.run(module);
}
```

可以看到OneFlow Dialect首先下降到Tosa Dialect，然后下降到Linalg Dialect，再然后是Loop Dialect，一直到最后的LLVM IR。在逐级下降的过程中，我们可以享受如Linalg Dialect带来的嵌套循环变换带来的优化机会以提升最终IR的性能。这里的Lowering过程是在OneFlow调用`MlirJitOp` 的Kernel时触发的（`oneflow/ir/oneflow-extension/extension.cpp` ），调用也是作为一个MLIR的Pass被加入到了优化流程中。JIT调用流程Pass的实现可以精简为：

```c++
class OutlineJitFunctionPass : public OutlineJitFunctionPassBase<OutlineJitFunctionPass> {
  void runOnOperation() override {
    Operation* op = getOperation();
    RewritePatternSet patterns(op->getContext());
    oneflow::populateFuserPasses(patterns);
    (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

std::unique_ptr<Pass> createOutlineJitFunctionPass() {
  return std::make_unique<OutlineJitFunctionPass>();
}

LogicalResult ApplyRoundTripPatterns(RoundTripOneFlowJobWrapperInterface& job_wrapper,
                                     MLIRContext* context, OwningModuleRef& module) {
  mlir::PassManager pm(context);
  pm.addNestedPass<mlir::FuncOp>(::mlir::createCanonicalizerPass());
  if (job_wrapper.IsLastIRPass() && std::getenv("ONEFLOW_MLIR_ENABLE_CODEGEN_FUSERS") != nullptr) {
    pm.addPass(oneflow::createOutlineJitFunctionPass());
  }
  ...
}
```

但这套流程还存在两个问题需要解决：

- 第一个问题是如何做Op融合。上面的JIT执行流程只考虑了不断Lowering，那么假如在OneFlow Dialect中有一些Operation是可以融合的，这个时候应该怎么做呢？很简单，我们沿用一下MLIR的DRR规则，还是用TableGen语法在`oneflow/ir/include/OneFlow/OneFlowPatterns.td` 中写一系列的Fuse Pattern即可，比如`bias_add`+`gelu` 这两个Op可以融合成OneFlow中的`fused_bias_add_gelu` Op，那么就可以写如下的规则。

```c++
def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;
def FusedBiasAddGeluPattern : Pat<
  (
    OneFlow_GeluOp : $gelu_op
    (
      OneFlow_BiasAddOp
        $a,
        $b,
        $bias_add_op_name,
        $bias_add_device_tag,
        $bias_add_device_name,
        $bias_add_scope_symbol_id,
        $bias_add_hierarchy,
        $axis
    ),
    $gelu_op_name,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy
  ),
  (OneFlow_FusedBiasAddGeluOp $a, $b,
    $gelu_op_name,
    $gelu_device_tag,
    $gelu_device_name,
    $gelu_scope_symbol_id,
    $gelu_hierarchy,
    $axis
  ),
  [
    (IsGPU $bias_add_device_tag),
    (IsGPU $gelu_device_tag)
  ]
>;
```

这里基于MLIR的DRR规则来做表达式匹配和重写，可以看到假如当前运行设备是GPU并且前后两个Op分别是`gelu`和`bias_add` 就将其进行融合为一个`fused_bias_add_gelu_op`，在CUDA上可以减少读写来提升执行效率。

- 第二个问题是如何让OneFlow的一些Operation享受MLIR基础设施中的更多优化？在多级Dialect 逐层下降时可以看到OneFlow的MLIR表达式的每个子函数都会被Lower。第一次会将其Lower到Tosa Dialect，这个时候如果这个子函数中的某个Operation没有定义转换到Tosa Dialect的方法，那么就不能Lower到Tosa Dialect。自然也就不能进一步下降为Linalg Dialect，享受不到一些循环变化带来的优化（我感觉可以类比TVM的scheduler优化）。为了解决这种情况我们需要额外再定义一个Pass来将当前需要转换为Tosa的Op或者模式提取成一个函数，里面的oneflow op都能够lower到tosa，然后生成一个 oneflow mlir jit op 来 call 这个函数：

```c++
def IsNotNestedInJit: Constraint<CPred<"(!$0.getDefiningOp()->getParentOfType<::mlir::FuncOp>()->hasAttr(\"llvm.emit_c_interface\"))">, "">;
def OutlineMulCast : NativeCodeCall<"::mlir::oneflow::OutlineMulCast($_builder, $0, $1)">;
// TODO: remove attr binding if possible
def MulCastPattern : Pat<
  (
    OneFlow_ScalarMulByTensorOp : $mul_op
    (
      OneFlow_CastOp : $cast_op
        $cast_x,
        $cast_op_name,
        $cast_device_tag,
        $cast_device_name,
        $cast_scope_symbol_id,
        $cast_hierarchy,
        $cast_dtype
    ),
    $scalar,
    $mul_op_name,
    $mul_device_tag,
    $mul_device_name,
    $mul_scope_symbol_id,
    $mul_hierarchy
  ),
  (OutlineMulCast $mul_op, $cast_op),
  [
    (IsNotNestedInJit $mul_op)
  ]
>;

::llvm::SmallVector<::mlir::Value, 4> OutlineMulCast(::mlir::PatternRewriter& rewriter,
                                                     mlir::OpResult mul_res,
                                                     mlir::OpResult cast_res) {
  if (auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_res.getDefiningOp())) {
    if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {
      // TODO: extract a function to generate op name for jit op from ops being fused
      SmallString<64> op_name_storage;
      auto op_name =
          (cast_op.op_name() + "__FUSE__" + mul_op.op_name()).toStringRef(op_name_storage);
      SmallVector<::mlir::Value, 2> operands;
      operands.push_back(cast_op.in());
      operands.push_back(mul_op.scalar());
      SmallVector<::mlir::Value, 1> results;
      results.push_back(mul_op.y());
      NamedAttrList attributes =
          GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), mul_op);
      SmallVector<Operation*, 4> ops = {cast_op, mul_op};
      auto function =
          GetOrInsertFuncOp(rewriter, mul_op->getLoc(), op_name, operands, results, ops);
      auto created = rewriter.create<MlirJitOp>(mul_op.getLoc(), function, attributes, operands);
      assert(DumpAssembly(rewriter, created).succeeded());
      cast_op->dropAllUses();
      cast_op.erase();
      return created->getResults();
    }
  }
  return {};
}

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<MulCastPattern>(patterns.getContext());
}
```

这里就是将MulCast这个Pattern手动实现了从OneFlow Dialect到Tosa Dialect的转换，最后将这个Pass加到优化流程中即可完成MLIR表达式中的这个Pattern会经过Tosa和Linalg这两个层次的Dialect，获得一些优化机会。



### 总结

这里以OneFlow为例讲解了一些MLIR的真实运行流程，即是如何通过MLIR来执行深度学习框架的计算图并且为其加速的，目前理解难免有不到位的地方，欢迎大家批评指正。