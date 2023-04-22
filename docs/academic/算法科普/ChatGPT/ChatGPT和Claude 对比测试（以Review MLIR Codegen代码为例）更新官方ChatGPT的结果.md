
> Claude在MLIR代码分析上完全超越了ChatGPT并表现十分惊艳，请阅读全文或者自己注册感受它的强大。结论：在本文的任务中，Claude > ChatGPT >> NewBing 
# 0x0. 前言

这里将以oneflow IR部分中的一个Codegen任务（目标是在mlir codegen中支持oneflow stream，用oneflow stream替换pass中自己生成的stream，PR链接为：https://github.com/Oneflow-Inc/oneflow/pull/10149）为例，来对比一下newibing(chatgpt)和claude对mlir的理解能力。claude是Anthropic公司推出的类似于chatgpt的聊天机器人，这家公司是OpenAI的最大竞争对手之一，因为创办这家公司的人也是OpenAI的前员工。然后Claude是参考这个issue: https://www.zhihu.com/question/594115372/answer/2988759047 将其直接添加到slack里进行对话。

# 0x1. PR简介

PR链接为：**https://github.com/Oneflow-Inc/oneflow/pull/10149**

这个PR实现了3个Pass (定义在 `OneFlowPasses.td`)，也就是：

```cpp
def EliminateAllocOpsPass : Pass<"eliminate-alloc-ops", "ModuleOp"> {
  let summary = "";
  let constructor = "mlir::oneflow::createEliminateAllocOpsPass()";
  let dependentDialects = ["pdl_interp::PDLInterpDialect", "pdl::PDLDialect"];
}

def AppendOneFlowStreamPass : Pass<"append-ofstream", "ModuleOp"> {
  let summary = "append oneflow stream to gpu function arguments";
  let constructor = "mlir::oneflow::createAppendOneFlowStreamPass()";
}

def MgpuToOneFlowStreamPass : Pass<"mgpu-to-ofstream", "ModuleOp"> {
  let summary = "convert mlir abi about mgpu to oneflow stream, this pass should be invoked after append-ofstream pass";
  let constructor = "mlir::oneflow::createMgpuToOneFlowStreamPass()";
}
```

EliminateAllocOpsPass用来消除IR中的无效memref.alloc指令，AppendOneFlowStreamPass给GPU相关的函数添加GPU启动kernel需要的stream参数，MgpuToOneFlowStreamPass发生在AppendOneFlowStreamPass执行之后(它生成了stream参数)并把mgpu相关的stream abi替换为oneflow stream abi。

我们分别使用newbing和claude来让它们分析一下这几行`OneFlowPasses.td`中定义的Pass意图：

newbing：

![在这里插入图片描述](https://img-blog.csdnimg.cn/c11564d2942c45968369dbe9eceef276.png)

newbing直接看不懂，其实我感觉claude也应该看不懂吧，抱着怀疑的态度问一下。

![在这里插入图片描述](https://img-blog.csdnimg.cn/7eb026cea1314bb3938dc20ae345f57c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/dc17ec634ca04a38a377bc24ab81f9f0.png)

太疯狂了，claude不仅读懂了td文件的代码，甚至为我们列出了这个代码涉及到的MLIR概念。感觉是训练数据考虑了MLIR相关的预料？接下来我们再对比下C++实现的Pass代码。

# 0x2. 对比具体实现

PR链接为：**https://github.com/Oneflow-Inc/oneflow/pull/10149**

## 0x2.1 EliminateAllocOpsPass

EliminateAllocOpsPass使用MLIR提供的PDL语言来完成Pattern的匹配和重写，具体实现在 `oneflow/ir/lib/OneFlow/PDLL/AllocEliminationPatterns.pdll` ：

```cpp
#include "OneFlow/OneFlowOps.td"

Constraint IsFuncArguments(value: Value) [{
  return success(llvm::dyn_cast<mlir::BlockArgument>(value));
}];

Pattern {
  let alloc = op<memref.alloc>();
  let copy = op<memref.copy>(alloc.0, arg: IsFuncArguments);

  rewrite alloc with {
    erase copy;
    replace alloc with arg;
  };
}
```

接下来，我们分别对比一下newbing和chatgpt对它的分析结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0cb5a211c3da4cbe8f452eaeb17797ce.png)

newbing并不能解析出这段代码是MLIR的PDL语言，当然也无法理解代码内容。我们可以再使用Claude试试。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b9748b417d354099a531f8efa4f0909e.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/4dcdf06a6bae4fb9acf2f40c0b63b21a.png)

个人感觉这个解释是非常强大且精准的，Claude的答案非常惊艳。
## 0x2.2 AppendOneFlowStreamPass

接下来我们看一下AppendOneFlowStreamPass的实现，这个实现是在`oneflow/ir/lib/OneFlow/Transform/OneFlowStream.cpp`这个文件，具体代码如下：

```cpp
struct AppendOneFlowStreamPattern final : public OpRewritePattern<func::FuncOp> {
 public:
  explicit AppendOneFlowStreamPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto ptr_type = LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
    if (llvm::dyn_cast<LLVM::LLVMPointerType>(op.getFunctionType().getInputs().back()))
      return success();

    llvm::SmallVector<Type> new_operand_type;
    for (auto type : op.getFunctionType().getInputs()) { new_operand_type.push_back(type); }
    new_operand_type.push_back(ptr_type);
    auto function_type =
        rewriter.getFunctionType(new_operand_type, op.getFunctionType().getResults());

    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), function_type);
    for (auto pair : op->getDialectAttrs()) { func->setAttr(pair.getName(), pair.getValue()); }
    op.getBody().addArgument(ptr_type, func->getLoc());
    IRMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);
    rewriter.eraseOp(op);
    return success();
  }
};
```

c++代码newbing（chatgpt）按道理可以看懂了，我们让它分析一下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/03231556bf3940b5b39e54d61dd924dc.png)

直接问chatgpt，它还是不懂这段代码。我手动提示了下它说，这段代码定义了一个mlir pattern，然后它先是重复我的话给出了一段回答。然后接下来就是胡说八道了，在这个例子中表现很差。接下来我们拷问一下Claude：


![在这里插入图片描述](https://img-blog.csdnimg.cn/58b0c17eec5a4760ab3ad0ed1c8c568b.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0d48cb6b887d482e9c7c7b7cc3d2a716.png)

我们继续问一下c++代码中的一些细节：

![在这里插入图片描述](https://img-blog.csdnimg.cn/08b6327eb2024a7587d61cffbba9423a.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1a9660ce29154367833792f9865081ea.png)

非常强大，给出的解释大多比较精准，并且似乎Claude真的完全理解了这段代码的逻辑。我们需要注意的是，这段代码是我同事今天才写的，模型的泛化性真的很好。



## MgpuToOneFlowStreamPass

我们最后再分析下MgpuToOneFlowStreamPass的实现。

```cpp
struct MgpuToOneFlowStreamPattern final : public OpRewritePattern<LLVM::CallOp> {
 public:
  explicit MgpuToOneFlowStreamPattern(mlir::MLIRContext* context)
      : OpRewritePattern<LLVM::CallOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(LLVM::CallOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto ptr_type = LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    auto callee = op.getCallee();
    if (!func || !callee) return failure();
    Value stream = func.getArguments().back();
    if (stream.getType() != ptr_type) {
      LOG(ERROR) << "failed to find stream in llvm.func block arguments";
      return failure();
    }

    DenseMap<StringRef,
             std::pair<std::function<bool(LLVM::CallOp&, Value&)>,
                       std::function<void(mlir::PatternRewriter&, LLVM::CallOp&, Value&)>>>
        oneflow_abi = {
            {"mgpuStreamCreate",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.replaceOp(op, {stream});
              }}},
            {"mgpuLaunchKernel",
             {[](LLVM::CallOp& op, Value& stream) {
                unsigned idx = op->getNumOperands();
                return op.getOperand(idx - 3) != stream;
              },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                unsigned idx = op->getNumOperands();
                auto target = op.getOperand(idx - 3).getDefiningOp();
                rewriter.replaceOp(target, {stream});
              }}},
            {"mgpuStreamSynchronize",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.eraseOp(op);
              }}},
            {"mgpuStreamDestroy",
             {[](LLVM::CallOp& op, Value& stream) { return true; },
              [](mlir::PatternRewriter& rewriter, LLVM::CallOp& op, Value& stream) {
                rewriter.eraseOp(op);
              }}},
        };
    auto out = oneflow_abi.find(callee.value().str());
    if (out != oneflow_abi.end() && out->getSecond().first(op, stream)) {
      out->getSecond().second(rewriter, op, stream);
    }
    return success();
  }
};
```

还是先让chatgpt分析下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e863973c2c2e4394b5cc870b2bbe09d1.png)

回答还是比较模棱两可，并且可以确定的事情是chatgpt完全没有理解这段代码。

接下来还是使用Claude来测试下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/12af066270d441339ff9b844697fbc96.png)

这个地方让我震惊的点是，它不仅理解了这段代码，而且知道在MLIR里面这段代码只是一个Pattern规则，如果要应用这个规则需要在MLIR里面再构建一个Pass。最后我们再让Claude给我们一些Review意见：

![在这里插入图片描述](https://img-blog.csdnimg.cn/9c935c86cb354e9d9f9b9eea001b6cb5.png)

这里的第4点提示让我感到有些疑惑，我还请教了下同事，顺便让同事补充一下注释。

![在这里插入图片描述](https://img-blog.csdnimg.cn/3c33c1f06f194b4e97e6b334fae190f5.png)

整体来说，在阅读MLIR代码方面，Claude已经相当智能，全面领先Newbing（Chatgpt），感觉以后可以日常用Claude来辅助Review IR相关代码。

# 0x3. 总结
我这里以MLIR的一个任务对比了一下ChatGpt和Claude，我感受到了Calude的强大之处。虽然暂时还没有评测过别的任务，但我已经被Calude表现出来的代码分析能力所震撼。我们甚至可以将Claude作为一个入门AI编译器的入门工具


--------------------------------分割线-------------------------------------

评论区有朋友提出newbing的一些功能被限制了，并不等价于chatgpt3.5，我借了一个官方的chatgpt账号重新测试了一下，以下是测试结果：

![在这里插入图片描述](https://img-blog.csdnimg.cn/a29128a56b484878bd07d7fd65d796f4.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/7a959c8e7f79477f9746ec8924d275e6.png)
就这个例子来说，chatgpt的解释没有Claude那么细节，Claude的结果确实比chatgpt的好一点，不过chatgpt确实知道这个是MLIR的Pass，不像newbing那样被限制。


### EliminateAllocOpsPass

接下来问问 EliminateAllocOpsPass 的实现：


![在这里插入图片描述](https://img-blog.csdnimg.cn/d9f41af1ef164b5fb56fe11a310f0af0.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/82d875fec1d64551b90bfcf6c53fba1b.png)
我们可以对比下上面Calude的结果，感觉针对这个问题ChatGPT的描述以及理解是不如Claude那么自然的。从这个回答里面我们并不能看出ChatGPT理解了这个实现的原理，而Claude则完全理解了。

![在这里插入图片描述](https://img-blog.csdnimg.cn/4dcdf06a6bae4fb9acf2f40c0b63b21a.png)
### AppendOneFlowStreamPattern
![在这里插入图片描述](https://img-blog.csdnimg.cn/88e12220bd8144d7a80bc8505fd4cf61.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/5c500f07cb1446f08f687ec63abb182a.png)

对比下Claude：

![在这里插入图片描述](https://img-blog.csdnimg.cn/0d48cb6b887d482e9c7c7b7cc3d2a716.png)

可以看到Claude的分析比ChatGPT好很多，它明确的知道	`if (llvm::dyn_cast<LLVM::LLVMPointerType>(op.getFunctionType().getInputs().back()))` 这行代码是检查当前函数是否已经有Stream参数，而ChatGPT的回答则不知道这个指针类型的参数就代表Stream。

接下来是细节分析。

![在这里插入图片描述](https://img-blog.csdnimg.cn/647dafd5de7443ceb014b11912a01c3a.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0f91969e1ee94440a67814fc87e4f12c.png)
对比下Claude

![在这里插入图片描述](https://img-blog.csdnimg.cn/08b6327eb2024a7587d61cffbba9423a.png)Claude的解释再次击败了ChatGPT


![在这里插入图片描述](https://img-blog.csdnimg.cn/6a6ca830ed784236bae1d319c09724c2.png)
对比下Claude

![在这里插入图片描述](https://img-blog.csdnimg.cn/1a9660ce29154367833792f9865081ea.png)可以看到Claude的结果显然也是更优的，不仅为我们解释了所有细节还列出了用到的MLIR相关属性和接口。



### MgpuToOneFlowStreamPass

我们最后再分析下MgpuToOneFlowStreamPass的实现。

![在这里插入图片描述](https://img-blog.csdnimg.cn/21710e3fc1c640e8bfa485bca0be920e.png)
对比Claude

![在这里插入图片描述](https://img-blog.csdnimg.cn/12af066270d441339ff9b844697fbc96.png)Claude的结果也显著优于ChatGPT，并且可以发现ChatGPT的回答里面还漏掉了一个mgpuStreamSynchronize ABI。最后，我们再问一下ChatGPT能不能给出一些修改意见。

![在这里插入图片描述](https://img-blog.csdnimg.cn/3c620543e56f4716af50168095e7057a.png)
感觉和Claude差不多。

# 结论2

整体来看，在这个Review MLIR代码的任务中，Claude > ChatGPT >> NewBing
