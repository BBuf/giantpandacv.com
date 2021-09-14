# 0x0. 前言
这篇文章基于TVM 0.8.0.dev版本。在[【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA) 这篇推文中已经简单介绍了Relay和Pass机制。但对Pass的基础设施（Pass Infrastructure）和Relay树结构都没有详细介绍，所以这篇文章主要介绍一下Pass Infrastructure和Relay树结构，再基于这些关键的基础知识详细了解一下Constant Folding Pass，相信读者读完这篇文章会对TVM的Pass有更深的理解，并且在阅读其它Pass和实现自定义Pass时可以很Relax。

# 0x1. Pass Infrastructure
首先来看Pass Infrastructure，基于官方文档进行介绍。

在讲解Pass通用的注册和运行流程前，先来介绍一下TVM的Pass Infrastructure。参考官方文档：https://tvm.apache.org/docs/dev/pass_infra.html 。

Relay 和 TVM IR 都包含一系列优化passes，可提高模型的性能指标，例如平均推理速度、内存占用或特定设备的功耗。 TVM有一套标准优化方法以及特定于机器学习的优化方法，包括常量折叠、死代码消除、运算符布局更改、算符融合、缓冲区处理和循环变换等。 每一个Pass都使用在traversal期间和/或之前收集的分析结果来构造ir-to-ir的pass。

然而，随着TVM的迅速发展，需要一种更系统、更有效的方法来管理这些passes。此外，一个可以管理跨TVM堆栈不同层（如Relay和tir）的passes的通用框架，为开发人员快速原型化并将实现的passes插入系统铺平了道路。

例如，许多现有的生产编译器，如 GCC 和 LLVM，都采用pass manager来有效管理passes的执行。 最初管理 pass 很简单，因为 pass 的数量很少，但成熟的编译器将包含数百个单独的 pass。 Often external users will want to have custom passes correctly scheduled without having to modify a single handcrafted pass order.

同样，现代深度学习框架，如 Pytorch 和 MXNet Gluon，也有分别通过 Sequential 和 Block 启用pass-style层构建方案的趋势。 有了这样的结构，这些现代框架能够方便地将模块/层添加到它们的容器中，并轻松地构建神经网络。 

Relay pass infra 的设计很大程度上受到 LLVM 中使用的分层pass manager和流行的深度学习框架中使用的block-style容器的启发。 pass infra 的主要目标包括： 

- 实现更好的optimizer编程编排。 这允许用户灵活地定制和构建自己的优化管道。 
- 提供一种用户友好的方式来调试passes。
-  减轻开发人员手动和分别解决passes之间的依赖关系。 
- 为开发人员简化实现新passes的难度。 例如，我们允许用户在 Python 中实现一个 pass 并让 pass infra 操纵它的执行。 

**The Design**

我们专注于为用户提供易于扩展的功能，让用户可以快速添加新passes而不会失去向后兼容性。 该设计包含后端和前端。 前者实现了 pass infra 的主要逻辑。 后者为用户提供简单的 API 进行交互，即允许用户快速创建自己的优化管道。 

**C++ Backend**

我们提供了一个 `PassInfo` 对象来包含一个pass所需的基本信息。 `name` 是 pass 名称，`opt_level` 指示将启用 pass 的优化级别， `required` 表示执行某个 pass 所需的 pass（更多详细信息请参见` include/tvm/ir/transform.h`）。 例如，在注册pass的时候（将在后面介绍），pass开发人员可以指定pass的名称、将执行的优化级别和/或所需的pass。 `opt_level` 可用于帮助 pass infra 识别在用户提供的优化级别下运行时是否需要执行某个 pass。 `required`字段可以由pass infra用来解决pass依赖关系。

 

```cpp
class PassInfoNode : public Object {
  String name;
  int opt_level;
  Array<String> required;
};
```

**PassContext**

`PassContext` 带有用于优化pass的有用信息。 例如，它包含错误报告系统，因此pass的作者可以提供有关优化失败原因的注释。 `PassContext` 还旨在替换旧的`BuildConfig`，它用于帮助用户配置编译选项，包括优化级别和必需/禁用的pass等。例如，我们可能有一个配置，它在 `opt_level=3` 时执行所有pass，除开使用 `PassContext` 提供的 `disabled_pass=xx`禁用的一些passes 。 现在我们可以在 `opt_level=3` 处对所有passes进行全局处理，并排除禁用pass列表中的那些pass。

这个类是为方便用户编写Python而设计的，它的语法可以在特定的配置下执行优化。 此外，用户可以通过 `PassContext::Current() `以线程安全的方式获取某个程序范围内可用的context，因为ThreadLocalStore用于保存创建的pass context对象，关于ThreadLocalStore建议看这篇文章：https://zhuanlan.zhihu.com/p/61587053，TVM模仿Java中的ThreadLocalStore在C++层自己实现了用来管理线程。 稍后将提供示例以展示我们如何使用 C++ 和 Python API 来创建使用pass context的编译管道。 

```cpp
class PassContextNode : public Object {
 public:
  ErrorReporter err_reporter;
  int opt_level{2};
  tvm::Array<tvm::Expr> required_pass;
  tvm::Array<tvm::Expr> disabled_pass;
};

class PassContext : public NodeRef {
 public:
  TVM_DLL static PassContext Create();
  TVM_DLL static PassContext Current();
  /* Other fields are omitted. */

 private:
  // The entry of a pass context scope.
  TVM_DLL void EnterWithScope();
  // The exit of a pass context scope.
  TVM_DLL void ExitWithScope();

  // Classes to get the Python `with` like syntax.
  friend class tvm::With<PassContext>;
};

struct PassContextThreadLocalEntry {
  /*! \brief The default pass context. */
  PassContext default_context;
  /*! \brief The current pass context. */
  std::stack<PassContext> context_stack;
  PassContextThreadLocalEntry() {
    default_context = PassContext(make_node<PassContextNode>());
  }
};

/*! \brief The thread-local store to hold the pass context. */
typedef dmlc::ThreadLocalStore<PassContextThreadLocalEntry>
     PassContextThreadLocalStore;
```

**Pass Constructs**

pass infra 是以分层方式设计的，它可以在不同粒度的Relay/tir 程序下工作。 引入了一个纯虚拟类 `PassNode` 作为不同优化pass的基础。 此类包含几个必须由子类在modules, functions, or sequences of passes实现的虚拟方法。 

```cpp
class PassNode : Object {
  virtual PassInfo Info() const = 0;
  virtual Module operator()(const IRModule& mod
                            const PassContext& pass_ctx) const = 0;
};
```

成员函数展示了一个pass应该如何实现，例如它始终在特定context下工作在 `IRModule `中，所有的pass都被设计在一个`Module` to `Module`的管理器中。因此，由 pass infra 控制的优化将始终更新整个module。


已经创建了几个子类来实现不同类型的优化pass，例如，function-level passes, module-level passes, and sequential passes。 每个子类本身都可以充当pass管理器。 例如，他们可以收集所需的passes并执行它们或基于给定的元数据构建依赖关系图。 它们的完整定义可以在`src/relay/ir/transform.cc 和 src/ir/transform.cc` 中找到。 


**Module-Level Passes**

Module Level Passes主要用于全局和过程间优化 (IPO)，类似于 LLVM 中使用的module pass。 Relay 中一些典型的 pass 需要一个模块的global picture，比如 A-normal form conversion 和 lambda lifting等，都属于这个集合。 在此级别，用户甚至可以在一个module中添加和/或删除function。 

```cpp
class ModulePassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  // Other members/methods are omitted
};
```

`pass_info` 维护module-level pass所需的信息。 `pass_func` 实现了真正的optimization。 例如，我们可能需要对module执行死代码消除。 我们可以在 `pass_func` 中实现算法并让它在module上运行。 然后它将删除死代码，包括module中未使用的函数。 请注意，该字段被设计为一个packed function，所以这个优化不仅可以使用C++还可以使用Python来实现。


 **Function-Level Passes**

Function-level passes用于为给定的 Relay/tir module实现各种内部函数级优化。 它一次从module的函数列表中获取一个函数以进行优化，并生成一个重写的 `Relay Function` 或 `tir PrimFunc`。 大多数pass可以归入这一类，例如Relay中的常见子表达式消除和inference simplification 以及tir中的向量化和flattening storage等。 

请注意，此级别的passes范围是 Relay Function或 tir PrimFunc。 因此，我们无法通过这些passes添加或删除函数，因为它们不知道全局信息。

```cpp
class FunctionPassNode : PassNode {
  PassInfo pass_info;
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
  bool SkipFunction(const Function& func) const;
  // Other members/methods are omitted...
};
```

`pass_info` 与我们刚刚在Module pass 中描述的相同。 `pass_func` 需要一个函数进行优化，它还需要一个Module，因为我们可能会使用它来报告错误。 一个函数可以用“SkipOptimization”注释，以便在优化过程中被忽略。

 **Sequential Passes**

 `SequentialPass` 类似于 Pytorch `nn.Sequential`，它包含许多用于执行的passes。

 

```cpp
class SequentialPassNode : PassNode {
  PassInfo pass_info;
  // Passes need to be executed.
  Array<Pass> passes;
  bool PassEnabled(const PassInfo& info) const;
  Module operator()(const Module& mod, const PassContext& pass_ctx) const final;
};
```

目前在Relay中只有少数passes 被放入这组中。 例如，`FoldScaleAxis` 需要在内部调度 `ForwardFoldScaleAxis` 和 `BackwardFoldScaleAxis`。 此外，建议先完成`BackwardFoldScaleAxis`。 因此，该pass是`SequentialPass`的理想候选者。 

以下代码显示了如何调用sequential pass中的各个pass。 

```cpp
Module SequentialNode::operator()(const Module& module,
                                  const PassContext& pass_ctx) const {
  Module mod = module;
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!PassEnabled(pass_info))  continue;
    for (const auto& it : pass_info->required) {
      const auto* name = it.as<tvm::ir::StringImm>();
      ICHECK(name);
      mod = GetPass(name->value)(mod, pass_ctx);
    }
    mod = pass(mod, pass_ctx);
  }
  return mod;
}
```

在调用pass时，我们首先检查是否启用了此pass。 这是通过首先检查用户是否明确禁用该pass，然后检查它是否被用户指定为必需pass来完成的。 如果仍然不确定是否启用了此传递，则将检查其 `opt_level`。 只有当它的`opt_level`不低于pass context中配置的优化级别时，才会启用并因此执行此pass。

要执行pass，我们首先需要使用pass name在 TVM packed function注册表中已注册的pass。 这是可能的，因为每个pass都注册了一个 API 接口，我们将在后面展示。 

```cpp
Pass GetPass(const std::string& pass_name) {
  using tvm::runtime::Registry;
  std::string fpass_name = "relay._transform." + pass_name;
  const auto* f = Registry::Get(fpass_name);
  ICHECK(f != nullptr) << "Cannot find " << fpass_name
                      << "to create the pass " << pass_name;
  return (*f)();
}
```

提供了一些helper function来创建上述每种类型的Pass。 这些helper function也暴露给 Python 前端，以便用户可以方便地使用 Python API 来创建特定的 pass 对象。 

```cpp
Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass CreateModulePass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level,
    String name,
    Array<String> required);

Pass Sequential(tvm::Array<Pass> passes, PassInfo pass_info);
```

**Pass Registration**

我们已经介绍了不同级别pass的概念和用于编译的context。 用户可以多么轻松地注册pass是一件有意义的事。，我们以constant folding为例。 这个 pass 已经被实现来折叠 Relay Function中的常量（在 `tvm/src/relay/transforms/fold_constant.cc` 中找到）。 

提供了一个 API 来执行 `Expr` 到 `Expr` 的转换。

 

```cpp
Expr FoldConstant(const Expr& expr);
```

为了将这个pass注册到pass infra，我们首先需要决定这个pass将在哪个级别执行。 由于常量折叠发生在单个函数上，我们应该直观地通过 `CreateFunctionPass`为其创建一个 `FunctionPass`。 `pass_func` 作为packed function返回，该函数在 `IRModule` 中的每个function上调用 `Expr` to `Expr` API。 `{}` 表示此pass不需要先决条件。 否则，pass开发人员必须识别并列出它们。

 

```cpp
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(FoldConstant(f));
  };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);

}  // namespace transform
```

为了允许其他 C++ 模块应用此pass，我们在 `include/tvm/relay/transform.h `中声明了一个free function，如下所示： 

```cpp
TVM_DLL Pass FoldConstant();
```

**Python Frontend**

python前端只需要一些简单的 APIs。 例如，我们可以为用户提供以下 APIs 来创建和执行一个 pass（完整的实现在 `python/tvm/relay/transform.py` 和 `python/tvm/ir/transform.py` 中提供）。 后端接收信息并决定它应该使用哪个函数来创建 Pass 对象。 

**PassContext**

Python 前端为 `PassContext` 提供了一个包装器，通过覆盖 `__enter__` 和 `__exit__` 来启用 `with` 语法。 为用户提供了一个 `current` 静态方法来获取在特定范围内使用的上下文。 

```python
@tvm._ffi.register_object("transform.PassContext")
class PassContext(tvm.runtime.Object):
    def __enter__(self):
        _transform.EnterPassContext(self)
        return self

    def __exit__(self, ptype, value, trace, config):
        _transform.ExitPassContext(self)

    @staticmethod
    def current():
        """Return the current pass context."""
        return _transform.GetCurrentPassContext()
```

PassContext 用于配置编译选项，包括优化级别和必需/禁用的pass。 它还可以带一个配置字典，以便不同的pass可以方便地获取passed的数据，例如回退设备信息和循环展开的步数/深度等。 为了能够获取所需的配置，必须通过`TVM_REGISTER_PASS_CONFIG_OPTION`注册关键字。 例如，loop unrolling pass使用以下内容：

```python
TVM_REGISTER_PASS_CONFIG_OPTION("tir.UnrollLoop", UnrollLoopConfig);
```

更多细节请参考 `src/tir/transforms/unroll_loop.cc`。 

**Pass Objects**

`Pass` 是所有 pass 对象的基类。 这里的所有方法都只是在后端实现的简单包装器。 它们是为了用户方便地与 Python 中的基类进行交互而定义的。 在 pass 基类中只定义了一个` __call__ `来使子类成为可调用对象，以便它们可以很容易地被调用（例如 `pass_xx(arg)`）来执行。

 

```python
@register_relay_node
class Pass(RelayNode):
   def __call__(self, mod):
       return _transform.RunPass(self, mod)
```

提供了一些辅助 APIs 以支持从 Python 前端轻松创建pass并让pass infra控制执行。 比如提供给用户`module_pass`、`function_pass`、`sequential`，让他们可以自定义自己的pass或者pass管道。 

对于在C++后端实现的所有pass，我们分别在`python/tvm/ir/transform.py`和`python/tvm/relay/transform.py`中提供了相应的Python API。 例如，const 折叠有一个 Python API，如下所示： 

```python
def FoldConstant():
    return _transform.FoldConstant()
```

用户可以通过装饰器像下面这样构建一个pass： 


```python
 @relay.transform.module_pass(opt_level=2)
 def transform(mod, ctx):
    tp = relay.TensorType((10,), "float32")
    x = relay.var("x", tp)
    gv = relay.GlobalVar("abs")
    func = relay.Function([x], relay.abs(x))
    new_mod = relay.Module({gv: func})
    new_mod.update(mod)
    return new_mod

module_pass = transform
assert isinstance(module_pass, transform.ModulePass)
assert module_pass.info.opt_level == 2
```

这里的`transform`函数向输入的module添加了一个` abs` 函数，但它可以是module level的任何自定义pass。 创建此 `module_pass` 后，用户可以将其应用于任何 `Relay` 模块。 例如，我们可以构建一个empty module并应用此pass来添加 `abs` 函数。

 

```python
mod = relay.Module()
mod = module_pass(mod)
```

相应地，我们也为 `function_pass` 提供了这样的功能。 例如，一个示例function-level pass可以写成如下：

 

```python
@relay.transform.function_pass(opt_level=1)
class TestReplaceFunc:
   def __init__(self, new_func):
      self.new_func = new_func
      def transform_function(self, func, mod, ctx):
         # Just for demo purposes
         # Transform func to new_func
         return self.new_func

x = relay.var("x", shape=(10, 20))
f1 = relay.Function([x], x)
f2 = relay.Function([x], relay.log(x))
# fpass is now a special pass that replaces every
# function to f1
fpass = TestReplaceFunc(f1)
# Now every function in input_mod is replaced by f1
res_mod = fpass(input_mod)
```

或者，用户也可以不使用装饰器直接注册pass，然后调用它。 有关如何自定义您自己的优化管道以及调试 Relay 和 tir pass 的更多示例，请参阅 use pass infra 教程（`https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_infra.py`）。 



# 0x2. TVM Relay树结构

## AST

> 摘自wiki

> 在计算机科学中，抽象语法树（Abstract Syntax Tree，AST），或简称语法树（Syntax tree），是源代码语法结构的一种抽象表示。它以树状的形式表现编程语言的语法结构，树上的每个节点都表示源代码中的一种结构。之所以说语法是“抽象”的，是因为这里的语法并不会表示出真实语法中出现的每个细节。比如，嵌套括号被隐含在树的结构中，并没有以节点的形式呈现；而类似于 if-condition-then 这样的条件跳转语句，可以使用带有三个分支的节点来表示。

> 和抽象语法树相对的是具体语法树（通常称作分析树）。一般的，在源代码的翻译和编译过程中，语法分析器创建出分析树，然后从分析树生成AST。一旦AST被创建出来，在后续的处理过程中，比如语义分析阶段，会添加一些信息。 

之前在解析TVM Relay的ONNX前端的时候，已经提到在完成每个OP转换之后需要使用`IRModule.from_expr`将所有转换后的Relay Function包起来返回，过程如下，这里关心最后一行代码即可：

```python
def from_onnx(self, graph, opset, get_output_expr=False):
        """基于ONNX模型构建Relay IR。

        参数
        ----------
        graph : onnx protobuf 对象
           加载进来的ONNX Graph

        opset : 操作集版本

        get_output_expr: bool
            如果设置为true，则此转换将返回每个输出表达式，而不是打包的模块。 
            将子图转换为Relay时，这可能很有用。 

        Returns
        -------
        mod : tvm.IRModule
            The returned relay module

        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        self.opset = opset
        # 解析网络的输入到relay中, 又叫参数，onnx的initializer就是用来保存模型参数的
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            # 具体实现就是先把这个TensorProto使用get_numpy函数获得值，再reshape到特定形状，再基于这个numpy构造tvm.nd.array。
            array = self._parse_array(init_tensor)
            # 前面解释过，如果设置冻结参数，则将这个参数设置为Relay中的常量OP
            if self._freeze_params:
                
                self._nodes[init_tensor.name] = _expr.const(array)
            else:
                self._params[init_tensor.name] = array
                self._nodes[init_tensor.name] = new_var(
                    init_tensor.name,
                    shape=self._params[init_tensor.name].shape,
                    dtype=self._params[init_tensor.name].dtype,
                )
        # 解析ONNX模型的输入
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            # 获取i这个输入的名字，shape，数据类型以及shape每个维度对应的名字
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            # 判断i这个输入是权重参数还是输入
            if i_name in self._params:
                # i is a param instead of input
                self._num_param += 1
                self._params[i_name] = self._params.pop(i_name)
                self._nodes[i_name] = new_var(
                    i_name, shape=self._params[i_name].shape, dtype=self._params[i_name].dtype
                )
            # 输入节点已经在Relay IR中了就不用处理了
            elif i_name in self._nodes:
                continue
            else:
                # 真正的输入节点，依赖用户进行指定
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]
        # Only check user inputs in the outer-most graph scope.
        if self._old_manager is None:
            assert all(
                [name in self._input_names for name in self._shape.keys()]
            ), "User specified the shape for inputs that weren't found in the graph: " + str(
                self._shape
            )
        # 获取不支持的算子列表
        convert_map = _get_convert_map(opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        # 输出不支持的算子集合
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)
        # 到这里说明这个ONNX模型的所有算子都被Relay支持，可以正常进行转换了
        for node in graph.node:
            op_name = node.op_type
            # 解析attribute参数
            attr = self._parse_attr(node.attribute)
            # 创建并填充onnx输入对象。
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    # self._renames.get(i, i)用来获取ONNX Graph每个节点的输入
                    inputs[i] = self._nodes[self._renames.get(i, i)]
                else:
                    inputs[i] = None
            i_name = self._parse_value_proto(node)
            node_output = self._fix_outputs(op_name, node.output)
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(node_output)
   # 执行转换操作
            op = self._convert_operator(op_name, inputs, attr, opset)
            # op的输出可能只有一个也可能有多个
            if not isinstance(op, _expr.TupleWrapper):
                outputs_num = 1
            else:
                outputs_num = len(op)
            if outputs_num > 1:
                # ONNX的某些节点支持可选输出
                # 这一块在ONNX的Graph中搜索缺失的输出并移除不需要的节点
                valid_outputs = [False] * outputs_num
                for i, output in enumerate(node_output):
                    if output != "":
                        valid_outputs[i] = True
                # If we have outputs ONNX isn't expecting, we need to drop them
                # 如果我们有ONNX不期望出现的输出，我们需要删除它们
                if not all(valid_outputs):
                    tup = op.astuple()
                    # TupleWrapper can also wrap ops with TupleType outputs
                    if isinstance(tup, _expr.Tuple):
                        # For tuples, we extract the fields instead of using GetTupleItem
                        outputs = [tup.fields[i] for i, valid in enumerate(valid_outputs) if valid]
                    else:
                        # For call nodes, we need to GetTupleItem
                        outputs = [op[i] for i, valid in enumerate(valid_outputs) if valid]
                    # Create the new op with valid outputs
                    if len(outputs) == 1:
                        op = outputs[0]
                    else:
                        op = _expr.TupleWrapper(outputs, len(outputs))
                    # Drop invalid outputs for the onnx node
                    outputs_num = len(outputs)
                    node_output = [output for output in node_output if output != ""]
            assert (
                len(node_output) == outputs_num
            ), "Number of output mismatch {} vs {} in {}.".format(
                len(node_output), outputs_num, op_name
            )
            # 输出只有一个有可能是常量OP，可以执行一次常量折叠功能
            if outputs_num == 1:
                self._nodes[node_output[0]] = fold_constant(op)
            else:
                op = _expr.TupleWrapper(fold_constant(op.astuple()), len(op))
                for k, i in zip(list(node_output), range(len(node_output))):
                    self._nodes[k] = op[i]

        # 解析ONNX模型的输出
        outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
        outputs = outputs[0] if len(outputs) == 1 else _expr.Tuple(outputs)
        # 如果需要直接返回转换后的表达式，在这里return
        if get_output_expr:
            return outputs
        # 保持来自ONNX Graph的输入和参数顺序，但仅仅包含这些需要执行转换到Relay的节点
        free_vars = analysis.free_vars(outputs)
        nodes = {v: k for k, v in self._nodes.items()}
        free_vars = [nodes[var] for var in free_vars]
        for i_name in self._params:
            if i_name in free_vars and i_name not in self._inputs:
                self._inputs[i_name] = self._nodes[i_name]
        # 根据我们的输出表达式和所有输入变量创建一个函数。 
        func = _function.Function([v for k, v in self._inputs.items()], outputs)
        # 把这个函数用IRModule包起来返回，并同时返回权重参数
        return IRModule.from_expr(func), self._params
```

这里`IRModule.from_expr(func)`就完成了Relay 抽象语法树结构的构建，TVM将这个树结构定义为`tvm.IRModule`这个类，也即Relay IR。

## Relay 树结构
现在来学习一下Relay 抽象语法树，也就是`tvm.IRModule`相关的数据结构。

### 节点定义
树的节点定义为在`/include/tvm/relay/expr.h` 中，主要有以下几种类型：ConstantNode、VarNode、TupleNode、CallNode、LetNode、IfNode。

这些Node都继承了在`include/tvm/ir/expr.h`定义的`RelayExprNode`，而`RelayExprNode`又继承了`BaseExprNode`，`RelayExprNode`可以做什么可以参考这几行注释：

```cpp
/*!
 * \brief Base node of all non-primitive expressions.
 *
 * RelayExpr supports tensor types, functions and ADT as
 * first class citizens. The life-cycle of the corresponding
 * objects are implicitly managed by the language.
 *
 * \sa RelayExpr
 */

/*!
  * \brief 所有非原始表达式的基节点。
  *
  * RelayExpr 支持张量类型、函数和 ADT 作为
  * 一等公民。 对应的生命周期
  * 对象由语言隐式管理。
  *
  * \sa RelayExpr
  */ 
```

然后这里以IfNode和CallNode为例看一下它们的实现：

```cpp
class IfNode : public ExprNode {
 public:
  /*! \brief The condition */
  Expr cond;
  /*! \brief The expression evaluated when condition is true. */
  Expr true_branch;
  /*! \brief The expression evaluated when condition is false */
  Expr false_branch;

};

class CallNode : public ExprNode {
 public:
  /*!
   * \brief The operator(function) being invoked
   *
   *  - It can be tvm::Op which corresponds to the primitive operators.
   *  - It can also be user defined functions (Function, GlobalVar, Var).
   */
  Expr op;

  /*! \brief The arguments(inputs) of the call */
  tvm::Array<relay::Expr> args;

  /*! \brief The additional attributes */
  Attrs attrs;

};
```

这里展示了这些节点的成员变量，可以大致了解到这些节点的内部结构。

### 节点的数据访问
在了解了Relay模型树节点后，我们需要知道TVM是如何去访问这些节点的数据的。在官方文档中可以找到这样一句话：`ExprVisitor`用于不修改程序而是执行程序分析和收集信息的passes。而`ExprVisitor`又继承自`ExprFunctor`（定义在`tvm/include/tvm/relay/expr_functor.h`），`ExprFunctor`设置了`VisitExpr_`的虚函数，在解析时会回到`ExprVisitor`来解析节点。 `ExprFunctor`提供了一个`public`接口方法`VisitExpr`，它接受一个表达式和零个或多个参数并返回某种类型的实例。 当你扩展这个类时，你通过为每种类型的表达式覆盖`VisitExpr_`的实现来定义 AST 遍历模式。

`VisitExpr`和`VisitExpr_`之间的关系与调度有关。 每个`VisitExpr_`定义针对特定类型的表达式，但你并不总是知道你将访问节点是哪种类型。 为了解决这个问题，`ExprFunctor`提供了一个`VisitExpr`函数，它从给定的表达式路由到处理它的`VisitExpr_`case。 尽管 C++ 已经提供了动态调度，但`ExprFunctor`定义了自己的 vtable，`VisitExpr`使用它。 通过定义我们自己的vtable，我们可以更好地控制调度。 例如，如果我们想定义一个`PrintVisitor`遍历器，在每次访问之前打印“Here”，我们可以覆盖`VisitExpr`：

 

```cpp
void PrintVisitor::VisitExpr(const Expr& expr) {
  std::cout << "Here" << std::endl;
  ExprFunctor::VisitExpr(expr);
}
```

`ExprFunctor`本身是一个非常通用的类，这就是为什么通常会扩展`ExprVisitor`或`ExprMutator`的原因。 这些类扩展了`ExprFunctor` 并提供`VisitExpr_ `的默认实现，用于捕获每个表达式类型的常见遍历模式。 拥有这些默认实现意味着我们只需要为需要不同行为的表达式类型提供进行重写`VisitExpr_ `方法即可。 


比如对于`tvm/src/relay/transforms/fold_constant.cc`中的`ConstantChecker`这个类，就继承了`ExprVisitor`，并通过`VisitExpr(expr)`，访问数据。`ExprVisitor`的`VisitExpr`成员函数实现如下：

```cpp
void ExprVisitor::VisitExpr(const Expr& expr) {
  auto it = visit_counter_.find(expr.get());
  if (it != visit_counter_.end()) {
    ++it->second;
  } else {
    using TParent = ExprFunctor<void(const Expr&)>;
    TParent::VisitExpr(expr);
    visit_counter_.insert({expr.get(), 1});
  }
}

```

可以看到这个类实际上调用的是父类(`ExprFunctor`)的`VisitExpr`，而`ExprFunctor`的`VisitExpr`的实现如下：

```cpp
virtual R VisitExpr(const Expr& n, Args... args) {
    ICHECK(n.defined()) << "Found null pointer node while traversing AST. The previous pass may "
                           "have generated invalid data.";
    static FType vtable = InitVTable();
    return vtable(n, this, std::forward<Args>(args)...);
  }
```

可以看到`ExprFunctor`设置了`VisitExpr`虚函数，在解析时会回到`ExprVisitor`来解析节点，而`ConstantChecker`这个类继承了`ExprVisitor`，这样我们只需要在`ConstantChecker`类中重写`VisitExpr_`就可以了。

在`ExprFunctor`的`VisitExpr`实现中有一个`RELAY_EXPR_FUNCTOR_DISPATCH`宏，这个宏的定义如下：

```cpp
#define RELAY_EXPR_FUNCTOR_DISPATCH(OP)                                                    \
  vtable.template set_dispatch<OP>([](const ObjectRef& n, TSelf* self, Args... args) {     \
    return self->VisitExpr_(static_cast<const OP*>(n.get()), std::forward<Args>(args)...); \
  });

```
这里的`self`即为`ExprFunctor`的`VisitExpr`的实现中的`vtable(n, this, std::forward<Args>(args)...)`，而`this`指向`ExprFunctor`。又因为`ExprVisitor::VisitExpr`方法调用的是`ExprFunctor`的函数，所以这里的`this`指向的是`ExprVisitor`实例。

以IfNode为例子，看看`ExprVisitor`的`VisitExpr_`实现。由于`this`指向的是`ExprVisitor`实例，最后会在`ExprVisitor`实例中生成`visit_counter_`的列表。

```cpp
void ExprVisitor::VisitExpr_(const IfNode* op) {
  this->VisitSpan(op->span);
  this->VisitExpr(op->cond);
  this->VisitExpr(op->true_branch);
  this->VisitExpr(op->false_branch);
}
```

`visit_counter_`是在`ExprVisitor`中定义的一个`unordered_map`，来标记在遍历Relay AST时某种Expr是否出现，同时记录下出现的次数。

```cpp
// Internal visiting counter
  std::unordered_map<const Object*, size_t> visit_counter_;
```

### 节点修改
pass是对Relay 树结构，也可以说计算图进行优化，优化必然设计到对图结构的修改。这就是上面提到的`ExprMutator`子类，它和`ExprVisitor`一样继承自`ExprFunctor`。类的定义如下：

```cpp
class ExprMutator : public ::tvm::relay::ExprFunctor<Expr(const Expr&)> {
 public:
  /*!
   * \brief Mutate is alias for VisitExpr
   * \return expr.
   */
  Expr Mutate(const Expr& expr) { return this->VisitExpr(expr); }
  Expr VisitExpr(const Expr& expr) override;
  Expr VisitExpr_(const VarNode* op) override;
  Expr VisitExpr_(const ConstantNode* op) override;
  Expr VisitExpr_(const GlobalVarNode* op) override;
  Expr VisitExpr_(const OpNode* op) override;
  Expr VisitExpr_(const TupleNode* op) override;
  Expr VisitExpr_(const FunctionNode* op) override;
  Expr VisitExpr_(const CallNode* call_node) override;
  Expr VisitExpr_(const LetNode* op) override;
  Expr VisitExpr_(const IfNode* op) override;
  Expr VisitExpr_(const TupleGetItemNode* op) override;
  Expr VisitExpr_(const RefCreate来表记Node* op) override;
  Expr VisitExpr_(const RefReadNode* op) override;
  Expr VisitExpr_(const RefWriteNode* op) override;
  Expr VisitExpr_(const ConstructorNode* op) override;
  Expr VisitExpr_(const MatchNode* op) override;

  /*!
   * \brief Used to visit the types inside of expressions.
   *
   * Can be overloaded to transform the types in arbitrary
   * ways, one way would be to define a sub-class of type
   * visitor for types which transform them appropriately.
   */
  virtual Type VisitType(const Type& t);
  virtual Clause VisitClause(const Clause& c);
  virtual Pattern VisitPattern(const Pattern& c);

 protected:
  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> memo_;
};

```

我们需要关注的是`memo_`这个成员变量，然后我们看一下这个类的`VisitExpr`实现：


```cpp
Expr ExprMutator::VisitExpr(const Expr& expr) {
  auto it = this->memo_.find(expr);
  if (it != this->memo_.end()) {
    return it->second;
  } else {
    Expr new_expr = ExprFunctor::VisitExpr(expr);
    memo_[expr] = new_expr;
    return new_expr;
  }
}
```

可以看到`memo_`存储了图中的各个节点。参考IfNode的实现：

```c++
Expr ExprMutator::VisitExpr_(const IfNode* op) {
  auto guard = this->Mutate(op->cond);
  auto true_b = this->Mutate(op->true_branch);
  auto false_b = this->Mutate(op->false_branch);
  if (op->cond.same_as(guard) && op->true_branch.same_as(true_b) &&
      op->false_branch.same_as(false_b)) {
    return GetRef<Expr>(op);
  } else {
    return If(guard, true_b, false_b, op->span);
  }
}
```

如果IFNode的子节点都没有被修改，那么就返回这个节点本身。否则创建新的节点`If(guard, true_b, false_b, op->span);`并返回。这里构造新节点的类`If`的定义和实现分别在`tvm/src/relay/ir/expr.h`和`tvm/src/relay/ir/expr.cc`中：

```cpp
class If : public Expr {
 public:
  /*!
   * \brief The constructor
   * \param cond The condition of a if node.
   * \param true_branch The fall through branch
   * \param false_branch The branch for execution when condition is false.
   * \param span The source span of the expression.
   */
  TVM_DLL If(Expr cond, Expr true_branch, Expr false_branch, Span span = Span());

  TVM_DEFINE_OBJECT_REF_METHODS(If, RelayExpr, IfNode);
};

If::If(Expr cond, Expr true_branch, Expr false_branch, Span span) {
  ObjectPtr<IfNode> n = make_object<IfNode>();
  n->cond = std::move(cond);
  n->true_branch = std::move(true_branch);
  n->false_branch = std::move(false_branch);
  n->span = std::move(span);
  data_ = std::move(n);
}
```

### 总结

这一节主要解析了Relay 表达式树的数据结构，TVM的所有pass都是基于在`tvm/include/tvm/relay/expr.h`中定义的各种Node组成的表达式树来完成的，也可以说是计算图。另外还讲解了TVM为了方便对这些表达式节点进行访问和操作抽象出了`ExprFunctor`这个类，并在`ExprFunctor`这个类的基础上扩展`ExprVisitor`或`ExprMutator`，这在实现各个Pass的C++后端代码时非常有用。最后我以IfNode的实现和常量折叠Pass中的`ConstantChecker`类实现为例，展示了这些类的具体用法。


# 0x3. Function Pass的C++后端通用创建流程

这里先基于Constant Folding Pass讲解一下Function Pass的C++后端通用创建流程。这里先看一下FoldConstant的定义：

```cpp
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldConstant(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform
```

`CreateFunctionPass`这个函数用来创建FunctionPass，相关代码如下：

```cpp

Pass CreateFunctionPass(
    const runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return FunctionPass(pass_func, pass_info);
}

FunctionPass::FunctionPass(
    runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<FunctionPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}
```

可以看到在`FunctionPass`中创建了一个`FunctionPassNode`实例并将其放到`data_`中，`data_`来自于`ObjectRef`这个类的成员变量，这里`FunctionPass->Pass->ObjectRef `。

如果将上述代码生成的`Pass`对象提供给Pass Infrastructure，它将确保将 AST 遍历应用于给定 Relay Module中的每个Function，这是我们对Constant Folding Pass所期望的行为（它应该尽可能折叠所有常量）。

函数`CreateFunctionPass`允许注册传递的优化级别（在本例中为2），可用于根据pass的通用效用、pass的名称以及pass的任何依赖关系将pass组合在一起。一个pass的依赖是一系列可能会对这个pass的结果产生影响的pass。 `FoldConstant`没有任何依赖。但是很多Relay pass确实依赖于类型信息，所以`InferType`是一个常见的依赖；others may depend on the program’s being in A-normal form, via the ToANormalForm pass.

注意，`PassContext` 对象包含传递用于错误报告和配置选项的信息； `FoldConstant`不需要此信息，但其它Pass可能会引用它们的`PassContext`对象。 

现在可以通过Pass Infrastructure调用pass，不过最好也为 pass 添加 Python 绑定，如以下代码片段所示： 


```cpp
TVM_REGISTER_GLOBAL("relay._transform.FoldConstant")
.set_body_typed(FoldConstant);
```

一旦以上述方式定义了 Pass 对象，就可以使用 Pass 基础结构的 Sequential 构造调用它们，该构造采用传递列表并将它们按顺序应用于Relay 模块，从而获得转换后的Module。 例如，下面的代码将 FoldConstant 和 ToANormalForm Pass（一个接一个）应用于`mod`中的每个函数并获得一个新Module。

 

```cpp
seq = transform.Sequential([
    relay.transform.FoldConstant(),
    relay.transform.ToANormalForm()
])
new_mod = seq(mod)
```

我们可以看一下`Sequential`的调用流程：

```cpp
// TODO(zhiics): we currenlty only sequentially execute each pass in
// a Sequential without the consideration of their orders. The phase
// ordering problem needs to be handled in the future.
IRModule SequentialNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  for (const Pass& pass : passes) {
    ICHECK(pass.defined()) << "Found undefined pass for optimization.";
    const PassInfo& pass_info = pass->Info();
    if (!pass_ctx.PassEnabled(pass_info)) continue;
    // resolve dependencies
    for (const auto& it : pass_info->required) {
      mod = GetPass(it)(std::move(mod), pass_ctx);
    }
    mod = pass(std::move(mod), pass_ctx);
  }
  return mod;
}

```

这里分成两个部分，如果pass有依赖，则先运行依赖pass。GetPass会在`relay._transform`的列表中根据命名返回对应的pass。代码实现如下：

```cpp
Pass GetPass(const String& pass_name) {
  using tvm::runtime::Registry;
  const runtime::PackedFunc* f = nullptr;
  if (pass_name.operator std::string().find("transform.") != std::string::npos) {
    f = Registry::Get(pass_name);
  } else if ((f = Registry::Get("transform." + pass_name))) {
    // pass
  } else if ((f = Registry::Get("relay._transform." + pass_name))) {
  }
  ICHECK(f != nullptr) << "Cannot use " << pass_name << "to create the pass";
  return (*f)();
}

```

接着再跟进一下`mod = pass(std::move(mod), pass_ctx);`，代码实现如下：

```cpp
IRModule Pass::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassNode* node = operator->();
  ICHECK(node != nullptr);
  PassProfile::EnterPass(node->Info()->name);
  auto ret = node->operator()(std::move(mod), pass_ctx);
  PassProfile::ExitPass();
  return std::move(ret);
}

// 创建PassNode类型的node实例
const Object* operator->() const { return get(); }
// 虚函数，需要子类重写
virtual IRModule operator()(IRModule mod, const PassContext& pass_ctx) const = 0;
```

这里关注一下`virtual IRModule operator()(IRModule mod, const PassContext& pass_ctx)`的接口实现，具体到FunctionPassNode重写这个`operator()`方法，因为FunctionPassNode继承了PassNode，代码实现如下：

```cpp
// Perform Module -> Module optimizations at the Function level.
IRModule FunctionPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  DiagnosticContext previous = DiagnosticContext::Default(mod);

  if (pass_ctx->diag_ctx) {
    DiagnosticContext tmp = pass_ctx->diag_ctx.value();
    pass_ctx->diag_ctx = previous;
    previous = tmp;
  } else {
    pass_ctx->diag_ctx = previous;
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  const PassInfo& pass_info = Info();

  ICHECK(mod.defined());

  DLOG(INFO) << "Executing function pass : " << pass_info->name
             << " with opt level: " << pass_info->opt_level;

  pass_ctx.Trace(mod, pass_info, true);

  // Execute the pass function and return a new module.
  IRModule updated_mod =
      IRModule(mod->functions, mod->type_definitions, mod->Imports(), mod->source_map);

  std::vector<std::pair<GlobalVar, Function> > updates;
  for (const auto& it : updated_mod->functions) {
    // only picks up relay::Function
    if (auto* n = it.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(n);
      auto updated_func = SkipFunction(func) ? func : pass_func(func, updated_mod, pass_ctx);
      updates.push_back({it.first, updated_func});
    }
  }

  for (const auto& pair : updates) {
    updated_mod->Add(pair.first, pair.second, true);
  }

  ICHECK(pass_ctx->diag_ctx)
      << "The diagnostic context was set at the top of this block this is a bug.";

  pass_ctx->diag_ctx.value().Render();
  pass_ctx->diag_ctx = previous;

  pass_ctx.Trace(updated_mod, pass_info, false);

  // TODO(@jroesch): move away from eager type checking for performance reasons
  // make issue.
  return transform::InferType()(updated_mod);
}
```

这个实现比较复杂，但我们只需要关心这行`auto updated_func = SkipFunction(func) ? func : pass_func(func, updated_mod, pass_ctx);`，这是执行Pass的核心操作。

以上就是Function Pass的C++后端通用创建流程。

# 0x4. Constant Folding Pass

下面我们来看一下Constant Folding Pass的C++后端代码实现需要注意哪些东西，首先Constant Folding Pass属于Funtion-level Pass。入口依旧是Function Pass的注册接口：

```cpp
namespace transform {

Pass FoldConstant() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(FoldConstant(f, m));
      };
  return CreateFunctionPass(pass_func, 2, "FoldConstant", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FoldConstant").set_body_typed(FoldConstant);

}  // namespace transform
```

我们看一下`FoldConstant`的具体实现：

```cpp
Expr FoldConstant(const Expr& expr, const IRModule& mod) {
  return ConstantFolder(mod).Mutate(expr);
}

```

可以看到常量折叠主要调用了`ConstantFolder`这个类的`Mutate`函数。而ConstantFolder继承了MixedModeMutator这个类，MixedModeMutator这个类比较有趣，定义如下：

```cpp
/*! \brief 用于自定义重写Pass的非递归 DFS 图遍历
 *
 * MixedModeMutator 将 Expr 视为数据流图，并且每个 Expr 只重写一次。
 * mutated的结果将被记录到一个map中被重用，以便数据流上的本地transformation保留了图形结构
 *
* MixedModeMutator 提供与 ExprMutator 相同的递归 API，并使用
* 递归遍历大多数形式的 IR，但在幕后它扩展了嵌套的数据流区域
* 并迭代处理它们以防止堆栈溢出
 *
 * Uses Rewrite_ API of ExprRewriter for a cleaner split between recrusive and non-recursive
 * behavior.
 */
class MixedModeMutator : public ::tvm::relay::ExprMutator {
 public:
  MixedModeMutator(bool pre = false) : pre_{pre} {};
  Expr VisitExpr(const Expr& expr) final;

  virtual Expr DispatchVisitExpr(const Expr& expr);
  Expr VisitExpr_(const TupleNode* op) final { return Rewrite(op); };
  Expr VisitExpr_(const CallNode* call_node) final { return Rewrite(call_node); };
  Expr VisitExpr_(const TupleGetItemNode* op) final { return Rewrite(op); };
  /*!
   * \brief 用户应该重写 Rewrite_ 方法来实现他们的Pass。Rewrite_ functions will be able to rewrite
   *  the op only with data about the original node `pre` and the same node with modified 
   * inputs `post` and should not recurse.
   * \param pre 重写前的表达式节点。
   * \param post 具有重写输入的表达式。
   */
  virtual Expr Rewrite_(const TupleNode* pre, const Expr& post) { return post; }
  virtual Expr Rewrite_(const CallNode* pre, const Expr& post) { return post; }
  virtual Expr Rewrite_(const TupleGetItemNode* pre, const Expr& post) { return post; }

 protected:
  bool pre_;
  /*! \brief Implement Rewrite API by calling ExprMutator's VisitExpr_(op) to get a `post` node with
   * changed inputs.
   */
  template <typename T>
  Expr Rewrite(const T* op) {
    Expr post = ExprMutator::VisitExpr_(op);
    return Rewrite_(op, post);
  }

  virtual void VisitLeaf(const Expr& expr);
  virtual bool CheckVisited(const Expr& expr);
};
```

我们在0x2节讲到pass是对Relay 树结构，也可以说计算图进行优化，优化必然设计到对图结构的修改。这就是上面提到的`ExprMutator`子类，它和`ExprVisitor`一样继承自`ExprFunctor`，我们实现Pass其实就是为其重写`VisitExpr_`成员函数。但是在这个`MixedModeMutator `类中，`VisitExpr_`成员函数实际上又调用了`Rewrite_`，所以Constant Folding在修改节点时只需要重写这个`Rewrite_`成员函数即可。（绕来绕去的，要仔细看看）。

然后我们看一下`ConstantChecker`的实现，Constant Folding通过ConstantChecker递归实现了Expr的常量判断。可以从代码看出，Constant 主要是判断元素是否是ConstantNode，或者TupleNode里的元素都是ConstantNode。

```cpp
class ConstantChecker : private ExprVisitor {
 public:
  // Check whether an expression is constant. The results are memoized.
  bool Check(const Expr& expr) {
    // The `ConstantNode` case is common enough that we check directly for the
    // case here, to avoid the time overhead of dispatching through the vtable
    // and the space overhead of memoizing always-true results.
    if (expr.as<ConstantNode>()) {
      return true;
    }
    const auto it = memo_.find(expr);
    if (it != memo_.end()) return it->second;
    VisitExpr(expr);
    return memo_[expr];  // return memoized result or the default value false
  }

 private:
  std::unordered_map<Expr, bool, ObjectPtrHash, ObjectPtrEqual> memo_;

  void VisitExpr_(const TupleNode* n) final {
    bool result = true;
    for (const auto& field : n->fields) {
      if (!Check(field)) {
        result = false;
        break;
      }
    }
    memo_[GetRef<Tuple>(n)] = result;
  }
};

bool ConstantCheck(const Expr& e) { return ConstantChecker().Check(e); }
```

接下来我们就解析一下真正的常量融合发生的函数，即在ConstantFolder中重写的`Rewrite_`函数，代码实现如下：

```cpp
Expr Rewrite_(const CallNode* call, const Expr& post) final {
    if (inside_primitive) {
      return GetRef<Expr>(call);
    }
    static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");

    auto origin_args = call->args;
    call = post.as<CallNode>();
    // We don't constant fold function with zero arguments.
    // This is a heuristic that is useful.
    // For example it is harmful to fold ones(shape=(4, 5)).
    if (call->args.size() == 0) return post;
    const OpNode* op = call->op.as<OpNode>();
    if (op == nullptr) return post;
    // skip stateful ops.
    if (op_stateful.get(GetRef<Op>(op), false)) return post;
    // Try to evaluate shape_of op
    if (call->op == shape_of_op_ || call->op == vm_shape_of_op_) {
      return EvaluateShapeOf(post, origin_args, call->attrs);
    }

    if (call->op == ndarray_size_op_) {
      return EvaluateNdarraySize(post, origin_args, call->attrs);
    }

    // We should think about potentially constant evaluation over these ops too.
    static auto fnoncomputational = Op::GetAttrMap<TNonComputational>("TNonComputational");
    if (const auto* call_node = call->op.as<OpNode>()) {
      Op op = GetRef<Op>(call_node);
      if ((fnoncomputational.count(op) && fnoncomputational[op]) || (call->op == device_copy_op_)) {
        return GetRef<Call>(call);
      }
    }

    bool all_const_args = true;
    for (Expr arg : call->args) {
      if (!checker_.Check(arg)) {
        all_const_args = false;
      }
    }
    if (all_const_args) {
      return ConstEvaluate(post);
    } else {
      return post;
    }
  }
```


这里根据callnode的类型又分了几种情况，我们假设代码走到了`return ConstEvaluate(post);`这一行，这个函数为了完成常量折叠做了什么事情呢？

```cpp
// Constant evaluate an expression.
  Expr ConstEvaluate(Expr expr) {
    std::vector<transform::Pass> passes = {transform::FuseOps(0), transform::ToANormalForm(),
                                           transform::InferType()};
    Function func;
    if (expr.as<FunctionNode>()) {
      func = Downcast<Function>(expr);
    } else {
      // TODO(@jroesch): fix this
      func = Function(FreeVars(expr), expr, Type(), FreeTypeVars(expr, module_), {});
    }
    auto mod = IRModule({}, module_->type_definitions, module_->Imports());
    auto global = GlobalVar("main");
    mod->Add(global, func);
    auto seq = transform::Sequential(passes);
    mod = seq(mod);
    auto entry_func = Downcast<Function>(mod->Lookup("main"));
    expr = expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;

    using tvm::transform::PassContext;
    Device dev;
    dev.device_type = kDLCPU;
    dev.device_id = 0;
    Target target = Target("llvm");
    // use a fresh build context
    // in case we are already in a build context.
    // needed for both execution and creation(due to JIT)
    With<PassContext> fresh_build_ctx(PassContext::Create());

    FInterpreter executor = CreateInterpreter(mod, dev, target);
    return ObjectToExpr(executor(expr));
  }

```

可以看到这里增加了三个passes，接下来执行这三个Pass完成常量折叠的功能，这里有个ToANormalForm定义可以在wikipedia找到（https://en.wikipedia.org/wiki/A-normal_form）：

```cpp
passes = {transform::FuseOps(0), transform::ToANormalForm(),
transform::InferType()};
```

获得了Sequential Pass和target以及全局module信息之后执行ObjectToExpr完成常量值到表达式的转换。

```cpp
// Convert value to expression.
  Expr ObjectToExpr(const ObjectRef& value) {
    if (value->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(value);
      return Constant(nd_array);
    } else if (const auto* val = value.as<runtime::ADTObj>()) {
      runtime::ADT adt = GetRef<runtime::ADT>(val);
      Array<Expr> fields;
      for (size_t i = 0; i < adt.size(); ++i) {
        fields.push_back(ObjectToExpr(adt[i]));
      }
      return Tuple(fields);
    } else {
      LOG(FATAL) << "Cannot handle " << value->GetTypeKey();
      return Expr();
    }
  }
```

这个函数主要实现了基于runtime的结果生成新的Expr来代替原来的Expr。在`Rewrite_`函数中还有几种类型的CallNode的常量折叠实现这里就不介绍了，感兴趣的小伙伴可以自己看一下。


# 0x5. 笔者建议
我的建议是，如果你C++能力不是很出色，建议对于TVM的C++ backend Pass了解即可，不用深究，只需要按照第一节介绍的方法就可以将TVM已经实现的Pass玩的风声水起。如果你要自定义Pass，那么直接基于TVM提供的装饰器在Python层实现就可以了，具体参考：https://github.com/apache/tvm/blob/main/tutorials/dev/use_pass_infra.py。所以如果嫌弃文章太长，只看第一节就好了。

# 0x6. 总结

好像开头写好了，拷贝一下：这篇文章基于TVM 0.8.0.dev版本。在[【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA) 这篇推文中已经简单介绍了Relay和Pass机制。但对Pass的基础设施（Pass Infrastructure）和Relay树结构都没有详细介绍，所以这篇文章主要介绍一下Pass Infrastructure和Relay树结构，再基于这些关键的基础知识详细了解一下Constant Folding Pass，相信读者读完这篇文章会对TVM的Pass有更深的理解，并且在阅读其它Pass和实现自定义Pass时可以很Relax。


# 0x7. 推荐阅读

- [【从零开始学深度学习编译器】六，TVM的编译流程详解](https://mp.weixin.qq.com/s/CZzC5klWoFftUlOKkpvEZg)
- [【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA)
- [【从零开始学深度学习编译器】番外一，Data Flow和Control Flow](https://mp.weixin.qq.com/s/Kt4xDLo-NRui8Whl0DqcSA)
- [【从零开始学TVM】三，基于ONNX模型结构了解TVM的前端](https://mp.weixin.qq.com/s/KFxd3zf76EP3DFcCAPZjvQ)
- [【从零开始学深度学习编译器】二，TVM中的scheduler](https://mp.weixin.qq.com/s/fPpqKL3uaaJ5QlNS79DZ5Q)
- [【从零开始学深度学习编译器】一，深度学习编译器及TVM 介绍](https://mp.weixin.qq.com/s/sZLWjYebbHjCgQ6XAZCiOw)


# 0x8. 参考
- https://tvm.apache.org/docs
- https://zhuanlan.zhihu.com/p/151815380

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)