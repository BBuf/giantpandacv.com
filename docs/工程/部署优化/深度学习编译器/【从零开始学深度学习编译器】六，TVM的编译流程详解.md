# 一. 前言

上一篇文章对TVM Relay和Pass进行了介绍，但还没有介绍整体的编译流程。这一篇文章将继续介绍一下TVM的编译流程，即TVM是如何将深度学习框架的模型转换成Relay IR之后进一步编译和优化为硬件可以执行的IR，再将这个底层IR和运行时库以及模型参数打包为一个`tvm.Module`返回。关于为什么要将底层IR和运行时库以及模型参数打包，根据官方文档可以知道这样是为了可以更方便的保存底层IR和运行时库，做到一次编译，可持久化推理。

# 二. TVM编译流程详解

TVM的编译流程在Python端的调用方式非常简单：

```python
with tvm.transform.PassContext(opt_level=10):
    lib = relay.build(func, "llvm", params=params)
```

这里的`with tvm.transform.PassContext(opt_level=10)`是指定Pass的优化等级，在[【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA) 已经介绍了。这里就跟进一下`lib = relay.build(func, "llvm", params=params)`这行代码来看一下TVM的编译流程。

首先这里的`func`和`params`分别代表模型的**图结构**以及**权重参数**。`relay.build`这个函数定义在`tvm/python/tvm/relay/build_module.py`这个函数中，入口代码如下：

```python
@register_func("tvm.relay.build")
def _build_module_no_factory(mod, target=None, target_host=None, params=None, mod_name="default"):
    """A wrapper around build which discards the Python GraphFactoryRuntime.
    This wrapper is suitable to be used from other programming languages as
    the runtime::Module can be freely passed between language boundaries.
    """
    target, target_host = Target.check_and_update_host_consist(target, target_host)
    return build(mod, target, params=params, mod_name=mod_name).module

```

对于**上面调用的例子**，`target`为`llvm`代表这个模型会被TVM编译成CPU的可执行程序。`Target.check_and_update_host_consist`这个函数应该是用来检查目标设备类型`targer`以及`target`对应的`host`端是否指定正确的，如果指定正确则将这两个参数合并到一个`Target`类中并返回。`Target`这个类的实现在`tvm/python/tvm/target/target.py`这里，是用来管理TVM支持的设备后端的。

接着就来到了build这个函数，代码实现如下：

```python
def build(ir_mod, target=None, target_host=None, params=None, mod_name="default"):
    # fmt: off
    # pylint: disable=line-too-long
    """一个将Relay Function编译成可执行程序的函数

    参数
    ----------
    ir_mod : :py:class:`~tvm.IRModule`
        要编译的IR Module. 不推荐使用relay.Function

    target : str, :any:`tvm.target.Target`, or dict of str(i.e. device/context name) to str/tvm.target.Target, optional
        对于异构编译，它是一个指示context到target映射的字典。 对于同构编译，它是一个编译target。 

    target_host : str or :any:`tvm.target.Target`, optional
        主机编译target，如果target是device。 当 TVM 编译 CUDA 等device特定程序时，我们还需要主机（CPU）端代码与驱动程序交互，正确设置维度和参数。target_host 用于指定主机端代码生成target。 默认情况下，如果启用 llvm，则使用 llvm，否则使用 stackvm 解释器。 

    params : dict of str to NDArray
        在推理阶段不会更改的Graph的权重参数，用于常量折叠。 

    mod_name: Optional[str]
        The module name we will build

    Returns
    -------
    graph_json : str
        The json string that can be accepted by graph executor.

    mod : tvm.Module
        The module containing necessary libraries.

    params : dict
        The parameters of the final graph.
    """
    # pylint: enable=line-too-long
    # fmt: on
    if not isinstance(ir_mod, (IRModule, _function.Function)):
        raise ValueError("Type of input parameter mod must be tvm.IRModule")

    if isinstance(ir_mod, _function.Function):
        if params:
            ir_mod = bind_params_by_name(ir_mod, params)
        ir_mod = IRModule.from_expr(ir_mod)
        warnings.warn(
            "Please use input parameter mod (tvm.IRModule) "
            "instead of deprecated parameter mod (tvm.relay.function.Function)",
            DeprecationWarning,
        )
    target = _update_target(target)
    if isinstance(target_host, (str, Target)):
        target_host = Target(target_host)
    elif target_host:
        raise ValueError("target host must be the type of str, " + "tvm.target.Target, or None")

    target, target_host = Target.check_and_update_host_consist(
        target, target_host, target_is_dict_key=False
    )

    # If current dispatch context is fallback context (the default root context),
    # then load pre-tuned parameters from TopHub
    if isinstance(autotvm.DispatchContext.current, autotvm.FallbackContext):
        tophub_context = autotvm.tophub.context(list(target.values()))
    else:
        tophub_context = autotvm.utils.EmptyContext()

    with tophub_context:
        bld_mod = BuildModule()
        graph_json, runtime_mod, params = bld_mod.build(mod=ir_mod, target=target, params=params)
        executor_factory = _graph_executor_factory.GraphExecutorFactoryModule(
            ir_mod, target, graph_json, runtime_mod, mod_name, params
        )
        return executor_factory
```

在上面的函数中，首先将`relay.Function`和`params`组织成一个IRModule待用，并且再次检查和更新目标设备类型`target`和`target`对应的`host`端类型。接下来，Relay会寻找是否有AutoTVM预先Fintune的记录，如果没有那么就使用`autotvm.FallbackContext`这个环境上下文信息，如果有那么接下来的所有操作都在tophub_context 的 scope 之下(`with tophub_context:`)。值得一提的是 Relay考虑了异构情景下的代码生成，用户可以指定多个生成代码的目标(target)。

在`with tophub_context: `中，创建了一个`BuildModule `对象`bld_mod `，然后调用了`bld_mod `对象的`build`函数生成一个硬件可以执行的更底层的IR，以及包含各种必需运行时库的`tvm.Module`和优化后的计算图的参数。这里还有一个`_graph_executor_factory.GraphExecutorFactoryModule`函数，它的功能就是将上面的IR，运行时库以及参数打包成一个`tvm.Module`，这样用户只需要把这个`tvm.Module`存下来，下次就可以省去编译过程直接在硬件上执行了。

继续深入代码，我们现在知道TVM编译Relay IR的核心实现应该就是`BuildModule`类中的`build`函数了，我们接着分析：

```python
class BuildModule(object):
    """Build an IR module to run on TVM graph executor. This class is used
    to expose the `RelayBuildModule` APIs implemented in C++.
    """

    def __init__(self):
        self.mod = _build_module._BuildModule()
        self._get_graph_json = self.mod["get_graph_json"]
        self._get_module = self.mod["get_module"]
        self._build = self.mod["build"]
        self._optimize = self.mod["optimize"]
        self._set_params_func = self.mod["set_params"]
        self._get_params_func = self.mod["get_params"]

    def build(self, mod, target=None, target_host=None, params=None):
        """
        Parameters
        ----------
        mod : :py:class:`~tvm.IRModule`
            The IRModule to build.

        target : str, :any:`tvm.target.Target`, or dict of str(i.e.
        device/context name) to str/tvm.target.Target, optional
            For heterogeneous compilation, it is a dictionary indicating context
            to target mapping. For homogeneous compilation, it is a build target.

        target_host : str or :any:`tvm.target.Target`, optional
            Host compilation target, if target is device.
            When TVM compiles device specific program such as CUDA,
            we also need host(CPU) side code to interact with the driver
            to setup the dimensions and parameters correctly.
            target_host is used to specify the host side codegen target.
            By default, llvm is used if it is enabled,
            otherwise a stackvm intepreter is used.

        params : dict of str to NDArray
            Input parameters to the graph that do not change
            during inference time. Used for constant folding.

        Returns
        -------
        factory_module : tvm.relay.backend.graph_executor_factory.GraphExecutorFactoryModule
            The runtime factory for the TVM graph executor.
        """
        target = _update_target(target)
        target, target_host = Target.check_and_update_host_consist(
            target, target_host, target_is_dict_key=False
        )

        # Setup the params.
        if params:
            self._set_params(params)

        # Build the IR module. If auto_scheduler is not enabled,
        # then use the TOPI-defined schedule.
        use_auto_scheduler = PassContext.current().config.get(
            "relay.backend.use_auto_scheduler", False
        )

        # Turn off AutoTVM config not found warnings if auto_scheduler is enabled.
        old_autotvm_silent = autotvm.GLOBAL_SCOPE.silent
        autotvm.GLOBAL_SCOPE.silent = use_auto_scheduler

        self._build(mod, target, target_host)
        autotvm.GLOBAL_SCOPE.silent = old_autotvm_silent

        # Get artifacts
        graph_json = self.get_json()
        mod = self.get_module()
        params = self.get_params()

        return graph_json, mod, params
```

首先在`__init__`函数中，通过`self._build = self.mod["build"]`这行代码可以获取对应的C++函数，这是怎么做到的呢？首先看，`self.mod = _build_module._BuildModule()`这里的`_BuildModule()`是C++中注册到环境中的一个函数，实现在：`tvm/src/relay/backend/build_module.cc`。这里实现了一个`RelayBuildModule `类，这个类中有一个`GetFunction`函数，这个函数会通过名字查询要使用的函数，打包成PackedFunc返回，这个函数和上面`__init__`中的`self.mod[“build”]`等建立了映射关系。

```cpp
class RelayBuildModule : public runtime::ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_graph_json") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetGraphJSON(); });
    } else if (name == "get_module") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetModule(); });
    } else if (name == "build") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 3);
        this->Build(args[0], args[1], args[2]);
      });
    } else if (name == "list_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->ListParamNames(); });
    } else if (name == "get_params") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetParams(); });
    } else if (name == "set_params") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        Map<String, Constant> params = args[0];
        for (const auto& kv : params) {
          this->SetParam(kv.first, kv.second->data);
        }
      });
    } else if (name == "get_irmodule") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->graph_codegen_->GetIRModule();
      });
    } else if (name == "get_external_modules") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        *rv = this->graph_codegen_->GetExternalModules();
      });
    } else if (name == "optimize") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        ICHECK_EQ(args.num_args, 2);
        *rv = this->Optimize(args[0], args[1], this->params_);
      });
    } else {
      LOG(FATAL) << "Unknown packed function: " << name;
      return PackedFunc([sptr_to_self, name](TVMArgs args, TVMRetValue* rv) {});
    }
  }
```

PackedFunc是TVM中提供的Python的一个接口，任何函数都可以封装成PackedFunc，并给Python调用，当然Python的函数也可以伪装成PackedFunc提供给C++调用。对这个感兴趣的同学可以看一下这篇博客：`https://hjchen2.github.io/2020/01/10/TVM-PackedFunc%E5%AE%9E%E7%8E%B0%E6%9C%BA%E5%88%B6/` 。

继续跟进代码我们发现上面的`self.mod["build"]`在C++实现中主要就运行了`BuildRelay`这个函数：

```cpp
void BuildRelay(IRModule relay_module,
                  const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
    Target target_host = GetTargetHost();
    // If no target_host has been set, we choose a default one, which is
    // llvm if "codegen.LLVMModuleCreate" is accessible.
    const runtime::PackedFunc* pf = runtime::Registry::Get("codegen.LLVMModuleCreate");
    if (!target_host.defined()) target_host = (pf != nullptr) ? Target("llvm") : Target("stackvm");

    // Update all the targets in the targets_ TargetsMap
    CheckAndUpdateHostConsistency(&targets_, &target_host);

    // Relay IRModule -> IRModule optimizations.
    relay_module = Optimize(relay_module, targets_, params);
    // Get the updated function.
    auto func = Downcast<Function>(relay_module->Lookup("main"));

    // Generate code for the updated function.
    graph_codegen_ = std::unique_ptr<GraphCodegen>(new GraphCodegen());
    graph_codegen_->Init(nullptr, targets_);
    graph_codegen_->Codegen(func);

    ret_.graph_json = graph_codegen_->GetJSON();
    ret_.params = graph_codegen_->GetParams();

    auto lowered_funcs = graph_codegen_->GetIRModule();

    // Generate a placeholder function that attaches linked params as its arguments.
    if (target_host->GetAttr<Bool>("link-params").value_or(Bool(false))) {
      CHECK(pf != nullptr) << "Unable to link-params with no target_host and no llvm codegen.";
      auto param_ids = graph_codegen_->GetParamIds();
      auto link_params = Map<String, tir::LinkedParam>();
      for (auto param : ret_.params) {
        link_params.Set(param.first, tir::LinkedParam(param_ids[param.first], param.second));
      }

      Map<String, ObjectRef> dict;
      dict.Set(tvm::tir::attr::kLinkedParams, link_params);
      dict.Set(tvm::attr::kGlobalSymbol, String(::tvm::runtime::symbol::tvm_lookup_linked_param));
      DictAttrs attrs{dict};
      auto prim = tir::PrimFunc(Array<tir::Var>(), tir::SeqStmt(Array<tir::Stmt>()), VoidType(),
                                Map<tir::Var, tir::Buffer>(), attrs);
      if (lowered_funcs.find(target_host->str()) == lowered_funcs.end()) {
        lowered_funcs.Set(target_host->str(), IRModule(Map<GlobalVar, BaseFunc>({})));
      }
      lowered_funcs[target_host->str()]->Add(
          GlobalVar(::tvm::runtime::symbol::tvm_lookup_linked_param), prim);
    }

    // When there is no lowered_funcs due to reasons such as optimization.
    if (lowered_funcs.size() == 0) {
      if (target_host.defined() && target_host->kind->name == "llvm") {
        // If we can decide the target is LLVM, we then create an empty LLVM module.
        ret_.mod = (*pf)(target_host->str(), "empty_module");
      } else {
        // If we cannot decide the target is LLVM, we create an empty CSourceModule.
        // The code content is initialized with ";" to prevent complaining
        // from CSourceModuleNode::SaveToFile.
        ret_.mod = tvm::codegen::CSourceModuleCreate(";", "", Array<String>{});
      }
    } else {
      ret_.mod = tvm::build(lowered_funcs, target_host_);
    }

    auto ext_mods = graph_codegen_->GetExternalModules();
    ret_.mod = tvm::codegen::CreateMetadataModule(ret_.params, ret_.mod, ext_mods, GetTargetHost());
  }
```

在这个函数是编译流程的主要代码，可以看到它包含了Optimize，Codegen两个过程。而Optimize就是我们上一节讲过的Pass了，Codegen主要实现了内存分配以及指定设备上的代码生成。这里面还有很多细节，但本篇文章只是讲编译流程，所以Codegen相关细节不在这里继续展开。


# 三，总结
这篇文章跟进源码介绍了一下TVM的编译流程，可以看到TVM通过Relay IR来对接深度学习框架的模型并通过编译流程将Relay IR编译成了硬件可以执行的IR，再将这个底层IR和运行时库以及模型参数打包为一个`tvm.Module`返回。关于为什么要将底层IR和运行时库以及模型参数打包，根据官方文档可以知道这样是为了可以方便的保存TVM的计算图和运行时库，可以做到一次编译，可持久化推理。

# 四，同系列文章
- [【从零开始学深度学习编译器】五，TVM Relay以及Pass简介](https://mp.weixin.qq.com/s/5JAWE9RTTXwDJR5HqlsCzA)
- [【从零开始学深度学习编译器】番外一，Data Flow和Control Flow](https://mp.weixin.qq.com/s/Kt4xDLo-NRui8Whl0DqcSA)
- [【从零开始学TVM】三，基于ONNX模型结构了解TVM的前端](https://mp.weixin.qq.com/s/KFxd3zf76EP3DFcCAPZjvQ)
- [【从零开始学深度学习编译器】二，TVM中的scheduler](https://mp.weixin.qq.com/s/fPpqKL3uaaJ5QlNS79DZ5Q)
- [【从零开始学深度学习编译器】一，深度学习编译器及TVM 介绍](https://mp.weixin.qq.com/s/sZLWjYebbHjCgQ6XAZCiOw)

# 五，参考
- https://zhuanlan.zhihu.com/p/91283238
- https://zhuanlan.zhihu.com/p/338550499

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)