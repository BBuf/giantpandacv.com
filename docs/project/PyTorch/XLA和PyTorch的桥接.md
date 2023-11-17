# XLA和PyTorch的链接

## 前言

[XLA (Accelerated Linear Algebra)](https://github.com/openxla/xla)是一个开源的机器学习编译器，对PyTorch、Tensorflow、JAX等多个深度学习框架都有支持。最初XLA实际上是跟Tensorflow深度结合的，很好地服务了Tensorflow和TPU，而与XLA的结合主要依赖于社区的支持，即[torch-xla](https://github.com/pytorch/xla)。

torch-xla在支持XLA编译的基础上，较大限度地保持了PyTorch的易用性，贴一个官方的DDP训练的例子：

```py
 import torch.distributed as dist
-import torch.multiprocessing as mp
+import torch_xla.core.xla_model as xm
+import torch_xla.distributed.parallel_loader as pl
+import torch_xla.distributed.xla_multiprocessing as xmp
+import torch_xla.distributed.xla_backend

 def _mp_fn(rank, world_size):
   ...

-  os.environ['MASTER_ADDR'] = 'localhost'
-  os.environ['MASTER_PORT'] = '12355'
-  dist.init_process_group("gloo", rank=rank, world_size=world_size)
+  # Rank and world size are inferred from the XLA device runtime
+  dist.init_process_group("xla", init_method='xla://')
+
+  model.to(xm.xla_device())
+  # `gradient_as_bucket_view=True` required for XLA
+  ddp_model = DDP(model, gradient_as_bucket_view=True)

-  model = model.to(rank)
-  ddp_model = DDP(model, device_ids=[rank])
+  xla_train_loader = pl.MpDeviceLoader(train_loader, xm.xla_device())

-  for inputs, labels in train_loader:
+  for inputs, labels in xla_train_loader:
     optimizer.zero_grad()
     outputs = ddp_model(inputs)
     loss = loss_fn(outputs, labels)
     loss.backward()
     optimizer.step()

 if __name__ == '__main__':
-  mp.spawn(_mp_fn, args=(), nprocs=world_size)
+  xmp.spawn(_mp_fn, args=())
```

将一段PyTorch代码改写为torch-xla代码，主要就是三个方面：

- 将模型和数据放到xla device上
- 适当的时候调用`xm.mark_step`
- 某些组件该用pytorchx-xla提供的，比如amp和spawn

其中第二条并没有在上面的代码中体现，原因是为了让用户少改代码，torch-xla将mark_step封装到了dataloader中，实际上不考虑DDP的完整训练的过程可以简写如下：

```py
device = xm.xla_device()
model = model.to(device)
for data, label in enumerate(dataloader):
    data, label = data.to(device), label.to(device)
    output = model(data)
    loss = func(output, label)
    loss.backward()
    optimizer.step()
    xm.mark_step()
```

`xm.mark_step`的作用就是"告诉"框架：现在对图的定义告一段落了，可以编译并执行计算了。既然如此，那么mark_step之前的内容是做了什么呢？因为要在mark_step之后才编译并计算，那么前面肯定不能执行实际的运算。这就引出了Trace和LazyTensor的概念。

其实到了这里，如果对tensorflow或者torch.fx等比较熟悉，就已经很容易理解了，在mark_step之前，torch-xla将torch Tensor换成了LazyTensor，进而将原本是PyTorch中eager computation的过程替换成了trace的过程，最后生成一张计算图来优化和执行。简而言之这个过程是PyTorch Tensor -> XLATensor -> HLO IR，其中HLO就是XLA所使用的IR。在每次调用到torch op的时候，会调用一次`GetIrValue`，这时候就意味着一个节点被写入了图中。更具体的信息可以参考[XLA Tensor Deep Dive](https://github.com/pytorch/xla/blob/46643801a5da36106c346d0434387c29d3a7818d/API_GUIDE.md#xla-tensor-deep-dive)这部分文档。需要注意的是，trace这个过程是独立于mark_step的，即便你的每个循环都不写mark_step，这个循环也可以一直持续下去，只不过在这种情况下，永远都不会发生图的编译和执行，除非在某一步trace的时候，发现图的大小已经超出了pytorch-xla允许的上限。


## PyTorch与torch-xla的桥接

知晓了Trace过程之后，就会好奇一个问题：当用户执行一个PyTorch函数调用的时候，torch-xla怎么将这个函数记录下来的？

最容易想到的答案是“torch-xla作为PyTorch的一个编译选项，打开的时候就会使得二者建立起映射关系”，但很可惜，这个答案是错误的，仔细看PyTorch的CMake文件以及torch-xla的编译方式就会明白，torch-xla是几乎单向依赖于PyTorch的（为什么不是全部后面会讲）。既然PyTorch本身在编译期间并不知道torch-xla的存在，那么当用户使用一个xla device上的Tensor作为一个torch function的输入的时候，又经历了怎样一个过程调用到pytorch-xla中的东西呢？

### 从XLATensor开始的溯源

尽管我们现在并不知道怎么调用到torch-xla中的，但我们知道PyTorch Tensor一定要转换成XLATensor（参考[tensor.h](https://github.com/pytorch/xla/blob/46643801a5da36106c346d0434387c29d3a7818d/torch_xla/csrc/tensor.h)），那么我们只需要在关键的转换之处打印出调用堆栈，自然就可以找到调用方，这样虽然不能保证找到PyTorch中的位置，但是能够找到torch-xla中最上层的调用。注意到XLATensor只有下面这一个创建函数接受`at::Tensor`作为输入，因此就在这里面打印调用栈。

```cpp
XLATensor XLATensor::Create(const at::Tensor& tensor, const Device& device)
```

测试的用例很简单，我们让两个xla device上的Tensor相乘：

```
import torch_xla.core.xla_model as xm
import torch

device = xm.xla_device()
a = torch.normal(0, 1, (2, 3)).to(device)
b = torch.normal(0, 1, (2, 3)).to(device)

c = a * b
```

在上述位置插入堆栈打印代码并重新编译、安装后运行用例，可以看到以下输出（截取部分）：

```
usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(_ZN9torch_xla15TensorToXlaDataERKN2at6TensorERKNS_6DeviceEb+0x64d) [0x7f086098b9ed]
/usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(_ZNK9torch_xla9XLATensor19GetIrValueForTensorERKN2at6TensorERKNS_6DeviceE+0xa5) [0x7f0860853955]
/usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(_ZNK9torch_xla9XLATensor10GetIrValueEv+0x19b) [0x7f0860853d5b]
/usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(_ZN9torch_xla9XLATensor3mulERKS0_S2_N3c108optionalINS3_10ScalarTypeEEE+0x3f) [0x7f086087631f]
/usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(_ZN9torch_xla18XLANativeFunctions3mulERKN2at6TensorES4_+0xc4) [0x7f08606d4da4]
/usr/local/lib/python3.8/dist-packages/_XLAC.cpython-38-x86_64-linux-gnu.so(+0x19d158) [0x7f08605f7158]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so(_ZN2at4_ops10mul_Tensor10redispatchEN3c1014DispatchKeySetERKNS_6TensorES6_+0xc5) [0x7f0945c9d055]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so(+0x2b8986c) [0x7f094705986c]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so(+0x2b8a37b) [0x7f094705a37b]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_cpu.so(_ZN2at4_ops10mul_Tensor4callERKNS_6TensorES4_+0x157) [0x7f0945cee717]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_python.so(+0x3ee91f) [0x7f094e4b391f]
/usr/local/lib/python3.8/dist-packages/torch/lib/libtorch_python.so(+0x3eeafb) [0x7f094e4b3afb]
python() [0x5042f9]
```
明显可以看到是从python的堆栈调用过来的，分析一下可以得知`_ZN2at4_ops10mul_Tensor10redispatchEN3c1014DispatchKeySetERKNS_6TensorES6_+0xc5`对应的定义是`at::_ops::mul_Tensor::redispatch(c10::DispatchKeySet, at::Tensor const&, at::Tensor const&)+0xc5`

虽然这里意义仍有些不明，但我们已经可以做出推测了：redistpatch函数是根据DispatchKeySet来决定将操作dispatch到某个backend上，xla的device信息就被包含在其中。而后面两个输入的`const at::Tensor&`就是乘法操作的两个输入。

根据上面的关键字redispatch来寻找，我们可以找到这样一个文件[gen.py](https://github.com/pytorch/pytorch/blob/6e73ae20225e0794eed0079fcd43cf72524d2a31/torchgen/gen.py)，其中的codegen函数很多，但最显眼的是下面的OperatorGen：

```py
@dataclass(frozen=True)
class ComputeOperators:
    target: Union[
        Literal[Target.DECLARATION],
        Literal[Target.DEFINITION]
    ]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()
        call_method_name = 'call'
        redispatch_method_name = 'redispatch'

        if self.target is Target.DECLARATION:
            return f"""
struct TORCH_API {name} {{
  using schema = {sig.type()};
  using ptr_schema = schema*;
  // See Note [static constexpr char* members for windows NVCC]
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})
  static {sig.defn(name=call_method_name, is_redispatching_fn=False)};
  static {sig.defn(name=redispatch_method_name, is_redispatching_fn=True)};
}};"""
        elif self.target is Target.DEFINITION:
            defns = f"""
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")
STATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})

// aten::{f.func}
static C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow({name}::name, {name}::overload_name)
      .typed<{name}::schema>();
}}
"""

            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ', '.join(['dispatchKeySet'] + [a.name for a in sig.arguments()])
                    dispatcher_call = 'redispatch'
                    method_name = f'{name}::{redispatch_method_name}'
                else:
                    dispatcher_exprs_str = ', '.join([a.name for a in sig.arguments()])
                    dispatcher_call = 'call'
                    method_name = f'{name}::{call_method_name}'

                defns += f"""
// aten::{f.func}
{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{
    static auto op = create_{name}_typed_handle();
    return op.{dispatcher_call}({dispatcher_exprs_str});
}}
"""
            return defns
        else:
            assert_never(self.target)
```

对于每个算子，PyTorch会（在编译前）在这里生成许多类，这些类会有静态成员`call`或者`redispatch`，其中redispatch负责分发具体的实现。这里的codegen比较繁琐，这里就不再细讲。

### 注册PyTorch库实现

即便我们找到了上面redispatch和codegen的线索，看起来仍然不足以解释PyTorch到torch-xla的桥接，因为PyTorch和torch-xla两个库之间的调用，必须要有符号的映射才可以，而不是一些函数形式上的相同。PyTorch中是有Dispatcher机制的，这个机制很常见于很多框架，比如oneflow也是有一套类似的Dispatcher机制。这套机制最大的好处就是在尽可能减少侵入式修改的前提下保证了较高的可扩展性。简而言之，我们的op有一种定义，但可以有多种实现方式，并且这个实现的代码可以不在框架内部，这样就使得框架在保持通用性的同时，易于在特定环境下做针对性的扩展。这套机制本质上就是建立了一个字典，将op映射到函数指针，那么每次调用一个op的时候，我们可以根据一些标识（比如tensor.device）来判断应该调用哪一种实现。

PyTorch中提供了一个宏用来将实现注册，从而让dispatcher可以调用：

```cpp
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)                             \
  static void C10_CONCATENATE(                                         \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&);    \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE(        \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)(              \
      torch::Library::IMPL,                                            \
      c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check( \
          c10::DispatchKey::k)>(                                       \
          []() {                                                       \
            return &C10_CONCATENATE(                                   \
                TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid);           \
          },                                                           \
          []() { return [](torch::Library&) -> void {}; }),            \
      #ns,                                                             \
      c10::make_optional(c10::DispatchKey::k),                         \
      __FILE__,                                                        \
      __LINE__);                                                       \
  void C10_CONCATENATE(                                                \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)
```

这个宏如果完全展开会是下面这样：

```cpp
static void TORCH_LIBRARY_IMPL_init_aten_CPU_0(torch::Library&);
static const torch::detail::TorchLibraryInit TORCH_LIBRARY_IMPL_static_init_aten_CPU_0(
      torch::Library::IMPL,
      (c10::impl::dispatch_key_allowlist_check(c10::DispatchKey::CPU)
           ? &TORCH_LIBRARY_IMPL_init_aten_CPU_0
           : [](torch::Library&) -> void {}),
      "aten",
      c10::make_optional(c10::DispatchKey::CPU),
      __FILE__,
      __LINE__);
void TORCH_LIBRARY_IMPL_init_aten_CPU_0(torch::Library & m)
```

这里比较需要注意的是第二行的`TORCH_LIBRARY_IMPL_static_init_aten_CPU_0`并不是一个函数，而是一个静态变量，它的作用就是在torch_xla库初始化的时候，将xla定义的op注册到PyTorch中。


关于这部分更详细的介绍可以参考[https://zhuanlan.zhihu.com/p/648578629](https://zhuanlan.zhihu.com/p/648578629)。


### 从PyTorch调用到torch_xla

xla调用上面所说的宏进行注册的位置在`RegisterXLA.cpp`这个文件中（codegen的结果），如下：

```cpp
ORCH_LIBRARY_IMPL(aten, XLA, m) {
  m.impl("abs",
  TORCH_FN(wrapper__abs));

  ...
}
```

其中，wrapper__abs的定义如下：

```cpp
at::Tensor wrapper__abs(const at::Tensor & self) {
  return torch_xla::XLANativeFunctions::abs(self);
}
```

显然，这个定义和PyTorch框架内部的算子是完全一致的，只是修改了实现。而`XLANativeFunctions::abs`的实现可以在`aten_xla_type.cpp`中找到，如下所示：

```cpp
at::Tensor XLANativeFunctions::abs(const at::Tensor& self) {
  XLA_FN_COUNTER("xla::");
  return bridge::AtenFromXlaTensor(XLATensor::abs(bridge::GetXlaTensor(self)));
}
```

到这里已经比较明朗了，注册之后，PyTorch上对于op的调用最终会进入torch_xla的native function中调用对应的op实现，而这些实现的根本都是对XLATensor进行操作，在最终操作执行完成之后，会将作为结果的XLATensor重新转换为torch Tensor，但要注意，这里的结果不一定被实际计算了，也可能只是记录了一下IR，将节点加入图中，这取决于具体的实现。

## 总结

其实torch-xla官方的文档里是有关于代码生成和算子注册这个过程的描述的，只不过一开始我没找到这个文档，走了一点弯路，但是自己探索也会觉得更明了这个过程。官方文档中的描述如下（节选）：

All file mentioned below lives under the xla/torch_xla/csrc folder, with the exception of codegen/xla_native_functions.yaml

1. xla_native_functions.yaml contains the list of all operators that are lowered. Each operator name must directly match a pytorch operator listed in native_functions.yaml. This file serves as the interface to adding new xla operators, and is an input to PyTorch's codegen machinery. It generates the below 3 files: XLANativeFunctions.h, RegisterXLA.cpp, and RegisterAutogradXLA.cpp
2. XLANativeFunctions.h and aten_xla_type.cpp are entry points of PyTorch to the pytorch_xla world, and contain the manually written lowerings to XLA for each operator. XLANativeFunctions.h is auto-generated through a combination of xla_native_functions.yaml and the PyTorch core native_functions.yaml file, and contains declarations for kernels that need to be defined in aten_xla_type.cpp. The kernels written here need to construct 'XLATensor' using the input at::Tensor and other parameters. The resulting XLATensor needs to be converted back to the at::Tensor before returning to the PyTorch world.
3. RegisterXLA.cpp and RegisterAutogradXLA.cpp are auto-generated files that register all lowerings to the PyTorch Dispatcher. They also include auto-generated wrapper implementations of out= and inplace operators.

大概意思就是实际上torch-xla就是根据`xla_native_functions.yaml`这个文件来生成算子的定义，然后再生成对应的`RegisterXLA.cpp`中的注册代码，这也跟PyTorch的codegen方式一致。

综合这一整个过程可以看出，PyTorch是保持了高度的可扩展性的，不需要多少侵入式的修改就可以将所有的算子全部替换成自己的，这样的方式也可以让开发者不用去关注dispatcher及其上层的实现，专注于算子本身的逻辑。

## 参考资料

- [XLA (Accelerated Linear Algebra)](https://github.com/openxla/xla)
- [https://zhuanlan.zhihu.com/p/648578629](https://zhuanlan.zhihu.com/p/648578629)
- [pytorch repo](https://github.com/pytorch/pytorch)
- [XLA Tensor Deep Dive](https://github.com/pytorch/xla/blob/46643801a5da36106c346d0434387c29d3a7818d/API_GUIDE.md#xla-tensor-deep-dive)
