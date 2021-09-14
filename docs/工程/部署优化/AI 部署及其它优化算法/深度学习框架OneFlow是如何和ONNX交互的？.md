> oneflow-onnx 工具开发者：daquexian， bbuf
# 0x0. 介绍
在开始阅读本篇文章之前，如果你对ONNX不是很了解介意先阅读我之前写的这几篇介绍ONNX文章：

- [ONNX初探](https://mp.weixin.qq.com/s/FWEmk3PmaMhIG0GwaAOcjA)
- [ONNX 再探](https://mp.weixin.qq.com/s/_iNhfZNR5-swXLhHKjYRkQ)
- [onnx2pytorch和onnx-simplifier新版介绍](https://mp.weixin.qq.com/s/NDv-quXeBrPeDcCbg97FHA)

以及大老师的：

[onnx simplifier 和 optimizer](https://mp.weixin.qq.com/s/q0Aa2LRpeCPCnIzRJbMmaQ)

然后，这篇文章不会继续探索ONNX本身的东西，而是聊聊另外一个有趣的话题，即深度学习框架是如何和ONNX进行交互的？我最近配合大老师基于OneFlow深度学习框架做了一些和ONNX有关的工作，感觉自己对OneFlow和ONNX的交互过程也算熟悉一些了。因此，在这篇文章我将分享OneFlow和ONNX交互的具体实现思路以及介绍oneflow-onnx这个开源工具的一些特性。让读者了解OneFlow的模型是如何转换为ONNX模型，以及ONNX模型是如何转回OneFlow的模型（X2OneFlow）的。个人认为OneFlow目前和ONNX交互的做法是比较优雅并且具有较好扩展性的，因此我们将这项工作转换成了开源成果并分享实现思路，github地址为：`https://github.com/Oneflow-Inc/oneflow_convert_tools `。这个工具作为OneFlow 生态系统的一部分会被我们持续维护，同时这个工具也被我们制作成了一个wheel包，感兴趣的用户只需要pip安装oneflow-onnx即可快速体验。在下面的第二节以及工程的README也有详细的安装步骤。

oneflow-onnx工具包含两个功能，一个是将OneFlow导出ONNX，另外一个是将各个训练框架导出的ONNX模型转换为OneFlow的模型。本工程已经适配了TensorFlow/Pytorch/PaddlePaddle框架的预训练模型通过导出ONNX转换为OneFlow（我们将这一功能叫作X2OneFlow）。更多使用示例以及相关文档和源码均可以在开源`https://github.com/Oneflow-Inc/oneflow_convert_tools`工程中获得。

# 0x1. 算子支持和模型支持

## OneFlow2ONNX
#### OneFlow2ONNX 支持的OP列表

> 目前OneFlow2ONNX 支持60+的ONNX OP，我们在下面的列表中列出了目前OneFlow支持导出的全部ONNX OP


| 序号 | OP         | 序号 | OP             | 序号 | OP          | 序号 | OP                 |
| ---- | ---------- | ---- | -------------- | ---- | ----------- | ---- | ------------------ |
| 1    | GatherND   | 2    | Transpose      | 3    | Add         | 4    | Sub                |
| 5    | Mul        | 6    | Div            | 7    | Sum         | 8    | LeakyRelu          |
| 9    | Softplus   | 10   | Softplus       | 11   | Abs         | 12   | Ceil               |
| 13   | Elu        | 14   | Exp            | 15   | Floor       | 16   | Log                |
| 17   | Neg        | 18   | Sigmoid        | 19   | Sqrt        | 20   | Tanh               |
| 21   | Reciprocal | 22   | Relu           | 23   | Acos        | 24   | Asin               |
| 25   | Atan       | 26   | Cos            | 27   | Sin         | 28   | Tan                |
| 29   | Acosh      | 30   | Asinh          | 31   | Atanh       | 32   | Cosh               |
| 33   | Sinh       | 34   | Min            | 35   | Max         | 36   | Clip               |
| 37   | Softmax    | 38   | Sign           | 39   | MatMul      | 40   | Erf                |
| 41   | FloorMod   | 42   | Round          | 43   | Not         | 44   | And                |
| 45   | Or         | 46   | Equal          | 47   | NotEqual    | 48   | Greater            |
| 49   | Less       | 50   | Pad            | 51   | AveragePool | 52   | MaxPool            |
| 53   | Conv       | 54   | QuantizeLinear | 56   | ReduceMin   | 57   | BatchNormalization |
| 58   | ReduceSum  | 59   | ReduceProd     | 60   | ArgMax      | 61   | ArgMin             |
| 62   | Reshape    | 63   | Squeeze        | 64   | Transpose   | 65   | Concat             |
| 66   | Cast       | 67   | Identity       | 68   | Mul         |      |                    |

#### OneFlow2ONNX 模型测试库

> 目前OneFlow2ONNX 支持60+的ONNX OP，我们在下面的模型列表中测试了OneFlow2ONNX的转换。


| 模型        | 来源                                                         | operator version |
| ----------- | ------------------------------------------------------------ | ---------------- |
| AlexNet     | [OneFlow-AlexNet](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | 10               |
| MobileNetV2 | [Oneflow-MobileNetV2](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | 10               |
| ResNet50    | [OneFlow-ResNet50](https://github.com/Oneflow-Inc/OneFlow-Benchmark/tree/master/Classification/cnns) | 10               |


## X2OneFlow

### X2OneFlow 支持的OP列表

> 目前X2OneFlow 支持40+的ONNX OP，30+的Tensorflow/Pytorch/PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。OP的单元测试代码会逐渐移步到工程的examples目录下，并支持更多的OP。



#### ONNX

| 序号 | OP          | 序号 | OP                 | 序号 | OP                | 序号 | OP              |
| ---- | ----------- | ---- | ------------------ | ---- | ----------------- | ---- | --------------- |
| 1    | Conv        | 2    | BatchNormalization | 3    | MaxPool           | 4    | AveragePool     |
| 5    | Concat      | 6    | ReLU               | 7    | AdaptiveMaxPool   | 8    | Softmax         |
| 9    | Unsqueeze   | 10   | Transpose          | 11   | Clip              | 12   | Gather          |
| 13   | Slice       | 14   | Split              | 15   | Flatten           | 16   | Add             |
| 17   | Sub         | 18   | Mul                | 19   | Div               | 20   | Sqrt            |
| 21   | Pow         | 22   | Tanh               | 23   | Sigmoid           | 24   | Cast            |
| 25   | Pad         | 26   | ReduceMean         | 27   | Reshape           | 28   | AdaptiveAvgPool |
| 29   | Squeeze     | 30   | Expand             | 31   | Gather            | 32   | Slice           |
| 33   | Split       | 34   | Min                | 35   | Max               | 36   | Constant        |
| 37   | HardSigmoid | 38   | Gemm               | 39   | MatMul            | 40   | Erf             |
| 41   | Cast        | 42   | GlobalMaxPool      | 43   | GlobalAveragePool | 44   | ReduceMax       |
| 45   | Identity    |      |                    |      |                   |      |                 |
#### TensorFlow

| 序号 | OP            | 序号 | OP              | 序号 | OP          | 序号 | OP            |
| ---- | ------------- | ---- | --------------- | ---- | ----------- | ---- | ------------- |
| 1    | relu          | 2    | concatenate     | 3    | expand_dims | 4    | transpose     |
| 5    | batchnorm     | 6    | slice           | 7    | gather      | 8    | clip_by_value |
| 9    | conv2d        | 10   | depthwiseconv2d | 11   | flatten     | 12   | add           |
| 13   | sub           | 14   | mul             | 15   | div         | 16   | pow           |
| 17   | sqrt          | 18   | tanh            | 19   | sigmoid     | 20   | erf           |
| 21   | cast          | 22   | pad             | 23   | maxpool     | 24   | avgpool       |
| 25   | globalavgpool | 26   | globalmaxpool   | 27   | reduce_mean | 28   | reshape       |
| 29   | softmax       | 30   | relu6           |      |             |      |               |

- 分组卷积存在问题，已给TensorFlow2ONNX团队PR。

#### Pytorch

| 序号 | OP            | 序号 | OP              | 序号 | OP               | 序号 | OP        |
| ---- | ------------- | ---- | --------------- | ---- | ---------------- | ---- | --------- |
| 1    | relu          | 2    | cat             | 3    | unsqueeze        | 4    | transpose |
| 5    | batchnorm     | 6    | slice           | 7    | gather           | 8    | clamp     |
| 9    | conv2d        | 10   | depthwiseconv2d | 11   | flatten          | 12   | add       |
| 13   | sub           | 14   | mul             | 15   | div              | 16   | pow       |
| 17   | sqrt          | 18   | tanh            | 19   | sigmoid          | 20   | erf       |
| 21   | cast          | 22   | pad             | 23   | maxpool          | 24   | avgpool   |
| 25   | globalavgpool | 26   | globalmaxpool   | 27   | reduce_mean      | 28   | reshape   |
| 29   | softmax       | 30   | relu6           | 31   | CrossEntropyLoss |      |           |


#### PaddlePaddle

| 序号 | OP              | 序号 | OP                 | 序号 | OP          | 序号 | OP            |
| ---- | --------------- | ---- | ------------------ | ---- | ----------- | ---- | ------------- |
| 1    | relu            | 2    | concatenate        | 3    | expand_dims | 4    | transpose     |
| 5    | batchnorm       | 6    | slice              | 7    | gather      | 8    | clip_by_value |
| 9    | conv2d          | 10   | depthwiseconv2d    | 11   | flatten     | 12   | add           |
| 13   | sub             | 14   | mul                | 15   | div         | 16   | pow           |
| 17   | sqrt            | 18   | tanh               | 19   | sigmoid     | 20   | ~~erf~~       |
| 21   | cast            | 22   | pad                | 23   | maxpool     | 24   | avgpool       |
| 25   | adaptiveavgpool | 26   | ~~adptivemaxpool~~ | 27   | reduce_mean | 28   | reshape       |
| 29   | softmax         | 30   | relu6              |      |             |      |               |

相关issue：

- https://github.com/PaddlePaddle/Paddle2ONNX/issues/221
- https://github.com/PaddlePaddle/Paddle2ONNX/issues/220

### X2OneFlow模型测试库

目前X2OneFlow 支持40+的ONNX OP，30+的Tensorflow/Pytorch/PaddlePaddle OP，覆盖了大部分CV分类模型常用的操作。我们在如下模型列表中测试了X2OneFlow的转换。

#### Pytorch

| 模型         | 是否支持 |
| ------------ | -------- |
| AlexNet      | Yes      |
| VGGNet       | Yes      |
| GoogleNet    | Yes      |
| ResNet       | Yes      |
| ResNext      | Yes      |
| SENet        | Yes      |
| MobileNetV1  | Yes      |
| MobileNetV2  | Yes      |
| MobileNetV3  | Yes      |
| RegNet       | Yes      |
| DenseNet     | Yes      |
| EfficientNet | Yes      |
| InceptionNet | Yes      |
| ShuffleNetV1 | Yes      |
| ShuffleNetV2 | Yes      |
| SqueezeNet   | Yes      |

#### TensorFlow

| 模型         | 是否支持 |
| ------------ | -------- |
| VGGNet       | Yes      |
| ResNet       | Yes      |
| ResNetV2     | Yes      |
| XceptionNet  | Yes      |
| MobileNetV1  | Yes      |
| MobileNetV2  | Yes      |
| MobileNetV3  | Yes      |
| DenseNet     | Yes      |
| EfficientNet | Yes      |
| InceptionNet | Yes      |

#### PaddlePaddle

| 模型               | 是否支持                                                     |
| ------------------ | ------------------------------------------------------------ |
| AlexNet            | Yes                                                          |
| VGGNet             | Yes                                                          |
| GoogleNet          | Yes                                                          |
| ResNet             | Yes                                                          |
| ResNext            | Yes                                                          |
| SE_ResNext         | Yes                                                          |
| SENet              | Yes                                                          |
| MobileNetV1        | Yes                                                          |
| MobileNetV2        | Yes                                                          |
| MobileNetV3        | Yes                                                          |
| RegNet             | Yes                                                          |
| DenseNet           | No（msg: "op_name: Concat_58 already exist in job: job_eval"） |
| EfficientNet       | Yes                                                          |
| InceptionNet       | Yes                                                          |
| ShuffleNetV2       | Yes                                                          |
| SqueezeNet         | Yes                                                          |
| DPNNet             | Yes                                                          |
| DarkNet            | Yes                                                          |
| GhostNet           | Yes                                                          |
| RepVGG             | Yes                                                          |
| XceptionNet        | Yes                                                          |
| Xception_DeepLab   | Yes                                                          |
| Vision_Transformer | No（"op_name: Constant_20 already exist in job: job_eval"）  |
| Res2Net            | No（split op bug，working）                                  |
| Unet               | No（OneFlow的上采样OP和Paddle未对齐）                        |


- 模型的测试代码均可以在工程的examples中找到。


# 0x2. 快速体验
##### 用户环境配置

```sh
python>=3.5
onnx>=1.8.0
onnx-simplifier>=0.3.3
onnxoptimizer>=0.2.5
onnxruntime>=1.6.0
oneflow>=0.3.4
```

如果你想使用X2OneFlow（X代表TensorFlow/Pytorch/PaddlePaddle）需要安装对应的深度学习框架，需要安装对应的深度学习框架，依赖如下：

```sh
pytorch>=1.7.0
paddlepaddle>=2.0.0
tensorflow>=2.0.0
```

#### 安装

##### 安装方式1

```sh
pip install oneflow_onnx
```

#### 安装方式2

```
git clone https://github.com/Oneflow-Inc/oneflow_convert_tools
cd oneflow_onnx
python3 setup.py install
```

使用方法见工程的samples下的示例。

# 0x3. OneFlow-ONNX思路分享
我们将在这一节分享一下OneFlow的模型是如何被转换为ONNX的，这里我们以将OneFlow定义的AlexNet导出ONNX模型为例来分析源码。首先我们`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/examples/oneflow2onnx/test_alexnet.py#L133`进到这里，可以看到下面调用代码：

```python
def test_alexnet():
    @flow.global_function()
    def alexnet_eval_job(x: tp.Numpy.Placeholder((1, 227, 227, 3))):
        return alexnet(x, None, False)

    convert_to_onnx_and_check(alexnet_eval_job, flow_weight_dir=None, onnx_model_path="/tmp")
```

这里通过`flow.global_function()`定义了一个预测用于eval的AlexNet `job`，网络的完整定义可以通过上面的链接访问，可以看到这里通过`convert_to_onnx_and_check`函数将OneFlow定义的AlexNet转换为了ONNX模型，我们跟进这个函数，就来到了这里：`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/oneflow_onnx/oneflow2onnx/util.py#L65-L73`，代码为：

```python
 while not os.path.exists(os.path.join(flow_weight_dir, "snapshot_done")):
     pass
 onnx_model_dir = onnx_model_path
 onnx_model_path = os.path.join(onnx_model_dir, "model.onnx")
 flow.onnx.export(
     job_func,
     flow_weight_dir,
     onnx_model_path,
     opset=opset,
     external_data=external_data,
 )
```

可以看到完成ONNX模型转换的核心函数就是这个`flow.onnx.export`函数，我们继续跳转到这个函数`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/oneflow_onnx/oneflow2onnx/flow2onnx.py#L229-L281`，代码如下：

```python
def Export(
    job_func: Callable,
    model_save_dir: Text,
    onnx_filename: Text,
    continue_on_error: bool = False,
    opset: Optional[int] = None,
    extra_opset: Optional[int] = None,
    shape_override: Optional[Dict[Text, List[int]]] = None,
    external_data: bool = False,
):
    r"""Export a oneflow model into ONNX format.
    Args:
        job_func: OneFlow的作业函数
        model_save_dir: 包含OneFlow定义的模型权重的文件夹. 这个模型权重是用oneflow的check_point.save接口保存的。
        onnx_filename: 输出ONNX模型文件名，字符串类型
        continue_on_error: 如果某个OP无法处理（即没有映射），是否继续
        opset: ONNX Opset版本号，默认为10
        extra_opset: 额外Opset的列表，例如自定义操作使用的Opset
        shape_override: 带有输入信息的字典，覆盖OneFlow给定的输入形状
        external_data: 将权重另存为ONNX外部数据，通常是为了绕过protobuf的2GB文件大小限制。
    """
    assert os.getenv("ENABLE_USER_OP") != "False"
    # 确定模型的路径是存在的
    assert os.path.isdir(model_save_dir)
    # 通过c_api_util.GetJobSet()方法获取当前的所有job
    job_set = c_api_util.GetJobSet()
    # 我们要转的模型被定义在job_func中，所以我们先记录下它的名字
    job_name = job_func.__name__
    # 编译job_set，找到定义模型的job
    for job in job_set.job:
        # TODO(OYY) Modify the interface before modifying it
        if job.job_conf.job_name == job_name:
            # job找到了，可以开始进行下面的步骤，我们在外面详细解释
            onnx_graph = ProcessFlowGraph(
                job,
                model_save_dir,
                continue_on_error=continue_on_error,
                opset=opset,
                extra_opset=extra_opset,
                shape_override=shape_override,
            )
            onnx_graph = optimizer.OptimizeGraph(onnx_graph)
            model_proto = onnx_graph.MakeModel(
                job_name, onnx_filename, external_data=external_data
            )
            with open(onnx_filename, "wb") as f:
                try:
                    f.write(model_proto.SerializeToString())
                except ValueError as e:
                    raise ValueError(
                        "Error occured when running model_proto.SerializeToString(). If the model is larger than 2GB, please specify external_data=True when calling flow.onnx.export. Original error message:\n{}".format(
                            e
                        )
                    )
            return
    raise ValueError('Cannot find job "{}" in jobset'.format(job_name))
```

可以看到这个函数首先编译了OneFlow中的job_set，然后找到了我们最开始定义AlexNet模型的那个job，然后就进入了`ProcessFlowGraph`函数，这个函数主要做了三件事情并最终获得了**初版的合法ONNX模型**（初版的意思是还没有经过优化以及填ONNX节点的权重），我们跟进这个函数，代码如下。

```python
def ProcessFlowGraph(
    flow_graph,
    model_save_dir,
    continue_on_error=False,
    opset=None,
    extra_opset=None,
    shape_override=None,
):
	# 这个函数用来获取导出的ONNX的Opset Version，OneFlow里面最高为10
    opset = util.FindOpset(opset)
    logger.info("Using opset <onnx, %s>", opset)
    # 判断当前的ONNX版本是否支持上面的Opset Version
    if opset > schemas.get_max_supported_opset_version():
        logger.warning(
            "Currently installed onnx package %s is too low to support opset %s, "
            "please upgrade onnx package to avoid potential conversion issue.",
            util.get_onnx_version(),
            opset,
        )

    if shape_override is None:
        shape_override = {}
	# 用于将oneflow 的各个 node 转换为 onnx node 的格式，保持 op 类型、输入输出和属性值不变，这一步产生的还不是合法的 onnx 模型
    (onnx_nodes, op_cnt, attr_cnt, dtypes, output_shapes,) = FlowToOnnxNaive(
        flow_graph, shape_override
    )
	# 构造一个 Graph 类，用于后续方便的修改 onnx 网络结构
    g = Graph(onnx_nodes, model_save_dir, output_shapes, dtypes, opset, extra_opset,)

    # create ops mapping for the desired opsets
    ops_mapping = handler.flow_op.CreateMapping(g.opset, g.extra_opset)

    # some nodes may already copied into inner Graph, so remove them from main Graph.
    TopologicalSort(g, continue_on_error)
	# FlowOnnxMapping 函数调用各个转换函数（通过 @flow_op 注册）逐个转换 op，转换后产生的是合法的 onnx 模型
    mapped_op, unmapped_op, exceptions = FlowOnnxMapping(g, ops_mapping)
    if unmapped_op:
        logger.error("Unsupported ops: %s", unmapped_op)
    if exceptions and not continue_on_error:
        raise exceptions[0]

    # onnx requires topological sorting
    TopologicalSort(g, continue_on_error)

    g.UpdateProto()

    logger.debug(
        "Summay Stats:\n"
        "\toneflow ops: {}\n"
        "\toneflow attr: {}\n"
        "\tonnx mapped: {}\n"
        "\tonnx unmapped: {}".format(op_cnt, attr_cnt, mapped_op, unmapped_op)
    )

    return g
```


`FlowToOnnxNaive `这个函数用于将oneflow 的各个 node 转换为 onnx node 的格式，保持 op 类型、输入输出和属性值不变，最后将转换后的ONNX节点（这个地方这些ONNX节点还不是真正的合法ONNX节点，要后面执行一对一转换之后才是合法的ONNX节点）全部返回。接下来利用这些ONNX节点来构造Graph类，方便后续对ONNX模型进行修改。Graph类的实现在`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/18e041d92654cfc8b03e16c906c451a405c99fd2/oneflow_onnx/onnx_wrapper.py`，这个文件主要是定义了onnx graph和node的wrapper，包含各种修改 onnx 图结构的 api，这里复用了tensorflow-onnx项目的相关代码。注意构造Graph类之后还并没有构造ONNX模型，因为OneFlow的OP还没有一对一的转换为ONNX的OP。

接下来，我们调用`handler.flow_op.CreateMapping(g.opset, g.extra_opset)`这个函数，代码实现如下：

```python
def CreateMapping(max_onnx_opset_version, extra_opsets):
        """Create the final mapping dictionary by stacking domains and opset versions.

        :param max_onnx_opset_version: The highest onnx opset the resulting graph may use.
        :param extra_opsets: Extra opsets the resulting graph may use.
        """
        mapping = {constants.ONNX_DOMAIN: max_onnx_opset_version}
        if extra_opsets:
            for extra_opset in extra_opsets:
                mapping[extra_opset.domain] = extra_opset.version
        ops_mapping = {}
        for domain, opsets in flow_op.get_opsets().items():
            for target_opset, op_map in enumerate(opsets):
                print('='*100)
                print(target_opset)
                print(op_map)
                m = mapping.get(domain)
                if m:
                    if target_opset <= m and op_map:
                        ops_mapping.update(op_map)

        flow_op._MAPPING = ops_mapping
        return ops_mapping
```

这个函数做的事情就是将每个ONNX Opset版本号（也就是for循环中的`domain`）和(OneFlow OP和ONNX OP的mapper，这个mapper是如何获得的请看后文)关联起来并返回，我们打印一下`target_opset`和`op_map`就可以理解了。以AlexNet为例打印如下：


```python
====================================================================================================
0
{}
====================================================================================================
1
{'add_n': (<bound method AddN.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.AddN'>>, 'Sum', {}), 'leaky_relu': (<bound method DirectOpSinceOpset1.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOpSinceOpset1'>>, 'LeakyRelu', {}), 'softplus': (<bound method DirectOpSinceOpset1.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOpSinceOpset1'>>, 'Softplus', {}), 'abs': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Abs', {}), 'ceil': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Ceil', {}), 'elu': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Elu', {}), 'exp': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Exp', {}), 'floor': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Floor', {}), 'log': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Log', {}), 'neg': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Neg', {}), 'sigmoid': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sigmoid', {}), 'sigmoid_v2': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sigmoid', {}), 'sqrt': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sqrt', {}), 'tanh': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Tanh', {}), 'reciprocal': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Reciprocal', {}), 'relu': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Relu', {}), 'broadcast_maximum': (<bound method MinMaxOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.MinMaxOp'>>, 'Max', {}), 'broadcast_minimum': (<bound method MinMaxOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.MinMaxOp'>>, 'Min', {}), 'clip_by_scalar': (<bound method ClipByValueOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'clip_by_scalar_min': (<bound method ClipByValueOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'clip_by_scalar_max': (<bound method ClipByValueOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'softmax': (<bound method Softmax.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Softmax'>>, 'Softmax', {}), 'square': (<bound method Square.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Square'>>, None, {}), 'rsqrt': (<bound method Rsqrt.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Rsqrt'>>, None, {}), 'squared_difference': (<bound method SquaredDifference.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.SquaredDifference'>>, None, {}), 'sign': (<bound method Sign.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Sign'>>, 'Sign', {}), 'matmul': (<bound method MatMul.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.MatMul'>>, 'MatMul', {}), 'batch_matmul': (<bound method MatMul.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.MatMul'>>, 'MatMul', {}), 'erf': (<bound method Erf.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Erf'>>, 'Erf', {}), 'logical_not': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Not', {}), 'broadcast_logical_or': (<bound method BroadcastOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Or', {}), 'broadcast_logical_and': (<bound method BroadcastOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'And', {}), 'input': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.DirectOp'>>, None, {}), 'return': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.DirectOp'>>, None, {}), 'variable': (<bound method DirectOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.DirectOp'>>, None, {}), 'distribute_split': (<bound method BoxingOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.BoxingOp'>>, 'Identity', {}), 'distribute_concat': (<bound method BoxingOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.BoxingOp'>>, 'Identity', {}), 'distribute_clone': (<bound method BoxingOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.BoxingOp'>>, 'Identity', {}), 'distribute_add': (<bound method BoxingOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.misc.BoxingOp'>>, 'Identity', {}), 'conv2d': (<bound method ConvOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.ConvOp'>>, None, {}), 'max_pool_2d': (<bound method PoolOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'MaxPool', {}), 'avg_pool_2d': (<bound method PoolOp.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'AveragePool', {}), 'reduce_prod': (<bound method ReduceOpBase.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceProd', {}), 'reduce_sum': (<bound method ReduceOpBase.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceSum', {}), 'reduce_min': (<bound method ReduceOpBase.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceMin', {}), 'argmax': (<bound method ArgMax.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ArgMax'>>, 'ArgMax', {}), 'argmin': (<bound method ArgMax.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ArgMax'>>, 'ArgMin', {}), 'squeeze': (<bound method Squeeze.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Squeeze'>>, 'Squeeze', {}), 'transpose': (<bound method Transpose.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Transpose'>>, 'Transpose', {}), 'concat': (<bound method Concat.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Concat'>>, 'Concat', {}), 'identity': (<bound method Identity.Version_1 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Identity'>>, 'Identity', {})}
====================================================================================================
2
{'pad': (<bound method Pad.Version_2 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.Pad'>>, 'Pad', {})}
====================================================================================================
3
{}
====================================================================================================
4
{}
====================================================================================================
5
{'reshape': (<bound method Reshape.Version_5 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Reshape'>>, 'Reshape', {})}
====================================================================================================
6
{'broadcast_div': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Div', {}), 'scalar_div_by_tensor': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Div', {}), 'multiply': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Mul', {}), 'broadcast_mul': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Mul', {}), 'scalar_mul_by_tensor': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Mul', {}), 'broadcast_sub': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Sub', {}), 'scalar_sub_by_tensor': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Sub', {}), 'broadcast_add': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Add', {}), 'scalar_add_by_tensor': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Add', {}), 'scalar_add': (<bound method ScalarBinaryOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ScalarBinaryOp'>>, 'Add', {}), 'scalar_mul': (<bound method ScalarBinaryOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ScalarBinaryOp'>>, 'Mul', {}), 'bias_add': (<bound method BiasAdd.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BiasAdd'>>, 'Add', {}), 'abs': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Abs', {}), 'ceil': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Ceil', {}), 'elu': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Elu', {}), 'exp': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Exp', {}), 'floor': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Floor', {}), 'log': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Log', {}), 'neg': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Neg', {}), 'sigmoid': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sigmoid', {}), 'sigmoid_v2': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sigmoid', {}), 'sqrt': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Sqrt', {}), 'tanh': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Tanh', {}), 'reciprocal': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Reciprocal', {}), 'relu': (<bound method DirectOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.DirectOp'>>, 'Relu', {}), 'broadcast_logical_or': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'Or', {}), 'broadcast_logical_and': (<bound method BroadcastOp.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.BroadcastOp'>>, 'And', {}), 'normalization': (<bound method BatchNorm.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.BatchNorm'>>, None, {}), 'cast': (<bound method Cast.Version_6 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Cast'>>, 'Cast', {})}
====================================================================================================
7
{'acos': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Acos', {}), 'asin': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Asin', {}), 'atan': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Atan', {}), 'cos': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Cos', {}), 'sin': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Sin', {}), 'tan': (<bound method TrigOpSinceOpset7.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset7'>>, 'Tan', {}), 'broadcast_floor_mod': (<bound method FloorMod.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.FloorMod'>>, 'FloorMod', {}), 'broadcast_equal': (<bound method Equal.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Equal'>>, 'Equal', {}), 'broadcast_not_equal': (<bound method Equal.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Equal'>>, 'NotEqual', {}), 'broadcast_greater': (<bound method GreaterLess.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.GreaterLess'>>, 'Greater', {}), 'broadcast_less': (<bound method GreaterLess.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.GreaterLess'>>, 'Less', {}), 'broadcast_less_equal': (<bound method GreaterLessEqual.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.GreaterLessEqual'>>, 'Greater', {}), 'broadcast_greater_equal': (<bound method GreaterLessEqual.Version_7 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.GreaterLessEqual'>>, 'Less', {})}
====================================================================================================
8
{}
====================================================================================================
9
{'acosh': (<bound method TrigOpSinceOpset9.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset9'>>, 'Acosh', {}), 'asinh': (<bound method TrigOpSinceOpset9.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset9'>>, 'Asinh', {}), 'atanh': (<bound method TrigOpSinceOpset9.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset9'>>, 'Atanh', {}), 'cosh': (<bound method TrigOpSinceOpset9.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset9'>>, 'Cosh', {}), 'sinh': (<bound method TrigOpSinceOpset9.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.TrigOpSinceOpset9'>>, 'Sinh', {}), 'sign': (<bound method Sign.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Sign'>>, 'Sign', {}), 'erf': (<bound method Erf.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Erf'>>, 'Erf', {}), 'normalization': (<bound method BatchNorm.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.BatchNorm'>>, None, {}), 'cast': (<bound method Cast.Version_9 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Cast'>>, 'Cast', {})}
====================================================================================================
10
{'max_pool_2d': (<bound method PoolOp.Version_10 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'MaxPool', {}), 'avg_pool_2d': (<bound method PoolOp.Version_10 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'AveragePool', {}), 'min_max_observer': (<bound method MinMaxObserver.Version_10 of <class 'oneflow_onnx.oneflow2onnx.handlers.quantize.MinMaxObserver'>>, None, {}), 'moving_average_min_max_observer': (<bound method MovingAverageMinMaxObserver.Version_10 of <class 'oneflow_onnx.oneflow2onnx.handlers.quantize.MovingAverageMinMaxObserver'>>, None, {}), 'fake_quantization': (<bound method FakeQuantization.Version_10 of <class 'oneflow_onnx.oneflow2onnx.handlers.quantize.FakeQuantization'>>, 'QuantizeLinear', {})}
====================================================================================================
11
{'clip_by_scalar': (<bound method ClipByValueOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'clip_by_scalar_min': (<bound method ClipByValueOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'clip_by_scalar_max': (<bound method ClipByValueOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.ClipByValueOp'>>, 'Clip', {}), 'softmax': (<bound method Softmax.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Softmax'>>, 'Softmax', {}), 'round': (<bound method Round.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Round'>>, 'Round', {}), 'broadcast_equal': (<bound method Equal.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Equal'>>, 'Equal', {}), 'broadcast_not_equal': (<bound method Equal.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.math.Equal'>>, 'NotEqual', {}), 'conv2d': (<bound method ConvOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.ConvOp'>>, None, {}), 'max_pool_2d': (<bound method PoolOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'MaxPool', {}), 'avg_pool_2d': (<bound method PoolOp.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.PoolOp'>>, 'AveragePool', {}), 'pad': (<bound method Pad.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.nn.Pad'>>, 'Pad', {}), 'reduce_prod': (<bound method ReduceOpBase.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceProd', {}), 'reduce_sum': (<bound method ReduceOpBase.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceSum', {}), 'reduce_min': (<bound method ReduceOpBase.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ReduceOpBase'>>, 'ReduceMin', {}), 'argmax': (<bound method ArgMax.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ArgMax'>>, 'ArgMax', {}), 'argmin': (<bound method ArgMax.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.reduce.ArgMax'>>, 'ArgMin', {}), 'squeeze': (<bound method Squeeze.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Squeeze'>>, 'Squeeze', {}), 'concat': (<bound method Concat.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.Concat'>>, 'Concat', {}), 'gather_nd': (<bound method GatherND.Version_11 of <class 'oneflow_onnx.oneflow2onnx.handlers.array.GatherND'>>, 'GatherND', {})}
====================================================================================================
12
{}
====================================================================================================
13
{'min_max_observer': (<bound method MinMaxObserver.Version_13 of <class 'oneflow_onnx.oneflow2onnx.handlers.quantize.MinMaxObserver'>>, None, {}), 'fake_quantization': (<bound method FakeQuantization.Version_13 of <class 'oneflow_onnx.oneflow2onnx.handlers.quantize.FakeQuantization'>>, 'QuantizeLinear', {})}
```

可以看到对于ONNX的每一个Opset Version的OP都对应了OneFlow实现的OP，需要特别注意的是这个OP Mapper过程在是在`https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/oneflow_onnx/oneflow2onnx/handlers`这里完成的，只要安装了oneflow-onnx这个包或者编译了oneflow-onnx工程源码，Python就会自动将OneFlow的OP和ONNX的OP进行映射，这是通过`
@flow_op(["avg_pool_2d"], onnx_op="AveragePool")`装饰器来实现的，`flow_op`装饰器的具体实现在`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/oneflow_onnx/oneflow2onnx/handler.py#L34`这里。

完成了ONNX每个Opset版本的OP和OneFlow OP的mapper之后，我们需要对`Graph`里面的ONNX节点（注意现在的ONNX节点并不是合法的ONNX节点，因为还没有执行一对一的转换，只是复制了OneFlow OP的类型、输入输出和属性值）先执行拓扑排序，然后再一对一的转换。这个地方很有意思，为什么要进行拓扑排序呢？

我们首先需要了解一下拓扑序算法，拓扑排序要解决的问题是给一个图的所有节点排序。

以下对拓扑排序的解释引自oi.wiki。


我们可以拿大学选课的例子来描述这个过程，比如学习大学课程中有：单变量微积分，线性代数，离散数学概述，概率论与统计学概述，语言基础，算法导论，机器学习。当我们想要学习 算法导论 的时候，就必须先学会 离散数学概述 和 概率论与统计学概述，不然在课堂就会听的一脸懵逼。当然还有一个更加前的课程 单变量微积分。这些课程就相当于几个顶点 $u$, 顶点之间的有向边 $(u,v)$ 就相当于学习课程的顺序。显然拓扑排序不是那么的麻烦，不然你是如何选出合适的学习顺序。下面将介绍如何将这个过程抽象出来，用算法来实现。

但是如果某一天排课的老师打瞌睡了，说想要学习 算法导论，还得先学 机器学习，而 机器学习 的前置课程又是 算法导论，然后你就一万脸懵逼了，我到底应该先学哪一个？当然我们在这里不考虑什么同时学几个课程的情况。在这里，算法导论 和 机器学习 间就出现了一个环，显然你现在没办法弄清楚你需要学什么了，于是你也没办法进行拓扑排序了。因而如果有向图中存在环路，那么我们就没办法进行 拓扑排序 了。

因此我们可以说 在一个 DAG（有向无环图），我们将图中的顶点以线性方式进行排序，使得对于任何的顶点 $u$ 到 $v$ 的有向边 $(u,v)$, 都可以有 $u$ 在 $v$ 的前面。

还有给定一个 DAG，如果从 $i$ 到 $j$ 有边，则认为 $j$ 依赖于 $i$。如果 $i$ 到 $j$ 有路径（$i$ 可达 $j$），则称 $j$ 间接依赖于 $i$。

**拓扑排序的目标是将所有节点排序，使得排在前面的节点不能依赖于排在后面的节点。** 伪代码实现如下：

```cpp
void TopologicalSort(Graph G){
	InitStack(S);
	for(i = 0;i < G.vexnum; i++){
		if(indegrdd[i]==0)
			Push(S, i);
	}
 
	int count =0;
	while(!Empty(S)){
		Pop(S,i);
		print[count++] = i;
		for(p = G.vertices[i].firstarc; p; p = p->nextarc){
			v = p->adjvex;
			if(!(--indegree[v]))
				Push(S, v);
		}
	}
 
	if(count < G.vexnum)
		return false;
	else
		return true;
}
```

上面加粗的这句话即是拓扑排序的核心。一般深度学习模型也是一个DAG（有向无环图），我们这里同样使用了拓扑排序算法使得我们在一对一转换OP时和真实的网络结构是完全一致的。另外考虑到这里可能插入了一些新的节点如Identity可能会破坏原Graph的拓扑序，以及时刻需要判断计算图是否是一个完整合法的DAG，使用拓扑排序都是没有坏处的。

完成拓扑排序之后我们就可以执行`FlowOnnxMapping`完成OneFlow OP和ONNX OP的一对一转换了，代码如下：

```python
def FlowOnnxMapping(g, ops_mapping):
    logger.debug("Mapping Oneflow node to ONNX node(s)")
    mapped_op = collections.Counter()
    unmapped_op = collections.Counter()
    exceptions = []

    ops = list(g.get_nodes())
    for node in ops:
        logger.debug("Process node: %s\n%s", node.name, node.summary)

        if node.skip_conversion:
            logger.debug("explicitly skip node " + node.name)
            continue

        op = node.op_type
        map_info = ops_mapping.get(op)
        if map_info is None:
            unmapped_op[op] += 1
            logger.error("oneflow op [%s: %s] is not supported", node.name, op)
            continue
        mapped_op[op] += 1

        func, onnx_op, kwargs = map_info
        if onnx_op is not None:
            node.op_type = onnx_op
        try:
            func(g, node, **kwargs)
            node.skip_conversion = True
        except Exception as ex:
            logger.error(
                "Failed to convert node %s\n%s", node.name, node.summary, exc_info=1
            )
            exceptions.append(ex)

    return mapped_op, unmapped_op, exceptions
```

执行完这个函数会返回map上的OP容器，以及没有map上的OP容器，当然如果`Graph`中有OP没有map上也就是转换失败会抛出错误信息给用户。在转换完成之后，我们调用`Graph`中的每个`Node`的`UpdateProto()`构造函数将之前的假ONNX节点信息更新成真的ONNX节点信息。

接下来，我们调用各种 optimizer 优化网络结构，例如尽可能消除 nhwc->nchw 带来的 transpose op（Export 函数内的 optimizer.OptimizeGraph），即`https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/main/oneflow_onnx/oneflow2onnx/flow2onnx.py#L264`。在oneflow-onnx里面主要有以下几种optimizer：

![oneflow-onnx里面的optimizer，获得更优的ONNX模型](https://img-blog.csdnimg.cn/2021040411182963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

这些 optimizer 继承自 tensorflow-onnx，我们后续会将其中的一部分用 onnx 原生的 optimizer 替代。


在优化了ONNX模型之后，最后调用下面的函数取磁盘中保存的 oneflow 权重，赋给 onnx 模型对象，并返回 protobuf 格式的 onnx 模型对象。至此就完成了创建合法的ONNX模型。

```python
model_proto = onnx_graph.MakeModel(
                job_name, onnx_filename, external_data=external_data
            )
```

我们的X2OneFlow分为X2ONNX和ONNX2Oneflow两个步骤，其中ONNX2OneFlow和OneFlow2ONNX共用了一套基础代码，所以需要修改的地方仅仅是将`handles`里面的注册OP转换的装饰器改个方向即可，这里不再赘述。

想了解更多细节可以看我们的源码`https://github.com/Oneflow-Inc/oneflow_convert_tools/tree/main/oneflow_onnx`。


# 0x4. 总结
在这篇文章中，我们分享了最近维护OneFlow和ONNX做的一系列工作，这项工作目前已经开源并在持续维护中，欢迎对OneFlow框架以及ONNX感兴趣的小伙伴体验和PR。点击阅读原文快速关注和体验oneflow-onnx工具。

# 0x5. 相关链接
- OneFlow：https://github.com/Oneflow-Inc/oneflow
- https://github.com/onnx/tensorflow-onnx
- https://github.com/OI-wiki/OI-wiki