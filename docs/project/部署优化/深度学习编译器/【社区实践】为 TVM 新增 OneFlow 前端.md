# 0x0. 背景
去年在Summer Code的时候我刚好开始入门 TVM（虽然现在仍然也还是入门阶段，没做过什么有意义的工作），并且恰好来到OneFlow 工作就想着给 TVM 添加一个 OneFlow 前端。但可惜在 Summer Code 发起了这个项目后因为系统选人的 BUG 导致没有选到合适的候选人。后来我私下联系了申请这个项目的第二位候选人胡伽魁同学问他是否愿意来 OneFlow 实习并花1-2个月完成这件事，他同意了并在实习期间做了一个初版出来。感谢胡伽魁同学的贡献。

在这个初版的基础上，我做了一系列 代码重构，BUG 修复，文档编写，支持更多算子和模型转换之后 使其达到了一个相对稳定的状态。所以这篇文章来分享一下做这个小项目的经历和技术细节，希望对想做开源项目但还没有做过的读者提供一个参考。

# 0x1. 效果

![文档预览](https://img-blog.csdnimg.cn/2b5e886ca6894adf82dd71d0a6dc5a33.png)

这里没有截图全，可以去官方查看 https://tvm.apache.org/docs/how_to/compile_models/from_oneflow.html 。

Python API预览：

![Python API预览](https://img-blog.csdnimg.cn/9a88a95ce51e4e13b122183325a74240.png)

现在已经成功支持了 ResNet, MobileNet, ShuffleNet，GhostNet，YOLOV3，SRGAN，Vision Transformer在类的多种视觉模型，欢迎大家使用。使用方法见 https://tvm.apache.org/docs/how_to/compile_models/from_oneflow.html 。


# 0x2. PR历程

下面的截图展示了这一工作的 PR 流程，在4月合并了基础功能的 PR 后基本做的都是 Op 支持和模型支持以及 BUG 修复。

![PR历程](https://img-blog.csdnimg.cn/e5537b3365a845f8a08dfc1de7f9fc46.png)

十分感谢TVM社区的 **@masahi** 在 PR 过程中的热心帮助。 

# 0x3. 技术细节
实际上并没有什么细节可讲，基本上就是将OneFlow的IR进行逐一遍历以及逐 Op 转换。我之前已经介绍过 TVM 的 ONNX 前端的技术细节了：[【从零开始学TVM】三，基于ONNX模型结构了解TVM的前端](https://mp.weixin.qq.com/s/KFxd3zf76EP3DFcCAPZjvQ)  ，所以这里就不再重复类似的细节了。我这里只列举一下 OneFlow 前端实现中的一些特殊一点的细节。

- 形状和类型推导：对输入 Tensor 进行形状和类型推导，功能由 TVM 提供。代码见：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/common.py#L524-L532
- 形状和类型提升：对于如 concat 之类的 Op 来说，如果输入 Tensor 是不同类型或者不同形状并且符合提升原则的，那么就可以将其提升到最高类型或者固定的形状然后再转到 Relay IR。具体实现见：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L95-L112
- 消除获取 OneFlow Op 的输入 Tensor 名字的随机性：这个问题是因为 OneFlow 的 IR 由 Protobuf 做的序列化，所以导致遍历某个 Node 的时候拿到输入的名字是随机的，可能会造成 BUG 。为了解决这一问题，在获取名字的时候维护了一个有序的列表。具体实现在：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L1756-L1765
- Relay IR输入的确定：在 OneFlow IR 中输入节点名字都是带有`_input.`这个特征的，所以根据这个特征可以确定 Relay IR 的输入节点。具体实现：https://github.com/apache/tvm/blob/main/python/tvm/relay/frontend/oneflow.py#L1816-L1840

确定了输入 以及 Op 转换的规则后，整个 Relay IR 就可以方便的被构造出来了，感觉就没有什么可说的了。如果还想了解更多 TVM 前端具体细节就看上面那个链接吧。

# 0x4. 总结

本文简要介绍了笔者和胡伽魁给 TVM 新增 OneFlow 前端的工作，希望对想做开源项目但还没有做过的读者提供一个参考。

# 0x5. 参考链接
- https://github.com/apache/tvm
- https://github.com/Oneflow-Inc/oneflow