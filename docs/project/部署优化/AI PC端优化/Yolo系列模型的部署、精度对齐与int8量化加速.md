#  Yolo系列模型的部署、精度对齐与int8量化加速

> 本文写于2023-11-02晚
>
> 若需转载请联系 [haibintian@foxmail.com](mailto:haibintian@foxmail.com)

大家好，我是海滨。写这篇文章的目的是为宣传我在23年初到现在完成的一项工作---Yolo系列模型在TensorRT上的部署与量化加速，目前以通过视频的形式在B站发布（不收费，只图一个一剑三连）。

麻雀虽小但五脏俱全，本项目系统介绍了YOLO系列模型在TensorRT上的量化方案，工程型较强，我们给出的工具可以实现不同量化方案在Yolo系列模型的量化部署，无论是工程实践还是学术实验，相信都会对你带来一定的帮助。

>B站地址（求关注和三连）：https://www.bilibili.com/video/BV1Ds4y1k7yr/
>
>Github开源地址（求star）：https://github.com/thb1314/mmyolo_tensorrt/

当时想做这个的目的是是为了总结一下目标检测模型的量化加速到底会遇到什么坑，只是没想到不量化坑都会很多。

比如即使是以FP32形式推理，由于TensorRT算子参数的一些限制和TRT和torch内部实现的不同，导致torch推理结果会和TensorRT推理结果天然的不统一，至于为什么不统一这里卖个关子大家感兴趣可以看下视频。

下面说一下我们这个项目做了哪些事情

1. **YOLO系列模型在tensorrt上的部署与精度对齐**

   该项目详细介绍了Yolo系列模型在TensorRT上的FP32的精度部署，基于mmyolo框架导出各种yolo模型的onnx，在coco val数据集上对齐torch版本与TensorRT版本的精度。

   在此过程中我们发现，由于TopK算子限制和NMS算子实现上的不同，我们无法完全对齐torch和yolo模型的精度，不过这种风险是可解释且可控的。

2. **详解TensorRT量化的三种实现方式**

   TensorRT量化的三种实现方式包括trt7自带量化、dynamic range api，trt8引入的QDQ算子。

   Dynamic range api会在采用基于MQbench框架做PTQ时讲解。

   TensorRT引入的QDQ算子方式在针对Yolo模型的PTQ和QAT方式时都有详细的阐述，当然这个过程也没有那么顺利。

   在基于PytorchQuantization导出的含有QDQ节点的onnx时，我们发现尽管量化版本的torch模型精度很高，但是在TensorRT部署时精度却很低，TRT部署收精度损失很严重，通过可视化其他量化形式的engine和问题engine进行对比，我们发现是一些层的int8量化会出问题，由此找出问题量化节点解决。

3. **详解MQbench量化工具包在TensorRT上的应用**

   我们研究了基于MQbench框架的普通PTQ算法和包括Adaround高阶PTQ算法，且启发于Adaround高阶PTQ算法。

   我们将torch版本中的HistogramObserver引入到MQBench中，activation采用HistogramObserver
   weight采用MinMaxObserver，在PTQ过程中，weight的校准前向传播一次，activation的校准需要多次
   因此我们将weight的PTQ过程和activation的PTQ过程分开进行，加速PTQ量化。
   实践证明，我们采用上述配置的分离PTQ量化在yolov8上可以取得基本不掉点的int8量化精度。

4. **针对YoloV6这种难量化模型，分别采用部分量化和QAT来弥补量化精度损失**

   在部分量化阶段，我们采用量化敏感层分析技术来判断哪些层最需要恢复原始精度，给出各种metric的量化敏感层实现。

   在QAT阶段，不同于原始Yolov6论文中蒸馏+RepOPT的方式，我们直接采用上述部分量化后的模型做出初始模型进行finetune，结果发现finetune后的模型依然取得不错效果。

5. **针对旋转目标检测，我们同样给出一种端到端方案，最后的输出就是NMS后的结果。**
   通过将TensorRT中的EfficientNMS Plugin和mmcv中旋转框iou计算的cuda实现相结合，给出EfficientNMS for rotated box版本，经过简单验证我们的TRT版本与Torch版本模型输出基本对齐。



以上就是我们这个项目做的事情，欢迎各位看官关注b站和一剑三连。同时，如果各位有更好的想法也欢迎给我们的git仓库提PR。
