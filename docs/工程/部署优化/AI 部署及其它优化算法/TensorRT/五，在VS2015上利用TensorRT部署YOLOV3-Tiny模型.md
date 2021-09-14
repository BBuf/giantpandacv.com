# 1. 前言
大家好，最近在VS2015上尝试用TensorRT来部署检测模型，中间走了两天弯路，感觉对于一个完全新手来说要做成功这件事并不会那么顺利。所以这里写一篇部署文章，希望能让使用TensorRT来部署YOLOV3-Tiny检测模型的同学少走一点弯路。

# 2. 确定走哪条路？
这里我是将AlexeyAB版本DarkNet训练出来的YOLOV3-Tiny检测模型（包含`*.weights`和`*.cfg`）利用TensorRT部署在NVIDIA的1060显卡上。我选择的模型转换道路是`DarkNet->ONNX->TRT`。我们知道TensorRT既可以直接加载ONNX也可以加载ONNX转换得到的TRT引擎文件，而ONNX模型转TRT引擎文件是非常简单的，这个可以直接在代码里面完成，所以我们首先需要关注的是DarkNet模型转换到ONNX模型。

# 3. DarkNet2ONNX
现在已经明确，首要任务是模型转换为ONNX模型。这个我们借助Github上的一个工程就可以完成了，工程地址为：`https://github.com/zombie0117/yolov3-tiny-onnx-TensorRT`。具体操作步骤如下：
- 克隆工程。
- 使用Python2.7。
- 执行`pip install onnx=1.4.1`
- 将YOLOV3-Tiny的`cfg`文件末尾手动添加一个空行。
- 修改`yolov3_to_onnx.py`的`cfg`和`weights`文件的路径以及ONNX模型要保存的路径。
- 执行`yolov3_to_onnx.py`脚本，获得`yolov3-tiny.onnx`模型。

我们来看一下`yolov3-tiny.onnx`模型的可视化结果（使用Neutron），这里只看关键部分：

![yolov3-tiny.onnx可视化](https://img-blog.csdnimg.cn/20200327164426219.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![yolov3-tiny.onnx可视化](https://img-blog.csdnimg.cn/20200327164441502.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到ONNX模型里面最后一层`YOLO`层是不存在了(ONNX不支持`YOLO`层，所以就忽略了)，最后的两个输出层是做特征映射的$1\times 1$卷积层，这就意味着后面的BBox和后处理NMS都是需要我们在代码中手动完成的。


# 4. ONNX2TRT
在获得了YOLOV3-Tiny的ONNX模型后，我们可以就可以将ONNX转为TensorRT的引擎文件了，这一转换的代码如下：

```
// ONNX模型转为TensorRT引擎
bool onnxToTRTModel(const std::string& modelFile, // onnx文件的名字
	const std::string& filename,  // TensorRT引擎的名字 
	IHostMemory*& trtModelStream) // output buffer for the TensorRT model
{
	// 创建builder
	IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
	assert(builder != nullptr);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();

	// 解析ONNX模型
	auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());


	//可选的 - 取消下面的注释可以查看网络中每层的星系信息
	//config->setPrintLayerInfo(true);
	//parser->reportParsingInfo();

	//判断是否成功解析ONNX模型
	if (!parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.getReportableSeverity())))
	{
		gLogError << "Failure while parsing ONNX file" << std::endl;
		return false;
	}

	// 建立推理引擎
	builder->setMaxBatchSize(BATCH_SIZE);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setFp16Mode(true);
	builder->setInt8Mode(gArgs.runInInt8);

	if (gArgs.runInInt8)
	{
		samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
	}

	cout << "start building engine" << endl;
	ICudaEngine* engine = builder->buildCudaEngine(*network);
	cout << "build engine done" << endl;
	assert(engine);

	// 销毁模型解释器
	parser->destroy();

	// 序列化引擎
	trtModelStream = engine->serialize();

	// 保存引擎
	nvinfer1::IHostMemory* data = engine->serialize();
	std::ofstream file;
	file.open(filename, std::ios::binary | std::ios::out);
	cout << "writing engine file..." << endl;
	file.write((const char*)data->data(), data->size());
	cout << "save engine file done" << endl;
	file.close();

	// 销毁所有相关的东西
	engine->destroy();
	network->destroy();
	builder->destroy();

	return true;
}
```

执行了这个函数之后就会在指定的目录下生成`yolov3-tiny.trt`，从下图可以看到这个引擎文件有48.6M，而原始的`weights`文件是34.3M。

![yolov3-tiny.trt](https://img-blog.csdnimg.cn/20200327171439319.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 5. 前向推理&后处理
这部分就没有什么细致讲解的必要了，直接给出源码吧。由于篇幅原因，我把源码上传到Github了。地址为：`https://github.com/BBuf/cv_tools/blob/master/trt_yolov3_tiny.cpp`。 **注意我是用的TensorRT版本为6.0**。修改ONNX模型的路径和图片路径就可以正确得到推理结果了。

# 6. 后记
这篇文章就是为大家分享了一个DarkNet2ONNX的工具，以及提供了在VS2015中利用TensorRT完成ONNX到TRT引擎文件的转换并进行预测的C++代码，希望可以帮助到刚使用TensorRT的同学，仅此而已。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)