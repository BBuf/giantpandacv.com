# AiDB: 一个集合了6大推理框架的AI工具箱 | 加速你的模型部署

> 项目地址: https://github.com/TalkUHulk/ai.deploy.box
>
> 网页体验: https://www.hulk.show/aidb-webassembly-demo/
>
> PC: https://github.com/TalkUHulk/aidb_qt_demo
>
> Android: https://github.com/TalkUHulk/aidb_android_demo
>
> Go Server: https://github.com/TalkUHulk/aidb_go_demo
>
> Python Server: https://github.com/TalkUHulk/aidb_python_demo
>
> Lua: https://github.com/TalkUHulk/aidb_lua_demo


## 导读

  本文介绍了一个开源的AI模型部署工具箱--AiDB。该项目使用C++开发，将主流深度学习推理框架抽象成统一接口，包括ONNXRUNTIME、MNN、NCNN、TNN、PaddleLite和OpenVINO，支持Linux、MacOS、Windows、Android、Webassembly等平台。AiDB提供C/C++/Python/Lua等多种API接口。并且提供多种场景的部署实例Demo(Server、PC、Android等)。目前，AiDB集成了数十个开源算法(如Yolo系列、MobileSAM等)，约300个模型，并且持续更新。


![](https://files.mdnice.com/user/48619/175506ab-301c-471f-bc9b-3b955393275d.jpg)



  AiDB具备以下特点：
  
- 集成了市面上主流的推理框架，并抽象成统一的接口；
- 支持Linux、Windows、MacOS、Android、Webassembly等各种平台部署；
- 支持C/C++、Python、Lua接口；
- 使用友好，支持docker一键安装，开箱即用；
- 丰富的部署实例，包括Android(kotlin)、PC(Qt5)、Server(Go Zeros | Python FastApi)、Web(Webassembly)；
- 提供C++/Python/Go/Lua的Colab demo；
- 内置丰富的模型，涵盖检测、关键点、分类、分割、生成等十几种开源算法，300余个模型;


## 整体架构

整个项目的架构如下：


![](https://files.mdnice.com/user/48619/5ca22723-d7ed-4a9e-a068-9c759b4f9c4b.png)

底层封装了六类推理框架，集成前后处理和日志模块，支持各类平台。内置十余种开源算法。提供C/C++、Python、Lua等接口。上层提供各种场景调用实例。

### Backend封装

  主流推理框架的调用接口其实大同小异。主要可以概括为4大步: 1.初始化；2.数据输入；3.预测；4.获取结果。但每个推理框架的具体参数和一些细节又各有不同，如MNN动态输入等。所以为了后续可以动态选择不同的backend，AiDB抽象出一个基类：
  ```
      class AIDB_PUBLIC Engine {
    public:
        Engine() = default;
        virtual StatusCode init(const Parameter&) = 0;
        virtual StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) = 0;
        virtual ~Engine(){};
        virtual void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) = 0;
        virtual void forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) = 0;

        std::vector<std::string> _output_node_name;
        std::vector<std::string> _input_node_name;
        std::map<std::string, std::vector<int>> _input_nodes;  /*!< 输入节点信息*/
        bool _dynamic=false;
        std::string _model_name = "default";
        std::string _backend_name = "default";

    };
  ```
所有的backend通过Paramater初始化。当每个模型初始化后，通过forward函数完成预测操作。这里设计了两个forward函数。1.x版本只支持single input。这个函数可能已经满足了大部分模型的需求，但随着更多的模型加入，有些模型需要multi-input，如最近加入的MobileSAM。所以后续重新设计了forward，支持任意数量的输入和输出。

后面每类backend继承这个类实现各自的forward和init即可。比如我们要实现MNN backend：

```
 class MNNEngine: public Engine{
    public:
        MNNEngine();
        StatusCode init(const Parameter&) override;
        StatusCode init(const Parameter&, const void *buffer_in1, const void* buffer_in2) override;
        ~MNNEngine() override;
        void forward(const void *frame, int frame_width, int frame_height, int frame_channel, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
        void forward(const std::vector<void*> &input, const std::vector<std::vector<int>> &input_shape, std::vector<std::vector<float>> &outputs, std::vector<std::vector<int>> &outputs_shape) override;
    private:
        void reshape_input(const std::vector<int>&);
        std::shared_ptr<MNN::Tensor> get_output_by_name(const char *name);
        MNN::Tensor* get_input_tensor(const char *node_name);
        MNN::Tensor* get_input_tensor();

    private:
        std::shared_ptr<MNN::Interpreter> _mnn_net;
        MNN::ScheduleConfig _net_cfg;
        MNN::Session *_mnn_session;

    };
```

通过如上操作，分别实现onnx、mnn、ncnn、paddlelite和openvino的backend部分，之后我们就可以利用c++多态特性，通过配置文件动态初始化不同的backend:

```
switch(engineType(model_node["backend"].as<std::string>())){

            case ONNX:{
#ifdef ENABLE_ORT
                ONNXParameter param = ONNXParameter(model_node);
                ptr_engine = new ONNXEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }

            case MNN:{
#ifdef ENABLE_MNN
                MNNParameter param = MNNParameter(model_node);
                ptr_engine = new MNNEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }

            case NCNN:{
#ifdef ENABLE_NCNN
                NCNNParameter param = NCNNParameter(model_node);
                ptr_engine = new NCNNEngine();
                status = ptr_engine->init(param);
#endif
                break;
            }
            case TNN:{
#ifdef ENABLE_TNN
                TNNParameter param = TNNParameter(model_node);
                ptr_engine = new TNNEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            case OPENVINO:{
#ifdef ENABLE_OPV
                OPVParameter param = OPVParameter(model_node);
                ptr_engine = new OPVEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            case PADDLE_LITE:{
#ifdef ENABLE_PPLite
                PPLiteParameter param = PPLiteParameter(model_node);
                ptr_engine = new PPLiteEngine();
                status = ptr_engine->init(param);
#endif
            }
                break;
            default:
                break;
        }
```


### 预处理

  每个模型的inference代码区别不大，差异主要集中在预处理和后处理阶段。后处理部分根据各个任务的不同(分类、检测等)，很难抽象出统一的接口。但预处理可以很简单的实现统一。这里AiDB实现了一个简单的预处理类：
  
```
class ImageInput: public AIDBInput{
    public:
        explicit ImageInput(const YAML::Node& input_mode);
        explicit ImageInput(const std::string& input_str);
        ~ImageInput() override;
        void forward(const cv::Mat &image, cv::Mat &blob) override;
        void forward(const std::string &image_path, cv::Mat &blob) override;
    private:
        void Normalize(cv::Mat &image);
        void Permute(const cv::Mat &image, cv::Mat &blob);
        void Resize(const cv::Mat &image, cv::Mat &resized);
        void cvtColor(const cv::Mat &image, cv::Mat &converted);
    };
```

使用yaml配置文件，为每个模型设置设置预处理操作：

```
BISENET: &bisenet_detail
    num_thread: 4
    device: "CPU"
    PreProcess:
      shape: &shape
        width: 512
        height: 512
        channel: 3
        batch: 1
      keep_ratio: true
      mean:
        - 123.675
        - 116.28
        - 103.53
      var:
        - 58.395
        - 57.12
        - 57.375
      imageformat: "RGB"
      inputformat: &format "NCHW"
    input_node1: &in_node1
      input_name: "input"
      format: *format
      shape: *shape
    input_nodes:
      - *in_node1
    output_nodes:
      - "output"

```

### 接口

  考虑Ai模型的主要部署场景，AiDB实现了两套接口，分别起名H-mode和S-mode，即静态模式和动态模式。以下两图展示了两种模式的不同(上-H-mode;下:S-mode):
  

  
<![H1](https://files.mdnice.com/user/48619/0e0e8dbd-311d-4cf4-b4d5-7f0382e880ed.jpg), ![H2](https://files.mdnice.com/user/48619/0d8143d9-e8d6-4b21-8d97-046d2d9ad964.jpg), ![H3](https://files.mdnice.com/user/48619/8c696343-1f81-46c7-a2e1-3acdaabd2ddf.jpg), ![H4](https://files.mdnice.com/user/48619/f4d51f90-e635-4e91-8b50-ce8bcd26b82c.jpg)>


<![S1](https://files.mdnice.com/user/48619/f38530d0-6733-41e3-920d-abe5f1d3416c.png), ![S2](https://files.mdnice.com/user/48619/0dae9a0a-1fc1-4371-99fe-b0f988ccbdfc.png), ![S3](https://files.mdnice.com/user/48619/973c0cc1-9400-4063-8812-3e8e7fa0df5b.png), ![S4](https://files.mdnice.com/user/48619/573f09f8-85e0-45ab-bf42-3514ed168c22.png)>


  动态模式适合用在服务端，可以方便的实现热插拔，而静态模式更注重效率和性能，适合在边缘上设备使用。
  

### 内置模型

  目前，AiDB内置了十余种开源算法，约300个不同的模型。未来，AiDB会持续更新，加入更多不同的模型。
  

![当前模型列表](https://files.mdnice.com/user/48619/de81d03e-8f58-43b4-b41a-e2fbc54d43f4.jpg)


## 部署实例

  AiDB的最大目的就是加速AI模型的部署。所以以下内容展示了不同场景的部署实例。
  
### Python


  Python的语法相对简单明了，具有更高的可读性。在Ai领域, Python使用是比较广泛的。因此AiDB支持Python接口，简化调用难度。
  AiDB使用pybind11实现python绑定。目前只支持从源码安装pyAiDB:
  ```
  python setup.py build_ext --inplace
  ```
  
  详细过程可以可以参考Colab中的python编译调用全过程(https://colab.research.google.com/drive/1gVKxkeIvgrnC56dVQOImyqQqVns-NtkR)。当完成编译安装，我们可以按如下方式调用对应的模型算法:
  
  ```
  from pyAiDB import AiDB, AIDB_ModelID
  import cv2
  import numpy as np
  from PIL import Image, ImageDraw, ImageFont

  ImagePath = "./doc/test/face.jpg" 
  Model = "scrfd_500m_kps" # @param {"type": "string"}
  Backend = "mnn" # @param {"type": "string"}
  bgr = cv2.imread(ImagePath)
  h, w, c = bgr.shape
  aidb = AiDB()
  models = [Model]
  backend = [Backend]
  aidb.init(AIDB_ModelID.SCRFD, models, backend, "./config")
  result = aidb.forward(bgr.data, w, h)
  ```
  当然，python 绑定/调用C++的方式有很多，为了满足不同需求，这里也提供了ctypes调用so的例子(https://link.zhihu.com/?target=https%3A//github.com/TalkUHulk/aidb_python_demo/tree/master)（fastapi搭建的AI服务）


### Go

  公司实际业务中，我们常会使用Go/Java，为了更贴近实际业务，AiDB提供了基于Go Zeros的服务实例(https://github.com/TalkUHulk/aidb_go_demo)。
  
![Go Server](https://files.mdnice.com/user/48619/b5deab81-427b-4059-837c-62ae56acd7ae.jpg)

  
  Go调用AiDB通过CGO的方式，如果你对此感兴趣，可以参考Colab(https://colab.research.google.com/drive/15DTMnueAv2Y3UMk7lhXMMN_VVUCBA0qh?usp=drive_link)：
  
### Android

  MNN、NCNN等推理框架主要针对移动端设计优化，AiDB也因此继承式地支持手机端的部署。
这里给出一个Android部署实例。重点就是实现JNI部分，开发语言使用Kotlin。【测试机器:Google Pixel 4, Android:13]

<![3ddfa-dense](https://files.mdnice.com/user/48619/20db74df-454c-4770-965c-73606da52c6e.png),![3ddfa-base](https://files.mdnice.com/user/48619/f6362bd0-6260-4ebe-905a-ce8c575068fe.png),![yolox](https://files.mdnice.com/user/48619/d3cb5452-edba-4de8-9c36-e2af9bb71341.png),![ocr](https://files.mdnice.com/user/48619/bc3a0375-151d-401a-b8b7-40c9cd74ee16.png)>



### PC(Qt5)

  实际业务或是开发过程中，我们需要将自己的模型show出来，或是演示，或是作为一个里程碑，亦或是一个demo产品。鉴于此，AiDB提供一个桌面级部署实例,考虑跨平台需求，选用Qt5开发。

<![mobile-sam1](https://files.mdnice.com/user/48619/8e86306e-7fc3-4cd2-9cf7-da7224da4aac.png),![mobile-sam2](https://files.mdnice.com/user/48619/e4b6005f-0468-4ad3-9d04-4b527cbd9075.png),![face](https://files.mdnice.com/user/48619/1aa5a224-68dc-47e8-84b8-be4b4c6397c6.png),![ocr](https://files.mdnice.com/user/48619/76d231e7-a589-4445-b8f5-11efe9d75e30.png)>


### Web(Webassembly)

  WebAssembly即WASM， WebAssembly是一种新的编码格式并且可以在浏览器中运行,它让我们能够使用JavaScript以外的语言（例如C，C++）编写程序，然后将其编译成WebAssembly，进而生成一个加载和执行速度非常快的Web应用程序。
  目前NCNN和OpenVINO都支持wasm，AiDB已经支持了NCNN wasm版本，openvino列入计划。同时，AiDB也提供了一个wasm的demo,同时支持在线体验(http://www.hulk.show/aidb-webassembly-demo/):

<![wasm](https://files.mdnice.com/user/48619/50761866-cfdd-4af0-bdcc-3f47c4b94892.png), ![yolox2](https://files.mdnice.com/user/48619/9c05cc6f-fb51-4ea6-b504-46f809016916.png), ![yolox3](https://files.mdnice.com/user/48619/e4ad17d9-b77f-4e4f-a246-bd678433398a.png), ![face](https://files.mdnice.com/user/48619/924411fd-0998-4384-bf05-096a0102408e.png)>




### 彩蛋

  在Rasberry Pi4部署AiDB:
  

<![face1](https://files.mdnice.com/user/48619/9bce93ac-ca8d-4082-a736-e2ce918d8a4d.png), ![yolox2](https://files.mdnice.com/user/48619/b25decfd-d2e5-4c0d-97de-7f4c683d2fb5.png)>



## 拾遗

  AiDB开发过程中遇到了很多问题，主要集中在移动端，相关趟坑已经记录在github中。问题比较多的是paddle-lite和openvino的移动端部署。paddle-lite更多的是转模型过程中版本对应的问题。openvino则全网几乎没有移动端部署教程。官方给的也是java接口的调用。openvino的调用和mnn、ncnn这些对比，调用方式还是有很大不同的。
  总结下android端c++中调用openvino的方法：

1. 编译对应平台的库（以下为AiDB使用的）

```
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=android-ndk-r25c/build/cmake/android.toolchain.cmake -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=30 -DANDROID_STL=c++_shared  -DENABLE_SAMPLES=OFF  -DENABLE_OPENCV=OFF -DENABLE_CLDNN=OFF -DENABLE_VPU=OFF  -DENABLE_GNA=OFF -DENABLE_MYRIAD=OFF -DENABLE_TESTS=OFF  -DENABLE_GAPI_TESTS=OFF  -DENABLE_BEH_TESTS=OFF ..
```

2. 把需要的.so扔到assets下（如果是ir模型，只需要基础的so和ir plugin）

3. 如果你的设备没root，libc++.so 和 libc++_shared.so 也一起扔进 assets

4. 在c++ cmakelist中做好相关配置

```
add_library(openvino SHARED IMPORTED)

set_target_properties(
                openvino
                PROPERTIES IMPORTED_LOCATION
                ${CMAKE_CURRENT_LIST_DIR}/../libs/android/openvino/libopenvino.so)
```
以及kotlin中

```
System.loadLibrary("openvino");
```

## 如何增加模型

在对应的config里增加模型配置，比如onnx_config.yaml

```
SCRFD_2_5G_KPS: &scrfd_2_5g_kps
name: "SCRFD_2.5G_KPS"
model: *mp_scrfd_2_5g
backend: "ONNX"
detail: *scrfd_detail
```

mp_scrfd_2_5g为模型路径：

```
SCRFD_2_5G_KPS: &mp_scrfd_2_5g "./models/onnx/scrfd/scrfd_2.5g_kps_simplify"
```

scrfd_detail为详细的模型相关信息:

```
SCRFD: &scrfd_detail
    encrypt: false
    num_thread: 4
    device: "CPU"
    PreProcess:
      shape: &shape
        width: 640
        height: 640
        channel: 3
        batch: 1
      keep_ratio: true
      mean:
        - 127.5
        - 127.5
        - 127.5
      var:
        - 128.0
        - 128.0
        - 128.0
      border_constant:
        - 0.0
        - 0.0
        - 0.0
      imageformat: "RGB"
      inputformat: &format "NCHW"

    input_node1: &in_node1
      input_name: "images"
      format: *format
      shape: *shape
    input_nodes:
      - *in_node1
    output_nodes:
      - "out0"
      - "out1"
      - "out2"
      - "out3"
      - "out4"
      - "out5"
      - "out6"
      - "out7"
      - "out8"
```

最后在AIDBZOO里声明下模型：
```
scrfd_2.5g_kps: *scrfd_2_5g_kps
```
如果新加入模型有额外预处理操作，则需要增加该部分代码。

> 更多详情，敬请登陆github，欢迎Star。





  
