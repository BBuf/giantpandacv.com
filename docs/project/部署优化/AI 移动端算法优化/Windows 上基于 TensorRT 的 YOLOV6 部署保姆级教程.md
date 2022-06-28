### 部署环境和相关依赖包
- Cuda 11.0.2
- Cudnn 8.0.4.30
- TensorRT 8.4.0.6
- OpenCV 4.1.1
- VS2019

项目所需的安装包均放到到如下百度云链接
链接：https://pan.baidu.com/s/1C4jYSKAN2P_GSpFiikhY_g 
提取码：71ou 

### 部署流程
- 第一步把VS2019装好，安装略，百度云提供了安装包，可自行安装
- 第二步Cuda安装流程如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/ec3aea80201d4feba34c1d1eb36c2970.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/898a15b8cc6041a0b48779d087ee92bc.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/e26ff4c5ac784f658fd54c92955a5a71.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/dcac53f0dfdf43e796314b7b046b7510.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b62402809b5a4f28a3a214f55ee2d1d6.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/72a11472fc5a49ae94aa8340021517d4.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/b7e2b4dd2dc546ebb3b50974d88ee60d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/fb7dc8fd9d844057b3f25421120d65af.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/26c3b2d9e18f4baabfeb8b446d6fc33c.png)

- 第三步：解压如上cudnn压缩包，把如下目录的文件拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin目录下

![在这里插入图片描述](https://img-blog.csdnimg.cn/5012dda5491b4bdcb33d00c15a15f6d8.png)
把如下目录的文件拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\include目录下

![在这里插入图片描述](https://img-blog.csdnimg.cn/24a796bc0a63454aaa7c9f8359bbf656.png)
把如下目录的文件拷贝到C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64目录下

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb622cbf9e64404e85bb169d5e570c48.png)

- 第四步解压OpenCV和TensorRT备用

![在这里插入图片描述](https://img-blog.csdnimg.cn/c06463423cc74987931eec39546368cf.png)
安装cmake

![在这里插入图片描述](https://img-blog.csdnimg.cn/8ff7626aaeca4e69abc4ce7311b52987.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/5fea0db6e58e4958b237cfe0cfbf658c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/945172b6770449f79d1242ca755cf826.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/574af970efbf4292a9cf3c851bd70c30.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bcb21adae8584966b46207dd2fea8d78.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/950df7a217a241fc8fcd1cd2e31cee98.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1f95c9c051774526a3b9d95012f71339.png)

- 第五步onnx转tensorrt引擎，流程如下：

从GitHub下载部署代码https://github.com/zhiqwang/yolov5-rt-stack
进入如下目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/b74250b897fb401cb157f36bb4077c94.png)

编辑CMakeLists，设置OpenCV和TensorRT目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/9e7c1b153052459d9bbea044ccd8d846.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/56fd024f43f14fa8a28d2775ac343176.png)

指定CMakeLists.txt目录以及要build的目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/501e5f3611764eedb9977cd39aab8505.png)

点击cmake的tools->Configure

![在这里插入图片描述](https://img-blog.csdnimg.cn/0ad63c11f1564fc6afcfcc6953900b62.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3c710bfc5d7c4b8e9a56b88a8e597e69.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/733e779250a8440383d9a253d623cf3f.png)

进入build目录，打开build工程

![在这里插入图片描述](https://img-blog.csdnimg.cn/fb981dd5407b4c71ac4abf8f11c4313e.png)

选择Release

![在这里插入图片描述](https://img-blog.csdnimg.cn/10e487d78c3f4540a7a42258608a41d5.png)

点击生成->生成解决方案，编译完成后进入生成的exe目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/a263acb6f13046ff9ff49f1788f0d299.png)
把tensorrt的dll文件放到exe目录下

![在这里插入图片描述](https://img-blog.csdnimg.cn/54b4ab29d6264e0ca7b5718a79fa285f.png)

进入C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\bin目录

把以下8个dll拷贝到exe目录

- cublas64_11.dll
- cublasLt64_11.dll
- cudart64_110.dll
- cudnn_cnn_infer64_8.dll
- cudnn_ops_infer64_8.dll
- cudnn64_8.dll
- nvrtc64_110_0.dll
- nvrtc-builtins64_110.dll

![在这里插入图片描述](https://img-blog.csdnimg.cn/d2dbe0169523493b99804f440c5ce840.png)

把官方的提供的onnx模型放到exe目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/852fe6e8a56f47e58484cdc39bc88727.png)

在exe目录打开命令行，输入build_model.exe yolov6n.onnx yolov6n.engine生成tensorrt引擎

- 第六步tensorrt引擎推理流程如下：

进入D:\yolov6\yolov5-rt-stack\deployment\tensorrt-yolov6目录

编辑CMakeLists，设置OpenCV和TensorRT目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/9acee4ef1cc147868ef19621143341b5.png)

打开桌面上的cmake,设置tensorrt推理代码的CMakeLists.txt路径以及要build的目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/eb26130e53934da1a2a3847769b04637.png)

点击cmake的tools->Configure

![在这里插入图片描述](https://img-blog.csdnimg.cn/f1567862c9314966b86711f0b58d895e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/78707a0bbfac4a6ba8f63fb413a733a6.png)

点击Generate

![在这里插入图片描述](https://img-blog.csdnimg.cn/ffe7baf1dc3540ed9b9d7bcf2bdfc224.png)

打开build工程，选择release，点击生成->生成解决方案，生成tensorrt推理的exe

![在这里插入图片描述](https://img-blog.csdnimg.cn/17c45809a65e47c1851a3bb6bff33f02.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/4dbb77d330dd4289b485280169ff4c38.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/40842389b2384f2289123f49628d319b.png)

进入生成tensorrt引擎的exe目录（上一次build的目录）

![在这里插入图片描述](https://img-blog.csdnimg.cn/7cc9c0bb104b4fb7ad437516cb9d1952.png)

拷贝dll以及tensorrt推理引擎到tensorrt推理的exe目录（当前build的目录）

![在这里插入图片描述](https://img-blog.csdnimg.cn/2e5828263bb94b54918f403d4e7b9940.png)

把OpenCV的opencv_world411.dll拷贝到推理exe目录

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe5d62bdba114427bebfe739c8e22041.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/d033fcd0b1a0493080db19022c093386.png)

在推理exe目录打开命令行，输入推理命令

yolov6.exe -model_path yolov6n.engine -image_path zidane.jpg

指定推理引擎路径以及推理图片路径，在推理exe目录生成推理可视化结果

![在这里插入图片描述](https://img-blog.csdnimg.cn/2288b88e00fb46f190bb7a4f7f4bc405.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/328b16344c69456bab9f58412d4a2533.png)