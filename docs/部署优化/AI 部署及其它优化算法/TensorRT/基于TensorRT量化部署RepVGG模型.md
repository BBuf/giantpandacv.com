【GiantPandaCV导语】本文为大家介绍了一个TensorRT int8 量化部署 RepVGG 模型的教程，并开源了全部代码。主要是教你如何搭建tensorrt环境，对pytorch模型做onnx格式转换，onnx模型做tensorrt int8量化，及对量化后的模型做推理，实测在1070显卡做到了不到1ms一帧！

# 0x0. RepVGG简介
一个简单但功能强大的卷积神经网络架构，该架构具有类似于VGG的推理时间主体，该主体仅由3×3卷积和ReLU的堆栈组成，而训练时间模型具有多分支拓扑。训练时间和推理时间体系结构的这种解耦是通过结构性重新参数化技术实现的，因此模型是名为RepVGG。论文解读可以看：[RepVGG|让你的ConVNet一卷到底，plain网络首次超过80%top1精度](https://mp.weixin.qq.com/s/g-LnAtOnmGYX4onGZMRkhQ)

开源代码地址如下：https://github.com/Wulingtian/RepVGG_TensorRT_int8

# 0x1. 环境配置

- ubuntu：18.04
- cuda：11.0
- cudnn：8.0
- tensorrt：7.2.16
- OpenCV：3.4.2
- cuda，cudnn，tensorrt和OpenCV安装包（编译好了，也可以自己从官网下载编译）可以从链接: https://pan.baidu.com/s/1Nl5XTAsUOyTbY6VbigsMNQ 密码: c4dn
- **cuda安装**
- 如果系统有安装驱动，运行如下命令卸载
- sudo apt-get purge nvidia*
- 禁用nouveau，运行如下命令
- sudo vim /etc/modprobe.d/blacklist.conf
- 在末尾添加
- blacklist nouveau
- 然后执行
- sudo update-initramfs -u
- chmod +x cuda_11.0.2_450.51.05_linux.run
- sudo ./cuda_11.0.2_450.51.05_linux.run 
- 是否接受协议: accept 
- 然后选择Install 
- 最后回车
- vim ~/.bashrc 添加如下内容：
- export PATH=/usr/local/cuda-11.0/bin:$PATH
- export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
- source .bashrc 激活环境
- **cudnn 安装**
- tar -xzvf cudnn-11.0-linux-x64-v8.0.4.30.tgz
- cd cuda/include
- sudo cp *.h /usr/local/cuda-11.0/include
- cd cuda/lib64
- sudo cp libcudnn* /usr/local/cuda-11.0/lib64
- **tensorrt及OpenCV安装**
- 定位到用户根目录
- tar -xzvf TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-11.0.cudnn8.0.tar.gz 
- cd TensorRT-7.2.1.6/python，该目录有4个python版本的tensorrt安装包
- sudo pip3 install tensorrt-7.2.1.6-cp37-none-linux_x86_64.whl（根据自己的python版本安装）
- pip install pycuda 安装python版本的cuda
- 定位到用户根目录
- tar -xzvf opencv-3.4.2.zip 以备推理调用

# 0x2. RepVGG模型训练以及转换onnx

- 定位到用户根目录
- git clone https://github.com/Wulingtian/RepVGG_TensorRT_int8.git
- cd RepVGG_TensorRT_int8/models
- vim convert_model.py 
- 设置 num_classes，例如：我训练的是猫狗识别，则设置为2
- python convert_model.py 生成可加载的ImageNet预训练模型路径
- cd RepVGG_TensorRT_int8
- vim repvgg.py 定位到154行，修改类别数
- vim train.py 修改IMAGENET_TRAINSET_SIZE参数 指定训练图片的数量
- 根据自己的训练数据及配置设置data（数据集路径），arch（我选择的是最小的模型RepVGG-A0），epochs，lr，batch-size，model_path（设置ImageNet预训练模型路径，就是上面convert_model.py转换得到的模型）等参数
- python train.py，开始训练，模型保存在当前目录，名为model_best.pth.tar
- python convert.py model_best.pth.tar RepVGG-A0-deploy.pth -a RepVGG-A0（指定模型类型，我训练的是RepVGG-A0）
- vim export_onnx.py
- 设置arch，weights_file（convert.py生成的模型），output_file（输出模型名称），img_size（图片输入大小），batch_size（推理的batch）
- python export_onnx.py得到onnx模型

# 0x3. onnx模型转换为 int8 tensorrt引擎

- cd RepVGG_TensorRT_int8/repvgg_tensorrt_int8_tools
- vim convert_trt_quant.py 修改如下参数
- BATCH_SIZE 模型量化一次输入多少张图片
- BATCH 模型量化次数
- height width 输入图片宽和高
- CALIB_IMG_DIR 量化图片路径(把训练的图片放到一个文件夹下，然后把这个文件夹设置为此参数，注意BATCH_SIZE*BATCH要小于或等于训练图片数量）
- onnx_model_path onnx模型路径（上面运行export_onnx.py得到的onnx模型）
- python convert_trt_quant.py 量化后的模型存到models_save目录下

# 0x4. tensorrt模型推理

- cd RepVGG_TensorRT_int8/repvgg_tensorrt_int8
- cd yolov5_tensorrt_int8
- vim CMakeLists.txt
- 修改USER_DIR参数为自己的用户根目录
- vim repvgg_infer.cc修改如下参数
- output_name repvgg模型有1个输出
- 我们可以通过netron查看模型输出名
- pip install netron 安装netron
- vim netron_repvgg.py 把如下内容粘贴
    - import netron
	- netron.start('此处填充简化后的onnx模型路径', port=3344)
- python netron_repvgg.py 即可查看 模型输出名
- trt_model_path 量化的tensorrt推理引擎（models_save目录下trt后缀的文件）
- test_img 测试图片路径
- INPUT_W INPUT_H 输入图片宽高
- NUM_CLASS 训练的模型有多少类
- 参数配置完毕

```shell
mkdir build
cd build
cmake ..
make
```

./RepVGGsEngine 输出平均推理时间，实测平均推理时间小于1ms一帧，不得不说，RepVGG真的很香！至此，部署完成！
由于我训练的是猫狗识别下面放一张猫狗同框的图片结尾。

顺便放一下我的数据集链接
链接: https://pan.baidu.com/s/1Mh6GxTLoXRTCRQh-TPUc3Q 密码: 3dt3


![数据集中一张图片展示](https://img-blog.csdnimg.cn/20210204194700353.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)