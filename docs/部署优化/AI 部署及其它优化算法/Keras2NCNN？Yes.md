> 本文首发于GiantPandaCV公众号
# 1. 前言
这篇文章是记录笔者最近想尝试将自己开发的分割工程模型利用NCNN部署所做的一些工作，经过一些尝试和努力算是找到了一种相对简单的方法。因此这篇文章将笔者的工作分享出来，希望对使用Keras训练模型但苦于无法部署到移动端，或者使用Keras模型通过ONNX转到其它推理框架时碰到各种OP支持无法解决的读者带来些许帮助。

# 2. 转换路线
我的转换路线为：

Keras->Caffe->NCNN

首先Caffe->NCNN是NCNN默认支持的，所以不需要我做任何工作，所以我的工作主要就是Keras->Caffe。

然后我们来看一下Keras的HDF5模型的内存排布方式以及Caffe模型的内存排布方式。

## 2.1 Caffe模型内存排布方式

Caffe使用Blob结构在CNN网络中存储、传递数据。对于批量2D图像数据，Blob的维度为

图像数量N × 通道数C × 图像高度H × 图像宽度W

简写为[N, C, H, W]

## 2.2 Keras模型内存排布方式

笔者的环境为TF1.13.1/Keras 2.2.4，即Keras的后端仍为TF1.x，如果你是使用TF2.0也不要紧，因为TF2.0也可以将模型保存为HDF5的形式，所以仍然可以沿用本文介绍的方法。

然后Keras的Tensor的内存排布方式分为两种，一种是[N,H,W,C]，另外一种为[N,C,H,W]，默认是[N,H,W,C]，这里以默认的内存排布方式为例，我的代码库训练出来的UNet分割模型也是这种方式。

## 2.3 HDF5数据文件简介

Keras的模型保存方式为HDF5文件，HDF5全称Hierarchical Data Format，是美国伊利诺伊大学厄巴纳-香槟分校 UIUC (University of Illinois at Urbana-Champaign) 开发的，是一种跨平台的数据存储文件，然后Keras的模型一般保存为这种文件。一种最简单的理解是可以把hdf5文件看成一个字典，它会保存Keras搭建的CNN的每一层的名字，类型，配置参数，权重，参数等，我们可以通过访问字典的方式获得这些信息。Keras的HDF5模型解析是比较简单的，最后我们只需要将网络层的参数以及权重写进Caffe的模型和权重就可以了。

# 3. Keras2Caffe
Keras2Caffe的工具开源在：https://github.com/BBuf/Keras-Semantic-Segmentation/tree/master/tools，目前支持的Op如下：

- InputLayer
- Conv2D/Convolution2D
- Conv2DTranspose
- DepthwiseConv2D
- SeparableConv2D
- BatchNormalization
- Dense
- ReLU
- ReLU6
- LeakyReLU
- SoftMax
- SigMoid
- Cropping2D
- Concatenate
- Merge
- Add
- Flatten
- Reshape
- MaxPooling2D
- AveragePooling2D
- Dropout
- GlobalAveragePooling2D
- UpSampling2D
- ...

目前已支持的网络如下：

- VGG16
- SqueezeNet
- InceptionV3
- InceptionV4
- Xception V1
- UNet
- ...



# 4. 使用方法

进入`https://github.com/BBuf/Keras-Semantic-Segmentation/tree/master/tools`：

## 4.1 依赖
```
python3 
Pycaffe 需要自己编译，CPU/GPU版均可
```

## 4.2 转换命令

```
python convert2caffe.py xxx
```

- `--model_name` 字符串类型，代表测试时使用哪个模型，支持`enet`,`unet`,`segnet`,`fcn8`等多种模型，默认为`unet`。
- `--n_classes` 整型，代表分割图像中有几种类别的像素，默认为`2`。
- `--input_height`整型，代表要分割的图像需要`resize`的长，默认为`224`。
- `--input_width` 整型，代表要分割的图像需要`resize`的宽，默认为`224`。
- `--input_model` 字符串类型，代表模型的输入路径，如`../weights/unet.05.xxx.hdf5`。
- `--output_model` 字符串类型，代表转换后的caffe模型输出的文件夹路径，默认为`./unet.prototxt`，即当前目录。
- `--output_weights` 字符串类型，代表转换后的caffe模型名字，默认为`./unet.weights`。

## 4.3 Caffe模型升级

如果是旧版Caffe模型，需要在Caffe环境中转换为新版Caffe模型，调用如下命令：

```sh
 ~/caffe/build/tools/upgrade_net_proto_text unet.prototxt new.prototxt

 ~/caffe/build/tools/upgrade_net_proto_binary unet.caffemodel new.caffemodel
```


## 4.4 转换为NCNN模型

编译NCNN，执行模型转换命令：

```sh
~/ncnn/build/tools/caffe/caffe2ncnn new.prototxt new.caffemodel new.param new.bin
```

# 5. 使用NCNN进行推理
## 5.1 编写CmakeLists.txt

```sh
# 设置cmake版本，如果cmake版本过高，可能会出现错误
cmake_minimum_required(VERSION 3.5.1)
project(NCNN_test)

# 设置C++编译版本
set(CMAKE_CXX_STANDARD 11)

# 设置程序编译出的可执行文件
set(MAIN_FILE main.cpp)
set(EXECUTABLE_OUTPUT_PATH ./)

# 分别设置ncnn的链接库和头文件
set(NCNN_LIBS /home/pc/bbuf_project/ncnn/build/install/lib/libncnn.a)
set(NCNN_INCLUDE_DIRS /home/pc/bbuf_project/ncnn/build/install/include/ncnn)

# 配置OpenMP
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# 配置OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${NCNN_INCLUDE_DIRS})

# 建立链接依赖
add_executable(NCNN_SEG main.cpp)
target_link_libraries(NCNN_SEG ${NCNN_LIBS})
target_link_libraries(NCNN_SEG ${OpenCV_LIBS})
```

## 5.2 编写NCNN推理代码

```cpp
#include "net.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <map>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define INPUT_WIDTH     224
#define INPUT_HEIGHT    224

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("illegal parameters!");
        exit(0);
    }

    ncnn::Net Unet;

    Unet.load_param("/home/pc/Keras-Semantic-Segmentation-master/ncnn/models/unet.param");
    Unet.load_model("/home/pc/Keras-Semantic-Segmentation-master/ncnn/models/unet.bin");

    int64 tic, toc;

    tic = cv::getTickCount();

    cv::Scalar value = Scalar(0,0,0);
    cv::Mat src;
    cv::Mat tmp;
    src = cv::imread(argv[1]);
    float width = src.size().width;
    float height = src.size().height;
    int top = 0, bottom = 0;
    int left = 0, right = 0;

    if (width > height) {
        top = (width - height) / 2;
        bottom = (width - height) - top;
        cv::copyMakeBorder(src, tmp, top, bottom, 0, 0, BORDER_CONSTANT, value);
    } else {
        left = (height - width) / 2;
        right = (height - width) - left;
        cv::copyMakeBorder(src, tmp, 0, 0, left, right, BORDER_CONSTANT, value);
    }

    top = (INPUT_HEIGHT*top)/width;
    bottom = (INPUT_HEIGHT*bottom)/width;
    left = (INPUT_WIDTH*left)/height;
    right = (INPUT_WIDTH*right)/height;

    std::cout << "top " << top << " bottom " << bottom << " left " << left << " right " << right << std::endl;

    cv::Mat tmp1;
    cv::resize(tmp, tmp1, cv::Size(INPUT_WIDTH, INPUT_HEIGHT), CV_INTER_CUBIC);

    cv::Mat image;
    tmp1.convertTo(image, CV_32FC3);

    std::cout << "image element type "<< image.type() << " " << image.cols << " " << image.rows << std::endl;

    // cv32fc3 的布局是 hwc ncnn的Mat布局是 chw 需要调整排布
    float *srcdata = (float*)image.data;
    float *data = new float[INPUT_WIDTH*INPUT_HEIGHT*3];
    for (int i = 0; i < INPUT_HEIGHT; i++)
       for (int j = 0; j < INPUT_WIDTH; j++)
           for (int k = 0; k < 3; k++) {
              data[k*INPUT_HEIGHT*INPUT_WIDTH + i*INPUT_WIDTH + j] = srcdata[i*INPUT_WIDTH*3 + j*3 + k]/255.0;
           }
    ncnn::Mat in(image.rows*image.cols*3, data);
    in = in.reshape(image.rows, image.cols, 3);

    ncnn::Extractor ex = Unet.create_extractor();

    ex.set_light_mode(true);
    //ex.set_num_threads(4);

    ex.input("data", in);

    ncnn::Mat mask;
    ex.extract("reshape_1_activation_21", mask);

    {
        toc = cv::getTickCount() - tic;

        double time = toc / double(cv::getTickFrequency());
        double fps = double(1.0) / time;
        std::cout << "fps:" << fps << std::endl;
    }

    std::cout << "whc " << mask.w << " " << mask.h << " " << mask.c << std::endl;
#if 1
    cv::Mat cv_img = cv::Mat::zeros(INPUT_WIDTH,INPUT_HEIGHT,CV_8UC1);
//    mask.to_pixels(cv_img.data, ncnn::Mat::PIXEL_GRAY);

    {
    float *srcdata = (float*)mask.data;
    unsigned char *data = cv_img.data;

    for (int i = 0; i < mask.h; i++)
       for (int j = 0; j < mask.w; j++) {
#if 1
         float tmp = srcdata[0*mask.w*mask.h+i*mask.w+j];
         int maxk = 0;
         for (int k = 0; k < mask.c; k++) {
           if (tmp < srcdata[k*mask.w*mask.h+i*mask.w+j]) {
             tmp = srcdata[k*mask.w*mask.h+i*mask.w+j];
             maxk = k;
           }
           //std::cout << srcdata[k*mask.w*mask.h+i*mask.w+j] << " ";
         }
         //cout << endl;
         data[i*INPUT_WIDTH + j] = maxk;

         if ((left > 0) && (right > 0) && ((j < left) || (j >= INPUT_WIDTH - right)))
           data[i*INPUT_WIDTH + j] = 0;

         if ((top > 0) && (bottom > 0) && ((i < top) || (i >= INPUT_HEIGHT - bottom)))
           data[i*INPUT_WIDTH + j] = 0;
#else
         if (srcdata[1*mask.w*mask.h+i*mask.w+j] > 0.999)
           data[i*INPUT_WIDTH + j] = 1;
         else
           data[i*INPUT_WIDTH + j] = 0;
#endif
       }
    }

    {
        toc = cv::getTickCount() - tic;

        double time = toc / double(cv::getTickFrequency());
        double fps = double(1.0) / time;
        std::cout << "fps:" << fps << std::endl;
    }

    cv_img *= 255;
    cv::imshow("test", cv_img);
    cv::waitKey();
#endif
    return 0;
}
```

# 6. 速度测试&效果展示

输入一张原始图像，看一下NCNN获得的推理结果：

![原图](https://img-blog.csdnimg.cn/20200721182622782.jpg)

![分割结果图](https://img-blog.csdnimg.cn/20200721182635935.jpg)

可以看到推理结果是正确的，下面来看一下在CPU上的速度测试：

| 平台                      | 分辨率  | 推理位宽 | FPS  | 线程数 |
| ------------------------- | ------- | -------- | ---- | ------ |
| Intel Xeon CPU E5-2678 v3 | 224x224 | FP32     | 3.01 | 1      |

这里只给了一个x86平台的BaseLine，速度是比较慢的，速度优化的工程就留给有需要的读者自己做了。


# 7. 欢迎Star
代码公开在 https://github.com/BBuf/Keras-Semantic-Segmentation 这个库中，感兴趣的可以来点个star。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)