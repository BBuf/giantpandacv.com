# 前言
前几天加了两个Openvino群，准备请教一下关于Openvino对YOLOv3-tiny的int8量化怎么做的，没有得到想要的答案。但缺发现有那么多人Openvino并没有用好，都是在网络上找资料，我百度了一下中文似乎没有靠谱的目标检测算法的部署资料，实际上这个并不难，用官方提供的例子改一改就可以出来。所以我答应了几个同学写一个部署流程希望对想使用Openvino部署YOLOv3-tiny(其他目标检测算法类推)到cpu或者Intel神经棒上(1代或者2代)都是可以的。

# YOLOv3-tiny模型训练
这部分我就不过多介绍了，我使用的是AlexeyAB版本darknet训练的YOLOv3-tiny模型(地址见附录)，得到想要的weights文件，并调用命令测试图片的检测效果无误。具体训练过程可以看我之前写的一篇博客，地址放附录了。
# Darknet模型转pb模型
- 克隆OpenVINO-YoloV3 工程，完整地址见附录。
- 修改工程下面的coco.names改成和自己训练的时候一样。
- 确保你要使用的python环境有tensorflow版本，1.8和1.9应该都没什么问题。
- 执行：

```python
python3 convert_weights_pb.py 
--class_names voc.names 
--weights_file yolov3_tiny_200000.weights
--data_format NHWC 
--tiny --output_graph frozen_tiny_yolo_v3.pb
```
- 不出意外会在你的OpenVINO-YoloV3文件下生成了frozen_tiny_yolo_v3.pb文件，这个文件就是我们需要的pb文件。

# 在Windows上将pb文件转换为IR模型
我这里使用了OpenVINO2019.1.087，只要OpenVINO某个版本里面extension模块包含了YOLORegion Layer应该都是可以的。转换步骤如下：
- 拷贝frozen_tiny_yolo_v3.pb到OpenVINO所在的`F:\IntelSWTools\openvino_2019.1.087\deployment_tools\model_optimizer`文件夹下，注意这个文件夹是我安装OpenVINO的路径，自行修改一下即可。
- 新建一个yolov3-tiny.json文件，放在`F:\IntelSWTools\openvino_2019.1.087\deployment_tools\model_optimizer`文件夹下。内容是，注意一下里面classes是你的数据集中目标类别数：

```python
[
  {
    "id": "TFYOLOV3",
    "match_kind": "general",
    "custom_attributes": {
      "classes": 3,
      "coords": 4,
      "num": 6,
      "mask": [0,1,2],
      
      "anchors":[10,14,23,27,37,58,81,82,135,169,344,319],
      "entry_points": ["detector/yolo-v3-tiny/Reshape","detector/yolo-v3-tiny/Reshape_4"]
    }
  }
]
```
- 在`F:\IntelSWTools\openvino_2019.1.087\deployment_tools\model_optimizer`文件夹下，执行下面的命令来完成pb文件到OpenVINO的IR文件转换过程。

```python
python mo_tf.py --input_model frozen_darknet_yolov3_model.pb 
--tensorflow_use_custom_operations_config yolo_v3_tiny.json 
--input_shape=[1,416,416,3]  --data_type=FP32
```
- 不出意外的话就可以获得`frozen_darknet_yolov3_model.bin`和`frozen_darknet_yolov3_model.xml`了。

# 利用VS2015配合OpenVINO完成YOLOv3-tiny的前向推理

因为yolov3-tiny里面的yoloRegion Layer层是openvino的扩展层，所以在vs2015配置`lib`和`include`文件夹的时候需要把`cpu_extension.lib`和`extension文件夹`加进来。最后`include`和`lib`文件夹分别有的文件如下：
- include文件夹：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209111316707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- lib文件夹：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209111350792.png)
其中`cpu_extension.lib`在安装了OpenVINO之后可能是没有的，这时候就需要手动编译一下。这个过程很简单，我在后边放了一个链接讲得很清楚了。

把`include`和`lib`配置好之后就可以编写代码进行预测了。代码只需要在OpenVINO-YoloV3工程的cpp目录下提供的main.cpp稍微改改就可以了。因为我这里使用的不是原始的Darknet，而是AlexeyAB版本的darknet，所以图像resize到`416`的时候是直接resize而不是`letter box`的方式。具体来说修改部分的代码为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209112834301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
然后除了这个地方，由于使用的YOLOv3-tiny，OpenVINO-YoloV3里面的cpp默认使用的是YOLOv3的Anchor，所以Anchor也对应修改一下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191209113050831.png)这两个地方改完之后就可以顺利完成前向推理过程了，经过我的测试，相比于原始的darknet测试结果在小数点后两位开始出现差距，从我在几千张图片的测试结果来看，精度差距在1/1000到1/500，完全是可以接受的。

注意github上面的cpp那些打印信息都是没有必要可以注释掉，然后异步策略在测试时候也可以选择不用，我改好了一个可以直接运行的cpp，如果需要可以关注我的微信公众号回复"交流群"入群获取（没有引号）。当然加我微信也是可以的，微信名：hellotopython。

# 附录
- AlexAB版本Darknet：https://github.com/AlexeyAB/darknet
- 利用Darket 和YOLOV3训练自己的数据集(制作VOC)：https://blog.csdn.net/just_sort/article/details/81389571
- OpenVINO-YoloV3: https://github.com/PINTO0309/OpenVINO-YoloV3 
- Windows10下使用OpenVINO手动编译cpu_extension.lib： https://www.jianshu.com/p/32d12abc6e6a


# 后记
本文详细介绍了将AlexAB版本Darknet框架下训练的YOLOv3-tiny模型通过OpenVINO部署的完整流程，希望可以帮助到大家。

# 维护了一个微信公众号，分享论文，算法，比赛，生活，欢迎加入。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019102917521565.jpg)