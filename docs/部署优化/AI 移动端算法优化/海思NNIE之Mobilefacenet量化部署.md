> 本文首发于我的知乎：https://zhuanlan.zhihu.com/p/107548509

# 0. 前言
那就直接更新下整理好的代码链接吧！

`https://github.com/hanson-young/nniefacelib`

当您点进这篇文章，我想肯定不需要过多的去向您介绍华为海思35xx系列芯片的型号参数或者强大之处。另外这个教程也是建立已经配置好环境，并掌握Ruyi Studio的基本使用前提下的。如果还没有跑过其中的一些sample，网上也有一些教程，推荐看刘山老师的博客（地址为：`https://blog.csdn.net/avideointerfaces/article/details/88585654`）。

这篇文章的目录如下：

- 简介
- 目录结构
- mobilefacenet.cfg文件的配置
- 生成NNIE mk模型
- Vector Comparision
- NNIE mobilefacenet板上特征提取
- 附录

# 1. 简介
海思35xx系列芯片对比起nvidia TX2、Intel Movidius神经计算棒等一众边缘计算产品，有其惊艳的地方，因其集成了强大的算力模块，集成度和功能模块齐全，最重要的是成本低，成为了安防行业的首选芯片。但是也有一些麻烦的地方，主要是在于其开发难度的提高，大家都是摸着石头过河（3288老玩家转行也是能体会到痛苦的）。在转自己的模型时，坑比想象的要多，并且海思官方SDK也存在一些错误之处，让人很难捉摸，所以有时候需要自己多去独立思考。这次我记录了在转换人脸识别模型mobilefacenet（`https://github.com/deepinsight/insightface`）下了比较坑的三个点，毕竟是个新玩意儿，多半是版本发布时候不统一造成的：

- CNN_convert_bin_and_print_featuremap.py 代码出现错误，cfg中的【image_list】这个字段并没有在代码中出现，代码中只有【image_file】，因此需要修改这一地方。
- CNN_convert_bin_and_print_featuremap.py和Get Caffe Output这里的预处理方式都是先乘以【data_scale】，再减均值【mean_file】，而在量化生成 .mk 文件时却是先减均值再乘以scale的。
- 量化需要使用多张图片，而CNN_convert_bin_and_print_featuremap.py各层产生的feature仅仅是一张图片，这在做【Vector Comparision】时候就难以清楚的明白到底最后mk文件是第几张图像。

# 2. 目录结构
![在Ruyi Studio中量化mobilefacenet的目录结构](https://img-blog.csdnimg.cn/20200530154354447.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 3. mobilefacenet.cfg文件的配置
可以从github上下载mxnet2caffe的mobilefacenet模型（`https://github.com/honghuCode/mobileFacenet-ncnn/tree/feature/mobilefacenet-mxnet2caffe`），首先需要修改mobilefacenet.prototxt（`https://github.com/honghuCode/mobileFacenet-ncnn/blob/feature/mobilefacenet-mxnet2caffe/mobilefacenet.prototxt`）的输入层以符合NNIE caffe网络的结构标准：

![更改前](https://img-blog.csdnimg.cn/20200530154625107.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![更改后](https://img-blog.csdnimg.cn/20200530154635651.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
而量化mk使用的【mean_file】pixel_mean.txt是特别需要注意的

![pixel_mean.txt](https://img-blog.csdnimg.cn/20200530154651101.png)
我从agedb_30人脸数据库里面挑选了10张图像来做量化处理，为什么需要多张量化，请参考文章`https://zhuanlan.zhihu.com/p/58182172`，我们选择【10.jpg】来做 【Vector Comparision】，其实就是imageList.txt里的排列在最后的那张图片

![做量化的校验集](https://img-blog.csdnimg.cn/20200530154730952.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
具体配置如下：

![mobilefacenet.cfg文件的配置视图](https://img-blog.csdnimg.cn/20200530154755958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

```sh
[prototxt_file] ./mark_prototxt/mobilefacenet_mark_nnie_20190723102335.prototxt
[caffemodel_file] ./data/face/mobilefacenet.caffemodel
[batch_num] 256
[net_type] 0
[sparse_rate] 0
[compile_mode] 0
[is_simulation] 0
[log_level] 3
[instruction_name] ./data/face/mobilefacenet_inst
[RGB_order] RGB
[data_scale] 0.0078125
[internal_stride] 16
[image_list] ./data/face/images/imageList20190723102419.txt
[image_type] 1
[mean_file] ./data/face/pixel_mean.txt
[norm_type] 5
```


# 4. 生成NNIE mk模型

```sh
Start [RuyiStudio Wk NNIE Mapper] [E:\Code\nnie\windows\RuyiStudio-2.0.31\workspace\HeilsFace\mobilefacenet.cfg] HeilsFace (2019-07-23 10:48:17)
Mapper Version 1.1.2.0_B050 (NNIE_1.1) 1812171743151709
begin net parsing....
.end net parsing
begin prev optimizing....
....end prev optimizing....
begin net quantalizing(GPU)....


....................**********************************************************
WARNING: file: Inference::computeNonlinearQuantizationDelta  line: 92
data containing only zeros; set max value to 1e-6.
**********************************************************
WARNING: file: Inference::computeNonlinearQuantizationDelta  line: 92
data containing only zeros; set max value to 1e-6.
.......................................


end quantalizing
begin optimizing....
.end optimizing
begin NNIE[0] mem allocation....
...end NNIE[0] memory allocating
begin NNIE[0] instruction generating....
.............end NNIE[0] instruction generating
begin parameter compressing....
.end parameter compressing
begin compress index generating....
end compress index generating
begin binary code generating....
...................................................................................
...................................................................................
..................................................................................
...................................................................................
.............end binary code generating
begin quant files writing....
end quant files writing
===============E:\Code\nnie\windows\RuyiStudio-2.0.31\workspace\HeilsFace\mobilefacenet.cfg Successfully!===============
```

结束之后会生成：

- mobilefacenet_inst.wk文件
- mapper_quant文件夹，里面有量化输出的结果，如图 Fig.4.1，也就是./data/face/images/10.jpg

![Fig.4.1 [image_list]./data/face/images/imageList20190723102419.txt](https://img-blog.csdnimg.cn/20200530154948256.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**记住，mk量化过程在【mapper_quant】文件夹中生成的features是最后一张图片的inference结果，这也是文章最开始说的第三个存在问题的地方**


# 5. Vector Comparision
这一步，主要就是对比量化前后模型输出的精度损失，**最重要的就是要debug一遍CNN_convert_bin_and_print_featuremap.py**

因为这个脚本里确实藏了很多雷，我们先要比较原框架原模型inference的结果与这一脚本得出来的结果是否一致，如果存在不一致的情况，需要去核查一遍原因

文章开篇说到的第一个问题点 CNN_convert_bin_and_print_featuremap.py 中加载了mobilefacenet.cfg文件，**但脚本中并不存在【image_list】这个字段，取而代之的是【image_file】这个字段**

生成NNIE mk中，mobliefacenet.cfg 的【image_list】：

![Fig.5.1 生成NNIE mk中的mobliefacenet.cfg](https://img-blog.csdnimg.cn/20200530155058357.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
CNN_convert_bin_and_print_featuremap.py 中加载.cfg代码片段：

![CNN_convert_bin_and_print_featuremap.py 中加载.cfg代码片段：](https://img-blog.csdnimg.cn/20200530155117291.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
因此需要根据实际情况修改 mobliefacenet.cfg ，这里最好是复制一份新的，旧的用于生成NNIE wk，在复制后的mobliefacenet.cfg中修改一下：


![修改后的mobliefacenet.cfg](https://img-blog.csdnimg.cn/20200530155143281.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
另外，我们需要特别注意预处理这一个环节，如文章开篇所阐述的第二点

![预处理](https://img-blog.csdnimg.cn/20200530155237775.png)
我们注意到这里，data是uint8类型的array，是先乘以了【data_scale】的，也就是说和NNIE 生成wk中的操作顺序是不一致的。

**(data - 128.0) * 0.0078125 <==> data * 0.0078125 - 1**

因此这里需要做的修改就是需要将【mean_file】pixel_mean.txt修改为

![修改后的mean_file](https://img-blog.csdnimg.cn/20200530155257932.png)
修改完以上，然后直接运行代码，将最终模型提取的features fc1_output0_128_caffe.linear.float和caffe_forward.py（`https://github.com/honghuCode/mobileFacenet-ncnn/blob/feature/mobilefacenet-mxnet2caffe/caffe_forward.py`）中的进行比对，如果以上都没问题，可以看到结果是几乎一致的

caffe_forward.py生成的结果：

```sh
[-0.82475293 -0.33066949 -0.9848339   2.44199681  0.41715512  0.67809981
  0.29879519  1.14293635 -0.42905819  0.32940909 -1.20455348  1.01217067
  0.83146936 -0.84349883 -1.49177814 -0.91509151 -1.39441037  0.00413842
  0.97043389 -1.77688181  0.28639579 -1.06645989 -0.8570649  -2.09743094
 -0.1394622  -1.15035641 -0.81590587 -3.93798804 -0.35600579  1.90367532
  1.27935755 -2.07778478 -0.42563218  0.06624207  1.02597868 -0.52002895
 -0.905873   -0.41364694 -1.40032899 -1.37654066  0.03066693 -0.18659458
 -1.53931415 -0.55896652  2.42570448 -0.3044413   0.18183242  0.50442797
 -2.36735368 -0.12376076  0.15200013  0.13939141  0.56305337 -0.10047323
  1.50704932  0.05429612 -1.97527623 -0.75790995  1.89399767  0.56089604
 -2.34883094  0.22600658  1.00399816 -0.55099922  1.77083731  0.10722937
  2.21140814  0.06182361  0.03354079  0.97481596 -2.00423741  0.73168194
 -1.79977489 -0.85182911 -0.06020565 -0.14835797 -1.93012297 -3.09269047
 -0.60087907 -1.02915597  1.40985525  1.85411906 -1.21282506 -2.53264689
 -0.63467324 -1.15255475 -0.59994221  0.21181655  1.30336523 -1.73625863
  0.00861333  0.99906266  1.90666902  0.51179212  0.62143475  1.01997399
 -1.65181398  1.55190873  0.43448481 -0.85371047 -0.68216199  1.28038061
  0.4629558  -0.59671575  1.00122356  1.74233603  1.50384009  0.49827856
  0.67030573 -1.20388556  1.00168729 -0.71768999  1.06416941 -2.55346298
 -1.85579956 -2.18774438 -1.79652691  1.50856853  2.10628557  1.12313557
  2.76396179  0.60242128  0.0550903  -1.31998527 -0.6896565  -0.07160443
  1.21242583 -1.06733179]
```


CNN_convert_bin_and_print_featuremap.py生成的结果（由于特征值太多，就不一一打印出来了）：

![caffe_forward.py生成的结果：](https://img-blog.csdnimg.cn/20200530155526708.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**然后再生成，并进行【Vector Comparision】，量化终于成功了**

![中间层特征值比较图1](https://img-blog.csdnimg.cn/20200530155644744.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![中间层特征值比较图2](https://img-blog.csdnimg.cn/20200530155652426.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![中间层特征值比较图3](https://img-blog.csdnimg.cn/20200530155659125.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 6. NNIE mobilefacenet板上特征提取
做完了模型的量化，就可以进行仿真或者是在板子上进行实际测试了，这一步的坑并不是很多，主要还是得靠一些编程技巧了，建议熟悉C语言，这部分要熟悉sample代码，如果说非常熟悉c/c++混编，也可以使用c++。

## 6.1 修改例程
这里参考了`https://blog.csdn.net/u011728480/article/details/92069793`，其写法几乎一致，如下Fig.6.1 Fig.6.2是我所修改的代码片段，找到smp/a7_linux/mpp/sample/svp/nnie/sample/sample_nnie.c中该函数
```cpp
void SAMPLE_SVP_NNIE_Cnn(void)
```
只用修改了该函数的前后两处代码

![Fig.6.1 函数开头修改pcSrcFile和pcModeName](https://img-blog.csdnimg.cn/20200530155849263.png)
![Fig.6.2 函数结尾增加输出层的打印信息](https://img-blog.csdnimg.cn/20200530155857641.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
我们调用了 SAMPLE_SVP_NNIE_PrintReportResult 函数输出两个结果报表文件，结果分析当中会用到

```bash
seg0_layer38_output0_inst.linear.hex
seg0_layer3605_output0_inst.linear.hex
```

整段函数代码参见文章末尾【附录】


## 6.2 bgr文件的生成
注意到上文中我使用了pcSrcFile，这也是例程中主流的格式bgr，那么我们一般的图片都是.jpeg格式的，为了更好的利用NNIE，所以就需要利用opencv来转化以下。

首先.bgr文件是可以由opencv Mat转换的，但完成转换代码的编写之前我们必须清楚像素的空间排列顺序。注意，以下转换代码简单采用像素复制，并没有考虑优化，运行会比较慢！参考博客

.bgr ==> BBBBBB...GGGGGG...RRRRRR

cv::Mat ==> BGRBGRBGR...BGRBGRBGR

**.bgr --> cv::Mat**

![Fig.6.3 .bgr 转 mat](https://img-blog.csdnimg.cn/20200530155951657.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

```c++
/*bgr格式 转 cv::Mat代码 */
int bgr2mat(cv::Mat& img, int width, int height, int channel, const char* pth)
{
	if (pth)
	{
		FILE* fp;
		unsigned char *img_data = NULL;
		unsigned char *img_data_conv = NULL;
		img_data = (unsigned char*)malloc(sizeof(unsigned char) * width * height * channel);
		//unsigned char img_data[300 * 300 * 3];
		img_data_conv = (unsigned char*)malloc(sizeof(unsigned char) * width * height * channel);

		fp = fopen(pth, "rb");
		if (!fp)
		{
			return 0;
		}
		fread(img_data, 1, width * height * channel, fp);
		fclose(fp);

		for (size_t k = 0; k < channel; k++)
			for (size_t i = 0; i < height; i++)
				for (size_t j = 0; j < width; j++)
					img_data_conv[channel * (i * width + j) + k] = img_data[k * height * width + i * width + j];
		img = cv::Mat(height, width, CV_8UC3, img_data_conv);
		//free(img_data_conv);
		//img_data_conv = NULL;
		free(img_data);
		img_data = NULL;
		return 1;
	}
	return 0;
}
```

**cv::Mat -->.bgr** 

![Fig.6.4 mat转.bgr](https://img-blog.csdnimg.cn/20200530160020177.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

```c++
/*cv::Mat 转 bgr格式代码 */
int mat2bgr(cv::Mat& img, const char* bgr_path)
{
	if (bgr_path)
	{
		FILE* fp = fopen(bgr_path, "wb");
		int step = img.step;
		int h = img.rows;
		int w = img.cols;
		int c = img.channels();
		std::cout << step<< std::endl;
		for (int k = 0; k < c; k++)
			for (int i = 0; i < h; i++)
				for (int j = 0; j < w; j++)
				{
					//两种写法
					//fwrite(&img.data[i*step + j * c + k], sizeof(uint8_t), 1, fp);
					fwrite(&img.data[c*(i * w + j) + k], sizeof(uint8_t), 1, fp);
				}
		fclose(fp);
		//cv::Mat tmp;
		//bgr2mat(tmp, w, h, 3, bgr_path);
		//cv::imshow("tmp", tmp);
		//cv::waitKey(0);
		return 1;
	}
	return 0;
}
```

## 6.3 模型额外问题

pc上运行

E:\Code\nnie\software\sample_simulator\Release\sample_simulator.exe

板上运行

/nfsroot/Hi3516CV500_SDK_V2.0.1.0/smp/a7_linux/mpp/sample/svp/nnie # ./sample_nnie_main 4

可能会出现如下（Fig.6.5，Fig.6.6）错误，原因是生成NNIE wk文件的mapper工具有版本要求，下面错误当中使用的nnie mapper 版本是V1.1.2.0，而指令仿真或者是板上的SDK是V1.2的，解决办法就是使用nnie mapper V1.2版本重新生成一下wk模型，如（Fig.6.7），生成inst/chip.wk的时间比较久，在我机器上大概要2个小时，因为inst.wk实际上是需要进行参数压缩和二进制代码生成，这可能也是inst.mk比func.wk文件大的原因（如Fig.6.8），而生成func.wk的时间会比较短，建议在PC上调试的时候选择func/simulation模型

![Fig.6.5 PC运行仿真例程sample_simulator会出现该log](https://img-blog.csdnimg.cn/20200530160104437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![Fig.6.6 板上测试SDK修改的例程](https://img-blog.csdnimg.cn/2020053016011611.png)
![Fig.6.7 改变工程依赖的NNIE版本为指定芯片](https://img-blog.csdnimg.cn/20200530160127275.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![Fig.6.8 模型尺寸比较](https://img-blog.csdnimg.cn/20200530160139775.png)
![HISI 开发文档的提示](https://img-blog.csdnimg.cn/20200530160149255.png)

## 6.4 运行结果及分析
修改完sample_nnie.c中的代码后，在宿主机上进行make，然后到海思板子上运行可执行文件即可

![Fig.6.9 板上运行结果](https://img-blog.csdnimg.cn/20200530160241112.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
拷贝出生成的两个打印报表文件到Ruyi studio，进行比对测试

```bash
seg0_layer38_output0_inst.linear.hex
seg0_layer3605_output0_inst.linear.hex
```

如Fig.6.10，Fig.6.11，虽然说板上和仿真情况下还是会有一定的差别，但总体的误差是比较小的，基本可以接受，如果无法接受，可以尝试int16模型

![结果向量的对比](https://img-blog.csdnimg.cn/20200530160314681.png)
![Fig.6.10 量化模型在板子上的输出结果和pc上的结果比对（cosine similarity 99.6）](https://img-blog.csdnimg.cn/20200530160336993.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![Fig.6.11 无量化caffe输出与板上量化输出比对（cosine similarity  99.1）](https://img-blog.csdnimg.cn/20200530160405228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 7. 附录

```c++
void SAMPLE_SVP_NNIE_Cnn(void)
{
    HI_CHAR *pcSrcFile = "./data/nnie_image/rgb_planar/10.bgr";
    HI_CHAR *pcModelName = "./data/nnie_model/face/mobilefacenet_inst.wk";
    HI_U32 u32PicNum = 1;
    HI_S32 s32Ret = HI_SUCCESS;
    SAMPLE_SVP_NNIE_CFG_S   stNnieCfg = {0};
    SAMPLE_SVP_NNIE_INPUT_DATA_INDEX_S stInputDataIdx = {0};
    SAMPLE_SVP_NNIE_PROCESS_SEG_INDEX_S stProcSegIdx = {0};

    /*Set configuration parameter*/
    stNnieCfg.pszPic= pcSrcFile;
    stNnieCfg.u32MaxInputNum = u32PicNum; //max input image num in each batch
    stNnieCfg.u32MaxRoiNum = 0;
    stNnieCfg.aenNnieCoreId[0] = SVP_NNIE_ID_0;//set NNIE core
    s_stCnnSoftwareParam.u32TopN = 5;

    /*Sys init*/
    SAMPLE_COMM_SVP_CheckSysInit();

    /*CNN Load model*/
    SAMPLE_SVP_TRACE_INFO("Cnn Load model!\n");
    s32Ret = SAMPLE_COMM_SVP_NNIE_LoadModel(pcModelName,&s_stCnnModel);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_COMM_SVP_NNIE_LoadModel failed!\n");

    /*CNN parameter initialization*/
    /*Cnn software parameters are set in SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit,
     if user has changed net struct, please make sure the parameter settings in
     SAMPLE_SVP_NNIE_Cnn_SoftwareParaInit function are correct*/
    SAMPLE_SVP_TRACE_INFO("Cnn parameter initialization!\n");
    s_stCnnNnieParam.pstModel = &s_stCnnModel.stModel;
    s32Ret = SAMPLE_SVP_NNIE_Cnn_ParamInit(&stNnieCfg,&s_stCnnNnieParam,&s_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_ParamInit failed!\n");

    /*record tskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_AddTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_AddTskBuf failed!\n");

    /*Fill src data*/
    SAMPLE_SVP_TRACE_INFO("Cnn start!\n");
    stInputDataIdx.u32SegIdx = 0;
    stInputDataIdx.u32NodeIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_FillSrcData(&stNnieCfg,&s_stCnnNnieParam,&stInputDataIdx);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_FillSrcData failed!\n");

    /*NNIE process(process the 0-th segment)*/
    stProcSegIdx.u32SegIdx = 0;
    s32Ret = SAMPLE_SVP_NNIE_Forward(&s_stCnnNnieParam,&stInputDataIdx,&stProcSegIdx,HI_TRUE);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Forward failed!\n");

    /*Software process*/
    /*if user has changed net struct, please make sure SAMPLE_SVP_NNIE_Cnn_GetTopN
     function's input datas are correct*/
    s32Ret = SAMPLE_SVP_NNIE_Cnn_GetTopN(&s_stCnnNnieParam,&s_stCnnSoftwareParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_CnnGetTopN failed!\n");

    /*Print result*/
    SAMPLE_SVP_TRACE_INFO("Cnn result:\n");
    s32Ret = SAMPLE_SVP_NNIE_Cnn_PrintResult(&(s_stCnnSoftwareParam.stGetTopN),
        s_stCnnSoftwareParam.u32TopN);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_1,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,SAMPLE_SVP_NNIE_Cnn_PrintResult failed!\n");

    /*Print results*/
    {
        printf("features:\n{\n");
        printf("stride: %d\n",s_stCnnNnieParam.astSegData[0].astDst[0].u32Stride);
        printf("blob type :%d\n",s_stCnnNnieParam.astSegData[0].astDst[0].enType);
        printf("{\n\tc :%d", s_stCnnNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Chn);
        printf("\n\th :%d", s_stCnnNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Height);
        printf("\n\tw :%d \n}\n", s_stCnnNnieParam.astSegData[0].astDst[0].unShape.stWhc.u32Width);
        HI_S32* ps32Score = (HI_S32* )((HI_U8* )s_stCnnNnieParam.astSegData[0].astDst[0].u64VirAddr);
        printf("blobs fc1:\n[");
        for(HI_U32 i = 0; i < 128; i++)
        {
            printf("%f ,",*(ps32Score + i) / 4096.f);
        }
        
        printf("]\n}\n");
    }
    s32Ret = SAMPLE_SVP_NNIE_PrintReportResult(&s_stCnnNnieParam);
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret, CNN_FAIL_1, SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,SAMPLE_SVP_NNIE_PrintReportResult failed!");

CNN_FAIL_1:
    /*Remove TskBuf*/
    s32Ret = HI_MPI_SVP_NNIE_RemoveTskBuf(&(s_stCnnNnieParam.astForwardCtrl[0].stTskBuf));
    SAMPLE_SVP_CHECK_EXPR_GOTO(HI_SUCCESS != s32Ret,CNN_FAIL_0,SAMPLE_SVP_ERR_LEVEL_ERROR,
        "Error,HI_MPI_SVP_NNIE_RemoveTskBuf failed!\n");

CNN_FAIL_0:
    SAMPLE_SVP_NNIE_Cnn_Deinit(&s_stCnnNnieParam,&s_stCnnSoftwareParam,&s_stCnnModel);
    SAMPLE_COMM_SVP_CheckSysExit();
}
```