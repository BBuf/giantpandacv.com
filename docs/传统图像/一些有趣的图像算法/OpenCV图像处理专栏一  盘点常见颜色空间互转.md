# 前言 
今天是OpenCV传统图像处理算法的第一篇，我们来盘点一下常见的6种颜色空间互转算法，并给出了一些简单的加速方案，希望可以帮助到学习OpenCV图像处理的同学。这6种算法分别是：
- RGB和GRAY互转
- RGB和YUV互转
- RGB和HSV互转
- RGB和HSI互转
- RGB和YCbCr互转
- RGB和YDbDr互转

# 算法原理和代码实现
## 一，RGBGG转GRAY
RGB是依据人眼识别的颜色定义出的空间，可表示大部分颜色。是图像处理中最基本、最常用、面向硬件的颜色空间，是一种光混合的体系。

RGB颜色空间最常用的用途就是显示器系统，彩色阴极射线管,彩色光栅图形的显示器都使用R、G、B数值来驱动R、G、B 电子枪发射电子，并分别激发荧光屏上的R、G、B三种颜色的荧光粉发出不同亮度的光线，并通过相加混合产生各种颜色。扫描仪也是通过吸收原稿经反射或透射而发送来的光线中的R、G、B成分，并用它来表示原稿的颜色。

首先是RGB2GRAY，也就是彩色图转灰度图的算法。RGB值和灰度的转换，实际上是人眼对于彩色的感觉到亮度感觉的转换，这是一个心理学问题，有一个公式：
Grey = 0.299$\times$R + 0.587$\times$G + 0.114$\times$B。直接计算复杂度较高，考虑优化可以将小数转为整数，除法变为移位，乘法也变为移位，但是这种方法也会带来一定的精度损失，我们可以根据实际情况选择需要保留的精度位数。下面给出不同精度(2-20位)的计算公式：

```
Grey = (R*1 + G*2 + B*1) >> 2

Grey= (R*2 + G*5 + B*1) >> 3

Grey= (R*4 + G*10 + B*2) >> 4

Grey = (R*9 + G*19 + B*4) >> 5

Grey = (R*19 + G*37 + B*8) >> 6

Grey= (R*38 + G*75 + B*15) >> 7

Grey= (R*76 + G*150 + B*30) >> 8

Grey = (R*153 + G*300 + B*59) >> 9

Grey = (R*306 + G*601 + B*117) >> 10

Grey = (R*612 + G*1202 + B*234) >> 11

Grey = (R*1224 + G*2405 + B*467) >> 12

Grey= (R*2449 + G*4809 + B*934) >> 13

Grey= (R*4898 + G*9618 + B*1868) >> 14

Grey = (R*9797 + G*19235 + B*3736) >> 15

Grey = (R*19595 + G*38469 + B*7472) >> 16

Grey = (R*39190 + G*76939 + B*14943) >> 17

Grey = (R*78381 + G*153878 + B*29885) >> 18

Grey =(R*156762 + G*307757 + B*59769) >> 19

Grey= (R*313524 + G*615514 + B*119538) >> 20
```
再给出保留20位精度的计算代码(使用了Openmp多线程优化)：

```
//RGB2GRAY优化
Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<uchar>(i, j) = ((src.at<Vec3b>(i, j)[0] << 18) + (src.at<Vec3b>(i, j)[0] << 15) + (src.at<Vec3b>(i, j)[0] << 14) +
				(src.at<Vec3b>(i, j)[0] << 11) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 7) + (src.at<Vec3b>(i, j)[0] << 5) +
				(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 2) +
				(src.at<Vec3b>(i, j)[1] << 19) + (src.at<Vec3b>(i, j)[1] << 16) + (src.at<Vec3b>(i, j)[1] << 14) + (src.at<Vec3b>(i, j)[1] << 13) +
				(src.at<Vec3b>(i, j)[1] << 10) + (src.at<Vec3b>(i, j)[1] << 8) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1) +
				(src.at<Vec3b>(i, j)[2] << 16) + (src.at<Vec3b>(i, j)[2] << 15) + (src.at<Vec3b>(i, j)[2] << 14) + (src.at<Vec3b>(i, j)[2] << 12) +
				(src.at<Vec3b>(i, j)[2] << 9) + (src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 1) >> 20);
		}
	}
	return dst;
}
```

## 二，RGB和YUV互转
首先介绍一下YUV颜色空间，YUV(亦称YCrCb)是被欧洲电视系统所采用的一种颜色编码方法。在现代彩色电视系统中，通常采用三管彩色摄像机或彩色CCD摄影机进行取像，然后把取得的彩色图像信号经分色、分别放大校正后得到RGB，再经过矩阵变换电路得到亮度信号Y和两个色差信号R-Y(即U)、B-Y(即V)，最后发送端将亮度和两个色差总共三个信号分别进行编码，用同一信道发送出去。这种色彩的表示方法就是所谓的YUV色彩空间表示。采用YUV色彩空间的重要性是它的亮度信号Y和色度信号U、V是分离的。如果只有Y信号分量而没有U、V信号分量，那么这样表示的图像就是黑白灰度图像。彩色电视采用YUV空间正是为了用亮度信号Y解决彩色电视机与黑白电视机的兼容问题，使黑白电视机也能接收彩色电视信号。
YUV主要用于优化彩色视频信号的传输，使其向后相容老式黑白电视。与RGB视频信号传输相比，它最大的优点在于只需占用极少的频宽（RGB要求三个独立的视频信号同时传输）。其中“Y”表示明亮度（Luminance或Luma），也就是灰阶值；而“U”和“V” 表示的则是色度（Chrominance或Chroma），作用是描述影像色彩及饱和度，用于指定像素的颜色。“亮度”是透过RGB输入信号来建立的，方法是将RGB信号的特定部分叠加到一起。“色度”则定义了颜色的两个方面─色调与饱和度，分别用Cr和Cb来表示。其中，Cr反映了RGB输入信号红色部分与RGB信号亮度值之间的差异。而Cb反映的是RGB输入信号蓝色部分与RGB信号亮度值之同的差异。
1，RGB转YUV

Y = 0.299R + 0.587G + 0.114B
U = -0.147R - 0.289G + 0.436B
V = 0.615R - 0.515G - 0.100B

2，YUV转RGB

R = Y + 1.14V
G = Y - 0.39U - 0.58V
B = Y + 2.03U
## 优化1：去掉浮点运算
基于这一点，我们做如下操作：

```
Y * 256 = 0.299 * 256R + 0.587 * 256G + 0.114 * 256B

U * 256 = -0.147 * 256R - 0.289 * 256G + 0.436 * 256B
V * 256 = 0.615 * 256R - 0.515 * 256G - 0.100 * 256B

R * 256 = Y * 256 + 1.14 * 256V
G * 256 = Y * 256 - 0.39 * 256U - 0.58 * 256V
B * 256 = Y * 256 + 2.03 * 256U
```

简化上面的公式如下：

```
256Y = 76.544R + 150.272G + 29.184B

256U = -37.632R - 73.984G + 111.616B

256V = 157.44R - 131.84G - 25.6B

256R = 256Y + 291.84V

256G = 256Y - 99.84U - 148.48V

256B = 256Y + 519.68U
```
然后，我们就可以对上述公式进一步优化，彻底干掉小数，注意这里是有精度损失的。

```
256Y = 77R + 150G + 29B

256U = -38R - 74G + 112B

256V = 158R - 132G - 26B

256R = 256Y + 292V

256G = 256Y - 100U - 149V

256B = 256Y + 520U

```

实际上就是四舍五入，这是乘以256是为了缩小误差，当然乘数越大，误差越小。和RGB2GRAY一样的套路。
## 优化二：乘法和除法变为移位运算
先将除法变为移位运算：
```
Y = (77R + 150G + 29B) >> 8

U = (-38R - 74G + 112B) >> 8

V = (158R - 132G - 26B) >> 8

R = (256Y + 292V) >> 8

G = (256Y - 100U - 149V) >> 8

B = (256Y + 520U) >> 8

```
公式中还有很多乘法运算，乘法跟移位运算相比，还是效率太低了，因此，我们将把所有乘法都改成移位运算。如何将常数乘法改成移位运算？这里给个例子：Y=Y*9可以改为：Y=(Y<<3)+Y。因此，我们可以讲YUV的公式继续改为最简：
RGB转YUV：
```
Y = ((R << 6) + (R << 3) + (R << 2) + R + (G << 7) + (G << 4) + (G << 2) + (G << 1) + (B << 4) + (B << 3) + (B << 2) + B) >> 8;
U = (-((R << 5) + (R << 2) + (R << 1)) - ((G << 6) + (G << 3) + (G << 1)) + ((B << 6) + (B << 5) + (B << 4))) >> 8;
V = ((R << 7) + (R << 4) + (R << 3) + (R << 2) + (R << 1) - ((G << 7) + (G << 2)) - ((B << 4) + (B << 3) + (B << 1))) >> 8;
```
YUV转RGB：
```
R = ((Y << 8) + ((V << 8) + (V << 5) + (V << 2))) >> 8;
G = ((Y << 8) - ((U << 6) + (U << 5) + (U << 2)) - ((V << 7) + (V << 4) + (V << 2) + V)) >> 8;
B = ((Y << 8) + (U << 9) + (U << 3)) >> 8;
```
使用OpemMP和上诉优化的互转代码如下：注意一下，imread读取的图片通道顺序默认是BGR。

```
//RGB2YUV优化
Mat speed_rgb2yuv(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3b>(i, j)[0] =
				((src.at<Vec3b>(i, j)[2] << 6) + (src.at<Vec3b>(i, j)[2] << 3) + (src.at<Vec3b>(i, j)[2] << 2) + src.at<Vec3b>(i, j)[2] +
				(src.at<Vec3b>(i, j)[1] << 7) + (src.at<Vec3b>(i, j)[1] << 4) + (src.at<Vec3b>(i, j)[1] << 2) + (src.at<Vec3b>(i, j)[1] << 1) +
					(src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 3) + (src.at<Vec3b>(i, j)[0] << 2) + src.at<Vec3b>(i, j)[0]) >> 8;
			dst.at<Vec3b>(i, j)[1] = (-((src.at<Vec3b>(i, j)[2] << 5) + (src.at<Vec3b>(i, j)[2] << 2) + (src.at<Vec3b>(i, j)[2] << 1)) -
				((src.at<Vec3b>(i, j)[1] << 6) + (src.at<Vec3b>(i, j)[1] << 3) + (src.at<Vec3b>(i, j)[1] << 1)) +
				((src.at<Vec3b>(i, j)[0] << 6) + (src.at<Vec3b>(i, j)[0] << 5) + (src.at<Vec3b>(i, j)[0] << 4))) >> 8;
			dst.at<Vec3b>(i, j)[2] = ((src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 4) + (src.at<Vec3b>(i, j)[2] << 3) + (src.at<Vec3b>(i, j)[2] << 2) + (src.at<Vec3b>(i, j)[2] << 1) -
				((src.at<Vec3b>(i, j)[1] << 7) + (src.at<Vec3b>(i, j)[1] << 2)) - ((src.at<Vec3b>(i, j)[0] << 4) + (src.at<Vec3b>(i, j)[0] << 3) + (src.at<Vec3b>(i, j)[0] << 1))) >> 8;
		}
	}
	return dst;
}


//YUV2RGB优化
Mat speed_yuv2rgb(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC3);
	#pragma omp parallel for num_threads(4)
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			dst.at<Vec3b>(i, j)[0] = ((src.at<Vec3b>(i, j)[0] << 8) + (src.at<Vec3b>(i, j)[1] << 9) + (src.at<Vec3b>(i, j)[1] << 3)) >> 8;
			dst.at<Vec3b>(i, j)[1] = ((src.at<Vec3b>(i, j)[0] << 8) - ((src.at<Vec3b>(i, j)[1] << 6) + (src.at<Vec3b>(i, j)[1] << 5) +
			(src.at<Vec3b>(i, j)[1] << 2)) - ((src.at<Vec3b>(i, j)[2] << 7) + (src.at<Vec3b>(i, j)[2] << 4) +
			(src.at<Vec3b>(i, j)[2] << 2) + src.at<Vec3b>(i, j)[2])) >> 8;
			dst.at<Vec3b>(i, j)[2] = ((src.at<Vec3b>(i, j)[0] << 8) + ((src.at<Vec3b>(i, j)[2] << 8) + (src.at<Vec3b>(i, j)[2] << 5) +
			(src.at<Vec3b>(i, j)[2] << 2))) >> 8;
		}
	}
	return dst;
}
```


## 三，RGB和HSV互转
HSV是一种将RGB色彩空间中的点在倒圆锥体中的表示方法。HSV即色相(Hue)、饱和度(Saturation)、明度(Value)，又称HSB(B即Brightness)。色相是色彩的基本属性，就是平常说的颜色的名称，如红色、黄色等。饱和度（S）是指色彩的纯度，越高色彩越纯，低则逐渐变灰，取0-100%的数值。明度（V），取0-max(计算机中HSV取值范围和存储的长度有关)。HSV颜色空间可以用一个圆锥空间模型来描述。圆锥的顶点处，V=0，H和S无定义，代表黑色。圆锥的顶面中心处V=max，S=0，H无定义，代表白色。
RGB颜色空间中，三种颜色分量的取值与所生成的颜色之间的联系并不直观。而HSV颜色空间，更类似于人类感觉颜色的方式，封装了关于颜色的信息：“这是什么颜色？深浅如何？明暗如何？”。
从RGB转到HSV的计算公式为：设max等于r、g和b中的最大者，min为最小者。对应的HSV空间中的(h,s,v)值为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190213154738127.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)h在0到360°之间，s在0到100%之间，v在0到max之间。
从HSV空间转回RGB空间的公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190213154827434.png)
代码实现，效果测试无误：
```
Mat RGB2HSV(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(r, min(g, b));
			float maxx = max(r, max(g, b));
			dst.at<Vec3f>(i, j)[2] = maxx; //V
			float delta = maxx - minn;
			float h, s;
			if (maxx != 0) {
				s = delta / maxx;
			}
			else {
				s = 0;
			}
			if (r == maxx) {
				h = (g - b) / delta;
			}
			else if (g == maxx) {
				h = 2 + (b - r) / delta;
			}
			else {
				h = 4 + (r - g) / delta;
			}
			h *= 60;
			if (h < 0)
				h += 360;
			dst.at<Vec3f>(i, j)[0] = h;
			dst.at<Vec3f>(i, j)[1] = s;
		}
	}
	return dst;
}

Mat HSV2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	float r, g, b, h, s, v;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			h = src.at<Vec3f>(i, j)[0];
			s = src.at<Vec3f>(i, j)[1];
			v = src.at<Vec3f>(i, j)[2];
			if (s == 0) {
				r = g = b = v;
			}
			else {
				h /= 60;
				int offset = floor(h);
				float f = h - offset;
				float p = v * (1 - s);
				float q = v * (1 - s * f);
				float t = v * (1 - s * (1 - f));
				switch (offset)
				{
				case 0: r = v; g = t; b = p; break;
				case 1: r = q; g = v; b = p; break;
				case 2: r = p; g = v; b = t; break;
				case 3: r = p; g = q; b = v; break;
				case 4: r = t; g = p; b = v; break;
				case 5: r = v; g = p; b = q; break;
				default:
					break;
				}
			}
			dst.at<Vec3b>(i, j)[0] = int(b * 255);
			dst.at<Vec3b>(i, j)[1] = int(g * 255);
			dst.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	return dst;
}
```
## 四，RGB和HSI互转
HSI色彩空间是从人的视觉系统出发，用色调(Hue)、饱和度(Saturation或Chroma)和亮度 (Intensity或Brightness)来描述色彩。
- H——表示颜色的相位角。红、绿、蓝分别相隔120度；互补色分别相差180度，即颜色的类别。
- S——表示成所选颜色的纯度和该颜色最大的纯度之间的比率，范围：[0,  1]，即颜色的深浅程度。
- I——表示色彩的明亮程度，围：[0, 1]，人眼对亮度很敏感！
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190213163428719.png)
可以看到HSI色彩空间和RGB色彩空间只是同一物理量的不同表示法，因而它们之间存在着转换关系：HSI颜色模式中的色调使用颜色类别表示，饱和度与颜色的白光光亮亮度刚好成反比，代表灰色与色调的比例，亮度是颜色的相对明暗程度。
转自：https://blog.csdn.net/aoshilang2249/article/details/38070663
RGB转换为HSI的公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190213163800518.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
HSI转RGB的公式为：给定 HSI空间中的 (h, s, l) 值定义的一个颜色，带有 h 在指示色相角度的值域 [0, 360）中，分别表示饱和度和亮度的s 和 l 在值域 [0, 1] 中，相应在 RGB 空间中的 (r, g, b) 三原色，带有分别对应于红色、绿色和蓝色的 r, g 和 b 也在值域 [0, 1] 中，它们可计算为：
首先，如果 s = 0，则结果的颜色是非彩色的、或灰色的。在这个特殊情况，r, g 和 b 都等于 l。注意 h 的值在这种情况下是未定义的。当 s ≠ 0 的时候，可以使用下列过程：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190213163928189.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
源码实现，测试无误：

```
Mat RGB2HSI(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_32FC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float b = src.at<Vec3b>(i, j)[0] / 255.0;
			float g = src.at<Vec3b>(i, j)[1] / 255.0;
			float r = src.at<Vec3b>(i, j)[2] / 255.0;
			float minn = min(b, min(g, r));
			float maxx = max(b, max(g, r));
			float H = 0;
			float S = 0;
			float I = (minn + maxx) / 2.0f;
			if (maxx == minn) {
				dst.at<Vec3f>(i, j)[0] = H;
				dst.at<Vec3f>(i, j)[1] = S;
				dst.at<Vec3f>(i, j)[2] = I;
			}
			else {
				float delta = maxx - minn;
				if (I < 0.5) {
					S = delta / (maxx + minn);
				}
				else {
					S = delta / (2.0 - maxx - minn);
				}
				if (r == maxx) {
					if (g > b) {
						H = (g - b) / delta;
					}
					else {
						H = 6.0 + (g - b) / delta;
					}
				}
				else if (g == maxx) {
					H = 2.0 + (b - r) / delta;
				}
				else {
					H = 4.0 + (r - g) / delta;
				}
				H /= 6.0; //除以6，表示在那个部分
				if (H < 0.0)
					H += 1.0;
				if (H > 1)
					H -= 1;
				H = (int)(H * 360); //转成[0, 360]
				dst.at<Vec3f>(i, j)[0] = H;
				dst.at<Vec3f>(i, j)[1] = S;
				dst.at<Vec3f>(i, j)[2] = I;
			}
		}
	}
	return dst;
}

float get_Ans(double p, double q, double Ht) {
	if (Ht < 0.0)
		Ht += 1.0;
	else if (Ht > 1.0)
		Ht -= 1.0;
	if ((6.0 * Ht) < 1.0)
		return (p + (q - p) * Ht * 6.0);
	else if ((2.0 * Ht) < 1.0)
		return q;
	else if ((3.0 * Ht) < 2.0)
		return (p + (q - p) * ((2.0F / 3.0F) - Ht) * 6.0);
	else
		return (p);
}

Mat HSI2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float r, g, b, M1, M2;
			float H = src.at<Vec3f>(i, j)[0];
			float S = src.at<Vec3f>(i, j)[1];
			float I = src.at<Vec3f>(i, j)[2];
			float hue = H / 360;
			if (S == 0) {//灰色
				r = g = b = I;
			}
			else {
				if (I <= 0.5) {
					M2 = I * (1.0 + S);
				}
				else {
					M2 = I + S - I * S;
				}
				M1 = (2.0 * I - M2);
				r = get_Ans(M1, M2, hue + 1.0 / 3.0);
				g = get_Ans(M1, M2, hue);
				b = get_Ans(M1, M2, hue - 1.0 / 3.0);
			}
			dst.at<Vec3b>(i, j)[0] = (int)(b * 255);
			dst.at<Vec3b>(i, j)[1] = (int)(g * 255);
			dst.at<Vec3b>(i, j)[2] = (int)(r * 255);
		}
	}
	return dst;
}
```

## 五，RGB和YCbCr互转
 在常用的几种颜色空间中，YCbCr颜色空间在学术论文中出现的频率是相当高的，常用于肤色检测等等。其和RGB空间之间的相互转换公式在网上也有多种，我们这里取http://en.wikipedia.org/wiki/YCbCr 描述的JPG转换时使用的计算公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190409145140619.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)和RGB与CIEXYZ空间互转的优化套路一样，测试无误的代码如下：

```
const float YCbCrYRF = 0.299F;              // RGB转YCbCr的系数(浮点类型）
const float YCbCrYGF = 0.587F;
const float YCbCrYBF = 0.114F;
const float YCbCrCbRF = -0.168736F;
const float YCbCrCbGF = -0.331264F;
const float YCbCrCbBF = 0.500000F;
const float YCbCrCrRF = 0.500000F;
const float YCbCrCrGF = -0.418688F;
const float YCbCrCrBF = -0.081312F;

const float RGBRYF = 1.00000F;            // YCbCr转RGB的系数(浮点类型）
const float RGBRCbF = 0.0000F;
const float RGBRCrF = 1.40200F;
const float RGBGYF = 1.00000F;
const float RGBGCbF = -0.34414F;
const float RGBGCrF = -0.71414F;
const float RGBBYF = 1.00000F;
const float RGBBCbF = 1.77200F;
const float RGBBCrF = 0.00000F;

const int Shift = 20;
const int HalfShiftValue = 1 << (Shift - 1);

const int YCbCrYRI = (int)(YCbCrYRF * (1 << Shift) + 0.5);         // RGB转YCbCr的系数(整数类型）
const int YCbCrYGI = (int)(YCbCrYGF * (1 << Shift) + 0.5);
const int YCbCrYBI = (int)(YCbCrYBF * (1 << Shift) + 0.5);
const int YCbCrCbRI = (int)(YCbCrCbRF * (1 << Shift) + 0.5);
const int YCbCrCbGI = (int)(YCbCrCbGF * (1 << Shift) + 0.5);
const int YCbCrCbBI = (int)(YCbCrCbBF * (1 << Shift) + 0.5);
const int YCbCrCrRI = (int)(YCbCrCrRF * (1 << Shift) + 0.5);
const int YCbCrCrGI = (int)(YCbCrCrGF * (1 << Shift) + 0.5);
const int YCbCrCrBI = (int)(YCbCrCrBF * (1 << Shift) + 0.5);

const int RGBRYI = (int)(RGBRYF * (1 << Shift) + 0.5);              // YCbCr转RGB的系数(整数类型）
const int RGBRCbI = (int)(RGBRCbF * (1 << Shift) + 0.5);
const int RGBRCrI = (int)(RGBRCrF * (1 << Shift) + 0.5);
const int RGBGYI = (int)(RGBGYF * (1 << Shift) + 0.5);
const int RGBGCbI = (int)(RGBGCbF * (1 << Shift) + 0.5);
const int RGBGCrI = (int)(RGBGCrF * (1 << Shift) + 0.5);
const int RGBBYI = (int)(RGBBYF * (1 << Shift) + 0.5);
const int RGBBCbI = (int)(RGBBCbF * (1 << Shift) + 0.5);
const int RGBBCrI = (int)(RGBBCrF * (1 << Shift) + 0.5);

Mat RGB2YCbCr(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0];
			int Green = src.at<Vec3b>(i, j)[1];
			int Red = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(i, j)[0] = (int)((YCbCrYRI * Red + YCbCrYGI * Green + YCbCrYBI * Blue + HalfShiftValue) >> Shift);
			dst.at<Vec3b>(i, j)[1] = (int)(128 + ((YCbCrCbRI * Red + YCbCrCbGI * Green + YCbCrCbBI * Blue + HalfShiftValue) >> Shift));
			dst.at<Vec3b>(i, j)[2] = (int)(128 + ((YCbCrCrRI * Red + YCbCrCrGI * Green + YCbCrCrBI * Blue + HalfShiftValue) >> Shift));
		}
	}
	return dst;
}

Mat YCbCr2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Y = src.at<Vec3b>(i, j)[0];
			int Cb = src.at<Vec3b>(i, j)[1] - 128;
			int Cr = src.at<Vec3b>(i, j)[2] - 128;
			int Red = Y + ((RGBRCrI * Cr + HalfShiftValue) >> Shift);
			int Green = Y + ((RGBGCbI * Cb + RGBGCrI * Cr + HalfShiftValue) >> Shift);
			int Blue = Y + ((RGBBCbI * Cb + HalfShiftValue) >> Shift);
			if (Red > 255) Red = 255; else if (Red < 0) Red = 0;
			if (Green > 255) Green = 255; else if (Green < 0) Green = 0;    // 编译后应该比三目运算符的效率高
			if (Blue > 255) Blue = 255; else if (Blue < 0) Blue = 0;
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}
```

## 六，RGB和YDbDr互转
 YDbDr颜色空间和YCbCr颜色空间类似，其和RGB空间之间的相互转换公式里取http://en.wikipedia.org/wiki/YDbDr 所描述的。
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20190711114608334.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)代码：

```
const float YDbDrYRF = 0.299F;              // RGB转YDbDr的系数(浮点类型）
const float YDbDrYGF = 0.587F;
const float YDbDrYBF = 0.114F;
const float YDbDrDbRF = -0.1688F;
const float YDbDrDbGF = -0.3312F;
const float YDbDrDbBF = 0.5F;
const float YDbDrDrRF = -0.5F;
const float YDbDrDrGF = 0.4186F;
const float YDbDrDrBF = 0.0814F;

const float RGBRYF = 1.00000F;            // YDbDr转RGB的系数(浮点类型）
const float RGBRDbF = 0.0002460817072494899F;
const float RGBRDrF = -1.402083073344533F;
const float RGBGYF = 1.00000F;
const float RGBGDbF = -0.344268308442098F;
const float RGBGDrF = 0.714219609001458F;
const float RGBBYF = 1.00000F;
const float RGBBDbF = 1.772034373903893F;
const float RGBBDrF = 0.0002111539810593343F;

const int Shift = 20;
const int HalfShiftValue = 1 << (Shift - 1);

const int YDbDrYRI = (int)(YDbDrYRF * (1 << Shift) + 0.5);         // RGB转YDbDr的系数(整数类型）
const int YDbDrYGI = (int)(YDbDrYGF * (1 << Shift) + 0.5);
const int YDbDrYBI = (int)(YDbDrYBF * (1 << Shift) + 0.5);
const int YDbDrDbRI = (int)(YDbDrDbRF * (1 << Shift) + 0.5);
const int YDbDrDbGI = (int)(YDbDrDbGF * (1 << Shift) + 0.5);
const int YDbDrDbBI = (int)(YDbDrDbBF * (1 << Shift) + 0.5);
const int YDbDrDrRI = (int)(YDbDrDrRF * (1 << Shift) + 0.5);
const int YDbDrDrGI = (int)(YDbDrDrGF * (1 << Shift) + 0.5);
const int YDbDrDrBI = (int)(YDbDrDrBF * (1 << Shift) + 0.5);

const int RGBRYI = (int)(RGBRYF * (1 << Shift) + 0.5);              // YDbDr转RGB的系数(整数类型）
const int RGBRDbI = (int)(RGBRDbF * (1 << Shift) + 0.5);
const int RGBRDrI = (int)(RGBRDrF * (1 << Shift) + 0.5);
const int RGBGYI = (int)(RGBGYF * (1 << Shift) + 0.5);
const int RGBGDbI = (int)(RGBGDbF * (1 << Shift) + 0.5);
const int RGBGDrI = (int)(RGBGDrF * (1 << Shift) + 0.5);
const int RGBBYI = (int)(RGBBYF * (1 << Shift) + 0.5);
const int RGBBDbI = (int)(RGBBDbF * (1 << Shift) + 0.5);
const int RGBBDrI = (int)(RGBBDrF * (1 << Shift) + 0.5);

Mat RGB2YDbDr(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Blue = src.at<Vec3b>(i, j)[0];
			int Green = src.at<Vec3b>(i, j)[1];
			int Red = src.at<Vec3b>(i, j)[2];
			dst.at<Vec3b>(i, j)[0] = (uchar)((YDbDrYRI * Red + YDbDrYGI * Green + YDbDrYBI * Blue + HalfShiftValue) >> Shift);
			dst.at<Vec3b>(i, j)[1] = (uchar)(128 + ((YDbDrDbRI * Red + YDbDrDbGI * Green + YDbDrDbBI * Blue + HalfShiftValue) >> Shift));
			dst.at<Vec3b>(i, j)[2] = (uchar)(128 + ((YDbDrDrRI * Red + YDbDrDrGI * Green + YDbDrDrBI * Blue + HalfShiftValue) >> Shift));
		}
	}
	return dst;
}

Mat YDbDr2RGB(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Y = src.at<Vec3b>(i, j)[0];
			int Db = src.at<Vec3b>(i, j)[1] - 128;
			int Dr = src.at<Vec3b>(i, j)[2] - 128;
			int Red = Y + ((RGBRDbI * Db + RGBRDrI * Dr + HalfShiftValue) >> Shift);
			int Green = Y + ((RGBGDbI * Db + RGBGDrI * Dr + HalfShiftValue) >> Shift);
			int Blue = Y + ((RGBBDbI * Db + RGBBDrI * Dr + HalfShiftValue) >> Shift);
			if (Red > 255) Red = 255;
			else if (Red < 0) Red = 0;
			if (Green > 255) Green = 255;
			else if (Green < 0) Green = 0;
			if (Blue > 255) Blue = 0;
			else if (Blue < 0) Blue = 0;
			dst.at<Vec3b>(i, j)[0] = Blue;
			dst.at<Vec3b>(i, j)[1] = Green;
			dst.at<Vec3b>(i, j)[2] = Red;
		}
	}
	return dst;
}
```

# 后记
本文盘点了6种常见的颜色空间互转算法，因为这六种算法是我在学习过程和图像处理论文复现中有使用到的，所以如果还有其他我这里没提到的的转换算法，欢迎留言和我交流哦。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPadaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)