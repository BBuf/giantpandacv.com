# 前言
这是OpenCV图像处理专栏的第十二篇文章，今天为大家介绍一个用于解决光照不均匀的图像自适应校正算法。光照不均匀其实是非常常见的一种状况，为了提升人类的视觉感受或者是为了提升诸如深度学习之类的算法准确性，人们在解决光照不均衡方面已经有大量的工作。一起来看看这篇论文使用的算法吧，论文名为：《基于二维伽马函数的光照不均匀图像自适应校正算法》。

# 算法原理
论文使用了Retinex的多尺度高斯滤波求取**光照分量**，然后使用了二**维Gamma函数**针对原图的**HSV空间的V(亮度)分量**进行亮度改变，得到结果。原理还是蛮简单的，因为是中文论文，且作者介绍得很清楚，我就不细说了，可以自己看论文，论文地址见附录。本文的重点在于对算法步骤的解读和OpenCV复现。

# 算法步骤
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190315105109777.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70) 
# 需要注意的点
文中公式5(二维Gamma变换) 有误，公式5为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020020319173958.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
其中$\gamma$ 的指数应该是$m-I(x,y)$，而不是$I(x,y)-m$，如果使用后者会得到错误结果，应该是作者笔误了。

# OpenCV C++代码复现

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

Mat work(Mat src) {
	int row = src.rows;
	int col = src.cols;
	Mat now = RGB2HSV(src);
	Mat H(row, col, CV_32FC1);
	Mat S(row, col, CV_32FC1);
	Mat V(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			H.at<float>(i, j) = now.at<Vec3f>(i, j)[0];
			S.at<float>(i, j) = now.at<Vec3f>(i, j)[1];
			V.at<float>(i, j) = now.at<Vec3f>(i, j)[2];
		}
	}
	int kernel_size = min(row, col);
	if (kernel_size % 2 == 0) {
		kernel_size -= 1;
	}
	float SIGMA1 = 15;
	float SIGMA2 = 80;
	float SIGMA3 = 250;
	float q = sqrt(2.0);
	Mat F(row, col, CV_32FC1);
	Mat F1, F2, F3;
	GaussianBlur(V, F1, Size(kernel_size, kernel_size), SIGMA1 / q);
	GaussianBlur(V, F2, Size(kernel_size, kernel_size), SIGMA2 / q);
	GaussianBlur(V, F3, Size(kernel_size, kernel_size), SIGMA3 / q);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			F.at <float>(i, j) = (F1.at<float>(i, j) + F2.at<float>(i, j) + F3.at<float>(i, j)) / 3.0;
		}
	}
	float average = mean(F)[0];
	Mat out(row, col, CV_32FC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			float gamma = powf(0.5, (average - F.at<float>(i, j)) / average);
			out.at<float>(i, j) = powf(V.at<float>(i, j), gamma);
		}
	}
	vector <Mat> v;
	v.push_back(H);
	v.push_back(S);
	v.push_back(out);
	Mat merge_;
	merge(v, merge_);
	Mat dst = HSV2RGB(merge_);
	return dst;
}
```

# 效果


![原图1](https://img-blog.csdnimg.cn/20190315110434546.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![效果图1](https://img-blog.csdnimg.cn/2019031511054243.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![原图2](https://img-blog.csdnimg.cn/20190315110557148.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![效果图2](https://img-blog.csdnimg.cn/20190315110633745.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 结论
可以看到这个算法对光照不均匀的图像校正效果还是不错的，且没有像Retiex那样在亮度突变处出现色晕现象。

# 附录
- 论文原文：https://wenku.baidu.com/view/3570f2c255270722182ef74e.html 

# 同期文章
- [OpenCV图像处理专栏一 | 盘点常见颜色空间互转](https://mp.weixin.qq.com/s/c_7cdSmqkr8tXMXDORSA-Q)
- [OpenCV图像处理专栏二 |《Local Color Correction 》论文阅读及C++复现](https://mp.weixin.qq.com/s/z7tIiD0wLikcjFwtwZaV8w)
- [OpenCV图像处理专栏三 | 灰度世界算法原理和实现](https://mp.weixin.qq.com/s/aiVIci0NQVyUTJ7V8ElH3g)
- [OpenCV图像处理专栏四 | 自动白平衡之完美反射算法原理及C++实现](https://mp.weixin.qq.com/s/AVHB9cC-FJwD4SKUQ51-YA)
- [OpenCV图像处理专栏五 | ACE算法论文解读及实现](https://mp.weixin.qq.com/s/aPi7haF7eTDabbYi4c75cA)
- [OpenCV图像处理专栏六 | 来自何凯明博士的暗通道去雾算法(CVPR 2009最佳论文)](https://mp.weixin.qq.com/s/PCvTDqEt53voZFij9jNWKQ)
- [OpenCV图像处理专栏七 | 直方图均衡化算法及代码实现](https://mp.weixin.qq.com/s/tWqjYd0YXwv6cAVVQdR4wA)
- [OpenCV图像处理专栏八 | 《Contrast image correction method》 论文阅读及代码实现](https://mp.weixin.qq.com/s/ZO_KE6MrQ0yvTm2xsy5bOA)
- [OpenCV图像处理专栏九 | 基于直方图的快速中值滤波算法](https://mp.weixin.qq.com/s/C-e-uUQSIbHSQRRaNFXgbA)
- [OpenCV图像处理专栏十 | 利用中值滤波进行去雾](https://mp.weixin.qq.com/s/gxyShyUR_UWL1QnOb6y9Pg)
- [OpenCV图像处理专栏十一 | IEEE Xplore 2015的图像白平衡处理之动态阈值法](https://mp.weixin.qq.com/s/4kOs2YmOk5PlKv4zlU5vuQ)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)