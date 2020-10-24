# 前言
对于光照不均匀的图像,用通常的图像分割方法不能取得满意的效果。为了解决这个问题，论文《一种基于亮度均衡的图像阈值分割技术》提出了一种实用而简便的图像分割方法。该方法针对图像中不同亮度区域进行亮度补偿,使得整个图像亮度背景趋于一致后,再进行常规的阈值分割。实验结果表明，用该方法能取得良好的分割效果。关于常规的阈值分割不是我这篇推文关注的，我这里只实现前面光照补偿的部分。算法的原理可以仔细看论文。论文原文见附录。

# 算法步骤
- 如果是RGB图需要转化成灰度图。
- 求取原始图src的平均灰度，并记录rows和cols。
- 按照一定大小，分为$DX \times DY$个方块，求出每块的平均值，得到子块的亮度矩阵$D$。
- 用矩阵$D$的每个元素减去原图的平均灰度，得到子块的亮度差值矩阵$E$。
- 用双立方插值法，将矩阵$E$ `resize`成和原图一样大小的亮度分布矩阵$R$。
- 得到矫正后的图像：$result=I-R$。

# 代码实现

```
Mat speed_rgb2gray(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
#pragma omp parallel for num_threads(12)
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


Mat unevenLightCompensate(Mat src, int block_Size) {
	int row = src.rows;
	int col = src.cols;
	Mat gray(row, col, CV_8UC1);
	if (src.channels() == 3) {
		gray = speed_rgb2gray(src);
	}
	else {
		gray = src;
	}
	float average = mean(gray)[0];
	int new_row = ceil(1.0 * row / block_Size);
	int new_col = ceil(1.0 * col / block_Size);
	Mat new_img(new_row, new_col, CV_32FC1);
	for (int i = 0; i < new_row; i++) {
		for (int j = 0; j < new_col; j++) {
			int rowx = i * block_Size;
			int rowy = (i + 1) * block_Size;
			int colx = j * block_Size;
			int coly = (j + 1) * block_Size;
			if (rowy > row) rowy = row;
			if (coly > col) coly = col;
			Mat ROI = src(Range(rowx, rowy), Range(colx, coly));
			float block_average = mean(ROI)[0];
			new_img.at<float>(i, j) = block_average;
		}
	}
	new_img = new_img - average;
	Mat new_img2;
	resize(new_img, new_img2, Size(row, col), (0, 0), (0, 0), INTER_CUBIC);
	Mat new_src;
	gray.convertTo(new_src, CV_32FC1);
	Mat dst = new_src - new_img2;
	dst.convertTo(dst, CV_8UC1);
	return dst;
}
```

# 效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190314142018712.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
可以看到经过这个算法处理之后，亮度确实被均衡了一些，从视觉效果上来看还是有作用的。

# 附录
- 论文原文：https://wenku.baidu.com/view/f74cc087e53a580216fcfe52.html?from=search


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
- [OpenCV图像处理专栏十二 |《基于二维伽马函数的光照不均匀图像自适应校正算法》](https://mp.weixin.qq.com/s/rz9sKcmq1saIVkDNkzyh2A)
- [OpenCV图像处理专栏十三 | 利用多尺度融合提升图像细节](https://mp.weixin.qq.com/s/ixJ7kt1i1qKkG4sdbRPHNA)
- [OpenCV图像处理专栏十四 | 基于Retinex成像原理的自动色彩均衡算法(ACE)](https://mp.weixin.qq.com/s/Jr1191839j3IdOlpRsrM6w)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)