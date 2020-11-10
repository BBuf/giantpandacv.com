# 前言
这是OpenCV图像处理专栏的第十篇文章，介绍一种利用中值滤波来实现去雾的算法。这个方法发表于国内的一篇论文，链接我放附录了。
# 算法原理

这个算法和之前He Kaiming的暗通道去雾都基于大气散射模型即：
$I(x)=J(x)t(x)+A(1-t(x))$
其中$I(x)$就是输入图像，需要求去雾后的输出图像$J(x)$，所以我们只要计算出全局大气光值$A$和透射率$t(x)$就可以了。其他的一些介绍和背景交代可以去看原文，这里我直接给出论文的算法核心步骤。



- 1、定义$F(x)=A(1-t(x))$，焦作大气光幕或者雾浓度。
- 2、计算$M(x)=min_{c\in (r,g,b)}(I(x))$，即是求暗通道，这一点在[OpenCV图像处理专栏六 | 来自何凯明博士的暗通道去雾算法(CVPR 2009最佳论文)](https://mp.weixin.qq.com/s/PCvTDqEt53voZFij9jNWKQ) 我已经详细说明了。
- 3、计算$A(x,y)=median_s(M(x,y))$，即对$M(x,y)$进行中值滤波得到$A$。
- 4、计算$B(x,y)=A(x,y)-median_s(|A(x,y)-M(x,y)|)$，注意式子中取了绝对值。
- 5、计算$F(x,y)=max(min(pB(x,y), M(x,y)), 0)$，式子中$P$是控制去雾浓度的系数，取值为$[0,1]$。
- 6、通过式子$J(x,y)=\frac{I(x)-F(x)}{1-F(x)/A}$获得去雾后的图像，这个式子就是把原始子移项变形得到的。
- 7、自此，算法结束，得到了利用中值滤波实现的去雾后的结果。

# 代码实现

```
int rows, cols;
//获取最小值矩阵
int **getMinChannel(cv::Mat img) {
	rows = img.rows;
	cols = img.cols;
	if (img.channels() != 3) {
		fprintf(stderr, "Input Error!");
		exit(-1);
	}
	int **imgGray;
	imgGray = new int *[rows];
	for (int i = 0; i < rows; i++) {
		imgGray[i] = new int[cols];
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			int loacalMin = 255;
			for (int k = 0; k < 3; k++) {
				if (img.at<Vec3b>(i, j)[k] < loacalMin) {
					loacalMin = img.at<Vec3b>(i, j)[k];
				}
			}
			imgGray[i][j] = loacalMin;
		}
	}
	return imgGray;
}

//求暗通道
int **getDarkChannel(int **img, int blockSize = 3) {
	if (blockSize % 2 == 0 || blockSize < 3) {
		fprintf(stderr, "blockSize is not odd or too small!");
		exit(-1);
	}
	//计算pool Size
	int poolSize = (blockSize - 1) / 2;
	int newHeight = rows + poolSize - 1;
	int newWidth = cols + poolSize - 1;
	int **imgMiddle;
	imgMiddle = new int *[newHeight];
	for (int i = 0; i < newHeight; i++) {
		imgMiddle[i] = new int[newWidth];
	}
	for (int i = 0; i < newHeight; i++) {
		for (int j = 0; j < newWidth; j++) {
			if (i < rows && j < cols) {
				imgMiddle[i][j] = img[i][j];
			}
			else {
				imgMiddle[i][j] = 255;
			}
		}
	}
	int **imgDark;
	imgDark = new int *[rows];
	for (int i = 0; i < rows; i++) {
		imgDark[i] = new int[cols];
	}
	int localMin = 255;
	for (int i = poolSize; i < newHeight - poolSize; i++) {
		for (int j = poolSize; j < newWidth - poolSize; j++) {
			for (int k = i - poolSize; k < i + poolSize + 1; k++) {
				for (int l = j - poolSize; l < j + poolSize + 1; l++) {
					if (imgMiddle[k][l] < localMin) {
						localMin = imgMiddle[k][l];
					}
				}
			}
			imgDark[i - poolSize][j - poolSize] = localMin;
		}
	}
	return imgDark;
}

Mat MedianFilterFogRemoval(Mat src, float p = 0.95, int KernelSize = 41, int blockSize=3, bool meanModel = false, float percent = 0.001) {
	int row = src.rows;
	int col = src.cols;
	int** imgGray = getMinChannel(src);
	int **imgDark = getDarkChannel(imgGray, blockSize = blockSize);
	//int atmosphericLight = getGlobalAtmosphericLightValue(imgDark, src, meanModel = meanModel, percent = percent);
	int Histgram[256] = { 0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Histgram[imgDark[i][j]]++;
		}
	}
	int Sum = 0, atmosphericLight = 0;
	for (int i = 255; i >= 0; i--) {
		Sum += Histgram[i];
		if (Sum > row * col * 0.01) {
			atmosphericLight = i;
			break;
		}
	}
	int SumB = 0, SumG = 0, SumR = 0, Amount = 0;
	//printf("%d\n", atmosphericLight);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (imgDark[i][j] >= atmosphericLight) {
				SumB += src.at<Vec3b>(i, j)[0];
				SumG += src.at<Vec3b>(i, j)[1];
				SumR += src.at<Vec3b>(i, j)[2];
				Amount++;
			}
		}
	}
	SumB /= Amount;
	SumG /= Amount;
	SumR /= Amount;
	Mat Filter(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			Filter.at<uchar>(i, j) = imgDark[i][j];
		}
	}
	Mat A(row, col, CV_8UC1);
	medianBlur(Filter, A, KernelSize);
	Mat temp(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Diff = Filter.at<uchar>(i, j) - A.at<uchar>(i, j);
			if (Diff < 0) Diff = -Diff;
			temp.at<uchar>(i, j) = Diff;
		}
	}
	medianBlur(temp, temp, KernelSize);
	Mat B(row, col, CV_8UC1);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Diff = A.at<uchar>(i, j) - temp.at<uchar>(i, j);
			if (Diff < 0) Diff = 0;
			B.at<uchar>(i, j) = Diff;
		}
	}
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int Min = B.at<uchar>(i, j) * p;
			if (imgDark[i][j] > Min) {
				B.at<uchar>(i, j) = Min;
			}
			else {
				B.at<uchar>(i, j) = imgDark[i][j];
			}
		}
	}
	Mat dst(row, col, CV_8UC3);
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int F = B.at<uchar>(i, j);
			int Value;
			if (SumB != F) {
				Value = SumB * (src.at<Vec3b>(i, j)[0] - F) / (SumB - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[0];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[0] = Value;

			if (SumG != F) {
				Value = SumG * (src.at<Vec3b>(i, j)[1] - F) / (SumG - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[1];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[1] = Value;

			if (SumR != F) {
				Value = SumR * (src.at<Vec3b>(i, j)[2] - F) / (SumR - F);
			}
			else {
				Value = src.at<Vec3b>(i, j)[2];
			}
			if (Value < 0) Value = 0;
			else if (Value > 255) Value = 255;
			dst.at<Vec3b>(i, j)[2] = Value;
		}
	}
	return dst;
}
```



# 效果
均是原图和算法处理后的结果的顺序，可以看到这个算法得到了还不错的结果。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425172828752.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425172836388.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173021339.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173034295.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173112219.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/2019042517312236.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173157669.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173207173.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173233991.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173243102.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173301720.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173308782.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173329180.jpg)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190425173350753.jpg)

# 附录

论文原文：https://wenku.baidu.com/view/dfe4191459eef8c75fbfb38a.html

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)