# 算法介绍
这是OpenCV图像处理专栏的第十一篇文章，之前介绍过两种处理白平衡的算法，分别为灰度世界算法和完美反射算法。今天来介绍另外一个自动白平衡的算法，即动态阈值法，一个看起来比较厉害的名字。论文原文链接放在附录。
# 算法原理
和灰度世界法和完美反射算法类似，动态阈值算法仍然分为两个步骤即白点检测和白点调整，具体如下：
## 白点检测

- 1、把尺寸为$w\times h$的原图像从$RGB$空间转换到$YCrCb$空间。

- 2、把图像分成$3\times 4$个块。
- 3、对每个块分别计算$Cr$，$Cb$的平均值$Mr$，$Mb$。
- 4、判定每个块的近白区域（`near-white region`）。判别准则为：
      - $Cb(i, j) − (Mb + Db\times sign(Mb)) < 1.5\times Db$ 
      - $Cr(i, j) − (1.5\times Mr + Dr \times sign(Mr )) < 1.5\times Dr$，其中sign为符号函数，即正数返回1，负数返回0。
- 5、设一个“参考白色点”的亮度矩阵$RL$，大小为$w\times h$。
- 6、若符合判别式，则作为“参考白色点”,并把该点$（i，j）$的亮度（$Y$分量）值赋给$RL(i,j)$。若不符合，则该点的$RL(i,j)$值为0。

## 白点调整

- 1、选取参考“参考白色点”中最大的`10%`的亮度（`Y`分量）值，并选取其中的最小值`Lu_min`。
- 2、调整`RL`，若`RL(i,j)<Lu_min`, `RL(i,j)=0`; 否则，`RL(i,j)=1`。
- 3、分别把`R`，`G`，`B`与`RL`相乘，得到`R2`，`G2`，`B2`。  分别计算`R2`，`G2`，`B2`的平均值，`Rav`，`Gav`，`Bav`。
- 4、 得到调整增益：定义`Ymax=double(max(max(Y)))`，则`Rgain=Ymax/Rav`,`Ggain=Ymax/Gav`, `Bgain=Ymax/Bav`。
- 5、调整原图像：`Ro= R*Rgain`; `Go= G*Ggain`; `Bo= B*Bgain`;

# 代码实现
块的大小取了100，没处理长或者宽不够100的结尾部分，这个可以自己添加。
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

template<typename T>
inline T sign(T const &input) {
	return input >= 0 ? 1 : -1;
}

Mat AutomaticWhiteBalanceMethod(Mat src) {
	int row = src.rows;
	int col = src.cols;
	if (src.channels() == 4) {
		cvtColor(src, src, CV_BGRA2BGR);
	}
	Mat input = RGB2YCbCr(src);
	Mat mark(row, col, CV_8UC1);
	int sum = 0;
	for (int i = 0; i < row; i += 100) {
		for (int j = 0; j < col; j += 100) {
			if (i + 100 < row && j + 100 < col) {
				Rect rect(j, i, 100, 100);
				Mat temp = input(rect);
				Scalar global_mean = mean(temp);
				double dr = 0, db = 0;
				for (int x = 0; x < 100; x++) {
					uchar *ptr = temp.ptr<uchar>(x) + 1;
					for (int y = 0; y < 100; y++) {
						dr += pow(abs(*ptr - global_mean[1]), 2);
						ptr++;
						db += pow(abs(*ptr - global_mean[2]), 2);
						ptr++;
						ptr++;
					}
				}
				dr /= 10000;
				db /= 10000;
				double cr_left_criteria = 1.5 * global_mean[1] + dr * sign(global_mean[1]);
				double cr_right_criteria = 1.5 * dr;
				double cb_left_criteria = global_mean[2] + db * sign(global_mean[2]);
				double cb_right_criteria = 1.5 * db;
				for (int x = 0; x < 100; x++) {
					uchar *ptr = temp.ptr<uchar>(x) + 1;
					for (int y = 0; y < 100; y++) {
						uchar cr = *ptr;
						ptr++;
						uchar cb = *ptr;
						ptr++;
						ptr++;
						if ((cr - cb_left_criteria) < cb_right_criteria && (cb - cr_left_criteria) < cr_right_criteria) {
							sum++;
							mark.at<uchar>(i + x, j + y) = 1;
						}
						else {
							mark.at<uchar>(i + x, j + y) = 0;
						}
					}
				}
			}
		}
	}

	int Threshold = 0;
	int Ymax = 0;
	int Light[256] = { 0 };
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mark.at<uchar>(i, j) == 1) {
				Light[(int)(input.at<Vec3b>(i, j)[0])]++;
			}
			Ymax = max(Ymax, (int)(input.at<Vec3b>(i, j)[0]));
		}
	}
	printf("maxY: %d\n", Ymax);
	int sum2 = 0;
	for (int i = 255; i >= 0; i--) {
		sum2 += Light[i];
		if (sum2 >= sum * 0.1) {
			Threshold = i;
			break;
		}
	}
	printf("Threshold: %d\n", Threshold);
	printf("Sum: %d Sum2: %d\n", sum, sum2);
	double Blue = 0;
	double Green = 0;
	double Red = 0;
	int cnt2 = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			if (mark.at<uchar>(i, j) == 1 && (int)(input.at<Vec3b>(i, j)[0]) >= Threshold) {
				Blue += 1.0 * src.at<Vec3b>(i, j)[0];
				Green += 1.0 * src.at<Vec3b>(i, j)[1];
				Red += 1.0 * src.at<Vec3b>(i, j)[2];
				cnt2++;
			}
		}
	}
	Blue /= cnt2;
	Green /= cnt2;
	Red /= cnt2;
	printf("%.5f %.5f %.5f\n", Blue, Green, Red);
	Mat dst(row, col, CV_8UC3);
	double maxY = Ymax;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			int B = (int)(maxY * src.at<Vec3b>(i, j)[0] / Blue);
			int G = (int)(maxY * src.at<Vec3b>(i, j)[1] / Green);
			int R = (int)(maxY * src.at<Vec3b>(i, j)[2] / Red);
			if (B > 255) B = 255; else if (B < 0) B = 0;
			if (G > 255) G = 255; else if (G < 0) G = 0;
			if (R > 255) R = 255; else if (R < 0) R = 0;
			dst.at<Vec3b>(i, j)[0] = B;
			dst.at<Vec3b>(i, j)[1] = G;
			dst.at<Vec3b>(i, j)[2] = R;
		}
	}
	return dst;
}
```

# 效果
图像均为算法处理前和处理后的顺序。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212433721.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212446373.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212457237.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212540682.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212553768.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191226212627785.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 附录
论文原文：http://140.112.114.62/bitstream/246246/200704191001444/1/01465458.pdf

参考文章：https://www.cnblogs.com/Imageshop/archive/2013/04/20/3032062.html


# 后记
对比前面的灰度世界算法和完美反射算法后，这个算法的效果确实要好很多，原文的内容基本上我的博客就写完了，感兴趣可以再去读读原文。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
 
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)