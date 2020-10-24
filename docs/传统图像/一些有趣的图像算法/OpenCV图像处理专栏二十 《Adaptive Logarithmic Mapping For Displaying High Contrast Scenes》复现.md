# 1. 前言
今天分享一篇2003年的低照度图像增强论文《Adaptive Logarithmic Mapping For Displaying High Contrast Scenes》，论文地址为：`https://domino.mpi-inf.mpg.de/intranet/ag4/ag4publ.nsf/0/53A4B81D590A3EEAC1256CFD003CE441/$file/logmap.pdf`。
# 2. 原理
首先论文第一个重要的公式是：

$L_d=\frac{log(L_w+1)}{log(L_{max}+1)}$

其中$L_d$代表每个像素的显示亮度，$L_w$代表原图像亮度，$L_{max}$是原图像亮度的最大值。

第二个重要的公式是：

$log_{base}x=\frac{log(x)}{log(base)}$

这就是高中学过的换底公式了。

第三个重要公式是：

$bias_b(t)=t^{\frac{log(b)}{log(0.5)}}$

这个公式的来源为：经过实验表明人眼对亮度的适应比较符合对数曲线，为了使得对数变换变得"平滑"，使用了上述的$bias$变换。这个变换就是将一个数值$t$做一个指数变换，来达到调节的目的，当$b=0.5$时，即是$bias_b(t)=t$，当$b$取$0.73$时，得到的调整函数最接近$\gamma = 2.2$的伽马矫正结果，论文还尝试$0.65，0.75，0.85，0.95$的不同恢复结果，最后在代码实现部分选择了$0.85$，这个值看起来是最优秀的。

第4个重要的公式为：

$L_d=\frac{0.01L_{dmax}}{log_{10}(L_{wmax}+1)}*\frac{log(L_w+1)}{log(2+8(\frac{L_w}{L_{wmax}})^{\frac{log(b)}{log(0.5)}})}$

其中$L_{dmax}$是设定的一个比例因子，根据不同的显示器需要进行调整，CRT显示器可以取$L_{max}=100cm/m^2$，这部分的细节和$Bias$的取值建议去看原论文，最后作者在实现过程中使用的是将原始的`rgb`转成`ciexyz`色彩空间，然后在`ciexyz`空间进行变换后再转回`rgb`颜色空间。

按照这几个公式以及自己的理解去复现了一把，和论文展示的结果是比较类似的，应该不会有大问题。


# 3. 代码复现


```
double Transform(double x)
{
	if (x <= 0.05)return x * 2.64;
	return 1.099*pow(x, 0.9 / 2.2) - 0.099;
}
struct zxy {
	double x, y, z;
}s[2500][2500];

int work(cv::Mat input_img, cv::Mat out_img) {
	int rows = input_img.rows;
	int cols = input_img.cols;
	double r, g, b;
	double lwmax = -1.0, base = 0.75;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			b = (double)input_img.at<Vec3b>(i, j)[0] / 255.0;
			g = (double)input_img.at<Vec3b>(i, j)[1] / 255.0;
			r = (double)input_img.at<Vec3b>(i, j)[2] / 255.0;
			s[i][j].x = (0.4124*r + 0.3576*g + 0.1805*b);
			s[i][j].y = (0.2126*r + 0.7152*g + 0.0722*b);
			s[i][j].z = (0.0193*r + 0.1192*g + 0.9505*b);
			lwmax = max(lwmax, s[i][j].y);
		}
	}
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			double xx = s[i][j].x / (s[i][j].x + s[i][j].y + s[i][j].z);
			double yy = s[i][j].y / (s[i][j].x + s[i][j].y + s[i][j].z);
			double tp = s[i][j].y;
			//修改CIE:X,Y,Z
			s[i][j].y = 1.0 * log(s[i][j].y + 1) / log(2 + 8.0*pow((s[i][j].y / lwmax), log(base) / log(0.5))) / log10(lwmax + 1);
			double x = s[i][j].y / yy*xx;
			double y = s[i][j].y;
			double z = s[i][j].y / yy*(1 - xx - yy);

			//转化为用RGB表示
			r = 3.2410*x - 1.5374*y - 0.4986*z;
			g = -0.9692*x + 1.8760*y + 0.0416*z;
			b = 0.0556*x - 0.2040*y + 1.0570*z;

			if (r < 0)r = 0; if (r>1)r = 1;
			if (g < 0)g = 0; if (g>1)g = 1;
			if (b < 0)b = 0; if (b>1)b = 1;

			//修正补偿
			r = Transform(r), g = Transform(g), b = Transform(b);
			out_img.at<Vec3b>(i, j)[0] = int(b * 255);
			out_img.at<Vec3b>(i, j)[1] = int(g * 255);
			out_img.at<Vec3b>(i, j)[2] = int(r * 255);
		}
	}
	return 0;
}
```

# 4. 效果
![原图1](https://img-blog.csdnimg.cn/20200401213946852.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![结果图1](https://img-blog.csdnimg.cn/20200401213916630.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)



![原图2](https://img-blog.csdnimg.cn/20200401214038606.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)
![结果图2](https://img-blog.csdnimg.cn/20200401214057367.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)
对于常态的图片，一般也能起到一定的视觉增强效果：


![原图3](https://img-blog.csdnimg.cn/20200401214118886.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![结果图3](https://img-blog.csdnimg.cn/20200401214211619.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

# 5. 总结

从结果图看出效果还是很不错的，这里提供了朴素实现的代码，其实我也有优化版，这里先卖个关子哈。请持续关注本公众号吧，我决心会在图像算法优化上下很大的功夫，希望和大家一起进步。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)