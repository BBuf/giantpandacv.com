# 前言
偶然在IPOL见到了这篇paper，虽然之前复现的一些paper已经可以较好的处理低照度下的色彩恢复，然而在光度强度很大的情况下怎么恢复还不清楚，并且如果出现图片中既有很亮的部分，又有很暗的部分，又不知道怎么处理了。这篇paper，正式为了解决这一问题，他的局步颜色矫正，和He KaiMing的暗通道去雾有相似的想法，值得借鉴。论文地址为：http://www.ipol.im/pub/art/2011/gl_lcc/ 。IPOL是一个非常好的学习数字图像处理的网站，上面的论文都是提供配套源码的，如果平时在数字图像处理方面想找一些Idea，不妨上去看看。
# 算法原理 
首先对于太亮和太暗的图像，我们可以使用Gamma校正和直方图均衡化来提高对比度。 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126133007795.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)						**分别代表较暗图像，Gamma系数为0.5的Gamma校正，直方图均衡化**
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019120514023074.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
						**分别代表较亮的原始图像，Gamma系数为2.5的Gamma校正，直方图均衡化**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126133241336.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
							**分别代表原始图像，Gamma系数为0.5,2.5,0.75,1.5的Gamma校正图像**

使用Gamma校正后可以提高图像的动态范围，实际上作者讲这么多实际是要说，如果当图像既有较亮又有较暗的区域时，如果仅仅使用一个Gamma矫正输出的图像效果反而会变差，这是因为Gamma矫正是全局的方法，某一部分相近的像素将被映射到相同的灰度值，并没有考虑待到像素邻域的信息。对于普通的过亮和过暗的图像，当图像的平均灰度大于127.5使用$\gamma >1$，对图像的亮度进行抑制；当图像的灰度信息均值小于127.5时使用$\gamma <1$对图像亮度进行增强。这里我们假设图像用无符号8bit表示，那么$\gamma = 2^{\frac{u-127.5}{127.5}}$。在既有较暗又有较亮的区域的图像中，全局Gamma失效，这时候作者就提出了利用图像邻域的信息，进行Gamma矫正。对较暗的区域进行增加亮度，对较亮的区域降低亮度。局部颜色校正的方法可以根据邻域内像素的灰度值情况，把统一输入像素值，映射成不同水平的像素灰度值。
# 算法步骤
- 根据输入图像计算出掩膜图像
- 结合输入图像和掩模图像计算出最终结果
掩膜图像一般根据彩色图像各个通道的图像灰度值获得。假设RGB图像各个通道的像素灰度值为R，G，B，则掩膜图像可以表示为$I = (R + G + B) / 3$，之后对掩膜图像进行高斯滤波:$M(x,y) = (Gaussian*(255-I))(x,y)$，高斯滤波时，选取较大值进行滤波，以保证对比度不会沿着边缘方向过度减小。上述的输出结果表明：图像哪部分需要提亮，哪部分需要减暗。最后输出图像为：$Output(x, y)=255(\frac{Input(x,y)}{255})^{2^{\frac{128-M(x,y)}{128}}}$，如果掩膜图像大于128，将得到一个大于1的指数，并对图像该点的亮度移植，反之增加亮度。如果等于128，则不改变该像素点亮度。
# C++代码

```
Mat LCC(const Mat &src){
    int rows = src.rows;
    int cols = src.cols;
    int **I;
    I = new int *[rows];
    for(int i = 0; i < rows; i++){
        I[i] = new int [cols];
    }
    int **inv_I;
    inv_I = new int *[rows];
    for(int i = 0; i < rows; i++){
        inv_I[i] = new int [cols];
    }
    Mat Mast(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            I[i][j] = (src.at<Vec3b>(i, j)[0] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[1]) / 3.0;
            inv_I[i][j] = 255;
            *data = inv_I[i][j] - I[i][j];
            data++;
        }
    }
    GaussianBlur(Mast, Mast, Size(41, 41), BORDER_DEFAULT);
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        uchar *data = Mast.ptr<uchar>(i);
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 3; k++){
                float Exp = pow(2, (128 - data[j]) / 128.0);
                int value = int(255 * pow(src.at<Vec3b>(i, j)[k] / 255.0, Exp));
                dst.at<Vec3b>(i, j)[k] = value;
            }
        }
    }
    return dst;
}
```
# 效果
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142124460.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142137162.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/201811261421550.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142215291.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142242689.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142254612.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142311906.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20181126142334178.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
所有的图片顺序均为**处理前**和**处理后**的顺序。

# 后记
今天介绍了一篇IPOL的对不均匀光照图像的局部校正论文，希望通过原理分析和代码实现对大家有所帮助。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)