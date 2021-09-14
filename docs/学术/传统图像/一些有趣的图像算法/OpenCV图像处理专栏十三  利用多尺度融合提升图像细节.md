# 前言
今天为大家介绍一个利用多尺度来提升图像细节的算法。这个算法来自于论文《DARK IMAGE ENHANCEMENT BASED ON PAIRWISE TARGET CONTRAST AND MULTI-SCALE DETAIL BOOSTING》，如果你想自己的图片细节看起来更加丰富的话可以尝试一下这个算法。


# 算法原理
核心就是，论文使用了Retinex方法类似的思路，使用了多个尺度的高斯核对原图滤波，然后再和原图做减法，获得不同程度的细节信息，然后通过一定的组合方式把这些细节信息融合到原图中，从而得到加强原图信息的能力。公式十分简单，注意到第一个系数有点特殊，实现的话，直接看下图的几个公式即可。

**从深度学习中特征金字塔网络的思想来看，这个算法实际上就是将不同尺度上的特征图进行了融合，不过这个方式是直接针对原图进行，比较粗暴，但有个好处就是这个算法用于预处理阶段是易于优化的，关于如何优化后面讲SSE指令集优化的时候再来讨论，今天先提供原始的实现啦。**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181214174426609.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 代码实现

```
void separateGaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma){
    CV_Assert(src.channels()==1 || src.channels() == 3); //只处理单通道或者三通道图像
    //生成一维的
    double *matrix = new double[ksize];
    double sum = 0;
    int origin = ksize / 2;
    for(int i = 0; i < ksize; i++){
        double g = exp(-(i-origin) * (i-origin) / (2 * sigma * sigma));
        sum += g;
        matrix[i] = g;
    }
    for(int i = 0; i < ksize; i++) matrix[i] /= sum;
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BORDER_CONSTANT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    //水平方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i, j+k);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i, j+k);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    //竖直方向
    for(int i = border; i < rows; i++){
        for(int j = border; j < cols; j++){
            double sum[3] = {0};
            for(int k = -border; k<=border; k++){
                if(channels == 1){
                    sum[0] += matrix[border + k] * dst.at<uchar>(i+k, j);
                }else if(channels == 3){
                    Vec3b rgb = dst.at<Vec3b>(i+k, j);
                    sum[0] += matrix[border+k] * rgb[0];
                    sum[1] += matrix[border+k] * rgb[1];
                    sum[2] += matrix[border+k] * rgb[2];
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1)
                dst.at<Vec3b>(i, j) = static_cast<uchar>(sum[0]);
            else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    delete [] matrix;
}

Mat MultiScaleDetailBoosting(Mat src, int Radius){
    int rows = src.rows;
    int cols = src.cols;
    Mat B1, B2, B3;
    separateGaussianFilter(src, B1, Radius, 1.0);
    separateGaussianFilter(src, B2, Radius*2-1, 2.0);
    separateGaussianFilter(src, B3, Radius*4-1, 4.0);
    float w1 = 0.5, w2 = 0.5, w3 = 0.25;
    Mat dst(rows, cols, CV_8UC3);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            for(int k = 0; k < 3; k++){
                int D1 = src.at<Vec3b>(i, j)[k] - B1.at<Vec3b>(i, j)[k];
                int D2 = B1.at<Vec3b>(i, j)[k] - B2.at<Vec3b>(i, j)[k];
                int D3 = B2.at<Vec3b>(i, j)[k] - B3.at<Vec3b>(i, j)[k];
                int sign = D1 > 0 ? 1 : -1;
                dst.at<Vec3b>(i, j)[k] = saturate_cast<uchar>((1 - w1*sign) * D1 - w2 * D2 + w3 * D3 + src.at<Vec3b>(i, j)[k]);
            }
        }
    }
    return dst;
}
```

# 效果

![原图](https://img-blog.csdnimg.cn/20181214174846124.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![效果图](https://img-blog.csdnimg.cn/20181214174905452.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

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

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)