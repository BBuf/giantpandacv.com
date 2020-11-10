# 前言
这是OpenCV图像处理专栏的第七篇文章，这篇文章是在之前的推文 [OpenCV图像处理专栏二 |《Local Color Correction 》论文阅读及C++复现](https://mp.weixin.qq.com/s/z7tIiD0wLikcjFwtwZaV8w)基础上进行了改进，仍然针对数字图像的光照不均衡现象进行校正。
# 算法原理
首先在《Local Color Correction》中有$O(i,j)=255[\frac{I(i,j)}{255}]^{\gamma}$指数部分为$\gamma[i,j,N(i,j)]=2^{[128-BFmask(i,j)/128]}$，具体去看上篇文章，这篇论文优化了2个地方：



- 第一，高斯模糊的mask使用双边滤波来代替，因为双边滤波的保边特性，可以更好的保持边缘信息。
- 第二，常数`2`使用$\alpha$来代替，并且是和图像内容相关的，当图像的整体平均值小于128时，使用:$\alpha=\frac{ln(I_{ave}/255)}{ln(0.5)}$计算，当平均值大于128时，使用$\alpha=\frac{ln(0.5)}{ln(I_{ave}/255)}$,意思就是说对于低对比度的图像，应该需要比较强的矫正幅度，所以$\alpha$应该偏大，反之对于高对比度的图像，只需要较弱的校正幅度。
- 但是这里有个trick，就是说对于第二条，实际上存在很大的问题，比如对于我们下面测试的原图，由于他上半部分为天空，下半部分比较暗，且基本各占一般，因此其平均值非常靠近128，因此计算出的α也非常接近1，这样如果按照改进后的算法进行处理，则基本上图像无什么变化，显然这是不符合实际的需求的，因此，个人认为作者这一改进是不合理的，还不如对所有的图像该值都取2，靠mask值来修正对比度，我实现的代码也是取了2。
# 算法实现
接下来我们实现算法需要对RGB图像进行处理，我们可以像我之前那篇推文那样对RGB通道分别处理，但是可能会存在色偏，所以可以在YUV或者CIELAB等等空间只对亮度的通道进行处理，最后再转回RGB，并且作者提出在对Y分量做处理后，再转换到RGB空间，图像会出现饱和度一定程度丢失的现象，看上去图像似乎色彩不足。所以论文提出了一个修正的公式为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181214160616438.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 代码实现

```
Mat ContrastImageCorrection(Mat src){
    int rows = src.rows;
    int cols = src.cols;
    Mat yuvImg;
    cvtColor(src, yuvImg, CV_BGR2YUV_I420);
    vector <Mat> mv;
    split(yuvImg, mv);
    Mat OldY = mv[0].clone();
//    for(int i = 0; i < rows; i++){
//        for(int j = 0; j < cols; j++){
//            mv[0].at<uchar>(i, j) = 255 - mv[0].at<uchar>(i, j);
//        }
//    }
    Mat temp;
    bilateralFilter(mv[0], temp, 9, 50, 50);
    //GaussianBlur(mv[0], temp, Size(41, 41), BORDER_DEFAULT);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float Exp = pow(2, (128 - (255 - temp.at<uchar>(i, j))) / 128.0);
            int value = int(255 * pow(OldY.at<uchar>(i, j) / 255.0, Exp));
            temp.at<uchar>(i, j) = value;
        }
    }
    Mat dst(rows, cols, CV_8UC3);
//    mv[0] = temp;
//    merge(mv, dst);
//    cvtColor(dst, dst, CV_YUV2BGRA_I420);
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++) {
            if (OldY.at<uchar>(i, j) == 0) {
                for (int k = 0; k < 3; k++) dst.at<Vec3b>(i, j)[k] = 0;
            } else {
                //channel B
                dst.at<Vec3b>(i, j)[0] =
                        (temp.at<uchar>(i, j)  * (src.at<Vec3b>(i, j)[0] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[0] - OldY.at<uchar>(i, j)) >> 1;
                //channel G
                dst.at<Vec3b>(i, j)[1] =
                        (temp.at<uchar>(i, j)  * (src.at<Vec3b>(i, j)[1] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[1] - OldY.at<uchar>(i, j)) >> 1;
                //channel R
                dst.at<Vec3b>(i, j)[2] =
                        (temp.at<uchar>(i, j) * (src.at<Vec3b>(i, j)[2] + OldY.at<uchar>(i, j)) / OldY.at<uchar>(i, j) +
                         src.at<Vec3b>(i, j)[2] - OldY.at<uchar>(i, j)) >> 1;
            }
        }
    }
//    for(int i = 0; i < rows; i++){
//        for(int j = 0; j < cols; j++){
//            for(int k = 0; k < 3; k++){
//                if(dst.at<Vec3b>(i, j)[k] < 0){
//                    dst.at<Vec3b>(i, j)[k] = 0;
//                }else if(dst.at<Vec3b>(i, j)[k] > 255){
//                    dst.at<Vec3b>(i, j)[k] = 255;
//                }
//            }
//        }
//    }
    return dst;
}

int main(){
    Mat src = imread(../1.jpg");
    Rect rect(0, 0, (src.cols-1)/2*2, (src.rows-1)/2*2); //保证长宽都是偶数
    Mat newsrc = src(rect);
    Mat dst = ContrastImageCorrection(newsrc);
    imshow("origin", newsrc);
    imshow("result", dst);
    waitKey(0);
    return 0;
}
```

# 效果

**原图**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181214160820520.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)**YUV直接转回RGB**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181214160853707.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**使用作者的修正公式**
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181214160923936.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 附录
论文原文：https://www.researchgate.net/publication/220051147_Contrast_image_correction_method

# 后记
今天就讲到这里了，希望《Contrast image correction method》 这篇论文可以帮助到大家。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)