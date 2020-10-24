# 前言
这是OpenCV图像处理专栏的第五篇文章，分享一下《Real-time adaptive contrast enhancement for imaging sensors》论文解读及实现，论文地址见附录。本文的算法简称为ACE算法是用来做图像对比度增强的算法。图像对比度增强的算法在很多场合都有用处，特别是在医学图像中，这是因为在众多疾病的诊断中，医学图像的视觉检查时很有必要的。而医学图像由于本身及成像条件的限制，图像的对比度很低。因此，在这个方面已经开展了很多的研究。这种增强算法一般都遵循一定的视觉原则。众所周知，人眼对高频信号（边缘处等）比较敏感。虽然细节信息往往是高频信号，但是他们时常嵌入在大量的低频背景信号中，从而使得其视觉可见性降低。因此适当的提高高频部分能够提高视觉效果并有利于诊断。
#  算法原理
一张图片，总是由低频部分和高频部分构成的，低频部分可以由图像的低通滤波来得到，而高频部分可以由原图减去低频部分来得到。而本算法的目标是增强代表细节的高频部分，即是对高频部分乘上一个系数，然后重组得到增强的图像。所以本算法的核心就是高频部分增益系数(又叫CG)的计算，一种方法是将这个系数设为一个常数，第二种方法是将增益表示为与方差相关的量。假设图像中的某个点表示为$x(i, j)$，那么以$(i, j)$为中心，窗口大小为$(2n+1)\times (2n+1)$，其局部均值和局部方差为：
 - $m_x(i, j) = \frac{1}{(2n+1)^2}\sum_{k=i-n}^{i+n}\sum_{l=j-n}^{j+n}x(k, l)$
 - $\sigma_x^2(i, j)=\frac{1}{(2n+1)^2}\sum_{k=i-n}^{i+n}\sum_{l=j-n}^{j+n}[x(k, l)-m_x(i, j)]^2$。

上面的式子中$\sigma_x(i,j)$就是所谓的局部标准差(LSD)。定义$f(i, j)$表示$x(i, j)$对应的增强后的像素值。则ACE算法可以表示为：
$f(i, j)=m_x(i, j)+G(i,j)[x(i,j)-m_x(i,j)]$
其中系数$G(i,j)$就是上面说的CG。一般情况下CG总是大于1的，这样高频部分就可以得到增强。CG的取值有2种，一种是直接取一个常数C(C>1)，这样上面的式子可以写成：
$f(i, j)=m_x(i, j)+C[x(i, j)-m_x(i, j)]$，其中C是一个大于1的数。
这种情况下，图像中所有的高频部分都被同等放大，可能有些高频部分会出现过增强的现象的。
而第二种方法是对每个位置使用不同的增益，Lee等人提出了下面的解决方案：
$f(i, j)=m_x(i, j)+\frac{D}{\sigma_x(i,j)}[x(i,j)-m_x(i,j)]$

其中D是一个常数，这样CG系数是空间自适应的，并且和局部均方差成反比，在图像的边缘或者其他变化剧烈的地方，局部均方差比较大，因此CG的值就比较小，这样就不会产生振铃效应。然而，在平滑的区域，局部均方差就会很小，这样CG的值比较大，从而引起了噪音的放大，所以需要对CG的最大值做一定的限制才能获得更好的效果。


 我们使用第二种方法，因为在图像的高频区域，局部方差较大，此时增益值就较小，这样就不会出现过亮的情况。但是在图像平滑的区域，局部均方差很小，此时增益值较大，从而可能会方法噪声信号，所以需要对增益最大值做一定的限制。D这个常数一些文章认为取图像的全局均值，而我这里参考ImageShop大牛的文章使用了全局均方差。下面给出一些代码实现和效果测试。

# 代码实现

```
//自适应对比度增强算法，C表示对高频的直接增益系数,n表示滤波半径，maxCG表示对CG做最大值限制
Mat ACE(Mat src, int C = 3, int n = 3, float MaxCG = 7.5){
    int row = src.rows;
    int col = src.cols;
    Mat meanLocal; //图像局部均值
    Mat varLocal; //图像局部方差
    Mat meanGlobal; //全局均值
    Mat varGlobal; //全局标准差
    blur(src.clone(), meanLocal, Size(n, n));
    Mat highFreq = src - meanLocal;
    varLocal = highFreq.mul(highFreq);
    varLocal.convertTo(varLocal, CV_32F);
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            varLocal.at<float>(i, j) = (float)sqrt(varLocal.at<float>(i, j));
        }
    }
    meanStdDev(src, meanGlobal, varGlobal);
    Mat gainArr = meanGlobal / varLocal; //增益系数矩阵
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            if(gainArr.at<float>(i, j) > MaxCG){
                gainArr.at<float>(i, j) = MaxCG;
            }
        }
    }
    printf("%d %d\n", row, col);
    gainArr.convertTo(gainArr, CV_8U);
    gainArr = gainArr.mul(highFreq);
    Mat dst1 = meanLocal + gainArr;
    //Mat dst2 = meanLocal + C * highFreq;
    return dst1;
}

int main(){
    Mat src = imread("../test.png");
    vector <Mat> now;
    split(src, now);
    int C = 150;
    int n = 5;
    float MaxCG = 3;
    Mat dst1 = ACE(now[0], C, n, MaxCG);
    Mat dst2 = ACE(now[1], C, n, MaxCG);
    Mat dst3 = ACE(now[2], C, n, MaxCG);
    now.clear();
    Mat dst;
    now.push_back(dst1);
    now.push_back(dst2);
    now.push_back(dst3);
    cv::merge(now, dst);
    imshow("origin", src);
    imshow("result", dst);
    imwrite("../result.jpg", dst);
    waitKey(0);
    return 0;
}
```
# 效果测试
 int C = 150;
 int n = 5;
 float MaxCG = 3;
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222155317201.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)**原图**
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222155342562.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)**结果**

    int C = 4;
    int n = 50;
    float MaxCG = 5;
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181222160212436.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
 **原图**
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/2018122216030822.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
 **效果图**


# 附录
论文地址：https://www.researchgate.net/publication/253622155_Real-Time_Adaptive_Contrast_Enhancement_For_Imaging_Sensors
参考博客：https://www.cnblogs.com/Imageshop/p/3324282.html



---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)