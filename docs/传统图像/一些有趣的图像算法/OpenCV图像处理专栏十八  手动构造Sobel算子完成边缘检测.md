# 1. 前言
众所周知，在传统的图像边缘检测算法中，最常用的一种算法是利用Sobel算子完成的。Sobel算子一共有$2$个，一个是检测水平边缘的算子，另一个是检测垂直边缘的算子。

# 2. Sobel算子优缺点
Sobel算子的优点是可以利用快速卷积函数，简单有效，且对领域像素位置的影响做了加权，可以降低边缘模糊程度，有较好效果。然而Sobel算子并没有基于图像的灰度信息进行处理，所以在提取图像边缘信息的时候可能不会让人视觉满意。

# 3. 手动构造Sobel算子
我们来看一下怎么构造Sobel算子？

Sobel算子是在一个坐标轴的方向进行非归一化的高斯平滑，在另外一个坐标轴方向做一个差分，$kszie\times ksize$大小的Sobel算子是由**平滑算子**和**差分算子**全卷积得到，其中$ksize$代表Sobel算子的半径，必须为奇数。

对于窗口大小为$ksize$的**非归一化Sobel平滑算子**等于$ksize-1$阶的二项式展开式的系数，而**Sobel平滑算子**等于$ksize-2$阶的二项式展开式的系数两侧补$0$，然后向前差分。

举个例子：构造一个$4$阶的Sobel非归一化的**Sobel平滑算子和Sobel差分算子**：

**Sobel平滑算子**： 取二项式的阶数为$n=3$，然后计算展开式系数为，$[C_3^0, C_3^1, C_3^2, C_3^3]$ 也即是$[1, 3, 3, 3]$，这就是$4$阶的非归一化的Sobel平滑算子。

**Sobel差分算子**：取二项式的阶数为$n=4-2=2$，然后计算二项展开式的系数，即为：$[C_2^0, C_2^1, C_2^2]$，两侧补$0$ 并且前向差分得到$[1, 1, -1, -1]$，第$5$项差分后可以直接删除。

**Sobel算子**：**将$4$阶的Sobel平滑算子和Sobel差分算子进行全卷积**，即可得到$4\times 4$的Sobel算子。

其中$x$方向的Sobel算子为：

$soble_x=\begin{bmatrix} 1 \\3\\3\\1 \end{bmatrix} * \begin{bmatrix} 1 &1 &-1& -1\end{bmatrix}=\begin{bmatrix} 1 &1&-1&-1\\3&3&-3&-3\\3&3&-3&-3\\1&1&-1&-1 
\end{bmatrix}$

而$y$方向的Sobel算子为：

$sobel_y=\begin{bmatrix} 1 &3&3&1 \end{bmatrix}*\begin{bmatrix} 1 \\1\\-1\\-1 \end{bmatrix}=\begin{bmatrix} 1 &3&3&1\\1&3&3&1\\-1&-3&-3&-1\\-1&-3&-3&-1 
\end{bmatrix}$

# 4. 代码实现

```
const int fac[9]={1, 1, 2, 6, 24, 120, 720, 5040, 40320};
//Sobel平滑算子
Mat getSmmoothKernel(int ksize){
    Mat Smooth = Mat::zeros(Size(ksize, 1), CV_32FC1);
    for(int i = 0; i < ksize; i++){
        Smooth.at<float>(0, i) = float(fac[ksize-1]/(fac[i] * fac[ksize-1-i]));
    }
    return Smooth;
}
//Sobel差分算子
Mat getDiffKernel(int ksize){
    Mat Diff = Mat::zeros(Size(ksize, 1), CV_32FC1);
    Mat preDiff = getSmmoothKernel(ksize-1);
    for(int i = 0; i < ksize; i++){
        if(i == 0){
            Diff.at<float>(0, i) = 1;
        }else if(i == ksize-1){
            Diff.at<float>(0, i) = -1;
        }else{
            Diff.at<float>(0, i) = preDiff.at<float>(0, i) - preDiff.at<float>(0, i-1);
        }
    }
    return Diff;
}
//调用filter2D实现卷积
void conv2D(InputArray src, InputArray kernel, OutputArray dst, int dep, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat kernelFlip;
    flip(kernel, kernelFlip, -1);
    filter2D(src, dst, dep, kernelFlip, anchor, 0.0, borderType);
}
//先进行垂直方向的卷积，再进行水平方向的卷积
void sepConv2D_Y_X(InputArray src, OutputArray dst, int dep, InputArray kernelY, InputArray kernelX, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat Y;
    conv2D(src, kernelY, Y, dep, anchor, borderType);
    conv2D(Y, kernelX, dst, dep, anchor, borderType);
}
//先进行水平方向的卷积，再进行垂直方向的卷积
void sepConv2D_X_Y(InputArray src, OutputArray dst, int dep, InputArray kernelX, InputArray kernelY, Point anchor=Point(-1,-1), int borderType=BORDER_DEFAULT){
    Mat X;
    conv2D(src, kernelX, X, dep, anchor, borderType);
    conv2D(X, kernelY, dst, dep, anchor, borderType);
}
//Sobel算子提取边缘信息
Mat Sobel(Mat &src, int x_flag, int y_flag, int kSize, int borderType){
    Mat Smooth = getSmmoothKernel(kSize);
    Mat Diff = getDiffKernel(kSize);
    Mat dst;
    if(x_flag){
        sepConv2D_Y_X(src, dst, CV_32FC1, Smooth.t(), Diff, Point(-1, -1), borderType);
    }else if(x_flag == 0 && y_flag){
        sepConv2D_X_Y(src, dst, CV_32FC1, Smooth, Diff.t(), Point(-1, -1), borderType);
    }
    return dst;
}
int main(){
    Mat src = imread("../lena.jpg");
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);
    Mat dst1 = Sobel(gray, 1, 0, 3, BORDER_DEFAULT);
    Mat dst2 = Sobel(gray, 0, 1, 3, BORDER_DEFAULT);
    //转8位灰度图显示
    convertScaleAbs(dst1, dst1);
    convertScaleAbs(dst2, dst2);
    imshow("origin", gray);
    imshow("result-X", dst1);
    imshow("result-Y", dst2);
    imwrite("../result.jpg", dst1);
    imwrite("../result2.jpg", dst2);
    waitKey(0);
    return 0;
}
```

# 5. 效果
![经典人物Lena](https://img-blog.csdnimg.cn/2018121517094254.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![先进行Y方向的Sobel运算，然后再进行X方向的结果](https://img-blog.csdnimg.cn/20181215171002371.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![先进行X方向的Sobel运算，然后再进行Y方向的结果](https://img-blog.csdnimg.cn/20181215171037373.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![虞姬原图](https://img-blog.csdnimg.cn/20200315212457739.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![先进行Y方向的Sobel运算，然后再进行X方向的结果](https://img-blog.csdnimg.cn/20200315212522346.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![先进行X方向的Sobel运算，然后再进行Y方向的结果](https://img-blog.csdnimg.cn/20200315212550535.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

可以看到两种不同的操作顺序会获得不完全一样的边缘检测效果。

# 6. 结论
这篇文章介绍了边缘检测是如何手动构造的，只要熟记二项式展开的系数，以此为出发点就比较好分析了。后面的源码实现也是比较朴素的实现，如果你想加速那么重心可以放在`filter2D`也即是卷积操作上，以后会来分享的。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)