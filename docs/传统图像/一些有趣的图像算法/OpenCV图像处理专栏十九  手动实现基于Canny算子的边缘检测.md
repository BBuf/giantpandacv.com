# 1. 前言
接着昨天手动构造Sobel算子实现检测，今天来讲讲如何手动实现Canny边缘检测。由于要实现这个算法的需要的先验知识比较多，所以在学习这个算法的实现之前我们先来学习一下用于图像二值化的OSTU大津法。
# 2. OTSU算法原理
Ostu方法又名最大类间差方法，通过统计整个图像的直方图特性来实现全局阈值$t$的自动选取。

它将图像的像素按灰度级分成$2$个部分，使得$2$个部分之间的灰度值差异最小，通过方差的计算来寻找一个合适的灰度级别来划分。

OTSU算法的计算很简单，不受到图像亮度和对比度的影响，因此使得类间方差最大的分割意味着错分概率最小。

设$t$为灰度级阈值，从$L$(一般为$255$)个灰度级遍历$t$，使得$t$为某个值的时候，前景和背景的方差最大，这个$t$值便是我们要求的灰度级阈值。

设$w0$表示分开后前景像素点占图像像素比例，$u0$表示分开后前景像素点的平均灰度，$w1$表示分开后背景像素点数占图像比例，$u1$表示分开后背景像素点的平均灰度。

图像的总灰度被定义为：？

$u=w0*u0+w1*u1..............(1)$

图像的方差计算公式为：

$g=w0*(u0-w0)*(u0-w0)+w1*(u1-w1)*(u1-w1)...............(2)$

并且有：

$w0+w1=1...................................(3)$

所以可以把公式(2)展开化简为：

$g=w0*w1*(u0-u1)*(u0-u1)$


公式推导完毕，我们来看一下OTSU算法的详细步骤。

# 3. OTSU算法流程

- 先计算图像的直方图，即将图像所有的像素点按照$0~255$共$256$个bin，统计落在每个bin的像素点数量。
- 归一化直方图，也即将每个bin中像素点数量除以总的像素点。
- $i$表示分类的阈值，也即一个灰度级，从$0$开始迭代。
- 通过归一化的直方图，统计$[0，i]$灰度级的像素(假设像素值在此范围的像素叫做前景像素) 所占整幅图像的比例$w0$，并统计前景像素的平均灰度$u0$。然后统计$[i，255]$灰度级的像素(假设像素值在此范围的像素叫做背景像素) 所占整幅图像的比例$w1$，并统计背景像素的平均灰度$u1$。
- 计算前景像素和背景像素的方差$g=w0*w1*(u0-u1)*(u0-u1)$（已经推导了）。
- 不断$i++$，转到第$4$个步骤。
- 将最大$g$相应的$i$值作为图像的全局阈值。

# 4. OTSU算法代码实现
```
int OTSU(Mat src){
    int row = src.rows;
    int col = src.cols;
    int PixelCount[256]={0};
    float PixelPro[256]={0};
    int total_Pixel = row * col;
    float threshold = 0;
    //统计灰度级中每个像素在整副图中的个数
    for(int i = 0; i < row; i++){
        for(int j = 0; j < col; j++){
            PixelCount[src.at<int>(i, j)]++;
        }
    }
    //计算每个像素在整副图中的个数
    for(int i = 0; i < 256; i++){
        PixelPro[i] = (float)(PixelCount[i]) / (float)(total_Pixel);
    }
    //遍历灰度级[0,255]，计算出方差最大的灰度值，为最佳阈值
    float w0, w1, u0tmp, u1tmp, u0, u1, u, deltaTmp, deltaMax = 0;
    for(int i = 0; i < 256; i++){
        w0 = w1 = u0tmp = u1tmp = u0 = u1 = u = deltaTmp = 0;
        for(int j = 0; j < 256; j++){
            if(j <= i){//以i为阈值分类，第一类总的概率
                w0 += PixelPro[j];
                u0tmp += j * PixelPro[j];
            }else{//前景部分
                w1 += PixelPro[j];
                u1tmp += j * PixelPro[j];
            }
        }
        u0 = u0tmp / w0; //第一类的平均灰度
        u1 = u1tmp / w1; //第二类的平均灰度
        u = u0 + u1; //整副图像的平均灰度
        //计算类间方差
        deltaTmp = w0 * (u0 - u) * (u0 - u) + w1 * (u1 - u) * (u1 - u);
        //找出最大类间方差以及对应阈值
        if(deltaTmp > deltaMax){
            deltaMax = deltaTmp;
            threshold = i;
        }
    }
    return threshold;
}
```

# 5. 边缘检测的一般标准
边缘检测有下面几个标准：
(1) 以低的错误率检测边缘，也即意味着需要尽可能准确的捕获图像中尽可能多的边缘。
(2) 检测到的边缘应精确定位在真实边缘的中心。
(3) 图像中给定的边缘应只被标记一次，并且在可能的情况下，图像的噪声不应产生假的边缘。

# 6. Canny算子边缘检测步骤
有了上面的铺垫，我们来到今天的主题，我们直接来看一下基于Canny算子进行边缘检测的步骤，我会在第6节详细讲解每一个步骤。基于Canny算子边缘检测的步骤如下：

- 使用高斯滤波算法，以平滑图像，滤除噪声
- 计算图像中每个像素点的梯度强度和方向
- 应用非极大值抑制，以消除边缘检测带来的杂散响应
- 应用双阈值检测来确定真正的边缘和潜在的边缘
- 通过抑制孤立的弱边缘最终完成边缘检测
# 7. 对应算法步骤的详细解释
## 7.1 高斯滤波

首先高斯函数的定义为$h(x, y) = e^{-\frac{x^2+y^2}{2\sigma^2}}$，其中$(x,y)$是图像中像素点的坐标，$\sigma$为标准差，高斯模板就是利用这个函数来计算的。

接下来看高斯模板是如何构造的，假设高斯模板的尺寸为$(2k+1)\times (2k+1)$。

为什么长宽都为奇数，这主要是保证整个模板有唯一中心元素，便于计算。

高斯模板中的元素值为：

$H_{i,j}=\frac{1}{2\pi {\sigma^2}}e^{-\frac{(i-k-1)^2+(j-k-1)^2}{2\sigma^2}}$

然后在实现生成高斯模板时，又有两种形式，即整数形式和小数形式。对于小数形式的就是按照上面的公式直接构造，不需要其他处理，而整数形式的需要归一化，即将模板左上角的值归一化为$1$。

使用整数高斯模板时，需要在模板前面加一个系数，这个系数为$\frac{1}{\sum_{(i,j)\in w}w_{i,j}}$，就是模板系数和的导数。

生成小数高斯模板代码如下：

```
#define PI 3.1415926
//生成小数形式的高斯模板
void generateGaussianTemplate(double window[][11], int ksize, double sigma){
    int center = ksize / 2; //模板的中心位置，也就是坐标原点
    double x2, y2;
    for(int i = 0; i < ksize; i++){
        x2 = pow(i - center, 2);
        for(int j = 0; j < ksize; j++){
            y2 = pow(j - center, 2);
            double g = exp(-(x2 + y2)) / (2 * sigma * sigma);
            g /= 2 * PI * sigma;
            window[i][j] = g;
        }
    }
    //归一化左上角的数为1
    double k = 1 / window[0][0];
    for(int i = 0; i < ksize; i++) {
        for (int j = 0; j < ksize; j++) {
            window[i][j] *= k;
        }
    }
}
```
整数部分基本一样，就不说了。对于$ksize=3$，$\sigma=0.8$的高斯模板结果为：

```sh
1.00000 2.71828 1.00000 
2.71828 7.38906 2.71828 
1.00000 2.71828 1.00000 
```

这里想说一下$\sigma$的作用，当$\sigma$比较小的时候，生成的高斯模板中心的系数比较大，而周围的系数比较小，这样对图像的平滑效果不明显。而当$\sigma$比较大时，生成的模板的各个系数相差就不是很大，比较类似于均值模板，对图像的平滑效果比较明显。

## 7.2 计算梯度强度和方向
利用Sobel算子返回水平$Gx$和垂直$Gy$的一阶导数值。以此来计算梯度强度$G$和梯度方向。

$G=\sqrt{G_x^2+G_y^2}$

$\theta = arctan(\frac{G_y}{G_x})$

$G_x=\sum_i\sum_jSobelx_{i,j}*img_{i,j}$

$G_y=\sum_i\sum_jSobely_{i,j}*img_{i,j}$

关于Sobel算子的构造请看昨天的推文：[OpenCV图像处理专栏十八 | 手动构造Sobel算子完成边缘检测](https://mp.weixin.qq.com/s/ISxINIo2PiA-LSVnWDK3wQ)

## 7.3 应用非极大值抑制，以消除边缘检测带来的杂散响应
非极大值抑制是一种边缘稀疏技术，作用在于**瘦边**。在对图片进行梯度计算之后，仅仅基于梯度值提取的边缘仍然比较模糊。根据**标准(3)**，对边缘有且应当只有一个准确的响应。而非极大值抑制可以帮助将将局部最大值之外的所有梯度值抑制为$0$，对梯度图像中每个像素进行非极大值抑制的算法步骤为：

(1) 将当前像素的梯度强度与沿正负梯度方向上的两个像素进行比较。

(2) 如果当前像素的梯度值与另外两个像素值相比最大，则该像素点保留为边缘点，否则该像素点被抑制。

通常为了更加精确的计算，**在跨梯度方向的两个相邻像素之间使用线性插值来得到要比较的像素梯度**。

![Fig1. 梯度方向分割例子](https://img-blog.csdnimg.cn/20200316221244178.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

如Fig1所示，将梯度分为$8$个方向，分别为E、NE、N、NW、W、SW、S、SE。其中$0$代表$[0^o,45^o]$,$1$代表$[45^o,90^o]$，$2$代表$[-90^o,-45^o]$，$3$代表$[-45^o,0^o]$。像素点$P$的梯度方向为$\theta$，则像素点$P1$和$P2$的梯度线性插值为： 

![梯度线性插值](https://img-blog.csdnimg.cn/20200316221549649.png)

因此非极大值抑制的伪代码如下：

![非极大值抑制的伪代码](https://img-blog.csdnimg.cn/20200316221630887.png)

需要注意的是，如何标志方向并不重要，重要的是梯度方向的计算要和梯度算子的选取保持一致。

## 7.4 双阈值检测
在施加非极大值抑制之后，剩余的像素可以更准确地表示图像中的实际边缘。然而，仍然存在由于噪声和颜色变化引起的一些边缘像素。为了解决这些杂散响应，必须用弱梯度值过滤边缘像素，并保留具有高梯度值的边缘像素，可以通过选择高低阈值来实现。如果边缘像素的梯度值高于高阈值，则将其标记为强边缘像素；如果边缘像素的梯度值小于高阈值并且大于低阈值，则将其标记为弱边缘像素；如果边缘像素的梯度值小于低阈值，则会被抑制。阈值的选择取决于给定输入图像的内容。

双阈值检测的伪代码如下：

![双阈值检测伪代码](https://img-blog.csdnimg.cn/20200316221921543.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

## 7.5 抑制孤立弱边缘完成边缘检测
到目前为止，被划分为强边缘的像素点已经被确定为边缘，因为它们是从图像中的真实 边缘中提取出来的。然而，对于弱边缘像素，将会有一些争论，因为这些像素可以从真实边缘提取也可以是因噪声或颜色变化引起的。为了获得准确的结果，应该抑制由后者引起的弱边缘。通常，由真实边缘引起的弱边缘像素将连接到强边缘像素，而噪声响应未连接。为了跟踪边缘连接，通过查看弱边缘像素及其$8$个邻域像素，只要其中一个为强边缘像素，则该弱边缘点就可以保留为真实的边缘。

这部分的伪代码如下：

![抑制孤立弱边缘完成边缘检测](https://img-blog.csdnimg.cn/20200316222052776.png)



# 8. C++代码实现

按照上面的步骤一步步实现基于Canny的边缘检测算法，代码如下：

```
const double PI = 3.1415926;

double getSum(Mat src){
    double sum = 0;
    for(int i = 0; i < src.rows; i++){
        for(int j = 0; j < src.cols; j++){
            sum += (double)src.at<double>(i, j);
        }
    }
    return sum;
}

Mat CannyEdgeDetection(cv::Mat src, int kSize, double hightThres, double lowThres){
//    if(src.channels() == 3){
//        cvtColor(src, src, CV_BGR2GRAY);
//    }
    cv::Rect rect;
    Mat gaussResult;
    int row = src.rows;
    int col = src.cols;
    printf("%d %d\n", row, col);
    Mat filterImg = Mat::zeros(row, col, CV_64FC1);
    src.convertTo(src, CV_64FC1);
    Mat dst = Mat::zeros(row, col, CV_64FC1);
    int gaussCenter = kSize / 2;
    double  sigma = 1;
    Mat guassKernel = Mat::zeros(kSize, kSize, CV_64FC1);
    for(int i = 0; i < kSize; i++){
        for(int j = 0; j < kSize; j++){
            guassKernel.at<double>(i, j) = (1.0 / (2.0 * PI * sigma * sigma)) * (double)exp(-(((double)pow((i - (gaussCenter+1)), 2) + (double)pow((j-(gaussCenter+1)), 2)) / (2.0*sigma*sigma)));
        }
    }
    Scalar sumValueScalar = cv::sum(guassKernel);
    double sum = sumValueScalar.val[0];
    guassKernel = guassKernel / sum;
    for(int i = gaussCenter; i < row - gaussCenter; i++){
        for(int j = gaussCenter; j < col - gaussCenter; j++){
            rect.x = j - gaussCenter;
            rect.y = i - gaussCenter;
            rect.width = kSize;
            rect.height = kSize;
            //printf("%d %d\n", i, j);
            //printf("%d %d %.5f\n", i, j, cv::sum(guassKernel.mul(src(rect))).val[0]);
            filterImg.at<double>(i, j) = cv::sum(guassKernel.mul(src(rect))).val[0];
        }
    }
    Mat gradX = Mat::zeros(row, col, CV_64FC1); //水平梯度
    Mat gradY = Mat::zeros(row, col, CV_64FC1); //垂直梯度
    Mat grad = Mat::zeros(row, col, CV_64FC1); //梯度幅值
    Mat thead = Mat::zeros(row, col, CV_64FC1); //梯度角度
    Mat whereGrad = Mat::zeros(row, col, CV_64FC1);//区域
    //x方向的Sobel算子
    Mat Sx = (cv::Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    //y方向的Sobel算子
    Mat Sy = (cv::Mat_<double >(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
    //计算梯度赋值和角度
    for(int i=1; i < row-1; i++){
        for(int j=1; j < col-1; j++){
            rect.x = j-1;
            rect.y = i-1;
            rect.width = 3;
            rect.height = 3;
            Mat rectImg = Mat::zeros(3, 3, CV_64FC1);
            filterImg(rect).copyTo(rectImg);
            //梯度和角度
            gradX.at<double>(i, j) += cv::sum(rectImg.mul(Sx)).val[0];
            gradY.at<double>(i, j) += cv::sum(rectImg.mul(Sy)).val[0];
            grad.at<double>(i, j) = sqrt(pow(gradX.at<double>(i, j), 2) + pow(gradY.at<double>(i, j), 2));
            thead.at<double>(i, j) = atan(gradY.at<double>(i, j)/gradX.at<double>(i, j));
            if(0 <= thead.at<double>(i, j) <= (PI/4.0)){
                whereGrad.at<double>(i, j) = 0;
            }else if(PI/4.0 < thead.at<double>(i, j) <= (PI/2.0)){
                whereGrad.at<double>(i, j) = 1;
            }else if(-PI/2.0 <= thead.at<double>(i, j) <= (-PI/4.0)){
                whereGrad.at<double>(i, j) = 2;
            }else if(-PI/4.0 < thead.at<double>(i, j) < 0){
                whereGrad.at<double>(i, j) = 3;
            }
        }
    }
    //printf("success\n");
    //梯度归一化
    double gradMax;
    cv::minMaxLoc(grad, &gradMax);
    if(gradMax != 0){
        grad = grad / gradMax;
    }
    //双阈值确定
    cv::Mat caculateValue = cv::Mat::zeros(row, col, CV_64FC1); //grad变成一维
    resize(grad, caculateValue, Size(1, grad.rows * grad.cols));
    cv::sort(caculateValue, caculateValue, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);//升序
    long long highIndex= row * col * hightThres;
    double highValue = caculateValue.at<double>(highIndex, 0); //最大阈值
    double lowValue = highValue * lowThres;
    //NMS
    for(int i = 1 ; i < row-1; i++ ){
        for( int j = 1; j<col-1; j++){
            // 八个方位
            double N = grad.at<double>(i-1, j);
            double NE = grad.at<double>(i-1, j+1);
            double E = grad.at<double>(i, j+1);
            double SE = grad.at<double>(i+1, j+1);
            double S = grad.at<double>(i+1, j);
            double SW = grad.at<double>(i-1, j-1);
            double W = grad.at<double>(i, j-1);
            double NW = grad.at<double>(i -1, j -1); // 区域判断，线性插值处理
            double tanThead; // tan角度
            double Gp1; // 两个方向的梯度强度
            double Gp2; // 求角度，绝对值
            tanThead = abs(tan(thead.at<double>(i,j)));
            switch ((int)whereGrad.at<double>(i,j)) {
                case 0: Gp1 = (1- tanThead) * E + tanThead * NE; Gp2 = (1- tanThead) * W + tanThead * SW; break;
                case 1: Gp1 = (1- tanThead) * N + tanThead * NE; Gp2 = (1- tanThead) * S + tanThead * SW; break;
                case 2: Gp1 = (1- tanThead) * N + tanThead * NW; Gp2 = (1- tanThead) * S + tanThead * SE; break;
                case 3: Gp1 = (1- tanThead) * W + tanThead *NW; Gp2 = (1- tanThead) * E + tanThead *SE; break;
                default: break;
            }
            // NMS -非极大值抑制和双阈值检测
            if(grad.at<double>(i, j) >= Gp1 && grad.at<double>(i, j) >= Gp2){
                //双阈值检测
                if(grad.at<double>(i, j) >= highValue){
                    grad.at<double>(i, j) = highValue;
                    dst.at<double>(i, j) = 255;
                } else if(grad.at<double>(i, j) < lowValue){
                    grad.at<double>(i, j) = 0;
                } else{
                    grad.at<double>(i, j) = lowValue;
                }
            } else{
                grad.at<double>(i, j) = 0;
            }
        }
    }
    //抑制孤立低阈值点 3*3. 找到高阈值就255
    for(int i = 1; i < row - 1; i++){
        for(int j = 1; j < col - 1; j++){
            if(grad.at<double>(i, j) == lowValue){
                //3*3 区域强度
                rect.x = i-1;
                rect.y = j-1;
                rect.width = 3;
                rect.height = 3;
                for(int x = 0; x < 3; x++){
                    for(int y = 0; y < 3; y++){
                        if(grad(rect).at<double>(x, y)==highValue){
                            dst.at<double>(i, j) = 255;
                            break;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

```
# 9. 效果

![虞姬原图](https://img-blog.csdnimg.cn/20200316224522625.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![边缘检测结果](https://img-blog.csdnimg.cn/20200316224623462.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
边缘检测的效果不如OpenCV自带的，需要再研究研究。

# 10. 结论
本文介绍了图像处理算法中最常用的边缘检测算法的原理以及一个C++复现，然而可惜的是效果没有OpenCV自带算法的效果好，并且速度也会稍慢，一定是某个细节没有处理完美，但作为科普来讲大概已经够用了，如果想实现完美的Sobel边缘检测请查看OpenCV源码或者和我一起讨论。

# 11. 参考
- https://blog.csdn.net/jmu201521121021/article/details/80622011

- https://www.cnblogs.com/techyan1990/p/7291771.html

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)