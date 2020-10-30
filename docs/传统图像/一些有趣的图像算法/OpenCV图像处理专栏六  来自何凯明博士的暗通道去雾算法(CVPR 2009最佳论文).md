# 前言
这是OpenCV图像处理专栏的第六篇文章，我们一起来看看何凯明博士这篇CVPR 2009最佳论文。这篇论文的灵感来自于作者的两个观察，第一个是在3D游戏中的雾使得作者坚信人眼有特殊的东西去感知雾，而不仅仅是靠对比度。第二个是作者阅读了之前的一篇去雾方面的论文《Single Image Dehazing》，发现这篇论文中的Dark Object Subtraction可以处理均匀的雾，但是非均匀的就处理不好，所以作者尝试在局部使用了Dark Object Subtraction，然后得到了惊人的效果。
# 原理
- 暗通道先验：首先说在绝大多数非天空的局部区域里，某一些像素总会有至少一个颜色通道具有很低的值，也就是说该区域光强是一个很小的值。所以给暗通道下了个数学定义，对于任何输入的图像J,其暗通道可以用下面的公式来表示：![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115195716365.png)
其中$J^C$表示彩色图像每个通道，$Ω(x)$表示以像素X为中心的一个窗口。要求暗通道的图像是比较容易的，先求出每个像素在3个通道的最小值，存到一个二维Mat中(灰度图)，然后做一个最小值滤波，滤波的半径由窗口大小决定，这里窗口大小为$WindowSize$，公式表示为$WindowsSize=2*Radius+1$，其中$Radius$表示滤波半径。
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/2018111520041669.png) 
暗通道先验理论得出的结论，这个我不知道如何证明，不过论文给出了几个原因：
- a)汽车、建筑物和城市中玻璃窗户的阴影，或者是树叶、树与岩石等自然景观的投影；
- b)色彩鲜艳的物体或表面，在RGB的三个通道中有些通道的值很低（比如绿色的草地／树／植物，红色或黄色的花朵／叶子，或者蓝色的水面）；
- c)颜色较暗的物体或者表面，例如灰暗色的树干和石头。

总之，自然景物中到处都是阴影或者彩色，这些景物的图像的暗原色总是很灰暗的。作者在论文中，统计了5000多副图像的特征，也都基本符合这个先验。因此，我们可以认为它是一条定理。

- 基于这个先验，就是该论文中最核心的部分了。首先，在计算机视觉和图像处理中，下面这个雾生成模型被广泛的应用：$I(x)=J(x)t(x)+A(1-t(x))$，其中$I(x)$是我们待处理的图像，$J(x)$是我们要恢复的没有雾的图像，$A$是全球大气光成分，$t(x)$为透射率。现在已知了$I(X)$，我们需要求取$J(X)$，显然这个不定方程有无数解，所以还需要定义一些先验。
- 将上式处理变形得到:$\frac{I^c(x)}{A^c}=t(x)\frac{J^c(x)}{A^c}+1-t(x)$，其中上标$c$代表$R、G、B$三个通道。然后假设在每一个窗口中透射率$t(x)$是一个常数，定义为$\hat{t(x)}$并且$A$值已经给定，然后对这个式子左右两边同时取2次最小值，得到下面的式子：
- ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115201846184.png)
其中$\hat{t(x)}$就是公式(8)中那个$t(x)$部分，因为我不知道怎么用markdown语法写这个符号。
上式中，$J$是待求的无雾的图像，根据前述的暗原色先验理论有：![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115202255705.png) 
因此，可推导出：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115202313428.png)  
把式(10)带入式(8)中，得到：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115202331971.png)
这就是透射率$\hat{t(x)}$的预估值。
 在现实生活中，即使是晴天白云，空气中也存在着一些颗粒，因此，看远处的物体还是能感觉到雾的影响，另外，雾的存在让人类感到景深的存在，因此，有必要在去雾的时候保留一定程度的雾，这可以通过在式（11）中引入一个在[0,1] 之间的系数，则式（11）被修正为：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115202438127.png) 
 本推文中所有的测试结果依赖于：  $ω=0.95$。
- 上述的推导是基于A已知的情况下，然而事实是A还不知道呢？A怎么计算呢？在实际中，我们可以借助于暗通道图来从有雾图像中获取该值。具体步骤如下：(1)从暗通道图中按照亮度的大小取前0.1%的像素。(2)在这些位置中，在原始有雾图像$I$中寻找对应的具有最高亮度的点的值，作为A值。  到这一步，我们就可以进行无雾图像的恢复了。由$I(x)=J(x)t(x)+A(1-t(x))$，推出 $J(x)=(I(x)-A)/t(x)+A$，现在$I、A、t$都已经求得了，因此，完全进行出J，也就是去雾后的图像了。当投射图$t$的值很小时，会导致$J$的值偏大，从而使得图像整体向白场过度，因此一般可设置一阈值$t_0$，当$t$值小于$t_0$时，令$t=t_0$，本推文中所有效果图均以$t_0=0.1$为标准计算得来。 
- 最终的结果计算表示为：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181115203002367.png)
按照上面的公式复现了论文，给几张图片测试结果，都是**原图和算法处理后的图**这样的顺序：
![原图1](https://img-blog.csdnimg.cn/20181119140950542.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![去雾后额图片](https://img-blog.csdnimg.cn/20181120113921207.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120114643152.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120114023509.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)[](https://img-blog.csdnimg.cn/20181119141043379.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![原图3](https://img-blog.csdnimg.cn/20181119141157264.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20181120114054245.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 代码实现

```
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace cv;
using namespace std;

int rows, cols;
//获取最小值矩阵
int **getMinChannel(cv::Mat img){
    rows = img.rows;
    cols = img.cols;
    if(img.channels() != 3){
        fprintf(stderr, "Input Error!");
        exit(-1);
    }
    int **imgGray;
    imgGray = new int *[rows];
    for(int i = 0; i < rows; i++){
        imgGray[i] = new int [cols];
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            int loacalMin = 255;
            for(int k = 0; k < 3; k++){
                if(img.at<Vec3b>(i, j)[k] < loacalMin){
                    loacalMin = img.at<Vec3b>(i, j)[k];
                }
            }
            imgGray[i][j] = loacalMin;
        }
    }
    return imgGray;
}

//求暗通道
int **getDarkChannel(int **img, int blockSize = 3){
    if(blockSize%2 == 0 || blockSize < 3){
        fprintf(stderr, "blockSize is not odd or too small!");
        exit(-1);
    }
    //计算pool Size
    int poolSize = (blockSize - 1) / 2;
    int newHeight = rows + blockSize - 1;
    int newWidth = cols + blockSize - 1;
    int **imgMiddle;
    imgMiddle = new int *[newHeight];
    for(int i = 0; i < newHeight; i++){
        imgMiddle[i] = new int [newWidth];
    }
    for(int i = 0; i < newHeight; i++){
        for(int j = 0; j < newWidth; j++){
            if(i < rows && j < cols){
                imgMiddle[i][j] = img[i][j];
            }else{
                imgMiddle[i][j] = 255;
            }
        }
    }
    int **imgDark;
    imgDark = new int *[rows];
    for(int i = 0; i < rows; i++){
        imgDark[i] = new int [cols];
    }
    int localMin = 255;
    for(int i = poolSize; i < newHeight - poolSize; i++){
        for(int j = poolSize; j < newWidth - poolSize; j++){
            localMin = 255;
            for(int k = i-poolSize; k < i+poolSize+1; k++){
                for(int l = j-poolSize; l < j+poolSize+1; l++){
                    if(imgMiddle[k][l] < localMin){
                        localMin = imgMiddle[k][l];
                    }
                }
            }
            imgDark[i-poolSize][j-poolSize] = localMin;
        }
    }
    return imgDark;
}

struct node{
    int x, y, val;
    node(){}
    node(int _x, int _y, int _val):x(_x),y(_y),val(_val){}
    bool operator<(const node &rhs){
        return val > rhs.val;
    }
};

//估算全局大气光值
int getGlobalAtmosphericLightValue(int **darkChannel, cv::Mat img, bool meanMode = false, float percent = 0.001){
    int size = rows * cols;
    std::vector <node> nodes;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            node tmp;
            tmp.x = i, tmp.y = j, tmp.val = darkChannel[i][j];
            nodes.push_back(tmp);
        }
    }
    sort(nodes.begin(), nodes.end());
    int atmosphericLight = 0;
    if(int(percent*size) == 0){
        for(int i = 0; i < 3; i++){
            if(img.at<Vec3b>(nodes[0].x, nodes[0].y)[i] > atmosphericLight){
                atmosphericLight = img.at<Vec3b>(nodes[0].x, nodes[0].y)[i];
            }
        }
    }
    //开启均值模式
    if(meanMode == true){
        int sum = 0;
        for(int i = 0; i < int(percent*size); i++){
            for(int j = 0; j < 3; j++){
                sum = sum + img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
            }
        }
    }
    //获取暗通道在前0.1%的位置的像素点在原图像中的最高亮度值
    for(int i = 0; i < int(percent*size); i++){
        for(int j = 0; j < 3; j++){
            if(img.at<Vec3b>(nodes[i].x, nodes[i].y)[j] > atmosphericLight){
                atmosphericLight = img.at<Vec3b>(nodes[i].x, nodes[i].y)[j];
            }
        }
    }
    return atmosphericLight;
}

//恢复原图像
// Omega 去雾比例 参数
//t0 最小透射率值
cv::Mat getRecoverScene(cv::Mat img, float omega=0.95, float t0=0.1, int blockSize=15, bool meanModel=false, float percent=0.001){
    int** imgGray = getMinChannel(img);
    int **imgDark = getDarkChannel(imgGray, blockSize=blockSize);
    int atmosphericLight = getGlobalAtmosphericLightValue(imgDark, img, meanModel=meanModel, percent=percent);
    float **imgDark2, **transmission;
    imgDark2 = new float *[rows];
    for(int i = 0; i < rows; i++){
        imgDark2[i] = new float [cols];
    }
    transmission = new float *[rows];
    for(int i = 0; i < rows; i++){
        transmission[i] = new float [cols];
    }
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            imgDark2[i][j] = float(imgDark[i][j]);
            transmission[i][j] = 1 - omega * imgDark[i][j] / atmosphericLight;
            if(transmission[i][j] < 0.1){
                transmission[i][j] = 0.1;
            }
        }
    }
    cv::Mat dst(img.rows, img.cols, CV_8UC3);
    for(int channel = 0; channel < 3; channel++){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                int temp = (img.at<Vec3b>(i, j)[channel] - atmosphericLight) / transmission[i][j] + atmosphericLight;
                if(temp > 255){
                    temp = 255;
                }
                if(temp < 0){
                    temp = 0;
                }
                dst.at<Vec3b>(i, j)[channel] = temp;
            }
        }
    }
    return dst;
}

int main(){
    cv::Mat src = cv::imread("/home/zxy/CLionProjects/Acmtest/4.jpg");
    rows = src.rows;
    cols = src.cols;
    cv::Mat dst = getRecoverScene(src);
    cv::imshow("origin", src);
    cv::imshow("result", dst);
    cv::imwrite("../zxy.jpg", dst);
    waitKey(0);
}
```

# 参考文章
论文原文：https://ieeexplore.ieee.org/document/5567108
参考博客：https://www.cnblogs.com/Imageshop/p/3281703.html
我的github链接：https://github.com/BBuf/Image-processing-algorithm
# 后记
关于何凯明博士的暗通道去雾算法就介绍到这里了，希望对你有帮助。代码也可以在我的github获取哦。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。
![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)