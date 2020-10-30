# 算法原理
双边滤波是一种非线性滤波方法，是结合了图像的邻近度和像素值相似度的一种折中，在滤除噪声的同时可以保留原图的边缘信息。整个双边滤波是由两个函数构成：一个函数是由空间距离决定的滤波器系数，另外一个诗由像素差值决定的滤波器系数。整个双边滤波的公式如下：
$g(i,.j)=\frac{\sum_{k,l}f(k,l)w(i,j,k,l)}{\sum_{k,l}w(i,j,k,l)}$,权重系数$w(i,j,k,l)$取决于定义域核$d(i,j,k,l)=exp(-\frac{(i-k)^2+(j-l)^2}{2\sigma_d^{2}})$和值域核$r(i,j,k,l)=exp(-\frac{||f(i,j)-f(k,l)||^2}{2\sigma_r^{2}})$的乘积。
$w(i,j,k,l)=exp(-\frac{(i-k)^2+(j-l)^2}{2\sigma_d^{2}}-\frac{||f(i,j)-f(k,l)||^2}{2\sigma_r^{2}})$，其中第一个模板是全局模板，只需要生成就可以了。第二个模板需要对每个像素点滑动进行计算。双边滤波相对于高斯滤波多了一个高斯方差$\sigma_d$，它是给予空间分布的高斯滤波函数，所以在边缘附近，离得较远的像素不会太多影响到边缘的像素，所以可以更好的保留边缘信息，但是由于保留了过多高频信息，对于RGB图像中的高频噪声，双边滤波不能干净的滤掉，只能对低频噪声较好的滤除。
# 算法实现
OpenCV中提供了双边滤波的实现，我们直接调用即可。

    void bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT );

其中：
- InputArray src: 输入图像，可以是Mat类型，图像必须是8位或浮点型单通道、三通道的图像。
- OutputArray dst: 输出图像，和原图像有相同的尺寸和类型。
- int d: 表示在过滤过程中每个像素邻域的直径范围。如果这个值是非正数，则函数会从第五个参数sigmaSpace计算该值。
- double sigmaColor: 颜色空间过滤器的sigma值，这个参数的值越大，表明该像素邻域内有越宽广的颜色会被混合到一起，产生较大的半相等颜色区域。
- double sigmaSpace: 坐标空间中滤波器的sigma值，如果该值较大，则意味着颜色相近的较远的像素将相互影响，从而使更大的区域中足够相似的颜色获取相同的颜色。当d>0时，d指定了邻域大小且与sigmaSpace无关，否则d正比于sigmaSpace.
- int borderType=BORDER_DEFAULT: 用于推断图像外部像素的某种边界模式，有默认值BORDER_DEFAULT.
$\quad$双边滤波的复杂度相对于其他滤波器也是比较高的，我调用Opencv实现了一个可以调节双边滤波3个可变参数进行滤波的程序放在下面。
# 代码

```
#include "opencv2/opencv.hpp"
#include "iostream"
#include "algorithm"
#include "vector"
using namespace std;
using namespace cv;

const int g_ndMaxValue = 100;
const int g_nsigmaColorMaxValue = 200;
const int g_nsigmaSpaceMaxValue = 200;
int g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue;
Mat src, dst;
void on_bilateralFilterTrackbar(int, void*){
    bilateralFilter(src, dst, g_ndValue, g_nsigmaColorValue, g_nsigmaSpaceValue);
    imshow("filtering", dst);
}

int main(){
//    Mat src = imread("/home/streamax/CLionProjects/Paper/101507_588686279_15.jpg");
//    Mat dst;
//    bilateralFilter(src, dst, 9, 50, 50);
//    imshow("src", src);
//    imshow("dst", dst);
//    imwrite("../result.jpg", dst);
//    waitKey(0);
    src = imread("../101507_588686279_15.jpg");
    namedWindow("origin", WINDOW_AUTOSIZE);
    imshow("origin", src);
    g_ndValue = 9;
    g_nsigmaColorValue = 50;
    g_nsigmaSpaceValue = 50;
    namedWindow("filtering", WINDOW_AUTOSIZE);
    char dname[20];
    sprintf(dname, "Neighborhood diamter %d", g_ndMaxValue);
    char sigmaColorName[20];
    sprintf(sigmaColorName, "sigmaColor %d", g_nsigmaColorMaxValue);
    char sigmaSpaceName[20];
    sprintf(sigmaSpaceName, "sigmaSpace %d", g_nsigmaSpaceMaxValue);
    //创建轨迹条
    createTrackbar(dname, "filtering", &g_ndValue, g_ndMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_ndValue, 0);
    createTrackbar(sigmaColorName, "filtering", &g_nsigmaColorValue, g_nsigmaColorMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaColorValue, 0);
    createTrackbar(sigmaSpaceName, "filtering", &g_nsigmaSpaceValue, g_nsigmaSpaceMaxValue, on_bilateralFilterTrackbar);
    on_bilateralFilterTrackbar(g_nsigmaSpaceValue, 0);
    waitKey(0);
    return 0;
}
```