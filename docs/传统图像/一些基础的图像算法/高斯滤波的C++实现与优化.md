# 高斯模板
$\quad$首先高斯函数的定义为$h(x, y) = e^{-\frac{x^2+y^2}{2\sigma^2}}$，其中(x,y)是图像中的点的坐标，$\sigma$为标准差，高斯模板就是利用这个函数来计算的，我们来看高斯模板，假设大小为(2k+1)*(2k+1)为什么长宽都为奇数，这主要是保证整个模板有唯一中心元素，便于计算。高斯模板中的元素值为:$H_{i,j}=\frac{1}{2\pi {\sigma^2}}e^{-\frac{(i-k-1)^2+(j-k-1)^2}{2\sigma^2}}$，然后在实现生成高斯模板时，又有两种形式，即整数和小数，对于小数形式的就是按照公式直接计算，不需要其他处理，而整数形式的需要归一化，将模板左上角的值归一化为1,。使用整数的模板时，需要在模板前面加一个系数，这个系数为$\frac{1}{\sum_{(i,j)\in w}w_{i,j}}$，就是模板系数和的导数。
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
整数部分基本一样，就不说了。对于$ksize=3$，$\sigma=0.8$的高斯模板结果为，
1.00000 2.71828 1.00000 
2.71828 7.38906 2.71828 
1.00000 2.71828 1.00000 
这里想说一下$\sigma$的作用，当$\sigma$比较小的时候，生成的高斯模板中心的系数比较大，而周围的系数比较小，这样对图像的平滑效果不明显。而当$\sigma$比较大时，生成的模板的各个系数相差就不是很大，比较类似于均值模板，对图像的平滑效果比较明显。

# 高斯滤波
1、按照公式暴力高斯滤波

```
//O(m * n * ksize^2)
void GaussianFilter(const Mat &src, Mat &dst, int ksize, double sigma)
{
    CV_Assert(src.channels() || src.channels() == 3); //只处理3通道或单通道的图片
    double **GaussianTemplate = new double *[ksize];
    for(int i = 0; i < ksize; i++){
        GaussianTemplate[i] = new double [ksize];
    }
    generateGaussianTemplate(GaussianTemplate, ksize, sigma);
    //padding
    int border = ksize / 2;
    copyMakeBorder(src, dst, border, border, border, border, BORDER_CONSTANT);
    int channels = dst.channels();
    int rows = dst.rows - border;
    int cols = dst.cols - border;
    for(int i = border; i < rows; i++){
        for(int j = border; j< cols; j++){
            double sum[3] = {0};
            for(int a = -border; a <= border; a++){
                for(int b = -border; b <= border; b++){
                    if(channels == 1){
                        sum[0] += GaussianTemplate[border+a][border+b] * dst.at<uchar>(i+a, j+b);
                    }else if(channels == 3){
                        Vec3b rgb = dst.at<Vec3b>(i+a, j+b);
                        auto k = GaussianTemplate[border+a][border+b];
                        sum[0] += k * rgb[0];
                        sum[1] += k * rgb[1];
                        sum[2] += k * rgb[2];
                    }
                }
            }
            for(int k = 0; k < channels; k++){
                if(sum[k] < 0) sum[k] = 0;
                else if(sum[k] > 255) sum[k] = 255;
            }
            if(channels == 1){
                dst.at<uchar >(i, j) = static_cast<uchar >(sum[0]);
            }else if(channels == 3){
                Vec3b rgb = {static_cast<uchar>(sum[0]), static_cast<uchar>(sum[1]), static_cast<uchar>(sum[2])};
                dst.at<Vec3b>(i, j) = rgb;
            }
        }
    }
    for(int i = 0; i < ksize; i++)
        delete[] GaussianTemplate[i];
    delete[] GaussianTemplate;
}
```
2、分两步进行高斯滤波，先在水平方向高斯滤波，然后垂直方向高斯滤波，在核的size较大的情况下有很大的优化作用，代码如下

```
//分离实现高斯滤波
//O(m*n*k)
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
```