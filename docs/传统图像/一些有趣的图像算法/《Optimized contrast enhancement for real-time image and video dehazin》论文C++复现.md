# 前言
本文提出了一种针对含有雾的图像和视频快速、完善的去雾算法。观察发现有雾的图像普遍具有低对比度，我们通过增强对比度来修复图像。然后多度的增强这些低对比度会截断像素值以及导致信息丢失。因此，我们引入一个包含对比项以及信息丢失项的损失函数。通过最小化损失函数，该算法不仅增强了对比度而且有效的保留了图像信息。另外，我们将图片去雾算法扩展到视频去雾。我们通过计算透射率的相关性减少对视频去雾时的闪烁程度。实验证明该算法去雾的有效性以及快速地进行实时去雾。
# 算法原理
一般，图像去雾问题可以用一个雾形成模型来描述：$I(p)=t(p)J(p)+(1−t(p))A$,其中，$J(p)=(Jr(p),Jg(p),Jb(p))$代表原始图像(R,G,B)3个通道，$I(p)=(Ir(p),Ig(p),Ib(p))$代表有误的图像。$A=(A_r,A_g,A_b)$是全球大气光值，它表示周围环境的大气光。$t(p)∈[0,1]$是反射光的透射率, 由场景点到照相机镜头之间的距离所决定。因为光传播的距离越远，光就越分散且越发减弱。所以上面这个公式的意思就是，本来没有被雾所笼罩的图像 J 与大气光 A 按一定比例进行混合后就得到我们最终所观察到的有雾图像。大气光 A 通常用图像中最明亮的颜色来作为估计。因为大量的灰霾通常会导致一个发亮（发白）的颜色。然而，在这个框架下，那些颜色比大气光更加明亮的物体通常会被选中，因而便会导致一个本来不应该作为大气光参考值的结果被用作大气光的估计。为了更加可靠的对大气光进行估计，算法的作者利用了这样一个事实：通常，那些灰蒙蒙的区域（也就是天空）中像素的方差（或者变动）总体来说就比较小。基于这个认识，算法的作者提出了一个基于四叉树子空间划分的层次搜索方法。如下图所示，我们首先把输入图像划分成四个矩形区域。然后，为每个子区域进行评分，这个评分的计算方法是“用区域内像素的平均值减去这些像素的标准差”。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181209214301834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
并记下来，选择具有最高得分的区域，并将其继续划分为更小的四个子矩形。我们重复这个过程直到被选中的区域小于某个提前指定的阈值。例如下图中的红色部分就是最终被选定的区域。在这被选定的区域里，我们选择使得距离 $||(Ir(p),Ig(p),Ib(p))−(255,255,255)||$ 最小化的颜色（包含 r,g,b 三个分量）来作为大气光的参考值。注意，这样做的意义在于我们希望选择那个离纯白色最近的颜色（也就是最亮的颜色）来作为大气光的参考值。
我们假设在一个局部的小范围内，场景深度是相同的（也就是场景内的各点到相机镜头的距离相同），所以在一个小块内（例如32×32）我们就可以使用一个固定的透射率 t，所以前面给出的有雾图像与原始（没有雾的）图像之间的关系模型就可以改写为$J(p)=\frac{1}{t}(I(p)−A)+A$，所以，在求得大气光 A 的估计值之后，我们希望复原得到的原始（没有雾的）图像 $J(p)$ 将依赖于透射率$t$。
总的来说，一个有雾的块内，对比度都是比较低的，而被恢复的块内的对比度则随着$t$的估计值的变小而增大，我们将设法来估计一个最优的$t$值，从而使得去雾后的块能够得到最大的对比度。
下面我写的计算大气光值A(4叉树递归)C++代码：


```
//计算大气光值
vector <int> m_anAirlight;
void AirlightEstimation(cv::Mat src)
{
    int nMinDistance = 65536;
    int nDistance;
    int nMaxIndex;
    double dpScore[3];
    float afScore[4] = {0};
    float nMaxScore = 0;
    int cols = src.cols;
    int rows = src.rows;
    //4 sub-block
    Mat R = Mat(rows / 2, cols / 2, CV_8UC1);
    Mat G = Mat(rows / 2, cols / 2, CV_8UC1);
    Mat B = Mat(rows / 2, cols / 2, CV_8UC1);
    Rect temp1(0, 0, cols / 2, rows / 2);
    Mat UpperLeft = src(temp1);
    Rect temp2(cols / 2, 0, cols / 2, rows / 2);
    Mat UpperRight = src(temp2);
    Rect temp3(0, rows / 2, cols / 2, rows / 2);
    Mat LowerLeft = src(temp3);
    Rect temp4(cols / 2, rows / 2, cols / 2, rows / 2);
    Mat LowerRight = src(temp4);
    if(rows * cols > 200){
        vector <Mat> channels;
        //upper left sub-block
        split(UpperLeft, channels);

        B = channels[0];
        G = channels[1];
        R = channels[2];
        Mat tmp_m, tmp_std;
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[0] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        nMaxScore = afScore[0];
        nMaxIndex = 0;
        //upper right sub-block
        split(UpperRight, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[1] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[1] > nMaxScore){
            nMaxScore = afScore[1];
            nMaxIndex = 1;
        }
        //lower left sub-block
        split(LowerLeft, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[2] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[2] > nMaxScore){
            nMaxScore = afScore[2];
            nMaxIndex = 2;
        }
        //lower right sub-block
        split(LowerRight, channels);
        B = channels[0];
        G = channels[1];
        R = channels[2];
        meanStdDev(R, tmp_m, tmp_std);
        dpScore[0] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(G, tmp_m, tmp_std);
        dpScore[1] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        meanStdDev(B, tmp_m, tmp_std);
        dpScore[2] = tmp_m.at<double>(0,0) - tmp_std.at<double>(0,0);
        afScore[3] = (float)(dpScore[0] + dpScore[1] + dpScore[2]);
        if(afScore[3] > nMaxScore){
            nMaxScore = afScore[3];
            nMaxIndex = 3;
        }
        //select the sub-block, which has maximum score

        switch (nMaxIndex){
            case 0:
                AirlightEstimation(UpperLeft); break;
            case 1:
                AirlightEstimation(UpperRight); break;
            case 2:
                AirlightEstimation(LowerLeft); break;
            case 3:
                AirlightEstimation(LowerRight); break;
        }
    }else{
        //在子快中寻找最亮的点作为A
        printf("%d %d\n", src.rows, src.cols);
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                nDistance = int(sqrt(float(255 - src.at<Vec3b>(i, j)[0]) * float(255 - src.at<Vec3b>(i, j)[0])) +
                        sqrt(float(255 - src.at<Vec3b>(i, j)[1]) * float(255 - src.at<Vec3b>(i, j)[1])) +
                                        sqrt(float(255 - src.at<Vec3b>(i, j)[2]) * float(255 - src.at<Vec3b>(i, j)[2])));
                if(nMinDistance > nDistance){
                    m_anAirlight.clear();
                    nMinDistance = nDistance;
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[0]);
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[1]);
                    m_anAirlight.push_back(src.at<Vec3b>(i, j)[2]);
                }
            }
        }
        printf("success\n");
    }
}
```

计算出了A，接下来就是如何计算透射率$t$的问题了。我们首先给出图像对比度度量的方法（论文中，原作者给出了三个对比度定义式，我们只讨论其中第一个）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181209215631945.png)
其中$c\in{r,g,b}$是颜色通道，$I_c$是$I_c(p)$的均值，$p=1,...,N$为图像中像素的个数。且有：$J(p)=\frac{1}{t}(I(p)−A)+A$。由上2式可知，t越小，还原图片的对比度越高，反之越低。Fig.4 表明如果输入像素值在[α,β]范围可以映射到[0, 255]的输出。红色部分为阶段区域，t越小[α,β]范围越小。在图像中，对于有雾的部分减少t获取较高的对比度，对无雾的部分增加t减少图像信息损失。Fig.5 显示了不同的t值对还原图像的效果。

![](https://img-blog.csdnimg.cn/20181209221335437.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)**Fig. 4. An example of the transformation function. Input pixel values are mapped to output pixel values according to the transformation function, depicted by the black line. The red regions represent the information loss due to the truncation of output pixel values. (For interpretation of the references to color in this figure legend, the reader is referred to the web version of this article.)** 
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181209221348603.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)**Fig. 5. Relationship between the transmission value and the information loss. A smaller transmission value causes more severe truncation of pixel values and a larger amount of information loss. (a) An input hazy image. The restored dehazed images with transmission values of (b) t=0.1, (c) t=0.3, (d)t=0.5, and (e) t=0.7。**

为了解决在增强对比度的同时尽可能保留原图像信息。我们定义了两个损失函数，分别为对比度，信息丢失函数：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181209222612301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
上式中，$h_c(i)$为图像颜色c像素i所在的直方图值，$\lambda$为权重因子。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20181209223202586.png)

C++代码：

```
//计算透射率
float NFTrsEstimationColor(cv::Mat src, float lamda=5.0){
    int rows = src.rows;
    int cols = src.cols;
    int nOutR, nOutG, nOutB, nSquaredOut, nSumofOuts, nSumofSquaredOuts;
    float fTrans, fOptTrs;
    int nTrans, nSumofLoss;
    float fCost, fMinCost, fMean;
    int nNumberofPixels, nLossCount;
    fTrans = 0.4f;
    nTrans = 427;
    nNumberofPixels = rows * cols * 3;
    for(int cnt = 0; cnt < 5; cnt++){
        nSumofLoss = 0;
        nLossCount = 0;
        nSumofSquaredOuts = 0;
        nSumofOuts = 0;
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                nOutB = ((src.at<Vec3b>(i, j)[0] - m_anAirlight[0]) * nTrans + 128 * m_anAirlight[0]) >> 7; //(I-A)/t+A-->((I-A)*k*128+A*128)/128
                nOutG = ((src.at<Vec3b>(i, j)[1] - m_anAirlight[1]) * nTrans + 128 * m_anAirlight[1]) >> 7;
                nOutR = ((src.at<Vec3b>(i, j)[2] - m_anAirlight[2]) * nTrans + 128 * m_anAirlight[2]) >> 7;
                if(nOutR>255){
                    nSumofLoss += (nOutR-255)*(nOutR-255);
                    nLossCount++;
                }else if(nOutR<0){
                    nSumofLoss += nOutR*nOutR;
                    nLossCount++;
                }
                if(nOutG>255){
                    nSumofLoss += (nOutG-255)*(nOutG-255);
                    nLossCount++;
                }else if(nOutG<0){
                    nSumofLoss += nOutG*nOutG;
                    nLossCount++;
                }
                if(nOutB>255){
                    nSumofLoss += (nOutB-255)*(nOutB-255);
                    nLossCount++;
                }else if(nOutB<0){
                    nSumofLoss += nOutB*nOutB;
                    nLossCount++;
                }
                nSumofSquaredOuts += nOutB*nOutB + nOutR*nOutR + nOutG*nOutG;
                nSumofOuts += nOutB + nOutG + nOutR;
            }
        }
        fMean = (float)(nSumofOuts)/(float)(nNumberofPixels);
        fCost = lamda * (float)nSumofLoss / (float)(nNumberofPixels) - ((float)nSumofSquaredOuts/(float)nNumberofPixels-fMean*fMean);
        if(cnt == 0 || fMinCost > fCost){
            fMinCost = fCost;
            fOptTrs = fTrans;
        }
        fTrans += 0.1f;
        nTrans = (int)(1.0f/fTrans*128.0f);
    }
    return fOptTrs;
}
```
有了t也有了A就可以按照$J(p)=\frac{1}{t}(I(p)−A)+A$得到原始的RGB图像去雾后的结果了。

# 完整代码
我尝试按照上面讲的原理实现这篇论文，可是我得到的结果却很不尽如人意，产生了大量噪声。(作者的代码里用到了导向滤波，我因为下载不了论文原因还不知道导向滤波用在了哪个步骤)我这里就不贴代码了。所幸，作者给出了他的C++代码，并且风格十分优美。强烈建议直接去看作者的代码。看懂了实现也就不难了。
http://mcl.korea.ac.kr/projects/dehazing/#userconsent# 我实现的时候主要是计算透射率的时候没有像作者那样考虑，不过这里我发现将复现何凯明论文中的透射率的最小值该大，对天空进行去雾的效果也可以更好。如果要研究视频以及实时去雾的话，一定还是要去研究作者的源码和论文。

# 效果
可以看下这篇文章的效果：https://www.cnblogs.com/Imageshop/p/3925461.html

# 参考文章
https://blog.csdn.net/xx116213/article/details/51848429