> 【GiantPandaCV导语】**卷积操作在深度学习中的重要性,想必大家都很清楚了。接下来将通过图解的方式,使用cpp一步一步从简单到复杂来实现卷积操作。**

- **符号约定**

**F**为输入;

**width**为输入的宽;

**height**为输入的高;

**channel**为输入的通道;

**K**为kernel;

**kSizeX**为kernel的宽;

**kSizeY**为kernel的高;

**filters**为kernel的个数;

**O**为输出;

**outWidth**为输出的宽;

**outHeight**为输出的高;

**outChannel**为输出的通道;
<br/>

- **卷积输出尺寸计算公式**

  $W_2=\frac{W_1+2*padding - (dilation*(ksize-1)+1)}{stride}+1$
  <br/>

- **1. 最简单的3x3卷积**

  首先, 我们不考虑任何padding, stride, 多维度等情况,来一个最简单的3x3卷积操作.计算思路很简单, 对应元素相乘最后相加即可.此处:

  - **width=3**
  - **height=3**
  - **channel=1**
  - **paddingX=0**
  - **paddingY=0**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=1**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=1**
  - **outHeight=1**
  - **outChannel=1**

![图1 最简单的3x3卷积](https://img-blog.csdnimg.cn/2020123122425830.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


  cpp代码:

```
void demo0()
{
    float F[] = {1,2,3,4,5,6,7,8,9};
    float K[] = {1,2,3,4,5,6,7,8,9};
    float O = 0;

    int width  = 3;
    int height = 3;
    int kSizeX = 3;
    int kSizeY = 3;

    for(int m=0;m<kSizeY;m++)
    {
        for(int n=0;n<kSizeX;n++)
        {
            O+=K[m*kSizeX+n]*F[m*width+n];
        }
    }

    std::cout<<O<<" ";
}
```

<br/>

- **2. 最简单卷积(1)**

  接下来考虑能适用于任何尺寸的简单卷积, 如输入为4x4x1, kernel为3x3x1. 这里考虑卷积步长为1, 则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=1**
  - **paddingX=0**
  - **paddingY=0**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=1**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=1**
  
    ![](https://img-blog.csdnimg.cn/img_convert/da90224e961fdd33d113d4a8b4d23497.png)

    <center>图2 最简单卷积(1)</center>

  cpp代码:

```
void demo1()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4,5,6,7,8,9};
    float O[] = {0,0,0,0};

    int padX = 0;
    int padY = 0;

    int dilationX = 1;
    int dilationY = 1;

    int strideX  = 1;
    int strideY  = 1;

    int width = 4;
    int height = 4;

    int kSizeX = 3;
    int kSizeY = 3;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    for(int i=0;i<outH;i++)
    {
        for(int j=0;j<outW;j++)
        {
            for(int m=0;m<kSizeY;m++)
            {
                for(int n=0;n<kSizeX;n++)
                {
                    O[i*outW+j]+=K[m*kSizeX+n]*F[(m+i)*width+(n+j)];
                }
            }
        }
    }

    for (int i = 0; i < outH; ++i)
    {
        for (int j = 0; j < outW; ++j)
        {
            std::cout<<O[i*outW+j]<<" ";
        }
        std::cout<<std::endl;
    }
}
```

<br/>

- **3. 最简单卷积(2)**

  接下来考虑在步长上为任意步长的卷积,  如输入为4x4x1, kernel为2x2x1. 这里考虑卷积步长为2, 则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=1**
  - **paddingX=0**
  - **paddingY=0**
  - **strideX=2**
  - **strideY=2**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=2**
  - **kSizeY=2**
  - **filters=1**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=1**

![图3 最简单卷积(2)](https://img-blog.csdnimg.cn/20201231224216218.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)



  cpp代码:

```
void demo2()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4};
    float O[] = {0,0,0,0};

    int padX = 0;
    int padY = 0;

    int dilationX = 1;
    int dilationY = 1;

    int strideX  = 2;
    int strideY  = 2;

    int width = 4;
    int height = 4;

    int kSizeX = 2;
    int kSizeY = 2;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    for(int i=0;i<outH;i++)
    {
        for(int j=0;j<outW;j++)
        {
            for(int m=0;m<kSizeY;m++)
            {
                for(int n=0;n<kSizeX;n++)
                {
                    O[i*outW+j]+=K[m*kSizeX+n]*F[(m+i*strideY)*width+(n+j*strideX)];
                }
            }
        }
    }

    for (int i = 0; i < outH; ++i)
    {
        for (int j = 0; j < outW; ++j)
        {
            std::cout<<O[i*outW+j]<<" ";
        }
        std::cout<<std::endl;
    }
}
```

<br/>

- **4. 带padding的卷积**

  接下来考虑带padding的卷积,  如输入为4x4x1, kernel为3x3x1. 卷积步长为1, padding为1 则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=1**
  - **paddingX=1**
  - **paddingY=1**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=1**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=1**

    ![](https://img-blog.csdnimg.cn/img_convert/0fad4724a7f9eab6f13ae3f9576f497e.png)

    <center>图4 考虑padding卷积</center>

  cpp代码:

```
void demo3()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4,5,6,7,8,9};
    float O[] = {0,0,0,0};

    int padX = 1;
    int padY = 1;

    int dilationX = 1;
    int dilationY = 1;

    int strideX  = 2;
    int strideY  = 2;

    int width = 4;
    int height = 4;

    int kSizeX = 3;
    int kSizeY = 3;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    for(int i=0;i<outH;i++)
    {
        for(int j=0;j<outW;j++)
        {
            for(int m=0;m<kSizeY;m++)
            {
                for(int n=0;n<kSizeX;n++)
                {
                    float fVal = 0;  
                    //考虑边界强情况
                    if((n+j*strideX-padX)>-1&&(m+i*strideY-padY>-1)&&(n+j*strideX-padX)<=width&&(m+i*strideY-padY>-1)<=height) 
                    {
                        fVal = F[(m+i*strideY-padX)*width+(n+j*strideX-padY)];
                    }
                    O[i*outW+j]+=K[m*kSizeX+n]*fVal;
                }
            }
        }
    }

    for (int i = 0; i < outH; ++i)
    {
        for (int j = 0; j < outW; ++j)
        {
            std::cout<<O[i*outW+j]<<" ";
        }
        std::cout<<std::endl;
    }
}
```

<br/>

- **5. 多通道卷积**

  接下来考虑多通道卷积,  如输入为4x4x1, kernel为3x3x1. 卷积步长为1, padding为1, 输入通道为2, 则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=2**
  - **paddingX=1**
  - **paddingY=1**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=1**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=1**
  
    ![](https://img-blog.csdnimg.cn/img_convert/92558b26d90c7590f436592bd4ace397.png)

    <center>图5 多通道卷积</center>

  cpp代码:

```
void demo4()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9};
    float O[] = {0,0,0,0};

    int padX = 1;
    int padY = 1;

    int dilationX = 1;
    int dilationY = 1;

    int strideX  = 2;
    int strideY  = 2;

    int width = 4;
    int height = 4;

    int kSizeX = 3;
    int kSizeY = 3;

    int channel = 2;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    for (int c = 0; c < channel; ++c)
    {
        for(int i=0;i<outH;i++)
        {
            for(int j=0;j<outW;j++)
            {
                for(int m=0;m<kSizeY;m++)
                {
                    for(int n=0;n<kSizeX;n++)
                    {
                        float fVal = 0;
                        if((n+j*strideX-padX)>-1&&(m+i*strideY-padY>-1)&&(n+j*strideX-padX)<=width&&(m+i*strideY-padY>-1)<=height)
                        {
                            fVal = F[c*width*height + (m+i*strideY-padX)*width+(n+j*strideX-padY)];
                        }
                        O[i*outW+j]+=K[c*kSizeX*kSizeY+m*kSizeX+n]*fVal;
                    }
                }
            }
        }
    }

    for (int i = 0; i < outH; ++i)
    {
        for (int j = 0; j < outW; ++j)
        {
            std::cout<<O[i*outW+j]<<" ";
        }
        std::cout<<std::endl;
    }
}
```

<br/>

- **6. 多kernel卷积**

  接下来考虑多kernel卷积,  如输入为4x4x1, kernel为3x3x1. 卷积步长为1, padding为1, 输入通道为2, filters为2, 则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=2**
  - **paddingX=1**
  - **paddingY=1**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=1**
  - **dilationY=1**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=2**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=2**

![图6 多kernel卷积](https://img-blog.csdnimg.cn/20201231224331687.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

  cpp代码:

```
void demo5()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,
                 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
                };
    float O[] = {0,0,0,0,0,0,0,0};

    int padX = 1;
    int padY = 1;

    int dilationX = 1;
    int dilationY = 1;

    int strideX  = 2;
    int strideY  = 2;

    int width = 4;
    int height = 4;

    int kSizeX = 3;
    int kSizeY = 3;

    int channel = 2;

    int filters = 2;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    int outC = filters;

    for (int oc = 0; oc < outC; ++oc)
    {
        for (int c = 0; c < channel; ++c)
        {
            for(int i=0;i<outH;i++)
            {
                for(int j=0;j<outW;j++)
                {
                    for(int m=0;m<kSizeY;m++)
                    {
                        for(int n=0;n<kSizeX;n++)
                        {
                            float fVal = 0;
                            if((n+j*strideX-padX)>-1&&(m+i*strideY-padY>-1)&&(n+j*strideX-padX)<=width&&(m+i*strideY-padY>-1)<=height)
                            {
                                fVal = F[c*width*height + (m+i*strideY-padX)*width+(n+j*strideX-padY)];
                            }
                            O[oc*outH*outW+i*outW+j]+=K[oc*outC*kSizeX*kSizeY+c*kSizeX*kSizeY+m*kSizeX+n]*fVal;
                        }
                    }
                }
            }
        }
    }

    for (int oc = 0; oc < outC; ++oc)
    {
        for (int i = 0; i < outH; ++i)
        {
            for (int j = 0; j < outW; ++j)
            {
                std::cout<<O[oc*outH*outW+i*outW+j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl<<std::endl;
    }
}
```

<br/>

- **7. 膨胀卷积**

  接下来考虑多膨胀卷积,  如输入为4x4x1, kernel为3x3x1. 卷积步长为1, padding为1, 输入通道为2, filters为2, dilate为2则此处的参数为:

  - **width=4**
  - **height=4**
  - **channel=2**
  - **paddingX=1**
  - **paddingY=1**
  - **strideX=1**
  - **strideY=1**
  - **dilationX=2**
  - **dilationY=2**
  - **kSizeX=3**
  - **kSizeY=3**
  - **filters=2**

    可根据卷积输出尺寸计算公式,得到:

  - **outWidth=2**
  - **outHeight=2**
  - **outChannel=2**

![图7 膨胀卷积](https://img-blog.csdnimg.cn/20201231224605654.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

  cpp代码:

```
void demo6()
{
    float F[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    float K[] = {1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,
                 1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9
                };
    float O[] = {0,0,0,0,0,0,0,0};

    int padX = 1;
    int padY = 1;

    int dilationX = 2;
    int dilationY = 2;

    int strideX  = 1;
    int strideY  = 1;

    int width = 4;
    int height = 4;

    int kSizeX = 3;
    int kSizeY = 3;

    int channel = 2;

    int filters = 2;

    int outH = (height+2*padY-(dilationY*(kSizeY-1)+1)) / strideY + 1;
    int outW = (width+2*padX-(dilationX*(kSizeX-1)+1)) / strideX + 1;

    int outC = filters;

    for (int oc = 0; oc < outC; ++oc)
    {
        for (int c = 0; c < channel; ++c)
        {
            for(int i=0;i<outH;i++)
            {
                for(int j=0;j<outW;j++)
                {
                    for(int m=0;m<kSizeY;m++)
                    {
                        for(int n=0;n<kSizeX;n++)
                        {
                            float fVal = 0;
                            if( ((n+j*strideX)*dilationX-padX)>-1 && ((m+i*strideY)*dilationY-padY)>-1&&
                               ((n+j*strideX)*dilationX-padX)<=width && ((m+i*strideY)*dilationY-padY>-1)<=height)
                            {
                                fVal = F[c*width*height + ((m+i*strideY)*dilationX-padX)*width+((n+j*strideX)*dilationY-padY)];
                            }
                            O[oc*outH*outW+i*outW+j]+=K[oc*outC*kSizeX*kSizeY+c*kSizeX*kSizeY+m*kSizeX+n]*fVal;
                        }
                    }
                }
            }
        }
    }

    for (int oc = 0; oc < outC; ++oc)
    {
        for (int i = 0; i < outH; ++i)
        {
            for (int j = 0; j < outW; ++j)
            {
                std::cout<<O[oc*outH*outW+i*outW+j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<std::endl;
    }
}
```

<br/>

- **git源码**

  https://github.com/msnh2012/ConvTest

- **最后**

  欢迎关注我和BBuf及公众号的小伙伴们一块维护的一个深度学习框架Msnhnet:  https://github.com/msnh2012/Msnhnet