> 上一节的RGB转灰度图算法我又做了两个相关优化，加入了多线程以及去掉了上次SSE计算中的一些重复计算，现在相对于传统实现已经可以获得4倍加速。同时我也在做一个AVX2的优化，所以不久后我将发布一个RGB转灰度图算法优化的升级版，尝试触摸这一个算法的优化极限，我会尽快做完实验发出来的。今天我先介绍一个有趣的自然饱和度算法，并讲解如何一步步进行优化。

# 1. 原始实现
今天要介绍的自然饱和度算法是一个开源图像处理软件PhotoDemon（地址：`https://github.com/tannerhelland/PhotoDemon`）上的，原版是C#的，代码如下：

```c#
For x = initX To finalX
        quickVal = x * qvDepth
        For y = initY To finalY
            'Get the source pixel color values
            r = ImageData(quickVal + 2, y)
            g = ImageData(quickVal + 1, y)
            b = ImageData(quickVal, y)
            
            'Calculate the gray value using the look-up table
            avgVal = grayLookUp(r + g + b)
            maxVal = Max3Int(r, g, b)
            
            'Get adjusted average
            amtVal = ((Abs(maxVal - avgVal) / 127) * vibranceAdjustment)
            
            If r <> maxVal Then
                r = r + (maxVal - r) * amtVal
            End If
            If g <> maxVal Then
                g = g + (maxVal - g) * amtVal
            End If
            If b <> maxVal Then
                b = b + (maxVal - b) * amtVal
            End If
            
            'Clamp values to [0,255] range
            If r < 0 Then r = 0
            If r > 255 Then r = 255
            If g < 0 Then g = 0
            If g > 255 Then g = 255
            If b < 0 Then b = 0
            If b > 255 Then b = 255
            
            ImageData(quickVal + 2, y) = r
            ImageData(quickVal + 1, y) = g
            ImageData(quickVal, y) = b
        Next
    Next
```

然后将其翻译为C++就获得了原始实现，代码如下：

```c++
//Adjustment如果为正值，会增加饱和度
void VibranceAlgorithm_FLOAT(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Adjustment) {
	float VibranceAdjustment = -0.01 * Adjustment;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
		for (int X = 0; X < Width; X++) {
			int Blue = LinePS[0], Green = LinePS[1], Red = LinePS[2];
			int Avg = (Blue + Green + Green + Red) >> 2;
			int Max = max(max(Blue, Green), Red);
			float AmtVal = (abs(Max - Avg) / 127.0f) * VibranceAdjustment;
			if (Blue != Max) Blue += (Max - Blue) * AmtVal;
			if (Green != Max) Green += (Max - Green) * AmtVal;
			if (Red != Max) Red += (Max - Red) * AmtVal;
			if (Red < 0) Red = 0;
			else if (Red > 255) Red = 255;
			if (Green < 0) Green = 0;
			else if (Green > 255) Green = 255;
			if (Blue < 0) Blue = 0;
			else if (Blue > 255) Blue = 255;
			LinePD[0] = Blue;
			LinePD[1] = Green;
			LinePD[2] = Red;
			LinePS += 3;
			LinePD += 3;
		}
	}
}
```

代码看起来非常简单，我们可以使用这个代码去对人像进行处理，效果如下：

![原图](https://img-blog.csdnimg.cn/2020040810093552.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![Adjustment=50，面色红润有精神](https://img-blog.csdnimg.cn/20200408101025937.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

![Adjustment=-50，面色苍白](https://img-blog.csdnimg.cn/20200408101122922.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

接下来看一下这个算法原始实现的速度测试：

| 分辨率    | 优化     | 循环次数 | 速度     |
| --------- | -------- | -------- | -------- |
| 4032x3024 | 原始实现 | 100      | 115.36ms |

# 2. 自然饱和度算法优化第一版
首先，我们可以考虑去掉算法中的浮点运算，即是将`float AmtVal = (abs(Max - Avg) / 127.0f) * VibranceAdjustment;`这里的`127.0f`优化为乘法，怎么优化为乘法呢？这里$127$是接近$128$的，如果我们把它换成$128$，那么我们可以用位运算来代替这个除法，实测将$127$换成$128$基本不影响算法的效果，所以这里直接采用了这种优化技巧。另外，Adjustment默认的范围为`[-100,100]`，如果把它的范围线性扩大一些，比如扩大$1.28$倍变成$[-128,128]$，这样在最后我们一次性移位，减少中间的损失。再然后，我们将这VibranceAdjustment里面的`*0.01`变成`*0.01=1.28/128`，然后把128放到下面的计算中并将VibranceAdjustment 重新设置为：`int VibranceAdjustment = -1.28 * Adjustment;`。最后还有一个点就是**这个算法中的绝对值运算完全可以去掉，因为平均值肯定是小于最大值的**。可能有点小晕哈，但是看代码很容易就理解了，下面给出优化后的代码：

```c++
void VibranceAlgorithm_INT(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Adjustment) {
	int VibranceAdjustment = -1.28 * Adjustment;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
		for (int X = 0; X < Width; X++) {
			int Blue, Green, Red, Max;
			Blue = LinePS[0], Green = LinePS[1], Red = LinePS[2];
			int Avg = (Blue + Green + Green + Red) >> 2;
			if (Blue > Green)
				Max = Blue;
			else
				Max = Green;
			if (Red > Max)
				Max = Red;
			int AmtVal = (Max - Avg) * VibranceAdjustment;
			if (Blue != Max) Blue += (((Max - Blue) * AmtVal) >> 14);
			if (Green != Max) Green += (((Max - Green) * AmtVal) >> 14);
			if (Red != Max) Red += (((Max - Red) * AmtVal) >> 14);
			if (Red < 0) Red = 0;
			else if (Red > 255) Red = 255;
			if (Green < 0) Green = 0;
			else if (Green > 255) Green = 255;
			if (Blue < 0) Blue = 0;
			else if (Blue > 255) Blue = 255;
			LinePD[0] = Blue;
			LinePD[1] = Green;
			LinePD[2] = Red;
			LinePS += 3;
			LinePD += 3;
		}
	}
}
```

下面看一下速度测试：

| 分辨率    | 优化     | 循环次数 | 速度     |
| --------- | -------- | -------- | -------- |
| 4032x3024 | 原始实现 | 100      | 115.36ms |
|4032x3024|第一版优化|100|62.43ms

可以看到稍加优化，速度快了近2倍，还是比较可观的。

# 3. 自然饱和度算法优化第二版
在上面算法的基础上如果使用多线程(OpenMP)来优化的话那么会获得多少加速呢？我们来试试，源码如下：

```c++
void VibranceAlgorithm_INT_OpenMP(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride, int Adjustment) {
	int VibranceAdjustment = -1.28 * Adjustment;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Stride;
#pragma omp parallel for num_threads(4)
		for (int X = 0; X < Width; X++) {
			int Blue, Green, Red, Max;
			Blue = LinePS[X*3 + 0], Green = LinePS[X*3 + 1], Red = LinePS[X*3 + 2];
			int Avg = (Blue + Green + Green + Red) >> 2;
			if (Blue > Green)
				Max = Blue;
			else
				Max = Green;
			if (Red > Max)
				Max = Red;
			int AmtVal = (Max - Avg) * VibranceAdjustment;
			if (Blue != Max) Blue += (((Max - Blue) * AmtVal) >> 14);
			if (Green != Max) Green += (((Max - Green) * AmtVal) >> 14);
			if (Red != Max) Red += (((Max - Red) * AmtVal) >> 14);
			if (Red < 0) Red = 0;
			else if (Red > 255) Red = 255;
			if (Green < 0) Green = 0;
			else if (Green > 255) Green = 255;
			if (Blue < 0) Blue = 0;
			else if (Blue > 255) Blue = 255;
			LinePD[X*3 + 0] = Blue;
			LinePD[X*3 + 1] = Green;
			LinePD[X*3 + 2] = Red;
		}
	}
}
```

我们来看一下加速效果：

| 分辨率    | 优化     | 循环次数 | 速度     |
| --------- | -------- | -------- | -------- |
| 4032x3024 | 原始实现 | 100      | 115.36ms |
|4032x3024|第一版优化|100|62.43ms
|4032x3024|第二版优化(4线程)|100|28.89ms|

可以看到使用OpenMP开启4线程，可以将我们的算法又优化接近2倍，仍然是可观的。接下来我们开始今天的主角，使用SSE指令集对这段代码进行优化。

# 4. 自然饱和度算法优化第三版

注意，在这个例子中，我们一次性加载48个图像数据到内存中，刚好可以放在3个`__m128i`变量中，同时看了我第一篇优化的人应该知道48正好被3整除，也就是说我们完整的加载了16个24位像素，这不会出现上一篇文章中的断层现象，使得下面48个像素可以和现在的48个像素使用同样的方法进行处理。上篇文章传送门：[【AI PC端算法优化】一，一步步优化RGB转灰度图算法](https://mp.weixin.qq.com/s/itvuHfLwtgyE43LyoQc8UA)

首先，对于这样单像素点且邻域无关的算法，为了利用SSE提高运行速度，一个核心步骤就是把各个颜色分量分离为单独的连续的变量。然后在计算完之后，我们又需要把单独连续的变量重新分解成BGR（**注意OpenCV默认读图方式是BGR**）的形式，这两部分的代码实现如下：

- BGRBGR->BBGGRR

```cpp
Src1 = _mm_loadu_si128((__m128i *)(LinePS + 0)); //B1,G1,R1,B2,G2,R2,B3,G3,R3,B4,G4,R4,B5,G5,R5,B6
Src2 = _mm_loadu_si128((__m128i *)(LinePS + 16));//G6,R6,B7,G7,R7,B8,G8,R8,B9,G9,R9,B10,G10,R10,B11,G11
Src3 = _mm_loadu_si128((__m128i *)(LinePS + 32));//R11,B12,G12,R12,B13,G13,R13,B14,G14,R14,B15,G15,R15,B16,G16,R16

Blue8 = _mm_shuffle_epi8(Src1, _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
Blue8 = _mm_or_si128(Blue8, _mm_shuffle_epi8(Src2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1)));
Blue8 = _mm_or_si128(Blue8, _mm_shuffle_epi8(Src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13)));

Green8 = _mm_shuffle_epi8(Src1, _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
Green8 = _mm_or_si128(Green8, _mm_shuffle_epi8(Src2, _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1)));
Green8 = _mm_or_si128(Green8, _mm_shuffle_epi8(Src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14)));

Red8 = _mm_shuffle_epi8(Src1, _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
Red8 = _mm_or_si128(Red8, _mm_shuffle_epi8(Src2, _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1)));
Red8 = _mm_or_si128(Red8, _mm_shuffle_epi8(Src3, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15)));
```

- BBGGRR->BGRBGR

```cpp
Dest1 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5));
Dest1 = _mm_or_si128(Dest1, _mm_shuffle_epi8(Green8, _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1)));
Dest1 = _mm_or_si128(Dest1, _mm_shuffle_epi8(Red8, _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1)));

Dest2 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1));
Dest2 = _mm_or_si128(Dest2, _mm_shuffle_epi8(Green8, _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10)));
Dest2 = _mm_or_si128(Dest2, _mm_shuffle_epi8(Red8, _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1)));

Dest3 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1));
Dest3 = _mm_or_si128(Dest3, _mm_shuffle_epi8(Green8, _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1)));
Dest3 = _mm_or_si128(Dest3, _mm_shuffle_epi8(Red8, _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15)));
```

这两个过程就是巧妙的利用`__mm_shuffle_epi8`指令完成的，而那个`_mm_or_si128`指令就是实现了将这次操作时所有的B或者G或者R放在了一个SSE向量中，对照后面的注释就很好理解了，也可以自己手推一下这个过程。如果想看详细步骤可以参考ImageShop大佬的博客，链接如下：`https://www.cnblogs.com/Imageshop/p/7234463.html`。

接下来我们看看其它代码的实现，由于uchar数据类型的表示范围非常有限，除了少数几个操作能针对字节类型直接处理外（比如这段代码中的求RGB的Max值，就可以直接用下面的SIMD指令实现）

```cpp
Max8 = _mm_max_epu8(_mm_max_epu8(Blue8, Green8), Red8);
```

其他的一些操作无法在这样的范围(`uchar`)内进行了，所以我们需要将数据扩展到`short`或者`int/float`范围内，在SSE中完成这种操作是有直接命令的，例如`byte`扩展到`short`，则可以用`_mm_unpacklo_epi8`和`_mm_unpackhi_epi8`配合`Zero`来实现：

```cpp
BL16 = _mm_unpacklo_epi8(Blue8, Zero);
BH16 = _mm_unpackhi_epi8(Blue8, Zero);
```

其中`_mm_unpacklo_epi8`是将两个`__m128i`的低8位交错布置形成一个新的128位数据，如果其中一个参数为0，则就是把另外一个参数的低8个字节无损的扩展为16位了，以上述BL16为例，其内部布局为：

| 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   | 11   | 12   | 13   | 14   | 15   | 16   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| B0   | 0    | B1   | 0    | B2   | 0    | B3   | 0    | B4   | 0    | B5   | 0    | B6   | 0    | B7   | 0    |

接下来我们需要来实现`int Avg = (Blue + Green + Green + Red) >> 2;`这行代码了，可以看到里的计算是无法在uchar范围内完成的，中间的`Blue + Green + Green + Red`在大部分情况下都会超出255并且绝对小于$255\times 4$，因此我们需要扩展数据到16位(`short`)，按照上面介绍的指令集对`Blue8\Green8\Red8\Max8`进行扩展，代码如下所示：

```cpp
BL16 = _mm_unpacklo_epi8(Blue8, Zero);
BH16 = _mm_unpackhi_epi8(Blue8, Zero);
GL16 = _mm_unpacklo_epi8(Green8, Zero);
GH16 = _mm_unpackhi_epi8(Green8, Zero);
RL16 = _mm_unpacklo_epi8(Red8, Zero);
RH16 = _mm_unpackhi_epi8(Red8, Zero);
MaxL16 = _mm_unpacklo_epi8(Max8, Zero);
MaxH16 = _mm_unpackhi_epi8(Max8, Zero);
```

接下来计算Avg就简单了，代码如下：

```cpp
AvgL16 = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(BL16, RL16), _mm_slli_epi16(GL16, 1)), 2);
AvgH16 = _mm_srli_epi16(_mm_add_epi16(_mm_add_epi16(BH16, RH16), _mm_slli_epi16(GH16, 1)), 2);
```

上面的分析都非常常规，接下来要到本文的硬核部分了，首先SSE对于跳转处理是很不好做的，它擅长的是序列化的处理一件事情，虽然SSE提供了比较指令，但很多复杂的跳转情况，SSE仍然无能为力。而我们这段代码中跳转情况不算复杂，我们可以使用SSE中的比较指令得到个`Mask`，`Mask`中符合比较结果的值会为`F`，不符合的为`0`，然后把这个Mask和后面需要计算的某个值进行`And`操作，由于和`F`进行`And`操作不会改变操作数本身，和`0`进行`And`操作则变为`0`。因此，这种操作方式是无论你符合条件与否，都要进行后面的计算，只是不符合条件的计算并不会影响结果，这种多余的计算可能会低效SSE优化的部分提速效果，这个就要具体情况具体分析了。

然后参考ImageShop博主的思路，仔细观察我们的代码可以发现，这里跳转语句的本意是如果最大值和某个分量的值不相同，则进行后面的调整操作，否则就不进行调整。而后面的调整操作中有最大值减去这个分量的逻辑，也就是说如果满足条件两者相减则为$0$，调整量此时也为$0$，并不会对结果产生影响。**基于此，我们可以直接把这个条件判断去掉，这并不会影响结果。** 同时我们能节省一些SSE指令，并且也更加适合SSE的处理。

继续分析，由于代码中有`((Max - Blue) * AmtVal) >> 14`这行逻辑，其中`AmtVal = (Max - Avg) * Adjustment`，展开即为：

`((Max - Blue) * (Max - Avg) * Adjustment）>>14`


这三个数据相乘很大程度上会超出`short`所能表达的范围，因此，我们还需要对上面的$16$位数据进行扩展，扩展到$32$位，这样就多了很多指令，那么有没有不需要扩展的方式呢？ImageShop博主提出了一种方式，这里搬运一下：

在SSE指令集中有一个指令叫做`_mm_mulhi_epi16`，我们看一下这个指令能干什么？


![_mm_mulhi_epi16 指令](https://img-blog.csdnimg.cn/20200408140839625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

这个指令剋一次性处理8个16位的数据，其计算结果相当于对于$(a*b）>>16$，但$a$和$b$必须是`short`类型所能表达的范围。而我们要求解的等式是：
`((Max - Blue) * (Max - Avg) * Adjustment）>>14`
因此，我们这里首先将其扩展为移位16位的结果，变成：
` ((Max - Blue) * 4 * (Max - Avg) * Adjustment）>>16`
然后，已知的是Adjustment我们已经将他限定在了`[-128,128]`之间，而`(Max - Avg)`理论上的最大值为`255 - 85=170`,（即`RGB`分量有一个是`255`，其他的都为`0`），最小值为`0`，因此，两者在各自范围内的成绩不会超出`short`所能表达的范围，而(`Max-Blue`)的最大值为`255`，最小值为`0`，再乘以`4`也在`short`类型所能表达的范围内。所以SSE代码就呼之欲出了。
- 原始代码段。
```cpp
int AmtVal = (Max - Avg) * Adjustment;                                //    Get adjusted average
if (Blue != Max)    Blue += (((Max - Blue) * AmtVal) >> 14);
if (Green != Max)    Green += (((Max - Green) * AmtVal) >> 14);
if (Red != Max)        Red += (((Max - Red) * AmtVal) >> 14);
```
- 按照上述思路改成SSE代码段。

```cpp
AmtVal = _mm_mullo_epi16(_mm_sub_epi16(MaxL16, AvgL16), Adjustment128);
BL16 = _mm_adds_epi16(BL16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxL16, BL16), 2), AmtVal));
GL16 = _mm_adds_epi16(GL16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxL16, GL16), 2), AmtVal));
RL16 = _mm_adds_epi16(RL16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxL16, RL16), 2), AmtVal));
            
AmtVal = _mm_mullo_epi16(_mm_sub_epi16(MaxH16, AvgH16), Adjustment128);
BH16 = _mm_adds_epi16(BH16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxH16, BH16), 2), AmtVal));
GH16 = _mm_adds_epi16(GH16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxH16, GH16), 2), AmtVal));
RH16 = _mm_adds_epi16(RH16, _mm_mulhi_epi16(_mm_slli_epi16(_mm_sub_epi16(MaxH16, RH16), 2), AmtVal));
```

最后一步就是将获得的B8,G8,R8别转换为不连续存储的形式即是BGR的格式，然后再存储即可。

```cpp
Dest1 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1, 5));
Dest1 = _mm_or_si128(Dest1, _mm_shuffle_epi8(Green8, _mm_setr_epi8(-1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1, -1)));
Dest1 = _mm_or_si128(Dest1, _mm_shuffle_epi8(Red8, _mm_setr_epi8(-1, -1, 0, -1, -1, 1, -1, -1, 2, -1, -1, 3, -1, -1, 4, -1)));

Dest2 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(-1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10, -1));
Dest2 = _mm_or_si128(Dest2, _mm_shuffle_epi8(Green8, _mm_setr_epi8(5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1, 10)));
Dest2 = _mm_or_si128(Dest2, _mm_shuffle_epi8(Red8, _mm_setr_epi8(-1, 5, -1, -1, 6, -1, -1, 7, -1, -1, 8, -1, -1, 9, -1, -1)));

Dest3 = _mm_shuffle_epi8(Blue8, _mm_setr_epi8(-1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1, -1));
Dest3 = _mm_or_si128(Dest3, _mm_shuffle_epi8(Green8, _mm_setr_epi8(-1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15, -1)));
Dest3 = _mm_or_si128(Dest3, _mm_shuffle_epi8(Red8, _mm_setr_epi8(10, -1, -1, 11, -1, -1, 12, -1, -1, 13, -1, -1, 14, -1, -1, 15)));

_mm_storeu_si128((__m128i *)(LinePD + 0), Dest1);
_mm_storeu_si128((__m128i *)(LinePD + 16), Dest2);
_mm_storeu_si128((__m128i *)(LinePD + 32), Dest3);
```

完整代码实现请看：`https://github.com/BBuf/Image-processing-algorithm-Speed/blob/master/speed_rgb2gray_sse.cpp`。


我们来看一下速度测试：


| 分辨率    | 优化     | 循环次数 | 速度     |
| --------- | -------- | -------- | -------- |
| 4032x3024 | 原始实现 | 100      | 115.36ms |
|4032x3024|第一版优化|100|62.43ms
|4032x3024|第二版优化(4线程)|100|28.89ms|
|4032x3024|第三版优化(SSE)|100|12.69ms|


# 5. 结论
这篇文章介绍了如何一步步优化一个自然饱和度算法，从原始算法的115.36ms优化到了13.04ms，**加速比达到了9.09倍**，还是比较可观的。


# 6. 参考
- https://www.cnblogs.com/Imageshop/p/7234463.html
- https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=4155,3983,3956&text=_mm_mulhi_epi16

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)