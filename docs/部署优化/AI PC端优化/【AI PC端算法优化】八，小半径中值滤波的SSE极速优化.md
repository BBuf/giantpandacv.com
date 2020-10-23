# 1. 前言
大家应该经常碰到这种需求，那就是使用$3\times 3$或者$5\times 5$这种相对比较小的窗口进行中值滤波，而如果在图像的分辨率比较大的情况下这种操作也是比较耗时的。所以在这种固定场景下定制一个优化算法是有意义的。（这里针对PC端，而非Arm端）。

# 2. 普通的3*3中值滤波实现
普通的实现没什么好说，就是直接在窗口区域内遍历寻找中位数即可，这里获取中值直接使用了c语言的qsort。代码实现如下：


```cpp
int ComparisonFunction(const void *X, const void *Y) {
	unsigned char Dx = *(unsigned char *)X;
	unsigned char Dy = *(unsigned char *)Y;
	if (Dx < Dy) return -1;
	else if (Dx > Dy) return 1;
	else return 0;
}

void MedianBlur3X3_Ori(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	int Channel = Stride / Width;
	if (Channel == 1) {
		unsigned char Array[9];
		for (int Y = 1; Y < Height - 1; Y++) {
			unsigned char *LineP0 = Src + (Y - 1) * Stride + 1;
			unsigned char *LineP1 = LineP0 + Stride;
			unsigned char *LineP2 = LineP1 + Stride;
			unsigned char *LinePD = Dest + Y * Stride + 1;
			for (int X = 1; X < Width - 1; X++) {
				Array[0] = LineP0[X - 1];        Array[1] = LineP0[X];    Array[2] = LineP0[X + 1];
				Array[3] = LineP1[X - 1];        Array[4] = LineP1[X];    Array[5] = LineP2[X + 1];
				Array[6] = LineP2[X - 1];        Array[7] = LineP2[X];    Array[8] = LineP2[X + 1];
				qsort(Array, 9, sizeof(unsigned char), &ComparisonFunction);
				LinePD[X] = Array[4];
			}
		}
	}
	else {
		unsigned char ArrayB[9], ArrayG[9], ArrayR[9];
		for (int Y = 1; Y < Height - 1; Y++) {
			unsigned char *LineP0 = Src + (Y - 1) * Stride + 3;
			unsigned char *LineP1 = LineP0 + Stride;
			unsigned char *LineP2 = LineP1 + Stride;
			unsigned char *LinePD = Dest + Y * Stride + 3;
			for (int X = 1; X < Width - 1; X++){
				ArrayB[0] = LineP0[-3];       ArrayG[0] = LineP0[-2];       ArrayR[0] = LineP0[-1];
				ArrayB[1] = LineP0[0];        ArrayG[1] = LineP0[1];        ArrayR[1] = LineP0[2];
				ArrayB[2] = LineP0[3];        ArrayG[2] = LineP0[4];        ArrayR[2] = LineP0[5];

				ArrayB[3] = LineP1[-3];       ArrayG[3] = LineP1[-2];       ArrayR[3] = LineP1[-1];
				ArrayB[4] = LineP1[0];        ArrayG[4] = LineP1[1];        ArrayR[4] = LineP1[2];
				ArrayB[5] = LineP1[3];        ArrayG[5] = LineP1[4];        ArrayR[5] = LineP1[5];

				ArrayB[6] = LineP2[-3];       ArrayG[6] = LineP2[-2];       ArrayR[6] = LineP2[-1];
				ArrayB[7] = LineP2[0];        ArrayG[7] = LineP2[1];        ArrayR[7] = LineP2[2];
				ArrayB[8] = LineP2[3];        ArrayG[8] = LineP2[4];        ArrayR[8] = LineP2[5];

				qsort(ArrayB, 9, sizeof(unsigned char), &ComparisonFunction);
				qsort(ArrayG, 9, sizeof(unsigned char), &ComparisonFunction);
				qsort(ArrayR, 9, sizeof(unsigned char), &ComparisonFunction);

				LinePD[0] = ArrayB[4];
				LinePD[1] = ArrayG[4];
				LinePD[2] = ArrayR[4];

				LineP0 += 3;
				LineP1 += 3;
				LineP2 += 3;
				LinePD += 3;
			}
		}
	}
}
```

来测一把耗时情况：

| 分辨率    | 算法优化 | 循环次数 | 速度       |
| --------- | -------- | -------- | ---------- |
| 4032x3024 | 普通实现 | 100      | 8293.79 ms |

# 3. 一个简单的改进
由于排序耗时是非常多的，而这里实际上就是在9个元素中找到中位数，这个其实不需要排序就可以办到，只要我们按照下面的方法进行比较就可以获得中位数。事实上只需要19次比较，就可以获得中位数，我们先看源码：

```cpp
void Swap(int &X, int &Y) {
	X ^= Y;
	Y ^= X;
	X ^= Y;
}

void MedianBlur3X3_Faster(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	int Channel = Stride / Width;
	if (Channel == 1) {

		for (int Y = 1; Y < Height - 1; Y++) {
			unsigned char *LineP0 = Src + (Y - 1) * Stride + 1;
			unsigned char *LineP1 = LineP0 + Stride;
			unsigned char *LineP2 = LineP1 + Stride;
			unsigned char *LinePD = Dest + Y * Stride + 1;
			for (int X = 1; X < Width - 1; X++) {
				int Gray0, Gray1, Gray2, Gray3, Gray4, Gray5, Gray6, Gray7, Gray8;
				Gray0 = LineP0[X - 1];        Gray1 = LineP0[X];    Gray2 = LineP0[X + 1];
				Gray3 = LineP1[X - 1];        Gray4 = LineP1[X];    Gray5 = LineP1[X + 1];
				Gray6 = LineP2[X - 1];        Gray7 = LineP2[X];    Gray8 = LineP2[X + 1];

				if (Gray1 > Gray2) Swap(Gray1, Gray2);
				if (Gray4 > Gray5) Swap(Gray4, Gray5);
				if (Gray7 > Gray8) Swap(Gray7, Gray8);
				if (Gray0 > Gray1) Swap(Gray0, Gray1);
				if (Gray3 > Gray4) Swap(Gray3, Gray4);
				if (Gray6 > Gray7) Swap(Gray6, Gray7);
				if (Gray1 > Gray2) Swap(Gray1, Gray2);
				if (Gray4 > Gray5) Swap(Gray4, Gray5);
				if (Gray7 > Gray8) Swap(Gray7, Gray8);
				if (Gray0 > Gray3) Swap(Gray0, Gray3);
				if (Gray5 > Gray8) Swap(Gray5, Gray8);
				if (Gray4 > Gray7) Swap(Gray4, Gray7);
				if (Gray3 > Gray6) Swap(Gray3, Gray6);
				if (Gray1 > Gray4) Swap(Gray1, Gray4);
				if (Gray2 > Gray5) Swap(Gray2, Gray5);
				if (Gray4 > Gray7) Swap(Gray4, Gray7);
				if (Gray4 > Gray2) Swap(Gray4, Gray2);
				if (Gray6 > Gray4) Swap(Gray6, Gray4);
				if (Gray4 > Gray2) Swap(Gray4, Gray2);

				LinePD[X] = Gray4;
			}
		}

	}
	else {
		for (int Y = 1; Y < Height - 1; Y++) {
			unsigned char *LineP0 = Src + (Y - 1) * Stride + 3;
			unsigned char *LineP1 = LineP0 + Stride;
			unsigned char *LineP2 = LineP1 + Stride;
			unsigned char *LinePD = Dest + Y * Stride + 3;
			for (int X = 1; X < Width - 1; X++) {
				int Blue0, Blue1, Blue2, Blue3, Blue4, Blue5, Blue6, Blue7, Blue8;
				int Green0, Green1, Green2, Green3, Green4, Green5, Green6, Green7, Green8;
				int Red0, Red1, Red2, Red3, Red4, Red5, Red6, Red7, Red8;
				Blue0 = LineP0[-3];        Green0 = LineP0[-2];    Red0 = LineP0[-1];
				Blue1 = LineP0[0];        Green1 = LineP0[1];        Red1 = LineP0[2];
				Blue2 = LineP0[3];        Green2 = LineP0[4];        Red2 = LineP0[5];

				Blue3 = LineP1[-3];        Green3 = LineP1[-2];    Red3 = LineP1[-1];
				Blue4 = LineP1[0];        Green4 = LineP1[1];        Red4 = LineP1[2];
				Blue5 = LineP1[3];        Green5 = LineP1[4];        Red5 = LineP1[5];

				Blue6 = LineP2[-3];        Green6 = LineP2[-2];    Red6 = LineP2[-1];
				Blue7 = LineP2[0];        Green7 = LineP2[1];        Red7 = LineP2[2];
				Blue8 = LineP2[3];        Green8 = LineP2[4];        Red8 = LineP2[5];

				if (Blue1 > Blue2) Swap(Blue1, Blue2);
				if (Blue4 > Blue5) Swap(Blue4, Blue5);
				if (Blue7 > Blue8) Swap(Blue7, Blue8);
				if (Blue0 > Blue1) Swap(Blue0, Blue1);
				if (Blue3 > Blue4) Swap(Blue3, Blue4);
				if (Blue6 > Blue7) Swap(Blue6, Blue7);
				if (Blue1 > Blue2) Swap(Blue1, Blue2);
				if (Blue4 > Blue5) Swap(Blue4, Blue5);
				if (Blue7 > Blue8) Swap(Blue7, Blue8);
				if (Blue0 > Blue3) Swap(Blue0, Blue3);
				if (Blue5 > Blue8) Swap(Blue5, Blue8);
				if (Blue4 > Blue7) Swap(Blue4, Blue7);
				if (Blue3 > Blue6) Swap(Blue3, Blue6);
				if (Blue1 > Blue4) Swap(Blue1, Blue4);
				if (Blue2 > Blue5) Swap(Blue2, Blue5);
				if (Blue4 > Blue7) Swap(Blue4, Blue7);
				if (Blue4 > Blue2) Swap(Blue4, Blue2);
				if (Blue6 > Blue4) Swap(Blue6, Blue4);
				if (Blue4 > Blue2) Swap(Blue4, Blue2);

				if (Green1 > Green2) Swap(Green1, Green2);
				if (Green4 > Green5) Swap(Green4, Green5);
				if (Green7 > Green8) Swap(Green7, Green8);
				if (Green0 > Green1) Swap(Green0, Green1);
				if (Green3 > Green4) Swap(Green3, Green4);
				if (Green6 > Green7) Swap(Green6, Green7);
				if (Green1 > Green2) Swap(Green1, Green2);
				if (Green4 > Green5) Swap(Green4, Green5);
				if (Green7 > Green8) Swap(Green7, Green8);
				if (Green0 > Green3) Swap(Green0, Green3);
				if (Green5 > Green8) Swap(Green5, Green8);
				if (Green4 > Green7) Swap(Green4, Green7);
				if (Green3 > Green6) Swap(Green3, Green6);
				if (Green1 > Green4) Swap(Green1, Green4);
				if (Green2 > Green5) Swap(Green2, Green5);
				if (Green4 > Green7) Swap(Green4, Green7);
				if (Green4 > Green2) Swap(Green4, Green2);
				if (Green6 > Green4) Swap(Green6, Green4);
				if (Green4 > Green2) Swap(Green4, Green2);

				if (Red1 > Red2) Swap(Red1, Red2);
				if (Red4 > Red5) Swap(Red4, Red5);
				if (Red7 > Red8) Swap(Red7, Red8);
				if (Red0 > Red1) Swap(Red0, Red1);
				if (Red3 > Red4) Swap(Red3, Red4);
				if (Red6 > Red7) Swap(Red6, Red7);
				if (Red1 > Red2) Swap(Red1, Red2);
				if (Red4 > Red5) Swap(Red4, Red5);
				if (Red7 > Red8) Swap(Red7, Red8);
				if (Red0 > Red3) Swap(Red0, Red3);
				if (Red5 > Red8) Swap(Red5, Red8);
				if (Red4 > Red7) Swap(Red4, Red7);
				if (Red3 > Red6) Swap(Red3, Red6);
				if (Red1 > Red4) Swap(Red1, Red4);
				if (Red2 > Red5) Swap(Red2, Red5);
				if (Red4 > Red7) Swap(Red4, Red7);
				if (Red4 > Red2) Swap(Red4, Red2);
				if (Red6 > Red4) Swap(Red6, Red4);
				if (Red4 > Red2) Swap(Red4, Red2);

				LinePD[0] = Blue4;
				LinePD[1] = Green4;
				LinePD[2] = Red4;

				LineP0 += 3;
				LineP1 += 3;
				LineP2 += 3;
				LinePD += 3;
			}
		}
	}
}
```

其实上面的代码很好理解，我们将R,G,B三个通道分开看，每个通道执行的都是完全一样的操作，随着比较的不断执行，最后最小的4个数会排在前4个位置，最大的4个数会排在后4个位置，中位数恰好就在中间。这个算法的流水情况比第一个算法好多了，自然也会得到较大的速度提升。

同样来测试一下速度：


| 分辨率    | 算法优化             | 循环次数 | 速度      |
| --------- | -------------------- | -------- | --------- |
| 4032x3024 | 普通实现             | 100      | 8293.79ms |
| 4032x3024 | 逻辑优化，更好的流水 | 100      | 83.75ms   |


# 4. SSE优化
这里是本文的重点了，似乎这个算法看起来是不好做SSE优化的，因为窗口中像素的$9$次比较不能直接用SIMD指令来做。根据ImageShop博主的提示，当我们看到下面这段代码`https://github.com/ARM-software/ComputeLibrary/blob/master/src/core/NEON/kernels/NEMedian3x3Kernel.cpp#L113`提示时，我们可以知道多个像素的比较是不相关的，（这个地方需要思考为什么不相关，因为我们比较的时候交换是使用临时变量，实际上是没有改变每个位置的像素的位置的）。所以SSE优化的思路就有了，现在可以一次性处理16个像素了。SSE优化的代码如下：

```cpp
inline void _mm_sort_ab(__m128i &a, __m128i &b) {
	const __m128i min = _mm_min_epu8(a, b);
	const __m128i max = _mm_max_epu8(a, b);
	a = min;
	b = max;
}

void MedianBlur3X3_Fastest(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	int Channel = Stride / Width;
	int BlockSize = 16, Block = ((Width - 2)* Channel) / BlockSize;
	for (int Y = 1; Y < Height - 1; Y++) {
		unsigned char *LineP0 = Src + (Y - 1) * Stride + Channel;
		unsigned char *LineP1 = LineP0 + Stride;
		unsigned char *LineP2 = LineP1 + Stride;
		unsigned char *LinePD = Dest + Y * Stride + Channel;
		for (int X = 0; X < Block * BlockSize; X += BlockSize, LineP0 += BlockSize, LineP1 += BlockSize, LineP2 += BlockSize, LinePD += BlockSize)
		{
			__m128i P0 = _mm_loadu_si128((__m128i *)(LineP0 - Channel));
			__m128i P1 = _mm_loadu_si128((__m128i *)(LineP0 - 0));
			__m128i P2 = _mm_loadu_si128((__m128i *)(LineP0 + Channel));
			__m128i P3 = _mm_loadu_si128((__m128i *)(LineP1 - Channel));
			__m128i P4 = _mm_loadu_si128((__m128i *)(LineP1 - 0));
			__m128i P5 = _mm_loadu_si128((__m128i *)(LineP1 + Channel));
			__m128i P6 = _mm_loadu_si128((__m128i *)(LineP2 - Channel));
			__m128i P7 = _mm_loadu_si128((__m128i *)(LineP2 - 0));
			__m128i P8 = _mm_loadu_si128((__m128i *)(LineP2 + Channel));

			_mm_sort_ab(P1, P2);		_mm_sort_ab(P4, P5);		_mm_sort_ab(P7, P8);
			_mm_sort_ab(P0, P1);		_mm_sort_ab(P3, P4);		_mm_sort_ab(P6, P7);
			_mm_sort_ab(P1, P2);		_mm_sort_ab(P4, P5);		_mm_sort_ab(P7, P8);
			_mm_sort_ab(P0, P3);		_mm_sort_ab(P5, P8);		_mm_sort_ab(P4, P7);
			_mm_sort_ab(P3, P6);		_mm_sort_ab(P1, P4);		_mm_sort_ab(P2, P5);
			_mm_sort_ab(P4, P7);		_mm_sort_ab(P4, P2);		_mm_sort_ab(P6, P4);
			_mm_sort_ab(P4, P2);

			_mm_storeu_si128((__m128i *)LinePD, P4);
		}

		for (int X = Block * BlockSize; X < (Width - 2) * Channel; X++, LinePD++) {
			int Gray0, Gray1, Gray2, Gray3, Gray4, Gray5, Gray6, Gray7, Gray8;
			Gray0 = LineP0[X - Block * BlockSize - Channel];        Gray1 = LineP0[X - Block * BlockSize];    Gray2 = LineP0[X - Block * BlockSize + Channel];
			Gray3 = LineP1[X - Block * BlockSize - Channel];        Gray4 = LineP1[X - Block * BlockSize];    Gray5 = LineP1[X - Block * BlockSize + Channel];
			Gray6 = LineP2[X - Block * BlockSize - Channel];        Gray7 = LineP2[X - Block * BlockSize];    Gray8 = LineP2[X - Block * BlockSize + Channel];

			if (Gray1 > Gray2) Swap(Gray1, Gray2);
			if (Gray4 > Gray5) Swap(Gray4, Gray5);
			if (Gray7 > Gray8) Swap(Gray7, Gray8);
			if (Gray0 > Gray1) Swap(Gray0, Gray1);
			if (Gray3 > Gray4) Swap(Gray3, Gray4);
			if (Gray6 > Gray7) Swap(Gray6, Gray7);
			if (Gray1 > Gray2) Swap(Gray1, Gray2);
			if (Gray4 > Gray5) Swap(Gray4, Gray5);
			if (Gray7 > Gray8) Swap(Gray7, Gray8);
			if (Gray0 > Gray3) Swap(Gray0, Gray3);
			if (Gray5 > Gray8) Swap(Gray5, Gray8);
			if (Gray4 > Gray7) Swap(Gray4, Gray7);
			if (Gray3 > Gray6) Swap(Gray3, Gray6);
			if (Gray1 > Gray4) Swap(Gray1, Gray4);
			if (Gray2 > Gray5) Swap(Gray2, Gray5);
			if (Gray4 > Gray7) Swap(Gray4, Gray7);
			if (Gray4 > Gray2) Swap(Gray4, Gray2);
			if (Gray6 > Gray4) Swap(Gray6, Gray4);
			if (Gray4 > Gray2) Swap(Gray4, Gray2);

			LinePD[X] = Gray4;
			LineP0 += 1;
			LineP1 += 1;
			LineP2 += 1;
		}
	}
}
```

来测一下速度：

| 分辨率    | 算法优化             | 循环次数 | 速度      |
| --------- | -------------------- | -------- | --------- |
| 4032x3024 | 普通实现             | 100      | 8293.79ms |
| 4032x3024 | 逻辑优化，更好的流水 | 100      | 83.75ms   |
| 4032x3024 | SSE优化              | 100      | 11.93ms   |


# 5. AVX优化
显然，我们可以将SSE版本稍加修改获得AVX指令优化的版本，这样我们就可以一次性处理32个元素了，代码如下：


```cpp
inline void _mm_sort_AB(__m256i &a, __m256i &b) {
	const __m256i min = _mm256_min_epu8(a, b);
	const __m256i max = _mm256_max_epu8(a, b);
	a = min;
	b = max;
}

void MedianBlur3X3_Fastest_AVX(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	int Channel = Stride / Width;
	int BlockSize = 32, Block = ((Width - 2)* Channel) / BlockSize;
	for (int Y = 1; Y < Height - 1; Y++) {
		unsigned char *LineP0 = Src + (Y - 1) * Stride + Channel;
		unsigned char *LineP1 = LineP0 + Stride;
		unsigned char *LineP2 = LineP1 + Stride;
		unsigned char *LinePD = Dest + Y * Stride + Channel;
		for (int X = 0; X < Block * BlockSize; X += BlockSize, LineP0 += BlockSize, LineP1 += BlockSize, LineP2 += BlockSize, LinePD += BlockSize)
		{
			__m256i P0 = _mm256_loadu_si256((const __m256i*)(LineP0 - Channel));
			__m256i P1 = _mm256_loadu_si256((const __m256i*)(LineP0 - 0));
			__m256i P2 = _mm256_loadu_si256((const __m256i*)(LineP0 + Channel));
			__m256i P3 = _mm256_loadu_si256((const __m256i*)(LineP1 - Channel));
			__m256i P4 = _mm256_loadu_si256((const __m256i*)(LineP1 - 0));
			__m256i P5 = _mm256_loadu_si256((const __m256i*)(LineP1 + Channel));
			__m256i P6 = _mm256_loadu_si256((const __m256i*)(LineP2 - Channel));
			__m256i P7 = _mm256_loadu_si256((const __m256i*)(LineP2 - 0));
			__m256i P8 = _mm256_loadu_si256((const __m256i*)(LineP2 + Channel));

			_mm_sort_AB(P1, P2);		_mm_sort_AB(P4, P5);		_mm_sort_AB(P7, P8);
			_mm_sort_AB(P0, P1);		_mm_sort_AB(P3, P4);		_mm_sort_AB(P6, P7);
			_mm_sort_AB(P1, P2);		_mm_sort_AB(P4, P5);		_mm_sort_AB(P7, P8);
			_mm_sort_AB(P0, P3);		_mm_sort_AB(P5, P8);		_mm_sort_AB(P4, P7);
			_mm_sort_AB(P3, P6);		_mm_sort_AB(P1, P4);		_mm_sort_AB(P2, P5);
			_mm_sort_AB(P4, P7);		_mm_sort_AB(P4, P2);		_mm_sort_AB(P6, P4);
			_mm_sort_AB(P4, P2);

			_mm256_storeu_si256((__m256i *)LinePD, P4);
		}

		for (int X = Block * BlockSize; X < (Width - 2) * Channel; X++, LinePD++) {
			int Gray0, Gray1, Gray2, Gray3, Gray4, Gray5, Gray6, Gray7, Gray8;
			Gray0 = LineP0[X - Block * BlockSize - Channel];        Gray1 = LineP0[X - Block * BlockSize];    Gray2 = LineP0[X - Block * BlockSize + Channel];
			Gray3 = LineP1[X - Block * BlockSize - Channel];        Gray4 = LineP1[X - Block * BlockSize];    Gray5 = LineP1[X - Block * BlockSize + Channel];
			Gray6 = LineP2[X - Block * BlockSize - Channel];        Gray7 = LineP2[X - Block * BlockSize];    Gray8 = LineP2[X - Block * BlockSize + Channel];

			if (Gray1 > Gray2) Swap(Gray1, Gray2);
			if (Gray4 > Gray5) Swap(Gray4, Gray5);
			if (Gray7 > Gray8) Swap(Gray7, Gray8);
			if (Gray0 > Gray1) Swap(Gray0, Gray1);
			if (Gray3 > Gray4) Swap(Gray3, Gray4);
			if (Gray6 > Gray7) Swap(Gray6, Gray7);
			if (Gray1 > Gray2) Swap(Gray1, Gray2);
			if (Gray4 > Gray5) Swap(Gray4, Gray5);
			if (Gray7 > Gray8) Swap(Gray7, Gray8);
			if (Gray0 > Gray3) Swap(Gray0, Gray3);
			if (Gray5 > Gray8) Swap(Gray5, Gray8);
			if (Gray4 > Gray7) Swap(Gray4, Gray7);
			if (Gray3 > Gray6) Swap(Gray3, Gray6);
			if (Gray1 > Gray4) Swap(Gray1, Gray4);
			if (Gray2 > Gray5) Swap(Gray2, Gray5);
			if (Gray4 > Gray7) Swap(Gray4, Gray7);
			if (Gray4 > Gray2) Swap(Gray4, Gray2);
			if (Gray6 > Gray4) Swap(Gray6, Gray4);
			if (Gray4 > Gray2) Swap(Gray4, Gray2);

			LinePD[X] = Gray4;
			LineP0 += 1;
			LineP1 += 1;
			LineP2 += 1;
		}
	}
}
```

同样，来测一下速度：

| 分辨率    | 算法优化             | 循环次数 | 速度      |
| --------- | -------------------- | -------- | --------- |
| 4032x3024 | 普通实现             | 100      | 8293.79ms |
| 4032x3024 | 逻辑优化，更好的流水 | 100      | 83.75ms   |
| 4032x3024 | SSE优化              | 100      | 11.93ms   |
| 4032x3024 | AVX优化              | 100      | 9.32ms    |

可以看到AVX虽然一次处理了32个像素，但速度的提升幅度并不是很大，只有2ms左右。

这里就不打算继续做AVX指令集的多线程优化和测速了，感兴趣的可以自行实验，基本上速度提升是很少了，

# 6. 总结
本文以一个$3\times 3$的中值滤波作为切入点，讨论了一下针对这个具体问题的优化思路，速度也从最开始普通实现的8293.79ms优化到了9.32ms，还是有一定的参考意义的。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)