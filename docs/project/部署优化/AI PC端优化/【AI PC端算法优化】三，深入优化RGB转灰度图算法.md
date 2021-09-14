# 0. 目录
- 前言
- RGB2GRAY最简单实现
- RGB转GRAY优化第一版（float->INT）
- RGB转GRAY优化第二版（手动4路并行）
- RGB转GRAY优化第三版（OpenMP4线程）
- RGB转GRAY优化第四版（SSE优化，一次处理12个像素）
- RGB转GRAY优化第五版（SSE优化，一次处理15个像素）
- RGB转GRAY优化第六版（AVX2优化，一次处理10个像素）
- RGB转GRAY优化第七版（AVX2优化+std::async）

# 1. 前言
前几天发了一篇一步步优化RGB转灰度图算法，但实验做的并不完善，在上次的基础上我又补充了一些优化技巧，相对于传统实现将RGB转灰度图算法可以加速到近5倍左右。所以，这篇文章再次将所有涉及到的优化方法进行汇总，SSE优化相关的原理上一节已经讲得很清楚了，这里就不会再展开了，感兴趣可以查看上篇文章。[【AI PC端算法优化】一，一步步优化RGB转灰度图算法](https://mp.weixin.qq.com/s/itvuHfLwtgyE43LyoQc8UA) 这一节的速度测试环境为：

**测试CPU型号：Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz**

# 2. RGB转GRAY最简单实现

```c++
//origin
void RGB2Y(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		for (int X = 0; X < Width; X++, LinePS += 3) {
			LinePD[X] = int(0.114 * LinePS[0] + 0.587 * LinePS[1] + 0.299 * LinePS[2]);
		}
	}
}
```

耗时信息如下：

| 分辨率    | 优化     | 循环次数 | 速度     |
| --------- | -------- | -------- | -------- |
| 4032x3024 | 原始实现 | 1000     | 12.139ms |

# 3. RGB转GRAY优化第一版


直接计算复杂度较高，考虑优化可以将小数转为整数，除法变为移位，乘法也变为移位，但是这种方法也会带来一定的精度损失，我们可以根据实际情况选择需要保留的精度位数。下面给出不同精度(2-20位)的计算公式：

```sh
Gray = (R*1 + G*2 + B*1) >> 2

Gray= (R*2 + G*5 + B*1) >> 3

Gray= (R*4 + G*10 + B*2) >> 4

Gray = (R*9 + G*19 + B*4) >> 5

Gray = (R*19 + G*37 + B*8) >> 6

Gray= (R*38 + G*75 + B*15) >> 7

Gray= (R*76 + G*150 + B*30) >> 8

Gray = (R*153 + G*300 + B*59) >> 9

Gray = (R*306 + G*601 + B*117) >> 10

Gray = (R*612 + G*1202 + B*234) >> 11

Gray = (R*1224 + G*2405 + B*467) >> 12

Gray= (R*2449 + G*4809 + B*934) >> 13

Gray= (R*4898 + G*9618 + B*1868) >> 14

Gray = (R*9797 + G*19235 + B*3736) >> 15

Gray = (R*19595 + G*38469 + B*7472) >> 16

Gray = (R*39190 + G*76939 + B*14943) >> 17

Gray = (R*78381 + G*153878 + B*29885) >> 18

Gray =(R*156762 + G*307757 + B*59769) >> 19

Gray= (R*313524 + G*615514 + B*119538) >> 20
```

下面测试一下保留8位精度的代码实现和速度测试：

```c++
//int
void RGB2Y_1(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		for (int X = 0; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}

```

| 分辨率    | 优化                     | 循环次数 | 速度     |
| --------- | ------------------------ | -------- | -------- |
| 4032x3024 | 原始实现                 | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT） | 1000     | 7.629ms  |

这一优化将速度加速了10ms，还是比较可观的。


# 4. RGB转GRAY优化第二版
在第一版优化的基础上，使用**4路并行**，然后我们看看有没有进一步的加速效果。代码实现和速度测试如下：
```
//4路并行
void RGB2Y_2(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 4; X += 4, LinePS += 12) {
			LinePD[X + 0] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
			LinePD[X + 1] = (B_WT * LinePS[3] + G_WT * LinePS[4] + R_WT * LinePS[5]) >> 8;
			LinePD[X + 2] = (B_WT * LinePS[6] + G_WT * LinePS[7] + R_WT * LinePS[8]) >> 8;
			LinePD[X + 3] = (B_WT * LinePS[9] + G_WT * LinePS[10] + R_WT * LinePS[11]) >> 8;
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}
```


| 分辨率    | 优化                      | 循环次数 | 速度     |
| --------- | ------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                  | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）  | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行） | 1000     | 10.528ms |

这里测试的时候并没有加速效果，反而还慢了很多。


# 5. RGB转GRAY优化第三版
利用OpenMP进行4线程加速，代码如下：

```
//openmp
void RGB2Y_3(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT;
	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
#pragma omp parallel for num_threads(4)
		for (int X = 0; X < Width; X++) {
			LinePD[X] = (B_WT * LinePS[0 + X*3] + G_WT * LinePS[1 + X*3] + R_WT * LinePS[2 + X*3]) >> 8;
		}
	}
}
```

| 分辨率    | 优化                      | 循环次数 | 速度     |
| --------- | ------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                  | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）  | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行） | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程） | 1000     | 7.632ms  |

可以看到使用OpenMP进行多线程加速，算法的速度和第一版的速度也是差不多的，没有明显的加速效果。

# 6. RGB转GRAY优化第四版
这一次优化就是上篇推文讲的SSE优化，原理就不再多说了，可以去查看上篇推文，代码实现如下：

```
//sse 一次处理12个
void RGB2Y_4(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)

	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 12; X += 12, LinePS += 36) {
			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3

			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//4
			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//5
			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//6

			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//7
			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//8
			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//9

			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));//10
			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));//11
			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));//12

			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));//13
			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));//14
			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));//15
			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));//16
			__m128i sclaL = _mm_srli_epi16(sumaL, 8);//17
			__m128i sclaH = _mm_srli_epi16(sumaH, 8);//18
			__m128i sclbL = _mm_srli_epi16(sumbL, 8);//19
			__m128i sclbH = _mm_srli_epi16(sumbH, 8);//20
			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//21
			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 18, 24, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));//22
			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));//23
			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 24, 30, -1, -1, -1, -1));//24
			__m128i accumL = _mm_or_si128(shftaL, shftbL);//25
			__m128i accumH = _mm_or_si128(shftaH, shftbH);//26
			__m128i h3 = _mm_or_si128(accumL, accumH);//27
													  //__m128i h3 = _mm_blendv_epi8(accumL, accumH, _mm_setr_epi8(0, 0, 0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 1, 1, 1, 1));
			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}
```

| 分辨率    | 优化                                    | 循环次数 | 速度     |
| --------- | --------------------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                                | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）                | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行）               | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程）               | 1000     | 7.632ms  |
| 4032x3024 | 第四版优化（SSE优化，一次处理12个像素） | 1000     | 5.579ms  |

# 7. RGB转灰度图优化第五版
在上篇推文中提到 `_mm_storeu_si128`把处理的结果写入到目标内存中，会多写了**4个字节**的内存数据（128 - 12 * 8），但是我们后面又会把他们重新覆盖掉，这就造成了重复计算，因此简单修改将一次处理12个像素变成一次处理15个像素，**那么重复计算的数量就会下降**，代码如下：

```
//sse 一次处理15个
void RGB2Y_5(unsigned char *Src, unsigned char *Dest, int Width, int Height, int Stride) {
	const int B_WT = int(0.114 * 256 + 0.5);
	const int G_WT = int(0.587 * 256 + 0.5);
	const int R_WT = 256 - B_WT - G_WT; // int(0.299 * 256 + 0.5)

	for (int Y = 0; Y < Height; Y++) {
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 15; X += 15, LinePS += 45)
		{
			__m128i p1aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 0))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT)); //1
			__m128i p2aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 1))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT)); //2
			__m128i p3aL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 2))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT)); //3

			__m128i p1aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 8))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
			__m128i p2aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 9))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p3aH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 10))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));

			__m128i p1bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 18))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p2bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 19))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
			__m128i p3bL = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 20))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));

			__m128i p1bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 26))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));
			__m128i p2bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 27))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p3bH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 28))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));

			__m128i p1cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 36))), _mm_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT));
			__m128i p2cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 37))), _mm_setr_epi16(G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT));
			__m128i p3cH = _mm_mullo_epi16(_mm_cvtepu8_epi16(_mm_loadu_si128((__m128i *)(LinePS + 38))), _mm_setr_epi16(R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT));

			__m128i sumaL = _mm_add_epi16(p3aL, _mm_add_epi16(p1aL, p2aL));
			__m128i sumaH = _mm_add_epi16(p3aH, _mm_add_epi16(p1aH, p2aH));
			__m128i sumbL = _mm_add_epi16(p3bL, _mm_add_epi16(p1bL, p2bL));
			__m128i sumbH = _mm_add_epi16(p3bH, _mm_add_epi16(p1bH, p2bH));
			__m128i sumcH = _mm_add_epi16(p3cH, _mm_add_epi16(p1cH, p2cH));

			__m128i sclaL = _mm_srli_epi16(sumaL, 8);
			__m128i sclaH = _mm_srli_epi16(sumaH, 8);
			__m128i sclbL = _mm_srli_epi16(sumbL, 8);
			__m128i sclbH = _mm_srli_epi16(sumbH, 8);
			__m128i sclcH = _mm_srli_epi16(sumcH, 8);

			__m128i shftaL = _mm_shuffle_epi8(sclaL, _mm_setr_epi8(0, 6, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftaH = _mm_shuffle_epi8(sclaH, _mm_setr_epi8(-1, -1, -1, 2, 8, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftbL = _mm_shuffle_epi8(sclbL, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 0, 6, 12, -1, -1, -1, -1, -1, -1, -1));
			__m128i shftbH = _mm_shuffle_epi8(sclbH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 8, 14, -1, -1, -1, -1));
			__m128i shftcH = _mm_shuffle_epi8(sclcH, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 6, 12, -1));
			__m128i accumL = _mm_or_si128(shftaL, shftbL);
			__m128i accumH = _mm_or_si128(shftaH, shftbH);
			__m128i h3 = _mm_or_si128(accumL, accumH);
			h3 = _mm_or_si128(h3, shftcH);
			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			LinePD[X] = (B_WT * LinePS[0] + G_WT * LinePS[1] + R_WT * LinePS[2]) >> 8;
		}
	}
}
```

| 分辨率    | 优化                                    | 循环次数 | 速度     |
| --------- | --------------------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                                | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）                | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行）               | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程）               | 1000     | 7.632ms  |
| 4032x3024 | 第四版优化（SSE优化，一次处理12个像素） | 1000     | 5.579ms  |
| 4032x3024 | 第五版优化（SSE优化，一次处理15个像素） | 1000     | 5.843ms  |

很遗憾并没有取得什么提升，考虑原因可能是因为图片的宽度除以12和除以15这两个直拉不开明显的数量级差距，导致重复计算不是特别多，在速度上体现不出来。

# 8. RGB转灰度图优化第六版
这一版即是使用AVX/AVX2指令集来优化，我们知道AVX寄存器可以一次处理256位也就是32个uchar，因此使用AVX来进行优化获取能带来性能提升。参考Github工程`https://github.com/komrad36/RGB2Y/blob/master/RGB2Y.h`将代码重写如下，注意这里的原理已经在注释里面标明了，如果你有疑惑可以到官方网站(`https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX&expand=4155`)查询指令集的对应操作或者联系我为你解答疑惑。

在AVX这一节的优化中，**梁德澎作者对我理解这段代码提供了巨大的帮助，在此特别感谢**。

AVX2指令集实现的代码如下：

```
// AVX2
constexpr double B_WEIGHT = 0.114;
constexpr double G_WEIGHT = 0.587;
constexpr double R_WEIGHT = 0.299;
constexpr uint16_t B_WT = static_cast<uint16_t>(32768.0 * B_WEIGHT + 0.5);
constexpr uint16_t G_WT = static_cast<uint16_t>(32768.0 * G_WEIGHT + 0.5);
constexpr uint16_t R_WT = static_cast<uint16_t>(32768.0 * R_WEIGHT + 0.5);
static const __m256i weight_vec = _mm256_setr_epi16(B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT, G_WT, R_WT, B_WT);

void  _RGB2Y(unsigned char* Src, const int32_t Width, const int32_t start_row, const int32_t thread_stride, const int32_t Stride, unsigned char* Dest)
{
	for (int Y = start_row; Y < start_row + thread_stride; Y++)
	{
		//Sleep(1);
		unsigned char *LinePS = Src + Y * Stride;
		unsigned char *LinePD = Dest + Y * Width;
		int X = 0;
		for (; X < Width - 10; X += 10, LinePS += 30)
		{
			//B1 G1 R1 B2 G2 R2 B3 G3 R3 B4 G4 R4 B5 G5 R5 B6 
			__m256i temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS + 0)));
			__m256i in1 = _mm256_mulhrs_epi16(temp, weight_vec);

			//B6 G6 R6 B7 G7 R7 B8 G8 R8 B9 G9 R9 B10 G10 R10 B11
			temp = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(LinePS + 15)));
			__m256i in2 = _mm256_mulhrs_epi16(temp, weight_vec);


			//0  1  2  3   4  5  6  7  8  9  10 11 12 13 14 15    16 17 18 19 20 21 22 23 24 25 26 27 28   29 30  31       
			//B1 G1 R1 B2 G2 R2 B3 G3  B6 G6 R6 B7 G7 R7 B8 G8    R3 B4 G4 R4 B5 G5 R5 B6 R8 B9 G9 R9 B10 G10 R10 B11
			__m256i mul = _mm256_packus_epi16(in1, in2);

			__m256i b1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(
				//  B1 B2 B3 -1, -1, -1  B7  B8  -1, -1, -1, -1, -1, -1, -1, -1,
				0, 3, 6, -1, -1, -1, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1, B4 B5 B6 -1, -1  B9 B10 -1, -1, -1, -1, -1, -1
				-1, -1, -1, 1, 4, 7, -1, -1, 9, 12, -1, -1, -1, -1, -1, -1));

			__m256i g1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				// G1 G2 G3 -1, -1  G6 G7  G8  -1, -1, -1, -1, -1, -1, -1, -1, 
				1, 4, 7, -1, -1, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, -1  G4 G5 -1, -1, -1  G9  G10 -1, -1, -1, -1, -1, -1	
				-1, -1, -1, 2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1));

			__m256i r1 = _mm256_shuffle_epi8(mul, _mm256_setr_epi8(

				//  R1 R2 -1  -1  -1  R6  R7  -1, -1, -1, -1, -1, -1, -1, -1, -1,	
				2, 5, -1, -1, -1, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1,

				//  -1, -1, R3 R4 R5 -1, -1, R8 R9  R10 -1, -1, -1, -1, -1, -1
				-1, -1, 0, 3, 6, -1, -1, 8, 11, 14, -1, -1, -1, -1, -1, -1));



			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m256i accum = _mm256_adds_epu8(r1, _mm256_adds_epu8(b1, g1));


			// _mm256_castsi256_si128(accum)
			// B1+G1+R1  B2+G2+R2 B3+G3  0 0 G6+R6  B7+G7+R7 B8+G8 0 0 0 0 0 0 0 0

			// _mm256_extracti128_si256(accum, 1)
			// 0 0 R3 B4+G4+R4 B5+G5+R5 B6 0 R8 B9+G9+R9 B10+G10+R10 0 0 0 0 0 0

			__m128i h3 = _mm_adds_epu8(_mm256_castsi256_si128(accum), _mm256_extracti128_si256(accum, 1));

			_mm_storeu_si128((__m128i *)(LinePD + X), h3);
		}
		for (; X < Width; X++, LinePS += 3) {
			int tmpB = (B_WT * LinePS[0]) >> 14 + 1;
			tmpB = max(min(255, tmpB), 0);

			int tmpG = (G_WT * LinePS[1]) >> 14 + 1;
			tmpG = max(min(255, tmpG), 0);

			int tmpR = (R_WT * LinePS[2]) >> 14 + 1;
			tmpR = max(min(255, tmpR), 0);

			int tmp = tmpB + tmpG + tmpR;
			LinePD[X] = max(min(255, tmp), 0);
		}
	}
}

//avx2 
void RGB2Y_6(unsigned char *Src, unsigned char *Dest, int width, int height, int stride)
{
	_RGB2Y(Src, width, 0, height, stride, Dest);
}
```

| 分辨率    | 优化                                     | 循环次数 | 速度     |
| --------- | ---------------------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                                 | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）                 | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行）                | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程）                | 1000     | 7.632ms  |
| 4032x3024 | 第四版优化（SSE优化，一次处理12个像素）  | 1000     | 5.579ms  |
| 4032x3024 | 第五版优化（SSE优化，一次处理15个像素）  | 1000     | 5.843ms  |
| 4032x3024 | 第六版优化（AVX2优化，一次处理10个像素） | 1000     | 3.576ms  |

相比于SSE优化提高了2.xms，提升幅度还是比较大的。

# 9. RGB转灰度图优化第七版
在AVX2优化的基础上如果我们将多线程也加入进来，是否会获得提升呢？这里并非使用OpenMP而是使用C++中的std::async异步并行编程创建多个线程来执行整个任务，代码实现如下：

```
//avx2 + std::async异步编程
void RGB2Y_7(unsigned char *Src, unsigned char *Dest, int width, int height, int stride) {
	const int32_t hw_concur = std::min(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
	std::vector<std::future<void>> fut(hw_concur);
	const int thread_stride = (height - 1) / hw_concur + 1;
	int i = 0, start = 0;
	for (; i < std::min(height, hw_concur); i++, start += thread_stride)
	{
		fut[i] = std::async(std::launch::async, _RGB2Y, Src, width, start, thread_stride, stride, Dest);
	}
	for (int j = 0; j < i; ++j)
		fut[j].wait();
}
```

速度测试结果如下：


| 分辨率    | 优化                                     | 循环次数 | 速度     |
| --------- | ---------------------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                                 | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）                 | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行）                | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程）                | 1000     | 7.632ms  |
| 4032x3024 | 第四版优化（SSE优化，一次处理12个像素）  | 1000     | 5.579ms  |
| 4032x3024 | 第五版优化（SSE优化，一次处理15个像素）  | 1000     | 5.843ms  |
| 4032x3024 | 第六版优化（AVX2优化，一次处理10个像素） | 1000     | 3.576ms  |
| 4032x3024 | 第七版优化（AVX2优化+std::async）        | 1000     | 2.626ms  |

可以看到使用异步并行以及AVX优化后，我们已经将原始算法的速度优化了**接近5倍**，在PC端优化RGB转灰度图算法我能想到和调研到的一些相关算法大概就这么多了，如果还有其它相关的想法或者方法可以在评论区留言讨论。

# 10. 总结

| 分辨率    | 优化                                     | 循环次数 | 速度     |
| --------- | ---------------------------------------- | -------- | -------- |
| 4032x3024 | 原始实现                                 | 1000     | 12.139ms |
| 4032x3024 | 第一版优化（float->INT）                 | 1000     | 7.629ms  |
| 4032x3024 | 第二版优化（手动4路并行）                | 1000     | 10.528ms |
| 4032x3024 | 第三版优化（OpenMP4线程）                | 1000     | 7.632ms  |
| 4032x3024 | 第四版优化（SSE优化，一次处理12个像素）  | 1000     | 5.579ms  |
| 4032x3024 | 第五版优化（SSE优化，一次处理15个像素）  | 1000     | 5.843ms  |
| 4032x3024 | 第六版优化（AVX2优化，一次处理10个像素） | 1000     | 3.576ms  |
| 4032x3024 | 第七版优化（AVX2优化+std::async）        | 1000     | 2.626ms  |

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)