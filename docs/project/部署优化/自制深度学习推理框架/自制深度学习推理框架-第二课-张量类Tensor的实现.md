# Tensor张量类的实现

**视频链接:** [https://www.bilibili.com/video/BV1Ed4y1v7Gb](https://www.bilibili.com/video/BV1Ed4y1v7Gb)

**Github 链接:** [https://github.com/zjhellofss/KuiperInfer ](https://github.com/zjhellofss/KuiperInfer ) 欢迎star和PR

## 课程logo

`Kuiper`是太阳系小行星天体带，有兴趣的同学可以自行百度`柯伊伯带`。之所以取这个名字，我是想表达这个框架是具有一定“边缘”属性，同时希望更多的人像“小行星”一样加入到这个星带中来。
![在这里插入图片描述](https://img-blog.csdnimg.cn/f895bda32d0b4c2c8f93d096da7bed3a.jpeg)

## 关于维度的知识

在Tensor张量中，共有三维数据进行顺序存放，分别是Channels(维度)，Rows(行高), Cols(行宽)。

三维矩阵我们可以看作多个连续的二维矩阵组成，最简单的方法就是使用嵌套的vector数组，但是这种方法非常不利于数据的访问（尤其是内存不连续的问题）修改以及查询，特别是在扩容的时候非常不方便，不能很好地满足使用需求。

因此，综合考虑灵活性和开发的难易度，我们会以`Armadillo`类中的`arma::mat`(矩阵 `matrix`)类和`arma::cube`作为数据管理(三维矩阵)类来实现Tensor. 一个`cube`由多个`matrix`组成，`cube`又是我们代码中`Tensor`类中的数据实际管理者。

首先我们讲讲`Tensor`类和`Armadillo`中两个类的关系，什么是数据的实际管理者，可以从下方代码看出`Tensor`类中的数据均由`arma::cube`类进行管理、删改、扩充，我们设计的类以`arma::cube`为基础实现了`Tensor`类，我们主要是提供了更方便的访问方式和对外接口。

```cpp
class Tensor{
public:
	void Function1(); // 对外接口1
	void Function2(); // 对外接口2
private:
	std::vector<uint32_t> raw_shapes_;
	arma::fcube data_; //具体的数据管理者
}
```

`arma::cube`是一个三维矩阵，分别是通道维度(slices或者channels)，行维度(rows)和列维度(cols)，请看下图1, 图中是两个5行3列的矩阵，蓝色的区域是数据的实际存储区，灰色和白色部分仅用作示意，在内存中实际不存在。

![](https://img-blog.csdnimg.cn/img_convert/f5cf18a683b3251d1e327f0da1461708.jpeg)

一个`cube`类由多个这样的`Matrix`组成，图1中表示的情况是`arma::cube(2, 5, 3)`, 表示当前的三维矩阵共有2个矩阵构成，每个矩阵都是5行3列的。如果放在我们项目中需要这样声明: `Tensor tensor(2, 5, 3)`. 

下图2是这种情况下的三维结构图，可以看出一个`Cube`一共有两个`Matrix`，也就是共有两个Channel. 一个Channel中存放一个`Matrix`. `Matrix`的行宽均为Rows和Cols.
![](https://img-blog.csdnimg.cn/img_convert/79f7b9a5c7409c72a9e73959a3fc062f.jpeg)


## Tensor方法总览

我们从上面可以知道，我们的`Tensor`类是对`armdillo`中`cube`类的封装，`cube`是多个`Matrix`的集合（二维矩阵的集合），关系图如上图1、图2.  我们在这里对`KuiperInfer`中`Tensor`类的方法进行一个总览，其中我们会让大家亲自动手实现两个方法（加粗的两个），只有动手起来才能参与其中。


| leight-aligned 类名                       | 功能                                                         |
| :---------------------------------------- | :----------------------------------------------------------- |
| leight-aligned  rows()                    | 返回Tensor的行数                                             |
| cols()                                    | 返回Tensor的列数                                             |
| Fill(float value)                         | 填充Cube中的数据，以value值填充                              |
| **Padding(std::vector<uint32_t\>values)** | 调整Matrix的维度，让Rows和Cols变大一点：）                   |
| at(uint32_t channel,  row,  col)          | 返回Cube中第channel维，第row行，第col列的数据。              |
| index(uint32_t offset)                    | 以另外一种方法来返回数据，返回Cube中第offset个数据，比如说在row行，col列，c维的一个数据，除了可以用tensor.at(c, row, col)方法访问。我们也可以通过`tensor.index(c × Rows × Cols + row × Cols + col)`这种方式来访问。可以参考**图4,** 展平后的Matrix, at接口更适合用来存放展平后的数据。 |
| **Fill(std::vector<float\>values)**       | 另外一个Fill方法， 我们需要以values中的所有数据去填充Tensor中的数据管理器cube类，注意values中数据的数量要等于Cube的行数×列数×维度 |
| Flatten()                                 | 将三维的矩阵展开铺平为一维的。                               |


![](https://img-blog.csdnimg.cn/img_convert/b11fbe43855a97d8dd78ad64e0070b7a.jpeg)

## Tensor类模板

`Tensor`共有两个类型，一个类型是`Tensor<float>`，另一个类型是`Tensor<uint8_t>`, `Tensor<uint8_t>` 可能会在后续的量化课程中进行使用，目前还暂时未实现，所以在之后的文章中我们以`Tensor`来指代`Tensor<float>`.

### 如何创建一个Tensor

`Tensor<float> tensor(3, 5, 3)`. 在我们的`KuiperInfer`项目中，我们可以用一个非常简单的方式来创建一个张量实例，在如上的定义中，我们得到了一个通道数量为3,行数(rows)为5,列数(cols)为3的tensor变量。

### 如何访问Tensor中数据(我们要大家实现的功能)

我们将在这个项目中为Tensor类定义多种访问内部数据的方式。首先要讲的是顺序访问方式，在tensor变量中，我们可以使用`tensor.at(0, 1, 2)`得到tensor变量中第0通道，第1行，第2列中存放的元素。

另外一种，我们可以使用`tensor.index(0)`这种方法来得到tensor变量中第0个数据 。我会在作业系统中给予大家充分的提示，让大家准确无误地把代码写出来。

从下图中可以看出，`tensor.at(0,1,2)`就是访问图中对应位置的点。第1个矩阵(channel = 0)中第2行(row = 1)，第3列(col=2)中的数据。

![](https://img-blog.csdnimg.cn/img_convert/325b6b79d787b9e6f6faf79ca012be80.jpeg)

## 再谈谈Tensor类中数据的排布

我们以具体的图片作为例子，来讲讲`Tensor`中数据管理类`arma::cube`的数据排布方式，`Tensor`类是`arma::cube`对外更方便的接口，所以说`armadillo::cube`怎么管理内存的，`Tensor`类就是怎么管理内存的，希望大家的能理解到位。

如下图中的一个`Cube`，`Cube`的维度是2,每个维度上存放的是一个`Matrix`，一个`Matrix`中的存储空间被用来存放一张图像(Lena) .

 一个channel 是一个`Matrix`，`Matrix1`存放在`Cube`第1维度(channel 1)上，`Matrix2`存放在`Cube`的第2维度上(channel 2). `Matrix1`和`Matrix2`的Rows和Cols均代表着图像的高和宽，在本例中就是512和384.
![](https://img-blog.csdnimg.cn/img_convert/266d768269341a34057c070722b4202f.jpeg)



## 数据在内存中的顺序问题
如果将顺序的一组数据`[0,1,2,3,4,5,...,15]`存放到一个大小为4×4的Matrix中，那么大家需要注意一个问题，我们的数据管理类`Tensor(arma::cube)`是列主序的，这一点和`Opencv cv::Mat`或者`Python numpy`有一些不同。列主序在内存中的顺序如下表：
![](https://img-blog.csdnimg.cn/img_convert/f04eae56f89b80abdf38ccb3ad6b6195.jpeg)

## 第二节的作业

- git clone https://gitee.com/fssssss/KuiperCourse
- mkdir build
- git checkout second
- cd build
- cmake .\.
- make

**Tips:** 
Padding作业提示：https://arma.sourceforge.net/docs.html#insert 
Fill作业提示：请看视频提示
