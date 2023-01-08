## 课程logo

Kuiper是太阳系小行星天体带，有兴趣的同学可以自行百度。之所以取这个名字，我是想表达，这个框架是具有一定“边缘”属性，然后希望更多的人像“小行星”一样加入到这个星带中来。 

![image-20221222214001402](https://pic.imgdb.cn/item/63a45e1a08b6830163f6614b.jpg)

## 关于维度的预备知识

在Tensor张量中，共有三维数据进行顺序存放，分别是Channels(维度)，Rows(行高), Cols(行宽)，三维矩阵我们可以看作多个连续的二维矩阵组成，最简单的方法就是使用嵌套的vector数组，但是这种方法非常不利于数据的访问（尤其是内存不连续的问题）修改以及查询，特别是在扩容的时候非常不方便,能满足使用需求。

因此，综合考虑灵活性和开发的难易度，我们会以Armadillo类中的arma::mat(矩阵 matrix)类和arma::cube作为数据管理(三维矩阵)类来实现Tensor 我们库中类的主体，一个cube由多个matrix组成，cube又是Tensor类中的数据实际管理者。

首先我们讲讲Tensor类和Armadillo中两个类的关系，可以从下方图看出Tensor类中的数据均由arma::cube类进行管理扩充，我们设计的类以arma::cube为基础实现了Tensor类，我们主要是提供了更方便的访问方式和对外接口。

![](https://pic.imgdb.cn/item/63a45e6908b6830163f6dcd6.jpg)

arma::cube是一个三维矩阵，分别是通道维度(slices或者channels)，行维度(rows)和列维度(cols)，请看下图1, 图中是两个5行3列的矩阵，蓝色的区域是数据的实际存储区，灰色和和白色部分仅用作示意，在内存中实际不存在。

![](https://pic.imgdb.cn/item/63a45f2308b6830163f7eac7.jpg)

一个cube类由多个这样的Matrix组成，图1中表示的情况是arma::cube(2, 5, 3), 表示当前的三维矩阵共有2个矩阵构成，每个矩阵都是5行3列的。如果放在我们项目中会以这形式提供 Tensor tensor(2, 5, 3). 

下图2是这种情况下的三维结构图，可以看出一个Cube一共有两个Matrix，也就是共有两个Channel. 一个Channel放一个Matrix. Matrix的行宽均为Rows和Cols.

![](https://pic.imgdb.cn/item/63a45f7b08b6830163f86533.jpg)



## Tensor方法总览

我们从上面可以知道，我们的Tensor类是对armdillo库中cube类的封装，cube是多个Matrix的集合（二维矩阵的集合），关系图如上图1、图2.  我们在这里对KuiperInfer中Tensor类的方法进行一个总览，其中我们会让大家亲自动手实现两个方法（加粗的两个），只有动手起来才能参与其中。

| 类名                                      | 功能                                                         |
| ----------------------------------------- | ------------------------------------------------------------ |
| rows()                                    | 返回Tensor的行数                                             |
| cols()                                    | 返回Tensor的列数                                             |
| Fill(float value)                         | 填充Cube中的数据，以value值填充                              |
| **Padding(std::vector<uint32_t\>values)** | 调整Matrix的维度，让Rows和Cols变大一点：）                   |
| at(uint32_t channel,  row,  col)          | 返回Cube中第channel维，第row行，第col列的数据。              |
| index(uint32_t offset)                    | 以另外一种方法来返回数据，返回Cube中第offset个数据，比如说在row行，col列，c维的一个数据，除了可以用tensor.at(c, row, col)方法访问。我们也可以通过tensor.index(c × Rows × Cols + row × Cols + col)这种方式来访问。可以参考图4, 展平后的Matrix, at接口更适合用来存放展平后的数据。 |
| **Fill(std::vector<float\>values)**       | 另外一个Fill方法， 我们需要以values中的所有数据去填充Tensor中的数据管理器cube类，注意values中数据的数量要等于Cube的行数×列数×维度 |
| Flatten()                                 | 将三维的矩阵展开铺平为一维的。                               |

![](https://pic.imgdb.cn/item/63a460e108b6830163fa4d72.jpg)

## Tensor类模板

Tensor共有两个类型，一个类型是Tensor<float\>，另一个类型是Tensor<uint8_t>, Tensor<uint8_t> 可能会在后续的量化课程中进行使用，目前还暂时未实现，所以在之后的文章中我们以Tensor来指代Tensor<float\>.

### 如何创建一个Tensor

Tensor<float\> tensor(3, 5, 3). 在我们的KuiperInfer项目中，我们可以用一个非常简单的方式来创建一个张量实例，在如上的定义中，我们得到了一个通道数量为3,行数(rows)为5,列数(cols)为3的tensor变量。

### 如何访问Tensor中数据(我们要大家实现的功能)

我们将在这个项目中为Tensor类定义多种访问内部数据的方式。首先要讲的是顺序访问方式，在tensor变量中，我们可以使用tensor.at(0, 1, 2)得到tensor变量中第0通道，第1行，第2列中存放的元素。

另外一种，我们可以使用tensor.index(0)这种方法来得到tensor变量中第0个数据 。我会在作业系统中给予大家充分的提示，让大家准确无误地把代码写出来。从下图中可以看出，tensor.at(0,1,2)就是访问图中对应位置的点。第1个矩阵(channel = 0)中第2行(row = 1)，第3列(col=2)中的数据。

![](https://pic.imgdb.cn/item/63a4618908b6830163fb38e2.jpg)

## 再谈谈Tensor类中数据的排布

我们以具体的图片作为例子，来讲讲Tensor中数据管理类arma::cube的数据排布方式，Tensor类是arma::cube对外更方便的接口，所以说armadillo::cube怎么管理内存的，Tensor类就是怎么管理内存的，希望大家的能理解到位。

如下图中的一个Cube，Cube的维度是2,每个维度上存放的是一个Matrix，一个Matrix中的存储空间被用来存放一张图像(lena) . 一个框内(channel) 是一个Matrix，Matrix1存放在Cube第1维度(channel 1)上，Matrix2存放在Cube的第2维度上(channel 2). Matrix1和Matrix2的Rows和Cols均代表着图像的高和宽，在本例中就是512和384.

![](https://pic.imgdb.cn/item/63a4620b08b6830163fbf1fa.jpg)

如果将顺序的一组数据[0,1,2,3,4,5....128]存放到一个大小为4×4的Matrix中，那么大家需要注意一个问题，我们的数据管理类Tensor(arma::cube)是列主序的，这一点和Opencv cv::Mat或者python numpy有一些不同。列主序在内存中的顺序如下表：

![](https://pic.imgdb.cn/item/63a462a708b6830163fcc30a.jpg)

## 作业提示

1. git clone https://gitee.com/fssssss/KuiperCourse

2. mkdir build

3. git checkout second
4. cd build
5. cmake ..
6. make
7. Padding作业提示：https://arma.sourceforge.net/docs.html#insert
8. Fill作业提示：请看视频提示

