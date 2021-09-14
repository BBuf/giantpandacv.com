【GiantPandaCV导语】**以卷积和im2col+gemm实现卷积操作举例,来图解深度学习中Tensor的NC4HW4(其实应该是N{C/4+C%4>0?1:0}HW4),写成CN4HW4方便阅读**.

### 什么是NC4HW4？

- 对于卷积操作, 根据计算机内存排布特点, 按行进行处理.处理完一个通道的数据, 转入下一个通道继续按行处理. 

![卷积操作示意图](https://img-blog.csdnimg.cn/20201110220623890.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 对于一个nchw格式的Tensor来说, 其在计算机中的内存排布是这样的:

![NCHW的Tensor内存排布示意图](https://img-blog.csdnimg.cn/20201110220650313.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 使用cpp一次指令处理一个数据, 用来处理卷积操作,  即循环实现乘法相加即可.

![卷积实现示意图](https://img-blog.csdnimg.cn/20201110220724687.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 现在有一条指令处理4组数据的能力, 比如x86结构的sse指令,arm的neon指令.以及GPGPU的OpenGL和OpenCL,单次处理RGBA四组数据. 如果继续使用nchw内存排布的话, 是这样的. 

![想使用指令集加速卷积，不能直接计算](https://img-blog.csdnimg.cn/2020111022075151.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)



-  根据按行处理特点, 对于Feature和kernel的宽不是4倍数进行处理, 会出现错误. 图中的kernel很明显以已经到了第二行的值。那么有没有方法在按行处理的思想上, 一次处理4个数,而不受影响.答案是有的, 即NC4HW4.即把前4个通道合并在一个通道上, 依次类推, 在通道数不够4的情况下进行补0.


![NC4HW4.即把前4个通道合并在一个通道上](https://img-blog.csdnimg.cn/20201110221040913.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)
![在通道数不够4的情况下进行补0](https://img-blog.csdnimg.cn/20201110221040931.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 经过NC4HW4重排后的Tensor在内存中的排布情况如下:

![经过NC4HW4重排后的Tensor在内存中的排布示意图](https://img-blog.csdnimg.cn/2020111022113576.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 那么, 此时在进行单次指令处理4组数据的处理,就没有问题了.只不过处理结果也是NC4HW4结构的，需要在结果输出加上NC4HW4转nchw.

![使用指令集加速卷积，可以直接计算](https://img-blog.csdnimg.cn/20201110221153849.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


### NC4HW4中使用im2col+gemm实现卷积:

- im2col+gemm在深度学习中是最常用的对卷积进行加速计算的方案。最早在caffe框架中支持。思路如下:

![卷积示意图](https://img-blog.csdnimg.cn/20201110221237831.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

- 使用im2col+gemm进行计算:

![Im2Col图解](https://img-blog.csdnimg.cn/20201110221424763.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- 对于NC4HW4内存排布的Tensor来说,同样可以采用im2col+gemm来处理.
- 有如下卷积,可以使用NC4HW4内存排布方式,使用指令集优化对卷积进行加速.

![卷积示意图](https://img-blog.csdnimg.cn/20201110221450563.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

- NCHW转NC4HW4.

![NCHW转NC4HW4](https://img-blog.csdnimg.cn/20201110221512771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- NC4HW4对feature进行im2col

![NC4HW4对feature进行im2col示意图](https://img-blog.csdnimg.cn/20201110221615525.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


- NC4HW4对kernel进行im2col

![NC4HW4对kernel进行im2col](https://img-blog.csdnimg.cn/20201110221635718.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

- 使用SSE,Neon,OpenCL或OpenGL实现Gemm.

![使用SSE,Neon,OpenCL或OpenGL实现Gemm](https://img-blog.csdnimg.cn/20201110221650287.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


### 最后

- 欢迎关注我和BBuf及公众号的小伙伴们一块维护的一个深度学习框架Msnhnet:  https://github.com/msnh2012/Msnhnet


### 推荐阅读
- [多平台轻量级PyTorch模型推理框架MsnhNet](https://mp.weixin.qq.com/s/U47E4ZF8OPstCmgnIR7TuA)
- [Pytorch转Msnhnet模型思路分享](https://mp.weixin.qq.com/s/gSffbAQf8CcOJkusjN09FA)
- [视觉算法工业部署及优化学习路线分享](https://mp.weixin.qq.com/s/_24JHFw8HuNXFKfzM-Y5Eg)
- [基于how-to-optimize-gemm初探矩阵乘法优化](https://mp.weixin.qq.com/s/EgC2puTsIfEk1uvgWlHXZA)
- [详解Im2Col+Pack+Sgemm策略更好的优化卷积运算](https://mp.weixin.qq.com/s/lqVsMDutBwsjiiM_NkGsAg)
- [Im2Col+GEMM的改进方法MEC，一种更加高效的卷积计算策略](https://mp.weixin.qq.com/s/f8kixUwHalw2NNL005n4EQ)

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)