# 前言
今天来介绍一个工程上的常用Trick，即折叠Batch Normalization，也叫作折叠BN。我们知道一般BN是跟在卷积层后面，一般还会接上激活函数，也就是`conv+BN+relu`这种基本组件，但在部署的时候前向推理框架一般都会自动的将BN和它前面的卷积层折叠在一起，实现高效的前向推理网络，今天我就从原理和实现来讲一讲这个操作。

# 原理

**我们知道卷积层的计算可以表示为：**

$Y = W * X + B$

**然后BN层的计算可以表示为：**

$\mu = \frac{1}{m}\sum_{i=1}^mx_i$

$\sigma^2=\frac{1}{m}\sum_{i=1}^m(x_i-\mu)^2$

$x_i=\frac{x_i-\mu}{\sqrt{\sigma^2+ \epsilon}}$

$y_i=\gamma * x_i + \beta$

**我们把二者组合一下，公式如下：**

$Y=\gamma*(\frac{(W*X+B)-\mu}{\sqrt{\sigma^2+\epsilon}})+\beta$

$Y=\frac{\gamma*W}{\sqrt{\sigma^2+\epsilon}}*X+\frac{\gamma*(B-\mu)}{\sqrt{\sigma^2+\epsilon}}+\beta$

然后令$a = \frac{\gamma}{\sqrt{\delta^2+\epsilon}}$

**那么，合并BN层后的卷积层的权重和偏置可以表示为：**

$W_{merged}=W*a$

$B_{merged}=(B-\mu)*a+\beta$

**值得一提的是，一般Conv后面接BN的时候很多情况下是不带Bias的，这个时候上面的公式就会少第二项。**

# Pytorch代码实现
原理介绍完了，接下来我们来看一下代码实现，就以Pytorch的代码为例好了。


```cpp
def fuse(conv, bn):
    global i
    i = i + 1
    # ********************BN参数*********************
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight
    beta = bn.bias
    # *******************conv参数********************
    w = conv.weight
    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    if(i >= 2 and i <= 7):
        b = b - mean + beta * var_sqrt
    else:
        w = w * (gamma / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
        b = (b - mean)/var_sqrt * gamma + beta
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         groups=conv.groups,
                         bias=True)
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv
```

可以看到这个代码实现和上面介绍的公式完全对应，实现了折叠BN的效果。

# Caffe代码实现
下面这个代码实现了将Mobilenet Caffe模型的卷积和BN进行合并的代码，亲测无误，放心使用。

**由于完整代码太长，我放这个工程里了**：https://github.com/BBuf/cv_tools/blob/master/merge_bn.py

# 思考

- 从公式推导可知，只有Conv后接BN才可以折叠，Preact结构的ResNet不可以折叠BN。
- 如果是反卷积接的BN，如何合并？

# 效果

在参考2的博客中看到看到了折叠BN的速度测试结果，分别在1080Ti和i7 CPU上对比了Resnet50 模型合并BN层前后的性能，可以看到分类精度保持不变，速度显著提升，具体测试结果如下：

| 模型               | CPU前向时间 | GPU前向时间 |
| ------------------ | ----------- | ----------- |
| Resnet50（合并前） | 176.17ms    | 11.03ms     |
| Resnet50（合并后） | 161.69ms    | 7.3ms       |
| 提升               | 10%         | 51%         |


# 附录
- 参考1：https://github.com/666DZY666/model-compression
- 参考2：https://blog.csdn.net/kangdi7547/article/details/81348254


# 推荐阅读
- [深度学习算法优化系列一 | ICLR 2017《Pruning Filters for Efficient ConvNets》](https://mp.weixin.qq.com/s/hbx62XkEPF61VPiORGpxGw)
- [深度学习算法优化系列二 | 基于Pytorch的模型剪枝代码实战](https://mp.weixin.qq.com/s/4akToQe0Sy5ze1quKhDDBQ)
- [深度学习算法优化系列三 | Google CVPR2018 int8量化算法](https://mp.weixin.qq.com/s/SWp61rQObczIRMO6-D3bog)
- [深度学习算法优化系列四 | 如何使用OpenVINO部署以Mobilenet做Backbone的YOLOv3模型](https://mp.weixin.qq.com/s/PdqxB1olpT5lEqg8pHDLFw)
- [深度学习算法优化系列五 | 使用TensorFlow-Lite对LeNet进行训练后量化](https://mp.weixin.qq.com/s/MSnkltSHGsBF9ddwN4wZKQ)
- [深度学习算法优化系列六 | 使用TensorFlow-Lite对LeNet进行训练时量化](https://mp.weixin.qq.com/s/vbFegPQg5omMlwNn6tSmhQ)
- [深度学习算法优化系列七 | ICCV 2017的一篇模型剪枝论文，也是2019年众多开源剪枝项目的理论基础](https://mp.weixin.qq.com/s/2h9S-qk99NDTkOurrLX6_g)
- [深度学习算法优化系列八 | VGG，ResNet，DenseNe模型剪枝代码实战](https://mp.weixin.qq.com/s/f6IHgTctf0HqlWTuxi8qjA)
- [深度学习算法优化系列九 | NIPS 2015 BinaryConnect](https://mp.weixin.qq.com/s/o-ZqWzZBlljnpYLweZm2_A)
- [深度学习算法优化系列十 | 二值神经网络(Binary Neural Network，BNN)](https://mp.weixin.qq.com/s/fX3wYKJj3mtSCuZRdRykXw)

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)