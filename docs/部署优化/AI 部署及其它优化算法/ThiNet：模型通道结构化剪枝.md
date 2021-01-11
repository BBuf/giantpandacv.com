【GiantPandaCV】本文介绍了ThiNet模型通道结构化剪枝算法，文章的主要思路是：**ThiNet是基于filter即卷积核剪枝，大白话就是找到一组channel的输出跟全部channel的输出之间的误差最小（采用均方误差/最小二乘法去衡量），那么就可以用这组channel来代替全部channel。文章中的说法是，找到一组channle的子集（其误差最小）可以替代channel的全集**。本文由作者授权首发于GiantPandaCV公众号。

# 0、 介绍

ThiNet是南京大学lamda实验室出品，是ICCV 2017的文章，文章全名《ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression》。

文章的主要思路是：ThiNet是基于filter即卷积核剪枝，大白话就是找到一组channel的输出跟全部channel的输出之间的误差最小（采用均方误差/最小二乘法去衡量），那么就可以用这组channel来代替全部channel。文章中的说法是，找到一组channle的子集（其误差最小）可以替代channel的全集。

ThiNet剪枝流程：选择filter子集、剪枝、finetune，如下图



![ThiNet剪枝流程](https://img-blog.csdnimg.cn/20210111215720498.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)

所以算法的实现的核心在于**如何进行filter选择**。

# 一、通道选择（channel selection）

文章采用贪心算法选择filter子集（也就是剩下来的filter）。这里，很多思路是找到可以丢弃的filter，ThiNet是迭代式layer by layer的剪枝。

思路1（正向思路）：根据通道重要性判断找到重要的filter，保留下来，然后迭代式进行直到压缩率达到预设要求。

思路1的方法会有一个问题就是，留下来filter的数量是从大到小的变化的，那么按照思路1计算量会很大，因为留下来的filter（记为S）通常要去被移除的filter（记为T）多，所以有思路2：根据通道重要性判断找到要剪枝（丢弃）的filter，然后迭代式剪枝进行直到压缩率达到预设要求（丢弃一定数量的filter）。

公式1：

$$y=\sum_{c=1}^{C} \sum_{k_{1}=1}^{K} \sum_{k_{2}=1}^{K} \widehat{\mathcal{W}}_{c, k_{1}, k_{2}} \times x_{c, k_{1}, k_{2}}+b$$

公式2：

$$\hat{x}_{c}=\sum_{k_{1}=1}^{K} \sum_{k_{2}=1}^{K} \widehat{\mathcal{W}}_{c, k_{1}, k_{2}} \times x_{c, k_{1}, k_{2}}$$



公式1和公式2，可以简化为公式3：

$$\hat{y}=\sum_{c=1}^{C} \hat{x}_{c}$$， $$\hat{y}=y-b$$



公式1-3是为了简化公式表示的等效变换。



公式5：

$$\begin{aligned}
\underset{S}{\arg \min } & \sum_{i=1}^{m}\left(\hat{y}_{i}-\sum_{j \in S} \hat{\mathbf{x}}_{i, j}\right)^{2} \\
\text { s.t. } &|S|=C \times r, \quad S \subset\{1,2, \ldots, C\}
\end{aligned}$$

变为公式6：

$$\begin{aligned}
\underset{T}{\arg \min } & \sum_{i=1}^{m}\left(\sum_{j \in T} \hat{\mathbf{x}}_{i, j}\right)^{2} \\
\text { s.t. } &|T|=C \times(1-r), \quad T \subset\{1,2, \ldots, C\}
\end{aligned}$$

其中，S ∪ T = {1, 2, . . . , C}，S ∩ T = ∅，r是压缩率，C是filter数量。

基于贪心算法选择filter子集的算法如图:

![贪心算法](https://img-blog.csdnimg.cn/20210111215745661.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


```python
def channel_selection(inputs, module, sparsity=0.5, method='greedy'):
    """
    选择当前模块的输入通道，以及高度重要的通道。
    找到可以使现有输出最接近的输入通道。
    
    :param inputs: torch.Tensor, input features map
    :param module: torch.nn.module, layer
    :param sparsity: float, 0 ~ 1 how many prune channel of output of this layer
    :param method: str, how to select the channel
    :return:
        list of int, indices of channel to be selected and pruned
    """
    num_channel = inputs.size(1)  # 通道数
    num_pruned = int(math.ceil(num_channel * sparsity))  #  输入需要删除的通道数
    num_stayed = num_channel - num_pruned

    print('num_pruned', num_pruned)
    if method == 'greedy':
        indices_pruned = []
        while len(indices_pruned) < num_pruned:
            min_diff = 1e10
            min_idx = 0
            for idx in range(num_channel):
                if idx in indices_pruned:
                    continue
                indices_try = indices_pruned + [idx]
                inputs_try = torch.zeros_like(inputs)
                inputs_try[:, indices_try, ...] = inputs[:, indices_try, ...]
                output_try = module(inputs_try)
                output_try_norm = output_try.norm(2) #这里就是公式6
                if output_try_norm < min_diff:
                    min_diff = output_try_norm
                    min_idx = idx
            indices_pruned.append(min_idx)

        indices_stayed = list(set([i for i in range(num_channel)]) - set(indices_pruned))
        
    inputs = inputs.cuda()
    module = module.cuda()

    return indices_stayed, indices_pruned
```



# 二、最小化重构误差（Minimize the reconstruction error）

首先先来看看numpy.linalg.lstsq()，是线性矩阵方程的最小二乘法求解。

最小二乘法的公式为：$$J(\theta)=\min \sum_{i=1}^{m}\left(f\left(x_{i}\right)-y_{i}\right)^{2}$$ 

| **方法**                                                     | **描述**                         |
| ------------------------------------------------------------ | -------------------------------- |
| **[linalg.lstsq](https://numpy.org/devdocs/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)(a, b[, rcond])** | **返回线性矩阵方程的最小二乘解** |

```python
numpy.linalg.lstsq(a, b, rcond='warn')
# 将least-squares解返回线性矩阵方程。
```

$\hat{\mathbf{w}}=\underset{\mathbf{w}}{\arg \min } \sum_{i=1}^{m}\left(\hat{y}_{i}-\mathbf{w}^{\mathrm{T}} \hat{\mathbf{x}}_{i}^{*}\right)^{2}$

其中, $\hat{\mathbf{x}}_{i}$是通道选择后的训练样本，可以通过$\hat{\mathbf{w}}=\left(\mathbf{X}^{\mathrm{T}} \mathbf{X}\right)^{-1} \mathbf{X}^{\mathrm{T}} \mathbf{y}$ 求解

该方法是**每一个通道赋予权重来进一步地减小重构误差。文章说这相当于跟finetune一个很好的初始化**。

```python
def weight_reconstruction(module, inputs, outputs, use_gpu=False):
    """
    reconstruct the weight of the next layer to the one being pruned
    :param module: torch.nn.module, module of the this layer
    :param inputs: torch.Tensor, new input feature map of the this layer
    :param outputs: torch.Tensor, original output feature map of the this layer
    :param use_gpu: bool, whether done in gpu
    :return: void
    """
    if module.bias is not None:
        bias_size = [1] * outputs.dim()
        bias_size[1] = -1
        outputs -= module.bias.view(bias_size)  # 从 output feature 中减去 bias （y - b）
    if isinstance(module, torch.nn.Conv2d):
        unfold = torch.nn.Unfold(kernel_size=module.kernel_size, dilation=module.dilation,
                                 padding=module.padding, stride=module.stride)

        unfold.eval()
        x = unfold(inputs)  # 展开到以一个面片（reception field）为列的三维数组 （N * KKC * L （number of fields））
        x = x.transpose(1, 2)  #  transpose (N * KKC * L) -> (N * L * KKC)
        num_fields = x.size(0) * x.size(1)
        x = x.reshape(num_fields, -1)  # x: (NL * KKC)
        y = outputs.view(outputs.size(0), outputs.size(1), -1)  # 将一个特征映射展开为一行数组 （N * C * WH）
        y = y.transpose(1, 2)  #  transpose (N * C * HW) -> (N * HW * C), L == HW
        y = y.reshape(-1, y.size(2))  # y: (NHW * C),  (NHW) == (NL)

        if x.size(0) < x.size(1) or use_gpu is False:
            x, y = x.cpu(), y.cpu()
            
	#上面一系列的reshape的操作是为了调用np.linalg.lstsq这个函数，利用最小二乘法求解weight
    param, residuals, rank, s = np.linalg.lstsq(x.detach().cpu().numpy(),y.detach().cpu().numpy(),rcond=-1)

    param = param[0:x.size(1), :].clone().t().contiguous().view(y.size(1), -1)
    if isinstance(module, torch.nn.Conv2d):
        param = param.view(module.out_channels, module.in_channels, *module.kernel_size)
    del module.weight
    module.weight = torch.nn.Parameter(param)
```



# 三、对于VGG-16的ThiNet剪枝策略

1、对前面10层剪枝力度大，因为前面10层的feature map比较大，FLOPs占据了超过90%

2、全连接层占据了 86.41%的模型参数，所以将其改成global average pooling layer

3、剪枝是layer by layer，每剪完一个layer finetune一个epoch，到最后一层剪完 finetune 12个epoch

4、在ImagNet上，因为比较多的系列，比如每层压缩率，finetune学习率的变化等， VGG的剪枝细节可以看论文4.2节



# 四、对于ResNet的剪枝策略

对于残差块的剪枝，因为有个add的操作，相加时候维度必须保持一致，所以残差块最后一层输出的filter不改变而只剪枝前面两层，如下所示：

![ResNet剪枝](https://img-blog.csdnimg.cn/20210111215907460.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70#pic_center)


ResNet具体剪枝的学习率、压缩率跟细节，可以查看论文4.3部分



# 五、总结

ThiNet有三个要点：

1、如何进行通道选择，通道的子集与全部通道的全集之间的最小二次乘法误差来做通道重要新判断依据

2、最小化重构误差，给finetune一个初始化卷积核参数

3、对残差网络的剪枝做了适配



# 六，参考链接

原作中文解读：http://www.lamda.nju.edu.cn/luojh/project/ThiNet_ICCV17/ThiNet_ICCV17_CN.html

论文：https://arxiv.org/abs/1707.06342

代码：https://github.com/Roll920/ThiNet

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)