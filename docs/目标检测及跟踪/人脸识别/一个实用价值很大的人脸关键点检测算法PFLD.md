# 1. 前言

PFLD全称A Practical Facial Landmark Detector是一个精度高，速度快，模型小的人脸关键点检测模型。在移动端达到了超实时的性能（模型大小2.1Mb，在Qualcomm ARM 845 处理器上达到140fps），作者分别来自武汉大学，天津大学，腾讯AI Lab，美国天普大学，有较大的实用意义。

# 2. 挑战

人脸关键点检测作为人脸相关应用中的一个基础任务面临了很多挑战，包括检测精度，处理速度，模型大小这些因素都要考虑到，并且在现实场景中很难获取到质量非常高的人脸，所以人脸关键点检测主要面临下面几个挑战：

- **局部变化**：现实场景中人脸的表情，广告，以及遮挡情况都有较大的变化，如Figure1所示
- **全局变化**：姿态和成像质量是影响图像中人脸的表征的两个主要因素，人脸全局结构的错误估计将直接导致定位不准
- **数据不平衡**：不平衡的数据使得算法模型无法正确表示数据的特征
- **模型的性能**：由于手机和嵌入式设备计算性能和内存资源的限制，必须要求检测模型的size小处理速度快



![Figure 1](https://img-blog.csdnimg.cn/20200610165232301.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 3. 创新点

总的来说，本文设计的PFLD在复杂情况下也可以保持高精度。针对全局变化，PFLD采用辅助网络来估计人脸样本的集合信息。针对数据不平衡，设计新的损失函数，加大对难样本的惩罚力度。使用multi-scale fc层扩展感受野精确定位人脸的特征点。使用Mobilenet Block构建网络的Backbone提升模型的推理速度及减少模型的计算量。



# 4. PFLD网络结构

PFLD的网络结构如下图所示：



![Figure2 PFLD的整体结构](https://img-blog.csdnimg.cn/20200610165608519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)



其中黄色虚线圈起来的部分表示主分支网络，用于预测关键的位置。绿色虚线圈起来的是head pose辅助网络。这样在训练关键点回归的同时预测人脸姿态，从而修改损失函数，使得模型更加关注那些稀有以及姿态角度过大的样本，从而提高预测的精度。

可以看到在主分支网络中，PFLD并没有采用VGG16，ResNet50等大模型。但为了增强模型的表达能力，对MobilenetV2的输出特征进行了结构上的修改，如Figure2中主分支网络的右边所示。PFLD融合了3个尺度的特征来增加模型的表达能力。



## 4.1 损失函数设计

我们知道一般的回归损失是MSE或者Smooth L1 Loss，但它们都难以应对数据不均衡的情况，以MSE Loss为例，损失函数可以写成：

![公式1](https://img-blog.csdnimg.cn/20200610181201865.png)

其中$M$表示人脸样本的数量，$N$表示每张人脸预设的需要检测的特征点数目，$||.||$在本文表示L2距离，$\gamma_n$表示不同类型样本的不同权重。


而RetinaNet中提出的Focal Loss可以较好的应对二分类中的数据不均衡情况，受到这一启发，作者设计了下面的损失函数来缓解数据不均衡的情况：

![PFLD loss](https://img-blog.csdnimg.cn/20200610181453633.png)

- $\sum_{c=1}^Cw_n^c\sum_{k=1}^K(1-cos\theta_{n}^k)$代表权重$\gamma_n$。
- $\theta^1$,$\theta^2$,$\theta^3$ ($K=3$)分别表示GT和Prediction在**yaw、pitch、roll**三种角度之间的偏差，角度越大$cos$值越小，权重越大。其中**pitch**代表上下翻转，**yaw**代表水平翻转，**roll**代表平面内旋转，都表示人脸的一种姿态。
- $C$表示不同的类别的人脸: 正脸、侧脸、抬头、低头、表情以及遮挡情况，$w_n^c$根据样本类别分数进行调整，论文中使用的分数样本数的导数计算的。
- $d_n^m$由主分支网络计算得到，$\theta_n^k$由辅助网络计算得到，然后由Loss来建立联系。


## 4.2 辅助网络的细节 
PFLD在训练过程中引入了一个辅助网络用以监督PFLD网络模型的训练，如Figure2中绿色虚线里的部分。该子网络仅在训练的阶段起作用，在推理阶段不起作用。

该子网络对每一个输入的人脸样本进行三维欧拉角估计，它的Ground Truth由训练数据中的关键点信息进行估计，虽然估计不太精确，但是作为区分数据分布的依据已经足够了，因为这个辅助网络的目的是监督和辅助关键点检测主分支。另外需要注意的一点是，这个辅助网络的输入不是训练数据，而是PFLD主分支网络的中间输出（第4个Block）。

## 4.3 主分支网络和辅助网络的详细配置
主分支网络和辅助网络的详细配置表如下：

![主分支网络的配置](https://img-blog.csdnimg.cn/20200610184031628.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![辅助网络的配置](https://img-blog.csdnimg.cn/20200610184050302.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 5. 实验结果
下面的Figure3展示了在300W数据集上PFLD和其它一些经典算法的CED曲线对比：

![在300W数据集上PFLD和其它一些经典算法的CED曲线对比](https://img-blog.csdnimg.cn/20200610184313569.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

下面的Table3展示了PFLD在各个平台上的性能表现：

![Table3](https://img-blog.csdnimg.cn/20200610202202932.png)

下面的Table4展示了不同的评价标准和不同的数据子集的评价指标：

![Table4](https://img-blog.csdnimg.cn/20200610202328724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Table5还展示了FPLD在AFLW数据集上的表现：

![FPLD在AFLW数据集上的表现](https://img-blog.csdnimg.cn/20200610202418801.png)

最后Figure4还展示了一些在具有挑战性的样本上的表现：

![Figure4](https://img-blog.csdnimg.cn/2020061020263182.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 总结
总的来说FPLD是一个idea非常好并且实用价值比较大的人脸关键点检测算法，无论是人脸姿态估计子网络的引入还是针对数据不平衡重新设计损失函数都是值得借鉴的。

# 7. 参考文章

- https://zhuanlan.zhihu.com/p/73546427
- https://blog.csdn.net/wwwhp/article/details/88361422
- https://arxiv.org/pdf/1902.10859.pdf
- https://github.com/polarisZhao/PFLD-pytorch

---------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)