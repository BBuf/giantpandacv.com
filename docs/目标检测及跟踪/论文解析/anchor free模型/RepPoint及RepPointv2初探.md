# RepPoint及RepPointv2初探

【GiantPandaCV导语】本文对anchor-free类检测方案中的基于特征采样点的实现思路实现的RepPoint网络进行了解读。

RepPoint及RepPointv2为近年提出的目标检测模型，作者开放的源码基于mmdetection实现的。该两篇检测方向的论文给我的最大收获是对anchor-free类目标检测算法性能优越性的一定解释性，以及其在anchor-free类检测算法方向上的探索。

code:https://github.com/microsoft/RepPoints

## RepPoint（提出时间 2019年）

**提出背景及面向问题**：作者认为基于anchor的目标检测算法虽然在近年表现出了不错的性能，但是仍旧过于粗糙，主要表现在，检测和分类头从bbox中提取的特征可能受到背景和杂乱语义信息的影响。用我们实际项目中的例子来解释，就是假设我们在标注人体，进行人体检测时，若此人是张开双臂的，为了尽量框全该人体目标，所标记的bbox就会引入大量的背景信息；再者，如果我们提供的训练样本中，人体区域周边有大量的假人，或者行人重叠情况，则基于anchor类目标检测算法，所标记的框中就会包含很多的误导信息，也就是该篇作者说的背景和杂乱语义信息。而这些杂乱信息的引入，就会导致检测模型的性能降低，主要体现在检测框的漂移（检测框会发生晃动或检测结果框不全目标）以及分类置信度分布过于松散，从而降低了正样本和负样本的可分性。

**作者提供解决方案**：作者首先提出了RepPoint的结构通过建立一系列的自适应采样点来代替anchor内的完全采样，也就是在正样本区域通过学习出一组自适应的采样点来找到有代表性的特征，从而提高检测器的性能。然后作者基于该结构实现了一个卷积网络RPDet，并在COCO上进行了验证。

**RepPoint结构**  

![本文提出的RepPoint结构论文展示图](https://img-blog.csdnimg.cn/20210430111910489.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NEQ29sZA==,size_16,color_FFFFFF,t_70)  

如图为作者提出的RepPoint结构流程示意图，具体为：首先初始化$n$个采样点$R={(x_k,y_k)}^{n}_{k=1}$，作者的论文中设置的$n=9$，然后这9个点初始化为某个待检测目标的中心点，再通过可变形卷积得到每个采样点相对于目标中心点的偏移，就得到了当前目标的最终采样点。

**RPDet结构**  

![RepPoint论文提出RPDet网络结构图](https://img-blog.csdnimg.cn/20210430110617876.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NEQ29sZA==,size_16,color_FFFFFF,t_70)  

如图就是作者基于RepPoint结构实现的anchor-free检测网络，首先是输入图像通过backbone+FPN提取多层特征，然后对FPN中的每个像素位置，通过一个RepPoint结构来得到对FPN中每个像素位置的$n$个采样点的偏移，并利用该组RepPoint获得的采样点位置来确定目标的粗定位检测框，接着把第一个RepPint结构得到的偏移量传递到第二个RepPpint结构中，第二个RepPoint结构得到的偏移是相对第一个RepPoint的偏移量来叠加的，得到的是最终每个位置的采样点。并依据最终的采样点来得到目标的最小外接矩形框。分类则是基于第一个RepPoint结构提供的采样点所构成的目标外接矩形框进行的。

**算法其余实现细节**：

1、作者对每个正样本是基于中心点表示，作者解释**采用FPN的意义**就是在FPN中由于不同尺度目标会天然的归属到不同层级的特征图上，且FPN对小目标有较高的分辨率，也降低了两个相同尺度目标在同一个中心点的概率，所以就极大降低了出现目标中心点重叠的情况。

2、关于不同目标在不同的FPN层上的分配，首先的不同的FPN层上目标尺度满足公式$S(B)=\lfloor{log_2(\sqrt{W_b*H_b}/4)}\rfloor$且目标的中心点加上偏移量后还在当前特征图内。

3、损失函数的分配。对第一组RepPoint，损失函数为分类损失+预测的点所形成的目标框与实际的目标框左上角点和右下角点距离值；对第二组RepPoint，只有定位损失值。

**算法结果展示**：

![RPDet性能展示图](https://img-blog.csdnimg.cn/20210430110617940.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NEQ29sZA==,size_16,color_FFFFFF,t_70)  

**总结**：从本篇论文的提出时间和参考文献来看，作者是参考了cornernet和FSAF的，且与同期的FCOS，centernnet互相之间都无引用，也就是这三篇是接近同期推出的anchor-free检测网络。与FCOS和centernet不同的是，该网络生成目标的bbox过程是通过目标中心点到采样点的回归完成的，与FCOS和centernet的此处思路一致，区别就是后续对特征的利用，FCOS更偏向于anchor-base类算法的常用回归头和分类头设计，并通过额外的objectness-head来完成对不同bbox的加权；centernet则是通过较高的特征图分辨率和高斯掩膜的训练trick的设计来对每个目标proposal内的特征进行加权采样，从而强化检测网络的目标定位能力。总之，该篇anchor-free的检测网络通过将特征采样从原本的proposal内均匀加权采样修改为显式的关键点采样来降低干扰信息对前景目标定位和分类的干扰，整个网络的设计也很简洁，对anchor-free类网络为什么能够良好工作具有一定的解释性。

## RepPointV2（提出时间2020年）

**提出背景及面向的问题**：经过19年提出的RepPoint后，作者后续也是看到了同期的FCOS和centernet的良好性能，然后也发现了RPDet对RepPoint结构回归出的目标采样点的强依赖，而我们也知道网络定位部分的性能训练到是非精准是很难的。在V2中，作者就想继续提升网络的回归性能，从而提高整个网络的性能。

**相对于V1版本的改进**：作者首先是参考了验证算法（**Within-box foreground verification**），来得到了当前输入图像的前背景热力图和角点图像，然后用这个辅助分支和之前RPDet的主分支进行联合计算从而提升性能，类似的还有FCOS的objectness分支。不过两者之间也有不同之处，FCOS中的objectness分支是对最后网络生成的多个定位框进行的加权筛选，目的是去除多余的假阳性定位框；而REpPointv2中的辅助1分支，考虑到RepPoint结构是对一定范围内特征点的选取，其实更类似于label-assign过程，(label-assign任务大家可以看公众号之前的推文[《从label assign角度理解当前目标检测进展》](https://mp.weixin.qq.com/s/qYEMhaWS_0lRvzqvmI-Ajg)或者搜索论文ATSS)，通过一定的手段来强化正样本和负样本选取的能力，从而使得网络采样点更加具有代表力。

![RepPointv2网络流程说明图](https://img-blog.csdnimg.cn/20210430112141945.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1NEQ29sZA==,size_16,color_FFFFFF,t_70)

**RepPointv2性能图表**:

![RepPointv2-COCO上与其它网络性能对比表](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9aOHcyRXhyRmdEek5vMlFKTGt6UE9jaEs0QUpGSVJ2WVBWUXIxVHl3VWpkTlYyNE1FV1JUdWtCMW5pYzBEeWpteDRGeUtnblhWcGpJdGRTWGJabEVMWWcvNjQw?x-oss-process=image/format,png)

**总结**：RepPointv2论文相对v1论文，主要是通过添加辅助分支来强化了定位能力。辅助分支主要完成的工作就是对RepPoint提取采样特征点能力的提高，采用的方式是通过生成的前背景和角点热力图来达到参考索引的效果。



