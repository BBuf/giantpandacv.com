>Xception: Deep Learning with Depthwise Separable Convolutions 
# 1.前言
卷积神经网络学习路线这个系列旨在盘点从古至今对当前CV影响很大的那些经典网络。为了保证完整性我会将之前漏掉的一些网络补充下去，已经介绍了非常多的经典网络，这个系列可能也快要迎来完结了。接着[卷积神经网络学习路线（九）| 经典网络回顾之GoogLeNet系列](https://mp.weixin.qq.com/s/mXhVMHBsxrQQf_MV4_7iaw) 也就是Inception V3之后，Google提出了XceptionNet，这是对Inception V3的一种改进，主要使用了深度可分离卷积来替换掉Inception V3中的卷积操作。

# 2. 铺垫
为了更好的说明Xception网络，我们首先需要从Inception V3来回顾一下。下面的Figure1展示了Inception V3的结构图。可以看到Inception的核心思想是将多种特征提取方式如$1\times 1$卷积，$3\times 3$卷积，$5\times 5$卷积，$pooling$等产生的特征图进行了concate，达到融合多种特征的效果。

![InceptionV3 结构](https://img-blog.csdnimg.cn/20200331175320964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

然后，从Inception V3的结构联想到了一个简化的Inception结构，如Figure2所示。


![简化的Inception结构](https://img-blog.csdnimg.cn/20200331175834674.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

再然后将Figure2的结构进行改进，就获得了Figure3所示的结构。可以看到，在这个结构中先用一个$1\times 1$卷积，然后连接三个$3\times 3$卷积，这三个$3\times 3$卷积只将前面$1\times 1$卷积的一部分(这里是$\frac{1}{3}$的通道)作为每个$3\times 3$卷积的输出。同时Figure4则为我们展示了将这一想法应用到极致，即每个通道接一个$3\times 3$卷积的结构。

![极致的Inception](https://img-blog.csdnimg.cn/20200331210501302.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


# 3. Xception原理
Xception中主要采用了深度可分离卷积。这个卷积我们之前已经介绍的很清楚了，请看这篇推文：[【综述】神经网络中不同种类的卷积层](https://mp.weixin.qq.com/s/jXmIXP4e9l47vzYLH11Hvg) 。那么深度可分离卷积和上面Figure4中的极致Inception结构有什么区别呢？

- 极致的Inception。
	- 第一步：普通的$1\times 1$卷积。
	- 第二步：对$1\times 1$卷积结果的每个channel，分别进行$3\times 3$卷积操作，并将结果concate。
- 深度可分离卷积。
	- 第一步： **Depthwise 卷积** ，对输入的每个channel，分别进行$3\times 3$卷积操作，并将结果concate。
	- 第二步： **Pointwise 卷积** ，对 Depthwise 卷积中的concate结果，进行$1\times 1$卷积操作。

可以看到两者的操作顺序是不一致的，Inception先进行$1\times 1$卷积，再进行$3\times 3$卷积，深度可分离卷积恰好相反。作者在论文中提到这个顺序差异并不会对网络精度产生大的影响。同时作者还有一个有趣的发现，在Figure4展示的**极致的 Inception”模块**中，用于学习空间相关性的$3\times 3$卷积和用于学习通道相关性的$1\times 1$卷积**之间**如果不使用激活函数，收敛过程会更快，并且结果更好，如下图所示。

![点卷积之前是否使用激活函数实验](https://img-blog.csdnimg.cn/20200331211833477.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


# 4. Xception网络结构

Xception的网络结构如Figure5所示。

![Xception的结构](https://img-blog.csdnimg.cn/20200331210857679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

图里面的`sparsableConv`就是深度可分离卷积。另外，每个小块的连接采用的是residule connection（图中的加号），而不是原Inception中的concate。


# 5. 实验结果
Table1表示几种网络结构在ImageNet上的对比结果。

![几种网络结构在ImageNet上的对比结果](https://img-blog.csdnimg.cn/20200331212208630.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

Table2表示几种网络结构在**JFT数据集**上的对比。可以看到在大数据上的提升会比Table1好一点。


![Table2](https://img-blog.csdnimg.cn/2020033121233179.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 6. 总结
Xception主要是在Inception V3的基础上引入了深度可分离卷积，在基本不增加网络复杂度的前提下提高了模型的效果。

# 7. 代码实现
Keras代码实现如下：

```python
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'

def Xception():

	# Determine proper input shape
	input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)

	img_input = Input(shape=input_shape)

	# Block 1
	x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False)(img_input)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = Conv2D(64, (3, 3), use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	# Block 2
	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	# Block 2 Pool
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = layers.add([x, residual])

	residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	# Block 3
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	# Block 3 Pool
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = layers.add([x, residual])

	residual = Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	# Block 4
	x = Activation('relu')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = layers.add([x, residual])

	# Block 5 - 12
	for i in range(8):
		residual = x

		x = Activation('relu')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
		x = BatchNormalization()(x)

		x = layers.add([x, residual])

	residual = Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
	residual = BatchNormalization()(residual)

	# Block 13
	x = Activation('relu')(x)
	x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)

	# Block 13 Pool
	x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
	x = layers.add([x, residual])

	# Block 14
	x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Block 14 part 2
	x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Fully Connected Layer
	x = GlobalAveragePooling2D()(x)
	x = Dense(1000, activation='softmax')(x)

	inputs = img_input

	# Create model
	model = Model(inputs, x, name='xception')

	# Download and cache the Xception weights file
	weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

	# load weights
	model.load_weights(weights_path)

	return model
```

# 8. 参考
- 论文原文：https://arxiv.org/abs/1610.02357
- https://blog.csdn.net/u014380165/article/details/75142710
- https://blog.csdn.net/lk3030/article/details/84847879

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)