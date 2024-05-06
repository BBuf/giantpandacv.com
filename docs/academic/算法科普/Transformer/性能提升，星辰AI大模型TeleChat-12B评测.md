
# 0x0. 前言
2月份的时候评测过TeleChat-7B大模型，见[星辰AI大模型TeleChat-7B评测](https://mp.weixin.qq.com/s/ZQusIc6Ho3OO-GNQfREmjg)。最近中电信 AI 科技有限公司针对TeleChat-7B进行了性能升级，并开源了一个更大的模型TeleChat-12B，受邀对这个大模型进行新的评测。本文主要关注TeleChat-7B在做一些文学创作和代码生成方面相比于TeleChat-7B的提升。TeleChat-7B不仅在模型结构上有所微调，而且相比于TeleChat-7B的1.5T Tokens，TeleChat-12B使用了3T Tokens进行预训练，取得了更好的性能结果。下面红框部分是TeleChat-12B相比于TeleChat-7B在通用能力，推理和代码能力，语言理解能力等维度的数据集上的性能提升：


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/3c4a80c2e7fa4309b3c25f04a76138aa.png)

# 0x1. TeleChat-12B相比于TeleChat-7B的差异点
TeleChat-12B和TeleChat-7B均开源在https://github.com/Tele-AI/Telechat这个仓库中，并且在Huggingface,ModelScope等大模型托管平台进行托管，另外还开源了int8和int4两种低比特类型的模型方便部署。这里着重说明一下TeleChat-12B和TeleChat-7B的差异之处：
- 数据方面，TeleChat-12B使用了3T tokens进行预训练，而TeleChat-7B则只用了1.5T tokens。
- 在模型结构方面，相比TeleChat-7B模型，TeleChat-12B模型采用了词嵌入层与输出层解耦的结构，将词嵌入层和输出lm head层参数分开，有助于增强训练稳定性和收敛性。
- 在训练方法方面，TeleChat-12B训练时使用更科学的数据配比学习与课程学习的方法，使用小参数模型在多种数据配比的数据上拟合，得到对各个数据集难度的先验估计；训练过程中每隔一段时间自动化评估当前模型在所有数据集上的loss，以及在评测集上的生成效果，动态提升较难学习的数据集权重，保证模型在各个数据集上都有较佳的拟合效果。
- 后续通过对比TeleChat-7B和TeleChat-12B在文创和代码方面的一些例子可以发现TeleChat-12B在指令跟随，幻觉，补全文本的指令以及代码创作上都有较大提升。
# 0x2. 环境配置
可以使用官方提供的Docker镜像，也可以自己按照 https://github.com/Tele-AI/Telechat/blob/master/requirements.txt 来配置。我这里是直接使用了官方的镜像，基本没踩什么坑，按照 https://github.com/Tele-AI/Telechat/blob/master/docs/tutorial.md 这个教程操作就可以。
# 0x3. 文学创作能力测试
为了更加真实的观察模型的文学创作能力，这里不使用TeleChat官方开源仓库提供的例子，而是使用我们自己的一些prompt来进行测试。其中部分例子取自：https://github.com/SkyworkAI/Skywork#chat%E6%A8%A1%E5%9E%8B%E6%A0%B7%E4%BE%8B%E5%B1%95%E7%A4%BA 。然后来关注TeleChat-7B和TeleChat-13B的输出结果，测试代码为：

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
question="通过对“红楼梦中的人，都散在眼前”的理解，尝试创作一首描绘梦境与现实的五言律诗。"
print('==============Prompt===================')
print(question)
print('==============TeleChat-7B==============')
tokenizer = AutoTokenizer.from_pretrained('/bbuf/telechat-7B/', trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained('/bbuf/telechat-7B/', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained('/bbuf/telechat-7B/')

answer, history = model.chat(tokenizer = tokenizer, question=question, history=[], generation_config=generate_config, stream=False)
print(answer)

print('==============TeleChat-12B==============')

tokenizer = AutoTokenizer.from_pretrained('/mnt/data/cangshui/bbuf/TeleChat-12B/', trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained('/mnt/data/cangshui/bbuf/TeleChat-12B/', trust_remote_code=True, device_map="auto", torch_dtype=torch.float16)
generate_config = GenerationConfig.from_pretrained('/mnt/data/cangshui/bbuf/TeleChat-12B/')
answer, history = model.chat(tokenizer = tokenizer, question=question, history=[], generation_config=generate_config, stream=False)
print(answer)

```

另外我还做了一个改动，把两个模型文件夹下的generation_config.json里面的do_sample都改成了`true`，让结果更加丰富。

针对相同的输入prompt分别使用7b和12b的模型进行推理，对比输出结果。

- 诗词创作


![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/109c3ee6a55d4464a36ce02df731cc4e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8fab7bc687474ffa8299757f3e2faf2a.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/7bd37fa4fa834bf19e09d39fe87a5019.png)

测试一下写诗，发现TeleChat-7B和TeleChat-12B模型在诗词创作方面的能力都比较有限，虽然可以生成一些和prompt描述相关的文字，但是对五言，七言等诗歌形式往往不能正常理解。可能和数据里面没有特意微调这种情况有关系。

- 广告文案

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/79d1e83e17fe4b978df82c22361930ab.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/26823c5da165463eaeb1f0c8820a1299.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/b9d580deb69e4b4793e9e5ce0352850b.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/fa938b4dd50a41b7970fcd9aefe63aac.png)

从这几个例子可以看到TeleChat-12B的指令跟随能力比7B要更好一些，并且输出的内容质量也更高。

- 作文生成
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/3b7619e7e8594c78a7fbff39beecd4f9.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/fc2be0cdd6524cc18c050b3985f3b503.png)（上面2张图是一个prompt）

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/8af0ebb57a484a158d542a6f7b182f77.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4e2b3d685313481186e45ccf1836321b.png)

上面写了2篇作文，看起来12B的模型也是比7B的表现更好，重要的是对于字数的判断12B模型更加准确，而7B模型似乎忽略了prompt里面字数限制的指令。

- 演讲稿生成
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9f5dbb5144a4428eb00eb5a5a9434394.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/94019a80d6994eeb80678f6687dcab61.png)

上面两张图是同一个prompt。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/fd47e551a0e14ea897554fe851279017.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c7ffc69c942c4f02a6fdf98c1d39244b.png)

TeleChat-12B的创作质量相比于TeleChat-7B明显更高。

- 心得体会

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4a2c6b9e287945c48c59c0a73255a388.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f2fdd572ba924d27b9b21ea17e3ee8d6.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/5fd33fcf38464613b314d765f57ad60e.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/4b46543839584febaf6d7537c9ba7518.png)

上面三张图是一个prompt的输出。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/f7a7309f300341f58164b46e6e0f16e0.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0ad0964386c544dcbf83958b3a45f0ba.png)

上面两张图是一个prompt的输出。

同样，TeleChat-12B的创作质量相比于TeleChat-7B明显更高，并且更加丰富，对于发明专利写得更专业。

- 记录文
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/60c44a61373d4cae960d5c762f230888.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/6f8fd79078c34c71bded51afe2e14405.png)

上面两张图片是同一个prompt的输出。


总的来说，在文学创作方面，TeleChat-12B相比于TeleChat-7B无论是在质量还是指令跟随上都更好一些，可以获得更好的创作效果。

#  0x4. 代码能力对比测试
- Python中的深拷贝和浅拷贝有什么区别？ 请用代码示例解释这两种拷贝方式。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/2c5488c74a094c61a4136e5c8ca489dd.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/66241be07c9542558de67f39109fc204.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/52d7e7aa90954543ad7311ca4450e3c4.png)

对于深浅拷贝，TeleChat-7B的答案完全错误，而TeleChat-12B的答案则更靠谱，指出了浅拷贝是副本共享内部对象的引用，而深拷贝的副本不共享内部对象的引用，但在对比的时候代码注释仍然有个小错误。

- 如何合并两个已排序的数组？ 编写一个函数，将两个已排序的整数数组合并为一个排序后的数组。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/cf2f1c3c8f0e4733b89c37a789512ea9.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/e8c5b36ef52a43e5a443ff53e9516953.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/40edb679a8c141e5aecc80579b5e6d46.png)

- 判断一个数是否为质数。 编写一个函数，判断给定的整数是否是质数，并考虑优化算法的效率。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/9e22d90f7bbf4e6faac5f73dd9edd63d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/da2d01269af7496a8e9128bfd17dee6c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/77c9e671122149a18778b8f9b418458e.png)

相比于TeleChat-7B的答案。TeleChat-12B提供了两种方法并指明了第二种方法的时间复杂度更小，回答的质量更高。

- c++编写一个函数，将字符串中所有指定的单词替换为另一个单词。

![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/0d664dc8fc3f4abba93766d77c11946c.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/c2b707aa5590417b89cd87b368016238.png)

对于这个问题，TeleChat-12B可以正确理解并给出解决方案，而TeleChat-7B的答案是错误的。


通过上述一些编程问题，可以发现TeleChat-12B在编程能力上相比于TeleChat-7B有较大提升，这种提升的原因大概率是新增的1.5T数据包含了部分代码数据。

# 0x5. 总结

本文通过对比TeleChat-7B和TeleChat-12B在文创和代码方面的一些例子可以发现TeleChat-12B在指令跟随，幻觉，补全文本的指令以及代码创作上都有较大提升。希望后续能持续在网络结构，数据方面做出改进，做出更强的开源模型。




