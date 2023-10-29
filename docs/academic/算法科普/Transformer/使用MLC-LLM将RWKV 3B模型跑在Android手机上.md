# 0x0. 前言
这篇文章主要是填一下 [MLC-LLM 部署RWKV World系列模型实战（3B模型Mac M2解码可达26tokens/s）](https://mp.weixin.qq.com/s/4a9yC4LedJRUKl8UtV9qtQ) 这里留下来的坑，这篇文章里面介绍了如何使用 MLC-LLM 在A100/Mac M2上部署 RWKV 模型。但是探索在Android端部署一个RWKV对话模型的app时却碰到了诸多问题，解决的周期也很长，之前留了issue在MLC-LLM的repo，这周@chentianqi大佬回复说之前编译出的app会在模型初始化阶段卡住的问题已经解决了，所以我又重新开始踩了一些坑最终完成了在手机上运行RWKV World4 3B模型的目的。这里把踩的坑和Android编译方法都描述一下。

我这里编译了一个RWKV4 World 3B模型的权重int4量化版本的apk，地址为：https://github.com/BBuf/run-rwkv-world-4-in-mlc-llm/releases/download/v1.0.0/app-debug.apk 。感兴趣的小伙伴可以下载这个apk到android手机上来运行，需要注意的是由于要在线拉取HuggingFace的权重，所以手机上需要可以访问HuggingFace需要代理。

在我的Redmik50手机上进行测试，效果和速度如下：![在这里插入图片描述](https://img-blog.csdnimg.cn/a1dbc73470574089ab19ca994c0d8345.png)

每一秒大概可以解码8个token，我感觉速度勉强够用了。由于RWKV5迭代到了第5个版本，后续希望能支持RWKV5的模型，当然也可以寻求新的优化机会提升解码速度。
# 0x1. 踩坑
之前写这篇文章 [MLC-LLM 部署RWKV World系列模型实战（3B模型Mac M2解码可达26tokens/s）](https://mp.weixin.qq.com/s/4a9yC4LedJRUKl8UtV9qtQ)  的时候发现android app在初始化的时候一直会卡住，即使换成官方编译的app也是如此，所以提了issue之后就放弃了。现在这个bug被修复了，不过我没有找到具体的改动pr是什么，但我在mlc-llm的android部分没有发现相关改动，所以大概率是relax本身的bug，就不深究了。

这次仍然是按照之前的方法进行编译，但是也踩了几个坑，具体体现在下方的改动：

![在这里插入图片描述](https://img-blog.csdnimg.cn/55917082d66a4212bd7cc1ac394d9534.png)这个改动只是为了在本地可以编译出RWKV的android app，有坑的地方体现在下面的2个改动：

![在这里插入图片描述](https://img-blog.csdnimg.cn/88959131aafa4efa8422c9a64bd9d20c.png)
第一个坑是在dump_mlc_chat_config的时候，对于RWKV World模型应该使用工程下面的tokenzier_model文件作为tokenzie的文件，但是之前没考虑这个问题（dump出的config中tokenizer_files字段为空）就会导致编译出的app在初始化阶段报错：

![在这里插入图片描述](https://img-blog.csdnimg.cn/98dad8bf4d3347fb838d0a31e85060a4.png)
经过上面的修改之后重新在mlc-llm下面`pip install .`，然后编译模型就可以得到可以正常初始化的config了。这个问题是通过在Android Studio里面通过Device Explore查看下载的文件夹发现的，我发现少了一个tokenizer_model文件才注意的。

第二个坑是初始化完成之后聊天的时候不出字，我在mac上去复现了这个错误，然后发现是因为在RWKV里面把max_window_size这个属性设置成了1。

![在这里插入图片描述](https://img-blog.csdnimg.cn/9e0c40818d8b431ca523cecc87799bac.png)然后在mlc_chat.cc里面通过max_window_size判断结束符时没有考虑-1，所以第一个token生成之后程序就终止了。所以在这里加一个特判进行了修复。![在这里插入图片描述](https://img-blog.csdnimg.cn/3b636694c7ad4945aeebb5cb4a8e9cf1.png)

解决了上面2个问题，编译出新的apk之后就可以正常运行了。

# 0x2. 详细教程
下面是编译出apk的详细教程。在这之前请阅读：[MLC-LLM 部署RWKV World系列模型实战（3B模型Mac M2解码可达26tokens/s）](https://mp.weixin.qq.com/s/4a9yC4LedJRUKl8UtV9qtQ)  ，这是前置内容。

对于Android，你可以按照[https://mlc.ai/mlc-llm/docs/deploy/android.html](https://mlc.ai/mlc-llm/docs/deploy/android.html)的教程在你的手机上编译apk。

根据官方教程，这里有一些需要修改的地方：
1. 修改[这个文件](https://github.com/mlc-ai/mlc-llm/blob/main/android/MLCChat/app/src/main/assets/app-config.json)。更新的内容应该是：
```json
{
  "model_libs": [
    "RWKV-4-World-3B-q4f16_1"
  ],
  "model_list": [
    {
      "model_url": "https://huggingface.co/BBuf/RWKV-4-World-3B-q4f16_1/",
      "local_id": "RWKV-4-World-3B-q4f16_1"
    }
  ],
  "add_model_samples": []
}
```
2. 将[代码的这个部分](https://github.com/mlc-ai/mlc-llm/blob/main/android/MLCChat/app/build.gradle#L35-L41)修改为：
```java
compileOptions {
    sourceCompatibility JavaVersion.VERSION_17
    targetCompatibility JavaVersion.VERSION_17
}
kotlinOptions {
    jvmTarget = '17'
}
```
3. 如果你遇到错误：“Android Gradle插件要求运行Java 17。你目前使用的是Java 11”，请按照https://stackoverflow.com/questions/76362800/android-gradle-plugin-requires-java-17-to-run-you-are-currently-using-java-11 的方法清除缓存并重新编译。

一旦你完成了APK的编译，你可以在你的手机上启用开发者模式并安装APK以供使用。

以小米手机为例，你可以按照下面的教程启用开发者模式并将APK传输到你的手机上。

- **第一步：在手机上启用USB调试**
  - 首先，前往你的手机的"设置 -> 我的设备 -> 所有规格 -> MIUI版本"，连续点击"MIUI版本"七次以进入开发者模式。
  - 接下来，导航至"设置 -> 额外设置 -> 开发者选项"，打开"USB调试"和"USB安装"。

- **第二步：配置Android Studio**
  - 打开你的Android Studio项目，前往"运行 -> 编辑配置"，如下图所示，选择"打开选择部署目标对话框"。这将在每次你调试时提示设备选择对话框。注意：如果你直接选择"USB设备"，你可能无法在调试过程中检测到你的手机。

- **第三步：在线调试**
  - 通过USB将你的手机连接到电脑。通常会自动安装必要的驱动程序。当你运行程序时，将出现设备选择对话框。选择你的手机，APK将自动安装并运行。

一个编译好的apk: [https://github.com/BBuf/run-rwkv-world-4-in-mlc-llm/releases/download/v1.0.0/app-debug.apk](https://github.com/BBuf/run-rwkv-world-4-in-mlc-llm/releases/download/v1.0.0/app-debug.apk)


# 0x3. 总结
这篇文章分享了一下使用MLC-LLM将RWKV模型跑在Android手机上遭遇的坑以及编译的详细教程，接下来也会尝试一下RWKV5。想在andorid手机上本地运行开源大模型的伙伴们可以考虑一下MLC-LLM，他们的社区还是比较活跃的，如果你提出一些问题一般都会有快速的回复或者解决方法。

# 0x4. 相关link
- [https://github.com/mlc-ai/tokenizers-cpp/pull/14](https://github.com/mlc-ai/tokenizers-cpp/pull/14)
- https://github.com/mlc-ai/mlc-llm/pull/1136
- [https://github.com/mlc-ai/mlc-llm/pull/848](https://github.com/mlc-ai/mlc-llm/pull/848)
- [https://mlc.ai/mlc-llm/docs/](https://mlc.ai/mlc-llm/docs/)
- [StarRing2022/RWKV-4-World-1.5B](https://huggingface.co/StarRing2022/RWKV-4-World-1.5B)
- [StarRing2022/RWKV-4-World-3B](https://huggingface.co/StarRing2022/RWKV-4-World-3B)
- [StarRing2022/RWKV-4-World-7B](https://huggingface.co/StarRing2022/RWKV-4-World-7B)
- [https://github.com/mlc-ai/mlc-llm/issues/862](https://github.com/mlc-ai/mlc-llm/issues/862)
- [https://github.com/mlc-ai/mlc-llm/issues/859](https://github.com/mlc-ai/mlc-llm/issues/859)



