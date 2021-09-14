# 前言
主要步骤在这个博客有解释:https://blog.csdn.net/ourkix/article/details/84283775 。但是这个博客是讲如何配置windows下神经棒一代的python接口，对于我们这种要用c++部署的人来说显然是不友好的。所幸，我跑通了神经棒1代的c++接口，接下来就分享一下这个过程。
# 步骤
1. 下载https://github.com/LukaszGajowski/ncsdk
2. 假设大家都有vs2015，不然点进来干啥。
3. 下载zading2.3 https://zadig.akeo.ie/downloads/zadig-2.3.exe
4. 下载完ncsdk后，进入ncsdk文件夹里面，在进入api文件夹，再进入winsrc文件夹。用vs2015打开这个过程。
5. 然后选择Release-X64
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190910104159759.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
- 6. 配置项目属性
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190910104224949.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)![在这里插入图片描述](https://img-blog.csdnimg.cn/20190910104242968.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)include文件夹的内容为: .\libusb\include\libusb-1.0;.\pthread\include;

- 链接器设置:
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190910104346804.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 编写代码
接下来就可以编写代码进行神经网络的推理工作了。可以参考这个SequeezeNet的代码，我传到我博客的资源里面了，有C币的可以在上面下载，支持一下，没有C币可以私信我发给你。

# 提醒
一定要在我们克隆的工程下编写代码，实际上就是需要编译出来的那几个dll，不然可能会出现打开NCS设备失败的问题。