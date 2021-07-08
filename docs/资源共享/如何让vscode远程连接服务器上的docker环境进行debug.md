> 【GiantPandaCV导语】本文在[让vscode在远程连接服务器时候如本地般顺滑地debug(Python)](https://mp.weixin.qq.com/s/kHhL-K-El8PimVQn3YndBw) 的基础上分享了另外一种可以直接通过vscode在docker环境中进行debug的方法。


# 如何让vscode远程连接服务器上的docker环境进行debug

一般深度学习算法的训练和调试环境都在服务器端，想不做配置就直接使用vscode进行debug不太可能。而使用远程服务器时，一般用docker进行环境部署的情况比较多。

**使用vscode远程连接服务器debug**和**远程服务器上的docker容器进行debug**，两者关键区别在于后者在docker容器创建时需要注意端口映射问题。本文主要讲解**vscode远程连接服务器上的docker环境进行debug**的具体步骤。

> 注意：如果是**使用vscode远程连接服务器debug**，则无需执行步骤一，直接从步骤二中的2开始即可。

### 一、服务器端的docker容器创建时需要注意的问题

创建容器时，一般按照如下命令创建。其中，端口映射参数：**-p** **宿主机port:容器port**。

> **sudo docker run --gpus all  -it  -d -p 8010:22 --name 容器名称   -v 本地路径或服务器物理路径：容器内路径  -d 镜像id   /bin/bash**

*OPTIONS说明：*

> - -d: 后台运行容器，并返回容器ID；
> - -i: 以交互模式运行容器，通常与 -t 同时使用；
> - -P: 随机端口映射，容器内部端口随机映射到主机的端口；
> - -p: 指定端口映射，格式为：主机(宿主)端口:容器端口 ；
> - -t: 为容器重新分配一个伪输入终端，通常与 -i 同时使用；
> - --name="nginx-lb": 为容器指定一个名称；
> - --volume , -v:	 绑定一个卷。映射关系：本地路径或服务器物理路径：容器内路径；

上面的命令中**-p 8010:22**，就是将容器的22号端口（ssh服务端口）映射到宿主机（服务器）的8010端口。在本文中，因为需要使用ssh服务端口，所以，容器端口必须写22。（宿主机端口可以写成其他值，但是也不能乱写，防止端口冲突）。这样，在后续的vscode配置中，需要将连接端口写成宿主机（服务器端口），例如本文中的8010端口。下文中会介绍如何配置连接端口。

注意：在整个配置过程中，应该保持创建的docker容器处于运行状态，方便后续调试。

### 二、docker容器内部相关配置

本文介绍的方法需要使用ssh服务进行通信，因此，需要在环境中安装ssh。

#### 1、进入容器中，使用如下命令修改root用户密码：

> **passwd**

#### 2、检查容器内部是否安装 openssh-server与openssh-client，若没安装，执行如下命令：

> **apt-get install openssh-server**
> **apt-get install openssh-client**

#### 3、修改ssh配置文件以下选项:

> **vim /etc/ssh/sshd_config**

在末尾增加如下内容(直接复制即可)：

> *#PermitRootLogin prohibit-password # 默认打开 禁止root用户使用密码登陆，需要将其注释*

> *RSAAuthentication yes           #启用 RSA 认证*
> *PubkeyAuthentication yes    #启用公钥私钥配对认证方式*
> *PermitRootLogin yes             #允许root用户使用ssh登录*

#### 4、启动sshd服务

> **/etc/init.d/ssh restart**

####  5、退出容器，连接测试

> **ssh root@127.0.0.1 -p 8010**     注意，此处应该是测试8010端口。

输入密码成功进入容器内部即配置成功。

### 三、在vscode上的相关配置

#### 1、安装**remote-ssh**插件

> 在vscode最左侧应用“**扩展**”中搜索**remote-ssh**插件，然后安装。安装完成之后，会在“**扩展**”图标下方出现“**远程资源管理器**”图标。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221043321.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


#### 2、配置vscode的**config**文件

> 单击“**远程资源管理器**”图标，然后单击“**配置**”按钮进行配置，此时vscode会显示“Select SSH configuration file to update”，如下图所示，选择路径中带有“.ssh”的config文件。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221056888.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


填写config文件内容，注意按照如下格式填写：

> **Host**可以根据自己的喜好起一个标志名称。**HostName**必须填写需要远程连接的服务器IP地址。**User**此处因为远程的是服务器上配置的docker容器，默认用户名是**root**，此处需要改下为root。

**特别注意**：由于需要远程连接的是服务器上的docker容器，而且前面提到：ssh服务器的22号端口已经映射为8010,因此，务必增加一个**Port**，填写自己映射的端口。如果只是远程服务器，不需要用docker容器，则，不需要增加Port这一行。配置完成后，保存配置。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221116536.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


#### 3、开启远程连接

> 如下图所示，config文件中写的Host名称alias就会显示在最左侧。此时，单击“新建连接”按钮，vscode会重新打开一个窗口，提示输入远程服务器的密码，注意，此时必须填入docker容器中创建的用户密码。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221128605.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


> 在如下图中输入用户密码，回车即可。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221138653.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


> 回车之后，可能会提示选择远程服务器的平台是哪一种系统类型，选项有linux\windows\MAC。应该选择vscode安装的系统平台类型。
>
> 选择完成之后，回车即可。此时，在vscode的“终端”窗口可以看到进入docker容器的命令行格式。如下图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210708221152337.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


> 在“终端”窗口可以查看以下远程连接的环境是否正确。
>
> 打开远程服务器上的代码，可以在代码任意行最左侧打断点，按F5快捷键可以debug运行。



# 四，参考文献

- https://blog.csdn.net/hanchaobiao/article/details/84069299