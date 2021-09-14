#### 让vscode在远程连接服务器时候如本地般顺滑地debug

【GaintPandaCV导读】本文主要分享了python语言的使用vscode在远程连接服务器的debug，可以通过launch.json来传入python脚本的参数，这样就能够在该情况下用vscode调试，操作跟vscode在本地调试一样

##### 一、vscode 远程连接服务器

1、在vscode应用插件那里下载Remote SSH

![Remote SSH](https://img-blog.csdnimg.cn/20210629155804964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

2、连接远程服务器

![连接远程服务器](https://img-blog.csdnimg.cn/20210629160211812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

点击SSH TARGETS上面的加号，出现下面的图片，输入ssh username@IP地址，输入密码即可。

![SSH TARGETS](https://img-blog.csdnimg.cn/20210629155908398.png)

3、免密码登录：

在终端输入 ssh-copy-id username@IP地址，输入密码即可。

##### 二、使用vscode在远程服务器上debug

1、命令行的方式： ipdb

首先需要安装 ipdb：pip install ipdb

在终端上输入 python -m ipdb xxx.py就可以一行一行的调试了

或者，在xxx.py文件中在需要中断的地方插入上如下代码

“from ipdb import set_trace

set_trace()”

xxx.py程序跑的时候就会在你设置断点的位置停下来。

但是并不建议使用在源代码中插入代码来达到断点的作用，因为这样破坏了程序源代码的完整性。

纯命令行调试的一些常用指令：

- h(help)：帮助命令
- s(step into)：进入函数内部
- n(next)：执行下一行
- b(break): b line_number 打断点
- cl(clear): 清除断点
- c(continue): 一直执行到断点
- r(return): 从当前函数返回
- j(jump): j line_number，跳过代码片段，直接执行指定行号所在的代码
- l(list): 列出上下文代码
- a(argument): 列出传入函数所有的参数值
- p/pp: print 和 pretty print 打印出变量值
- r(restart): 重启调试器
- q(quit): 推出调试，清除所有信息

2、**直接点击vscode的run进行调试**：

重点来了，就是使用vscode进行调试，让我们在远程连接服务器的使用感与在本地上一样。没办法，pycharm据说连接远程服务器要收费啊，只能用vscode来做这个事情了。

首先在你项目的文件夹下，创建一个.vscode文件夹，其实也是也可以按按按键来生成的，在ubuntu下，mkdir不是更加便捷嘛hhhh～～。

然后，在.vscode文件夹下面创建3个json文件，launch.json、setting.json、task.json。

a).编写launch.json

```json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "python": "/home/ml/anaconda3/envs/py36/bin/python", #这个是虚拟环境 conda info --envs 可以看虚拟环境的地址
            "console": "integratedTerminal",
            "args": [
                "--lr",               "0.4",
                "--iter",             "4" ,
                "--epoch",            "30",
                "--model",            "CNN",
              ],
        }
    ]
}
```

补充一个如何创建虚拟环境和查看虚拟环境

```
创建虚拟环境: conda create -n name python=3.6
查看虚拟环境: conda info --envs
激活虚拟环境: conda activate
退出虚拟环境: conda deactivate
```



b).编写setting.json

```json
{
  "python.pythonPath": "/home/ml/anaconda3/envs/py36/bin/python" #这个是虚拟环境 conda info --envs 可以看虚拟环境的地址
}

```



c).编写task.json

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "python",
            "type": "shell",
            "command": "/home/ml/anaconda3/envs/py36/bin/python",  #这个是虚拟环境 conda info --envs 可以看虚拟环境的地址
            "args": [
                "${file}"
            ],
            "group": {
                "kind": "build",
                "isDefault": true
            },
            "problemMatcher": [
                "$eslint-compact"
            ]
        }
    ]
}
```



3、给调试传参数

这个主要是在launch.json里面，
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [

    {
           ....
            "args": [
                "--lr",               "0.4",
                "--iter",             "4" ,
                "--epoch",            "30",
                "--model",            "CNN",
              ],
             ....
        }
    ]
    }
```

在args里面 传入你自己设定的参数

最后点击 Run and Debug

![点击运行和DEBUG](https://img-blog.csdnimg.cn/20210629161631328.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDcyODk1OA==,size_16,color_FFFFFF,t_70)

接下来就是选择python解释器，如果没有就直接点击install即可。

这样就完成了，可以愉快地debug了。



-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)