**GiantPandaCV** 一直是以公众号的形式和大家见面，但现在随着分享干货的日益增多，我们不满足于公众号小小的窗口，为了方便大家快速检索自己感兴趣的知识，也为了增加和大家的交流， **GiantPandaCV.com** 应运而生。以下为网站搭建过程，有小伙伴需要搭建自己个人博客网站，项目文档网站，可以作为参考。

同时我们的网站所有内容也在 GitHub 开源，欢迎读者访问学习或者star。

地址： [https://github.com/BBuf/giantpandacv.com](https://github.com/BBuf/giantpandacv.com) 

# MkDocs 简介

公众号日常文章都以 Markdown 形式交付BB，通过一定方式转化为公众号可读形式给大家阅读。基于此，我们选择了MkDocs 作为网站首选。

 MkDocs 是一个用于创建项目文档的静态站点生成器，基于 Markdown 和 YAML 配置文件生成HTML文档。无论配置，使用，调试都非常方便。

# 安装

MkDocs 基于Python，官网支持 Python 2.6，2.7，3.3，3.4，实测3.6，3.7也可正常使用。需要使用 pip 安装，mkdocs 目前最新版本为 1.2.1 ，现在还未完全适配。我们使用的1.1.2版本。

```shell
$ pip install mkdocs==1.1.2
```



# MkDocs 基本使用

### 创建项目

```shell
$ mkdocs new giantpandacv
$ cd giantpandacv
```
![创建项目](https://img-blog.csdnimg.cn/20210711172702765.png)
项目创建完成后可以看到一个 `mkdocs.yml` 配置文件和 `docs` 文件夹，`docs`下包含一个 `index.md` 文件，为网站默认首页。

网站整体结构依据 `mkdocs.yml` 配置，所有 Markdown 文件分类后放到 `docs` 目录下。

### 调试
使用 `mkdocs serve`，启用内置服务器，控制台根据 `mkdocs.yaml` 生成临时本地网站预览。

```shell
[root@giantpandacv giantpandacv]# mkdocs serve
INFO    -  Building documentation... 
INFO    -  Cleaning site directory 
INFO    -  Documentation built in 0.08 seconds 
[I 210711 17:31:16 server:335] Serving on http://127.0.0.1:8000
INFO    -  Serving on http://127.0.0.1:8000
[I 210711 17:31:16 handlers:62] Start watching changes
INFO    -  Start watching changes
[I 210711 17:31:16 handlers:64] Start detecting changes
INFO    -  Start detecting changes
```

使用浏览器打开 http://127.0.0.1:8000/ ，可以看到如下页面
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711174536544.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
每次对 `mkdocs.yaml` 修改或者 `docs` 下的文件进行修改，会自动进行编译，浏览器刷新即可查看最新改动，十分方便。

### mkdocs.yml 基础属性
这里使用 `giantpandacv.com` 网站的 `mkdocs.yml` 配置文件进行讲解

- 首部主要是网站基本信息

```yaml
site_name: GiantPandaCV
site_description: GiantPandaCV
site_author: GiantPandaCV
site_url: http://giantpandacv.com
```

- `copyright`信息，如网站需要在公网上发布，记得备案

```yaml
copyright: 'Copyright &copy; GiantPandaCV Team <a href="http://beian.miit.gov.cn">蜀ICP备2020031210号</a> '
```
- 网站结构
`nav` 块为网站核心模块，`首页` 为网站大的模块划分，往下依次为一级目录，二级目录，文章。可以有多级目录，但从美观度来看，二级目录就可以了。再多会显得文字拥挤。
```yaml
nav:
    - 首页:
        - Getting Started: index.md
        - GiantPandaCV 简介: introduction/introduction.md
    - 传统图像: 
        - 专栏介绍: 传统图像/README.md
        - 一些有趣的图像算法: 
            - OpenCV图像处理专栏一  盘点常见颜色空间互转: 传统图像/一些有趣的图像算法/OpenCV图像处理专栏一  盘点常见颜色空间互转.md
            - OpenCV图像处理专栏二 《Local Color Correction 》论文阅读及C++复现: 传统图像/一些有趣的图像算法/OpenCV图像处理专栏二 《Local Color Correction 》论文阅读及C++复现.md
            - OpenCV图像处理专栏三  灰度世界算法原理和实现: 传统图像/一些有趣的图像算法/OpenCV图像处理专栏三  灰度世界算法原理和实现.md
            - OpenCV图像处理专栏四  自动白平衡之完美反射算法原理及C++实现:  传统图像/一些有趣的图像算法/OpenCV图像处理专栏四  自动白平衡之完美反射算法原理及C++实现.md
            - OpenCV图像处理专栏五  《Real-time adaptive contrast enhancement for imaging sensors》论文解读及实现: 传统图像/一些有趣的图像算法/OpenCV图像处理专栏五  《Real-time adaptive contrast enhancement for imaging sensors》论文解读及实现.md
            - OpenCV图像处理专栏六  来自何凯明博士的暗通道去雾算法(CVPR 2009最佳论文): 传统图像/一些有趣的图像算法/OpenCV图像处理专栏六  来自何凯明博士的暗通道去雾算法(CVPR 2009最佳论文).md
            ······
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711180443451.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
### 配置主题
官方有多种主题和参数进行配置，具体可参照 [mkdocs主题样式配置](https://mkdocs.zimoapps.com/user-guide/styling-your-docs/)
```yaml
theme:
  name: 'material'
  language: 'zh'
  palette:
    primary: 'white'
    accent: 'red'
    scheme: preference
    
  icon:
    logo: 'material/school'
  features:
    - navigation.tabs
  font:
    text: 'Noto Sans'
    code: 'Source Code Pro'
```

# 站点生成
文档调试完毕后，使用以下命令生成 HTML 文档

```shell
$	mkdocs build
```
第一次执行命令后，会在目录下生成 `site` 目录，所有内容均生成 `site` 目录中。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711181556654.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
将 `site` 目录中所有文件拷贝到你的 `Apache` 或者 `Nginx` 服务器网站目录中即可正常访问网站内容。
注：不需要 mysql，php 等环境支持。

更多介绍和使用可以参见  [MkDocs中文文档 (zimoapps.com)](https://mkdocs.zimoapps.com/) 

# MkDocs 各种转换 bug 踩坑

- 文字下方图片未正常换行

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182340861.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
修改方法：在图片下方加换行

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182402862.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182433169.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
- 数字标题序号未按正常序列进行排列
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182451772.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

修正方法一：直接**加粗**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182512131.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

修改方法二：修改为4级标题（这样改会修改正文右边的目录结构，所以我一般在上面个方法不生效的时候用）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182555690.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

- 公式未正常解析

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182613647.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

修改方法：源码模式给每个公式前后进行换行
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182629731.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182647896.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

- 无序标题未正常解析

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182729656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
修改方法：在无序标题前进行换行（只需要在第一行进行换行就行了，如果无需标题间包含较多复杂公式和代码，可能需要在多个无序标题之间进行换行）

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711182800834.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711183021416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

- 超链接显示与实际不一致

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711183051845.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

修改方法：在超链接前后加空格

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711183111933.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0ZUUU9PTw==,size_16,color_FFFFFF,t_70)

-  [Errno 2] No such file or directory: 'C:\\Users\\TIANQI~1\\AppData\\Local\\Temp\\mkdocs_xrkyz_r5\\........

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210711183137180.png)

修改方法：

检查文件名前后是否含有多余空格
![](https://img-blog.csdnimg.cn/20210711183215485.png)