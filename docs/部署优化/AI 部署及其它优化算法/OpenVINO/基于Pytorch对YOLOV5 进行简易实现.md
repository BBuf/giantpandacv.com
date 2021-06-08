【GiantPandaCV导语】这篇文章主要针对于YOLOV5-Pytorch版本的网络结构代码进行实现，简化代码的理解并简化配置文件，进一步梳理一些YOLOV5四种网络结构，在这个过程中对于V5的网络有着更加深入的理解。最后希望看完这篇文章的读者可以有所收获，对于代码中的一些写法上的优化希望可以和大家一起交流进步。

## 一、网络完整代码

###  1.实现思路

v5中的common代码结构进行了保留，因为这一部分代码是比较好理解的，整体代码看起来是比较简单的，主要是整体网络结构的搭建，通过解析yaml文件对于一些开发人员来说是不是很友好的。

###  2.网络中的一些变量

   ```
   c1：输入通道 c2：输出通道  k：卷积核大小  s：步长 p：padding g：分组  act；激活函数 e：扩展倍数
   gw：网络宽度因子  gd：网络深度因子  n：模块重复次数  nc：类别数
   ```

###  3.主干网络代码`CSPDarknet53`

###  4.  1

   ```python
   import torch
   import torch.nn as nn
   
   
   def autopad(k, p=None):  # kernel, padding
       # Pad to 'same'
       if p is None:
           p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
       return p
   
   
   class CBL(nn.Module):
   
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, e=1.0):
           super(CBL, self).__init__()
           c1 = round(c1 * e)
           c2 = round(c2 * e)
           self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
           self.bn = nn.BatchNorm2d(c2)
           self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
   
       def forward(self, x):
           return self.act(self.bn(self.conv(x)))
   
   
   class Focus(nn.Module):
   
       def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True, e=1.0):
           super(Focus, self).__init__()
           c2 = round(c2 * e)
           self.conv = CBL(c1 * 4, c2, k, s, p, g, act)
   
       def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
           flatten_channel = torch.cat([x[..., 0::2, 0::2],
                                        x[..., 1::2, 0::2],
                                        x[..., 0::2, 1::2],
                                        x[..., 1::2, 1::2]], dim=1)
           return self.conv(flatten_channel)
   
   
   class SPP(nn.Module):
   
       def __init__(self, c1, c2, k=(5, 9, 13), e=1.0):
           super(SPP, self).__init__()
           c1 = round(c1 * e)
           c2 = round(c2 * e)
           c_ = c1 // 2
           self.cbl_before = CBL(c1, c_, 1, 1)
           self.max_pool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
           self.cbl_after = CBL(c_ * 4, c2, 1, 1)
   
       def forward(self, x):  
           x = self.cbl_before(x)
           x_cat = torch.cat([x] + [m(x) for m in self.max_pool], 1)
           return self.cbl_after(x_cat)
   
   
   class ResUnit_n(nn.Module):
   
       def __init__(self, c1, c2, n):
           super(ResUnit_n, self).__init__()
           self.shortcut = c1 == c2
           res_unit = nn.Sequential(
               CBL(c1, c1, k=1, s=1, p=0),
               CBL(c1, c2, k=3, s=1, p=1)
           )
           self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])
   
       def forward(self, x):
           return x + self.res_unit_n(x) if self.shortcut else self.res_unit_n(x)
   
   
   class CSP1_n(nn.Module):
   
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
           super(CSP1_n, self).__init__()
   
           c1 = round(c1 * e[1])
           c2 = round(c2 * e[1])
           n = round(n * e[0])
           c_ = c2 // 2
           self.up = nn.Sequential(
               CBL(c1, c_, k, s, autopad(k, p), g, act),
               ResUnit_n(c_, c_, n),
               # nn.Conv2d(c_, c_, 1, 1, 0, bias=False) 这里最新yolov5结构中去掉了，与网上的结构图稍微有些区别
           )
           self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
           self.tie = nn.Sequential(
               nn.BatchNorm2d(c_ * 2),
               nn.LeakyReLU(),
               nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
           )
       def forward(self, x):
           total = torch.cat([self.up(x), self.bottom(x)], dim=1)
           out = self.tie(total)
           return out
   
   
   class CSPDarkNet(nn.Module):
   
       def __init__(self, gd=0.33, gw=0.5):
           super(CSPDarkNet, self).__init__()
           self.truck_big = nn.Sequential(
               Focus(3, 64, e=gw),
               CBL(64, 128, k=3, s=2, p=1, e=gw),
               CSP1_n(128, 128, n=3, e=[gd, gw]),
               CBL(128, 256, k=3, s=2, p=1, e=gw),
               CSP1_n(256, 256, n=9, e=[gd, gw]),
   
           )
           self.truck_middle = nn.Sequential(
               CBL(256, 512, k=3, s=2, p=1, e=gw),
               CSP1_n(512, 512, n=9, e=[gd, gw]),
           )
           self.truck_small = nn.Sequential(
               CBL(512, 1024, k=3, s=2, p=1, e=gw),
               SPP(1024, 1024, e=gw)
           )
   
       def forward(self, x):
           h_big = self.truck_big(x)  # torch.Size([2, 128, 76, 76])
           h_middle = self.truck_middle(h_big)
           h_small = self.truck_small(h_middle)
           return h_big, h_middle, h_small
   
   
   def darknet53(gd, gw, pretrained, **kwargs):
       model = CSPDarkNet(gd, gw)
       if pretrained:
           if isinstance(pretrained, str):
               model.load_state_dict(torch.load(pretrained))
           else:
               raise Exception(f"darknet request a pretrained path. got[{pretrained}]")
       return model
   ```

###  5.整体网络的构建

   ```python
   import torch
   import torch.nn as nn
   from cspdarknet53v5 import darknet53
   
   
   def autopad(k, p=None):  # kernel, padding
       # Pad to 'same'
       if p is None:
           p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
       return p
   
   
   class UpSample(nn.Module):
   
       def __init__(self):
           super(UpSample, self).__init__()
           self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
   
       def forward(self, x):
           return self.up_sample(x)
   
   
   class CBL(nn.Module):
   
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, e=1.0):
           super(CBL, self).__init__()
           c1 = round(c1 * e)
           c2 = round(c2 * e)
           self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
           self.bn = nn.BatchNorm2d(c2)
           self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
   
       def forward(self, x):
           return self.act(self.bn(self.conv(x)))
   
   
   class ResUnit_n(nn.Module):
   
       def __init__(self, c1, c2, n):
           super(ResUnit_n, self).__init__()
           self.shortcut = c1 == c2
           res_unit = nn.Sequential(
               CBL(c1, c1, k=1, s=1, p=0),
               CBL(c1, c2, k=3, s=1, p=1)
           )
           self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])
   
       def forward(self, x):
           return x + self.res_unit_n(x) if self.shortcut else self.res_unit_n(x)
   
   
   class CSP1_n(nn.Module):
   
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
           super(CSP1_n, self).__init__()
   
           c1 = round(c1 * e[1])
           c2 = round(c2 * e[1])
           n = round(n * e[0])
           c_ = c2 // 2
           self.up = nn.Sequential(
               CBL(c1, c_, k, s, autopad(k, p), g, act),
               ResUnit_n(c_, c_, n),
               # nn.Conv2d(c_, c_, 1, 1, 0, bias=False) 这里最新yolov5结构中去掉了，与网上的结构图稍微有些区别
           )
           self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
           self.tie = nn.Sequential(
               nn.BatchNorm2d(c_ * 2),
               nn.LeakyReLU(),
               nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
           )
   
       def forward(self, x):
           total = torch.cat([self.up(x), self.bottom(x)], dim=1)
           out = self.tie(total)
           return out
   
   
   class CSP2_n(nn.Module):
   
       def __init__(self, c1, c2, e=0.5, n=1):
           super(CSP2_n, self).__init__()
           c_ = int(c1 * e)
           cbl_2 = nn.Sequential(
               CBL(c1, c_, 1, 1, 0),
               CBL(c_, c_, 1, 1, 0),
           )
           self.cbl_2n = nn.Sequential(*[cbl_2 for _ in range(n)])
           self.conv_up = nn.Conv2d(c_, c_, 1, 1, 0)
           self.conv_bottom = nn.Conv2d(c1, c_, 1, 1, 0)
           self.tie = nn.Sequential(
               nn.BatchNorm2d(c_ * 2),
               nn.LeakyReLU(),
               nn.Conv2d(c_ * 2, c2, 1, 1, 0)
           )
   
       def forward(self, x):
           up = self.conv_up(self.cbl_2n(x))
           total = torch.cat([up, self.conv_bottom(x)], dim=1)
           out = self.tie(total)
           return out
   
   
   class yolov5(nn.Module):
   
       def __init__(self, nc=80, gd=0.33, gw=0.5):
           super(yolov5, self).__init__()
           # ------------------------------Backbone--------------------------------
           self.backbone = darknet53(gd, gw, None)
   
           # ------------------------------Neck------------------------------------
           self.neck_small = nn.Sequential(
               CSP1_n(1024, 1024, n=3, e=[gd, gw]),
               CBL(1024, 512, 1, 1, 0, e=gw)
           )
           self.up_middle = nn.Sequential(
               UpSample()
           )
           self.out_set_middle = nn.Sequential(
               CSP1_n(1024, 512, n=3, e=[gd, gw]),
               CBL(512, 256, 1, 1, 0, e=gw),
           )
           self.up_big = nn.Sequential(
               UpSample()
           )
           self.out_set_tie_big = nn.Sequential(
               CSP1_n(512, 256, n=3, e=[gd, gw])
           )
   
           self.pan_middle = nn.Sequential(
               CBL(256, 256, 3, 2, 1, e=gw)
           )
           self.out_set_tie_middle = nn.Sequential(
               CSP1_n(512, 512, n=3, e=[gd, gw])
           )
           self.pan_small = nn.Sequential(
               CBL(512, 512, 3, 2, 1, e=gw)
           )
           self.out_set_tie_small = nn.Sequential(
               CSP1_n(1024, 1024, n=3, e=[gd, gw])
           )
           # ------------------------------Prediction--------------------------------
           # prediction
           big_ = round(256 * gw)
           middle = round(512 * gw)
           small_ = round(1024 * gw)
           self.out_big = nn.Sequential(
               nn.Conv2d(big_, 3 * (5 + nc), 1, 1, 0)
           )
           self.out_middle = nn.Sequential(
               nn.Conv2d(middle, 3 * (5 + nc), 1, 1, 0)
           )
           self.out_small = nn.Sequential(
               nn.Conv2d(small_, 3 * (5 + nc), 1, 1, 0)
           )
   
       def forward(self, x):
           h_big, h_middle, h_small = self.backbone(x)
           neck_small = self.neck_small(h_small)  
           # ----------------------------up sample 38*38-------------------------------
           up_middle = self.up_middle(neck_small)
           middle_cat = torch.cat([up_middle, h_middle], dim=1)
           out_set_middle = self.out_set_middle(middle_cat)
   
           # ----------------------------up sample 76*76-------------------------------
           up_big = self.up_big(out_set_middle)  # torch.Size([2, 128, 76, 76])
           big_cat = torch.cat([up_big, h_big], dim=1)
           out_set_tie_big = self.out_set_tie_big(big_cat)
   
           # ----------------------------PAN 36*36-------------------------------------
           neck_tie_middle = torch.cat([self.pan_middle(out_set_tie_big), out_set_middle], dim=1)
           up_middle = self.out_set_tie_middle(neck_tie_middle)
   
           # ----------------------------PAN 18*18-------------------------------------
           neck_tie_small = torch.cat([self.pan_small(up_middle), neck_small], dim=1)
           out_set_small = self.out_set_tie_small(neck_tie_small)
   
           # ----------------------------prediction-------------------------------------
           out_small = self.out_small(out_set_small)
           out_middle = self.out_middle(up_middle)
           out_big = self.out_big(out_set_tie_big)
   
           return out_small, out_middle, out_big
   
   
   if __name__ == '__main__':
       # 配置文件的写法
       config = {
           #            gd    gw
           'yolov5s': [0.33, 0.50],
           'yolov5m': [0.67, 0.75],
           'yolov5l': [1.00, 1.00],
           'yolov5x': [1.33, 1.25]
       }
       # 修改一次文件名字
       net_size = config['yolov5x']
       net = yolov5(nc=80, gd=net_size[0], gw=net_size[1])
       print(net)
       a = torch.randn(2, 3, 416, 416)
       y = net(a)
       print(y[0].shape, y[1].shape, y[2].shape)
   
   ```

   

## 二、网络结构的解析

###  1.残差块ResUnit_n

   ```python
   class ResUnit_n(nn.Module):
   
       def __init__(self, c1, c2, n):
           super(ResUnit_n, self).__init__()
           self.shortcut = c1 == c2
           res_unit = nn.Sequential(
               CBL(c1, c1, k=1, s=1, p=0),
               CBL(c1, c2, k=3, s=1, p=1)
           )
           self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])
   
       def forward(self, x):
           return x + self.res_unit_n(x) if self.shortcut else self.res_unit_n(x)
   ```

   

###  2.CSP1_x结构

   > 构建思路： CSP1_n 代码进行优化，把CSP看做一个趴着的动物，头在左面，尾巴在右边； up是靠近天空的地方，bottom是靠近地的，tie就是动物的尾巴              

   ```python
   class CSP1_n(nn.Module):
   
       def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
           super(CSP1_n, self).__init__()
   
           c1 = round(c1 * e[1])
           c2 = round(c2 * e[1])
           n = round(n * e[0])
           c_ = c2 // 2
           self.up = nn.Sequential(
               CBL(c1, c_, k, s, autopad(k, p), g, act),
               ResUnit_n(c_, c_, n),
               # nn.Conv2d(c_, c_, 1, 1, 0, bias=False) 这里最新yolov5结构中去掉了，与网上的结构图稍微有些区别
           )
           self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
           self.tie = nn.Sequential(
               nn.BatchNorm2d(c_ * 2),
               nn.LeakyReLU(),
               nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
           )
   
       def forward(self, x):
           total = torch.cat([self.up(x), self.bottom(x)], dim=1)
           out = self.tie(total)
           return out
   ```

   

###  3.CSPDarknet主干网络构建

   ```python
   class CSPDarkNet(nn.Module):
   
       def __init__(self, gd=0.33, gw=0.5):
           super(CSPDarkNet, self).__init__()
           self.truck_big = nn.Sequential(
               Focus(3, 64, e=gw),
               CBL(64, 128, k=3, s=2, p=1, e=gw),
               CSP1_n(128, 128, n=3, e=[gd, gw]),
               CBL(128, 256, k=3, s=2, p=1, e=gw),
               CSP1_n(256, 256, n=9, e=[gd, gw]),
   
           )
           self.truck_middle = nn.Sequential(
               CBL(256, 512, k=3, s=2, p=1, e=gw),
               CSP1_n(512, 512, n=9, e=[gd, gw]),
           )
           self.truck_small = nn.Sequential(
               CBL(512, 1024, k=3, s=2, p=1, e=gw),
               SPP(1024, 1024, e=gw)
           )
   
       def forward(self, x):
           h_big = self.truck_big(x)  
           h_middle = self.truck_middle(h_big)
           h_small = self.truck_small(h_middle)
           return h_big, h_middle, h_small
   ```

   

### 4.整体网络构建

   ```python
   class yolov5(nn.Module):
   
       def __init__(self, nc=80, gd=0.33, gw=0.5):
           super(yolov5, self).__init__()
           # ------------------------------Backbone------------------------------------
           self.backbone = darknet53(gd, gw, None)
   
           # ------------------------------Neck------------------------------------
           self.neck_small = nn.Sequential(
               CSP1_n(1024, 1024, n=3, e=[gd, gw]),
               CBL(1024, 512, 1, 1, 0, e=gw)
           )
           # FPN： 2次上采样 自顶而下 完成语义信息增强
           self.up_middle = nn.Sequential(
               UpSample()
           )
           self.out_set_middle = nn.Sequential(
               CSP1_n(1024, 512, n=3, e=[gd, gw]),
               CBL(512, 256, 1, 1, 0, e=gw),
           )
           self.up_big = nn.Sequential(
               UpSample()
           )
           self.out_set_tie_big = nn.Sequential(
               CSP1_n(512, 256, n=3, e=[gd, gw])
           )
   
           # PAN： 2次下采样 自底而上 完成位置信息增强
           self.pan_middle = nn.Sequential(
               CBL(256, 256, 3, 2, 1, e=gw)
           )
           self.out_set_tie_middle = nn.Sequential(
               CSP1_n(512, 512, n=3, e=[gd, gw])
           )
           self.pan_small = nn.Sequential(
               CBL(512, 512, 3, 2, 1, e=gw)
           )
           self.out_set_tie_small = nn.Sequential(
               # CSP2_n(512, 512)
               CSP1_n(1024, 1024, n=3, e=[gd, gw])
           )
           # ------------------------------Prediction------------------------------------
           # prediction
           big_ = round(256 * gw)
           middle = round(512 * gw)
           small_ = round(1024 * gw)
           self.out_big = nn.Sequential(
               nn.Conv2d(big_, 3 * (5 + nc), 1, 1, 0)
           )
           self.out_middle = nn.Sequential(
               nn.Conv2d(middle, 3 * (5 + nc), 1, 1, 0)
           )
           self.out_small = nn.Sequential(
               nn.Conv2d(small_, 3 * (5 + nc), 1, 1, 0)
           )
   
       def forward(self, x):
           h_big, h_middle, h_small = self.backbone(x)
           neck_small = self.neck_small(h_small)  
           # ----------------------------up sample 38*38--------------------------------
           up_middle = self.up_middle(neck_small)
           middle_cat = torch.cat([up_middle, h_middle], dim=1)
           out_set_middle = self.out_set_middle(middle_cat)
   
           # ----------------------------up sample 76*76--------------------------------
           up_big = self.up_big(out_set_middle)  # torch.Size([2, 128, 76, 76])
           big_cat = torch.cat([up_big, h_big], dim=1)
           out_set_tie_big = self.out_set_tie_big(big_cat)
   
           # ----------------------------PAN 36*36-------------------------------------
           neck_tie_middle = torch.cat([self.pan_middle(out_set_tie_big), out_set_middle], dim=1)
           up_middle = self.out_set_tie_middle(neck_tie_middle)
   
           # ----------------------------PAN 18*18-------------------------------------
           neck_tie_small = torch.cat([self.pan_small(up_middle), neck_small], dim=1)
           out_set_small = self.out_set_tie_small(neck_tie_small)
   
           # ----------------------------prediction-------------------------------------
           out_small = self.out_small(out_set_small)
           out_middle = self.out_middle(up_middle)
           out_big = self.out_big(out_set_tie_big)
   
           return out_small, out_middle, out_big
   ```

   

### 5.四种尺寸的配置文件的写法

放在了config字典中，这是网络模型的配置参数，没有将其他的参数放到配置文件中，可以将类别也放到配置文件中。在上面的网络代码中宽度参数就是变量`e`然后传入到每个网络中去。

   ```python
   config = {
           #            gd    gw
           'yolov5s': [0.33, 0.50],
           'yolov5m': [0.67, 0.75],
           'yolov5l': [1.00, 1.00],
           'yolov5x': [1.33, 1.25]
       }
       # 修改一次文件名字
       net_size = config['yolov5x']
       net = yolov5(nc=80, gd=net_size[0], gw=net_size[1])
   ```

v5原始代码将v3中的Head部分单独写成了一个 Detect类，主要的原因是因为v5中使用了一些训练的技巧，在Detect中有训练和两个部分，v5原始代码对于初学者来说是比较困难的，首先网络的写法，对于编码的能力要求是相对比较高的。不过这种yaml配置文件来对网络进行配置的方法在很多公司已经开始使用，这可能是未来工程话代码的一个写法，还是需要掌握这种写法的。

## 三、总结

1. 我个人的感觉是对于这种网络的设计还有代码的写法要有天马行空的想象力，代码写起来也像武侠小说中那种飘逸感。（网络结构图，网上有很多，我是仿照这江大白的结构图，在其结构图的基础上并与最新的v5代码的基础上进行了调整）。

2. 最新的v5网络结构中出现了`Transformer`结构，有种CV领域工程化上要变天的节奏，大家可以去了解一些。

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)