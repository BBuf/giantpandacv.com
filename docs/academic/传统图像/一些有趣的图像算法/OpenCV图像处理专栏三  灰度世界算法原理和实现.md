# 前言
这是 OpenCV图像处理算法专栏的第三篇文章，为大家介绍一下灰度世界算法的原理和C++实现，这个算法可以起到白平衡的作用。
# 灰度世界算法原理
人的视觉系统具有颜色恒常性，能从变化的光照环境和成像条件下获取物体表面颜色的不变特性，但成像设备并不具有这样的调节功能，不同的光照环境会导致采集到的图像颜色与真实颜色存在一定程度的偏差，需要选择合适的颜色平衡算法去消除光照环境对颜色显示的影响。
灰度世界算法以灰度世界假设为基础，假设为：对于一幅有着大量色彩变化的图像，RGB三个分量的平均值趋于同一个灰度值$\bar Gray$。从物理意思上讲，灰度世界算法假设自然界景物对于光线的平均反射的均值在整体上是一个定值，这个定值近似为“灰色”。颜色平衡算法将这一假设强制应用于待处理的图像，可以从图像中消除环境光的影响，获得原始场景图像。

# 算法步骤

- 确定Gray有2种方法，一种是取固定值，比如最亮灰度值的一半，8位显示为128。另一种就是通过计算图像R,G,B的三个通道$\bar R$,$\bar G$,$\bar B$，取$\bar Gray=\frac{\bar R + \bar G + \bar B}{3}$
- 计算$R$,$G$,$B$，3个通道的增益系数：$k_r=\frac{\bar Gray}{\bar R}$,$k_g=\frac{\bar Gray}{\bar G}$,$k_b=\frac{\bar Gray}{\bar B}$
- 根据Von Kries对角模型，对于图像中的每个像素C，调整其分量R,G,B分量：$C(R')=C(R)*k_r$,$C(G')=C(G)*k_g$,$C'(B)=C(B)*k_b$
# 算法优缺点
此算法简单快速，但是当图像场景颜色并不丰富时，尤其是出现大量单色物体时，该算法会失效。
# 源码实现

```
Mat GrayWorld(Mat src) {
  vector <Mat> bgr;
  cv::split(src, bgr);
  double B = 0;
  double G = 0;
  double R = 0;
  int row = src.rows;
  int col = src.cols;
  Mat dst(row, col, CV_8UC3);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      B += 1.0 * src.at<Vec3b>(i, j)[0];
      G += 1.0 * src.at<Vec3b>(i, j)[1];
      R += 1.0 * src.at<Vec3b>(i, j)[2];
    }
  }
  B /= (row * col);
  G /= (row * col);
  R /= (row * col);
  printf("%.5f %.5f %.5f\n", B, G, R);
  double GrayValue = (B + G + R) / 3;
  printf("%.5f\n", GrayValue);
  double kr = GrayValue / R;
  double kg = GrayValue / G;
  double kb = GrayValue / B;
  printf("%.5f %.5f %.5f\n", kb, kg, kr);
  for (int i = 0; i < row; i++) {
    for (int j = 0; j < col; j++) {
      dst.at<Vec3b>(i, j)[0] = (int)(kb * src.at<Vec3b>(i, j)[0]) > 255 ? 255 : (int)(kb * src.at<Vec3b>(i, j)[0]);
      dst.at<Vec3b>(i, j)[1] = (int)(kg * src.at<Vec3b>(i, j)[1]) > 255 ? 255 : (int)(kg * src.at<Vec3b>(i, j)[1]);
      dst.at<Vec3b>(i, j)[2] = (int)(kr * src.at<Vec3b>(i, j)[2]) > 255 ? 255 : (int)(kr * src.at<Vec3b>(i, j)[2]);
    }
  }
  return dst;
}
```
# 效果
**原图**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190103203458190.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**灰度世界算法处理后**

![在这里插入图片描述](https://img-blog.csdnimg.cn/20191207144543324.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

# 后记
可以看到灰度世界算法有了白平衡的效果，并且该算法的执行速度也是非常的快。

---------------------------------------------------------------------------

欢迎关注我的微信公众号GiantPandaCV，期待和你一起交流机器学习，深度学习，图像算法，优化技术，比赛及日常生活等。

![图片.png](https://imgconvert.csdnimg.cn/aHR0cHM6Ly91cGxvYWQtaW1hZ2VzLmppYW5zaHUuaW8vdXBsb2FkX2ltYWdlcy8xOTIzNzExNS1hZDY2ZjRmMjQ5MzRhZmQx?x-oss-process=image/format,png)