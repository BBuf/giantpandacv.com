> 论文获取请在公众号后台回复：MaxMin
# 1. 前言
最近有点忙，今天水一下。来为大家介绍一个之前看到的一个有趣的常量阶最大值最小值滤波算法，这个算法可以在对每个元素的比较次数不超过3次的条件下获得任意半径区域内的最大值或者最小值，也即是说可以让最大最小值滤波算法的复杂度和半径无关。
# 2. 算法介绍
普通实现的最大最小值滤波复杂度是非常高的，因为涉及到遍历$r\times r$的滑动窗口中的所有值然后求出这个窗口所有值的最大和最小值。尽管可以使用sse优化，但速度仍然快不了多少（后面会介绍这个算法的SSE优化）。然后偶然看到了这篇论文，题目为：`STREAMING MAXIMUM-MINIMUM FILTER USING NO
MORE THAN THREE COMPARISONS PER ELEMENT`。它介绍了一个最大最小值滤波的优化方法，使得这两个滤波器算法的复杂度可以和滤波半径$r$无关。

# 3. 算法原理
算法的核心原理如下图所示：

![算法伪代码](https://img-blog.csdnimg.cn/20200418200704688.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
其实算法也是比较好理解的，即动态维护一个长度为$r$（滤波窗口大小）的单调队列，然后可以在任意位置获取以当前点为结束点的滤波窗口中的最大值或者最小值。

#  4. 代码实现

```
#include <bits/stdc++.h>
using namespace std;
const int maxn = 110;
deque <int> U, L;
int maxval[maxn], minval[maxn];

int main(){
    int a[10] = {0, 1, 9, 8, 2, 3, 7, 6, 4, 5};
    int w = 3;
    for(int i = 1; i < 10; i++){
        if(i >= w){
            maxval[i - w] = a[U.size() > 0 ? U.front() : i-1];
            minval[i - w] = a[L.size() > 0 ? L.front() : i-1];
        }
        if(a[i] > a[i-1]){
            L.push_back(i - 1);
            if(i == w + L.front()) L.pop_front();
            while(U.size() > 0){
                if(a[i] <= a[U.back()]){
                    if(i == w + U.front()) U.pop_front();
                    break;
                }
                U.pop_back();
            }
        }
        else{
            U.push_back(i-1);
            if(i == w + U.front()) U.pop_front();
            while(L.size() > 0){
                if(a[i] >= a[L.back()]){
                    if(i == w + L.front()) L.pop_front();
                    break;
                }
                L.pop_back();
            }
        }
    }
    maxval[10 - w] = a[U.size() > 0 ? U.front() : 9];
    minval[10 - w] = a[L.size() > 0 ? L.front() : 9];
    for(int i = 0; i <= 10 - w; i++){
        printf("Min index %d is: %d\n", i, minval[i]);
        printf("Max index %d is: %d\n", i, maxval[i]);
    }
    return 0;
}
```

# 5. 算法结果
得到的结果如下入所示：
![结果图](https://img-blog.csdnimg.cn/20190420221328881.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
# 6. 总结
上面的算法是对一个序列进行求长度为$w$的一维窗口的最大最小值，我们只需要把$2$维的Mat看成两个一维的序列，分别求一下然后综合一下两个维度的结果即可。我们最后可以发现整个最大最小值滤波的算法复杂度和滤波的半径没有任何关系，这确实是一个很优雅高效的算法。

---------------------------------------------------------------------------

欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)