> 【GiantPandaCV导语】**这是LeetCode的第42场双周赛的题解，公众号后面每周周末会以主条推文的方式更新当前周的Leetcode比赛的解题报告，并整理成一个面试刷题专栏，有需要的读者可以关注一下。另外，以前的比赛的解题报告我也会从当前这个时间点往前补，原创的解题报告会尽量以次条的方式放在CV相关的推文下方，不影响目前公众号的布局。本次周赛考察的知识点有：c++ stl，队列，堆栈，模拟，找规律，数论，前缀和等。如果你只是关心面试，看前面三题就好，最后一题有亿点难。**

# 比赛链接
- https://leetcode-cn.com/contest/biweekly-contest-42/

# 题目一：无法吃午餐的学生数量

![题面](https://img-blog.csdnimg.cn/20201227182231574.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：数据范围比较小，按照题意直接用c++ stl的deque和stack进行模拟即可，当队列里面所有元素都和栈顶元素不存在喜欢关系就退出模拟过程。

**时间复杂度：O(n)**

**解题代码**如下：

```cpp
class Solution {
public:
    int countStudents(vector<int>& students, vector<int>& sandwiches) {
        deque <int> q;
        stack <int> s;
        int len = students.size();
        for(int i=0; i<students.size(); i++){
            q.push_back(students[i]);
        }
        for(int i=len-1; i>=0; i--){
            s.push(sandwiches[i]);
        }
        int ans = len;
        int cnt = 0;
        while(q.size()){
            int x = q.front();
            q.pop_front();
            if(x == s.top()){
                ans--;
                s.pop();
                cnt = 0;
            }
            else{
                q.push_back(x);
                cnt++;
                if(cnt>100) break;
            }
        }
        return ans;
    }
};
```

# 题目2：平均等待时间

![题面](https://img-blog.csdnimg.cn/20201227182828821.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：由于做菜顺序是固定的，按照题意模拟即可，动态维护一下做完上一道菜的截至时间。

**时间复杂度：O(n)**

**解题代码**如下：

```cpp
class Solution {
public:
    double averageWaitingTime(vector<vector<int>>& customers) {
        double ans = 0;
        double now = 0;
        for(int i=0; i<customers.size(); i++){
            now = max(now, (double)customers[i][0]);
            ans = ans + (now + customers[i][1] - customers[i][0]);
            now = now + customers[i][1];
        }
        ans = ans / customers.size();
        return ans;
    }
};
```

# 题目3：修改后的最大二进制字符串

![题面](https://img-blog.csdnimg.cn/20201227183330812.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：看起来是一道比较复杂的题，实际上是个傻题。手推两组数组容易发现，无论输入是什么字符串最终都可以变成前面全0，后面全1的排列，并且把前面的全0使用操作1进行替换能获得最优解。需要注意一下，**前导1不需要处理**，直接输出即可。

**时间复杂度：O(n)**

**解题代码**如下：

```cpp
class Solution {
public:
    string maximumBinaryString(string binary) {
        int len = binary.size();
        string s = binary;

        int cnt1 = 0, cnt2 = 0;
        int pos = -1;
        for(int i=0; i<len; i++){
            if(s[i]=='0'){
                pos = i-1;
                break;
            }
        }
        for(int i=pos+1; i<len; i++){
            if(s[i]=='0') cnt1++;
            else cnt2++;
        }
        string t="";
        for(int i=0; i<=pos; i++) t+='1';
        for(int i=0; i<cnt1; i++) t+='0';
        for(int i=0; i<cnt2; i++) t+='1';
        for(int i=0; i<t.size()-1; i++){
            if(t[i]=='0'&&t[i+1]=='0'){
                t[i]='1';
            }
        }


        return t;
    }
};
```

# 题目4：得到连续 K 个 1 的最少相邻交换次数

![题面](https://img-blog.csdnimg.cn/20201227183908892.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：一个比较套路的题，比赛时没有做出来，看赛后题解学会了，下面分享一下做法。

根据题意知道，我们需要：
- 找到数组中$k$个连续的1。
- 找到数组中$k$个连续的位置。

然后将数组中$k$个连续的1放到对应的位置中，为什么这样做没有问题呢？想象不按照顺序放入$k$个位置，也就是存在交叉的情况，交换次数只会更多。

然后假设$1$的位置在$[p_0,...,p_{k-1}]$，然后放入的位置为$[q, ..., q+k-1]$，那么题意就转换成求下式的最小值：

$\sum_{i=0}^{k-1}|p_i-(q+i)|$

然后我们设$p_i^{'}=p_i-i$，那么上式等价于：

$\sum_{i=0}^{k-1}|(p_i^{'}+i)-(q+i)|=\sum_{i=0}^{k-1}|p_i^{'}-q|$

那么这个式子在何时取得最小值呢？显然当$q$为${p_i^{'}}$的中位数时，上式可以取得最小值。

所以，我们首先记录一下原数组中所有1的位置，用$f_0,f_1,..,f_{m-1}$来表示，然后再令$g_i={f_i-i}$。然后我们就可以用一个长度为$k$的滑动窗口在$g$上实时滑动并更新和计算答案了。

最后一个关键的点就是交换次数如何计算，假设当前我们的滑动窗口包含$[g_i,..,g_{i+k-1}]$，中位数$q$自然等于$\frac{g[(2*i+k-1)/2]}{2}$。那么此时交换次数计算推导如下：

$\sum_{j=i}^{i+k-1}|g_j-q|$

$=\sum_{j=1}^{mid-1}(q-g_j)+\sum_{j=mid+1}^{i+k-1}(g_j-q)$

$=(mid-i)q-\sum_{j=i}^{mid-1}g_j+\sum_{j=mid+1}^{i+k-1}g_j-(i+k-mid-1)q$

$=(2(mid-i)-k+1)q+\sum_{j=mid+1}^{i+k-1}g_j-\sum_{j=i}^{mid-1}$

其中后面的求和项可以通过提前维护一个前缀和来O(1)计算。

**时间复杂度：O(n)**

**解题代码**：


```cpp
class Solution {
public:
    int minMoves(vector<int>& nums, int k) {
        if (k == 1) {
            return 0;
        }
        
        int n = nums.size();
        vector<int> g;
        vector<int> sum = {0};
        int cnt = 0;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == 1) {
                g.push_back(i - cnt);
                sum.push_back(sum.back() + g.back());
                cnt += 1;
            }
        }
        
        int m = g.size();
        int ans = INT_MAX;
        for (int i = 0; i + k <= m; ++i) {
            int mid = (i + i + k - 1) / 2;
            int q = g[mid];
            ans = min(ans, (2 * (mid - i) - k + 1) * q + (sum[i + k] - sum[mid + 1]) - (sum[mid] - sum[i]));
        }
        
        return ans;
    }
};
```


# 参考：

https://leetcode-cn.com/problems/minimum-adjacent-swaps-for-k-consecutive-ones/solution/de-dao-lian-xu-k-ge-1-de-zui-shao-xiang-lpa9i/

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)