【GiantPandaCV导语】这是LeetCode的第222场周赛的题解，本期考察的知识点有**贪心，哈希，二分，LIS**等等。

# 比赛链接
- https://leetcode-cn.com/contest/weekly-contest-222/

# 题目一：卡车上的最大单元数
![题面](https://img-blog.csdnimg.cn/20210110093838102.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：贪心，优先选取装载的单元数量多的箱子即可。
**时间复杂度**：O(nlogn)
**解题代码**如下：

```cpp
class Solution {
public:
    struct node{
        int x, y;
        node(){}
        node(int x_, int y_):x(x_),y(y_){}
        bool operator<(const node&rhs) const{
            return y > rhs.y;
        }
    }a[1010];
    int maximumUnits(vector<vector<int>>& boxTypes, int truckSize) {
        int n = boxTypes.size();
        for(int i=0; i<n; i++){
            a[i]=node(boxTypes[i][0], boxTypes[i][1]);
        }
        sort(a, a+n);
        int ans = 0;
        int cnt = 0;
        for(int i=0; i<n; i++){
            for(int j=0; j<a[i].x; j++){
                if(cnt + 1 <= truckSize){
                    ans += a[i].y;
                    cnt += 1;
                }
                else break;
            }
        }
        return ans;
    }
};
```

# 题目二：大餐计数

![题面](https://img-blog.csdnimg.cn/20210110094904171.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：题目要求计算两道不同餐品的美味程度之和为2的幂次，想到利用哈希表的方式将复杂度降低到线性级别，我这里使用的是unordered_map。需要注意一下当前餐品的美味程度的2倍恰好等于枚举的美味程度的情况，统计答案的时候要减掉1，比如2+2等于4，实际上是只有一对餐品的。
**时间复杂度**：O(22*N)
**解题代码**如下：

```cpp
class Solution {
public:
    unordered_map <int, int> mp;
    long long mod=1e9+7;
    int countPairs(vector<int>& deliciousness) {
        mp.clear();
        for(int i=0; i<deliciousness.size(); i++){
            mp[deliciousness[i]]++;
        }
        long long ans=0;
        for(int i=0; i<deliciousness.size(); i++){
            int x = deliciousness[i];
            for(int j=0; j<22; j++){
                if(mp.count((1<<j)-x)){
                    if(2*x==(1<<j))
                        ans += mp[(1<<j)-x] - 1;
                    else
                        ans += mp[(1<<j)-x];
                }
            }
        }
        return ans / 2 % mod;
    }
};
```

# 题目三：将数组分成三个子数组的方案数
![题面](https://img-blog.csdnimg.cn/20210110100604963.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：首先维护一个前缀和设为pre，然后我们枚举第一个位置X，现在就是要找到一个位置Y使得题意中的条件成立，并且这个Y可能不是一个值，可能是一个区间，它的上下界可以用下面的公式来确定：

```cpp
// pre[X] <= pre[Y] - pre[X] <= pre[n] - pre[Y];
// 2 * Pre[X] <= Pre[Y]
// 2 * pre[Y] <= pre[n] + pre[X]
```

对于上下区间值的确定就可以直接用`lower_bound`和`upper_bound`了。

**时间复杂度**：O(nlogn)

**解题代码**如下：

```cpp
const int mod = 1e9 + 7;
typedef long long int64;
class Solution {
public:
    int waysToSplit(vector<int>& a) {
        int n = a.size();
        vector<int> pre(n + 1);
        for (int i = 0; i < n; ++i) pre[i + 1] = pre[i] + a[i];

        int ret = 0;
        for (int X = 1; X + 2 <= n; ++X) {
            int Y1 = lower_bound(pre.begin(), pre.end(), 2 * pre[X]) - pre.begin();
            int Y2 = upper_bound(pre.begin(), pre.end(), (pre[n] + pre[X]) / 2) - pre.begin();
            Y1 = max(Y1, X + 1);
            Y2 = min(n, Y2);
            ret += max(0, Y2 - Y1);
            ret %= mod;
        }
        return ret;
    }
};
```



# 题目四：得到子序列的最少操作次数

![题面](https://img-blog.csdnimg.cn/20210110102017724.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：脑洞题，我没有脑洞，所以不会QAQ。Copy一份官方题解吧。地址为：`https://leetcode-cn.com/problems/minimum-operations-to-make-a-subsequence/solution/zui-chang-shang-sheng-zi-xu-lie-huan-cho-zve8/`

![第4题解](https://img-blog.csdnimg.cn/20210110102350139.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**时间复杂度**：O(nlogn)
**解题代码**如下：

```cpp
// 本段代码也来自上面题解作者。
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) { // 最长上升子序列, O(nlogn)
        int len = 1, n = (int)nums.size();
        if (n == 0) {
            return 0;
        }
        vector<int> d(n + 1, 0);
        d[len] = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] > d[len]) {
                d[++len] = nums[i];
            } else {
                int l = 1, r = len, pos = 0;
                while (l <= r) {
                    int mid = (l + r) >> 1;
                    if (d[mid] < nums[i]) {
                        pos = mid;
                        l = mid + 1;
                    } else {
                        r = mid - 1;
                    }
                }
                d[pos + 1] = nums[i];
            }
        }
        return len;
    }

    int minOperations(vector<int>& target, vector<int>& arr) {
        // 「标号」
        unordered_map<int, int> m;
        int idx = 0;
        for (int i = 0; i < target.size(); i++) {
            m[target[i]] = i;
        }
        vector<int> actual;
        for (int x: arr) {
            if (m.find(x) != m.end()) {
                actual.push_back(m[x]);
            }
        }
        
        int lis = lengthOfLIS(actual);
        return target.size() - lis;
    }
};
```

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)