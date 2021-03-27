【GiantPandaCV导语】这是LeetCode的第47场双周赛的题解，本期考察的知识点有**暴力，贪心，STL，树状数组**等等。

 # 比赛链接

 - <https://leetcode-cn.com/contest/biweekly-contest-47/>

# 题目一：找到最近的有相同 X 或 Y 坐标的点

 **题解思路**：直接枚举每一个点，如果这个点的x坐标或者y坐标与目标点的对应坐标相等，则与答案取最小值，并记录下最小值的下标。**时间复杂度**：$O(n)$**解题代码**如下：

~~~c
class Solution {
public:
    int nearestValidPoint(int x, int y, vector<vector<int>>& points) {
        pair<int, int> ans(INT_MAX, -1);
        for (int i = 0; i < points.size(); i++)
            if (points[i][0] == x || points[i][1] == y)
                ans = min(ans, make_pair(abs(points[i][0] - x) + abs(points[i][1] - y), i));
        return ans.second;
    }
};
~~~

# 题目二：判断一个数字是否可以表示成三的幂的和

**题解思路**：首先，任何正整数，都可以表示为\sum_{i=0}^{\infty}3^ik_i，即三进制表示。题意即需要验证n的三进制表示中，是否每一位都是0或者1，直接模拟短除法即可。**时间复杂度**：$O(\log_3n)$**解题代码**如下：

~~~c
class Solution {
public:
    bool checkPowersOfThree(int n) {
        while (n) {
            if (n % 3 > 1)
                return false;
            n /= 3;
        }
        return true;
    }
};
~~~

# 题目三：所有子字符串美丽值之和

**题解思路**：暴力枚举每一个子串并求其美丽值，但是需要注意枚举的顺序。外层枚举子串的起始坐标，内层枚举子串的长度，记录当前子串每个字母出现的次数，并计算出现次数的最大值减去出现次数最小值的差的和。**时间复杂度**：$O(\26n^2)$**解题代码**如下：

~~~c
class Solution {
public:
    int beautySum(string s) {
        int ans = 0;
        for (int st = 0; st < s.size(); st++) {
            int cnt[26];
            memset(cnt, 0, sizeof(cnt));
            for (int len = 1; st + len <= s.size(); len++) {
                int idx = s[st + len - 1] - 'a';
                cnt[idx]++;
                int mx = 0;
                int mi = INT_MAX;
                for (int i = 0; i < 26; i++) {
                    if (cnt[i] > 0) {
                        mx = max(mx, cnt[i]);
                        mi = min(mi, cnt[i]);
                    }
                }
                ans += mx - mi;
            }
        }
        return ans;
    }
};
~~~

# 题目四：统计点对的数目

**题解思路**：

题意比较晦涩，需要仔细阅读。

首先，需要记录每个点的度du，然后记录每个点和他相邻的点的边的个数ecnt，为了方便，这里使用ecnt[u][v]表示点u和点v之间边的条数。

对于每一个点对(u,v)，存在两种情况：

1. u和v相邻，则此时连载u或v点上的边的条数为du[u]+du[v]-ecnt[u][v];
2. u和v不相邻，则此时连载u或v点上的边的条数为du[u]+du[v]。

对这两类情况分开讨论即可。
具体做法：

1. 建立一个数组数组，将每个点u在树状数组下标为du[u]的地方+1;
2. 枚举每一个点，再枚举每一个与之相邻的点，统计每个询问对于情况1的答案
3. 在树状数组对应位置减去u以及和u相邻的点的度。
4. 对于每一个询问queries[j]，需要知道有多少个与u不相邻，且du[v]>queries[j]-du[u]的v点的个数，即当前树状数组下标从queries[j]-du[u]+1到结束的这段区间的和。
5. 还原步骤3的操作，枚举下一个u点。

求出的答案还需要进行/2的操作。

时间复杂度**：$O(m*\log_2m + q*n\log_2m)$

**解题代码**如下：

~~~c
class Solution {
public:
    vector<int> countPairs(int n, vector<vector<int>>& edges, vector<int>& queries) {
        vector<unordered_map <int, int> > ecnt(n + 1);
        vector<int> du(n + 1);
        for (auto& e : edges) {
            du[e[0]]++;
            du[e[1]]++;
            ecnt[e[0]][e[1]]++;
            ecnt[e[1]][e[0]]++;
        }
        int m = edges.size();
        int qSize = queries.size();
        vector<int> ans(qSize, 0);
        m_c = vector<int>(edges.size() + 1);
        for (int u = 1; u <= n; u++) {
            add(du[u], 1, m);
        }
        for (int u = 1; u <= n; u++) {
            add(du[u], -1, m);
            for (auto& p : ecnt[u]) {
                int v = p.first;
                add(du[v], -1, m);
                for (int i = 0; i < qSize; i++) {
                    int cnt = queries[i];
                    if (du[u] + du[v] - p.second > cnt) {
                        ans[i]++;
                    }
                }
            }
            for (int i = 0; i < qSize; i++) {
                int cnt = queries[i];
                ans[i] += sum(m) - sum(cnt - du[u]);
            }
            for (auto& p : ecnt[u]) {
                int v = p.first;
                add(du[v], 1, m);
            }
            add(du[u], 1, m);
        }
        for (auto& a : ans) a /= 2;
        return ans;
    }
private:
    inline int lowbit(int x) {
        return x & -x;
    }
    inline void add(int pos, int val, int m) {
        if (pos == 0) {
            m_c[0] += val;
            return;
        }
        for (int i = pos; i <= m; i += lowbit(i)) {
            m_c[i] += val;
        }
    }
    inline int sum(int pos) {
        if (pos < 0) {
            return 0;
        }
        int ans = 0;
        for (int i = pos; i > 0; i -= lowbit(i)) {
            ans += m_c[i];
        }
        return ans + m_c[0];
    }
    vector<int> m_c;
};
~~~

**题解思路2**：
其实本题统计du[u]+du[v]-ecnt[u][v]>queries[j]的点的数目，可以先统计du[u]+du[v]>queries[j]的数目，这个可以将du数组排序后使用二指针的方法求得。然后在遍历ecnt，从总数中减去du[u]+du[v]-ecnt[u][v]<=queries[j]的数目即可。

~~~c
class Solution {
public:
    vector<int> countPairs(int n, vector<vector<int>>& edges, vector<int>& queries) {
        vector<int> du(n);
        unordered_map<int, int> cnt;
        vector<pair<int, int>> uedge;
        for (auto& e : edges) {
            e[0]--;
            e[1]--;
            du[e[0]]++;
            du[e[1]]++;
            int code = _getCode(e[0], e[1]);
            if (!cnt.count(code)) {
                uedge.emplace_back(e[0], e[1]);
            }
            cnt[code]++;
        }
        vector<int> sortDu = du;
        sort(sortDu.begin(), sortDu.end());
        vector<int> ans;
        for (int q : queries) {
            int sum = 0;
            int r = n - 1;
            for (int l = 0; l < n; l++) {
                while (r > l && sortDu[l] + sortDu[r] > q) r--;
                sum += n - max(l, r) - 1;
            }
            for (auto& e : uedge) {
                if (du[e.first] + du[e.second] > q &&
                    du[e.first] + du[e.second] - cnt[_getCode(e.first, e.second)] <= q) {
                    sum--;
                }
            }
            ans.emplace_back(sum);
        }
        return ans;
    }
private:
    int _getCode(int x, int y) {
        if (x > y)
            swap(x, y);
        return x << 16 | y;
    }
};
~~~

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)