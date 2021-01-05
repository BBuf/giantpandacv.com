【GiantPandaCV导语】这是LeetCode的第221场周赛的题解，本期考察的知识点有模拟，贪心，优先队列，01Trie树等。

# 比赛链接
- https://leetcode-cn.com/contest/weekly-contest-221/

# 题目一：判断字符串的两半是否相似

![题面](https://img-blog.csdnimg.cn/20210103091439130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：直接模拟即可。
**时间复杂度**：O(s.length)
**解题代码**如下：

```cpp
class Solution {
public:
    bool yuan(char c){
        if(c=='a'||c=='e'||c=='i'||c=='o'||c=='u'||c=='A'||c=='E'||c=='I'||c=='O'||c=='U') return 1;
        else return 0;
    }
    bool check(string t1, string t2){
        int cnt1=0, cnt2=0;
        for(int i=0; i<t1.size(); i++){
            if(yuan(t1[i])) cnt1++;
        }
        for(int i=0; i<t2.size(); i++){
            if(yuan(t2[i])) cnt2++;
        }
        return cnt1==cnt2;
    }
    bool halvesAreAlike(string s) {
        string t1 = "", t2 = "";
        int len = s.size();
        for(int i=0; i<len; i++){
            if(i<len/2) t1+=s[i];
            else t2+=s[i];
        }
        bool ans=check(t1, t2);
        return ans;
    }
};
```

# 题目二：吃苹果的最大数目
![题面](https://img-blog.csdnimg.cn/20210103091741575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：这道题的核心思路是要每次优先吃最早过期的苹果。然后，我们用优先队列保存到当前时间为止拥有的苹果（苹果有数量和过期时间两个属性），每次从优先队列里面取出最早过期的苹果。

如果取到的苹果已经过期，即过期的时间小于等于当前的天数，那么将其从优先队列中移除并继续取。直到取到第一个可以吃的苹果，并吃掉它，将它的数量减1，如果苹果的数量不为0，则再次将其放入优先队列。

直到队列为空，程序结束。

**时间复杂度**：O(nlogn)

**解题代码**：


```cpp
struct node{
    // cnt代表个数，idx代表过期时间
    int cnt, idx;
    node(){}
    node(int cnt_, int idx_):cnt(cnt_),idx(idx_){}
    bool operator<(const node& rhs) const{
        return idx > rhs.idx;
    }
};

class Solution {
public:
    int eatenApples(vector<int>& apples, vector<int>& days) {
        int n = apples.size();
        int ans = 0;
        priority_queue <node>  pq;
        int i = 0;
        int nn=n;   //定义可变右边界
        while(i<nn){
            while(!pq.empty() && pq.top().idx<=i){ //移除烂掉的苹果
                pq.pop();
            }
            if(i<n && apples[i]>0){  //添加有效苹果到优先队列里
                pq.push(node(apples[i],i+days[i]));
                nn=max(nn,i+days[i]+1); //比较，获得较大的右侧边界
            }
            if(!pq.empty()){
                node now=pq.top();  //拿出靠前的一堆苹果，并吃掉一个
                pq.pop();
                now.cnt-=1;
                ans++;
                if(now.cnt>0){  //如果这堆没吃完，再放回去
                    pq.push(now);
                }
            }
            i++;
        }
        return ans;
    }
};
```

# 题目三：球会落在何处
![题面](https://img-blog.csdnimg.cn/20210103094122517.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
![题面](https://img-blog.csdnimg.cn/20210103094137509.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：仔细观察可以发现
- 如果在当前格子中 为'\'，那么它右边格子也应该为'\'，小球才能移动到下一行。
- 如果在当前格子中 为'/'，那么它右边格子也应该为'/'，小球才能移动到下一行。
-   其它情况均无法移动到最后一行。
**时间复杂度**：O(n*m)

**解题代码**如下：

```cpp
class Solution {
public:
    vector<int> ans;
    vector<int> findBall(vector<vector<int>>& grid)
    {
        int m = grid.size();
        int n = grid[0].size();
        ans.resize(n);

        // 初始化每一个小球的位置
        for (int i = 0; i < n; i++) {
            ans[i] = i;
        }

        for (int i = 0; i < m; ++i) { // 计算小球将出现在下一行的哪个位置
            for (int j = 0; j < n; ++j) { // 计算每一列的每一个小球
                if (ans[j] == -1) { // 如果当前列的小球已经停止向下运动, 跳过
                    continue;
                }

                int now = ans[j]; // 得到小球在这一列的位置
                // 根据辅助图, 如果在当前格子中 为'\'，那么它右边格子也应该为'\'，小球才能移动到下一行
                if (grid[i][now] == 1 && now < n - 1 && grid[i][now + 1] == 1) {
                    ans[j] += 1;
                } else if (grid[i][now] == -1 && now >= 1 && grid[i][now - 1] == -1) { // 如果在当前格子中 为'/'，那么它右边格子也应该为'/'，小球才能移动到下一行
                    ans[j] -= 1;
                } else { // 其他情况都无法向下移动
                    ans[j] = -1;
                }
            }
        }

        return ans;
    }
};
```

# 题目四：与数组中元素的最大异或值

![题面](https://img-blog.csdnimg.cn/20210103094510225.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)
**解题思路**：我们可以使用01字典树来存储数组中的所有元素。另外，由于每个查询包含一个上界$m_i$，因此我们可以在字典树的每一个节点再维护一个以当前节点为根节点的子树的最小元素。它的作用是，如果当前节点的子树的最小元素大小都大于$m_i$，我们就没有必要继续对这个子树进行搜索了。

然后考虑每一次查询，我们从高位到低位进行依次处理，实际上有以下两种情况：

- 如果$x_i$的当前二进制位为1，那么我们应该在Trie树上优先走位为0的分支；否则，我们需要尝试走Trie树上位为1的分支，如果分支不存在或者分支的最小元素已经超过了$m_i$，则本地查询无解。
- 如果$x_i$的当前二进制位为0，那么我们应该优先走当前位为1的分支，但要求这一分支的最小元素不超过$m_i$；否则，我们继续走当前位为0的分支。

如果我们能顺利走到了最低位，那么我们就获得了这一查询的最优解。

**时间复杂度**：O(n+q*log(maxx))

**空间复杂度**：O(nlog(maxx)+q)

**解题代码**如下：

```cpp
struct TrieNode {
    int minn = 1e9;
    TrieNode* children[2]{};
};

class Solution {
public:
    vector<int> maximizeXor(vector<int>& nums, vector<vector<int>>& queries) {
        TrieNode* root = new TrieNode();

        for (int j = 0; j < nums.size(); j++) {
            int num = nums[j];
            TrieNode* p = root;

            for (int i = 30; i >= 0; --i) {
                int nxt = (num & (1 << i)) ? 1 : 0;
                if (!p->children[nxt]) p->children[nxt] = new TrieNode();
                p = p->children[nxt];
                p->minn = min(p->minn, num);
            }
        }

        vector<int> ans;

        for (int j = 0; j < queries.size(); j++) {
            int x = queries[j][0], limit = queries[j][1];
            int sum = 0;
            TrieNode* p = root;

            for (int i = 30; i >= 0; --i) {
                if (x & (1 << i)) {
                    if (p->children[0]) {
                        p = p->children[0];
                        sum ^= (1 << i);
                    } else if (!p->children[1] || (p->children[1]->minn > limit)) {
                        ans.push_back(-1);
                        break;
                    } else {
                        p = p->children[1];
                    }
                } else {
                    if (p->children[1] && (p->children[1]->minn <= limit)) {
                        p = p->children[1];
                        sum ^= (1 << i);
                    } else if (!p->children[0]) {
                        ans.push_back(-1);
                        break;
                    } else {
                        p = p->children[0];
                    }
                }
                if (i == 0) ans.push_back(sum);
            }
        }
        return ans;
    }
};
```

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)