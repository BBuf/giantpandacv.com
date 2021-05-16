【GiantPandaCV导语】本次周赛做得还算顺利，主要考察了模拟，二分，单调栈，拓扑排序以及树形DP等知识点，这里记录一下题解。



# A. 人口最多的年份

![题面](https://img-blog.csdnimg.cn/20210510230743847.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

**解题思路**：数据范围比较小，可以直接模拟，从年份开始循环，然后再循环logs数组，如果年份在出生和死亡年份之间就记录。

**AC代码**：


```cpp
class Solution {
public:
    int maximumPopulation(vector<vector<int>>& logs) {
        int mx = 0, ans = 0;
        for(int i=1950; i<=2050; i++){
            int t = 0;
            for(int j=0; j<logs.size(); j++){
                if(logs[j][0]<=i&&i<logs[j][1]){
                    t++;
                }
            }
            if(t > mx){
                mx = t;
                ans = i;
            }
        }
        return ans;
    }
};
```

# B. 下标对中的最大距离

![题面](https://img-blog.csdnimg.cn/20210510231350121.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


**解题思路**：看一下数据量1e5，并且数组是有序的，那么枚举一个下标，二分另外一个下标就可以了。不过更好的做法是双指针，我更擅长写二分，这里只接写的二分法的代码。

**AC代码**：

```cpp
class Solution {
public:
    int maxDistance(vector<int>& nums1, vector<int>& nums2) {
        int ans = 0;
        for(int i=0; i<nums1.size(); i++){
            int l = i, r = nums2.size()-1;
            int now = -1;
            while(l <= r){
                int mid=(l+r)>>1;
                if(nums2[mid]>=nums1[i]){
                    l=mid+1;
                    now=mid;
                }
                else{
                    r=mid-1;
                }
            }
            if(now!=-1){
                ans = max(ans, now-i);
            }
        }
        return ans;
    }
};
```

# C. 子数组最小乘积的最大值

![题面](https://img-blog.csdnimg.cn/20210510232133100.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)


**解题思路**：以当前$nums[i]$为中心，往左往右找不小于$nums[i]$最远处$l$和$r$，这样$sum[l-r]$最大，$nums[i] * sum[l-r]$最大。需要注意结果要取模。

**AC代码**:

```cpp
class Solution {
public:
    long long sum[100010];
    int mod=1e9+7;
    int maxSumMinProduct(vector<int>& nums) {
        long long maxx = 0;
        int n = nums.size();
        for(int i=1; i<=n; i++){
            sum[i] = sum[i-1] + nums[i-1];
        }
        stack <int> s;
        for(int i=0; i<=n; i++){
            int now = i == n ? -1 : nums[i];
            while(!s.empty() && now < nums[s.top()]){
                int id = s.top();
                s.pop();
                int num = nums[id];
                if(s.empty()){
                    maxx = max(maxx, sum[i] * num);
                }
                else{
                    maxx = max(maxx, (sum[i] - sum[s.top()+1]) * num);
                }
            }
            s.push(i);
        }
        maxx %= mod;
        return (int)maxx;
    }
};
```


# D. 有向图中最大颜色值
![题面上](https://img-blog.csdnimg.cn/20210510232319776.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

![题面下](https://img-blog.csdnimg.cn/20210510232345599.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

解题思路：显然，当这个有向图有回路的话结果无限大，输出-1。我们可以通过拓扑排序的方法判断有向图是否有环，若这个有向图不具有拓扑次序，那么就说明有环。然后就是对结果的求解，我们依次枚举每一个字母，求出这个字母在所有路中的某一条中出现次数的最大值。根据前面求出的拓扑次序DP一下即可，dp[i]表示以i点为起点的路径中，当前字母c的最大出现次数，那么转移就是$dp[i]=max(dp[i], dp[j])$，$dp[i]++, s[i]==c$其中j是i的儿子节点。



```cpp
#define mst(a,b) memset((a),(b),sizeof(a))
typedef long long ll;
const int maxn = 300005;
const ll mod = 1e9+7;
const ll INF = 1e9;
const double eps = 1e-9;

int n,m;
int degree[maxn];
char s[maxn];
int dp[maxn];
vector<int>vec[maxn];
vector<int>mp;

bool check()
{
    mp.clear();
    queue<int>q;
    for(int i=1;i<=n;i++)
    {
        if(degree[i]==0) q.push(i);
    }
    while(q.size())
    {
        int u=q.front();
        q.pop();
        mp.push_back(u);
        for(int i=0;i<vec[u].size();i++)
        {
            degree[vec[u][i]]--;
            if(degree[vec[u][i]]==0) q.push(vec[u][i]);
        }
    }
    return mp.size()==n;
}

int solve(int x)
{
    mst(dp,0);
    char c=x+'a';
    int cnt=0;
    for(int i=mp.size()-1;i>=0;i--)
    {
        int u=mp[i];
        for(int j=0;j<vec[u].size();j++)
        {
            dp[u]=max(dp[u],dp[vec[u][j]]);
        }
        if(s[u-1]==c) dp[u]++;
        cnt=max(cnt,dp[u]);
    }
    return cnt;
}

class Solution {
public:
    int largestPathValue(string colors, vector<vector<int>>& edges) {
        for(int i=0; i<colors.size(); i++) s[i] =colors[i];
        n = colors.size();
        for(int i=1;i<=n;i++) vec[i].clear();
        m = edges.size();
        mst(degree,0);
        for(int i=0; i<m; i++){
            edges[i][0]++;
            edges[i][1]++;
            vec[edges[i][0]].push_back(edges[i][1]);
            degree[edges[i][1]]++;
        }
        if(!check()) return -1;
        int ans = 0;
        for(int i=0;i<26;i++)
        {
            ans=max(ans,solve(i));
        }
        return ans;
    }
};

```


总的来说，是一次比较好的周赛体验QAQ。