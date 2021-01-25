【GiantPandaCV导语】这是LeetCode的第225场周赛的题解，本期考察的知识点有**暴力，前缀和，推导**等等。 

# 比赛链接

- <https://leetcode-cn.com/contest/weekly-contest-225/> 

# 题目一：替换隐藏数字得到的最晚时间  

![第一题题面](https://img-blog.csdnimg.cn/20210124133909771.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：直接从高位开始判断即可，注意"?5:00"这种情况下，首位只能填1。**时间复杂度**：$O(1)$**解题代码**如下：

~~~c
class Solution {
public:
    string maximumTime(string time) {
        if(time[0]=='?')
        {
            if(time[1]!='?'&&time[1]>='4')
                time[0]='1';
            else
                time[0]='2';
        }
        if(time[1]=='?')
        {
            if(time[0]=='2')
                time[1]='3';
            else
                time[1]='9';
        }
        if(time[3]=='?')
            time[3]='5';
        if(time[4]=='?')
            time[4]='9';
        return time;
    }
    
};
~~~

# 题目二：满足三条件之一需改变的最少字符数  

![第二题题面](https://img-blog.csdnimg.cn/20210124133936184.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

题面

**解题思路**：先来判断前面两个条件，这两个条件其实是同一种条件；这里假设结果是a>b,那么a中的最小字母要比b中的所有字母大，那么我们可以枚举这个最小字母，如果a中的当前的字母比这个字母更小，则cnt++，如果b中的字母大于等于这个字母，同样cnt++，这里注意枚举要从字母b开始，因为字母a不可能成为限定的条件，对于最后一个条件，直接枚举相同字母即可。**时间复杂度**：$O(26*n)$**解题代码**如下：

~~~c
class Solution {
public:
    int solve(string a,string b)
    {
        int ans=1000000;
        for(int i=1;i<26;i++)
        {
            int cnt=0;
            for(int j=0;j<a.size();j++)
                if(a[j]-'a'<i)
                    cnt++;
            for(int j=0;j<b.size();j++)
                if(b[j]-'a'>=i)
                    cnt++;
            ans=min(ans,cnt);
        }
        return ans;
    }
    int minCharacters(string a, string b) {
        int ans=min(solve(a,b),solve(b,a));
        for(int i=0;i<26;i++)
        {
            int cnt=0;
            for(int j=0;j<a.size();j++)
                if(a[j]-'a'!=i)
                    cnt++;
            for(int j=0;j<b.size();j++)
                if(b[j]-'a'!=i)
                    cnt++;
            ans=min(ans,cnt);
        }
        return ans;
    }
};
~~~

# 题目三：找出第 K 大的异或坐标值  

![第三题题面](https://img-blog.csdnimg.cn/20210124134003142.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：直接使用类似于二维前缀和的思想，维护一个前缀异或的数组，然后可以用堆或者排序求第k大$dp[i][j] = (dp[i-1][j]\oplus dp[i][j-1]\oplus dp[i-1][j-1]\oplus matrix[i][j]);$**时间复杂度**：$O(n^2)$**解题代码**如下：

~~~c
class Solution {
public:
    int dp[1005][1005];
    int kthLargestValue(vector<vector<int>>& matrix, int k) {
        int n=matrix.size(),m=matrix[0].size();
        vector<int> ans;
        for(int i=1;i<=n;i++)
            for(int j=1;j<=m;j++)
            {
                dp[i][j] = (dp[i-1][j]^dp[i][j-1]^dp[i-1][j-1]^matrix[i-1][j-1]);
                ans.push_back(dp[i][j]);
            }
        sort(ans.rbegin(),ans.rend());
        return ans[k-1];
    }
};
~~~

# 题目四：放置盒子  

![第四题题面](https://img-blog.csdnimg.cn/20210124134034130.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：我们假设示例3中的摆放为3层标准摆放，也就是3层高的最小占地摆放，那么可以推出1层标准摆放的盒子数为1；2层标准摆放的盒子数为1+3=4；3层标准摆放的盒子数为1+3+6=10；以此类推。现在假设我们有n个盒子，为了尽可能减小占地面积，我们希望能够尽可能贴近标准摆放，可以用前面的递推求出来使用不超过n个盒子的最大标准摆放，现在还剩下一些盒子，从第一层开始贴近当前的标准摆放进行堆叠，这里简单推导一下即可得到：假设第一层放$X$个，那么上一层可以放$X-1$个，再往上可以放$X-2$个。那么求个最小的前k项和，判断是否大于等于剩下盒子个数即可。

**时间复杂度**：$O(\sqrt{n})$**解题代码**如下：

~~~c
class Solution {
public:
    int minimumBoxes(int n) {
        int s=1,tot=0,res=0;
        while(tot<n)
        {
            for(int i=1;i<=s;i++)
            {
                tot += i;
                res ++;
                if(tot>=n)
                    break;
            }
            s++;
        }
        return res;
    }
};
~~~

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)