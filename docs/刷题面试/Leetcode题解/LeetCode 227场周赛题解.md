【GiantPandaCV导语】这是LeetCode的第227场周赛的题解，本期考察的知识点有**暴力，字符串，二进制枚举**等等。 

# 比赛链接

- https://leetcode-cn.com/contest/weekly-contest-227/

# 题目一：检查数组是否经排序和轮转得到

![题面](https://img-blog.csdnimg.cn/2021020819180183.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：数组是由非递减的源数组轮转得到，那么其结果只有三种可能，分别是非递增、非递减和两段非递增，直接判断一下即可。

**时间复杂度**：$O(n)$

**解题代码**如下：

~~~c
class Solution {
public:
    bool check(vector<int>& nums) {
        int cnt=0;
        for(int i=1;i<nums.size();i++)
            if(nums[i]<nums[i-1])
                cnt++;
        if(cnt==0)
            return true;
        else if(cnt>=2)
            return false;
        else if(nums.back()<=nums[0])
            return true;
        return false;
    }
};
~~~

# 题目二：移除石子的最大得分

![题面](https://img-blog.csdnimg.cn/20210208191736798.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：先对a、b、c从小到大排序，简单推导一下就可以发现，如果a+b<c，那么此时只能得到a+b的分数，否则即可把全部石子取完。

**时间复杂度**：$O(常数)$

**解题代码**如下：

~~~c
class Solution {
public:
    int maximumScore(int a, int b, int c) {
        vector<int> v={a,b,c};
        sort(v.begin(),v.end());
        if(v[0]+v[1]>=v[2])
            return (a+b+c)/2;
        else
            return v[0]+v[1];
    }
};
~~~

# 题目三：构造字典序最大的合并字符串 

![题面](https://img-blog.csdnimg.cn/20210208191714333.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：根据题意直接合并即可，在合并的时候如果两个字符串的首字母不相同，直接取大的；如果首字母相同，此时就要一直往后比较，直到比较到不同的时候，取较大字符串的首字母。这里直接使用字符串进行比较。

**时间复杂度**：$O(n^2)$

**解题代码**如下：

~~~c
class Solution {
public:
    string largestMerge(string word1, string word2) {
        string ans="";
        while(word1.size()||word2.size())
        {
            if((word1.size()&&word2.size()&&word1>word2)||word2.size()==0)
            {
                 ans += word1[0];
                 word1=word1.substr(1);
            }
            else
            {
                ans += word2[0];
                word2=word2.substr(1);
            }
        }
        return ans;
    }
};
~~~

# 题目四：最接近目标值的子序列和

![题面](https://img-blog.csdnimg.cn/20210208191637530.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：题目很简单，给出一个数组，选出子序列求和的值最接近Goal，对于每个数字也就两种状态:选与不选，数组的大小为40个，如果直接枚举可能会超时(可以尝试加一些剪枝)。这里把这个数组分为两半，每一部分就只有20个，现在就可以采用二进制枚举这20个数字，将两部分的结果采用二分查找就可以了，注意在每个部分要单独维护一下答案，在二分查找的时候需要向前后一位分别维护答案。

**时间复杂度**：$O(n/2*2^{(n/2)}*log(2^{(n/2)}))$

**解题代码**如下：

~~~c
class Solution {
public:
    int minAbsDifference(vector<int>& a, int b) {
        int l=a.size()/2,r=a.size()-l,ans = 1e9;
        int max_l = (1<<l)-1,max_r=(1<<r)-1;
        vector<int> v;
        for(int i=0;i<=max_l;i++) 
        {
            int tmp=0;
            for(int j=0;j<l;j++) 
                if((1<<j)&i) 
                    tmp+=a[j];
            ans=min(ans,abs(tmp-b));
            v.push_back(tmp);
        }
        sort(v.begin(), v.end());
        for(int i=0;i<=max_r;i++)
        {
            int tmp=0;
            for(int j=0;j<r;j++)
                if((1<<j)&i) 
                    tmp+=a[l+j];
            ans=min(ans,abs(tmp-b));
            int pos=lower_bound(v.begin(), v.end(), b-tmp)-v.begin();
            for(int j=pos-1;j<=pos+1;j++)
                if(j>=0&&j<v.size()) 
                    ans=min(ans, abs(v[j]+tmp-b));
        }
        return ans;
    }
};
~~~

-----------------------------------------------------------------------------------------------
欢迎关注GiantPandaCV, 在这里你将看到独家的深度学习分享，坚持原创，每天分享我们学习到的新鲜知识。( • ̀ω•́ )✧

有对文章相关的问题，或者想要加入交流群，欢迎添加BBuf微信：

![二维码](https://img-blog.csdnimg.cn/20200110234905879.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2p1c3Rfc29ydA==,size_16,color_FFFFFF,t_70)

为了方便读者获取资料以及我们公众号的作者发布一些Github工程的更新，我们成立了一个QQ群，二维码如下，感兴趣可以加入。

![公众号QQ交流群](https://img-blog.csdnimg.cn/20200517190745584.png#pic_center)