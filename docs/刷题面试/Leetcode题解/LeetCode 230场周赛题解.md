【GiantPandaCV导语】这是LeetCode的第230场周赛的题解，本期考察的知识点有**暴力，搜索，贪心，单调栈**等等。 

# 比赛链接

- https://leetcode-cn.com/contest/weekly-contest-230/

# 题目一： 统计匹配检索规则的物品数量

![](https://img-blog.csdnimg.cn/20210228213629672.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：先用一个flag记录下ruleKey是哪一种类型，然后根据ruleValue判断一下即可。**时间复杂度**：$O(n)$**解题代码**如下：

~~~c
class Solution {
public:
    int countMatches(vector<vector<string>>& items, string ruleKey, string ruleValue) {
        int flag=0;
        if(ruleKey=="color")
            flag=1;
        else if(ruleKey=="name")
            flag=2;
        int ans=0;
        for(int i=0;i<items.size();i++)
            if(ruleValue==items[i][flag])
                ans++;
        return ans;
    }
};
~~~

# 题目二：最接近目标价格的甜点成本

![](https://img-blog.csdnimg.cn/20210228213658433.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：首先看数据范围基料和配料都最多只有10种，而对于每种配料只有三种可能：不放，放一份，放两份。这里就直接采取暴力搜索把配料的可能性都存下来，然后直接暴力枚举基料与所有配料的结果，维护答案即可。**时间复杂度**：$O(n*3^n)$**解题代码**如下：

~~~c
class Solution {
public:
    vector<int> tot;
    void count(vector<int>& toppingCosts,int pos,int tmp)
    {
        tot.push_back(tmp);
        if(pos==toppingCosts.size())
            return ;
        for(int i=0;i<=2;i++)
            count(toppingCosts,pos+1,tmp+toppingCosts[pos]*i);
    }
    int closestCost(vector<int>& baseCosts, vector<int>& toppingCosts, int target) {
        count(toppingCosts,0,0);
        int ans=9999999;
        for(int i=0;i<baseCosts.size();i++)
            for(int j=0;j<tot.size();j++)
            {
                int now=baseCosts[i]+tot[j];
                if(abs(now-target)<abs(ans-target)||(abs(now-target)==abs(ans-target)&&now<ans))
                    ans=now;
            }
        return ans;
    }
};
~~~

# 题目三： 通过最少操作次数使数组的和相等



![](https://img-blog.csdnimg.cn/2021022821372385.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：首先判断不可能的情况，如果两个数组长度大于6倍，那么一定不可能，否者就是存在答案的，然后来判断最少操作次数，假设A数组的和大于B数组的和，那么对于A数组需要对里面的数进行减小，对B数组的数进行增大，每次操作即可缩小两个数组间的差值，为了使得操作次数最小化，每次操作都需要贪心选取缩小的差值最大化，那么对两个数组维护一下差值最大化，然后排个序，从大到小枚举直到数组间的差值小于0。

**时间复杂度**：$O(n*logn)$**解题代码**如下：

~~~c
class Solution {
public:
    int minOperations(vector<int>& nums1, vector<int>& nums2) {
        if(min(nums1.size(),nums2.size())*6<max(nums1.size(),nums2.size()))
            return -1;
        int ans1=0,ans2=0;
        for(int i=0;i<nums1.size();i++)
            ans1+=nums1[i];
        for(int i=0;i<nums2.size();i++)
            ans2+=nums2[i];
        vector<int> v;
        if(ans1>ans2)
            swap(nums1,nums2);
        for(int i=0;i<nums1.size();i++)
            v.push_back(abs(6-nums1[i]));
        for(int i=0;i<nums2.size();i++)
            v.push_back(abs(nums2[i]-1));
        int tot=abs(ans1-ans2),pos=0;
        sort(v.rbegin(),v.rend());
        while(tot>0)    
            tot -= v[pos++];
        return pos;
    }
};
~~~

# 题目四：车队 II

![](https://img-blog.csdnimg.cn/2021022821374542.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM0NTQyOTAz,size_16,color_FFFFFF,t_70)

**解题思路**：给出了n个车的起始位置和初速度，每个车的方向相同，如果车辆相遇会合并成一个车队，此时车队的速度为车队里的最慢速度。由于是求每辆车与下一辆车的相遇时间，那么首先想到的是能否与下一辆车相遇，那么就需要这辆车的速度大于下一辆车，否则就需要下一辆车被相遇后拖慢，除此之外就不能相遇，那么对于这辆车与后面相遇，对于前面的车是没有关系的，我们用一个单调栈来从左往右存车，车速逐渐降低，栈底最慢，栈顶最快，那么对于这辆车，如果栈顶快，就只能一直pop到慢车才能相遇，此时栈顶就是慢车了，如果这辆车不会消失，那么就肯定能相遇，如果它会消失，则是需要计算在它消失之前能否追上它这辆车，如果能追上，那么就与这辆车相遇，否则那就只有继续pop判断下一辆车。**时间复杂度**：$O(n)$**解题代码**如下：

~~~c
class Solution {
public:
    vector<double> getCollisionTimes(vector<vector<int>>& cars) {
        vector<double>ans(cars.size());
        stack<int>S;
        for(int i=cars.size()-1;i>=0;i--)
        {
            while(S.size())
            {
                if(cars[S.top()][1]>=cars[i][1])  //栈顶车速太快追不上，判断下一辆车
                    S.pop();
                else
                {
                    if(ans[S.top()]<0)     //还存在
                        break;
                    if(ans[S.top()]*(cars[i][1]-cars[S.top()][1])>cars[S.top()][0]-cars[i][0]) //已经消失了，但是在消失前就追上了
                        break;
                    //否则与这辆车就遇不到，那就看下一辆车
                    S.pop();
                }
            }
            //维护这辆车的答案
            if(S.empty())
                ans[i]=-1;
            else
                ans[i]=(double(cars[S.top()][0]-cars[i][0]))/(cars[i][1]-cars[S.top()][1]);
            S.push(i);
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