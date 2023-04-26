
> 虽然一次周赛的几个题目说明不了太多问题，比如这个周赛的Hard题目就是板子题，算不上Hard，也许把第三题和第四题的顺序换一下比较合适。但是，GPT4的表现已经严重超出了我的预期。对于这次周赛的四个题目，GPT4的表现如下：题目1: 错了一次，简单提示后修正了错误，2A了。题目2: 1A。题目3: 无法通过提示的方法让GPT4做对，WA。题目4: 1A。不得不说，GPT4在模拟，模板题方面已经具备了不错的理解和处理能力，感觉在更强大的数据和更大模型的加持下以后大语言模型的做题能力能赶上一大半人类。｜ 从提升算法能力的角度来说，我不建议任何读者使用GPT4来做Leetcode。

# 0x0. 前言
GPT4论文（https://cdn.openai.com/papers/gpt-4.pdf）的第4节展示了GPT4的各种能力，在表格的最后三行展示了GPT4做Leetcode的能力，我比较感兴趣，所以本文打算来探索一下这种能力。看一下GPT4配合一个只发出prompt的人的表现如何。

![在这里插入图片描述](https://img-blog.csdnimg.cn/d5137298f4864e6cba3023b672bdb24a.png)
为了公平起见，我这里选取了LeetCode第 102 场双周赛（https://leetcode.cn/contest/biweekly-contest-102/）也就是2023年4月15日的这一场。我打算用GPT4来尝试解开这场周赛的4道题目，但是不一定能都解开，只是测试一下GPT4的写算法的能力。

我将全程只指挥GPT4写代码来解题，我自己不做任何的Coding工作。

先建立一个GPT4的新对话：

![在这里插入图片描述](https://img-blog.csdnimg.cn/97fa1096573b42dcbc9466bbdad60a85.png)


# 0x1. 第一题

第一题是个Easy的题目，描述如下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/216d70d145714e60b47d7471be8ba80a.png)
接下来，我们先把题面输入到GPT4。
![在这里插入图片描述](https://img-blog.csdnimg.cn/0a38679df3904289be54fe8d9516e871.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/5dadc88dc3ca4ef3bc0a9bf549b820fa.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8a65857efab4421faef2d1b2f6845b1d.png)
GPT4给了我们一个回复，感觉挺对的，但是这种格式不能让Leetcode直接通过，我们再让GPT4更新一下格式：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e371df3652164f08932b27ba6d84261d.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/a3581bf83ad640a487659eb6e26209d1.png)
接下来就是紧张的时刻，我们把这个类的代码提交给Leetcode。

![在这里插入图片描述](https://img-blog.csdnimg.cn/e1013a68ced04b65a0a4d087a4d46794.png)
结果错误，Leetcode返回了错误的例子。我们把这个错误的例子再返回给GPT4让它自己debug。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0e56c30bde254e7c9d125006d3bf7cc9.png)
我们把它修正后的代码提交给Leetcode。

![在这里插入图片描述](https://img-blog.csdnimg.cn/fb6b4a16c5164d899913a3e9267de279.png)
现在GPT4顺利通过了第一道题目。

# 0x2. 第二题

![在这里插入图片描述](https://img-blog.csdnimg.cn/59201d3785a74110b4207ca9d8b7b5fb.png)

我们问一下GPT4
![在这里插入图片描述](https://img-blog.csdnimg.cn/89718b157bd945d58158807282b67338.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/1d61c6fdcf5a499c9b33c95cd87fdf11.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3f3eb22a3f0e495093ecc6b4a9dc9cfc.png)


我们提交给Leetcode试试。

![在这里插入图片描述](https://img-blog.csdnimg.cn/0412957c76c84e9a87db62f31f0fd4ba.png)
直接通过，GPT4真有点强。



# 0x3. 第三题

![在这里插入图片描述](https://img-blog.csdnimg.cn/cfba733e53ef4364847ab63effd11451.png)
问问GPT4：

![在这里插入图片描述](https://img-blog.csdnimg.cn/e1611fd9861e40d9a7956d4e5e67075f.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/87f70c6ca79946c9908dfe301ac0dac4.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3c6ab7b1609345a98d563a42e691b00a.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c9f9f06e782441a7998d9af1719555a7.png)
这个问题感觉还是比较麻烦的，GPT4给出的方案感觉一眼假，不过我们不妨运行一下它给的代码。
![在这里插入图片描述](https://img-blog.csdnimg.cn/6e9f06e8ec304280ab6984dfeb249020.png)
我们发现编译就报错了。我们返回这个结果给GPT4：

![在这里插入图片描述](https://img-blog.csdnimg.cn/03974b3d588a44329f5c496e5f5975b2.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/8853020b1101465fa04910ddc9d2e38c.png)
现在确实可以编译了，但我们发现这个程序连样例都无法通过。显然，GPT4根本没有理解这道题目的意思，结果倾向于是“胡说八道”。再加强一些提示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/ab20607b56ee4dd0827bee1e8584a550.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/0e2d0c3ce00b41d19e9236d273c80fdc.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/c6e96f7cdeed41f4aefd36e9c7e56ce6.png)
最后GPT4输出的代码仍然无法通过样例。这道题，GPT4应该是无能为力了。

从这道题目，我们可以看到GPT4的局限性，那就是对于这种思维性的题目，GPT4很难理解这道题目暗含的意思，设计的算法也是错误的。

题解可以参考：https://leetcode.cn/problems/cousins-in-binary-tree-ii/solution/bfssuan-liang-ci-pythonjavacgo-by-endles-b72a/
# 0x4. 第4题

接下来我们看一下第4题，这个题是比较模板的题，我们看一下GPT4能否解开。

![在这里插入图片描述](https://img-blog.csdnimg.cn/486c033b606f41caa313ff8f9c5cb0aa.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/bb5816ed01b24ddfa18d3185ed33e0e0.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/65695ee2336a4cfba25d637e8b7d1b4d.png)

![在这里插入图片描述](https://img-blog.csdnimg.cn/6daf3abd6b884fe4877f7d8197dce0e1.png)

我们来看一下GPT4给出的答案能否通过样例。

![在这里插入图片描述](https://img-blog.csdnimg.cn/c6190af2b7b04f04b0f313be97d1a0b9.png)
是可以通过给出的样例的。我们提交试试。

![在这里插入图片描述](https://img-blog.csdnimg.cn/59355866df644de79a706cfe22bfd5bc.png)
这道题GPT4的完整代码实现如下：

```cpp
class Graph {
public:
    Graph(int n, vector<vector<int>>& edges) {
        this->n = n;
        for (const auto& edge : edges) {
            adjList[edge[0]].push_back({edge[1], edge[2]});
        }
    }
    
    void addEdge(vector<int> edge) {
        adjList[edge[0]].push_back({edge[1], edge[2]});
    }
    
    int shortestPath(int node1, int node2) {
        vector<int> dist(n, INT_MAX);
        dist[node1] = 0;

        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, node1});

        while (!pq.empty()) {
            int currNode = pq.top().second;
            int currDist = pq.top().first;
            pq.pop();

            if (currNode == node2) return currDist;

            if (currDist > dist[currNode]) continue;

            for (const auto& neighbor : adjList[currNode]) {
                int nextNode = neighbor.first;
                int edgeCost = neighbor.second;

                if (currDist + edgeCost < dist[nextNode]) {
                    dist[nextNode] = currDist + edgeCost;
                    pq.push({dist[nextNode], nextNode});
                }
            }
        }
        return -1;
    }

private:
    int n;
    unordered_map<int, vector<pair<int, int>>> adjList;
};

```

虽然这道题很难说是Hrad的难度，但1A真的太吓人了。。。

# 0x5. 总结+预测
虽然一次周赛的几个题目说明不了太多问题，比如这个周赛的Hard题目就是板子题，算不上Hard，也许把第三题和第四题的顺序换一下比较合适。但是，GPT4的表现已经严重超出了我的预期。

对于这次周赛的四个题目，GPT4的表现如下：

题目1: 错了一次，简单提示后修正了错误，2A了。
题目2: 1A
题目3: 无法通过提示的方法让GPT4做对，WA。
题目4: 1A。

不得不说，GPT4在模拟，模板题方面已经具备了不错的理解和处理能力，感觉在更强大的数据和更大模型的加持下以后大语言模型的做题能力能赶上一大半人类。｜ 从提升算法能力的角度来说，我不建议任何读者使用GPT4来做Leetcode。

