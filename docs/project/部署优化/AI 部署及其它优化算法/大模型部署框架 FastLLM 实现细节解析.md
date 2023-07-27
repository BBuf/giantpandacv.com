![](https://img-blog.csdnimg.cn/dda3b72563cc46e19542cb106f2f7cf8.png)


# 0x0. 前言
接着 [大模型部署框架 FastLLM 简要解析](https://mp.weixin.qq.com/s/4Cws_gXUuGXbyURIr0SdGw) 这篇文章首先梳理了一下FastLLM的调用链和关键的数据结构，然后解析了 FastLLM 的一些实现细节和CPU/GPU后端实现采用的优化技巧。


# 0x1. 调用链和数据结构解析
以chatglm-6b的支持为例，函数入口在 https://github.com/ztxz16/fastllm/blob/master/src/models/chatglm.cpp#L626 ，这里的 `input` 就是输入的 context（string类型）。然后 https://github.com/ztxz16/fastllm/blob/master/src/models/chatglm.cpp#L633 这行代码对 `input` 进行 tokenizer encode并构造好`inputIds`，再构造好`attentionMask`之后就可以给Forward函数推理，拿到推理结果之后再使用tokenizer进行decode得到输出。

在这里，`inputIds`和`attentionMask`都是Data数据类型，类比于PyTorch的Tensor，来对输入数据以及device，shape等信息进行统一管理。下面的代码展示了Data数据结构的定义，源码在：https://github.com/ztxz16/fastllm/blob/master/include/fastllm.h#L201-L286

```cpp
class Data {
    public:
        bool lockInCPU = false; // 如果lock在CPU上，那么不允许移动到其余设备
        WeightType weightType = WeightType::NONE; // 权重类型，NONE代表非权重（或未知权重）

        DataType dataType = DataType::FLOAT32; // 数据类型
        int unitSize, unitSizeDiv = 1; // 单个元素的字节数 = unitSIze / unitSizeDiv

        std::vector <int> dims; // 数据形状
        std::vector <uint64_t> strides; // 跨度

        uint64_t expansionSize = 0; // 扩容后的尺寸
        uint64_t expansionBytes = 0; // 扩容后的字节数
        std::vector <int> expansionDims; // 预扩容的形状
        uint8_t *cpuData = nullptr; // 数据指针

	    void *cudaData = nullptr;
        std::vector <void*> extraCudaData;

        void *deviceData = nullptr;
        std::vector <void*> extraDeviceData;

        DataDevice dataDevice = DataDevice::CPU;

        // 这两个参数用于量化，对FLOAT数据不适用
        int perChannelAxis = -1; // 沿哪个轴分通道量化，-1代表没有分通道
        std::vector <LowBitConfig> perChannelsConfigs; // perChannelsConfigs[i]代表第i个通道的min, max; 如果没有分通道，perChannelsConfigs[0]代表全局min, max
        std::vector <float> scales, mins;
        std::vector <int> zeros;
        std::vector <int> weightSum; // 作为权重时，有时候需要存一些和加速计算

        std::string fileName;
        long long filePos;
        std::shared_ptr<FileMmap> m_file;

        Data () {};

        Data (DataType type);

        Data (DataType type, const std::vector <int> &dims); // 构造函数

        // 构造函数，创建好之后从data复制数据
        // data中是原始数据，如果type不是float那么需要量化
        Data (DataType type, const std::vector <int> &dims, const std::vector <float> &data);

        ~Data(); // 析构函数

        Data (const Data &ori); // 深拷贝

        void CopyFrom(const Data &ori); // 复制

        uint64_t GetBytes() const; // 获取总字节数

        void Allocate(); // 分配内存

        void Allocate(float v); // 分配内存并初始化

        void Expansion(const std::vector <int> &dims); // 预扩容到相应尺寸

        void MallocSpace(uint64_t size); // 在设备上分配

        void FreeSpace(); // 回收设备上的内存

        void UpdateUnitSize(); // 更新unitSize

        void Resize(const std::vector <int> &dims); // 更改尺寸

        void Reshape(const std::vector <int> &dims); // 更改尺寸,但不修改数据

        uint64_t Count(int i) const; // dims[i] * strides[i]

        void PrintShape() const; // 输出形状

        void Print() const; // 输出

        void CalcWeightSum(); // 计算WeightSum

        void ToDevice(DataDevice device); // 移动到指定device

        void ToDevice(void *device);

        void set_file(std::shared_ptr<FileMmap> file) {
            m_file = file;
        }
    };
```

在Forward函数里面，以Data为核心载体，运行chatglm-6b模型的流程，具体包含如下的一些算子：https://github.com/ztxz16/fastllm/blob/master/include/fastllm.h#L346-L408 。以Permute为例我们浏览下它的实现：

```cpp
void Permute(const Data &input, const std::vector<int> &axis, Data &output) {
        Data axisData = Data(DataType::INT32PARAM, {(int)axis.size()});
        axisData.Allocate();
        for (int i = 0; i < axisData.Count(0); i++) {
            ((int32_t*)axisData.cpuData)[i] = axis[i];
        }
        curExecutor->Run("Permute", {
                {"input", (Data*)&input}, {"axis", &axisData}, {"output", (Data*)&output}
        }, {}, {});
    }
```
这里的curExecutor负责根据FastLLM编译开启的后端选项把算子Dispatch到不同的device进行执行，`{"input", (Data*)&input}, {"axis", &axisData}, {"output", (Data*)&output}}` 这行代码表示的是一个DataDict对象，也就是一个值为data的字典，原始定义为`typedef std::map <std::string, Data*> DataDict;`。接着我们看一下curExecutor的定义和实现：

```cpp

namespace fastllm {
    class Executor {
    private:
        std::vector <BaseDevice*> devices;
        std::map <std::string, float> profiler;

    public:
        Executor (); // 创建默认的Executor

        ~Executor(); // 析构

        void ClearDevices(); // 清空 devices

        void AddDevice(BaseDevice *device); // 增加一个device

        // 运行一个op
        void Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                 const fastllm::IntDict &intParams);

        void ClearProfiler();

        void PrintProfiler();
    };
}
```

从Executor类的定义我们可以判断它负责了在设定的devices上根据opType和输入数据等执行Op的前向计算，也就是Run这个接口。由于Executor类是FastLLM的调度核心实现，所以我们来详细解析一下它的实现。

```cpp
namespace fastllm {
    Executor::Executor() {
        this->devices.clear();
#ifdef USE_CUDA
        // 将一个指向 CudaDevice 类对象的指针插入到 devices 向量的末尾。
        // 这里通过 new 运算符创建了一个 CudaDevice 对象，并将返回的指针进行类型转换为 BaseDevice* 类型。
        this->devices.push_back((BaseDevice*) new CudaDevice());
#endif
        this->devices.push_back((BaseDevice*) new CpuDevice());
    }

    Executor::~Executor() {
        // 释放 devices 向量中的每个指针元素所占用的内存。
        for (int i = 0; i < devices.size(); i++) {
            delete devices[i];
        }
    }

    void Executor::ClearDevices() {
        // this->devices 指的是当前对象的 devices 成员，即指向 BaseDevice 类对象的指针向量。
        this->devices.clear();
    }
    
    // 该函数用于向 devices 向量中添加一个指向 BaseDevice 类对象的指针。
    void Executor::AddDevice(fastllm::BaseDevice *device) {
        this->devices.push_back(device);
    }

    void Executor::Run(const std::string &opType, const fastllm::DataDict &datas, const fastllm::FloatDict &floatParams,
                       const fastllm::IntDict &intParams) {
        // 创建一个 st 变量，用于记录函数开始执行的时间。
        auto st = std::chrono::system_clock::now();
        // 创建一个布尔变量 lockInCPU，用于记录是否将数据锁定在 CPU 上。
        bool lockInCPU = false;
        // 在第一个 for 循环中，遍历数据字典 datas，查找是否有 "___batch" 后缀的参数，
        // 并根据情况设置 lockInCPU 的值。it.first 是数据字典中的键（key），it.second 
        // 是对应的值（value）。如果存在 "___batch" 后缀的参数，则将 lockInCPU 设置为
        // 对应数据的 lockInCPU 属性（布尔值），否则设置为当前数据的 lockInCPU 属性。
        for (auto &it: datas) {
            if (intParams.find(it.first + "___batch") != intParams.end()) {
                int batch = intParams.find(it.first + "___batch")->second;
                for (int i = 0; i < batch; i++) {
                    lockInCPU |= ((Data**)it.second)[i]->lockInCPU;
                }
            } else {
                lockInCPU |= it.second->lockInCPU;
            }
        }
        // 第二个 for 循环遍历 devices 向量中的所有设备指针 device。
        // 在循环中，首先检查 lockInCPU 是否为真，并且当前设备的类型不是 "cpu"，
        // 如果是，则跳过当前设备（continue）。这个检查是为了保证数据锁定在 CPU 上时，只执行 CPU 设备上的操作。
        for (auto device: devices) {
            if (lockInCPU && device->deviceType != "cpu") {
                continue;
            }
            // 然后，通过调用 device->CanRun(opType, datas, floatParams, intParams) 
            // 检查当前设备是否可以运行指定的操作 opType。如果可以运行，则进行以下操作：
            if (device->CanRun(opType, datas, floatParams, intParams)) {
                // 第三个 for 循环遍历数据字典 datas，如果存在 "___batch" 后缀的参数，
                // 则将对应数据转移到当前设备上；否则，将当前数据转移到当前设备上。
                for (auto &it: datas) {
                    if (intParams.find(it.first + "___batch") != intParams.end()) {
                        int batch = intParams.find(it.first + "___batch")->second;
                        for (int i = 0; i < batch; i++) {
                            ((Data**)it.second)[i]->ToDevice((void *) device);
                        }
                    } else {
                        it.second->ToDevice((void *) device);
                    }
                }
                // 调用 device->Reshape(opType, datas, floatParams, intParams) 
                // 进行形状推导，device上的形状推导调用了opType对应的op的形状推导，
                // 并且被各个不同的op重写。
                device->Reshape(opType, datas, floatParams, intParams);
                // 对opType对应的这个算子进行推理。
                device->Run(opType, datas, floatParams, intParams);
                break;
            }
        }
        // 最后，计算操作运行时间，并将其加入 profiler 成员变量，用于性能分析。
        float spend = GetSpan(st, std::chrono::system_clock::now());
        profiler[opType] += spend;
    }
    
    // 清除profile的信息
    void Executor::ClearProfiler() {
        profiler.clear();
    }
    
    // 打印profile信息，也即输出每个层的运行时间和模型的总运行时间
    void Executor::PrintProfiler() {
        float sum = 0.0;
        for (auto &it : profiler) {
            printf("%s spend %f\n", it.first.c_str(), it.second);
            sum += it.second;
        }
        printf("total spend %f\n", sum);
    }
}
```

自此，前向计算就顺利完成了，再把推理结果给 tokenizer 解码就结束了，整体的调度执行流程是很简单明了的。

# 0x2. tokenizer 解析
接着，我们来解析一下tokenizer的实现。先看一下tokenizer的定义（https://github.com/ztxz16/fastllm/blob/master/include/fastllm.h#L287-L310）：

```cpp
struct Tokenizer {
        struct TrieNode {
            int tokenId;
            std::map <int, TrieNode*> next;
            TrieNode();
        };
        TrieNode *root;

        std::unordered_map <int, std::string> tokenToStringDict;

        Tokenizer ();

        ~Tokenizer();

        void Clear(); // 清空分词器

        void Insert(const std::string &s, int tokenId); // 插入一个token

        Data Encode(const std::string &s); // 编码

        std::string Decode(const Data &data); // 解码

        std::string DecodeTokens(const std::vector <int> &tokens); // 解码
    };
```

我们从实现来看tokenizer的细节：

```cpp
   // 这是 Tokenizer 类的嵌套结构 TrieNode 的构造函数的实现。
   // 在构造函数中，将 tokenId 成员变量的值初始化为 -999999。
   // 这个值在构造函数中被硬编码，它是作为一个特殊标记来使用的。
	 Tokenizer::TrieNode::TrieNode() {
        this->tokenId = -999999;
    }
    
    // Tokenizer 类的构造函数的实现。
    // 在构造函数中，通过 new 运算符创建一个新的 TrieNode 对象，
    // 并将其指针赋值给 root 成员变量。这样，构造函数创建了一个空的字典树，
    // 并将其根节点指针存储在 root 中。
    Tokenizer::Tokenizer() {
        root = new TrieNode();
    }
    
    // Tokenizer 类的析构函数的实现。
    // 在析构函数中，首先调用 Clear() 函数，用于释放动态分配的资源和清空数据。
    // 然后，调用 delete 运算符释放通过 new 运算符创建的 root 对象的内存，从而释放整个字典树的内存。
    Tokenizer::~Tokenizer() {
        Clear();
        delete root;
    }
    
    // 这是 Tokenizer 类的成员函数 Clear() 的定义，用于清空分词器并释放动态分配的资源。
    void Tokenizer::Clear() {
        // 创建一个指向 TrieNode 的指针向量 q，用于辅助遍历字典树。
        std::vector <TrieNode*> q;
        // 将字典树的根节点 root 加入 q 向量，作为遍历的起始点。
        q.push_back(root);
        // 开始遍历 q 向量中的节点，这是一个广度优先搜索（BFS）的过程。
        for (int i = 0; i < q.size(); i++) {
            // 取出当前遍历到的节点 now。
            TrieNode *now = q[i];
            // 对当前节点 now 的所有子节点进行遍历。
            for (auto it : now->next) {
                // 将当前节点 now 的子节点加入 q 向量中，以便继续遍历子节点的子节点。
                q.push_back(it.second);
            }
        }
        // 当遍历完成后，q 向量中包含了字典树中的所有节点。
        // 创建一个新的 TrieNode 对象，并将其指针赋值给 root 成员变量，表示创建了一个空的字典树。
        root = new TrieNode();
        //  清空 tokenToStringDict 映射表，以确保所有 token 的映射被清空。
        tokenToStringDict.clear();
    }
    
    // 这是 Tokenizer 类的成员函数 Insert 的定义，用于向分词器中插入一个 token。
    void Tokenizer::Insert(const std::string &s, int tokenId) {
        // 创建一个指向 TrieNode 的指针 now，并将其初始化为指向字典树的根节点 root。
        TrieNode *now = this->root;
        // 开始遍历输入的字符串 s 中的每个字符。
        for (int i = 0; i < s.size(); i++) {
            // 检查当前字符 s[i] 是否已经存在于当前节点 now 的 next 映射表中。
            // 如果当前字符 s[i] 不存在于当前节点 now 的子节点中，
            // 在 now->next 中添加新的子节点，该子节点的键为当前字符 s[i] 的编码值，
            // 值为指向新创建的 TrieNode 对象的指针。这表示在字典树中添加了一个新的字符节点。
            if (now->next.find(s[i]) == now->next.end()) {
                now->next[s[i]] = new TrieNode();
            }
            // 将 now 移动到下一个字符 s[i] 对应的节点，以便继续处理下一个字符。
            now = now->next[s[i]];
        }
        // 遍历完成后，now 将指向字典树中最后一个字符的节点。
        // 设置当前节点的 tokenId 成员变量，表示当前节点代表一个 token，
        // 并使用传入的 tokenId 值来标识该 token。
        now->tokenId = tokenId;
        // 将传入的 tokenId 和对应的字符串 s 添加到 tokenToStringDict 
        // 映射表中，用于后续的解码过程。
        tokenToStringDict[tokenId] = s;
    }
    
    // 这是 Tokenizer 类的成员函数 Encode 的定义，用于对输入的字符串 s 进行编码。
    Data Tokenizer::Encode(const std::string &s) {
        // 创建一个浮点数向量 v，用于存储编码结果。该向量将存储找到的 token 对应的 tokenId 值。
        std::vector <float> v;
        // 开始遍历输入的字符串 s 中的每个字符。
        for (int i = 0; i < s.size(); i++) {
            // 创建两个整数变量 tokenId 和 pos，
            // 用于记录找到的 token 的 tokenId 值和 token 的结束位置。
            int tokenId = -999999, pos = i - 1;
            // 创建一个指向 TrieNode 的指针 now，并将其初始化为指向字典树的根节点 root。
            TrieNode *now = this->root;
            // 从当前字符 s[i] 开始继续遍历字符串 s。
            for (int j = i; j < s.size(); j++) {
            		// 检查当前字符 s[j] 是否存在于当前节点 now 的 next 映射表中。
            		// 如果存在，表示当前字符构成了一个 token 的一部分，继续遍历子节点。
                if (now->next.find(s[j]) != now->next.end()) {
                    // 将 now 移动到下一个字符 s[j] 对应的节点。
                    now = now->next[s[j]];
                    // 检查当前节点 now 是否代表一个 token，即它的 tokenId 是否有效。
                    if (now->tokenId != -999999) {
                        // 如果当前节点代表一个 token，将 tokenId 和当前位置 j 存储到 
                        // tokenId 和 pos 变量中，以便记录找到的 token 的信息。 
                        tokenId = now->tokenId;
                        pos = j;
                    }
                } else { // 如果当前字符不再是 token 的一部分，退出内层循环，继续外层循环。
                    break;
                }
            }
            // 如果 pos 大于等于当前位置 i，表示找到了一个 token。
            // 这里 pos 存储了找到的 token 的结束位置，i 移动到 pos 处，以便继续遍历下一个字符。
            if (pos >= i) {
                i = pos;
                v.push_back(tokenId);
                //printf("%d ", tokenId);
            }
        }
        //printf("\n");
        // 遍历完成后，v 向量中存储了输入字符串中所有找到的 token 对应的 tokenId 值。
        // 创建一个 Data 对象并返回，表示编码的结果。这里 Data 是一个数据结构，
        // 用于存储数据及其相关信息。编码结果是一个一维浮点数数组，
        // 表示输入字符串中所有找到的 token 对应的 tokenId 值。
        return Data (DataType::FLOAT32, {1, (int)v.size()}, v);
    }
    
    // 这是 Tokenizer 类的成员函数 DecodeTokens 的定义，
    // 用于对输入的 token 数组进行解码，将 token 转换回原始的字符串。
    std::string Tokenizer::DecodeTokens(const std::vector<int> &tokens) {
        // 创建一个空字符串 ret，用于存储解码结果。
        std::string ret = "";
        // 开始遍历输入的 token 数组 tokens。
        for (int i = 0; i < tokens.size(); i++) {
            // 获取当前 token 对应的原始字符串 s，通过查询 tokenToStringDict 映射表，
            // 将 tokens[i] 转换回字符串。
            std::string s = tokenToStringDict[tokens[i]];
            // 判断当前 token 是否需要特殊处理：
            // 如果 s 是类似 "<0xHH>" 格式的 token（其中 HH 表示十六进制数），
            // 则需要将其转换为对应的字符。首先，提取 HH，然后将其转换为对应的字符，
            // 并用空格代替原始的 token。
            if (s.size() == 6 && s.substr(0, 3) == "<0x" && s.back() == '>') {
                int c = 0;
                for (int i = 3; i < 5; i++) {
                    c *= 16;
                    if (s[i] >= '0' && s[i] <= '9') {
                        c += (s[i] - '0');
                    } else {
                        c += (s[i] - 'A' + 10);
                    }
                }

                s = " ";
                s[0] = c;
            }
            // 根据不同的 token 进行解码：
            if (s == "<n>") {
                ret += "\n";
            } else if (s == "<|tab|>") {
                ret += "\t";
            } else {
                ret += s;
            }
        }
        
        // 将特殊字符 "\xE2\x96\x81"（UTF-8 编码）替换为空格 " "，这是用于表示空格的特殊字符。
        std::string blank = "";
        blank += 226, blank += 150, blank += 129;
        while (true) {
            std::string::size_type pos(0);
            if ((pos = ret.find(blank)) != std::string::npos)
                ret.replace(pos, blank.length(), " ");
            else break;
        }
        // 检查是否有 "<|blank_数字>" 格式的特殊 token，如果有，将其解码成对应数量的空格字符。
        int pos = ret.find("<|blank_");
        if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }

        return ret;
    }

    std::string Tokenizer::Decode(const Data &data) {
        std::vector <int> tokens;
        for (int i = 0; i < data.Count(0); i++) {
            tokens.push_back((int) ((float *) data.cpuData)[i]);
        }
        return DecodeTokens(tokens);
    }

```

上面的：

```cpp
if (pos != -1) {
            int space_num = atoi(ret.substr(8, ret.size() - 10).c_str());
            return std::string(space_num, ' ');
        }
```

这行代码应该是有bug，假设 ret 的值为 "Hello<|blank_4>world!"，那么在解码时，pos 将是 8，而 space_num 将是 4。然后，函数将返回 " "，即包含四个空格字符的字符串。在这种情况下，特殊 token "<|blank_4>" 被成功解码成了四个空格字符，但是Hello和world!这部分被删掉了。所以最终的解码结果是不对的，需要修正一下。

对tokenizer的解析可以发现，在c++中使用字典树数据结构来实现tokenizer是相对比较简单方便的。

接下来，我们对CPU后端和GPU后端的算子实现进行解析。
# 0x3. CPU后端算子实现
主要就是对这个文件进行解析：https://github.com/ztxz16/fastllm/blob/master/src/devices/cpu/cpudevice.cpp 。

## 辅助函数

```cpp
	  // 这是 CpuDevice 类的成员函数 Malloc 的定义，用于在 CPU 上分配一块内存空间。
    bool CpuDevice::Malloc(void **ret, size_t size) {
        *ret = (void*)new uint8_t [size];
        return true;
    }
    
    // 这是 CpuDevice 类的成员函数 Free 的定义，用于在 CPU 上释放之前分配的内存。
    bool CpuDevice::Free(void *ret) {
        delete[] (uint8_t*)ret;
        return true;
    }
    
    // 这是 CpuDevice 类的成员函数 CopyDataFromCPU 的定义，用于将数据从 CPU 拷贝到指定的设备上。
    // 这里什么都不做，直接返回true。
    bool CpuDevice::CopyDataFromCPU(void *dst, void *src, size_t size) {
        return true;
    }
    
    // 这是 CpuDevice 类的成员函数 CopyDataToCPU 的定义，用于将数据从指定的设备拷贝到 CPU 上。
    bool CpuDevice::CopyDataToCPU(void *dst, void *src, size_t size) {
        return true;
    }

// 如果定义了 __AVX__ 和 __AVX2__，那么会启用第一个 DotU8U8 函数和 DotU4U8 函数。
// 如果只定义了 __AVX__，但没有定义 __AVX2__，那么会启用第二个 DotU8U8 函数和 DotU4U8 函数。

#ifdef __AVX__
#ifdef __AVX2__
    // 这是一段使用了 Intel AVX2 指令集（Advanced Vector Extensions 2）的代码，
    // 用于计算两个8位无符号整数数组的点积。
    // 定义了一个函数 DotU8U8，它接受两个指向 8 位无符号整数的指针 a 和 b，
    // 以及一个整数 n。这个函数的目的是计算数组 a 和 b 的点积，其中数组的长度为 n。
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        // 初始化一个 256 位的整数向量 acc，所有位都设置为零。这个向量用于存储点积的累加值。
        __m256i acc = _mm256_setzero_si256();
        //  初始化两个变量，i 用于循环计数，ans 用于存储最后的结果。
        int i = 0;
        int ans = 0;
        // 等这几行代码初始化了一些常量向量
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        const __m256i ones8 = _mm256_set1_epi8(1);
        const __m256i xors = _mm256_set1_epi8(-128);
        // 这是一个循环，每次处理 32 个元素。这是因为 AVX2 可以同时处理 32 个 8 位整数。
        for (; i + 31 < n; i += 32) {
            // 这两行代码从数组 a 和 b 中加载数据到 256 位的向量 bx 和 by。
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            
            // 这行代码将 by 中的每个元素减去 128，这对应于上面表达式中的 ((int)b[i] - 128)。
            by = _mm256_xor_si256(by, xors);
            // 这行代码对于那些原本是 0 的元素（在减去 128 后变为 -128 的元素）加 1，
            // 以避免后续乘法操作时的溢出。
            by = _mm256_add_epi8(by, _mm256_and_si256(_mm256_cmpeq_epi8(by, xors), ones8));
            
            //  这行代码将 bx 中的符号应用到 by 中，对应于上面表达式中的 ((int8_t*)a)[i]。
            by = _mm256_sign_epi8(by, bx);
            // 这行代码将 bx 中的所有非零元素变为 1，这是为了在后续的乘法操作中保持 by 中元素的原值。
            bx = _mm256_sign_epi8(bx, bx);
            
            // 这行代码先对 bx 和 by 进行乘法运算（这对应于上面表达式中的 * 操作），
            // 然后再与 acc 进行加法操作（这对应于上面表达式中的 += 操作）。
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(bx, by), ones));
        }
        // 这是另一个循环，用于处理数组中剩余的元素（数量小于 32）。
        // 这些元素通过常规的方式计算点积，然后累加到 ans 中。
        for (; i < n; i++) {
            ans += ((int8_t*)a)[i] * ((int)b[i] - 128);
        }
        
        // 最后，将 acc 中的所有元素相加，然后再加上 ans，返回最终的结果。
        return ans + I32sum(acc);
    };
#else
    // 定义了一个函数 DotU8U8，它接受两个指向 8 位无符号整数的指针 a 和 b，
    // 以及一个整数 n。这个函数的目的是计算数组 a 和 b 的点积，其中数组的长度为 n。
    int DotU8U8(uint8_t *a, uint8_t *b, int n) {
        // 初始化一个 256 位的整数向量 acc，所有位都设置为零。这个向量用于存储点积的累加值。
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        int ans = 0;
        // 这是一个循环，每次处理 32 个元素。这是因为 AVX 可以同时处理 32 个 8 位整数。
        for (; i + 31 < n; i += 32) {
            // 这两行代码从数组 a 和 b 中加载数据到 256 位的向量 bx 和 by。
            __m256i bx = _mm256_loadu_si256((const __m256i *) (a + i));
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            
            // 接下来的四行代码将 bx 和 by 中的 8 位整数扩展为 16 位整数。
            // 这是因为在后续的乘法和累加操作中，如果仍然使用 8 位整数，可能会发生溢出。
            __m256i mx0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 0));
            __m256i mx1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(bx, 1));

            __m256i my0 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 0));
            __m256i my1 = _mm256_cvtepu8_epi16(_mm256_extractf128_si256(by, 1));
            
            // 这两行代码首先对 mx0 和 my0，以及 mx1 和 my1 进行乘法累加操作，
            // 然后再与 acc 进行加法操作，结果存储在 acc 中。
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx0, my0));
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(mx1, my1));
        }
        //  这是另一个循环，用于处理数组中剩余的元素（数量小于 32）。
        // 这些元素通过常规的方式计算点积，然后累加到 ans 中。
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }
        
        // 最后，将 acc 中的所有元素相加，然后再加上 ans，返回最终的结果。
        return ans + I32sum(acc);
    };
#endif
    // 它接受两个指向 8 位无符号整数的指针 a 和 b，以及一个整数 n。
    // 这个函数的目的是计算数组 a 和 b 的点积，其中数组的长度为 n。
    int DotU4U8(uint8_t *a, uint8_t *b, int n) {
        // 初始化一个 256 位的整数向量 acc，所有位都设置为零。这个向量用于存储点积的累加值。
        __m256i acc = _mm256_setzero_si256();

        int i = 0;
        int ans = 0;
        // 初始化两个常量向量，lowMask 中的每个元素都是 0xf，ones 中的每个元素都是 1。
        const __m256i lowMask = _mm256_set1_epi8(0xf);
        const __m256i ones = _mm256_set1_epi16(1);
        for (; i + 31 < n; i += 32) {
            // 从数组 a 中加载 16 个元素到 128 位的向量 orix 中。
            // 这里 i / 2 的原因是每个元素实际上只有 4 位。
            __m128i orix = _mm_loadu_si128((const __m128i *) (a + i / 2));
            // 将 orix 中的元素分成高 4 位和低 4 位，然后将它们合并成一个 256 位的向量 bytex。
            __m256i bytex = _mm256_set_m128i(_mm_srli_epi16(orix, 4), orix);
            // 使用按位与操作，取 bytex 中的每个元素的低 4 位，结果存储在 bx 中。
            __m256i bx = _mm256_and_si256(lowMask, bytex);
            // 从数组 b 中加载数据到 256 位的向量 by。
            __m256i by = _mm256_loadu_si256((const __m256i *) (b + i));
            // 这行代码首先进行了两个向量的乘法累加操作，然后再与 acc 进行加法操作，结果存储在 acc 中。
            acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(by, bx), ones));
        }
        for (; i < n; i++) {
            ans += a[i] * b[i];
        }

        return ans + I32sum(acc);
    };
#endif
```

在启用AVX2进行点积计算时，有一个特殊的操作就是把b[i]转换为有符号的整数并减掉128。我没太懂这个操作的意义是什么，问了一下gpt4获得了如下的回答：

![在这里插入图片描述](https://img-blog.csdnimg.cn/fb09776de77048f1b8970aa629c2eb94.png)
然后这里有个疑问是在DotU4U8的实现中调用的指令应该是AVX2的指令集，但确是在AVX2宏关闭时调用的，不清楚这里是否会有bug。![在这里插入图片描述](https://img-blog.csdnimg.cn/2f05c09eaea94877bc3a5457871ba8a2.png)
上述函数中涉及到大量的intel Intrinsics指令细节，读者想详细了解可以参考官方文档：https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html 。

## CpuEmbedding 算子解析

```cpp
// CpuEmbedding 算子的形状推导函数，这个函数接受四个参数：
// 一个 std::string 类型的 opType，两个字典类型的 datas 和 floatParams，以及一个 intParams。
void CpuEmbedding::Reshape(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 这三行代码从 datas 字典中查找键为 "input"、"output" 和 "weight" 的元素，
        // 并将找到的元素的值赋给 input、output 和 weight。
        // 这里的 "input"、"output" 和 "weight" 可以理解为嵌入层的输入、输出和权重。
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        
        // 这行代码检查 weight 的维度数量是否为 2。如果不是，就会抛出一个错误。
        AssertInFastLLM(weight.dims.size() == 2, "Embedding's weight's dim should be 2.\n");
        // 这行代码检查 weight 的数据类型是否为 FLOAT32 或 BFLOAT16。如果不是，就会抛出一个错误。
        AssertInFastLLM(weight.dataType == DataType::FLOAT32 ||
                        weight.dataType == DataType::BFLOAT16, "Embedding's weight's type should be float32 or bfloat16.\n");
        // 这行代码检查 input 的数据类型是否为 FLOAT32。如果不是，就会抛出一个错误。
        AssertInFastLLM(input.dataType == DataType::FLOAT32, "Embedding's input's type should be float32.\n");
        
        // 这行代码将 weight 的 weightType 属性设置为 EMBEDDING。
        weight.weightType = WeightType::EMBEDDING;
        // 这行代码从 weight 的维度中提取词汇大小（vocabSize）和嵌入大小（embSize）。
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        // 这两行代码将 embSize 添加到 input 的维度中，形成一个新的维度。
        std::vector <int> dims = input.dims;
        dims.push_back(embSize);
        
        // 这两行代码将 output 的数据类型设置为 FLOAT32，并重新调整其维度。
        output.dataType = DataType::FLOAT32;
        output.Resize(dims);
    }
    
    // 这是一个名为 CpuEmbedding::Run 的函数，它在某个名为 CpuEmbedding 的类中被定义。
    // 这个函数接受四个参数：一个 std::string 类型的 opType，
    // 两个字典类型的 datas 和 floatParams，以及一个 intParams。
    // 这个函数的主要任务是执行嵌入层（Embedding layer）的运算。
    // 嵌入层通常用于将离散型特征（例如词汇）转换为连续的向量表示。
    // 具体的实现方法是，对于每个输入的索引，从权重矩阵中查找对应的行，
    // 然后将其复制到输出矩阵的对应位置。
    void CpuEmbedding::Run(const std::string &opType, const fastllm::DataDict &datas,
                               const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 这三行代码从 datas 字典中查找键为 "input"、"output" 和 "weight" 的元素，
        // 并将找到的元素的值赋给 input、output 和 weight。
        // 这里的 "input"、"output" 和 "weight" 可以理解为嵌入层的输入、输出和权重。
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);;

        output.Allocate(); // 这行代码为 output 分配内存。
        
        // 这行代码从 weight 的维度中提取词汇大小（vocabSize）和嵌入大小（embSize）。
        int vocabSize = weight.dims[0], embSize = weight.dims[1];
        // 这行代码计算 input 的长度。
        uint64_t inputLen = input.Count(0);
        // 这行代码获取 input 的数据，并将其转换为浮点数的指针。
        float *inputData = (float*)input.cpuData;
        
        // 接下来的代码根据内存模式和权重的数据类型的不同，分别处理了四种情况。
        // 这四种情况可以归纳为两个大类：内存模式和权重的数据类型。
        // 内存模式：如果 GetLowMemMode() 返回 true，则表示处于低内存模式。
        // 在这种模式下，权重数据不会一次性全部加载到内存中，而是每次只加载需要的部分。
        // 否则，权重数据会全部加载到内存中。
        if (GetLowMemMode()) {
            FILE *fi = fopen(weight.fileName.c_str(), "rb");
            // 权重的数据类型：如果权重的数据类型为 FLOAT32，则使用浮点数进行计算。
            // 如果权重的数据类型为 BFLOAT16，则使用 16 位浮点数进行计算。
            if (weight.dataType == DataType::FLOAT32) {
                float *outputData = (float *) output.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    // 这行代码从 inputData 中取出第 i 个元素，并将其四舍五入到最近的整数。
                    int token = (int) (inputData[i] + 1e-9);
                    // 这两行代码将文件指针移动到第 token 行的开始位置。
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(float) + weight.filePos, 0);
#endif
                    // 这行代码从文件中读取 embSize 个浮点数，并将它们存储在 outputData 的对应位置。
                    int ret = fread(outputData + i * embSize, sizeof(float), embSize, fi);
                }
            } else {
                // 如果权重的数据类型为 BFLOAT16，则使用 16 位浮点数进行计算。
                // 这部分代码的逻辑与 FLOAT32 部分的逻辑类似，只是多了一个步骤：
                // 将 16 位的浮点数转换为 32 位的浮点数。
                uint16_t *outputData = (uint16_t *) output.cpuData;
                uint16_t *weightData = new uint16_t[embSize];
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
#if defined(_WIN32) or defined(_WIN64)
                    _fseeki64(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#else
                    fseek(fi, (long long)token * embSize * sizeof(uint16_t) + weight.filePos, 0);
#endif
                    int ret = fread(weightData, sizeof(uint16_t), embSize, fi);
                    for (int j = 0; j < embSize; j++) {
                        outputData[i * embSize * 2 + j * 2] = 0;
                        outputData[i * embSize * 2 + j * 2 + 1] = weightData[j];
                    }
                }
                delete[] weightData;
            }
            // 最后，fclose(fi); 这行代码关闭了文件。
            fclose(fi);
        } else {
            if (weight.dataType == DataType::FLOAT32) {
                // 这两行代码获取 output 和 weight 的数据，并将它们转换为浮点数的指针。
                float *outputData = (float *) output.cpuData;
                float *weightData = (float *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
                    // 这行代码从 weightData 中复制 embSize 个浮点数到 outputData 的对应位置。
                    // 这里的 token 是索引，embSize 是嵌入向量的长度。
                    memcpy(outputData + i * embSize, weightData + token * embSize, embSize * sizeof(float));
                }
            } else {
                uint16_t *outputData = (uint16_t *) output.cpuData;
                uint16_t *weightData = (uint16_t *) weight.cpuData;
                for (int i = 0; i < inputLen; i++) {
                    int token = (int) (inputData[i] + 1e-9);
                    for (int j = 0; j < embSize; j++) {
                        outputData[i * embSize * 2 + j * 2] = 0;
                        outputData[i * embSize * 2 + j * 2 + 1] = weightData[token * embSize + j];
                    }
                }
            }
        }
    }
```

## CpuLayerNormOp 解析

```cpp
void CpuLayerNormOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                             const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        // 这四行代码从 datas 字典中查找键为 "input"、"output"、"gamma" 和 "beta" 的元素，
        // 并将找到的元素的值赋给 input、output、gamma 和 beta。
        // 这里的 "input" 是层归一化的输入，"output" 是输出，
        // "gamma" 和 "beta" 是用于对归一化后的结果进行缩放和移位的可学习参数。
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &gamma = *(datas.find("gamma")->second);
        Data &beta = *(datas.find("beta")->second);
        
        // 这行代码为 output 分配内存。
        output.Allocate();
        
        // 这行代码从 intParams 字典中查找键为 "axis" 的元素。
        // 如果找到，则使用找到的值作为归一化的轴；否则，使用默认值 -1。在层归一化中，轴通常是特征维度。
        int axis = intParams.find("axis") != intParams.end() ? intParams.find("axis")->second : -1;
        // 这两行代码计算 input 的维度数，并将 axis 转换为非负数。
        // 这是为了处理负数的轴值，因为在 Python 中，轴可以是负数，表示从后向前数的位置。
        int dimsLen = input.dims.size();
        axis = (axis % dimsLen + dimsLen) % dimsLen;
        
        // 这三行代码计算 outer、channels 和 inner。
        // outer 是归一化操作的外部维度的元素总数，channels 是归一化操作的轴的大小，
        // inner 是归一化操作的内部维度的元素总数。
        int outer = input.Count(0) / input.Count(axis);
        int channels = input.dims[axis];
        int inner = input.strides[axis];
        
        // 这行代码为 mean 和 var 分配内存，它们用于存储每个归一化组的均值和方差。
        float *mean = new float[inner], *var = new float[inner];
        float *inputData = (float *) input.cpuData;
        float *outputData = (float *) output.cpuData;
        float *gammaData = (float *) gamma.cpuData;
        float *betaData = (float *) beta.cpuData;
        
        // 在这个条件下，每个通道只有一个元素，所以可以并行地对每个通道进行层归一化。
        if (inner == 1) {
            // 这是一个循环，对 input 中的每一个外部元素进行处理。
            for (int i = 0; i < outer; i++) {
                // 这行代码定义了三个浮点数变量，分别用于存储均值、平方和和方差。
                float mean = 0.f, s2 = 0.f, var = 0.f;
                int j = 0;
                // 这是一段条件编译的代码，只有在目标平台为 ARM 架构时才会编译和执行。
                // 这段代码使用了 ARM 架构的 SIMD 指令来加速计算。
#ifdef __aarch64__
                float32x4_t sums = vdupq_n_f32(0.0);
                    float32x4_t sums2 = vdupq_n_f32(0.0);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t vi = vld1q_f32(inputData + j);
                        sums = vaddq_f32(sums, vi);
                        sums2 = vaddq_f32(sums2, vmulq_f32(vi, vi));
                    }
                    mean = sums[0] + sums[1] + sums[2] + sums[3];
                    s2 = sums2[0] + sums2[1] + sums2[2] + sums2[3];
#endif
#ifdef __AVX2__
                // 这是另一段条件编译的代码，只有在目标平台支持 AVX2 指令集时才会编译和执行。
                // 这段代码使用了 AVX2 的 SIMD 指令来加速计算。
                __m256 sum_vec = _mm256_setzero_ps();
                __m256 squared_sum_vec = _mm256_setzero_ps();

                for (; j < channels - 7; j += 8) {
                    __m256 data_vec = _mm256_loadu_ps(inputData + j);
                    sum_vec = _mm256_add_ps(sum_vec, data_vec);

                    __m256 squared_data_vec = _mm256_mul_ps(data_vec, data_vec);
                    squared_sum_vec = _mm256_add_ps(squared_sum_vec, squared_data_vec);
                }

                float sum_array[8];
                _mm256_storeu_ps(sum_array, sum_vec);
                mean = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3] +
                            sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

                float squared_sum_array[8];
                _mm256_storeu_ps(squared_sum_array, squared_sum_vec);
                s2 = squared_sum_array[0] + squared_sum_array[1] +
                                    squared_sum_array[2] + squared_sum_array[3] +
                                    squared_sum_array[4] + squared_sum_array[5] +
                                    squared_sum_array[6] + squared_sum_array[7];
#endif
                // 这是一个循环，对 input 中剩余的每一个通道进行处理。
                for (; j < channels; j++) {
                    mean += inputData[j];
                    s2 += inputData[j] * inputData[j];
                }
                // 这两行代码计算了均值和方差。
                mean /= channels;
                var = sqrt(s2 / channels - mean*mean + 1e-10);
                // 接下来是对output的每一个通道进行并行处理
                j = 0;
#ifdef __aarch64__
                float32x4_t means = vdupq_n_f32(mean);
                    float32x4_t vars = vdupq_n_f32(1.0 / var);
                    for (; j + 3 < channels; j += 4) {
                        float32x4_t va = vld1q_f32(gammaData + j), vb = vld1q_f32(betaData + j);
                        float32x4_t vi = vld1q_f32(inputData + j);
                        float32x4_t vo = vaddq_f32(vmulq_f32(vmulq_f32(vsubq_f32(vi, means), vars), va), vb);
                        vst1q_f32(outputData + j, vo);
                    }
#endif
                for (; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    outputData[j] = (inputData[j] - mean) / var * a + b;
                }
                
                // 这两行代码更新了 inputData 和 outputData 的指针位置，
                // 以便在下一轮循环中处理下一个外部元素。
                inputData += channels;
                outputData += channels;
            }
            return;
        } else {
            // 这段代码同样是执行层归一化（Layer Normalization）操作，但这次的操作更为通用，
            // 能处理 inner 不等于 1 的情况，即每个通道有多个元素的情况。
            // 这是一个循环，对 input 中的每一个外部元素进行处理。
            for (int i = 0; i < outer; i++) {
                // 这两行代码将 mean 和 var 数组的所有元素初始化为 0。
                std::fill(mean, mean + inner, 0.f);
                std::fill(var, var + inner, 0.f);
                // 这行代码定义了一个指针 inputWalk，指向 inputData。
                float *inputWalk = inputData;
                // 这是一个循环，对每个通道进行处理。
                for (int j = 0; j < channels; j++) {
                	  // 这是一个嵌套循环，对每个通道内的每个元素进行处理。
                    for (int k = 0; k < inner; k++) {
                        // 这行代码将当前元素的值加到对应的 mean 中，然后 inputWalk 指针向后移动。
                        mean[k] += *inputWalk++; 
                    }
                }
                // 这是另一个循环，计算每个通道的均值。
                for (int k = 0; k < inner; k++) {
                    mean[k] /= channels;
                }
                // 方差类似
                inputWalk = inputData;
                for (int j = 0; j < channels; j++) {
                    for (int k = 0; k < inner; k++) {
                        float x = (*inputWalk++) - mean[k];
                        var[k] += x * x;
                    }
                }
                for (int k = 0; k < inner; k++) {
                    var[k] = sqrt(var[k] / channels + 1e-5);
                }
                
                // 计算输出也是类似
                inputWalk = inputData;
                float *outputWalk = outputData;
                for (int j = 0; j < channels; j++) {
                    float a = gammaData[j], b = betaData[j];
                    for (int k = 0; k < inner; k++) {
                        *outputWalk++ = ((*inputWalk++) - mean[k]) / var[k] * a + b;
                    }
                }

                inputData += channels * inner;
                outputData += channels * inner;
            }
            delete[] mean;
            delete[] var;
        }
    }
```

## CPULinearOp 解析
最后简单读一下CPULinearOp这个算子。

```cpp
void CpuLinearOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                          const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
//auto st = std::chrono::system_clock::now();
        Data &input = *(datas.find("input")->second);
        Data &output = *(datas.find("output")->second);
        Data &weight = *(datas.find("weight")->second);
        Data &bias = *(datas.find("bias")->second);

        output.Allocate(0.0f);
        int n = input.Count(0) / input.dims.back();
        int m = input.dims.back();
        int k = output.dims.back();
        
        // 这段代码处理权重数据类型为FLOAT32的情况。首先，它将输入、权重、输出和
        // 偏置数据的指针分别转换为 float* 类型的指针。对于偏置数据，如果其维度长度大于0，
        // 则获取其数据指针，否则设为nullptr。
        if (weight.dataType == DataType::FLOAT32) {
            float *inputData = (float *) input.cpuData;
            float *weightData = (float *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            
            // 接下来，计算需要的线程数（threadNum）。这里用的是用户设定的线程数
            //（通过 GetThreads() 获得）。然后，每个线程负责的任务数（per）
            // 为 k（输出数据的最后一个维度）除以线程数。cur 用来表示当前任务的起始位置。
            int threadNum = GetThreads();
            int per = k / threadNum;
            int cur = 0;
            // 接着，创建线程池（通过 GetPool() 获取）和用于保存线程任务的std::future数组。
            // 对于每个线程，确定其需要处理的任务范围（从 cur 到 end），然后提交线程任务。
            // 线程任务是通过调用 FloatLinearPart 函数来执行的，该函数需要输入数据、
            // 权重数据、偏置数据、输出数据、输入维度（n）、权重维度（m）、输出维度（k）
            // 以及任务范围（从 cur 到 end）作为参数。
            auto pool = GetPool();
            std::vector <std::future <void> > futures;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                futures.push_back(pool->Submit(FloatLinearPart, inputData, weightData, biasData, outputData,
                                                  n, m, k, cur, end));
                cur = end;
            }
            
            // 然后，主线程也执行一部分任务，处理范围为从 cur 到 k。
            FloatLinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
            // 最后，主线程等待所有子线程完成工作。通过调用 std::future::get() 
            // 方法来阻塞主线程，直到对应的子线程完成任务。
            // 这样，可以保证所有的线程任务都完成后，主线程才继续执行。
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
        } else if (weight.dataType == DataType::FLOAT16) {
            float *inputData = (float *) input.cpuData;
            uint16_t *weightData = (uint16_t *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            uint16_t *temp = new uint16_t[n * m];
            for (int i = 0; i < n * m; i++) {
                temp[i] = float_to_half(inputData[i]);
            }
            inputData = (float*)temp;
#endif
            int threadNum = GetThreads();
            int per = k / threadNum;
            int cur = 0;
            auto pool = GetPool();
            std::vector <std::future <void> > futures;
            for (int i = 0; i < threadNum - 1; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                futures.push_back(pool->Submit(Float16LinearPart, inputData, weightData, biasData, outputData,
                                                  n, m, k, cur, end));
                cur = end;
            }

            Float16LinearPart(inputData, weightData, biasData, outputData, n, m, k, cur, k);
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
            delete[] temp;
#endif
        } else if (weight.dataType == DataType::INT8) { // 这段代码处理权重数据类型为 INT8 的情况。
            // 这段代码首先对输入、权重、输出和偏置数据的指针进行类型转换，
            // 并根据偏置数据的维度是否大于0来决定是否获取偏置数据的指针。然后，它计算了权重数据的总和。
            float *inputData = (float *) input.cpuData;
            uint8_t *weightData = (uint8_t *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            weight.CalcWeightSum();
            
            // 之后，代码创建一个std::vector<LowBitConfig>对象，
            // LowBitConfig是一个用于存储数据量化信息的类，包括最小值、最大值、位宽和零点。
            // 这些信息是通过遍历输入数据获得的。
            std::vector <LowBitConfig> inputConfigs;
            for (int i = 0; i < n; i++) {
                float minValue = 1e9, maxValue = -1e9;
                for (int j = 0; j < m; j++) {
                    minValue = std::min(minValue, inputData[i * m + j]);
                    maxValue = std::max(maxValue, inputData[i * m + j]);
                }
                inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
            }
            // 接着，创建一个std::vector<uint8_t>对象uinput，并将其大小设置为输入数据的大小（n * m）。
            // uinput中的每个元素都是输入数据元素经过inputConfigs中对应配置信息量化后的结果。
            // 注意这里的量化过程可能会根据是否定义了__AVX2__进行不同的处理。
            std::vector <uint8_t> uinput;
            uinput.resize(n * m);
            for (int i = 0; i < n * m; i++) {
#ifdef __AVX2__
                uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
                uinput[i] = (uinput[i] + !uinput[i]) ^ 128;
#else
                uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
#endif
            }
            
            // 随后，调用MultiplyMultiThread函数，使用多线程并行计算uinput和weightData的乘积，
            // 并将结果存储在outputData中。
            MultiplyMultiThread(uinput.data(), weightData, (int32_t*)outputData, n, m, k, GetThreads());
            // 这段代码的目的是把在使用INT8进行量化计算时由于量化造成的误差进行修正，
            // 使得结果更接近于使用浮点数进行计算的结果。也就是反量化过程。
            for (int i = 0; i < n; i++) {
                // 这一步中，对于每一个输入向量（i从0到n），代码首先初始化inputSum为0，
                // 然后遍历输入向量的每个元素（j从0到m），将元素值加到inputSum上。
                // 如果定义了__AVX2__，则在加到inputSum之前，元素值会先与128进行异或操作。
                uint32_t inputSum = 0;
                for (int j = 0; j < m; j++) {
#ifdef __AVX2__
                    inputSum += uinput[i * m + j] ^ 128;
#else
                    inputSum += uinput[i * m + j];
#endif
                }
                
                // 接下来，代码遍历每个输出元素（j从0到k），并按照以下步骤进行调整和缩放：
                for (int j = 0; j < k; j++) {
                    // 首先，获取输出元素的原始值value。
                    int value = ((int32_t*)outputData)[i * k + j];
#ifdef __AVX2__
                    // 如果定义了__AVX2__，则value会增加128 * weight.weightSum[j]、
                    // 128 * inputSum，并减去m * 128 * 128。
                    value += (128 * weight.weightSum[j]);
                    value += (128 * inputSum);
                    value -= m * 128 * 128;
#endif
                    value -= weight.weightSum[j] * inputConfigs[i].zeroPoint;
                    value -= inputSum * weight.perChannelsConfigs[j].zeroPoint;
                    value += (int)inputConfigs[i].zeroPoint * weight.perChannelsConfigs[j].zeroPoint * m;
                    outputData[i * k + j] = weight.perChannelsConfigs[j].scale * inputConfigs[i].scale * value +
                                            (biasData == nullptr ? 0.0 : biasData[j]);
                }
            }
        } else if (weight.dataType == DataType::INT4 || weight.dataType == DataType::INT4_NOZERO) {
            float *inputData = (float *) input.cpuData;
            uint8_t *weightData = (uint8_t *) weight.cpuData;
            float *outputData = (float *) output.cpuData;
            float *biasData = bias.dims.size() > 0 ? (float *) bias.cpuData : nullptr;
            weight.CalcWeightSum();

            std::vector <LowBitConfig> inputConfigs;
            for (int i = 0; i < n; i++) {
                float minValue = 1e9, maxValue = -1e9;
                for (int j = 0; j < m; j++) {
                    minValue = std::min(minValue, inputData[i * m + j]);
                    maxValue = std::max(maxValue, inputData[i * m + j]);
                }
                inputConfigs.push_back(LowBitConfig(minValue, maxValue, 8, 0));
            }
            std::vector <uint8_t> uinput;
            uinput.resize(n * m);
            for (int i = 0; i < n * m; i++) {
                uinput[i] = inputConfigs[i / m].quantization(inputData[i]);
            }
#ifdef __AVX__
            uint8_t *temp = new uint8_t[32];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j + 31 < m; j += 32) {
                    memcpy(temp, uinput.data() + i * m + j, 32);
                    for (int k = 0; k < 16; k++) {
                        uinput[i * m + j + k] = temp[k * 2 + 1];
                        uinput[i * m + j + k + 16] = temp[k * 2];
                    }
                }
            }
            delete[] temp;
#endif
            if (weight.dataType == DataType::INT4) {
                MultiplyInt4MultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                        weight.weightSum.data(), weight.zeros.data(), weight.scales.data(), biasData,
                                        inputConfigs, GetThreads());
            } else {
                MultiplyInt4NoZeroMultiThread(uinput.data(), weightData, (int32_t *) outputData, n, m, k,
                                        weight.weightSum.data(), weight.mins.data(), weight.scales.data(), biasData,
                                        inputConfigs, GetThreads());
            }
        } else {
            ErrorInFastLLM("Linear error: unsupport weight's dataType.\n");
        }
//float spend = GetSpan(st, std::chrono::system_clock::now());
//float gops = (float)n * m * k / spend / 1e9;
// printf("n = %d, m = %d, k = %d, spend %f s, gops = %f\n", n, m, k, spend, gops);
    }

```

在上面的实现中，MultiplyMultiThread完成了对量化输入的计算，我们看一下它的实现细节：

```cpp
//a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void MultiplyMultiThread(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int threadNum) {
        int per = k / threadNum;
        int cur = 0;
        if (threadNum == 1) {
            Multiply(a, b + cur * m, c + cur, n, m, k - cur, k);
        } else {
            auto pool = GetPool();
            std::vector<std::future<void> > futures;
            for (int i = 0; i < threadNum; i++) {
                int end = cur + per + (cur + per * (threadNum - i) < k);
                if (i == threadNum - 1) {
                    end = k;
                }
                futures.push_back(pool->Submit(Multiply, a, b + cur * m, c + cur, n, m, end - cur, k));
                cur = end;
            }
            for (int i = 0; i < futures.size(); i++) {
                futures[i].get();
            }
        }
    }
```

可以看到这段代码仍然是在用线程池来启动多个线程完成计算，核心部分是Multiply函数，这个函数的实现细节：

```cpp
	  //a = [n, m], b = [k, m], c = aT(b') = [n, k]
    void Multiply(uint8_t *a, uint8_t *b, int32_t *c, int n, int m, int k, int kstride) {
#ifdef __ARM_FEATURE_DOTPROD
        int block = 0;
        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                int value = 0;
                uint8_t *inputWalk = inputStart;
                int j = 0;
                uint32x4_t sum0 = {0, 0, 0, 0};
                for (; j + 31 < m; j += 32) {
                    uint8x16_t vi = vld1q_u8(inputWalk);
                    uint8x16_t vi0 = vld1q_u8(inputWalk + 16);
                    uint8x16_t vw = vld1q_u8(weightWalk);
                    uint8x16_t vw0 = vld1q_u8(weightWalk + 16);
                    sum0 = vdotq_u32(sum0, vi, vw);
                    sum0 = vdotq_u32(sum0, vi0, vw0);
                    inputWalk += 32;
                    weightWalk += 32;
                }

                value += sum0[0] + sum0[1] + sum0[2] + sum0[3];
                for (; j < m; j++) {
				    value += (int)(*(weightWalk++)) * (*(inputWalk++));
			    }
                c[block * kstride + i] = value;
            }
        }
#elif defined(__aarch64__)
        int block = 0;
        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                int value = 0;
                uint8_t *inputWalk = inputStart;

                int per = 64;
                int cnt = m / per;
                int sur = m % per;

                uint32x4_t sum = {0};
                uint16x8_t temp = {0};
                uint16x8_t temp1 = {0};
                uint16x8_t temp2 = {0};
                uint16x8_t temp3 = {0};
                uint16x8_t temp4 = {0};
                uint16x8_t temp5 = {0};
                uint16x8_t temp6 = {0};
                uint16x8_t temp7 = {0};

                while (cnt--) {
                    temp = vmull_u8(vld1_u8(inputWalk), vld1_u8(weightWalk));
                    temp1 = vmull_u8(vld1_u8(inputWalk + 8), vld1_u8(weightWalk + 8));
                    temp2 = vmull_u8(vld1_u8(inputWalk + 16), vld1_u8(weightWalk + 16));
                    temp3 = vmull_u8(vld1_u8(inputWalk + 24), vld1_u8(weightWalk + 24));
                    temp4 = vmull_u8(vld1_u8(inputWalk + 32), vld1_u8(weightWalk + 32));
                    temp5 = vmull_u8(vld1_u8(inputWalk + 40), vld1_u8(weightWalk + 40));
                    temp6 = vmull_u8(vld1_u8(inputWalk + 48), vld1_u8(weightWalk + 48));
                    temp7 = vmull_u8(vld1_u8(inputWalk + 56), vld1_u8(weightWalk + 56));

                    sum = vpadalq_u16(sum, temp);
                    sum = vpadalq_u16(sum, temp1);
                    sum = vpadalq_u16(sum, temp2);
                    sum = vpadalq_u16(sum, temp3);
                    sum = vpadalq_u16(sum, temp4);
                    sum = vpadalq_u16(sum, temp5);
                    sum = vpadalq_u16(sum, temp6);
                    sum = vpadalq_u16(sum, temp7);

                    inputWalk += per;
                    weightWalk += per;
                }

                value += (sum[0] + sum[1] + sum[2] + sum[3]);
                while (sur--) {
                    value += (int)(*(weightWalk++)) * (*(inputWalk++));
                }

                c[block * kstride + i] = value;
            }
        }
#elif defined(__AVX__)
        int block = 0;
        for (; block < n; block++) {
            uint8_t *weightWalk = b;
            uint8_t *inputStart = a + block * m;

            for (int i = 0; i < k; i++) {
                uint8_t *inputWalk = inputStart;

                c[block * kstride + i] = DotU8U8(inputWalk, weightWalk, m);
                weightWalk += m;
            }
        }
#else
        int block = 0;
	    for (; block < n; block++) {
		    uint8_t *weightWalk = b;
		    uint8_t *inputStart = a + block * m;

		    for (int i = 0; i < k; i++) {
			    int value = 0;
			    uint8_t *inputWalk = inputStart;
			    for (int j = 0; j < m; j++) {
				    value += (int)(*(weightWalk++)) * (*(inputWalk++));
			    }

			    c[block * kstride + i] = value;
		    }
	    }
#endif
    }
```

这段代码实现了两个矩阵的乘法。输入的两个矩阵是 \(a\) 和 \(b\)，结果矩阵是 \(c\)。矩阵 \(a\) 的形状是 \([n, m]\)，矩阵 \(b\) 的形状是 \([k, m]\)，所以矩阵 \(c = a^T b\) 的形状是 \([n, k]\)。

在这段代码中，使用了不同的方法进行矩阵乘法，取决于系统是否支持特定的优化硬件指令。

1. 如果系统支持 ARMv8.2 的点积指令（`__ARM_FEATURE_DOTPROD`），那么会使用这个指令进行矩阵乘法。在这种情况下，每次会同时处理32个元素，这样可以加速计算。

2. 如果系统支持 ARMv8（`__aarch64__`），但不支持 ARMv8.2 的点积指令，那么会使用 NEON SIMD 指令进行矩阵乘法。在这种情况下，每次会同时处理64个元素。

3. 如果系统支持 AVX（`__AVX__`），那么会使用 AVX 指令进行矩阵乘法。在这种情况下，会使用 `DotU8U8` 函数来计算向量的点积。

4. 如果系统不支持上述任何一种优化指令，那么会使用基础的方法进行矩阵乘法。在这种情况下，每次只处理一个元素。

这段代码的优化部分主要利用了 SIMD（单指令多数据）的并行化特性，通过同时处理多个元素来加速计算。而选择使用哪种优化方法，取决于系统支持哪种硬件指令。

CPU后端的算子解析就暂时讲到这里，我们发现CPU的算子实现不仅考虑了Intel CPU也考虑了Arm端的优化，这也是FastLLM可以在Arm边缘端部署大模型的原因。
# 0x4. GPU后端算子实现
GPU后端算子实现在 https://github.com/ztxz16/fastllm/blob/master/src/devices/cuda/cudadevice.cpp 和 https://github.com/ztxz16/fastllm/blob/master/src/devices/cuda/fastllm-cuda.cu 。我们还是挑几个算子来讲解。

## CudaLlamaRotatePosition2DOp

LLama的ROPE实现在：https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L92-L126 。

```python
# 这个类是用来创建旋转位置编码（Rotary Position Embedding）的。
# Llama模型引入了旋转位置编码，以改进长序列处理的性能。
class LlamaRotaryEmbedding(torch.nn.Module):
    # 这是类的初始化方法，接收四个参数：dim（嵌入的维度），max_position_embeddings
    # （最大的位置嵌入长度，默认为2048），base（基数，默认为10000）和device（设备类型，例如CPU或GPU）。
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim # 将输入的dim参数保存到self.dim属性中。
        # # 将输入的max_position_embeddings参数保存到self.max_position_embeddings属性中。
        self.max_position_embeddings = max_position_embeddings
        # 将输入的base参数保存到self.base属性中。
        self.base = base
        # 计算逆频率并保存到变量inv_freq中。逆频率是一种用于位置编码的技巧，
        # 它可以帮助模型更好地捕捉位置信息。
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        # 将inv_freq保存到模型的缓存中。register_buffer是PyTorch nn.Module的一个方法，
        # 它用于保存一些不需要计算梯度的变量。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        # 调用_set_cos_sin_cache方法，预先计算并保存正弦和余弦的缓存值。
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )
    
    # 这是一个私有方法，接收三个参数：seq_len（序列长度），device（设备类型）和dtype（数据类型）
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # 将输入的seq_len参数保存到self.max_seq_len_cached属性中。
        self.max_seq_len_cached = seq_len
        # 生成一个长度为max_seq_len_cached的序列，并保存到变量t中。
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        
        # 使用外积计算频率和t的乘积，结果保存到变量freqs中。
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        # 将频率的两份副本拼接在一起，结果保存到变量emb中。
        emb = torch.cat((freqs, freqs), dim=-1)
        # 计算emb的余弦值，然后将结果保存到模型的缓存中。
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        # 计算emb的正弦值，然后将结果保存到模型的缓存中。
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)
    
    # 这是模型的前向传播方法，接收两个参数：x（输入数据）和seq_len（序列长度）。
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # 如果输入的序列长度大于缓存的最大序列长度，那么调用_set_cos_sin_cache方法，更新缓存。
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        
        # 返回对应输入位置的正弦和余弦值。这些值将用于旋转位置编码。
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```


CudaLlamaRotatePosition2DOp对应的就是上面的Python代码。

```cpp
void CudaLlamaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                     const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        FastllmCudaLlamaRotatePosition2D(data, positionIds, sinData, cosData, rotaryDim);
    }
```

这里调用的是FastllmCudaLlamaRotatePosition2D这个函数，它的实现和解析如下：

```cpp
// 这是一个在 GPU 上运行的 CUDA 函数，用于执行 Llama 模型的位置编码旋转操作。
// data：输入的数据，这个数据将会被旋转。
// positionIds：位置编码的数据。
// sinData，cosData：用于旋转的 sin 和 cos 值。
// rotaryDim：旋转的维度。
bool FastllmCudaLlamaRotatePosition2D(fastllm::Data &data, const fastllm::Data &positionIds,
                                      const fastllm::Data &sinData, const fastllm::Data &cosData, int rotaryDim) {
    // 使用 FastllmCudaPrepareInput 函数将输入的数据从 CPU 复制到 GPU。
    // 这个函数会返回一个指向 GPU 内存的指针。                                  
    float *cudaData = (float *) FastllmCudaPrepareInput(data);
    float *cudaPositionIds = (float *) FastllmCudaPrepareInput(positionIds);
    float *cudaSin = (float *) FastllmCudaPrepareInput(sinData);
    float *cudaCos = (float *) FastllmCudaPrepareInput(cosData);
    
    // 计算旋转操作需要的一些参数，包括 outer，spatial，bs，len，n 和 m。
    // 这些参数是用于确定 CUDA 核函数的执行配置和一些数据操作的。
    int outer = data.dims[0] * data.dims[1];
    int spatial = data.Count(2);
    int bs = data.dims[0], len = data.dims[1];
    int n = data.dims[2], m = data.dims[3];
    // 调用 CUDA 核函数 FastllmLlamaRotatePosition2DKernel 来在 GPU 上执行位置编码的旋转操作。
    // <<<outer * n, min(rotaryDim, m / 2)>>> 是 CUDA 中定义并行线程块和线程的语法，
    // outer * n 是线程块的数量，min(rotaryDim, m / 2) 是每个线程块中的线程数量。
    // 核函数的参数包括之前准备的数据和一些计算参数。
    FastllmLlamaRotatePosition2DKernel <<< outer * n, min(rotaryDim, m / 2) >>> (cudaData, cudaPositionIds, cudaSin, cudaCos,
                                                                                 len, bs, spatial, n, m,
                                                                                 (int)positionIds.dims.back(), (int)sinData.dims[1], rotaryDim);

    // 使用 FastllmCudaFinishInput 函数释放 positionIds，sinData 和 cosData 在 GPU 上的内存。
    // 这些数据在这个函数中不再需要。
    FastllmCudaFinishInput(positionIds, cudaPositionIds);
    FastllmCudaFinishInput(sinData, cudaSin);
    FastllmCudaFinishInput(cosData, cudaCos);
    // 使用 FastllmCudaFinishOutput 函数将旋转后的数据从 GPU 复制回 CPU。
    // 这个函数也会释放 data 在 GPU 上的内存。
    FastllmCudaFinishOutput(data, cudaData);
    return true;
}
```

最后再解析下这个cuda kernel。

```cpp
// float *data：输入数据，大小为 [bs, len, n, m]，其中 bs 是批量大小，
// len 是序列长度，n 是头的数量，m 是每个头的维度。
// float *positionIds：位置编码的索引，大小为 [bs, len]。
// float *sin 和 float *cos：预先计算的正弦和余弦值，用于旋转编码。
// int len, int bs, int spatial, int n, int m：输入数据的各个维度大小。
// int partStride 和 int sinCosStride：用于索引 positionIds 和 sin/cos 的步长。
// int rotateDim：旋转维度。
__global__ void FastllmLlamaRotatePosition2DKernel(float *data, float *positionIds, float *sin, float *cos,
                                                   int len, int bs, int spatial, int n, int m, int partStride, int sinCosStride, int rotateDim) {
    // 首先，计算出当前线程应处理的位置 o，长度 l 和批次 b。
    int o = (blockIdx.x / n);
    int l = o % len;
    int b = o / len;
    int j = threadIdx.x;
    // 然后，根据 positionIds 获取对应的旋转角度的正弦值 curSin 和余弦值 curCos。
    int index = (int) (positionIds[b * partStride + l]);

    float curSin = sin[index * sinCosStride + j];
    float curCos = cos[index * sinCosStride + j];
    float *d = (float *) data + o * spatial + j;
    int i = blockIdx.x % n;
    // 接着，获取输入数据对应位置的值 va 和 vb。
    float va = d[i * m], vb = d[i * m + m / 2];
    // 最后，根据旋转矩阵的公式，计算旋转后的值，并将结果写回输入数据中。
    d[i * m] = va * curCos - vb * curSin;
    d[i * m + m / 2] = va * curSin + vb * curCos;
}

```

直接看这个cuda kernel可能比较难理解，可以结合https://github.com/ztxz16/fastllm/blob/master/src/devices/cpu/cpudevice.cpp#L2204-L2233 这里的cpu实现来看，这样来看设置batch * seq_length * n个block，每个block处理m个元素就是比较合理直观的。

```cpp
void CpuLlamaRotatePosition2DOp::Run(const std::string &opType, const fastllm::DataDict &datas,
                                    const fastllm::FloatDict &floatParams, const fastllm::IntDict &intParams) {
        Data &data = *(datas.find("input")->second);
        Data &positionIds = *(datas.find("positionIds")->second);
        Data &sinData = *(datas.find("sin")->second);
        Data &cosData = *(datas.find("cos")->second);
        int rotaryDim = intParams.find("rotaryDim") != intParams.end() ? intParams.find("rotaryDim")->second : 128;

        int bs = data.dims[0], len = data.dims[1];
        int spatial = data.Count(2);
        int n = data.dims[2], m = data.dims[3];
        int stride = (int)sinData.dims[1];
        for (int b = 0; b < bs; b++) {
            for (int l = 0; l < len; l++) {
                int index = (int) ((float *) positionIds.cpuData)[b * positionIds.dims.back() + l];
                float *sin = ((float *) sinData.cpuData) + stride * index;
                float *cos = ((float *) cosData.cpuData) + stride * index;
                float *d = (float *) data.cpuData + (b * len + l) * spatial;
                for (int i = 0; i < n; i++) {
                    for (int j = 0; j < rotaryDim && j < m / 2; j++) {
                        float a = d[j], b = d[j + m / 2];
                        d[j] = a * cos[j] - b * sin[j];
                        d[j + m / 2] = a * sin[j] + b * cos[j];
                    }

                    d += m;
                }
            }
        }
    }
```

FastLLM在cuda上的实现不算高校，不过优点在于它支持了完整的int8和int4量化的计算，有兴趣的读者可以自行研究这部分kernel实现。

# 0x5. LLMSamping解析

在 chatglm-6b 的实现中，在前向推理完成后以及tokenizer解码之前有一个根据logits取label的过程：https://github.com/ztxz16/fastllm/blob/master/src/models/chatglm.cpp#L267-L279 。

```cpp
if (generationConfig.IsSimpleGreedy()) {
            // 对 logits 进行 TopK 操作，将结果存储在 topk 中。
            // 这里的 TopK 操作是找到 logits 中最大的 K 个值，这里 K=1，所以是找到最大值。
            TopK(logits, topk, 1); 
            topk.ToDevice(DataDevice::CPU);
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b; // 计算基础索引值 base。
                // 将 topk 中对应索引的值取整并添加到 lastRet 中。
                lastRet.push_back((int) (((float *) topk.cpuData)[base * 2] + 1e-3));
            }
        } else {
            for (int b = 0; b < batch; b++) {
                int base = (maxLen - 1) * batch + b; // 计算基础索引值 base。
                // 使用 LLMSampling 方法进行抽样，将结果添加到 lastRet 中。
                lastRet.push_back(LLMSampling(logits, base, generationConfig, lastTokens.units[b]));
            }
        }
```

LLMSampling是一种常见的在序列生成任务中，根据不同的需求，使用不同的策略生成序列的方法。我们这里来研究一下它的实现。它的实现在：https://github.com/ztxz16/fastllm/blob/master/src/fastllm.cpp#L874-L916 。

```cpp
// 这段代码是一个用于从给定的 logits（通常表示预测的概率分布）进行采样的函数，
// 采样策略主要受 GenerationConfig 和 LastTokensUnit 参数的影响。
int LLMSampling(Data &logits, int outerOffset,
                    const GenerationConfig &config, const LastTokensUnit &tokens) {
        // 将 logits 数据从当前设备转移到 CPU。
        logits.ToDevice(DataDevice::CPU);
        // 从 logits 的维度中获取词汇量 vocabSize。
        int vocabSize = logits.dims.back();
        // 计算 base 指针，指向要处理的 logits 的开始位置。
        float *base = ((float*)logits.cpuData) + outerOffset * vocabSize;
        
        // 判断 config.repeat_penalty 是否不等于1，如果不等于1，
        // 则对 tokens.tokenSet 中每个 id 对应的 base[id] 值进行修改。
        if (fabs(config.repeat_penalty - 1.0) > 1e-6) {
            for (int id : tokens.tokenSet) {
                base[id] = (base[id] < 0 ? base[id] * config.repeat_penalty : base[id] / config.repeat_penalty);
            }
        }
        // 计算温度的倒数 invTemp。
        float invTemp = 1.0f / config.temperature;
        // 定义一个向量 v，用于存储 <logit值，索引>。
        std::vector <std::pair <float, int> > v;
        // 遍历每个 logit，将其值乘以 invTemp，并存入 v 中。
        for (int i = 0; i < vocabSize; i++) {
            v.push_back(std::make_pair(-base[i] * invTemp, i));
        }
        // 计算 topk，它是词汇量 vocabSize 和 config.top_k 中的较小值。
        int topk = std::min(vocabSize, config.top_k);
        // 对 v 中的前 topk 个元素进行排序。
        std::partial_sort(v.begin(), v.begin() + topk, v.end());
        // 初始化 psum 和 maxValue，maxValue 是 v 中最大的元素。
        float psum = 0.0, maxValue = -v.begin()->first;
        // 定义一个向量 ps，用于存储处理后的概率。
        std::vector <float> ps;
        // 遍历 v 中的前 topk 个元素，将其值取 exp 并减去 maxValue，存入 ps，同时更新 psum。
        for (int i = 0; i < topk; i++) {
            ps.push_back(expf(-v[i].first - maxValue));
            psum += ps.back();
        }
        float curSum = 0.0;
        // 遍历 ps，将其每个元素除以 psum 并更新 curSum，
        // 当 curSum 大于 config.top_p 时，更新 topk 并退出循环。
        for (int i = 0; i < topk; i++) {
            ps[i] /= psum;
            curSum += ps[i];
            if (curSum > config.top_p) {
                topk = i + 1;
                break;
            }
        }
        // 生成一个随机数 rnd。
        float rnd = fastllmRandom.randP();
        curSum = 0.0;
        // 遍历 ps 中的前 topk 个元素，将其累加到 curSum，
        // 当 curSum 大于 rnd 或者达到最后一个元素时，
        // 返回对应 v[i].second，也就是返回采样得到的 id。
        for (int i = 0; i < topk; i++) {
            curSum += ps[i];
            if (curSum > rnd || i == topk - 1) {
                return v[i].second;
            }
        }
        // 如果以上步骤都没有返回，那么返回 -1。
        return -1;
    }
```

LLMSampling实现了一种基于温度和惩罚的采样策略，用于从给定的 logits 中选择一个 id。这种采样的方法可以控制输出文本的多样性。

# 0x6. 总结
接着 [大模型部署框架 FastLLM 简要解析](https://mp.weixin.qq.com/s/4Cws_gXUuGXbyURIr0SdGw) 这篇文章首先梳理了一下FastLLM的调用链和关键的数据结构，然后解析了 FastLLM 的一些实现细节和CPU/GPU后端实现采用的优化技巧。

