> 本文首发于 GiantPandaCV
> ：[绑定cpu](https://link.zhihu.com/?target=https%3A//mp.weixin.qq.com/s%3F__biz%3DMzA4MjY4NTk0NQ%3D%3D%26mid%3D2247485852%26idx%3D1%26sn%3D5c5f0b3ca0212df33756291016343b34%26chksm%3D9f80b30aa8f73a1c501245b571a7930fc26f46f1ee572680b3aa410f1fef01c61f37f7c5a1a2%26token%3D1939593186%26lang%3Dzh_CN%23rd)

本文主要内容是介绍移动端优化会涉及到的绑定cpu（cpu affinity）`[2,3]`的概念和相关验证实验。

用过一些移动端推理框架比如ncnn和paddlelite等框架的同学都会发现这些框架提供了运行时功耗级别的设置，其实里面就是用绑核的方式实现的，通过绑定大核实现高功耗模式，绑定小核实现低功耗模式。关于arm大小核的概念可以参考`[1]`。

 **本文相关实验代码：**

[Ldpe2G/ArmNeonOptimization](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/tree/master/cpuAffinityExperiments)

##  **绑核的概念**

简单来说就是把一个进程绑定到一到多个cpu上，也就是让一些进程只运行在可运行的cpu上。比如：让一个进程只运行在某个cpu上或者让一些进程只运行在除了某个cpu以外的cpu上等等。

 **那绑核有什么作用呢？**

首先是可以控制进程运行耗时，比如大核主频高，绑定到大核上运行起来应该会比绑定到小核上运行耗时小。其次根据文章`[2]`的说法，可以优化缓存性能。

我个人理解就是如果进程没有绑定在一个cpu上，那么当该进程切换cpu的时候，新cpu 的 cache上并没有之前cpu cache上缓存的数据，就会导致cache
miss，然后需要从内存加载数据，然后过一段时间切回去原来cpu之后，可能原来的cache里面的内容也失效了，又导致cache miss，那么这样来回切换就很影响性能。

##  **ncnn & paddlelite绑核实现**

下面来看下一些推理框架上的实现：

paddlelite实现代码:

[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc#L453](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc%23L453)

```C++
int set_sched_affinity(const std::vector<int>& cpu_ids) {
// #define CPU_SETSIZE 1024
// #define __NCPUBITS  (8 * sizeof (unsigned long))
// typedef struct
// {
//    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
// } cpu_set_t;

// set affinity for thread
#ifdef __GLIBC__
  pid_t pid = syscall(SYS_gettid);
#else
  pid_t pid = gettid();
#endif
  cpu_set_t mask;
  CPU_ZERO(&mask);
  for (int i = 0; i < cpu_ids.size(); ++i) {
    CPU_SET(cpu_ids[i], &mask);
  }
  int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
  if (syscallret) {
    return -1;
  }
  return 0;
}
```


ncnn实现代码:

[https://github.com/Tencent/ncnn/blob/ee41ef4a378ef662d24f137d97f7f6a57a5b0eba/src/cpu.cpp#L319](https://link.zhihu.com/?target=https%3A//github.com/Tencent/ncnn/blob/ee41ef4a378ef662d24f137d97f7f6a57a5b0eba/src/cpu.cpp%23L319)

```C++
static int set_sched_affinity(size_t thread_affinity_mask)
{
    // cpu_set_t definition
    // ref http://stackoverflow.com/questions/16319725/android-set-thread-affinity
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
typedef struct
{
    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

#define CPU_SET(cpu, cpusetp) \
    ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
    memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
    pid_t pid = syscall(SYS_gettid);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    for (int i=0; i<(int)sizeof(size_t) * 8; i++)
    {
        if (thread_affinity_mask & (1 << i))
            CPU_SET(i, &mask);
    }

    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
    if (syscallret)
    {
        fprintf(stderr, "syscall error %d\n", syscallret);
        return -1;
    }

    return 0;
}
```


其实从实现上看两者基本大同小异，下面就以ncnn的实现去分析。

刚开始理解代码的时候，我谷歌了一下`sched_setaffinity`这个系统调用，搜到了android源码里的声明头文件：

[https://android.googlesource.com/platform/bionic.git/+/master/libc/include/sched.h](https://link.zhihu.com/?target=https%3A//android.googlesource.com/platform/bionic.git/%2B/master/libc/include/sched.h)

```C++
#ifndef _SCHED_H_
#define _SCHED_H_
#include <bits/timespec.h>
#include <linux/sched.h>
#include <sys/cdefs.h>
__BEGIN_DECLS
......
int sched_getcpu(void);

#ifdef __LP64__
#define CPU_SETSIZE 1024
#else
#define CPU_SETSIZE 32
#endif
#define __CPU_BITTYPE  unsigned long int  /* mandated by the kernel  */
#define __CPU_BITS     (8 * sizeof(__CPU_BITTYPE))
#define __CPU_ELT(x)   ((x) / __CPU_BITS)
#define __CPU_MASK(x)  ((__CPU_BITTYPE)1 << ((x) & (__CPU_BITS - 1)))
typedef struct {
  __CPU_BITTYPE  __bits[ CPU_SETSIZE / __CPU_BITS ];
} cpu_set_t;
int sched_setaffinity(pid_t __pid, size_t __set_size, const cpu_set_t* __set);
#define CPU_ZERO(set)          CPU_ZERO_S(sizeof(cpu_set_t), set)
#define CPU_SET(cpu, set)      CPU_SET_S(cpu, sizeof(cpu_set_t), set)
......
#define CPU_ZERO_S(setsize, set)  __builtin_memset(set, 0, setsize)
#define CPU_SET_S(cpu, setsize, set) \
  do { \
    size_t __cpu = (cpu); \
    if (__cpu < 8 * (setsize)) \
      (set)->__bits[__CPU_ELT(__cpu)] |= __CPU_MASK(__cpu); \
  } while (0)
......
#endif /* _SCHED_H_ */
```


为了可读性，我简化了代码，可以看到ncnn的实现里的`cpu_set_t`结构体和宏定义基本和源码里的一致。

下面详细分析下ncnn代码实现：

```C++
// thread_affinity_mask是一个size_t类型变量，假设是4个字节32bit，
// 则该变量每一比特位对应一个cpu编号，0表示不绑定，1表示绑定
// 比如只绑定cpu0,则该变量的比特位表示：00000000000000000000000000000001
// 比如只绑定cpu0和31，则该变量的比特位表示：10000000000000000000000000000001
// 如果要绑定所有核，则是：11111111111111111111111111111111
// 该变量每个bit位由用户根据需要绑定的cpu编号设定。
static int set_sched_affinity(size_t thread_affinity_mask)
{
#define CPU_SETSIZE 1024
#define __NCPUBITS  (8 * sizeof (unsigned long))
// 这里我机器上sizeof (unsigned long)==4，
// 所以cpu_set_t的__bits数组长度是1024/32=32，
// 也就是32个32bit的变量，因为一个32bit就能表示32个cpu
// 而手机一般也就8个cpu，所以感觉其实一个unsigned long变量就足够了
// 我也做实验验证了，后面实验部分会再详细说明
typedef struct
{
    unsigned long __bits[CPU_SETSIZE / __NCPUBITS];
} cpu_set_t;

// 设置mask变量比特位宏
#define CPU_SET(cpu, cpusetp) \
    ((cpusetp)->__bits[(cpu)/__NCPUBITS] |= (1UL << ((cpu) % __NCPUBITS)))

#define CPU_ZERO(cpusetp) \
    memset((cpusetp), 0, sizeof(cpu_set_t))

    // set affinity for thread
    pid_t pid = syscall(SYS_gettid);
    
    cpu_set_t mask;
    // 先把mask清零
    CPU_ZERO(&mask);
    // 然后遍历用户传入的thread_affinity_mask变量的每一比特位
    // 如果某一位为1，则设置cpu_set_t->__bits变量对应位置
    // 其实从CPU_SET宏的实现也可以看到，尽管__bits数组长度是32，
    // 但是只会访问到第0个变量，因为 i < 32，
    // 所以为什么我觉得直接一个unsigned long变量就足够了
    for (int i=0; i<(int)sizeof(size_t) * 8; i++)
    {
        if (thread_affinity_mask & (1 << i))
            CPU_SET(i, &mask);
    }
    // 最后调用`__NR_sched_setaffinity`，传入mask，完成绑定
    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(mask), &mask);
```


## **实验 &结果分析**

看懂ncnn实现代码之后，我自己简单设置了3个对比实验来验证绑核的作用，就是通过人为绑定大小核，来观察代码运行时间的变化。

 **完整实验代码见：**

[Ldpe2G/ArmNeonOptimization](https://link.zhihu.com/?target=https%3A//github.com/Ldpe2G/ArmNeonOptimization/tree/master/cpuAffinityExperiments)

 **每个实验共同部分都是：**

开4个线程，每个线程内调用一次BoxFilter，时间统计是从开启4个线程到4个线程都退出的总耗时。

 **三个实验的区别部分是：**

第一个实验，并不设置绑核； 第二个实验，绑定小核； 第三个实验，绑定大核。

首先给出上面提到的绑定函数实现上的简化（完整代码见github链接），这里可以看到直接用用户传入的mask即可：

```C++
static int set_sched_affinity(size_t thread_affinity_mask)
{
    pid_t pid = syscall(SYS_gettid);
    int syscallret = syscall(__NR_sched_setaffinity, pid, sizeof(thread_affinity_mask), &thread_affinity_mask);
    .....
    return 0;
}
```


还有这里对于大小核的设置有一点需要注意的是，大小核是一个相对的概念，就是对于一台设备来说，所谓小核就是在这台设备上，相对其他核，频率最低的，大核就是相对频率最高的。

一般推理框架内对于大小核都有判定逻辑，比如ncnn的策略：

```C++
static int setup_thread_affinity_masks()
{
    // 绑定所有核，那就是把mask对应所有cpu编号位设为1
    // 比如8个核，1左移8位就是 100000000,再减一就是1111111，
    // 相当于绑定0~7号cpu
    g_thread_affinity_mask_all = (1 << g_cpucount) - 1;

    // 统计所有cpu中的最大和最小主频
    int max_freq_khz_min = INT_MAX;
    int max_freq_khz_max = 0;
    std::vector<int> cpu_max_freq_khz(g_cpucount);
    for (int i=0; i<g_cpucount; i++)
    {
        // get_max_freq_khz函数获取cpu编号对应的主频
        // 具体实现见ncnn源码，或者github上代码
        int max_freq_khz = get_max_freq_khz(i);

        cpu_max_freq_khz[i] = max_freq_khz;

        if (max_freq_khz > max_freq_khz_max)
            max_freq_khz_max = max_freq_khz;
        if (max_freq_khz < max_freq_khz_min)
            max_freq_khz_min = max_freq_khz;
    }
    // 计算主频中值
    int max_freq_khz_medium = (max_freq_khz_min + max_freq_khz_max) / 2;
    // 就是如果主频低于中值则算为小核，否则算大核
    for (int i=0; i<g_cpucount; i++)
    {
        if (cpu_max_freq_khz[i] < max_freq_khz_medium)
            g_thread_affinity_mask_little |= (1 << i);
        else
            g_thread_affinity_mask_big |= (1 << i);
    }

    return 0;
}
```

paddlelite策略和ncnn类似，也是计算最大主频和最小主频取中间值，大于中间值的的就是大核，否则就是小核。具体代码见：

[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc#L345](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc%23L345)

不过paddlelite还多了一步，就是对于一些预定义好的SOC，会人为设定其大小核，感兴趣的读者可以去看下完整代码：

[https://github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc#L646](https://link.zhihu.com/?target=https%3A//github.com/PaddlePaddle/Paddle-Lite/blob/develop/lite/core/device_info.cc%23L646)

下面只列出部分实现代码：

```C++
bool DeviceInfo::SetCPUInfoByName() {
  ......
  else if (dev_name_.find("KIRIN980") != std::string::npos ||
             dev_name_.find("KIRIN990") !=
                 std::string::npos) {  // Kirin 980, Kirin 990
    core_num_ = 8;
    core_ids_ = {0, 1, 2, 3, 4, 5, 6, 7};
    big_core_ids_ = {4, 5, 6, 7};
    little_core_ids_ = {0, 1, 2, 3};
    cluster_ids_ = {1, 1, 1, 1, 0, 0, 0, 0};
    SetArchInfo(2, kA76, kA55);
    SetCacheInfo(0, 2, 64 * 1024, 32 * 1024);
    SetCacheInfo(1, 2, 512 * 1024, 128 * 1024);
    SetCacheInfo(2, 1, 4096 * 1024);
    SetFP16Info(1, 1);
    SetDotInfo(1, 1);
    return true;
    .....
```


然后回来看下实验代码：

```C++
// 线程函数实现，就是打印线程当前运行所在的cpu，然后执行boxfilter
// 注释掉的代码，读者感兴趣去实验的时候可以打开看看，
// 如果不是人为绑定到具体一个核上运行的话，
// 有几率会看到线程在睡眠前后切换了运行的cpu
std::mutex g_display_mutex;
void threadFun() {
    size_t cpu;
    auto syscallret = syscall(__NR_getcpu, &cpu, NULL, NULL);
    std::stringstream ss;
    ss << std::this_thread::get_id();

    g_display_mutex.lock();
    printf("thread %s, running on cpu: %d\n", ss.str().c_str(), cpu);
    g_display_mutex.unlock();

    // std::this_thread::sleep_for (std::chrono::microseconds(100));

    Boxfilter(7, 500, 500);

    // syscallret = syscall(__NR_getcpu, &cpu, NULL, NULL);
    // g_display_mutex.lock();
    // printf("thread %s, running on cpu: %d\n", ss.str().c_str(), cpu);
    // g_display_mutex.unlock();
}
```

我的实验机器是华为P30（Kirin 908），先看下核数和cpu的主频信息：

```bash
before sort
cpu_0:1805000, cpu_1:1805000, cpu_2:1805000, cpu_3:1805000, 
cpu_4:1920000, cpu_5:1920000, cpu_6:2600000, cpu_7:2600000,
after sort
cpu_6:2600000, cpu_7:2600000, cpu_4:1920000, cpu_5:1920000, 
cpu_2:1805000, cpu_3:1805000, cpu_0:1805000, cpu_1:1805000,
```


可以看到和paddlelite里面设置一致，cpu 0~3是小核，4~7是大核，当然6~7比4~5主频又要高一些。

给出实验代码，方便读者理解，完整代码见github：    

```C++
// bind all cores
    printf("bind all cores ex:\n");
    auto start = std::chrono::steady_clock::now(), stop = start;

    for (int i = 0; i < threadNum; ++i) {
      std::thread t(threadFun);
      threads.push_back(std::move(t));
    }
    for (int i = 0; i < threadNum; ++i) {
      threads[i].join();
    }

    stop = std::chrono::steady_clock::now();
    auto time = (0.000001 * std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    printf("bind all core time: %f\n\n", time);
    threads.clear();

    // bind little cores
    printf("bind little cores ex:\n");
    size_t mask2 = 0;
    for (int i = 0; i < g_cpucount; ++i) {
      if (cpu_max_freq_khz[i] == max_freq_khz_min) {
        mask2 |= (1 << cpu_idx[i]);
        printf("bind cpu: %d, ", cpu_idx[i]);
      }
    }
    printf("\n");
    int ret2 = set_sched_affinity(mask2);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < threadNum; ++i) {
      std::thread t(threadFun);
      threads.push_back(std::move(t));
    }
    for (int i = 0; i < threadNum; ++i) {
      threads[i].join();
    }

    stop = std::chrono::steady_clock::now();
    time = (0.000001 * std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    printf("bind little core time: %f\n\n", time);
    threads.clear();

    // bind big cores
    printf("bind big cores ex:\n");
    size_t mask = 0;
    for (int i = 0; i < g_cpucount; ++i) {
      if (cpu_max_freq_khz[i] >= max_freq_khz_medium) {
        mask |= (1 << cpu_idx[i]);
        printf("bind cpu: %d, ", cpu_idx[i]);
      }
    }
    printf("\n");
    int ret = set_sched_affinity(mask);

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < threadNum; ++i) {
      std::thread t(threadFun);
      threads.push_back(std::move(t));
    }
    for (int i = 0; i < threadNum; ++i) {
      threads[i].join();
    }

    stop = std::chrono::steady_clock::now();
    time = (0.000001 * std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count());
    printf("bind big core time: %f\n", time);

    printf("\n");
```


看下实验结果：

```bash
#########################
iteration 1

bind all cores ex:
thread -360738256, running on cpu: 3
thread -358632912, running on cpu: 7
thread -357580240, running on cpu: 5
thread -359685584, running on cpu: 7
bind all core time: 76.996875

bind little cores ex:
bind cpu: 2, bind cpu: 3, bind cpu: 0, bind cpu: 1, 
thread -357580240, running on cpu: 2
thread -358632912, running on cpu: 3
thread -360738256, running on cpu: 1
thread -359685584, running on cpu: 0
bind little core time: 368.479166

bind big cores ex:
bind cpu: 6, bind cpu: 7, 
thread -359685584, running on cpu: 4
thread -358632912, running on cpu: 5
thread -360738256, running on cpu: 5
thread -357580240, running on cpu: 4
bind big core time: 99.261979

#########################
iteration 2

bind all cores ex:
thread -220220880, running on cpu: 1
thread -222326224, running on cpu: 0
thread -221273552, running on cpu: 4
thread -219168208, running on cpu: 5
bind all core time: 257.230729

bind little cores ex:
bind cpu: 2, bind cpu: 3, bind cpu: 0, bind cpu: 1, 
thread -219168208, running on cpu: 1
thread -220220880, running on cpu: 0
thread -221273552, running on cpu: 2
thread -222326224, running on cpu: 3
bind little core time: 269.061458

bind big cores ex:
bind cpu: 6, bind cpu: 7, 
thread -222326224, running on cpu: 4
thread -220220880, running on cpu: 5
thread -219168208, running on cpu: 4
thread -221273552, running on cpu: 5
bind big core time: 100.983854
```


这里只列出了2次实验结果，在脚本我是设定了跑10次，这个读者实验的时候可以自己设置。

根据实验结果可以看到，对比绑定大核和小核，确实绑定大核上运行会比绑定小核运行速度要更快，不过这里大核我是显式绑定6和7，但是多数情况下会失败，绑到了4和5，不知道是不是用户自己绑定有什么限制，如果系统调度就可以跑到6和7。

然后看到迭代1和2的不绑定核（相当于绑定所有核）实验结果，如果刚好这次4个线程多数都跑在了大核上那么速度就会快否则就可能会慢。

###  **参考资料**

  * [1] [https://hexus.net/tech/tech-explained/cpu/48693-tech-explained-arm-biglittle-processing/](https://link.zhihu.com/?target=https%3A//hexus.net/tech/tech-explained/cpu/48693-tech-explained-arm-biglittle-processing/)
  * [2] [https://www.linuxjournal.com/article/6799](https://link.zhihu.com/?target=https%3A//www.linuxjournal.com/article/6799)
  * [3] [https://www.quora.com/What-is-CPU-affinity](https://link.zhihu.com/?target=https%3A//www.quora.com/What-is-CPU-affinity)
  * [4] [https://android.googlesource.com/platform/bionic.git/+/master/libc/include/sched.h](https://link.zhihu.com/?target=https%3A//android.googlesource.com/platform/bionic.git/%2B/master/libc/include/sched.h)
  * [5] [https://linux.die.net/man/2/sched_setaffinity](https://link.zhihu.com/?target=https%3A//linux.die.net/man/2/sched_setaffinity)

