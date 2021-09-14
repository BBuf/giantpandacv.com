# MSNHNET
  本专栏为专栏作者自主设计和实现的网络部署推理架构，用于将训练好的模型部署到各种实际的运行平台并在CPU/GPU下实现加速。
**git: https://github.com/msnh2012/Msnhnet**

#### Msnhnet现在以及未来计划
- 新增更多的算子(x86 + arm)。
- 对特殊算子进行优化, 例如对1x1, 2x2, 3x3, 4x4等卷积进行单独实现(x86 + arm)。
- 使用更优的算法对卷积层进行优化，如winograd, FFT, 大核分块等(x86 + arm)。
- 进一步对arm框架进行neon intrins支持和neon assembly支持。
- 支持fp16(arm)。
- 支持INT8/7<模型后量化>(x86 + arm)。
- 支持2BIT/3BIT/dorefa(x86 + arm)。
- 框架图优化。
- 算子精简(x86 + arm)。
- IO优化(x86 + arm)。
- 新增一些预处理+预处理优化(x86 + arm)。
- 新增一些后处理+后处理优化(x86 + arm)。
- 对vulkan/opengl加速支持(x86 + arm)。
- 对C#版更多网络的支持。
- 对python语言的支持。
- 对更多pytorch的op支持。
- ~~内存复用技术，减少数据搬运带来的大量时间开销。~~
- 将开发过程中的一些技术以及使用这种技术带来了何种效果使用公众号（GiantPandaCV）文章的方式及时公布，方便读者跟进项目进程以及学习一些优化技术。
