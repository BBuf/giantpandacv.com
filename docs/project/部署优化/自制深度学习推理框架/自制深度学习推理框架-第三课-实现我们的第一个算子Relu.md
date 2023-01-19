# 自制深度学习推理框架-起飞！实现我们的第一个算子ReLu
## 我们的课程主页
[https://github.com/zjhellofss/KuiperInfer](https://github.com/zjhellofss/KuiperInfer) 欢迎pr和点赞

### 本期视频位置
**请务必配合视频一起学习该课件.** [视频地址](https://www.bilibili.com/video/BV1bG4y1J7sQ)

### 本期代码位置

```shell
git clone https://gitee.com/fssssss/KuiperCourse.git
git checkout fouth
```

## ReLu算子的介绍

`ReLu`是一种非线性激活函数, 它有运算简单, 不会在边缘处出现梯度消失的特点, 而且它在一定程度上能够防止深度学习模型在训练中发生的过拟合现象. `ReLu`的公式表达如下所示, 如果对于深度学习基本概念不了解的同学, 可以将`ReLu`当作一个公式进行对待, 不用深究其背后的含义.

![在这里插入图片描述](https://img-blog.csdnimg.cn/84a86685ed9c4762b5c78ba92cdf8478.png)

我们今天的任务就是来完成这个公式中的操作, 值得注意的是, 在我们的项目中, `x`和`f(x)`可以理解为我们在第二、第三节中实现的张量类(`Tensor`). 完整的`Tensor`定义可以看我们的上游项目, [代码链接](https://github.com/zjhellofss/KuiperInfer/blob/c2979c7e4d14868fb9342e7bd87cb101a7ab0ed5/include/data/tensor.hpp#L23)

## Operator类

`Operator`类就是我们在第一节中说过的计算图中**计算节点或者说操作符**的概念, 计算图的另外一个组成是**数据流图**. 一个规定了数据是怎么流动的, 另外一个规定了数据到达某个节点后是怎么进行运算的.

在我们的代码中先定义一个`Operator`类, 它是一个父类, 其余的`Operator`包括我们本节要实现的`ReLuOperator`都是其派生类.`Operator`类中会存放计算节点相关的参数, 例如在`ConvOperator`中就会存放初始化卷积算子所需要的`stride`,  `padding`,  `kernel_size`等参数, 而本节的`ReLuOperator`就会带有`thresh`值.

我们从下方的代码中来了解`Operator`类和`ReLuOperator`类, 它们是父子关系, `Operator`是基类, `OpType`记录`Operator`的类型.

值得注意的是计算图中的具体计算操作并不放在`Operator`类中执行, **而是根据`Operator`中存放的参数去初始化对应的`Layer`, 在`ReLuOperator`中记录了初始化`ReLuLayer`运行所需要的`thresh`.**

整体的执行关系是这样的
1. 根据模型文件来定义`operator`,并将相关的参数存放在`operator`中
2. 根据`operator`中存放的参数, 去初始化对应的`Layer`
3. 获取输入数据, `Layer`进行运算

```cpp
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorReLu = 0,
};

class Operator {
 public:
  OpType kOpType = OpType::kOperatorUnknown;

  virtual ~Operator() = default;

  explicit Operator(OpType op_type);
};
```

`ReLuOperator`实现：

```cpp
class ReLuOperator : public Operator {
 public:
  ~ReLuOperator() override = default;

  explicit ReLuOperator(float thresh);

  void set_thresh(float thresh);

  float get_thresh() const;

 private:
  float thresh_ = 0.f;
};
```

## Layer类

我们会在`Operator`类中存放从**计算图结构文件**得到的信息, 例如在`ReLuOperator`中存放的`thresh`值作为一个参数, 这个参数就是我们从计算图结构文件中得到的.

下一步我们需要根据`ReLuOperator`类去完成`ReLuLayer`的初始化, 他们的区别在于`ReLuOperator`负责**存放从计算图中得到的计算节点参数信息**, 不负责计算. 而`ReLuLayer`则负责具体的计算操作. 同样, 所有的`Layer`类有一个公共父类`Layer`. 我们可以从下方的代码中来了解两者的关系.

```cpp
class Layer {
 public:
  explicit Layer(const std::string &layer_name);

  virtual void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                        std::vector<std::shared_ptr<Tensor<float>>> &outputs);

  virtual ~Layer() = default;
 private:
  std::string layer_name_;
};
```

其中`Layer`的`Forwards`方法是具体的执行函数, 负责将输入的`inputs`中的数据, 进行`ReLu`运算并存放到对应的`outputs`中.

```cpp
class ReLuLayer : public Layer {
 public:
  ~ReLuLayer() override = default;

  explicit ReLuLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

 private:
  std::shared_ptr<ReLuOperator> op_;
};
```

这是继承于`Layer`的`ReLuLayer`类, 我们可以看到其中有一个`op`成员, 是一个`ReLuOperator`指针, 这个指针指向的`operator`中负责存放`ReLuLayer`计算时所需要用到的一些参数. 此处`op_`存放的参数比较简单, 只有`ReLuOperator`中的`thresh`参数.

我们再看看是怎么使用`ReLuOperator`去初始化`ReLuLayer`的, 先通过统一接口传入`Operator`类, 再转换为对应的`ReLuOperator`派生类指针, 最后再通过指针中存放的信息去初始化`op_`.

```cpp
ReLuLayer::ReLuLayer(const std::shared_ptr<Operator> &op) : Layer("ReLu") {
  CHECK(op->kOpType == OpType::kOperatorReLu);
  ReLuOperator *ReLu_op = dynamic_cast<ReLuOperator *>(op.get());
  CHECK(ReLu_op != nullptr);
  this->op_ = std::make_shared<ReLuOperator>(ReLu_op->get_thresh());
}
```

我们来看一下具体`ReLuLayer`的`Forwards`过程, 它在执行具体的计算, 完成`ReLu`函数描述的功能.

```cpp
void ReLuLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->kOpType == OpType::kOperatorReLu);

  const uint32_t batch_size = inputs.size();
  for (int i = 0; i < batch_size; ++i) {
    CHECK(!inputs.at(i)->empty());
    const std::shared_ptr<Tensor<float>>& input_data = inputs.at(i);

    input_data->data().transform([&](float value) {
      float thresh = op_->get_thresh();
      if (value >= thresh) {
        return value;
      } else {
        return 0.f;
      }
    });
    outputs.push_back(input_data);
  }
}
```

在`for`循环中, 首先读取输入`input_data`, 再对`input_data`使用`armadillo`自带的`transform`依次遍历其中的元素, 如果`value`的值大于`thresh`则不变, 如果小于`thresh`就返回0. 最后, 我们写一个测试函数来验证我们以上的两个类, 节点`op`类和计算层`layer`类的正确性.

## 实验环节

先判断`Forwards`返回的`outputs`是否已经保存了`ReLu`层的输出, 输出大小应该`assert`为1.  随后再进行比对, 我们应该知道在thresh等于0的情况下, 第一个输出`index(0)`和第二个输出`index(1)`应该是0, 第三个输出应该是3.f.

```cpp
TEST(test_layer, forward_ReLu) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> ReLu_op = std::make_shared<ReLuOperator>(thresh);
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  ReLuLayer layer(ReLu_op);
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}
```

