## 我们的课程主页

https://github.com/zjhellofss/KuiperInfer 欢迎pr和点赞

[手把手教大家去写一个深度学习推理框架](https://space.bilibili.com/1822828582) B站视频课程

## Relu算子的介绍

Relu是一种非线性激活函数，它的特点有运算简单，不会在梯度处出现梯度消失的情况，而且它在一定程度上能够防止深度学习模型在训练中发生的过拟合现象。Relu的公式表达如下所示，**如果对于深度学习基本概念不了解的同学，可以将Relu当作一个公式进行对待，可以不用深究其背后的含义。**


$$
\begin{equation} f(x)=\left\{ \begin{aligned} x,x\ge thresh\\ 0,x\lt thresh \end{aligned} \right. \end{equation} \\
$$
我们今天的任务就是来完成这个公式中的操作，**值得注意的是，在我们的项目中，x和y可以理解为我们在第二、第三节中实现的张量类(tensor).**

## Operator类

Operator类就是我们在第一节中说过的计算图中**节点**的概念，计算图的另外一个概念是数据流图，如果同学们忘记了这个概念，可以重新重新翻看第一节课程。

在我们的代码中我们先定义一个**Operator**类，它是一个父类，其余的Operator，包括我们本节要实现的ReluOperator都是其派生类，**Operator中会存放节点相关的参数。**例如在**ConvOperator**中就会存放初始化卷积算子所需要的stride,  padding,  kernel_size等信息，本节的**ReluOperator**就会带有**thresh**值信息。

我们从下方的代码中来了解Operator类和ReluOperator类，它们是父子关系，Operator是基类，OpType记录Operator的类型。

```cpp
enum class OpType {
  kOperatorUnknown = -1,
  kOperatorRelu = 0,
};

class Operator {
 public:
  OpType kOpType = OpType::kOperatorUnknown;

  virtual ~Operator() = default;

  explicit Operator(OpType op_type);
};
```

   ReluOperator实现：

```cpp
class ReluOperator : public Operator {
 public:
  ~ReluOperator() override = default;

  explicit ReluOperator(float thresh);

  void set_thresh(float thresh);

  float get_thresh() const;

 private:
  float thresh_ = 0.f;
};
```

## Layer类

我们会在operator类中存放从**计算图结构文件**得到的信息，例如在ReluOperator中存放的thresh值作为一个参数就是我们从计算图结构文件中得到的，计算图相关的概念我们已经在第一节中讲过。

下一步我们需要根据ReLuOperator类去完成ReluLayer的初始化，**他们的区别在于ReluOperator负责存放从计算图中得到的节点信息，不负责计算**，而ReluLayer则**负责具体的计算操作**，同样，所有的Layer类有一个公共父类Layer. 我们可以从下方的代码中来了解两者的关系。

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

其中Layer的Forwards方法是具体的执行函数，负责将输入的inputs中的数据，进行relu运算并存放到对应的outputs中。

```cpp
class ReluLayer : public Layer {
 public:
  ~ReluLayer() override = default;

  explicit ReluLayer(const std::shared_ptr<Operator> &op);

  void Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                std::vector<std::shared_ptr<Tensor<float>>> &outputs) override;

 private:
  std::shared_ptr<ReluOperator> op_;
};
```

这是集成于Layer的ReluLayer类，我们可以看到其中有一个op成员，是一个ReluOperator指针，**这个指针中负责存放ReluLayer计算时所需要用到的一些参数**。此处op_存放的参数比较简单，只有ReluOperator中的thresh参数。

我们再看看是怎么使用ReluOperator去初始化ReluLayer的，先通过统一接口传入Operator类，再转换为对应的ReluOperator指针，最后再通过指针中存放的信息去初始化**op_**.

```cpp
ReluLayer::ReluLayer(const std::shared_ptr<Operator> &op) : Layer("Relu") {
  CHECK(op->kOpType == OpType::kOperatorRelu);
  ReluOperator *relu_op = dynamic_cast<ReluOperator *>(op.get());
  CHECK(relu_op != nullptr);
  this->op_ = std::make_shared<ReluOperator>(relu_op->get_thresh());
}
```

我们来看一下具体ReluLayer的Forwards过程，它在执行具体的计算，完成Relu函数描述的功能。

```cpp
void ReluLayer::Forwards(const std::vector<std::shared_ptr<Tensor<float>>> &inputs,
                         std::vector<std::shared_ptr<Tensor<float>>> &outputs) {
  CHECK(this->op_ != nullptr);
  CHECK(this->op_->kOpType == OpType::kOperatorRelu);

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

在for循环中，首先读取输入input_data, 再对input_data使用armadillo自带的transform按照我们给定的thresh过滤其中的元素，如果**value**的值大于thresh则不变，如果小于thresh就返回0.

最后，我们写一个测试函数来验证我们以上的两个类，节点op类，计算层layer类的正确性。先判断Forwards返回的outputs是否已经保存了relu层的输出，输出大小应该assert为1.  随后再进行比对，我们应该知道在thresh等于0的情况下，第一个输出index(0)和第二个输出index(1)应该是0，第三个输出应该是3.f.

```cpp
TEST(test_layer, forward_relu) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  ReluLayer layer(relu_op);
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}
```

## 本期代码仓库位置

```shell
git clone https://gitee.com/fssssss/KuiperCourse.git
git checkout fouth
```