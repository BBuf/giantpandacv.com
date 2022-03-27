# 0x0. 前言
这篇文章对MLIR的Pattern Rewrite机制进行翻译和总结。这几篇文档分别是`https://mlir.llvm.org/docs/PatternRewriter/` 和 `https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/`  和 `https://mlir.llvm.org/docs/Canonicalization/`。下面的第一节是阅读并翻译了这三篇文档之后的要点总结，方便读者可以快速把握这三篇文档的核心内容。

考虑到不浪费大家时间，所有的核心内容我都总结在下面第一节的几百个字，方便快速浏览。如果有需要可以收藏本文待需要使用时再查看。欢迎关注 `https://github.com/BBuf/tvm_mlir_learn` 这个仓库，我会在这里持续分享深度学习编译器相关的学习笔记，包括TVM，MLIR相关的论文阅读，工程实践等等，能点个star将更加感激。

# 0x1. 本文总览（也是全文的太长不看版）
本文介绍的Pattern Rewrite机制实际上是对[【从零开始学深度学习编译器】十九，MLIR的Pass机制实践](https://mp.weixin.qq.com/s/qmFpGtH0oB_ml0LQGPUqPA) 这篇文章的理论部分的补充。在MLIR中Pattern Rewrite机制十分重要，它不仅可以对IR做一些通用变换优化，还负责Op的规范化以及Dialect间以及Dialect内部的Op转换。因此要理解MLIR的一些关键组件，那么了解Pattern Rewrite机制是必要的。

本文的0x2节，是在介绍Pattern Rewrite机制前先介绍了一下通用的Dag Rewriter机制，Pattern Rewriter架构也是基于Dag Rewriter机制进行构建的。这一节首先说明了Dag的变换在编译器领域是十分常见的一个变换，然后MLIR由于是同一个基础架构下的多级别IR引入Dag Rewrite机制可以解决很多不同领域的问题。这里提了一下常量折叠这个例子，在MLIR里面可以通过`fold`机制来方便的做到。然后讲了一些Dag Rewriter机制的相关工作并提到MLIR的Dag Rewriter机制和LLVM DAG-to-DAG Instruction Selection Infrastructure是比较类似的，然后对LLVM DAG-to-DAG指令选择基础设施进行了介绍，介绍了一些优缺点。至此这一节就结束了，这里主要是引入了MLIR的Dag Rewriter机制，介绍了一些历史原因和相关工作等，不用太Attention。

本文的0x3节，详细介绍了MLIR中的Pattern Rewriter基础设施（通用 DAG 到 DAG 转换框架）的设计和 API。 该框架在整个 MLIR 中广泛用于规范化、转换（conversion）和通用变换。 然后介绍了Pattern的定义和应用，在Pattern的定义中提到Pattern是通过继承`RewritePattern` 类来定义的，然后构造函数的关键的成员有Benefit， Root Operation Name等。其中Benifit是匹配给定Pattern的代价，而Root Operation Name是匹配当前Pattern的Op的名字，也可以不指定但需要声明此Pattern为`MatchAnyOpTypeTag`类型以匹配任何Op类型。然后Pattern可以通过重写`match` & `rewrite` 或者 `matchAndRewrite` 函数来实现特定的Pattern匹配和重写功能。另外，还介绍了Pattern重写过程中的限制以及如何支持Pattern的递归应用以及如何调试Pattern的匹配重写过程。在这之后，介绍了`PatternRewriter`类的API列表，包含我们经常常用的`replaceOp/replaceOpWithNewOp/eraseOp`等等以及Pattern是如何被驱动程序应用的。接下来介绍了MLIR系统中集中常见的Pattern驱动程序，如Dialect Conversion Driver用于Op转换，Greedy Pattern Rewrite Driver用于规范化。在这一节的最后，还提供了Pattern Rewriter在应用过程中如何调试的技巧，可以可视化这个`match`和`rewrite`过程，功能感觉挺方便的。

本文的0x4节，介绍了MLIR的Op规范化。首先说到，在MLIR的这个多级别的IR中，任意级别都可以使用规范化Pass并且介绍了一些规范化的例子如消除恒等Op，减少操作数，常量折叠等等。接下来，文档介绍了两种定义规范化的方法：一般的Pattern Rewriter以及fold方法。具体来说，可以在Op的ODS定义时设置`hasCanonicalizeMethod` 并在源文件中提供`getCanonicalizationPatterns`方法的实现来做，也可以在Op中设置`hasFolder`字段然后在源文件中提供`fold`方法的实现来做到。具体细节可以看一下0x4节的介绍以及官方文档。

----------------------------------------------------------------------------
~~从这里开始就是细节了，如果没时间可以有需要再看，有错误欢迎指出~~

# 0x2. Generic DAG Rewriter Infrastructure Rationale
题目可以翻译为通用的Dag重写架构的基本原理。对应 `https://mlir.llvm.org/docs/Rationale/RationaleGenericDAGRewriter/` 这篇文档的内容。这里主要介绍了用于MLIR的通用Dag-to-Dag重写架构背后的基本原理。

## 0x2.1 介绍和动机
编译器IR目标是在各种抽象级别上表示代码 ，这在表示能力和易于变换方面提出了不同的折衷。 但是，表示代码的能力本身并不是很有用——您还需要能够实现这些变换。

编译器的变换有很多，这里主要介绍的是一种对MLIR目标非常重要且反复出现的变换：匹配一系列Op组成的Dag，然后将其替换为另外一个Dag。这是很多学习编译器不可或缺的一部分，对于诸如“消除identity（直连）节点”或者使用"x"替换"x+0"这种优化，通用规范化框架（比如LLVM的指令组合(`Instruction Combiner`)），以及为编译器在多个中间IR上实现优化算法提供了一个有用的抽象。

MLIR 的一个特殊优势（以及与 LLVM、GCC、XLA、TensorFlow 等其他编译器基础架构的主要区别）是它使用单个编译器 IR 来表示多个抽象级别的代码：MLIR 操作可以是“TensorFlow operation”、“XLA HLO”、仿射循环嵌套、LLVM IR 指令（可传递地包括 X86、Lanai、PTX 和其他目标特定指令）或 MLIR 算子系统可以合理表达的任何其它内容。 鉴于 MLIR 跨越了如此广泛的不同问题范围，用于执行图到图重写的单一基础架构可以帮助解决许多不同的领域挑战。

像 MLIR 这样的基于静态单赋值 (SSA) 的IR可以轻松访问Op的操作数和“users”。 因此，这些图到图重写的自然抽象是 DAG Pattern匹配的抽象：客户端定义 DAG tile模式（其中tile是定义 DAG 子图的一系列Op），并且每个Pattern都包含一个产生的结果 DAG 和产生它的成本（或者相反，叫作进行替换的好处(`benifit`)）。 一个通用的基础设施可以有效地找到并执行重写。

虽然上面提到的概念很简单，但细节很微妙。 这篇文档里定义并探索了可以解决范围广泛的不同问题的一组抽象，并预计可以应用于 MLIR 随着时间的推移将面临的许多不同类型的问题。

## 常量折叠（Constant Folding）
DAG 到 DAG Pattern匹配的一个退化但常见的情况是常量折叠：操作数包含常量的Op通常可以折叠为结果常量值。 

MLIR 的Op可能会覆盖`fold`来实现，与一般的 DAG 到 DAG Pattern匹配器相比，它暴露了一个更简单的 API，并适用于通用的匹配器不适用的情况。 例如，DAG 重写可以删除当前函数中的任意节点，这可能会使迭代器无效。 作为 API 的常量折叠则不会删除任何节点，它只是提供一个常量值（列表）并允许客户端根据需要更新其数据结构。

关于常量折叠请看一下后面的0X4节的示例讲解，是这篇`https://mlir.llvm.org/docs/Canonicalization` 文档的翻译。 


## 相关工作
考虑到几乎每个现有的编译器都必须多次解决这个问题，因此需要考虑大量相关工作。 一个统一的问题是，所有这些系统都旨在解决一个特定的、通常是狭窄的问题：另一方面，MLIR 希望在单个基础设施中解决许多这些问题。 以下是一些相关的Pattern Rewriter系统，以及它们工作的优缺点（与 MLIR 中存在的基础设施最相似的设计是 LLVM DAG-to-DAG 指令选择算法）。 

- AST 级Pattern匹配器：文本中存在大量的source-to-source的翻译器用来做等价变换以提升性能（比如把x*0变成0）。一个较大的例子是GCC fold函数，它对AST进行了很多优化。Clang具有应用于表达式的简单常量折叠的类似例子（C++的要求），但并不会对AST执行常见的优化。
AST 优化器的主要缺点是我们无法看到具有多种用途的Op。 众所周知，DAG Pattern匹配比树Pattern匹配更强大，但另一方面，DAG Pattern匹配会导致重复计算。 
- 第二种就不介绍了，感兴趣可以看官方文档。
- LLVM’s DAG-to-DAG Instruction Selection Infrastructure：LLVM 中的指令选择子系统是多年迭代和研究的结果，这是由于 LLVM 需要支持大量的目标代码生成、现代指令集（例如 X86）的代码生成器的复杂性以及狂热的追求跨目标重用代码。 Eli Bendersky 写了一篇关于它如何工作的简短概述，LLVM 文档更深入地描述了它，包括它的优点和局限性。 它允许编写这样的Pattern。 

```cpp
def : Pat<(or GR64:$src, (not (add GR64:$src, 1))),
          (BLCI64rr GR64:$src)>;
```
此示例为 X86 目标描述中的“blci”指令定义了一个匹配器，该文件中还有许多其他指令（查找 `Pat<>` Pattern，因为它们没有纠缠于编译器的细节，如汇编器/反汇编器生成逻辑）。

下面说了一些LLVM的这个DAG-to-DAG 指令选择机制的好处和坏处，截图放在下方。

![LLVM的这个DAG-to-DAG 指令选择机制的特点](https://img-blog.csdnimg.cn/a558224aa4f74b2184d070a8088e097d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)

## 小结
MLIR 面临着广泛的Pattern匹配和图重写问题，在多个级别上使用通用代码表示的主要优势之一是它允许投资并高度利用单一基础设施来完成此类工作。

这里后续还介绍了一些Dag重写机制的目标，包括它解决了哪些问题以及使用的匹配策略，以及良好的报错信息等等。


# 0x3. Pattern Rewriting : Generic DAG-to-DAG Rewriting

本文档详细介绍了 MLIR中存在的Pattern Rewriter基础设施（通用 DAG 到 DAG 转换框架）的设计和 API。 该框架在整个 MLIR 中广泛用于规范化、转换（conversion）和通用变换（transformation）。 

## 介绍
Pattern Rewriter框架在很大程度上可以分解为两部分：Pattern定义和Pattern应用。 

## Pattern定义
Pattern是通过继承 `RewritePattern` 类来定义的。 此类表示 MLIR 中所有Rewrite Pattern的基类，由以下组件组成： 

### Benefit
这是应用给定Pattern的预期好处。 这种好处在Pattern构建时是静态的，但可以在Pattern初始化时动态计算，例如允许从特定领域的信息（如目标架构）中获得好处。 这种限制允许执行Pattern融合并将Pattern编译成一个高效的状态机，并且 Thier、Ertl 和 Krall 已经证明，匹配谓词在几乎所有情况下都不需要动态的计算成本：我们可以简单地为每个可能的好处实例化一次相同的Pattern，并使用谓词来保护匹配。 

### Root Operation Name（可选） 
此Pattern匹配的根操作的名称。如果指定，只有具有给定根名称的Op才需要提供`match`和`rewrite`实现。 如果没有指定，可以提供任何操作类型。 应尽可能提供根操作名称，因为它可以在应用代价模型时简化Pattern分析。 要匹配任何Op类型，必须提供一个特殊标签来明确意图：`MatchAnyOpTypeTag`。 

#### match and rewrite 实现
这是与给定根操作匹配并执行 IR 重写的代码块。 `RewritePattern` 可以通过单独的 `matc`h 和 `rewrite` 方法或通过组合的 `matchAndRewrite` 方法来指定此实现。 使用组合 `matchAndRewrite` 方法时，在匹配成功之前不应发生 IR 突变。 当匹配和重写阶段需要non-trivially的可重计算信息时，组合的 `matchAndRewrite` 很有用。 请参阅下面的示例： 

```cpp
class MyPattern : public RewritePattern {
public:
  /// This overload constructs a pattern that only matches operations with the
  /// root name of `MyOp`.
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
  /// This overload constructs a pattern that matches any operation type.
  MyPattern(PatternBenefit benefit)
      : RewritePattern(benefit, MatchAnyOpTypeTag()) {}

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using the separate hooks.
  LogicalResult match(Operation *op) const override {
    // The `match` method returns `success()` if the pattern is a match, failure
    // otherwise.
    // ...
  }
  void rewrite(Operation *op, PatternRewriter &rewriter) {
    // The `rewrite` method performs mutations on the IR rooted at `op` using
    // the provided rewriter. All mutations must go through the provided
    // rewriter.
  }

  /// In this section, the `match` and `rewrite` implementation is specified
  /// using a single hook.
  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) {
    // The `matchAndRewrite` method performs both the matching and the mutation.
    // Note that the match must reach a successful point before IR mutation may
    // take place.
  }
};
```

### 限制
在Pattern的`match`部分中，应用以下约束： 
- 不允许IR突变。

在Pattern的`rewrite`部分中，应用以下约束： 
- 所有 IR 突变，包括创建，都必须由给定的 `PatternRewriter` 执行。 此类提供了用于执行Pattern中可能发生的所有可能突变的钩子。 例如，这意味着不应通过其`erase`方法来删除操作。 要删除操作，应使用适当的 `PatternRewriter` 钩子（在本例中为 `eraseOp`）。 
- 根操作必须是：inplace更新、替换或删除。

### 递归应用
递归是Pattern重写上下文中的一个重点主题，因为一个Pattern通常对自己的结果也是适用的。但递归也可能将Pattern匹配过程陷入死循环。Pattern重写基础设施保守的假设没有Pattern存在递归，如果检测到递归将发出失败信号。如果一个Pattern支持递归，则需要在Pattern初始化时显示的调用`setHasBoundedRewriteRecursion`发出信号来表明该Pattern的递归应用可能会发生，并且该Pattern可以安全的处理。


### Debug Names and Labels
为了帮助调试，Pattern可以指定： 调试名称（通过 `setDebugName`），它应该对应于唯一标识特定Pattern的标识符； 和一组调试标签（通过 `addDebugLabels`），它们对应于唯一标识Pattern组的标识符。 各种实用程序使用此信息来帮助调试Pattern重写，例如 在调试日志中，提供Pattern过滤等。一个简单的代码示例如下所示： 


```cpp
class MyPattern : public RewritePattern {
public:
  /// Inherit constructors from RewritePattern.
  using RewritePattern::RewritePattern;

  void initialize() {
    setDebugName("MyPattern");
    addDebugLabels("MyRewritePass");
  }

  // ...
};

void populateMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  // Debug labels may also be attached to patterns during insertion. This allows
  // for easily attaching common labels to groups of patterns.
  patterns.addWithLabel<MyPattern, ...>("MyRewritePatterns", ctx);
}
```

### 初始化
一些Pattern状态需要Pattern显式初始化，例如，如果Pattern可以安全地处理递归应用程序，则设置 `setHasBoundedRewriteRecursion`。 此Pattern状态可以在Pattern的构造函数中初始化，也可以通过实用程序`initialize` hook进行初始化。 使用 `initialize` hook不需要重新定义Pattern构造函数来注入额外的Pattern状态初始化。 一个例子如下所示： 

```cpp
class MyPattern : public RewritePattern {
public:
  /// Inherit the constructors from RewritePattern.
  using RewritePattern::RewritePattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  // ...
};
```
### 构造
应该使用静态 `RewritePattern::create<T>` 实用程序方法来构造 RewritePattern。 此方法可确保正确初始化Pattern并准备好插入 `RewritePatternSet`。


## Pattern Rewriter
`PatternRewriter` 是一个特殊的类，它允许Pattern与Pattern应用程序的驱动程序进行通信。 如上所述，所有 IR 突变，包括创建，都需要通过 `PatternRewriter` 类执行。 这是必需的，因为底层Pattern驱动程序可能具有在发生突变时会失效的状态。 下面显示了一些更流行的 `PatternRewriter` API 的示例，请参阅类文档（`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/PatternMatch.h#L235`）以获取可用 API 的更新列表： 

- 删除一个Op：`eraseOp`

此方法擦除没有结果或结果都已知无用的Op。（在IR里悬空的Op） 

- 通知`match`失败：`notifyMatchFailure` 

此方法允许在 `matchAndRewrite` 中提供关于Pattern无法匹配的原因的诊断消息。 该消息如何显示给用户取决于特定的Pattern驱动程序。 

- 代替一个Op。`replaceOp/replaceOpWithNewOp`

此方法用一组提供的值替换Op的结果，并删除这个Op。

 - 本地更新一个Op：`(start|cancel|finalize)RootUpdate`

这是一组方法，它们提供了一个 transaction-like的 API，用于在Pattern中就地更新操作的属性、位置、操作数或后继者。 就地更新transaction由 `startRootUpdate` 启动，可以分别用 `cancelRootUpdate` 和 `finalizeRootUpdate` 取消或终止。 还提供了一个wapper `updateRootInPlace` 将`start`和`finalize`包装为一个回调。

- OpBuilder API

`PatternRewriter` 继承自 `OpBuilder` 类，因此提供了 `OpBuilder` 中存在的所有相同功能。 这包括Op创建，以及许多有用的属性和类型构造方法。 


## Pattern Application
在定义了一组Pattern后，将它们收集起来并提供给特定的驱动程序以供应用。 一个驱动程序由几个高级部分组成：
 - Input `RewritePatternSet`。
驱动程序的输入Pattern以 `RewritePatternSet` 的形式提供。 此类提供了用于构建Pattern列表的简化 API。 
- Driver-specific `PatternRewriter`。
为了确保驱动程序状态不会因Pattern Rewriter中的 IR 突变而失效，驱动程序必须提供一个 `PatternRewriter` 实例，其中覆盖了必要的hook。 如果驱动程序不需要挂钩某些突变，则会提供一个默认实现来直接执行突变。 
- Pattern Application and Cost Model
每个驱动程序负责定义自己的Op访问顺序以及Pattern代价模型，但最终应用程序是通过 `PatternApplicator` 类执行的。 此类将 `RewritePatternSet` 作为输入，并根据提供的代价模型变换Pattern。 该成本模型使用任何必要的驱动程序特定信息计算给定Pattern的最终代价。 在计算成本模型后，驱动程序可以开始使用 `PatternApplicator::matchAndRewrite` 将Pattern与Op匹配。


```cpp
class MyPattern : public RewritePattern {
public:
  MyPattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MyOp::getOperationName(), benefit, context) {}
};

/// Populate the pattern list.
void collectMyPatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MyPattern>(/*benefit=*/1, ctx);
}

/// Define a custom PatternRewriter for use by the driver.
class MyPatternRewriter : public PatternRewriter {
public:
  MyPatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

/// Apply the custom driver to `op`.
void applyMyPatternDriver(Operation *op,
                          const RewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.
  MyPatternRewriter rewriter(op->getContext());

  // Create the applicator and apply our cost model.
  PatternApplicator applicator(patterns);
  applicator.applyCostModel([](const Pattern &pattern) {
    // Apply a default cost model.
    // Note: This is just for demonstration, if the default cost model is truly
    //       desired `applicator.applyDefaultCostModel()` should be used
    //       instead.
    return pattern.getBenefit();
  });

  // Try to match and apply a pattern.
  LogicalResult result = applicator.matchAndRewrite(op, rewriter);
  if (failed(result)) {
    // ... No patterns were applied.
  }
  // ... A pattern was successfully applied.
}
```

## 常见的Pattern驱动程序
MLIR 提供了几种常见的Pattern驱动程序，可服务于各种不同的用例。 

### Dialect Conversion Driver
该驱动程序提供了一个框架，使用“legality”概念在Dialect之间和Dialect内执行Op转换。 该框架允许通过一组基于Pattern的Op重写Pattern将非法Op转换为提供的转换目标支持的Op。 该框架还提供对类型转换的支持。 可以在此处找到有关此驱动程序的更多信息。 `https://mlir.llvm.org/docs/DialectConversion/` 。

### Greedy Pattern Rewrite Driver
该驱动程序遍历提供的Op并贪婪地应用本地最有利的pattern。 pattern的好处完全取决于模式上指定的`benifit`，以及pattern列表中pattern的相对顺序（当两个模式具有相同的局部好处时）。 pattern迭代地应用于Op，直到达到一个固定点，此时驱动程序完成。 此驱动程序可通过以下方式使用：`applyPatternsAndFoldGreedily` 和 `applyOpPatternsAndFold`。 后者仅将Pattern应用于提供的操作，并且不会遍历 IR。 

驱动程序是可配置的，并支持两种模式：1）您可以选择“自上而下”遍历，遍历Op产生工作列表，并在区域树上预先排序。 这通常在编译时更有效。 2) 默认是“自下而上”遍历，它使用区域树的后序遍历来构建初始工作列表。 这可能会将较大的Pattern与模棱两可的Pattern集合相匹配。 

这种驱动程序仅仅在MLIR的规范化Pass中使用，也就是本文0x4节介绍的内容。

### Debugging
要调试Greedy Pattern Rewrite驱动程序的执行，可以使用`-debug-only=greedy-rewriter`。 此命令行标志仅为Greedy Pattern Rewriter激活 LLVM 的调试日志基础设施。 输出被格式化为树结构，反映了Pattern应用过程的结构。 此输出包含Rewriter执行的所有Op、如何处理Op和应用Pattern以及它们失败的原因。 

```cpp
//===-------------------------------------------===//
Processing operation : 'cf.cond_br'(0x60f000001120) {
  "cf.cond_br"(%arg0)[^bb2, ^bb2] {operand_segment_sizes = dense<[1, 0, 0]> : vector<3xi32>} : (i1) -> ()

  * Pattern SimplifyConstCondBranchPred : 'cf.cond_br -> ()' {
  } -> failure : pattern failed to match

  * Pattern SimplifyCondBranchIdenticalSuccessors : 'cf.cond_br -> ()' {
    ** Insert  : 'cf.br'(0x60b000003690)
    ** Replace : 'cf.cond_br'(0x60f000001120)
  } -> success : pattern applied successfully
} -> success : pattern matched
//===-------------------------------------------===//
```

此输出描述 `cf.cond_br` Op的处理。 我们首先尝试应用 `SimplifyConstCondBranchPred`，但失败了。 从那里，另一个Pattern (`SimplifyCondBranchIdenticalSuccessors`) 被应用匹配 `cf.cond_br` 并将其替换为 `cf.br`。

## 调试
### Pattern Filtering

为了简化测试用例的定义和缩减，`FrozenRewritePatternSet` 类提供了内置支持来过滤哪些Pattern应该提供给应用程序的Pattern驱动程序。 在构造 `FrozenRewritePatternSet` 时，通过提供 `disabledPatterns` 和 `enabledPattern`s 列表来指定过滤行为。 `disabledPatterns` 列表应该包含一组在Pattern应用期间禁用的Pattern的调试名称或标签，即应该过滤掉哪些Pattern。 `enabledPatterns` 列表应该包含一组在模式应用期间启用的Pattern的调试名称或标签，不满足此约束的Pattern将被过滤掉。 请注意，由 `disabledPatterns` 列表指定的模式将被过滤掉，即使它们与 `enabledPatterns` 列表中的条件匹配。 一个例子如下所示： 

```cpp
void MyPass::initialize(MLIRContext *context) {
  // No patterns are explicitly disabled.
  SmallVector<std::string> disabledPatterns;
  // Enable only patterns with a debug name or label of `MyRewritePatterns`.
  SmallVector<std::string> enabledPatterns(1, "MyRewritePatterns");

  RewritePatternSet rewritePatterns(context);
  // ...
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```

### Common Pass Utilities
利用rewrite patterns的pass应该旨在提供一组通用的选项和切换，以简化在不同passes/projects/等之间切换时的调试体验。 为了帮助完成这项工作，MLIR 提供了一组通用实用程序，可以在自定义pass时轻松包含这些实用程序。 这些在 `mlir/RewritePassUtil.td` 中定义； 示例用法如下所示： 

```cpp
def MyRewritePass : Pass<"..."> {
  let summary = "...";
  let constructor = "createMyRewritePass()";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.
  let options = RewritePassUtils.options;
}
```

#### Rewrite Pass Options
本节记录了可用于控制 rewrite pattern应用程序行为的常见pass选项。 

##### Pattern Filtering
公开了两个常见的Pattern过滤选项，`disable-patterns` 和 `enable-patterns`，与上述模式过滤部分中描述的 `disabledPatterns` 和 `enabledPatterns` 列表的行为相匹配。 这些选项的 `tablegen` 定义片段如下所示：

 

```cpp
ListOption<"disabledPatterns", "disable-patterns", "std::string",
           "Labels of patterns that should be filtered out during application",
           "llvm::cl::MiscFlags::CommaSeparated">,
ListOption<"enabledPatterns", "enable-patterns", "std::string",
           "Labels of patterns that should be used during application, all "
           "other patterns are filtered out",
           "llvm::cl::MiscFlags::CommaSeparated">,
```

这些选项可用于在pass中构造任何 `FrozenRewritePatternSet` 时提供过滤行为： 

```cpp
void MyRewritePass::initialize(MLIRContext *context) {
  RewritePatternSet rewritePatterns(context);
  // ...

  // When constructing the `FrozenRewritePatternSet`, we provide the filter
  // list options.
  frozenPatterns = FrozenRewritePatternSet(rewritePatterns, disabledPatterns,
                                           enabledPatterns);
}
```


# 0x4. Operation Canonicalization(操作规范化)
规范化是编译器 IR 设计的重要组成部分：它使实现可靠的编译器转换和推理代码中的优劣变得更加容易，并引发了有关特定 IR 级别目标的有趣讨论。 Dan Gohman 写了一篇文章探讨这些问题； 如果你不熟悉这些概念，则值得一读。文章地址为：`https://sunfishcode.github.io/blog/2018/10/22/Canonicalization.html` 。

 大多数编译器都有规范化pass，有时它们还有许多不同类型的pass（例如 LLVM 中的 instcombine、dag combine 等）。 因为 MLIR 是一个多级 IR，我们可以提供一个单一的规范化基础设施，并在它所代表的许多不同的IR中重用它。这一节描述了通用的全局规范化方法，并提供了部分用来捕获特定于IR的规则以供参考。 

## 通用设计
MLIR 有一个单一的规范化pass，它以贪心的方式迭代地应用规范化变换，直到IR收敛。 这些变换由Op本身定义，允许每个方言一起定义自己的Op和规范化集合。规范化Pattern需要考虑的几点：
- Pattern的重复应用应该收敛。 不稳定或循环重写将导致规范化程序中的无限循环。 
- 当操作数重复时，朝着值使用较少的Op进行规范化通常会更好，因为某些Pattern仅在值具有单个user时才匹配。 例如，将“x + x”规范化为“x * 2”通常是好的，因为这会将 x 的使用次数减少一。 
- 在可能的情况下完全消除Op总是好的，例如 通过折叠已知的恒等（如“x + 0 = x”）。 

## 全局应用规则
这些变换被应用于所有级别的IR：
-  消除无副作用、无用处的Op。
-  常量折叠 - 例如 “(addi 1, 2)”到“3”。 常量折叠钩子由Op指定。
-  将常量操作数移动到右侧的可交换运算符 - 例如 “(addi 4, x)”到“(addi x, 4)”。 
- `constant-like` Op是唯一的，并被提升到第一个父barrier区域的入口块中。这是一个和上方隔离的区域，如函数的入口块，或者通过`DialectFoldInterface`上的`shouldMaterializeInto`方法标记为barrier的入口块。


## 定义Canonicalizations
有两种机制可用于定义规范化； 一般的 RewritePatterns 和 fold 方法。 
### Canonicalizing with RewritePattern
这种机制允许将规范化作为一组 RewritePatterns 提供，或者在 C++ 中强制定义或作为声明性重写规则（DRR）声明。 Pattern Rewriter基础结构允许表达许多不同类型的规范化。 这些转换可能就像用移位替换乘法一样简单，甚至可以用无条件分支替换条件分支。 

在ODS中，Op可以通过设置`hasCanonicalizer`位或者`hasCanonicalizeMethod`位以生成`getCanonicalizationPatterns`方法。

```cpp
def MyOp : ... {
  // I want to define a fully general set of patterns for this op.
  let hasCanonicalizer = 1;
}

def OtherOp : ... {
  // A single "matchAndRewrite" style RewritePattern implemented as a method
  // is good enough for me.
  let hasCanonicalizeMethod = 1;
}
```
然后可以在源文件中提供规范化Pattern：


```cpp
void MyOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                       MLIRContext *context) {
  patterns.add<...>(...);
}

LogicalResult OtherOp::canonicalize(OtherOp op, PatternRewriter &rewriter) {
  // patterns and rewrites go here.
  return failure();
}

```
### Canonicalizing with `fold` 方法
`fold`机制是一种有意限制但功能强大的机制，它允许在整个编译器的许多地方应用规范化。例如，在规范化pass之外 ,`fold`在Dialect Conversion基础架构中用作合法化机制，并且可以通过`OpBuilder::createOrFold`在任何地方使用`OpBuilder`直接调用。

`fold` 的限制是不能创建新的Op，只能替换根Op（但不能删除）。 它允许原地更新Op，或返回一组预先存在的值（或属性）以替换Op。 这确保了`fold`方法是一个真正的“原地”转换，并且可以在不需要Pattern Rewriter的情况下调用。

在 ODS 中，Op可以设置`hasFolder`位以生成`fold`方法的声明。 此方法采用不同的形式，具体取决于Op的结构。

```cpp
def MyOp : ... {
  let hasFolder = 1;
}
```

如果Op只有一个结果，将生成以下内容： 

```cpp
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return nullptr.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return the operation itself.
///  3. They can return an existing value or attribute that can be used instead
///     of the operation. The caller will remove the operation and use that
///     result instead.
///
OpFoldResult MyOp::fold(ArrayRef<Attribute> operands) {
  ...
}
```
否则将生成下面的内容：

```cpp
/// Implementations of this hook can only perform the following changes to the
/// operation:
///
///  1. They can leave the operation alone and without changing the IR, and
///     return failure.
///  2. They can mutate the operation in place, without changing anything else
///     in the IR. In this case, return success.
///  3. They can return a list of existing values or attribute that can be used
///     instead of the operation. In this case, fill in the results list and
///     return success. The results list must correspond 1-1 with the results of
///     the operation, partial folding is not supported. The caller will remove
///     the operation and use those results instead.
///
/// Note that this mechanism cannot be used to remove 0-result operations.
LogicalResult MyOp::fold(ArrayRef<Attribute> operands,
                         SmallVectorImpl<OpFoldResult> &results) {
  ...
}
```

在上面，为每个方法提供了一个 `ArrayRef<Attribute>`，它对应于每个操作数的常量属性值。 这些操作数是那些实现 `ConstantLike` 特征的操作数。 如果任何操作数是非常量，则提供 null `Attribute` 值。 例如，如果 MyOp 提供了三个操作数 [a, b, c]，但只有 b 是常量，则操作数的格式为 [Attribute(), b-value, Attribute()]。 

上面还展示了`OpFoldResult`的应用。此类表示`fold`一个op的可能结果：SSA `Value`或`Attribute`（对于常量结果）。 如果提供了 SSA `Value`，则它必须对应于现有值。 `fold` 方法不允许生成新`Value`。 返回的 `Attribute` 值的形式没有特定的限制，但重要的是要确保特定 `Type` 的 `Attribute` 表示形式是一致的。 

当Op上的`fold`钩子不成功时，Dialect可以通过实现 `DialectFoldInterface` 并覆盖`fold`钩子来提供fallback。


### 从属性产生常量
当 `fold` 方法返回一个 `Attribute` 作为结果时，它表示这个结果是“常量”。 `Attribute`是值的常量表示。 `fold` 方法的使用者，例如 canonicalizer pass，将获取这些 `Attributes` 并在 IR 中实现常量Op来表示它们。 要启用此实现，Op的Dialect必须实现 `materializeConstant` 钩子。 这个钩子接受一个`Attribute`值，通常由`fold`返回，并产生一个“constant-like”的Op来表示该值。

 在 ODS 中，Dialect可以设置 `hasConstantMaterializer` 位以生成 `materializeConstant` 方法的声明。


```cpp
def MyDialect_Dialect : ... {
  let hasConstantMaterializer = 1;
}
```

然后可以在源文件中具体化常量： 

```cpp
/// Hook to materialize a single constant operation from a given attribute value
/// with the desired resultant type. This method should use the provided builder
/// to create the operation without changing the insertion position. The
/// generated operation is expected to be constant-like. On success, this hook
/// should return the value generated to represent the constant value.
/// Otherwise, it should return nullptr on failure.
Operation *MyDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                          Type type, Location loc) {
  ...
}
```

# 0x5. 总结
本文对MLIR的Pattern Rewriter相关的几篇文章做了一遍拉通梳理和总结，希望可以帮助学习MLIR的朋友。