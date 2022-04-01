# 0x0. 前言
这篇文章用来了解一下MLIR中的Interfaces（接口）。MLIR是一个通用可扩展的框架，由不同层次的具有 特定属性，Operation以及Type的Dialects构成。正是由于Dialects的分层设计, 使得MLIR可以表达多种语意和抽象级别的Operation。但这个分级设计也存在一个缺点，那就是在不同的Dialect层次进行Operation转换或者做变换（Pass）的时候我们需要明确每个Dialect下的每个Operation的具体语意，否则就可能会转换或变换失败。其实基于MLIR开发过的读者应该碰到过组合一些MLIR Pass对一个MLIR文件进行Lower的时候，有可能出现Op转换失败的情况。为了缓解这种情况，MLIR提出了Interfaces。实际上在[【从零开始学深度学习编译器】十三，如何在MLIR里面写Pass？](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg) 这里我们已经利用过Interfaces来实现内联以及形状推导Pass了。这一节就更深入的了解一下MLIR中的Interfaces，最后还结合了OneFlow IR中的UserOpCompatibleInterface例子来进一步加深了解。

> 本文提到的Operation和操作是一个东西，都是MLIR Dialect下的操作。

# 0x1. 动机
Interfaces可以翻译成接口，MLIR的Interfaces提供了和IR交互的通用方式。Interfaces的设计目标是可以不用侵入到具体某个Dialect下的特定Operation和Dialect的特定知识就达到可以转换和分析MLIR表达式。这样就可以将转换，分析和新增一个Dialect和对应的Operation 进行解耦，大大增强MLIR的可扩展性。

# 0x2. Dialect Interfaces定义（细看）
Dialect Interfaces一般用在想对一组属性，Operation，类型进行通用的转换（Pass）或者分析，这些属性，Operation，类型可以是由不同的Dialect定义的。这些Interfaces一般会广泛覆盖各个级别的Dialects，仅用于少数分析和变换。因此，我们要明确Interface并不是Operation的核心，而是一些通用变换的核心。在[【从零开始学深度学习编译器】十三，如何在MLIR里面写Pass？](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg) 这里有一个使用内联Interface实现内联Pass的例子。内联通常查询的是有关Dialect中Operation的高级信息，例如cost modeling和合法性，而这些信息通常不特定于某个Dialect下的某个Operation单独存在。

Dialect Interface可以通过继承一个CRTP基类`DialectInterfaceBase::Base<>`来进行定义。CRTP的介绍可以参考：`https://zh.wikipedia.org/wiki/奇异递归模板模式`，我理解静态多态（CRTP）是因为MLIR里面会存在很多Dialect Interface要从这个`DialectInterfaceBase::Base<>`基类派生出来，为了性能考虑用CRTP比较合适。这个基类提供了Dialect Interface注册必须的一些接口，方便将来引用它们。当Interface被定义之后，Dialects就可以使用Dialect特定信息去重写它。被一个Dialect定义的Interfaces通过`addInterfaces<>`进行注册，和属性，Operation，Type的注册机制类似。下面举一个栗子：


```cpp
// 定义一个基础的内联Interface类以允许Dialect选择加入内联。 
class DialectInlinerInterface :
    public DialectInterface::Base<DialectInlinerInterface> {
public:
  /// 如果给定的区域 'src' 可以内联到该区域中，则返回 true。
  /// 'dest' 附加到注册到当前Dialect的Operation上。 
  /// 'valueMapping' 包含来自 'src' 区域内的任何重新映射的值。 
  /// 例如，这可用于检查哪些值将替换“src”区域中的条目参数。 
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    return false;
  }
};

/// 覆盖内联接口以添加对 AffineDialect 的支持以启用内联Affine Dialect的Operation。 
struct AffineInlinerInterface : public DialectInlinerInterface {
  /// Affine结构具有特定的内联约束。 
  bool isLegalToInline(Region *dest, Region *src,
                       BlockAndValueMapping &valueMapping) const final {
    ...
  }
};

/// 在Dialect下注册内联Interfaces
AffineDialect::AffineDialect(MLIRContext *context) ... {
  addInterfaces<AffineInlinerInterface>();
}
```

这些Interfaces被注册之后，在执行MLIR的变换和分析时就可以从Dialect中查到，并不需要确定特定的Dialect子类（如具体到某个Operation）。例如：

```cpp
Dialect *dialect = ...;
if (DialectInlinerInterface *interface
      = dialect->getRegisteredInterface<DialectInlinerInterface>()) {
  // Dialect提供了这个Interface的实现
  ...
}
```

例如，在`llvm/mlir/lib/IR/Dialect.cpp`这个文件中的`registerDelayedInterfaces`函数就展示了上面这种用法，这个函数用于注册加载进来的Dialect的Interfaces：

```cpp
void DialectRegistry::registerDelayedInterfaces(Dialect *dialect) const {
  auto it = interfaces.find(dialect->getTypeID());
  if (it == interfaces.end())
    return;

  // Add an interface if it is not already present.
  for (const auto &kvp : it->getSecond().dialectInterfaces) {
    if (dialect->getRegisteredInterface(kvp.first))
      continue;
    dialect->addInterface(kvp.second(dialect));
  }

  // Add attribute, operation and type interfaces.
  for (const auto &info : it->getSecond().objectInterfaces)
    std::get<2>(info)(dialect->getContext());
}
```

# 0x3. DialectInterfaceCollection（选看，我还没用过）
`DialectInterfaceCollection `提供了一个额外的实用程序。这个类允许收集已在 `MLIRContext` 实例中注册给定Interface的所有Dialect。 这对于隐藏和优化已注册Dialect Interface的查找很有用。 

```cpp
class InlinerInterface : public
    DialectInterfaceCollection<DialectInlinerInterface> {
  // 此类的钩子是DialectInlinerInterface的钩子镜像，默认实现为调用给定Dialect上Interface上的钩子。 
  virtual bool isLegalToInline(Region *dest, Region *src,
                               BlockAndValueMapping &valueMapping) const {
    auto *handler = getInterfaceFor(dest->getContainingOp());
    return handler ? handler->isLegalToInline(dest, src, valueMapping) : false;
  }
};

MLIRContext *ctx = ...;
InlinerInterface interface(ctx);
if(!interface.isLegalToInline(...))
   ...
```


# 0x4. 属性，操作，类型 Interfaces（选看）
顾名思义，属性/操作/类型Interface是在特定属性/操作/类型级别注册的那些。 这些Interface **通过提供**必须实现的虚接口来提供对派生对象的访问。 例如，许多分析和转换想要知道Operation的副作用以提高性能和正确性。 Operation的副作用通常与特定Operation的语义相关，例如 affine.load Operation具有读取效果（顾名思义）。 

这些Interface是通过覆盖特定 IR 实体的 CRTP 类来定义的； 分别是 `AttrInterface`、`OpInterface` 或 `TypeInterface`。 这些类将定义`Concept`和`Model`类的 `Traits` 类作为模板参数。 这些类提供了基于概念的多态性的实现，其中`Concept`定义了一组虚方法，这些方法被在具体实体类型上模板化的`Model`覆盖。 需要注意的是，这些类应该是纯的，不应包含非静态数据成员或其他可变数据。为了将Interface附加到对象，基类提供了一个可以附加到该对象的特征列表的 `Trait` 类（跳过下面的示例代码就可以看到解释）。 

```cpp
struct ExampleOpInterfaceTraits {
  /// 定义一个基础Concept类，指定要实现的虚拟接口。 
  struct Concept {
    virtual ~Concept();

    /// 这是对Operation的非静态钩子的示例 
    virtual unsigned exampleInterfaceHook(Operation *op) const = 0;

    /// 这是Operation的静态钩子示例。 静态钩子不需要Operation的具体实例。 实现是一个虚拟的钩子，和非静态的情况一样，因为钩  子本身的实现还是需要间接实现的。 
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  /// 为给定Operation类型的concept定义一个model类 
  template <typename ConcreteOp>
  struct Model : public Concept {
    /// 覆盖要在具体Operation上分发的方法 
    unsigned exampleInterfaceHook(Operation *op) const final {
      return llvm::cast<ConcreteOp>(op).exampleInterfaceHook();
    }

    /// 覆盖要在具体Operation上分发的方法
    unsigned exampleStaticInterfaceHook() const final {
      return ConcreteOp::exampleStaticInterfaceHook();
    }
  };
};

/// 定义分析和转换将与之交互的主Interface类。 
class ExampleOpInterface : public OpInterface<ExampleOpInterface,
                                              ExampleOpInterfaceTraits> {
public:
  /// 继承基类构造函数以支持 LLVM 样式的转换。 
  using OpInterface<ExampleOpInterface, ExampleOpInterfaceTraits>::OpInterface;

  /// Interface分发到`getImpl()`，这是一个由基本的`OpInterface`类提供的方法，它返回concept的一个实例。 
  unsigned exampleInterfaceHook() const {
    return getImpl()->exampleInterfaceHook(getOperation());
  }
  unsigned exampleStaticInterfaceHook() const {
    return getImpl()->exampleStaticInterfaceHook(getOperation()->getName());
  }
};

```

一旦定义了Interface，就可以通过添加提供的特征 `ExampleOpInterface::Trait` 将其注册到操作中，如前所述。 使用此接口就像使用任何其他派生操作类型，即强制转换： 

```cpp

/// 定义Operation时，Interface通过`OpInterface<>`基类提供的嵌套`Trait`类进行注册。
class MyOp : public Op<MyOp, ExampleOpInterface::Trait> {
public:
  /// The definition of the interface method on the derived operation.
  unsigned exampleInterfaceHook() { return ...; }
  static unsigned exampleStaticInterfaceHook() { return ...; }
};

/// 稍后，我们可以查询特定Operation（如“MyOp”）是否覆盖给定Interface。 
Operation *op = ...;
if (ExampleOpInterface example = dyn_cast<ExampleOpInterface>(op))
  llvm::errs() << "hook returned = " << example.exampleInterfaceHook() << "\n";
```

如果你读到这里并且看过之前在[【从零开始学深度学习编译器】十三，如何在MLIR里面写Pass？](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg) 使用内联Interface的例子，相信可以更加理解在Toy Dialect下注册内联Pass的几个步骤。

# 0x5. 属性、操作和类型Interfaces的外部Model（选看）
这可能需要为 IR 对象提供Interface实现而不修改所述对象的定义。 值得注意的是，这允许在定义它们的Dialect之外实现属性、操作和类型的Interface，例如，为built-in类型提供InterFace。 这是通过使用两个基于concept的多态性Model从 `Concept` 派生的扩展类来实现的，如下所示（注意注释）：


```cpp
struct ExampleTypeInterfaceTraits {
  struct Concept {
    virtual unsigned exampleInterfaceHook(Type type) const = 0;
    virtual unsigned exampleStaticInterfaceHook() const = 0;
  };

  template <typename ConcreteType>
  struct Model : public Concept { /*...*/ };

  /// 与 Model 不同，FallbackModel 将类型对象传递给
  /// 钩子，使其在方法体中可访问，即使该方法未在类本身中定义，
  /// 因此没有“this”访问权限。 ODS 自动为所有Interfaces生成此类。 
  template <typename ConcreteType>
  struct FallbackModel : public Concept {
    unsigned exampleInterfaceHook(Type type) const override {
      getImpl()->exampleInterfaceHook(type);
    }
    unsigned exampleStaticInterfaceHook() const override {
      ConcreteType::exampleStaticInterfaceHook();
    }
  };

  /// `ExternalModel` 通过显式地分离实现Interface的模型类和实现Interface的类型类，为Interface方法的默认实现提供了一个位置。 然后可以使用 `cast<ConcreteType>` 定义默认实现。 如果`ConcreteType` 没有提供默认实现所需的API，自定义实现可以直接使用`FallbackModel` 来覆盖默认实现。位于类模板中，它永远不会被实例化，也不会导致编译错误。 ODS 自动生成此类并将默认方法实现放入其中。 
  template <typename ConcreteModel, typename ConcreteType>
  struct ExternalModel : public FallbackModel<ConcreteModel> {
    unsigned exampleInterfaceHook(Type type) const override {
      // Default implementation can be provided here.
      return type.cast<ConcreteType>().callSomeTypeSpecificMethod();
    }
  };
};

```


通过派生 `FallbackMode` 或 `ExternalModel` 并通过在给定context中向相关类注册Model类，可以为属性、操作和类型接口提供外部Models。 除非注册，否则其它context将看不到该Interface。 例如：

```cpp
/// 具体类的外部Interface实现。 这不需要修改类型类本身的定义。 
struct ExternalModelExample
    : public ExampleTypeInterface::ExternalModel<ExternalModelExample,
                                                 IntegerType> {
  static unsigned exampleStaticInterfaceHook() {
    // 实现被提供在这里
    return IntegerType::someStaticMethod();
  }
  // 无需定义在“ExternalModel”中具有默认实现的“exampleInterfaceHook”。 但如果需要，它可以被覆盖。 
}

int main() {
  MLIRContext context;
  /* ... */;

  // 在使用之前，将interface model附加到给定context中的类型。 预计此时已加载包含该类型的Dialect。 
  IntegerType::attachInterface<ExternalModelExample>(context);
}
```

文档最后还提出了一个建议，即当我们“拥有”外部应用的Interface时才使用此机制。 这可以防止包含对象的Dialect的所有者和interface的所有者都不知道Interface实现的情况，这可能导致重复或发散的实现。 还没有碰到过需要使用这种机制的情况，这里不继续深入了。

# 0x6. OpInterface的Dialect Fallback（选看）
一些Dialects有一个开放的生态系统，并没有注册所有可能的Operation。 在这种情况下，仍然可以为实现这些操作的 `OpInterface` 提供支持。 当操作未注册或未提供Interface实现时，查询将fallback到Dialect本身。

第二种Model用于此类情况，并在ODS使用名为 `FallbackModel`（见下文）时自动生成。 可以为特定方言实现此Model：

```cpp
// This is the implementation of a dialect fallback for `ExampleOpInterface`.
struct FallbackExampleOpInterface
    : public ExampleOpInterface::FallbackModel<
          FallbackExampleOpInterface> {
  static bool classof(Operation *op) { return true; }

  unsigned exampleInterfaceHook(Operation *op) const;
  unsigned exampleStaticInterfaceHook() const;
};
```

然后，Dialect可以实例化此实现，并通过覆盖 `getRegisteredInterfaceForOp` 方法在特定Operation上返回它： 

```cpp
void *TestDialect::getRegisteredInterfaceForOp(TypeID typeID,
                                               StringAttr opName) {
  if (typeID == TypeID::get<ExampleOpInterface>()) {
    if (isSupported(opName))
      return fallbackExampleOpInterface;
    return nullptr;
  }
  return nullptr;
}
```

这一节也不是很懂，先记录着吧。通过上面对Interfaces的介绍，我们能留下一些基础的印象我觉得应该就够了，接下来要讲的是如何基于ODS去定义Interfaces，这才是我们这篇文章的重点。

# 0x7. 利用ODS框架定义Interface（重要）
如上所述，Interface允许属性、Operation和Type 暴露调用方法，而无需调用者知道特定的派生类型。 这种基础设施的缺点是它需要一些样板才能将所有部分连接在一起。 MLIR 提供了一种机制，用于在 ODS 中以声明方式定义接口，并自动生成 C++ 定义。 举个栗子：


```cpp
def ExampleOpInterface : OpInterface<"ExampleOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];

  let methods = [
    InterfaceMethod<
      "This is an example of a non-static hook to an operation.",
      "unsigned", "exampleInterfaceHook"
    >,
    StaticInterfaceMethod<
      "This is an example of a static hook to an operation.",
      "unsigned", "exampleStaticInterfaceHook"
    >,
  ];
}
```

提供 `AttrInterface`、`OpInterface` 或 `TypeInterface` 类的定义将自动生成接口的 C++ 类。 接口由以下组件组成： 

- C++ 类名（通过模板参数提供） 。
- 描述。(`description`)。
- C++ Namespace。（`cppNamespace`）。即Interface类应该产生在哪个C++名称空间下。
- Methods（`methods`）。由 IR 对象定义的Interfaces钩子方法列表。 
- Extra Class Declarations。可选的：`extraClassDeclaration`。在Interface类的声明中生成的附加 C++ 代码。 这允许在面向用户的Interface类上定义方法等，不需要钩到 IR 实体。 这些声明在接口方法的默认实现中不是隐式可见的，但可以使用全名限定访问静态声明。 

`OpInterface`类可能还额外包含Verifier（`verify`）。它是一个包含附加验证的 C++ 代码块应用于Interface所附加的Operation。此代码块的结构与 `Trait::verifyTrait` 方法的结构 1-1 对应。  

有两种类型的方法可以与Interface一起使用，`InterfaceMethod `和 `StaticInterfaceMethod`。 它们都由相同的核心组件组成，区别在于 `StaticInterfaceMethod` 为派生的 IR 对象上的静态方法建模。 Interface 方法有以下组件：

- Description：方法的描述信息，一个字符串。
- ReturnType：与方法的 C++ 返回类型对应的字符串。 
- MethodName：与方法的 C++ 名称对应的字符串。 
- Arguments (Optional)：分别对应于 C++ 类型和变量名称的字符串。 
- MethodBody (Optional)和DefaultImplementation。暂没有用到，以后有需要再查。

如果Operation使用 `DeclareOpInterfaceMethods` 指定Interface，则 ODS 还允许为Operation的 `InterfaceMethods` 生成声明（请参见下面的示例）。

```cpp
ef MyInterface : OpInterface<"MyInterface"> {
  let description = [{
    This is the description of the interface. It provides concrete information
    on the semantics of the interface, and how it may be used by the compiler.
  }];

  let methods = [
    InterfaceMethod<[{
      This method represents a simple non-static interface method with no
      inputs, and a void return type. This method is required to be implemented
      by all operations implementing this interface. This method roughly
      correlates to the following on an operation implementing this interface:

      ```c++
      class ConcreteOp ... {
      public:
        void nonStaticMethod();
      };
```
    }], "void", "nonStaticMethod"
    >,
    
    InterfaceMethod<[{
      This method represents a non-static interface method with a non-void
      return value, as well as an `unsigned` input named `i`. This method is
      required to be implemented by all operations implementing this interface.
      This method roughly correlates to the following on an operation
      implementing this interface:
    
      ```c++
      class ConcreteOp ... {
      public:
        Value nonStaticMethod(unsigned i);
      };
      ```
    }], "Value", "nonStaticMethodWithParams", (ins "unsigned":$i)
    >,
    
    StaticInterfaceMethod<[{
      This method represents a static interface method with no inputs, and a
      void return type. This method is required to be implemented by all
      operations implementing this interface. This method roughly correlates
      to the following on an operation implementing this interface:
    
      ```c++
      class ConcreteOp ... {
      public:
        static void staticMethod();
      };
      ```
    }], "void", "staticMethod"
    >,
    
    StaticInterfaceMethod<[{
      This method corresponds to a static interface method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:
    
      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...
    
        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(OpBuilder &builder, Location loc) const override {
            return builder.create<ConcreteOp>(loc);
          }
        }
      };
      ```
    
      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "Operation *", "create", (ins "OpBuilder &":$builder, "Location":$loc),
      /*methodBody=*/[{
        return builder.create<ConcreteOp>(loc);
    }]>,
    
    InterfaceMethod<[{
      This method represents a non-static method that has an explicit
      implementation of the method body. Given that the method body has been
      explicitly implemented, this method should not be defined by the operation
      implementing this method. This method merely takes advantage of properties
      already available on the operation, in this case its `build` methods. This
      method roughly correlates to the following on the interface `Model` class:
    
      ```c++
      struct InterfaceTraits {
        /// ... The `Concept` class is elided here ...
    
        template <typename ConcreteOp>
        struct Model : public Concept {
          Operation *create(Operation *opaqueOp, OpBuilder &builder,
                            Location loc) const override {
            ConcreteOp op = cast<ConcreteOp>(opaqueOp);
            return op.getNumInputs() + op.getNumOutputs();
          }
        }
      };
      ```
    
      Note above how no modification is required for operations implementing an
      interface with this method.
    }],
      "unsigned", "getNumInputsAndOutputs", (ins), /*methodBody=*/[{
        return $_op.getNumInputs() + $_op.getNumOutputs();
    }]>,
    
    InterfaceMethod<[{
      This method represents a non-static method that has a default
      implementation of the method body. This means that the implementation
      defined here will be placed in the trait class that is attached to every
      operation that implements this interface. This has no effect on the
      generated `Concept` and `Model` class. This method roughly correlates to
      the following on the interface `Trait` class:
    
      ```c++
      template <typename ConcreteOp>
      class MyTrait : public OpTrait::TraitBase<ConcreteType, MyTrait> {
      public:
        bool isSafeToTransform() {
          ConcreteOp op = cast<ConcreteOp>(this->getOperation());
          return op.getNumInputs() + op.getNumOutputs();
        }
      };
      ```
    
      As detailed in [Traits](Traits.md), given that each operation implementing
      this interface will also add the interface trait, the methods on this
      interface are inherited by the derived operation. This allows for
      injecting a default implementation of this method into each operation that
      implements this interface, without changing the interface class itself. If
      an operation wants to override this default implementation, it merely
      needs to implement the method and the derived implementation will be
      picked up transparently by the interface class.
    
      ```c++
      class ConcreteOp ... {
      public:
        bool isSafeToTransform() {
          // Here we can override the default implementation of the hook
          // provided by the trait.
        }
      };
      ```
    }],
      "bool", "isSafeToTransform", (ins), /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
    }]>,
  ];
}

// Operation interfaces can optionally be wrapped inside
// DeclareOpInterfaceMethods. This would result in autogenerating declarations
// for members `foo`, `bar` and `fooStatic`. Methods with bodies are not
// declared inside the op declaration but instead handled by the op interface
// trait directly.
def OpWithInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface>]> { ... }

// Methods that have a default implementation do not have declarations
// generated. If an operation wishes to override the default behavior, it can
// explicitly specify the method that it wishes to override. This will force
// the generation of a declaration for those methods.
def OpWithOverrideInferTypeInterfaceOp : Op<...
    [DeclareOpInterfaceMethods<MyInterface, ["getNumWithDefault"]>]> { ... }
```

注意：在 ODS 框架中可以通过 `OpInterfaceTrait` 类访问 C++ 中定义的现有Operation接口。 

# 0x8. Operation Interface列表
MLIR包括提供可能在许多不同Operation中通用的功能的标准Interface。 下面是一些可以被任何方言直接使用的关键Interface的列表。 每个Interface部分的标题格式如下： 

- `Interface class name`
	-  (`C++ class` – `ODS class`(if applicable))

标准Interface截图如下：

![MLIR的标准Interface类型](https://img-blog.csdnimg.cn/235f7b909da04375a1ce51928e62c0dd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAanVzdF9zb3J0,size_20,color_FFFFFF,t_70,g_se,x_16)
# 0x9. OneFlow的Interface
接下来我们以OneFlow IR为例，来看一下OneFlow Dialect中定义了哪些Interface。代码在：`oneflow/ir/include/OneFlow/OneFlowInterfaces.td` 这里。OneFlow基于ODS定义了`UserOpCompatibleInterface`，`ControlEdgeCompatibleInterface`，`NoGrad` 等Interface类型。我们以`UserOpCompatibleInterface`为例来看一下它的实现：

​```cpp
def UserOpCompatibleInterface : OpInterface<"UserOpCompatible"> {
  let description = [{
    Interface to getting the hard-coded bn
  }];

  let methods = [
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "inputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetInputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
    }]>,
    StaticInterfaceMethod<"",
        "const std::vector<std::string>*", "outputKeys", (ins), [{
        static std::vector<std::string> val(mlir::oneflow::support::GetOutputKeys(ConcreteOp::getOperationName().split('.').second.str()));
        return &val;
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSOperandIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSOperandIndexAndLength(index);
    }]>,
    InterfaceMethod<"",
        "std::pair<unsigned, unsigned>", "getODSResultIndexAndLength", (ins "unsigned":$index), [{
        return $_op.getODSResultIndexAndLength(index);
    }]>
  ];
}
```

可以看到`UserOpCompatibleInterface `使用第7节Interface ODS规范中的StaticInterfaceMethod和InterfaceMethod为这个Interface指定了获取Operation输入操作数名字，输出操作数名字，操作数以及长度，结果以及长度等方法。然后在OneFlow的`oneflow/ir/include/OneFlow/OneFlowUserOps.td`中使用`DeclareOpInterfaceMethods<UserOpCompatibleInterface>`来为各种Operation指定Interface，在生成的Operation代码中就会带上这个Interface声明。

那么这样做有什么好处吗？第一点就是由于OneFlow的UserOp都带上了UserOpCompatibleInterface，只要我们为OneFlow的UserOp实现一个通用的`GetInputKeys`函数，那么所有UserOp派生出来的Operation都拥有了这个函数的功能，因为它们都带上了UserOpCompatibleInterface这个接口。

更加通用的例子是基于InterFace来开发一些通用Pass，比如内联和形状推导Pass。见[【从零开始学深度学习编译器】十三，如何在MLIR里面写Pass？](https://mp.weixin.qq.com/s/3N9DK7aQtjoLgs-s0lP-jg)

# 0x10. 总结
这篇文章主要介绍了一下MLIR的Interface，在MLIR文档的基础上添加了一些自己的理解和描述，以及展示了OneFlow的一个例子，以此来说明Interface的好处以及如何使用ODS来写Interface。