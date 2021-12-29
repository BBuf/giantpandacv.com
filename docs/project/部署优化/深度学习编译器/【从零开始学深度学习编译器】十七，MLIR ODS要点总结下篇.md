# 前言
这一节在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 的基础上补充完整了ODS的要点。约束和属性的定义都是MLIR中相当重要的元素，至于类型的定义个人认为了解即可，等到我们需要自定义类型的时候再仔细研究。最后MLIR的语法比较晦涩，初学者可以借助`mlir-tblgen`来辅助debug。

在这两篇文章里，我跟着MLIR的ODS规范完整走了一遍并总结了14个要点，对于每一个要点我都在OneFlow MLIR的Op定义中进行了对照，并给出了一些示例代码和位置。希望对读者入门MLIR有帮助。

# 11. 约束（这个很重要）
约束(Constraint)是表驱动Operation定义中的一个核心概念：Operation验证和图Operation匹配都是基于约束来做的。因此，Operation定义和重写规则都直接涉及写入约束。MLIR在`OpBase.td`(`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`)中定义了`Constraint`基类。一个Operation的约束可以覆盖不同的范围，可能是：

- 仅关注单个属性（例如大于 5 的 32 位整数）
- 多个操作数和结果（例如，第一个结果的形状必须与第一个操作数（可理解为Tensor）相同）
- 操作本身固有的。（例如没有副作用，参考Transpose Op消除那个案例）

我们将它们分别称为单实体约束、多实体约束和特征。这里的概念了解下即可，我觉得写新的约束是最重要的。

- **单体约束**。单体约束作用域为单个操作数，属性或结果的约束在实体的声明位置进行指定，如**Operation arguments** 和 **Operation results** 中（在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 中总结了Operation arguments和Operation results需要注意的知识）。
- **多实体约束**。多实体约束在`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`中被建模为`PredOpTrait`类（是`OpTrait`的一个子类）。查看`OpBase.td`获取完整列表。
- **特征**。特征是Operation的内在属性，例如是否具有副作用、可交换与否、是否是终止符等。这些约束应指定为 Op 类模板参数，如[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 中第三节的Op的特征和约束(Operation traits and constraints) 所示。特征在`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td`中被建模成一个`NativeOpTrait`类（`OpTrait`的一个子类）。 它们得到支持并将被翻译成相应的 C++ `mlir::OpTrait` 类。 

- **如何指定新的约束**？要写一个新的约束，我们必须为它提供一个谓词并指定一个描述名。使用`Pred`类建模的谓词是构成约束的核心。约束的谓词通常以嵌套的方式构建，有两种类型的谓词：1.`CPred`：原始的叶子节点谓词。2.复合谓词：由使用谓词组合器的子谓词组成的谓词（conjunction: `And`, disjunction: `Or`, negation: `Neg`, substitution: `SubstLeaves`, concatenation: `Concat`）。`CPred` 是构成更复杂谓词的基础。 它是TableGen 视角下的“原子”谓词，是TableGen 与C++ 之间的“接口”。 里面已经是 C++ 代码了，它会被当作不透明的字符串来处理，并带有特殊的占位符来替换。 我们可以将任何返回布尔值的 C++ 代码放在 `CPred` 中，包括计算表达式、调用函数、调用类方法等。 

为了帮助与 C++ 环境交互，提供了一些特殊的占位符来引用使用该谓词的上下文中的实体。 它们充当封闭环境的“钩子”。 这包括 `$_builder`、`$_op` 和 `$_self`：

- `$_builder`会被替换成一个`mlir::Builder`实例，以便我们可以访问常见的构建方法。
- `$_op` 会被当前的Operation替换，以便我们可以访问当前Operation的信息。
- `$_self` 会被替换为该谓词所附加的实体。 例如，`BoolAttr` 是一个包含 `CPred<"$_self.isa<BoolAttr>()">` 的属性约束。 那么对于 `BoolAttr:$attr`，`$_self` 将被 `$attr` 替换。 对于类型约束，它有点特殊，因为我们希望每个类型定义的约束自然读取，并且我们希望将类型约束直接附加到操作数/结果，`$_self` 将被操作数/结果的类型替换。 例如，对于 `F32:$operand` 中的 `F32`，它的 `$_self` 将被扩展为`operand(...).getType()`。

例如，要写一个属性 `attr` 是一个 `IntegerAttr`，在 C++ 中我们可以调用 `attr.isa<IntegerAttr>()`来实现。 这行代码也可以作为 `$_self.isa<IntegerAttr>()` 包装在 `CPred` 中，其中 `$_self` 作为特殊占位符，在扩展时由当前属性 `attr` 替换来实现相同的功能（指在Tablegen中）。

对于更复杂的谓词，我们可以将其包装在单个 `CPred` 中，也可以使用谓词组合器将它们组合起来。 例如，要写出属性 `attr` 是 32 位或 64 位整数的约束，可以将其写为：

```cpp
And<[
  CPred<"$_self.isa<IntegerAttr>()">,
  Or<[
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(32)">,
    CPred<"$_self.cast<IntegerAttr>().getType().isInteger(64)">
  ]>
]>
```
（注意，上面只是用一个熟悉的例子来展示如何使用`CPred`和谓词组合器来编写复杂的谓词。具体来说，对于整数属性，`OpBase.td`已经定义了`I32Attr`和`I64Attr`。所以我们实际上可以重用它们来编写它 `Or<[I32Attr.predicate, I64Attr.predicate]>`.)

这里再以OneFlow的一个例子来讲解一下，我们定义了一个IsGPU的约束：

```cpp
def IsGPU: Constraint<CPred<"$0.getValue().equals(\"gpu\")">, "is GPU device">;
```

然后OneFlow在Transformer部分做了一个定制优化，就是将Scale和Tril这两个连续的Kernel融合成一个大的Kernel，这样可以省掉一部分内存读写的时间。但这个融合的kernel只在GPU的情况下生效，所以这个时候就需要判断当前计算图检测到的Scale和Tril这两个Operation的device是否是GPU的，就需要这个约束。FusedScaleTrilPattern这个Pass的实现如下，可以看到在最后使用了IsGPU这个约束。

```cpp
def FusedScaleTrilPattern : Pat<
  (
    OneFlow_TrilOp
    (
      OneFlow_ScalarMulOp
        $x,
        $scale_op_name,
        $scale_trainable,
        $scale_device_tag,
        $scale_device_name,
        $scale_scope_symbol_id,
        $scale_hierarchy,
        $has_int_operand,
        $has_float_operand,
        $int_operand,
        $float_operand
    ),
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value
  ),
  (OneFlow_FusedScaleTrilOp $x,
    $tril_op_name,
    $tril_trainable,
    $tril_device_tag,
    $tril_device_name,
    $tril_scope_symbol_id,
    $tril_hierarchy,
    $diagonal,
    $floating_fill_value,
    $integer_fill_value,
    $is_floating_fill_value,
    $float_operand,
    $int_operand,
    $has_float_operand
  ),
  [
    (IsGPU $tril_device_tag),
    (IsGPU $scale_device_tag)
  ]
>;
```

这个Pass的功能就是检测到连续的Scale+Tril Operation就将这两个Operation融合成一个FusedScaleTril Operation。

如果谓词用 `CPred` 和谓词组合器一起编写非常复杂，我们也可以将其编写为普通的 C++ 函数，并使用 `CPred` 作为“调用”函数的一种方式。 例如，要验证属性 `attr` 是否具有某些属性，我们可以编写一个 C++ 函数，如：

```cpp
bool HasSomeProperty(Attribute attr) { ... }
```

然后定义Op如下：

```cpp
def HasSomeProperty : AttrConstraint<CPred<"HasSomeProperty($_self)">,
                                     "has some property">;

def MyOp : Op<...> {
  let arguments = (ins
    ...
    HasSomeProperty:$attr
  );
}
```

至于我们是否应该使用单个 `CPred` 包装整个表达式、多个带有谓词组合器的 `CPreds` 或单个 `CPred` “调用”一个函数来定义谓词，没有明确的标准。 使用 `CPred` 和谓词组合器进行定义是可取的，因为它将更多信息（而不是隐藏 C++ 函数背后的所有逻辑）公开到操作定义规范中，以便它可以潜在地驱动更多的自动生成案例。 但它需要一个很好的通用谓词库作为构建块，以避免重复，目前正在研究中。 

# 12. 属性定义（很重要+1）
属性是编译期就知道的Operation的常量。ODS 在 C++ 属性类上提供属性包装器。 MLIR 的核心 IR 库中定义了一些常见的 C++ 属性类（`https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Attributes.h`）。ODS 允许在 TableGen 中使用这些属性来定义Operation，可能具有更细粒度的约束。 比如`StrAttr`直接映射到`StringAttr`； `F32Attr/F64Attr` 要求 `FloatAttr` 额外具有一定的位宽。 ODS属性被定义为具有存储类型（对应于存储属性的`mlir::Attribute`类），返回类型（对应于生成的`getters`帮助函数的C++返回类型）以及在内部存储类型和帮助函数进行互转的方法。

**属性装饰器**。 有一些重要的属性适配器/装饰器/修饰符可以应用于 ODS 属性以指定常见的附加属性，如可选性、默认值等。

- `DefaultValuedAttr`：为一个属性指定默认值。
- `OptionalAttr`：将一个属性指定为可选的。
- `Confined`：`Confined`作为一种通用机制被提供，以帮助对值类型带来的属性约束进行进一步建模。可以通过`Confined`将较为原始的约束组合成为复杂约束。举个例子，一个`32bit`的整型最小值为10，可以被表示为`Confined<I32Attr, [IntMinValue<10>]>`。还有一些其它例子，比如`IntMinValue<N>`：指定一个大于等于N的整型属性等等。

**枚举属性** 。某些属性只能从预定义的enum获取值，例如，比较op的比较类型。 为了定义这些属性，ODS 提供了几种机制：`StrEnumAttr`、`IntEnumAttr` 和 `BitEnumAttr`。

- `StrEnumAttr`：每个enum case 都是一个字符串，属性在op中存储为 `StringAttr`。 
- `IntEnumAttr`：每个enum case 都是一个整数，属性在op中存储为 `IntegerType`。 
- `BitEnumAttr`：每个 enum case 都是一个位，属性在 op 中存储为 `IntegerAttr`。

所有这些 `*EnumAttr` 属性都需要通过其对应的 `*EnumAttrCase` 完全指定所有允许的情况。 有了这个，ODS 能够生成额外的验证以只接受允许的案例。 为了促进 `*EnumAttrs` 和它们的 C++ 使用者之间的交互，EnumsGen(`https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/EnumsGen.cpp`) TableGen 后端可以生成一些常见的实用程序：C++ 枚举类、用于枚举类的 `llvm::DenseMapInfo`、从/到字符串的转换函数。 这是通过 `mlir-tblgen` 的 `-gen-enum-decls` 和 `-gen-enum-defs` 命令行选项控制的。 

例如，给定下面的`EnumAttr`：

 

```cpp
def Case15: I32EnumAttrCase<"Case15", 15>;
def Case20: I32EnumAttrCase<"Case20", 20>;

def MyIntEnum: I32EnumAttr<"MyIntEnum", "An example int enum",
                           [Case15, Case20]> {
  let cppNamespace = "Outer::Inner";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}
```
以下代码将通过 `mlir-tblgen -gen-enum-decls` 生成： 

```cpp
namespace Outer {
namespace Inner {
// An example int enum
enum class MyIntEnum : uint32_t {
  Case15 = 15,
  Case20 = 20,
};

llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t);
llvm::StringRef ConvertToString(MyIntEnum);
llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef);
inline constexpr unsigned getMaxEnumValForMyIntEnum() {
  return 20;
}

} // namespace Inner
} // namespace Outer

namespace llvm {
template<> struct DenseMapInfo<Outer::Inner::MyIntEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline Outer::Inner::MyIntEnum getEmptyKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getEmptyKey());
  }

  static inline Outer::Inner::MyIntEnum getTombstoneKey() {
    return static_cast<Outer::Inner::MyIntEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const Outer::Inner::MyIntEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const Outer::Inner::MyIntEnum &lhs, const Outer::Inner::MyIntEnum &rhs) {
    return lhs == rhs;
  }
};
}
```

以下代码将通过 `mlir-tblgen -gen-enum-defs` 生成： 


```cpp
namespace Outer {
namespace Inner {
llvm::StringRef ConvertToString(MyIntEnum val) {
  switch (val) {
    case MyIntEnum::Case15: return "Case15";
    case MyIntEnum::Case20: return "Case20";
  }
  return "";
}

llvm::Optional<MyIntEnum> ConvertToEnum(llvm::StringRef str) {
  return llvm::StringSwitch<llvm::Optional<MyIntEnum>>(str)
      .Case("Case15", MyIntEnum::Case15)
      .Case("Case20", MyIntEnum::Case20)
      .Default(llvm::None);
}
llvm::Optional<MyIntEnum> symbolizeMyIntEnum(uint32_t value) {
  switch (value) {
  case 15: return MyIntEnum::Case15;
  case 20: return MyIntEnum::Case20;
  default: return llvm::None;
  }
}

} // namespace Inner
} // namespace Outer
```

对于以下 `BitEnumAttr` 定义类似： 

```cpp
def None: BitEnumAttrCase<"None", 0x0000>;
def Bit1: BitEnumAttrCase<"Bit1", 0x0001>;
def Bit2: BitEnumAttrCase<"Bit2", 0x0002>;
def Bit3: BitEnumAttrCase<"Bit3", 0x0004>;

def MyBitEnum: BitEnumAttr<"MyBitEnum", "An example bit enum",
                           [None, Bit1, Bit2, Bit3]>;
```

我们得到：

```cpp
// An example bit enum
enum class MyBitEnum : uint32_t {
  None = 0,
  Bit1 = 1,
  Bit2 = 2,
  Bit3 = 4,
};

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t);
std::string stringifyMyBitEnum(MyBitEnum);
llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef);
inline MyBitEnum operator|(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) | static_cast<uint32_t>(rhs));
}
inline MyBitEnum operator&(MyBitEnum lhs, MyBitEnum rhs) {
  return static_cast<MyBitEnum>(static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs));
}
inline bool bitEnumContains(MyBitEnum bits, MyBitEnum bit) {
  return (static_cast<uint32_t>(bits) & static_cast<uint32_t>(bit)) != 0;
}

namespace llvm {
template<> struct DenseMapInfo<::MyBitEnum> {
  using StorageInfo = llvm::DenseMapInfo<uint32_t>;

  static inline ::MyBitEnum getEmptyKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getEmptyKey());
  }

  static inline ::MyBitEnum getTombstoneKey() {
    return static_cast<::MyBitEnum>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::MyBitEnum &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::MyBitEnum &lhs, const ::MyBitEnum &rhs) {
    return lhs == rhs;
  }
};
```

```cpp
std::string stringifyMyBitEnum(MyBitEnum symbol) {
  auto val = static_cast<uint32_t>(symbol);
  // Special case for all bits unset.
  if (val == 0) return "None";

  llvm::SmallVector<llvm::StringRef, 2> strs;
  if (1u & val) { strs.push_back("Bit1"); val &= ~1u; }
  if (2u & val) { strs.push_back("Bit2"); val &= ~2u; }
  if (4u & val) { strs.push_back("Bit3"); val &= ~4u; }

  if (val) return "";
  return llvm::join(strs, "|");
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(llvm::StringRef str) {
  // Special case for all bits unset.
  if (str == "None") return MyBitEnum::None;

  llvm::SmallVector<llvm::StringRef, 2> symbols;
  str.split(symbols, "|");

  uint32_t val = 0;
  for (auto symbol : symbols) {
    auto bit = llvm::StringSwitch<llvm::Optional<uint32_t>>(symbol)
      .Case("Bit1", 1)
      .Case("Bit2", 2)
      .Case("Bit3", 4)
      .Default(llvm::None);
    if (bit) { val |= *bit; } else { return llvm::None; }
  }
  return static_cast<MyBitEnum>(val);
}

llvm::Optional<MyBitEnum> symbolizeMyBitEnum(uint32_t value) {
  // Special case for all bits unset.
  if (value == 0) return MyBitEnum::None;

  if (value & ~(1u | 2u | 4u)) return llvm::None;
  return static_cast<MyBitEnum>(value);
}
```

在OneFlow-MLIR中同样也有枚举属性的定义用来处理OneFlow的各种数据类型，代码如下：

```cpp
#ifndef ONEFLOW_ENUMS
#define ONEFLOW_ENUMS

def OneFlow_InvalidDataType : I32EnumAttrCase<"DT_InvalidDataType", 0>;
def OneFlow_Char : I32EnumAttrCase<"DT_Char", 1>;
def OneFlow_Float : I32EnumAttrCase<"DT_Float", 2>;
def OneFlow_Double : I32EnumAttrCase<"DT_Double", 3>;
def OneFlow_Int8 : I32EnumAttrCase<"DT_Int8", 4>;
def OneFlow_Int32 : I32EnumAttrCase<"DT_Int32", 5>;
def OneFlow_Int64 : I32EnumAttrCase<"DT_Int64", 6>;
def OneFlow_UInt8 : I32EnumAttrCase<"DT_UInt8", 7>;
def OneFlow_OFRecord : I32EnumAttrCase<"DT_OFRecord", 8>;
def OneFlow_Float16 : I32EnumAttrCase<"DT_Float16", 9>;
def OneFlow_TensorBuffer: I32EnumAttrCase<"DT_TensorBuffer", 10>;

def OneFlow_DataType: I32EnumAttr<"DataType", "OneFlow Data Type enum",
  [
    OneFlow_InvalidDataType,
    OneFlow_Char,
    OneFlow_Float,
    OneFlow_Double,
    OneFlow_Int8,
    OneFlow_Int32,
    OneFlow_Int64,
    OneFlow_UInt8,
    OneFlow_OFRecord,
    OneFlow_Float16,
    OneFlow_TensorBuffer,
  ]
> {
  let cppNamespace = "::mlir::oneflow";
  let stringToSymbolFnName = "ConvertToEnum";
  let symbolToStringFnName = "ConvertToString";
}

#endif // ONEFLOW_ENUMS
```

我们可以观察一下它生成的enum属性声明：


```cpp
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Enum Utility Declarations                                                  *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace oneflow {
// OneFlow Data Type enum
enum class DataType : uint32_t {
  DT_InvalidDataType = 0,
  DT_Char = 1,
  DT_Float = 2,
  DT_Double = 3,
  DT_Int8 = 4,
  DT_Int32 = 5,
  DT_Int64 = 6,
  DT_UInt8 = 7,
  DT_OFRecord = 8,
  DT_Float16 = 9,
  DT_TensorBuffer = 10,
};

::llvm::Optional<DataType> symbolizeDataType(uint32_t);
::llvm::StringRef ConvertToString(DataType);
::llvm::Optional<DataType> ConvertToEnum(::llvm::StringRef);
inline constexpr unsigned getMaxEnumValForDataType() {
  return 10;
}


inline ::llvm::StringRef stringifyEnum(DataType enumValue) {
  return ConvertToString(enumValue);
}

template <typename EnumType>
::llvm::Optional<EnumType> symbolizeEnum(::llvm::StringRef);

template <>
inline ::llvm::Optional<DataType> symbolizeEnum<DataType>(::llvm::StringRef str) {
  return ConvertToEnum(str);
}

class DataTypeAttr : public ::mlir::IntegerAttr {
public:
  using ValueType = DataType;
  using ::mlir::IntegerAttr::IntegerAttr;
  static bool classof(::mlir::Attribute attr);
  static DataTypeAttr get(::mlir::MLIRContext *context, DataType val);
  DataType getValue() const;
};
} // namespace oneflow
} // namespace mlir

namespace llvm {
template<> struct DenseMapInfo<::mlir::oneflow::DataType> {
  using StorageInfo = ::llvm::DenseMapInfo<uint32_t>;

  static inline ::mlir::oneflow::DataType getEmptyKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getEmptyKey());
  }

  static inline ::mlir::oneflow::DataType getTombstoneKey() {
    return static_cast<::mlir::oneflow::DataType>(StorageInfo::getTombstoneKey());
  }

  static unsigned getHashValue(const ::mlir::oneflow::DataType &val) {
    return StorageInfo::getHashValue(static_cast<uint32_t>(val));
  }

  static bool isEqual(const ::mlir::oneflow::DataType &lhs, const ::mlir::oneflow::DataType &rhs) {
    return lhs == rhs;
  }
};
}
```

实现部分就不贴了，这里贴了过长的代码了。

# 13. 类型定义（我只是简单了解了一下）
MLIR 定义了 `TypeDef` 类层次结构，以支持根据其规范生成数据类型。 类型是通过特化 `TypeDef` 类来定义的，该类具有它所需的所有字段的具体内容。 例如，整数类型可以定义为： 


```cpp
// All of the types will extend this class.
class Test_Type<string name> : TypeDef<Test_Dialect, name> { }

// An alternate int type.
def IntegerType : Test_Type<"TestInteger"> {
  let mnemonic = "int";

  let summary = "An integer type with special semantics";

  let description = [{
    An alternate integer type. This type differentiates itself from the
    standard integer type by not having a SignednessSemantics parameter, just
    a width.
  }];

  let parameters = (ins "unsigned":$width);

  // We define the printer inline.
  let printer = [{
    $_printer << "int<" << getImpl()->width << ">";
  }];

  // The parser is defined here also.
  let parser = [{
    if ($_parser.parseLess())
      return Type();
    int width;
    if ($_parser.parseInteger(width))
      return Type();
    if ($_parser.parseGreater())
      return Type();
    return get($_ctxt, width);
  }];
}
```

- **Type name** : 生成的 C++ 类的名称默认为 `<classParamName>Type`（例如上例中的 `TestIntegerType`）。 这可以通过 `cppClassName` 字段覆盖。 `mnemonic` 是指定解析的asm名称。 它是可选的，不指定将意味着没有解析器或打印方法附加到此类。 
- **Type documentation**：存在`summary`和`description`字段，其使用方式与Operation中相同。 即，`summary`应该是单行的，而`description`应该是更长的解释。 
- **Type parameters**：`parameters`字段是类型参数的列表。 如果未指定任何参数（默认），则此类型被视为单例类型。 参数采用`“c++Type”:$paramName` 格式。 要将C++类型用作需要在存储构造函数中分配的参数，有两种选择： 1. 设置 `hasCustomStorageConstructor` 以生成带有刚刚声明的构造函数的 TypeStorage 类——没有定义——所以我们可以自己编写它。 2. 使用`TypeParameter` tablegen类而不是"c++Type"字符串。（后半句话我不是很懂，也还没用过。）

- **TypeParameter tablegen class** : 这用于进一步指定有关每个类型参数的属性。 它包括文档（`summary`和`syntax`）、要使用的 C++ 类型、要在存储构造函数方法中使用的自定义分配器，以及用于确定参数类型的两个实例是否相等的自定义比较器。 

```cpp
// DO NOT DO THIS!
let parameters = (ins "ArrayRef<int>":$dims);
```
默认存储构造函数盲目地按值复制字段。 它对类型一无所知。 在这种情况下，ArrayRef 需要使用 `dims = allocator.copyInto(dims)` 进行分配。 

```cpp
class ArrayRefIntParam :
    TypeParameter<"::llvm::ArrayRef<int>", "Array of ints"> {
  let allocator = "$_dst = $_allocator.copyInto($_self);";
}

...

let parameters = (ins ArrayRefIntParam:$dims);
```
`allocator`代码块由`$_allocator`（是在其中分配对象的 TypeStorageAllocator）和`$_dst`（是放置已分配数据的变量）组成。`comparator`代码块由`$_lhs`和`$_rhs`参数类型实例组成。

自定义Type还有不少内容，但目前我没有这方面的需求，所以就没有继续看了，这里只是简单了解了一下。感兴趣的读者可以自行查看文档进行深入研究：https://mlir.llvm.org/docs/OpDefinitions/ 。

# 14. DEBUG方法
使用`mlir-tblgen`来看产生的文本。TableGen 语法有时可能很晦涩。阅读生成的文本对于理解和调试问题非常有用。 要构建 `mlir-tblgen`，可以运行 `cmake --build 。 --target mlir-tblgen` 在我们的构建目录中，并在 `bin/` 子目录中找到 `mlir-tblgen` 二进制文件。 所有支持的生成器都可以通过 `mlir-tblgen --help` 找到。 

要查看生成的代码，请通过 `-I` 提供包含路径，使用 `mlir-tblgen` 调用特定生成器。 例如：

```cpp
# To see op C++ class declaration
mlir-tblgen --gen-op-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op C++ class definition
mlir-tblgen --gen-op-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op documentation
mlir-tblgen --gen-dialect-doc -I /path/to/mlir/include /path/to/input/td/file

# To see op interface C++ class declaration
mlir-tblgen --gen-op-interface-decls -I /path/to/mlir/include /path/to/input/td/file
# To see op interface C++ class definition
mlir-tblgen --gen-op-interface-defs -I /path/to/mlir/include /path/to/input/td/file
# To see op interface documentation
mlir-tblgen --gen-op-interface-doc -I /path/to/mlir/include /path/to/input/td/file
```



# 15. 总结
这一节在[【从零开始学深度学习编译器】十六，MLIR ODS要点总结上篇](https://mp.weixin.qq.com/s/SFHWUm63BqsD9SWwuW83mA) 的基础上补充完整了ODS的要点。约束和属性的定义都是MLIR中相当重要的元素，至于类型的定义个人认为了解即可，等到我们需要自定义类型的时候再仔细研究。最后MLIR的语法比较晦涩，初学者可以借助`mlir-tblgen`来辅助debug。

在这两篇文章里，我跟着MLIR的ODS规范完整走了一遍并总结了14个要点，对于每一个要点我都在OneFlow MLIR的Op定义中进行了对照，并给出了一些示例代码和位置。希望对读者入门MLIR有帮助。

 