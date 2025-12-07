# TinyAI Neural Network V2 增强报告

## 概述

本次更新参考 PyTorch 设计，对 `tinyai-deeplearning-nnet` 的 V2 模块进行了全面完善，新增了大量激活函数、归一化层、Transformer 组件以及增强了核心 Module 类的功能。

## 更新内容汇总

### 一、激活函数扩展 ✅

| 激活函数 | 文件 | 公式 | 应用场景 |
|---------|------|------|---------|
| **GELU** | `layer/activation/GELU.java` | x * Φ(x) ≈ 0.5x(1+tanh(√(2/π)(x+0.044715x³))) | GPT, BERT, ViT |
| **SiLU** | `layer/activation/SiLU.java` | x * sigmoid(x) | EfficientNet, YOLOv5 |
| **LeakyReLU** | `layer/activation/LeakyReLU.java` | max(αx, x), α=0.01 | 解决神经元死亡 |
| **ELU** | `layer/activation/ELU.java` | x if x≥0, α(eˣ-1) otherwise | 负值饱和，加速学习 |
| **LogSoftmax** | `layer/activation/LogSoftmax.java` | log(softmax(x)) | 配合NLLLoss使用 |

**底层实现（func模块）**：
- `func/math/SiLU.java` - 底层 SiLU Function
- `func/math/LeakyReLU.java` - 底层 LeakyReLU Function
- `func/math/ELU.java` - 底层 ELU Function
- `func/math/LogSoftmax.java` - 底层 LogSoftmax Function

**Variable 扩展方法**：
- `gelu()` - GELU激活
- `silu()` - SiLU激活
- `leakyRelu(float negativeSlope)` - LeakyReLU激活
- `elu(float alpha)` - ELU激活
- `logSoftmax(int axis)` - LogSoftmax激活

---

### 二、归一化层增强 ✅

| 归一化层 | 文件 | 公式 | 应用场景 |
|---------|------|------|---------|
| **RMSNorm** | `layer/norm/RMSNorm.java` | y = x/RMS(x) * weight | LLaMA, DeepSeek等LLM |

**RMSNorm 特点**：
- 比 LayerNorm 更高效（去掉均值中心化）
- 只有 weight 参数，没有 bias
- 默认 eps = 1e-6

---

### 三、Transformer 组件完善 ✅

#### 1. MultiHeadAttention 增强

**新增功能**：
- ✅ 支持 `attnMask`（注意力掩码，如因果掩码）
- ✅ 支持 `keyPaddingMask`（键填充掩码）
- ✅ 支持不同长度的 query/key/value 序列

**新增静态方法**：
```java
// 生成因果掩码
Variable causalMask = MultiHeadAttention.generateCausalMask(seqLen);

// 生成可广播的因果掩码
Variable causalMaskBatched = MultiHeadAttention.generateCausalMaskBatched(seqLen);

// 生成填充掩码
Variable paddingMask = MultiHeadAttention.generatePaddingMask(batchSize, maxLen, actualLengths);

// 组合因果掩码和填充掩码
Variable combinedMask = MultiHeadAttention.combineCausalAndPaddingMask(seqLen, paddingMask);
```

#### 2. TransformerEncoder 容器

**文件**: `layer/transformer/TransformerEncoder.java`

**功能**：
- 堆叠多个 `TransformerEncoderLayer`
- 支持可选的最终层归一化（Pre-LN架构）
- 支持源序列掩码

**使用示例**：
```java
TransformerEncoder encoder = new TransformerEncoder(
    "encoder",
    numLayers,    // 层数
    dModel,       // 模型维度
    numHeads,     // 注意力头数
    dFF,          // FFN隐藏层维度
    dropout,      // dropout比率
    preLayerNorm  // 是否Pre-LN
);

Variable output = encoder.forward(src, srcMask);
```

#### 3. TransformerDecoder 容器

**文件**: `layer/transformer/TransformerDecoder.java`

**功能**：
- 堆叠多个 `TransformerDecoderLayer`
- 支持可选的最终层归一化
- 支持目标序列掩码（因果掩码）
- 支持编码器输出掩码

**使用示例**：
```java
TransformerDecoder decoder = new TransformerDecoder(
    "decoder",
    numLayers,
    dModel,
    numHeads
);

Variable output = decoder.forward(tgt, memory, tgtMask);
```

#### 4. 完整 Transformer 模型

**文件**: `layer/transformer/Transformer.java`

**功能**：
- 组合 Encoder + Decoder
- 支持分离的 encode/decode 方法（用于推理）
- 提供生成因果掩码的便捷方法

**使用示例**：
```java
// 创建完整Transformer
Transformer transformer = new Transformer(
    "transformer",
    dModel,           // 512
    numHeads,         // 8
    numEncoderLayers, // 6
    numDecoderLayers  // 6
);

// 训练时：联合编解码
Variable output = transformer.forward(src, tgt, tgtMask);

// 推理时：分离式
Variable memory = transformer.encode(src);
Variable output = transformer.decode(tgt, memory, causalMask);
```

---

### 四、Module 类增强 ✅

**新增方法**：

| 方法 | 功能 | 返回类型 |
|------|------|---------|
| `freeze()` | 冻结所有参数 | Module |
| `unfreeze()` | 解冻所有参数 | Module |
| `requiresGrad(boolean)` | 设置是否需要梯度 | Module |
| `numParameters(boolean)` | 统计参数数量 | long |
| `numParameters()` | 统计所有参数数量 | long |
| `parameterSummary()` | 获取参数摘要 | String |
| `copyStateDict()` | 深拷贝状态字典 | Map |
| `extraRepr()` | 额外信息表示 | String |
| `numChildren()` | 子模块数量 | int |
| `children()` | 所有直接子模块 | Collection |
| `modules()` | 所有模块（含自身） | Iterable |
| `evalAndFreeze()` | 评估模式+冻结 | Module |
| `trainAndUnfreeze()` | 训练模式+解冻 | Module |

**使用示例**：
```java
// 统计参数
System.out.println("Total params: " + model.numParameters());
System.out.println("Trainable params: " + model.numParameters(true));

// 参数摘要
System.out.println(model.parameterSummary());

// 冻结/解冻
model.freeze();  // 冻结所有参数
model.getModule("encoder").freeze();  // 只冻结编码器
model.unfreeze();  // 解冻所有参数

// 推理模式
model.evalAndFreeze();
```

---

### 五、Functional API 扩展 ✅

**新增激活函数**：
```java
Functional.gelu(input)
Functional.silu(input)
Functional.leakyRelu(input, negativeSlope)
Functional.elu(input, alpha)
Functional.logSoftmax(input, axis)
```

**新增归一化**：
```java
Functional.rmsNorm(input, weight, eps)
```

**新增注意力**：
```java
Functional.scaledDotProductAttention(query, key, value, attnMask, dropout, training)
```

**新增损失函数**：
```java
Functional.crossEntropyLoss(input, target)
Functional.nllLoss(input, target)
Functional.mseLoss(input, target)
Functional.binaryCrossEntropyLoss(input, target)
Functional.binaryCrossEntropyWithLogitsLoss(input, target)
```

---

## 新增文件列表

### 底层 func 模块
```
tinyai-deeplearning-func/src/main/java/io/leavesfly/tinyai/func/math/
├── SiLU.java          🆕
├── LeakyReLU.java     🆕
├── ELU.java           🆕
└── LogSoftmax.java    🆕
```

### V2 层实现
```
tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/v2/layer/
├── activation/
│   ├── GELU.java           🆕
│   ├── SiLU.java           🆕
│   ├── LeakyReLU.java      🆕
│   ├── ELU.java            🆕
│   └── LogSoftmax.java     🆕
├── norm/
│   └── RMSNorm.java        🆕
└── transformer/
    ├── MultiHeadAttention.java   ✏️ 增强
    ├── TransformerEncoder.java   🆕
    ├── TransformerDecoder.java   🆕
    └── Transformer.java          🆕
```

### 修改的现有文件
```
tinyai-deeplearning-func/src/main/java/io/leavesfly/tinyai/func/Variable.java  ✏️
tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/v2/core/Module.java  ✏️
tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/v2/functional/Functional.java  ✏️
```

---

## 与 PyTorch 对标

| PyTorch | TinyAI V2 | 状态 |
|---------|-----------|------|
| `nn.GELU` | `layer.activation.GELU` | ✅ |
| `nn.SiLU` | `layer.activation.SiLU` | ✅ |
| `nn.LeakyReLU` | `layer.activation.LeakyReLU` | ✅ |
| `nn.ELU` | `layer.activation.ELU` | ✅ |
| `nn.LogSoftmax` | `layer.activation.LogSoftmax` | ✅ |
| `nn.RMSNorm` | `layer.norm.RMSNorm` | ✅ |
| `nn.MultiheadAttention` | `layer.transformer.MultiHeadAttention` | ✅ 增强 |
| `nn.TransformerEncoder` | `layer.transformer.TransformerEncoder` | ✅ |
| `nn.TransformerDecoder` | `layer.transformer.TransformerDecoder` | ✅ |
| `nn.Transformer` | `layer.transformer.Transformer` | ✅ |
| `Module.freeze()` | `Module.freeze()` | ✅ |
| `Module.parameters()` | `Module.namedParameters()` | ✅ |
| `F.gelu` | `Functional.gelu` | ✅ |
| `F.scaled_dot_product_attention` | `Functional.scaledDotProductAttention` | ✅ |
| `F.cross_entropy` | `Functional.crossEntropyLoss` | ✅ |

---

## 后续计划

### 短期
- [ ] 完善单元测试
- [ ] 添加更多激活函数 (Mish, Hardswish, PReLU)
- [ ] 添加 GroupNorm, BatchNorm2d

### 中期
- [ ] RNN 层增强（多层、双向）
- [ ] 添加 LSTMCell, GRUCell
- [ ] Conv1d, ConvTranspose2d

### 长期
- [ ] 位置编码扩展 (RoPE, ALiBi)
- [ ] Flash Attention 优化
- [ ] 混合精度训练支持

---

## 使用建议

1. **现代 LLM 开发**：使用 `GELU` + `RMSNorm` + `Transformer`
2. **图像分类**：使用 `SiLU` + 现有卷积层
3. **迁移学习**：使用 `freeze()` / `unfreeze()` 管理参数
4. **模型分析**：使用 `parameterSummary()` 查看模型结构

---

**更新时间**: 2024年
**版本**: V2.1

