# DeepSeek-R1 模型实现

> 基于TinyAI框架实现的DeepSeek-R1推理和反思模型

## 📋 概述

DeepSeek-R1是一个具备深度推理和自我反思能力的大语言模型，通过多步推理和反思机制实现复杂任务的可解释性处理。

### 核心特性

- 🧠 **多步推理** - 支持最多7步迭代推理过程
- 🔍 **自我反思** - 从5个维度评估推理质量（逻辑性、完整性、正确性、清晰度、有用性）
- 📊 **置信度评估** - 动态评估每步推理的可信度
- 🏗️ **Pre-LayerNorm架构** - 提升训练稳定性
- ✨ **基于V2 API** - 完全使用TinyAI nnet v2组件，不依赖V1接口

## 🏛️ 架构设计

### 模型层次结构

```
DeepSeekR1Model (Model)
    └── DeepSeekR1Block (Module)
        ├── DeepSeekR1TokenEmbedding (Module)
        ├── DeepSeekR1TransformerBlock[] (Module)
        │   ├── MultiHeadAttention (Layer)
        │   ├── LayerNorm (Layer)
        │   ├── Linear (Layer)
        │   └── GELU (Layer)
        ├── DeepSeekR1ReasoningBlock (Module)
        │   ├── Linear (Layer)
        │   ├── LayerNorm (Layer)
        │   └── Sigmoid (Layer)
        ├── DeepSeekR1ReflectionBlock (Module)
        │   ├── Linear (Layer)
        │   ├── LayerNorm (Layer)
        │   └── Sigmoid (Layer)
        ├── LayerNorm (Layer)
        └── Linear (Layer)
```

### 数据流

```
Token IDs 
    ↓ (Token + Position Embedding)
Token Embeddings 
    ↓ (Transformer Layers × N)
Hidden States
    ↓ (Reasoning Module)
Reasoning Output (+ Confidence Scores)
    ↓ (Reflection Module)
Reflection Output (+ Quality Scores)
    ↓ (LayerNorm + Output Projection)
Logits
```

## 📦 核心组件

### 1. DeepSeekR1Config
配置类，包含所有模型超参数：
- 基础配置：vocabSize, nEmbd, nLayer, nHead, etc.
- 推理配置：maxReasoningSteps, reasoningHiddenDim, confidenceThreshold
- 反思配置：reflectionHiddenDim, qualityScoreDim, maxSuggestions

### 2. DeepSeekR1TokenEmbedding
Token嵌入层，负责：
- Token嵌入（词汇表 → 向量）
- 位置嵌入（位置 → 向量）
- Dropout正则化

### 3. DeepSeekR1TransformerBlock
Transformer块（Pre-LayerNorm），包含：
- 多头自注意力子层（带因果掩码）
- 前馈神经网络子层
- 残差连接和LayerNorm

### 4. DeepSeekR1ReasoningBlock
推理模块，实现：
- 多步迭代推理（最多7步）
- 置信度动态评估
- 推理状态管理

### 5. DeepSeekR1ReflectionBlock
反思模块，实现：
- 质量多维评分（5个维度）
- 问题识别
- 改进建议生成

### 6. DeepSeekR1Block
主体块，整合所有组件

### 7. DeepSeekR1Model
模型类，提供统一接口

## 🚀 快速开始

### 基本使用

```java
// 创建模型（三种预设规模）
DeepSeekR1Model model = DeepSeekR1Model.createTinyModel("R1-Tiny");      // 微型（测试）
// DeepSeekR1Model model = DeepSeekR1Model.createSmallModel("R1-Small");  // 小型（实验）
// DeepSeekR1Model model = DeepSeekR1Model.createStandardModel("R1");     // 标准（完整）

// 准备输入 [batch_size, seq_len]
NdArray tokenIds = NdArray.of(new float[][]{{1, 2, 3, 4, 5}});

// 基础推理
Variable logits = model.predict(new Variable(tokenIds));
System.out.println("输出形状: " + logits.getValue().getShape());
```

### 带详细信息的推理

```java
// 执行推理并获取详细结果
DeepSeekR1Model.ReasoningOutput result = model.performReasoning(new Variable(tokenIds));

// 查看推理详情
System.out.println("推理步骤数: " + result.numSteps);
System.out.println("平均置信度: " + result.averageConfidence);
System.out.println("质量评分: " + result.qualityScore);
```

### 序列生成

```java
// 准备提示词
NdArray promptIds = NdArray.of(new float[][]{{1, 2, 3}});

// 生成新token
NdArray generated = model.generateSequence(promptIds, 10);  // 生成10个新token
System.out.println("生成序列形状: " + generated.getShape());
```

### 自定义配置

```java
// 创建自定义配置
DeepSeekR1Config config = new DeepSeekR1Config();
config.setVocabSize(10000);
config.setNEmbd(256);
config.setNLayer(6);
config.setNHead(8);
config.setMaxReasoningSteps(5);
config.setConfidenceThreshold(0.8);

// 验证配置
config.validate();

// 创建模型
DeepSeekR1Model model = new DeepSeekR1Model("R1-Custom", config);
```

## 📊 模型规模对比

| 模型 | 参数量 | 层数 | 维度 | 注意力头 | 推理步骤 | 适用场景 |
|------|--------|------|------|----------|----------|----------|
| Tiny | ~100M | 6 | 256 | 8 | 5 | 快速测试 |
| Small | ~500M | 8 | 512 | 8 | 6 | 学习实验 |
| Standard | ~1B | 12 | 768 | 12 | 7 | 研究应用 |

## 🎯 质量评分维度

反思模块从5个维度评估推理质量：

1. **逻辑性** (Logic Score) - 推理步骤的逻辑连贯性
2. **完整性** (Completeness Score) - 是否考虑了所有相关因素
3. **正确性** (Correctness Score) - 结论的准确性
4. **清晰度** (Clarity Score) - 表达的清晰程度
5. **有用性** (Usefulness Score) - 对解决问题的帮助程度

每个维度的分数范围：[0, 1]，总体评分为5个维度的平均值。

## 📚 示例代码

查看 [DeepSeekR1Demo.java](./DeepSeekR1Demo.java) 获取完整示例，包括：
- 示例1: 创建模型并打印信息
- 示例2: 基础推理
- 示例3: 带详细信息的推理
- 示例4: 序列生成
- 示例5: 自定义配置模型
- 示例6: 对比不同规模的模型

运行示例：
```bash
java io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Demo
```

## 🔧 技术细节

### 架构特点

1. **Pre-LayerNorm架构**
   - 在子层之前应用LayerNorm
   - 提升训练稳定性
   - 流程：LN → SubLayer → Dropout → Add(Residual)

2. **因果掩码**
   - 自注意力使用下三角掩码
   - 确保自回归特性
   - 防止信息泄露

3. **推理机制**
   - 最多7步迭代推理
   - 每步评估置信度
   - 低于阈值继续推理

4. **反思机制**
   - 多维质量评分
   - 改进建议生成
   - 自适应阈值控制

### 依赖关系

本实现完全基于TinyAI框架的V2 API：
- `tinyai-deeplearning-nnet` v2.core.Module
- `tinyai-deeplearning-nnet` v2.layer.*
- `tinyai-deeplearning-ml` Model
- `tinyai-deeplearning-func` Variable
- `tinyai-deeplearning-ndarr` NdArray

**严格遵守**：
- ✅ 使用 v2.core.Module（而非 v1 Block/Layer）
- ✅ 使用 v2.layer.* 组件
- ❌ 禁止使用 v1 接口
- ❌ 不依赖 v3 目录代码

## 📝 类继承规范

根据TinyAI框架规范，本实现严格遵循以下继承规则：

- **DeepSeekR1Model** → extends `Model`
- **DeepSeekR1Block** → extends `Module` (v2)
- **DeepSeekR1TokenEmbedding** → extends `Module` (v2)
- **DeepSeekR1TransformerBlock** → extends `Module` (v2)
- **DeepSeekR1ReasoningBlock** → extends `Module` (v2)
- **DeepSeekR1ReflectionBlock** → extends `Module` (v2)

## 🎓 学习资源

相关文档：
- [DeepSeek概述](../../../../book/part2-llm/chapter14_2-deepseek/14.2.1-deepseek-overview.md)
- [R1推理与反思机制](../../../../book/part2-llm/chapter14_2-deepseek/14.2.2-r1-reasoning-reflection.md)
- [TinyAI Neural Network V2](../../../tinyai-deeplearning/tinyai-deeplearning-nnet/doc/v2/README.md)

## 📄 许可证

本项目遵循TinyAI框架的许可证。

## 👥 贡献者

- leavesfly - 初始实现

---

**注意**: 本实现为教育和研究目的，展示DeepSeek-R1的核心架构思想。实际生产环境需要更多优化和完善。
