# GPT-1 模型完整实现

## 概述

GPT-1（Generative Pre-trained Transformer 1）是OpenAI于2018年发布的第一代生成式预训练Transformer模型，开创了"预训练+微调"的范式。本实现完全基于TinyAI框架V2 API,提供从预训练到推理的完整流程。

## 核心特性

### 架构特点
- **Post-LayerNorm架构**: 在子层之后应用层归一化（遵循原始Transformer设计）
- **标准Transformer解码器**: 因果掩码的自注意力机制
- **序列长度**: 512（相比GPT-2/3的1024/2048较短）
- **参数规模**: 117M（标准配置）

### 与GPT-2/GPT-3的差异

| 特性 | GPT-1 | GPT-2 | GPT-3 |
|------|-------|-------|-------|
| LayerNorm位置 | Post-LN | Pre-LN | Pre-LN |
| 计算方式 | 串行 | 串行 | 并行 |
| 序列长度 | 512 | 1024 | 2048 |
| 默认参数 | 117M | 117M-1.5B | 125M-175B |

## 文件结构

```
gpt1/
├── GPT1Config.java           # 配置类
├── GPT1TokenEmbedding.java   # Token和位置嵌入
├── GPT1TransformerBlock.java # Transformer块(Post-LN)
├── GPT1MainBlock.java        # 主体块
├── GPT1Model.java            # 模型类
├── GPT1Demo.java             # 演示程序
├── training/                 # 训练和推理模块
│   ├── GPT1Dataset.java      # 数据集处理
│   ├── GPT1Pretrain.java     # 预训练器
│   ├── GPT1Finetune.java     # 微调训练器
│   ├── GPT1Inference.java    # 推理引擎
│   └── GPT1TrainDemo.java    # 训练演示
└── README.md                 # 本文档
```

## 技术实现

### 1. 完全基于V2 API
- 所有组件继承自`io.leavesfly.tinyai.nnet.v2.core.Module`
- 使用V2的`Linear`, `LayerNorm`, `Dropout`, `GELU`, `MultiHeadAttention`
- 保持与TinyAI框架的架构一致性

### 2. 完全独立实现
- 不依赖GPT-2/GPT-3的任何代码
- 独立的配置和组件
- 遵循模块独立性原则

### 3. Post-LayerNorm架构
```
input -> Attention -> Dropout -> Add -> LayerNorm ->
      -> FeedForward -> Dropout -> Add -> LayerNorm -> output
```

## 训练流程

### 预训练 (Pretrain)

**目标**: 学习语言的通用表示和模式

**配置**:
- 任务: 因果语言建模（预测下一个token）
- 数据: 大规模无标注文本（原论文使用BooksCorpus约7000本书）
- 学习率: 2.5e-4 (warmup + cosine decay)
- 优化器: Adam (β1=0.9, β2=0.999)
- 梯度裁剪: 1.0
- Batch大小: 64
- 序列长度: 512

**代码示例**:
```java
// 1. 创建模型
GPT1Model model = GPT1Model.createStandardModel("gpt1-pretrain");

// 2. 准备数据
GPT1Dataset.SimpleTokenizer tokenizer = new GPT1Dataset.SimpleTokenizer();
GPT1Dataset dataset = new GPT1Dataset(512, 64, tokenizer.getVocabSize());
dataset.loadFromFile("data/pretrain.txt", tokenizer);

// 3. 配置训练器
GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
trainer.configure(
    10,        // maxEpochs
    2.5e-4f,   // learningRate
    2000,      // warmupSteps
    1.0f       // maxGradNorm
).setCheckpoint("./checkpoints/pretrain", 5000);

// 4. 开始训练
trainer.train();
```

### 微调 (Finetune/Posttrain)

**目标**: 适应特定下游任务

**配置**:
- 任务: 文本分类、问答、文本蕴含等
- 数据: 任务相关的标注数据
- 学习率: 2.5e-5 (比预训练小10倍)
- 早停: 验证集损失不再下降
- Epoch: 通常3-5个epoch

**代码示例**:
```java
// 1. 加载预训练模型
GPT1Model model = GPT1Model.loadModel("checkpoints/pretrain/final.model");

// 2. 准备微调数据
GPT1Dataset trainDataset = new GPT1Dataset(512, 32, vocabSize);
trainDataset.loadFromFile("data/finetune_train.txt", tokenizer);

GPT1Dataset valDataset = new GPT1Dataset(512, 32, vocabSize);
valDataset.loadFromFile("data/finetune_val.txt", tokenizer);

// 3. 配置微调训练器
GPT1Finetune finetuner = new GPT1Finetune(model, trainDataset, valDataset);
finetuner.configure(
    5,        // maxEpochs
    2.5e-5f,  // learningRate
    3         // patience
).setCheckpoint("./checkpoints/finetune", 100);

// 4. 开始微调
finetuner.train();
```

## 推理策略

支持多种文本生成方法:

### 1. 贪婪解码 (Greedy Decoding)
- **特点**: 始终选择概率最高的token
- **适用**: 需要确定性输出的任务
- **示例**:
```java
GPT1Inference inference = new GPT1Inference(model);
int[] result = inference.generateGreedy(promptIds, 50);
```

### 2. Temperature采样
- **特点**: 控制输出的随机性
- **参数**: temperature < 1更确定，> 1更随机
- **示例**:
```java
int[] result = inference.generateWithTemperature(promptIds, 50, 0.8f);
```

### 3. Top-K采样
- **特点**: 从概率最高的K个token中采样
- **适用**: 避免采样到低概率token
- **示例**:
```java
int[] result = inference.generateTopK(promptIds, 50, 40, 1.0f);
```

### 4. Top-P采样 (Nucleus Sampling)
- **特点**: 动态候选集，从累积概率达到p的最小token集合中采样
- **适用**: 平衡质量与多样性
- **示例**:
```java
int[] result = inference.generateTopP(promptIds, 50, 0.9f, 1.0f);
```

### 5. Beam Search
- **特点**: 维护多个候选序列，选择全局最优
- **适用**: 需要最高生成质量
- **示例**:
```java
int[] result = inference.generateBeamSearch(promptIds, 50, 5);
```

## 快速开始

### 1. 模型创建
```java
// 创建标准配置模型(117M参数)
GPT1Model model = GPT1Model.createStandardModel("my-gpt1");

// 创建小型配置(28M参数)
GPT1Model model = GPT1Model.createSmallModel("my-gpt1-small");

// 创建微型配置(2.3M参数,用于测试)
GPT1Model model = GPT1Model.createTinyModel("my-gpt1-tiny");

// 自定义配置
GPT1Config config = new GPT1Config();
config.setVocabSize(50000);
config.setNEmbd(768);
config.setNLayer(12);
GPT1Model model = new GPT1Model("my-gpt1-custom", config);
```

### 2. 模型信息
```java
// 打印模型详细信息
model.printModelInfo();

// 获取配置摘要
String summary = model.getConfigSummary();
System.out.println(summary);

// 获取参数统计
long paramCount = model.getConfig().estimateParameterCount();
```

### 3. 完整训练演示
```java
// 运行完整训练演示
GPT1TrainDemo.main(new String[]{});

// 或单独演示各阶段
GPT1TrainDemo.runCompleteWorkflow();
```

## 预设配置

### Tiny配置(用于测试)
- 词汇表: 1000
- 嵌入维度: 128
- 层数: 4
- 注意力头: 4
- FFN维度: 512
- 参数量: ~2.3M

### Small配置
- 词汇表: 10000
- 嵌入维度: 512
- 层数: 8
- 注意力头: 8
- FFN维度: 2048
- 参数量: ~28M

### Standard配置(原论文)
- 词汇表: 40478
- 嵌入维度: 768
- 层数: 12
- 注意力头: 12
- FFN维度: 3072
- 参数量: ~117M

## 性能优化建议

### 训练阶段
1. **梯度累积**: 模拟更大的batch size
2. **混合精度**: 使用FP16加速训练(需硬件支持)
3. **检查点**: 定期保存避免训练中断
4. **学习率调度**: warmup + cosine decay
5. **梯度裁剪**: 防止梯度爆炸

### 推理阶段
1. **批量推理**: 并行处理多个序列
2. **KV缓存**: 缓存注意力键值对(暂未实现)
3. **量化**: 模型参数量化减小内存占用
4. **选择合适的生成策略**: 
   - 速度优先: 贪婪解码
   - 质量优先: Beam Search
   - 平衡: Top-P采样

## 注意事项

1. **内存管理**: 训练大模型需要充足的内存
2. **数据质量**: 预训练数据质量直接影响模型性能
3. **超参数调优**: 不同任务可能需要不同的学习率
4. **早停机制**: 微调时监控验证集避免过拟合
5. **模型评估**: 使用困惑度(Perplexity)评估语言模型

## 参考资料

- 论文: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)
- 原始实现: https://github.com/openai/finetune-transformer-lm

## 贡献

欢迎提交问题和改进建议!

## 许可证

本实现遵循TinyAI项目的开源许可证。
