# MiniMind API 参考文档

## 📚 文档概述

本文档提供MiniMind模块所有公开API的完整参考,包括模型创建、训练、推理、分词等核心功能。

**文档版本**: v1.0.0  
**更新时间**: 2025-12-07  
**适用版本**: MiniMind 1.0+

---

## 目录

1. [模型API](#1-模型api)
   - [1.1 模型创建](#11-模型创建)
   - [1.2 模型推理](#12-模型推理)
   - [1.3 文本生成](#13-文本生成)
   - [1.4 模型管理](#14-模型管理)

2. [训练API](#2-训练api)
   - [2.1 预训练](#21-预训练)
   - [2.2 监督微调(SFT)](#22-监督微调sft)
   - [2.3 LoRA微调](#23-lora微调)
   - [2.4 DPO训练](#24-dpo训练)

3. [Tokenizer API](#3-tokenizer-api)
   - [3.1 编码/解码](#31-编码解码)
   - [3.2 批处理](#32-批处理)
   - [3.3 BPE训练](#33-bpe训练)

4. [配置管理API](#4-配置管理api)
   - [4.1 模型配置](#41-模型配置)
   - [4.2 训练配置](#42-训练配置)

5. [工具类API](#5-工具类api)
   - [5.1 词汇表管理](#51-词汇表管理)
   - [5.2 数据处理](#52-数据处理)

6. [异常处理](#6-异常处理)

---

## 1. 模型API

### 1.1 模型创建

#### 1.1.1 使用预设配置创建模型

**方法签名**:
```java
public static MiniMindModel create(String name, String modelSize)
```

**功能描述**:  
使用预设配置快速创建模型实例,支持三种规模:small(26M)、medium(108M)、moe(145M)。

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `name` | String | 是 | 模型名称,用于标识和日志 |
| `modelSize` | String | 是 | 模型规模,可选值:"small", "medium", "moe" |

**返回值**:  
`MiniMindModel` - 已初始化的模型实例

**异常**:  
- `IllegalArgumentException` - 当modelSize不在支持列表中时

**代码示例**:
```java
// 创建小型模型(26M参数)
MiniMindModel smallModel = MiniMindModel.create("my-small-model", "small");

// 创建中型模型(108M参数)
MiniMindModel mediumModel = MiniMindModel.create("my-medium-model", "medium");

// 创建MoE模型(145M参数,4专家)
MiniMindModel moeModel = MiniMindModel.create("my-moe-model", "moe");

// 打印模型信息
System.out.println(smallModel.getDescription());
// 输出: MiniMind Language Model - small with 26M parameters
```

**最佳实践**:
- 初学者推荐使用"small"配置,训练和推理速度快
- 生产环境推荐使用"medium"配置,性能更好
- MoE配置适用于需要大容量但受限于计算资源的场景

---

#### 1.1.2 使用自定义配置创建模型

**方法签名**:
```java
public MiniMindModel(String name, MiniMindConfig config)
```

**功能描述**:  
使用自定义配置创建模型,提供完全的灵活性。

**参数**:

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `name` | String | 是 | 模型名称 |
| `config` | MiniMindConfig | 是 | 模型配置对象 |

**返回值**:  
`MiniMindModel` - 模型实例

**代码示例**:
```java
// 创建自定义配置
MiniMindConfig config = new MiniMindConfig();
config.setVocabSize(8000);          // 词汇表大小
config.setMaxSeqLen(1024);          // 最大序列长度
config.setHiddenSize(512);          // 隐藏维度
config.setNumLayers(12);            // Transformer层数
config.setNumHeads(8);              // 注意力头数
config.setFfnHiddenSize(2048);      // FFN隐藏维度
config.setDropout(0.1f);            // Dropout比例
config.setActivationFunction("silu"); // 激活函数
config.setUseRoPE(true);            // 使用RoPE位置编码
config.setPreLayerNorm(true);       // 使用Pre-LN

// 创建模型
MiniMindModel model = new MiniMindModel("custom-model", config);
```

**配置验证**:
```java
// 配置会自动验证
try {
    config.validate(); // 检查配置的合法性
} catch (IllegalArgumentException e) {
    System.err.println("配置错误: " + e.getMessage());
}
```

---

### 1.2 模型推理

#### 1.2.1 前向传播

**方法签名**:
```java
public Variable predict(Variable tokenIds)
public NdArray predict(NdArray tokenIds)
```

**功能描述**:  
执行单次前向传播,计算输入token序列的logits输出。

**参数**:

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `tokenIds` | Variable/NdArray | Token IDs,形状[batch_size, seq_len] |

**返回值**:  
- `Variable` - Logits输出,形状[batch_size, seq_len, vocab_size]
- `NdArray` - Logits NdArray,形状[batch_size, seq_len, vocab_size]

**代码示例**:
```java
// 准备输入
int[][] tokenIds = {{1, 234, 567, 89}}; // batch=1, seq_len=4
NdArray inputArray = NdArray.of(tokenIds);
Variable inputVar = new Variable(inputArray);

// 前向传播
Variable logits = model.predict(inputVar);

// 获取输出形状
int[] shape = logits.getValue().getShape().getShapeDims();
System.out.println("输出形状: [" + shape[0] + ", " + shape[1] + ", " + shape[2] + "]");
// 输出: 输出形状: [1, 4, 6400]
```

**使用场景**:
- 训练时计算损失
- 批量推理
- 特征提取

---

### 1.3 文本生成

#### 1.3.1 自回归生成

**方法签名**:
```java
public int[] generate(int[] promptTokenIds, 
                      int maxNewTokens, 
                      float temperature, 
                      int topK, 
                      float topP)
```

**功能描述**:  
给定提示词token序列,自回归生成新的token。支持温度采样、Top-K采样和Nucleus(Top-P)采样。

**参数**:

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `promptTokenIds` | int[] | 是 | - | 提示词token IDs |
| `maxNewTokens` | int | 是 | - | 最大生成token数量 |
| `temperature` | float | 是 | 1.0 | 温度参数,0.0=贪婪,>1.0=随机 |
| `topK` | int | 否 | 0 | Top-K采样,0表示不使用 |
| `topP` | float | 否 | 0.0 | Top-P采样,0.0表示不使用 |

**返回值**:  
`int[]` - 生成的完整token序列(包含提示词)

**代码示例**:
```java
// 准备提示词
String prompt = "Hello, world!";
List<Integer> promptTokens = tokenizer.encode(prompt, true, false);
int[] promptArray = promptTokens.stream().mapToInt(i -> i).toArray();

// 生成文本 - 贪婪采样
int[] output1 = model.generate(promptArray, 50, 0.0f, 0, 0.0f);

// 生成文本 - 温度采样
int[] output2 = model.generate(promptArray, 50, 0.7f, 0, 0.0f);

// 生成文本 - Top-K采样
int[] output3 = model.generate(promptArray, 50, 1.0f, 40, 0.0f);

// 生成文本 - Top-P(Nucleus)采样
int[] output4 = model.generate(promptArray, 50, 1.0f, 0, 0.9f);

// 解码输出
String generatedText = tokenizer.decode(Arrays.stream(output1)
    .boxed().collect(Collectors.toList()));
System.out.println("生成文本: " + generatedText);
```

**采样策略说明**:

| 策略 | temperature | topK | topP | 特点 |
|------|-------------|------|------|------|
| 贪婪采样 | 0.0 | 0 | 0.0 | 确定性,每次选择概率最高的token |
| 温度采样 | 0.1-2.0 | 0 | 0.0 | 控制随机性,temperature越大越随机 |
| Top-K | 1.0 | 20-100 | 0.0 | 仅从概率最高的K个token中采样 |
| Top-P | 1.0 | 0 | 0.8-0.95 | 动态选择,累计概率达到P |

**性能优化**:
- 生成时使用KV-Cache加速,避免重复计算
- 批量生成时共享KV-Cache
- 遇到EOS token自动停止

---

### 1.4 模型管理

#### 1.4.1 训练模式切换

**方法签名**:
```java
public void setTraining(boolean training)
public boolean isTraining()
```

**功能描述**:  
切换模型的训练/评估模式,影响Dropout和BatchNorm等层的行为。

**代码示例**:
```java
// 切换到训练模式
model.setTraining(true);
// Dropout生效,参数可更新

// 切换到评估模式
model.setTraining(false);
// Dropout关闭,模型固定

// 检查当前模式
boolean isTraining = model.isTraining();
System.out.println("训练模式: " + isTraining);
```

---

#### 1.4.2 参数管理

**方法签名**:
```java
public List<Parameter> getAllParams()
public void clearGrads()
```

**功能描述**:  
获取所有可训练参数,清空梯度。

**代码示例**:
```java
// 获取所有参数
List<Parameter> params = model.getAllParams();
System.out.println("参数总数: " + params.size());

// 计算参数量
long totalParams = params.stream()
    .mapToLong(p -> p.getData().getBuffer().length)
    .sum();
System.out.println("参数量: " + totalParams);

// 清空梯度(训练前必须调用)
model.clearGrads();
```

---

#### 1.4.3 模型信息

**方法签名**:
```java
public String getName()
public String getDescription()
public MiniMindConfig getConfig()
```

**代码示例**:
```java
// 获取模型名称
String name = model.getName();

// 获取模型描述
String desc = model.getDescription();

// 获取配置
MiniMindConfig config = model.getConfig();
System.out.println("词汇表大小: " + config.getVocabSize());
System.out.println("层数: " + config.getNumLayers());
System.out.println("参数估算: " + config.estimateParameters());
```

---

## 2. 训练API

### 2.1 预训练

#### 2.1.1 预训练配置

**类名**: `PretrainConfig`

**配置项**:

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `dataPath` | String | 必填 | 训练数据路径 |
| `batchSize` | int | 32 | 批次大小 |
| `learningRate` | float | 3e-4 | 学习率 |
| `numEpochs` | int | 10 | 训练轮数 |
| `warmupSteps` | int | 1000 | 学习率预热步数 |
| `maxGradNorm` | float | 1.0 | 梯度裁剪阈值 |
| `saveSteps` | int | 1000 | 保存检查点间隔 |
| `logSteps` | int | 100 | 日志输出间隔 |
| `checkpointDir` | String | "./checkpoints" | 检查点保存目录 |

**代码示例**:
```java
// 创建预训练配置
PretrainConfig config = new PretrainConfig();
config.setDataPath("/path/to/pretrain/data");
config.setBatchSize(64);
config.setLearningRate(3e-4f);
config.setNumEpochs(20);
config.setWarmupSteps(2000);
config.setMaxGradNorm(1.0f);
config.setSaveSteps(5000);
config.setLogSteps(100);
config.setCheckpointDir("./checkpoints/pretrain");

// 创建训练器
PretrainTrainer trainer = new PretrainTrainer(config);

// 开始训练
model.setTraining(true);
trainer.train(model);
```

---

### 2.2 监督微调(SFT)

#### 2.2.1 SFT配置

**类名**: `SFTConfig`

**配置项**:

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `dataPath` | String | 必填 | SFT数据路径(JSONL格式) |
| `batchSize` | int | 16 | 批次大小 |
| `learningRate` | float | 5e-5 | 学习率(比预训练小) |
| `numEpochs` | int | 3 | 微调轮数 |
| `maxSeqLen` | int | 512 | 最大序列长度 |
| `lossOnOutputOnly` | boolean | true | 仅计算输出部分损失 |

**代码示例**:
```java
// SFT数据格式示例(JSONL):
// {"instruction": "写一首诗", "input": "", "output": "春眠不觉晓..."}
// {"instruction": "翻译", "input": "Hello", "output": "你好"}

// 创建SFT配置
SFTConfig sftConfig = new SFTConfig();
sftConfig.setDataPath("/path/to/sft/data.jsonl");
sftConfig.setBatchSize(16);
sftConfig.setLearningRate(5e-5f);
sftConfig.setNumEpochs(3);
sftConfig.setLossOnOutputOnly(true);

// 加载预训练模型
MiniMindModel model = MiniMindModel.create("sft-model", "small");
// 加载预训练权重
// model.load("checkpoints/pretrain/epoch_10.pt");

// 创建训练器并训练
SFTTrainer sftTrainer = new SFTTrainer(sftConfig);
sftTrainer.train(model);
```

---

### 2.3 LoRA微调

#### 2.3.1 LoRA配置

**类名**: `LoRAConfig`

**配置项**:

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `rank` | int | 8 | LoRA秩(r) |
| `alpha` | float | 16.0 | 缩放因子(α) |
| `dropout` | float | 0.1 | LoRA Dropout |
| `targetModules` | List<String> | ["q", "v"] | 目标模块名称 |
| `mergeWeights` | boolean | false | 是否合并权重 |

**代码示例**:
```java
// 创建LoRA配置
LoRAConfig loraConfig = new LoRAConfig();
loraConfig.setRank(8);              // 秩r=8
loraConfig.setAlpha(16.0f);         // alpha=16
loraConfig.setDropout(0.1f);
loraConfig.setTargetModules(Arrays.asList("q_proj", "v_proj")); // QV注意力
loraConfig.setMergeWeights(false);  // 保持分离

// 应用LoRA到模型
LoRAAdapter.applyLoRA(model, loraConfig);

// 训练(只更新LoRA参数,冻结原始权重)
SFTConfig sftConfig = new SFTConfig();
sftConfig.setDataPath("/path/to/sft/data.jsonl");
sftConfig.setLearningRate(1e-4f);  // LoRA可用更大学习率

SFTTrainer trainer = new SFTTrainer(sftConfig);
trainer.train(model);

// 保存LoRA权重(单独保存,约原模型1%)
LoRAAdapter.save(model, "lora_weights.pt");

// 加载LoRA权重
LoRAAdapter.load(model, "lora_weights.pt");

// 合并权重(可选)
if (loraConfig.isMergeWeights()) {
    LoRAAdapter.mergeWeights(model);
}
```

**LoRA优势**:
- 参数量小(仅训练1-2%参数)
- 训练速度快
- 显存占用低
- 可多个LoRA适配器切换

---

### 2.4 DPO训练

#### 2.4.1 DPO配置

**类名**: `DPOConfig`

**配置项**:

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `beta` | float | 0.1 | KL散度惩罚系数 |
| `refModelPath` | String | null | 参考模型路径(可选) |
| `dataPath` | String | 必填 | 偏好对数据路径 |
| `batchSize` | int | 16 | 批次大小 |
| `learningRate` | float | 5e-6 | 学习率 |
| `numEpochs` | int | 1 | 训练轮数 |

**代码示例**:
```java
// DPO数据格式(JSONL):
// {
//   "prompt": "写一首诗",
//   "chosen": "春眠不觉晓,处处闻啼鸟...",
//   "rejected": "床前明月光..."
// }

// 创建DPO配置
DPOConfig dpoConfig = new DPOConfig();
dpoConfig.setBeta(0.1f);           // KL惩罚系数
dpoConfig.setDataPath("/path/to/dpo/data.jsonl");
dpoConfig.setBatchSize(16);
dpoConfig.setLearningRate(5e-6f);  // DPO用小学习率
dpoConfig.setNumEpochs(1);

// 加载SFT模型作为策略模型
MiniMindModel policyModel = MiniMindModel.create("dpo-policy", "small");
// policyModel.load("checkpoints/sft/final.pt");

// 创建参考模型(冻结)
MiniMindModel refModel = MiniMindModel.create("dpo-ref", "small");
// refModel.load("checkpoints/sft/final.pt");
refModel.setTraining(false);

// 创建DPO训练器
DPOTrainer dpoTrainer = new DPOTrainer(dpoConfig, refModel);

// 开始DPO训练
dpoTrainer.train(policyModel);
```

**DPO损失公式**:
```
L_DPO = -log(σ(β · (log π_θ(y_w|x) - log π_θ(y_l|x) 
                    - log π_ref(y_w|x) + log π_ref(y_l|x))))
```

---

## 3. Tokenizer API

### 3.1 编码/解码

#### 3.1.1 文本编码

**方法签名**:
```java
public List<Integer> encode(String text)
public List<Integer> encode(String text, boolean addBos, boolean addEos)
```

**功能描述**:  
将文本编码为token ID序列。

**参数**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `text` | String | 必填 | 待编码文本 |
| `addBos` | boolean | true | 是否添加BOS token |
| `addEos` | boolean | true | 是否添加EOS token |

**返回值**:  
`List<Integer>` - Token IDs列表

**代码示例**:
```java
// 创建Tokenizer
MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(6400, 512);

// 编码文本(自动添加BOS/EOS)
List<Integer> tokens1 = tokenizer.encode("Hello, world!");

// 编码文本(不添加BOS/EOS)
List<Integer> tokens2 = tokenizer.encode("Hello, world!", false, false);

// 编码文本(仅添加BOS)
List<Integer> tokens3 = tokenizer.encode("Hello, world!", true, false);

System.out.println("Token IDs: " + tokens1);
// 输出: Token IDs: [1, 234, 567, 89, ..., 2]
```

---

#### 3.1.2 文本解码

**方法签名**:
```java
public String decode(List<Integer> tokenIds)
public String decode(List<Integer> tokenIds, boolean skipSpecialTokens)
```

**功能描述**:  
将token ID序列解码为文本。

**参数**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `tokenIds` | List<Integer> | 必填 | Token IDs列表 |
| `skipSpecialTokens` | boolean | true | 是否跳过特殊token |

**返回值**:  
`String` - 解码后的文本

**代码示例**:
```java
// 解码token序列(跳过特殊token)
String text1 = tokenizer.decode(tokens1);

// 解码token序列(保留特殊token)
String text2 = tokenizer.decode(tokens1, false);

System.out.println("解码文本: " + text1);
// 输出: 解码文本: Hello, world!
```

---

### 3.2 批处理

#### 3.2.1 批量编码

**方法签名**:
```java
public EncodedBatch encodeBatch(List<String> texts, boolean padding, int maxLength)
```

**功能描述**:  
批量编码多个文本,支持填充到相同长度。

**参数**:

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `texts` | List<String> | 文本列表 |
| `padding` | boolean | 是否填充到maxLength |
| `maxLength` | int | 最大长度 |

**返回值**:  
`EncodedBatch` - 编码批次,包含:
- `inputIds`: List<List<Integer>> - Token IDs
- `attentionMask`: List<List<Integer>> - 注意力掩码(1=有效,0=填充)

**代码示例**:
```java
// 准备批量文本
List<String> texts = Arrays.asList(
    "Hello, world!",
    "This is a longer sentence.",
    "Short."
);

// 批量编码(填充到最大长度)
EncodedBatch batch = tokenizer.encodeBatch(texts, true, 20);

// 获取input_ids和attention_mask
List<List<Integer>> inputIds = batch.getInputIds();
List<List<Integer>> attentionMask = batch.getAttentionMask();

// 转换为NdArray用于模型输入
NdArray inputArray = batch.toNdArray();
```

---

### 3.3 BPE训练

#### 3.3.1 从语料库训练BPE

**方法签名**:
```java
public static BPETrainer trainBPE(List<String> corpus, 
                                   int vocabSize, 
                                   int numMerges)
```

**功能描述**:  
从文本语料库学习BPE merge规则。

**参数**:

| 参数名 | 类型 | 说明 |
|--------|------|------|
| `corpus` | List<String> | 训练语料(文本列表) |
| `vocabSize` | int | 目标词汇表大小 |
| `numMerges` | int | BPE合并次数 |

**返回值**:  
`BPETrainer` - BPE训练器,包含学习的merge规则

**代码示例**:
```java
// 准备训练语料
List<String> corpus = new ArrayList<>();
corpus.add("Hello, world!");
corpus.add("This is a test.");
// ... 添加更多文本

// 训练BPE
BPETrainer bpeTrainer = BPETrainer.trainBPE(corpus, 6400, 5000);

// 保存BPE模型
bpeTrainer.save("./tokenizer_model");

// 从训练器创建Tokenizer
MiniMindTokenizer tokenizer = MiniMindTokenizer.fromBPETrainer(bpeTrainer, 512);

// 使用BPE编码
List<Integer> tokens = tokenizer.encode("Hello, world!");
```

---

#### 3.3.2 加载BPE模型

**方法签名**:
```java
public static BPETrainer load(String modelPath)
```

**代码示例**:
```java
// 加载已保存的BPE模型
BPETrainer loadedTrainer = BPETrainer.load("./tokenizer_model");

// 创建Tokenizer
MiniMindTokenizer tokenizer = MiniMindTokenizer.fromBPETrainer(loadedTrainer, 512);
```

---

## 4. 配置管理API

### 4.1 模型配置

#### 4.1.1 MiniMindConfig

**类名**: `MiniMindConfig`

**主要属性**:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `vocabSize` | int | 6400 | 词汇表大小 |
| `maxSeqLen` | int | 512 | 最大序列长度 |
| `hiddenSize` | int | 512 | 隐藏维度(d_model) |
| `numLayers` | int | 8 | Transformer层数 |
| `numHeads` | int | 16 | 注意力头数 |
| `ffnHiddenSize` | int | 1024 | FFN隐藏维度 |
| `dropout` | float | 0.1 | Dropout比例 |
| `activationFunction` | String | "silu" | 激活函数 |
| `useRoPE` | boolean | true | 是否使用RoPE |
| `preLayerNorm` | boolean | true | 是否使用Pre-LN |
| `useMoE` | boolean | false | 是否启用MoE |
| `numExperts` | int | 4 | MoE专家数量 |
| `numExpertsPerToken` | int | 2 | 每token激活专家数 |

**方法**:

```java
// 获取预设配置
public static MiniMindConfig createSmallConfig()   // 26M
public static MiniMindConfig createMediumConfig()  // 108M
public static MiniMindConfig createMoEConfig()     // 145M

// 获取计算属性
public int getHeadDim()                 // 每个头的维度
public String getModelSize()            // 模型规模标识
public long estimateParameters()        // 估算参数量

// 验证配置
public void validate()                  // 检查配置合法性
```

**代码示例**:
```java
// 使用预设配置
MiniMindConfig config = MiniMindConfig.createSmallConfig();

// 修改部分配置
config.setVocabSize(8000);
config.setMaxSeqLen(1024);

// 验证配置
config.validate();

// 获取信息
System.out.println("每头维度: " + config.getHeadDim());
System.out.println("参数估算: " + config.estimateParameters());
```

---

### 4.2 训练配置

#### 4.2.1 通用训练配置基类

所有训练配置继承自`TrainingConfig`,提供通用参数:

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `batchSize` | int | 32 | 批次大小 |
| `learningRate` | float | 3e-4 | 学习率 |
| `numEpochs` | int | 10 | 训练轮数 |
| `warmupSteps` | int | 1000 | 预热步数 |
| `maxGradNorm` | float | 1.0 | 梯度裁剪 |
| `weightDecay` | float | 0.01 | 权重衰减 |
| `logSteps` | int | 100 | 日志间隔 |
| `saveSteps` | int | 1000 | 保存间隔 |

---

## 5. 工具类API

### 5.1 词汇表管理

#### 5.1.1 Vocabulary

**类名**: `Vocabulary`

**主要方法**:

```java
// 构造函数
public Vocabulary(int maxSize)
public Vocabulary(Map<String, Integer> tokenToId)

// Token管理
public int addToken(String token)           // 添加token
public int getTokenId(String token)         // 获取token ID
public String getToken(int tokenId)         // 获取token字符串

// 特殊Token
public int getPadTokenId()                  // PAD token ID (0)
public int getBosTokenId()                  // BOS token ID (1)
public int getEosTokenId()                  // EOS token ID (2)
public int getUnkTokenId()                  // UNK token ID (3)

// 信息查询
public int getVocabSize()                   // 词汇表大小
public boolean containsToken(String token)  // 是否包含token
public Set<String> getAllTokens()           // 获取所有token

// 序列化
public void save(String filePath)           // 保存到文件
public static Vocabulary load(String path)  // 从文件加载
```

**代码示例**:
```java
// 创建词汇表
Vocabulary vocab = new Vocabulary(10000);

// 添加token
vocab.addToken("hello");
vocab.addToken("world");

// 查询
int id = vocab.getTokenId("hello");
String token = vocab.getToken(id);

// 获取特殊token
int padId = vocab.getPadTokenId();
int bosId = vocab.getBosTokenId();
int eosId = vocab.getEosTokenId();

// 保存/加载
vocab.save("vocab.txt");
Vocabulary loaded = Vocabulary.load("vocab.txt");
```

---

### 5.2 数据处理

#### 5.2.1 DataCollator

**功能**: 批量数据整理和填充

**代码示例**:
```java
// 准备批量数据
List<List<Integer>> batchTokens = Arrays.asList(
    Arrays.asList(1, 10, 20, 30),
    Arrays.asList(1, 15, 25),
    Arrays.asList(1, 12, 22, 32, 42)
);

// 填充到相同长度
DataCollator collator = new DataCollator(vocab.getPadTokenId());
CollatedBatch batch = collator.collate(batchTokens);

// 获取填充后的数据
NdArray inputIds = batch.getInputIds();      // [batch_size, max_len]
NdArray attentionMask = batch.getAttentionMask(); // [batch_size, max_len]
```

---

## 6. 异常处理

### 6.1 常见异常

#### 6.1.1 配置异常

**异常类**: `IllegalArgumentException`

**触发场景**:
- 无效的modelSize参数
- 配置验证失败
- 参数超出合法范围

**处理示例**:
```java
try {
    MiniMindModel model = MiniMindModel.create("model", "invalid_size");
} catch (IllegalArgumentException e) {
    System.err.println("配置错误: " + e.getMessage());
    // 使用默认配置
    model = MiniMindModel.create("model", "small");
}
```

---

#### 6.1.2 IO异常

**异常类**: `IOException`

**触发场景**:
- 模型保存/加载失败
- 数据文件读取失败
- Tokenizer模型加载失败

**处理示例**:
```java
try {
    BPETrainer trainer = BPETrainer.load("./tokenizer_model");
} catch (IOException e) {
    System.err.println("加载失败: " + e.getMessage());
    // 重新训练或使用备份
}
```

---

#### 6.1.3 形状不匹配异常

**异常类**: `IllegalArgumentException`

**触发场景**:
- 输入tensor形状不符合要求
- batch_size不一致

**处理示例**:
```java
try {
    // 输入形状必须是[batch_size, seq_len]
    Variable output = model.predict(invalidInput);
} catch (IllegalArgumentException e) {
    System.err.println("形状错误: " + e.getMessage());
    // 重新整理输入数据
}
```

---

## 7. 最佳实践

### 7.1 内存管理

```java
// 及时清空梯度
model.clearGrads();

// 推理时关闭训练模式
model.setTraining(false);

// 批量处理时使用适当的batch_size
int batchSize = availableMemory / estimatedBatchMemory;
```

---

### 7.2 性能优化

```java
// 使用KV-Cache加速生成
// generate()方法内部自动使用

// 批量推理
List<int[]> prompts = ...;
for (int[] prompt : prompts) {
    model.generate(prompt, 50, 0.7f, 0, 0.9f);
}
```

---

### 7.3 训练技巧

```java
// 1. 预训练使用大学习率
config.setLearningRate(3e-4f);

// 2. SFT使用小学习率
config.setLearningRate(5e-5f);

// 3. DPO使用极小学习率
config.setLearningRate(5e-6f);

// 4. 使用梯度裁剪防止爆炸
config.setMaxGradNorm(1.0f);

// 5. 使用warmup稳定训练
config.setWarmupSteps(2000);
```

---

## 8. 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2025-12-07 | 初始版本,完整API文档 |

---

## 9. 相关资源

- [快速开始指南](./快速开始指南.md)
- [使用示例](../examples/)
- [CLI工具指南](./CLI-GUIDE.md)
- [API服务指南](./API-GUIDE.md)
- [技术架构文档](./module-creation.md)

---

**文档维护**: TinyAI Team  
**问题反馈**: GitHub Issues
