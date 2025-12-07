# MiniMind 使用示例

本目录包含TinyAI MiniMind模块的8个完整使用示例代码。

## 📚 示例列表

### ✅ 01. 模型创建与推理
**文件**: `Example01_ModelCreationAndInference.java`  
**内容**:
- 创建不同规模的模型配置(Small/Medium/Tiny)
- 初始化模型和Tokenizer
- 单次推理和批量推理
- 模型信息查看和配置对比

**运行**: 
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example01_ModelCreationAndInference"
```

---

### ✅ 02. BPE Tokenizer训练
**文件**: `Example02_BPETokenizerTraining.java`  
**内容**:
- 准备训练语料
- 训练BPE Tokenizer
- 编码解码测试
- 保存和加载Tokenizer模型

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example02_BPETokenizerTraining"
```

---

### ✅ 03. 监督微调(SFT)
**文件**: `Example03_SFTFineTuning.java`  
**内容**:
- 准备SFT数据集(问答对、指令格式)
- 数据预处理
- SFT训练配置(小学习率、正则化)
- 训练步骤演示
- 最佳实践和注意事项

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example03_SFTFineTuning"
```

---

### ✅ 04. LoRA微调
**文件**: `Example04_LoRAFineTuning.java`  
**内容**:
- 创建LoRA适配器
- 参数效率分析
- LoRA应用策略说明
- Rank和Alpha参数选择建议

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example04_LoRAFineTuning"
```

---

### ✅ 05. 预训练
**文件**: `Example05_PreTraining.java`  
**内容**:
- 预训练数据准备(大规模语料)
- 预训练配置(学习率、warm-up、调度)
- 训练监控和评估
- 优化技巧(混合精度、梯度累积、并行)

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example05_PreTraining"
```

---

### ✅ 06. 文本生成策略
**文件**: `Example06_TextGenerationStrategies.java`  
**内容**:
- Greedy Search (贪心搜索)
- Temperature采样 (低温/高温)
- Top-K采样
- Top-P (Nucleus)采样
- 组合策略
- 策略选择建议

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example06_TextGenerationStrategies"
```

---

### ✅ 07. 模型评估
**文件**: `Example07_ModelEvaluation.java`  
**内容**:
- 评估指标(困惑度、准确率、生成质量)
- 困惑度计算
- 生成质量评估(流畅性、连贯性、相关性)
- 性能基准测试(延迟、吞吐量、内存)
- 问题诊断

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example07_ModelEvaluation"
```

---

### ✅ 08. 完整训练流程
**文件**: `Example08_CompleteTrainingPipeline.java`  
**内容**:
- 环境准备
- 数据准备(分割、预处理、DataLoader)
- 模型创建和初始化
- 训练配置(优化器、调度器、损失)
- 训练循环实现
- 验证评估
- 模型保存

**运行**:
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example08_CompleteTrainingPipeline"
```

---

## 🚀 快速开始

### 1. 编译项目
```bash
cd tinyai-model/tinyai-model-minimind
mvn clean compile
```

### 2. 运行示例
选择任一示例运行:
```bash
# 示例1: 模型创建与推理
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example01_ModelCreationAndInference"

# 示例2: BPE训练
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example02_BPETokenizerTraining"

# 示例3: SFT微调
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.minimind.examples.Example03_SFTFineTuning"

# ... 其他示例类似
```

### 3. 修改示例
所有示例代码都包含详细注释,可以直接修改参数进行实验。

---

## 📖 示例说明

### 示例难度
- 🟢 初级: Example01 (模型创建与推理)
- 🟡 中级: Example02 (BPE训练), Example06 (生成策略), Example07 (模型评估)
- 🔴 高级: Example03 (SFT微调), Example04 (LoRA微调), Example05 (预训练), Example08 (完整流程)

### 推荐学习顺序
1. **Example01** - 了解模型基本使用
2. **Example02** - 学习Tokenizer训练
3. **Example06** - 掌握文本生成策略
4. **Example07** - 学习模型评估
5. **Example04** - 深入参数高效微调
6. **Example03** - 掌握监督微调
7. **Example05** - 了解预训练流程
8. **Example08** - 掌握完整训练流程

---

## 💡 使用提示

### 模型配置选择
- **快速测试**: 使用Tiny配置 (64维, 2层, ~30K参数)
- **实验开发**: 使用Small配置 (512维, 8层, ~26M参数)
- **生产部署**: 使用Medium配置 (768维, 16层, ~108M参数)

### 常见问题

**Q: 示例运行时内存不足?**  
A: 减小模型配置,使用Tiny或降低batch size

**Q: 如何保存和加载训练好的模型?**  
A: 参考Example02中Tokenizer的保存/加载方式,模型类似

**Q: 生成文本质量不好?**  
A: 需要先训练模型,随机初始化的模型无法生成有意义文本

**Q: 如何选择生成策略?**  
A: 参考Example06中的详细说明和建议

**Q: 如何评估模型性能?**  
A: 参考Example07中的评估方法和指标

**Q: 完整训练需要注意什么?**  
A: 参考Example08中的训练流程和最佳实践

---

## 📚 相关文档

- [技术架构文档](../../../../../../../doc/module-creation.md)
- [LoRA实现指南](../../../../../../../doc/LoRAImplementationGuide.md)
- [API参考文档](../../../../../../../doc/API参考.md)
- [TODO清单](../../../../../../../doc/TODO.md)

---

## ⚠️ 注意事项

1. **内存管理**: 大模型需要较多内存,建议从小模型开始
2. **GPU支持**: 当前版本为CPU实现,GPU版本待开发
3. **训练数据**: 示例中的训练数据仅用于演示,实际使用需要更大规模数据
4. **模型权重**: 示例使用随机初始化,实际应用需要训练或加载预训练权重
5. **示例代码**: 所有示例都是演示性质,展示API使用方式和最佳实践

---

**最后更新**: 2025-12-07  
**作者**: leavesfly  
**示例总数**: 8个
