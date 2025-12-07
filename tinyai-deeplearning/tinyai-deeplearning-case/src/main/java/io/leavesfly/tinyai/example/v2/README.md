# TinyAI V2 模块示例代码

本目录包含了TinyAI深度学习框架V2版本的完整示例代码，展示了如何使用各种神经网络层和模块。

## 📚 示例列表

### 1. BasicUsageExample - 基础使用示例
**展示内容:**
- 创建简单的全连接网络
- 使用train()和eval()模式切换
- 访问和管理模型参数
- 查看子模块结构

**适合人群:** 初学者，刚开始使用V2模块

**关键概念:**
- Module基类的使用
- 参数注册和访问
- 训练/推理模式切换

**运行方式:**
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.v2.BasicUsageExample"
```

### 2. SequentialExample - Sequential容器示例
**展示内容:**
- 使用Sequential容器快速构建模型
- 链式调用添加模块
- Sequential容器的前向传播

**适合人群:** 需要快速构建简单线性网络的开发者

**关键概念:**
- Sequential容器的使用
- 链式调用模式
- 模块自动注册

**运行方式:**
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.v2.SequentialExample"
```

### 3. CNNExample - 卷积神经网络示例
**展示内容:**
- 构建LeNet-5风格的卷积神经网络
- 使用Conv2d、MaxPool2d等卷积层
- 处理图像数据的形状变换
- 实现完整的分类器

**适合人群:** 计算机视觉任务开发者

**关键概念:**
- 卷积层和池化层
- 特征提取和分类
- 4D张量的展平操作

**模型结构:**
```
输入 (28x28) 
  ↓
Conv2d (6通道, 5x5) → ReLU → MaxPool (2x2)
  ↓
Conv2d (16通道, 5x5) → ReLU → MaxPool (2x2)
  ↓
展平 → Linear (120) → ReLU → Dropout
  ↓
Linear (84) → ReLU
  ↓
Linear (10输出)
```

**运行方式:**
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.v2.CNNExample"
```

### 4. RNNExample - RNN序列建模示例
**展示内容:**
- 使用LSTM、GRU、SimpleRNN处理序列数据
- 管理RNN的隐藏状态
- 构建序列分类模型
- 比较不同RNN变体的参数量

**适合人群:** 自然语言处理和时序任务开发者

**关键概念:**
- 循环神经网络
- 隐藏状态管理
- 序列处理

**模型对比:**
- **LSTM**: 3个门（输入门、遗忘门、输出门）+ 细胞状态，参数最多
- **GRU**: 2个门（重置门、更新门），参数适中
- **SimpleRNN**: 无门控机制，参数最少

**运行方式:**
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.v2.RNNExample"
```

### 5. TransformerExample - Transformer模型示例
**展示内容:**
- 使用多头注意力机制
- 使用位置编码
- 构建Transformer编码器和解码器

**适合人群:** 需要处理序列到序列任务的开发者

**关键概念:**
- 多头自注意力
- 位置编码
- 编码器-解码器架构

**应用场景:**
- 机器翻译
- 文本摘要
- 问答系统
- 语言建模

**运行方式:**
```bash
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.v2.TransformerExample"
```

## 🚀 快速开始

1. **确保依赖已配置**
   - 项目已自动依赖 `tinyai-deeplearning-nnet` 模块

2. **运行示例**
   - 使用Maven执行插件运行任意示例
   - 或直接在IDE中运行main方法

3. **学习路径建议**
   - 初学者：从 `BasicUsageExample` 开始
   - 视觉任务：学习 `CNNExample`
   - 序列任务：学习 `RNNExample` 和 `TransformerExample`
   - 快速原型：使用 `SequentialExample`

## 📝 注意事项

1. **模式切换**: 训练时使用 `model.train()`，推理时使用 `model.eval()`
2. **参数管理**: 使用 `namedParameters()` 访问所有参数
3. **模块注册**: 子模块必须通过 `registerModule()` 注册才能被正确管理
4. **形状匹配**: 确保输入数据的形状与模型期望的形状匹配

## 🔗 相关资源

- V2模块详细文档：`tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/v2/examples/README.md`
- V2模块源码：`tinyai-deeplearning-nnet/src/main/java/io/leavesfly/tinyai/nnet/v2/`

## 📧 反馈

如有问题或建议，请提交Issue或Pull Request。

