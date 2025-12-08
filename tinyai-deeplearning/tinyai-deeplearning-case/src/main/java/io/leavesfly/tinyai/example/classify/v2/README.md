# TinyAI Neural Network V2 分类示例

## 概述

本目录包含使用 TinyAI Neural Network V2 API 重写的分类示例代码。这些示例展示了 V2 版本的主要特性：

- **现代化API设计**：使用 `Sequential` 容器和链式调用
- **统一参数管理**：`registerParameter()` 和 `namedParameters()`
- **灵活初始化**：`Initializers` 工具类提供丰富的初始化策略
- **模式切换**：`train()` / `eval()` 方法控制训练和推理模式
- **手动训练循环**：展示完整的训练流程控制

## 文件说明

### MnistMlpExamV2.java
**MNIST手写数字识别分类器**

- **网络结构**：3层MLP (784 → 100 → 100 → 10)
- **数据集**：MNIST手写数字数据集
- **训练方式**：手动训练循环
- **特性展示**：
  - Sequential容器构建网络
  - Kaiming均匀初始化
  - 训练进度监控
  - 参数统计

### SpiralMlpExamV2.java
**螺旋数据集分类器**

- **网络结构**：3层MLP (2 → 30 → 30 → 3)
- **数据集**：螺旋数据集（经典非线性分类问题）
- **训练方式**：
  - `test()`: 简化训练方式
  - `test1()`: 详细训练方式（包含可视化）
- **特性展示**：
  - 模式切换（train/eval）
  - 训练结果可视化
  - 多批次训练监控

## V1 vs V2 对比

### 网络构建对比

**V1版本**：
```java
Block block = new MlpBlock("MlpBlock", batchSize, Config.ActiveFunc.Sigmoid, inputSize, hiddenSize1, hiddenSize2, outputSize);
Model model = new Model("MnistMlpExam", block);
```

**V2版本**：
```java
Sequential model = new Sequential("MnistMlpV2")
    .add(new Linear("fc1", inputSize, hiddenSize1))
    .add(new ReLU())
    .add(new Linear("fc2", hiddenSize1, hiddenSize2))
    .add(new ReLU())
    .add(new Linear("fc3", hiddenSize2, outputSize));
```

### 初始化对比

**V1版本**：
```java
// 隐式初始化，硬编码在层内部
```

**V2版本**：
```java
model.apply(module -> {
    if (module instanceof Linear) {
        Linear linear = (Linear) module;
        Initializers.kaimingUniform(linear.getWeight().data());
        if (linear.getBias() != null) {
            Initializers.zeros(linear.getBias().data());
        }
    }
});
```

### 训练循环对比

**V1版本**：
```java
Trainer trainer = new Trainer(maxEpoch, new Monitor(), evaluator);
trainer.init(dataSet, model, loss, optimizer);
trainer.train(true);
trainer.evaluate();
```

**V2版本**：
```java
model.train();  // 训练模式
for (int epoch = 0; epoch < maxEpoch; epoch++) {
    for (Batch batch : batches) {
        Variable output = model.forward(batch.input);
        Variable loss = lossFunc.loss(batch.target, output);
        model.clearGrads();
        loss.backward();
        optimizer.update();
    }
}
model.eval();   // 推理模式
evaluator.evaluate();
```

## V2 API 优势

### 1. 更清晰的网络结构
- 显式定义每一层
- 链式调用，代码可读性强
- 容易理解和修改网络架构

### 2. 统一的参数管理
- `namedParameters()` 提供分层参数路径
- 支持参数冻结和部分更新
- 便于模型保存和加载

### 3. 灵活的初始化策略
- 丰富的内置初始化器
- 支持自定义初始化逻辑
- 通过 `apply()` 方法统一应用

### 4. 明确的模式控制
- `train()`: 启用训练特性（如Dropout）
- `eval()`: 禁用训练特性，使用推理逻辑
- 避免训练和推理混用的问题

### 5. 更好的扩展性
- 容易添加新的层类型
- 支持自定义模块组合
- 便于实现复杂的网络架构

## 运行示例

### 编译和运行
```bash
# 编译项目
mvn clean compile

# 运行MNIST分类器
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.classify.v2.MnistMlpExamV2"

# 运行螺旋数据集分类器
mvn exec:java -Dexec.mainClass="io.leavesfly.tinyai.example.classify.v2.SpiralMlpExamV2"
```

### 预期输出示例

**MnistMlpExamV2.java**:
```
开始训练 MNIST MLP 分类器 (V2 API)
模型结构:
Sequential(name=MnistMlpV2, modules=[
  Linear(name=fc1, in=784, out=100, bias=true),
  ReLU(name=null),
  Linear(name=fc2, in=100, out=100, bias=true),
  ReLU(name=null),
  Linear(name=fc3, in=100, out=10, bias=true)
])

epoch = 0, loss: 1.8379626, accuracy: 0.9143001
...
最终损失: 0.21598813
最终准确率: 0.9143001
```

**SpiralMlpExamV2.java**:
```
=== Spiral MLP 分类器 (V2 API - 详细训练方式) ===
i=0, loss:1.098612, acc:0.333333 (批次:30)
...
i=299, loss:0.012345, acc:0.996667 (批次:30)
```

## 迁移指南

如果您需要从V1代码迁移到V2，请参考以下步骤：

1. **更新导入**：将 `io.leavesfly.tinyai.nnet.*` 替换为 `io.leavesfly.tinyai.nnet.v2.*`
2. **重构网络**：使用 `Sequential` 替换 `Block`，使用 `Linear` 替换 `MlpBlock`
3. **更新初始化**：使用 `Initializers` 工具类替换硬编码初始化
4. **重写训练循环**：手动实现训练循环，添加 `train()`/`eval()` 模式切换
5. **测试验证**：确保训练效果与V1版本一致

详细的迁移指南请参考：`tinyai-deeplearning-nnet/doc/v2/migration-guide.md`

## 依赖说明

V2版本使用了以下组件：

- **V1组件**（保持兼容）：
  - `DataSet` / `MnistDataSet` / `SpiralDateSet`
  - `Evaluator` / `AccuracyEval`
  - `Loss` / `SoftmaxCrossEntropy`
  - `Optimizer` / `SGD`
  - `Plot`（可视化）

- **V2组件**（新增特性）：
  - `Sequential` - 顺序容器
  - `Linear` - 全连接层
  - `ReLU` - 激活函数
  - `Initializers` - 初始化工具类

## 性能对比

V2版本在保持相同底层计算引擎的前提下，提供：

- **更清晰的代码结构**：网络定义更直观
- **更好的调试体验**：参数路径清晰，便于调试
- **更强的扩展性**：容易添加新的网络组件
- **更完善的特性**：支持模式切换、延迟初始化等

实际训练性能与V1版本基本一致，额外的抽象层开销可以忽略不计。

## 下一步

- 探索更多V2特性：BatchNorm、Dropout、自定义模块
- 尝试不同的网络架构：CNN、RNN、Transformer
- 学习模型保存和加载：`stateDict()` 和 `loadStateDict()`
- 实现更复杂的训练策略：学习率调度、早停等
