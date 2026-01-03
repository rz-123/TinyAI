# tinyai-deeplearning-rl 计算图断裂问题修复总结

## 修复日期
2026-01-03

## 问题描述
tinyai-deeplearning-rl模块中存在多处计算图断裂问题，主要原因是在需要保持梯度传播的场景中，直接使用`getValue().get()`提取数值，导致计算图连接中断，梯度无法正确回传。

## 修复的文件及问题点

### 1. DQNAgent.java

#### 问题1: computeTargetQValues() - Line 254-273
**原问题:**
```java
// 直接提取数值，断开计算图
float maxNextQ = findMaxQValue(nextQValues);
targetValues[i] = rewards[i][0] + gamma * maxNextQ;
return new Variable(NdArray.of(targetValues, Shape.of(batchSize, 1)));
```

**修复方案:**
```java
// 使用Variable层面的max操作保持计算图连通性
maxNextQArray[i] = findMaxQValueVariable(nextQValuesArray[i]);

// 组装目标Q值：r + γ * max(Q(s', a'))
Variable rewardVar = new Variable(NdArray.of(rewards[i][0]));
Variable gammaVar = new Variable(NdArray.of(gamma));
Variable discountedQ = maxNextQArray[i].mul(gammaVar);
targetArray[i] = rewardVar.add(discountedQ);
```

**核心改进:**
- 新增`findMaxQValueVariable()`方法，使用`qValues.max(1, true)`保持计算图
- 使用Variable的add/mul操作构建目标值，而不是直接拼接数值
- 添加`stackVariables()`辅助方法合并批次Variable

#### 问题2: computeCurrentQValues() - Line 283-295
**原问题:**
```java
// 直接使用get()提取Q值
currentValues[i] = qValues.getValue().get(0, actionIndex);
return new Variable(NdArray.of(currentValues, Shape.of(batchSize, 1)));
```

**修复方案:**
```java
// 使用indexSelect保持计算图连通性
Variable indexVar = new Variable(NdArray.of(new float[]{actionIndex}));
currentQArray[i] = qValues.indexSelect(1, indexVar);
return stackVariables(currentQArray, batchSize);
```

**核心改进:**
- 使用`indexSelect()`替代直接索引访问，保持计算图连通
- 避免手动提取数值再构造Variable

### 2. REINFORCEAgent.java

#### 问题1: computeLogProbability() - Line 218-226
**原问题:**
```java
// 提取数值后手动计算log
float prob = probArray.get(0, action);
prob = Math.max(prob, 1e-8f);
float logProb = (float) Math.log(prob);
return new Variable(NdArray.of(logProb));
```

**修复方案:**
```java
// 使用indexSelect提取指定动作的概率，保持计算图连通
Variable indexVar = new Variable(NdArray.of(new float[]{action}));
Variable selectedProb = probabilities.indexSelect(1, indexVar);

// 使用Variable的log操作，保持计算图
Variable epsilon = new Variable(NdArray.of(1e-8f));
Variable clippedProb = selectedProb.add(epsilon);
return clippedProb.log();
```

**核心改进:**
- 使用`indexSelect()`而非`getValue().get()`
- 使用Variable的`log()`方法而非`Math.log()`
- 数值裁剪也在Variable层面完成

#### 问题2: computeBaselines() & updatePolicy() - Line 299-313, 362-389
**原问题:**
```java
// 提取baseline数值
float baseline = baselineValue.getValue().getNumber().floatValue();

// 手动创建advantage Variable
Variable advantageVar = new Variable(NdArray.of(-advantage));
```

**修复方案:**
```java
// 新增computeBaselinesVariable()返回Variable列表
private List<Variable> computeBaselinesVariable() {
    // 保持Variable类型，不提取数值
    baselines.add(baselineValue);
}

// updatePolicyVariable()使用Variable计算advantage
Variable advantageVar = returnVar;
if (baselines != null) {
    advantageVar = returnVar.sub(baselines.get(i));
}
Variable negAdvantage = advantageVar.mul(new Variable(NdArray.of(-1.0f)));
```

**核心改进:**
- 新增`computeBaselinesVariable()`保持Variable类型
- 在Variable层面计算advantage，使用`sub()`而非手动减法
- 优势函数的符号翻转也在Variable层面完成

### 3. BanditAgent.java

#### 问题: learn() - Line 80
**原问题:**
```java
int actionIndex = (int) experience.getAction().getValue().get(0);
```

**修复方案:**
```java
// 使用getNumber()方法替代get()
int actionIndex = experience.getAction().getValue().getNumber().intValue();
```

**说明:** 虽然多臂老虎机不需要梯度传播，但统一使用`getNumber()`更规范。

### 4. MultiArmedBanditEnvironment.java

#### 修复位置: Line 117, Line 176
**修复内容:**
```java
// 使用getNumber()避免计算图断裂
int armIndex = action.getValue().getNumber().intValue();
```

### 5. EpsilonGreedyPolicy.java

#### 修复位置: Line 91
**修复内容:**
```java
// 添加注释说明使用场景
NdArray probArray = probs.getValue();
return probArray.get(0, actionIndex);
```

## 修复原则总结

### 1. 需要梯度传播的场景
在训练过程中需要梯度回传的操作，必须保持计算图连通性：

✅ **正确做法:**
```java
// 使用Variable层面的操作
Variable result = var1.add(var2);
Variable selected = tensor.indexSelect(dim, indexVar);
Variable maxVal = tensor.max(dim, keepDim);
Variable logVal = prob.log();
```

❌ **错误做法:**
```java
// 直接提取数值再构造Variable
float value = var.getValue().get(index);
Variable result = new Variable(NdArray.of(value));
```

### 2. 仅用于输出显示的场景
当仅需要数值用于日志输出、统计等非训练场景：

✅ **推荐做法:**
```java
float value = var.getValue().getNumber().floatValue();
```

⚠️ **可接受但不推荐:**
```java
float value = var.getValue().get(index);  // 仅在确定是标量时
```

### 3. 关键操作替换对照表

| 原操作 | 替换为 | 适用场景 |
|--------|--------|----------|
| `getValue().get(index)` | `indexSelect(dim, indexVar)` | 需要保持计算图的索引操作 |
| `Math.log(value)` | `var.log()` | 需要梯度的对数运算 |
| `value1 + value2` | `var1.add(var2)` | 需要梯度的加法 |
| `value1 * value2` | `var1.mul(var2)` | 需要梯度的乘法 |
| `Math.max(values)` | `var.max(dim, keepDim)` | 需要梯度的最大值 |
| `new Variable(NdArray.of(scalar))` | 仅在必要时使用 | 引入常数到计算图 |

## 验证建议

1. **梯度检查:** 在训练循环中添加梯度打印，确认关键参数的梯度不为null
2. **反向传播测试:** 对修复后的方法进行单元测试，验证backward()能正常执行
3. **数值稳定性:** 关注log、div等操作的数值稳定性，确保添加必要的epsilon
4. **性能对比:** 对比修复前后的训练收敛性和最终性能

## 影响评估

### 预期改进
- ✅ DQN算法能够正确更新Q网络参数
- ✅ REINFORCE算法的策略梯度能够正确计算
- ✅ 基线网络能够正确学习价值函数
- ✅ 整体训练收敛性提升

### 潜在风险
- ⚠️ Variable层面操作可能增加计算开销
- ⚠️ 某些边缘情况可能需要额外测试

## 相关记忆

根据项目记忆系统，类似问题的最佳实践：
1. **计算图完整性:** 必须在Variable层面完成所有需要梯度的操作
2. **indexSelect使用:** 替代slice等可能断裂计算图的操作
3. **避免数值提取:** 在训练流程中尽量避免getValue().get()

## 后续建议

1. 在代码规范中明确禁止在训练流程中使用`getValue().get()`
2. 为Variable类添加更多便捷操作方法（如argmax、gather等）
3. 在CI/CD中添加计算图连通性检查
4. 编写计算图可视化工具辅助调试

---
**修复完成，建议进行完整的训练测试验证修复效果。**
