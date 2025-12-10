# Variable 算子补充技术方案

## 一、现状分析

### 1.1 已实现的算子

#### 基础运算（已完整）
- ✅ `add`, `sub`, `mul`, `div`, `neg` - 四则运算
- ✅ `squ`, `pow`, `exp`, `sin`, `cos`, `log`, `tanh`, `sigmoid`, `sqrt` - 基础数学函数

#### 激活函数（已完整）
- ✅ `relu`, `gelu`, `silu`, `leakyRelu`, `elu`, `logSoftmax` - 常用激活函数

#### 统计函数（已完整）
- ✅ `max`, `min`, `mean`, `var`, `sum`, `sumTo` - 统计聚合函数

#### 矩阵操作（部分完整）
- ✅ `matMul` - 矩阵乘法
- ✅ `transpose` - 转置
- ✅ `reshape` - 重塑
- ✅ `broadcastTo` - 广播
- ✅ `linear` - 线性变换
- ✅ `getItem` - 基础索引
- ✅ `concat` - 拼接（仅支持2D）
- ✅ `split` - 分割
- ✅ `gather` - 索引查找（用于Embedding）
- ✅ `unsqueeze` - 增加维度
- ✅ `permute` - 维度重排
- ✅ `topK` - Top-K选择
- ⚠️ `tril` - 下三角矩阵（未完成实现）
- ✅ `maskedFill` - 掩码填充

#### 损失函数（部分完整）
- ✅ `meanSquaredError` - 均方误差
- ✅ `softmaxCrossEntropy` - Softmax交叉熵

### 1.2 缺失的关键算子

基于 PyTorch 的 `torch.Tensor` 和 Transformer/LLM 训练需求，以下算子需要补充：

## 二、需要补充的算子分类

### 2.1 维度操作算子（高优先级）

#### 2.1.1 `squeeze` - 压缩维度
**用途**：移除大小为1的维度
**PyTorch对应**：`torch.squeeze()`
**Transformer应用**：
- 处理注意力权重维度：`[batch, 1, seq_len, seq_len] -> [batch, seq_len, seq_len]`
- 处理单头注意力输出：`[batch, 1, seq_len, head_dim] -> [batch, seq_len, head_dim]`

**实现要点**：
```java
public Variable squeeze() {
    // 移除所有大小为1的维度
}

public Variable squeeze(int dim) {
    // 移除指定维度（如果大小为1）
}
```

#### 2.1.2 `expand` - 扩展维度（不复制数据）
**用途**：将大小为1的维度扩展到指定大小（广播语义）
**PyTorch对应**：`torch.expand()`
**Transformer应用**：
- 扩展掩码维度：`[1, seq_len] -> [batch, seq_len]`
- 扩展位置编码：`[1, seq_len, dim] -> [batch, seq_len, dim]`

**实现要点**：
```java
public Variable expand(Shape targetShape) {
    // 只能扩展大小为1的维度
    // 不复制数据，返回view
}
```

#### 2.1.3 `repeat` - 重复张量
**用途**：沿指定维度重复张量（复制数据）
**PyTorch对应**：`torch.repeat()`
**Transformer应用**：
- Grouped Query Attention (GQA) 中重复KV头
- 扩展位置编码到batch维度

**实现要点**：
```java
public Variable repeat(int... repeats) {
    // repeats[i] 表示第i维重复次数
    // 实际复制数据
}
```

#### 2.1.4 `tile` - 平铺张量
**用途**：类似repeat，但语义更清晰
**PyTorch对应**：`torch.tile()`

### 2.2 索引和切片算子（高优先级）

#### 2.2.1 `scatter` / `scatterAdd` - 分散操作
**用途**：将源张量的值分散到目标张量的指定位置
**PyTorch对应**：`torch.scatter()`, `torch.scatter_add()`
**Transformer应用**：
- Embedding层的梯度回传（已在Gather中部分实现）
- 稀疏注意力机制
- 动态KV-Cache更新

**实现要点**：
```java
public Variable scatter(int dim, Variable index, Variable src) {
    // 将src的值根据index分散到指定维度
}

public Variable scatterAdd(int dim, Variable index, Variable src) {
    // scatter的累加版本
}
```

#### 2.2.2 `indexSelect` - 索引选择
**用途**：沿指定维度选择索引对应的元素
**PyTorch对应**：`torch.index_select()`
**Transformer应用**：
- 从序列中选择特定位置
- 实现因果掩码的增量推理

**实现要点**：
```java
public Variable indexSelect(int dim, Variable index) {
    // 沿dim维度选择index指定的元素
}
```

#### 2.2.3 `advancedIndexing` - 高级索引
**用途**：支持多维索引、布尔索引、整数数组索引
**PyTorch对应**：`tensor[indices]`, `tensor[mask]`
**Transformer应用**：
- 动态序列长度处理
- 掩码选择有效token

**实现要点**：
```java
public Variable index(Variable... indices) {
    // 支持多种索引方式
}
```

#### 2.2.4 `where` - 条件选择
**用途**：根据条件选择元素
**PyTorch对应**：`torch.where()`
**Transformer应用**：
- 掩码条件选择
- 梯度裁剪
- 数值稳定性处理

**实现要点**：
```java
public Variable where(Variable condition, Variable x, Variable y) {
    // condition为true选x，否则选y
}
```

### 2.3 归一化算子（高优先级）

#### 2.3.1 `rmsNorm` - RMS归一化
**用途**：Root Mean Square Layer Normalization（LLaMA、Qwen等使用）
**PyTorch对应**：自定义实现或`torch.nn.RMSNorm`
**Transformer应用**：
- LLaMA、Qwen、DeepSeek等现代LLM的归一化层
- 相比LayerNorm更高效（无需均值计算）

**实现要点**：
```java
public Variable rmsNorm(int[] normalizedShape, float eps) {
    // RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    // 需要支持指定归一化维度
}
```

#### 2.3.2 `layerNorm` - 层归一化（增强版）
**用途**：支持指定归一化维度
**当前状态**：可能已有实现，但需要确认是否支持任意维度
**Transformer应用**：
- Pre-LN和Post-LN Transformer
- 需要支持`[batch, seq_len, hidden_dim]`在最后一维归一化

### 2.4 注意力相关算子（高优先级）

#### 2.4.1 `bmm` - 批量矩阵乘法
**用途**：高效的批量矩阵乘法
**PyTorch对应**：`torch.bmm()`
**Transformer应用**：
- 多头注意力的核心计算：`Q @ K^T` 和 `Attention @ V`
- 比循环调用matMul更高效

**实现要点**：
```java
public Variable bmm(Variable other) {
    // 输入: [batch, n, m] @ [batch, m, p] -> [batch, n, p]
    // 比循环matMul更高效
}
```

#### 2.4.2 `einsum` - 爱因斯坦求和约定
**用途**：灵活的矩阵运算表达式
**PyTorch对应**：`torch.einsum()`
**Transformer应用**：
- 复杂的注意力计算
- 多维度矩阵运算
- 实现各种注意力变体

**实现要点**：
```java
public static Variable einsum(String equation, Variable... inputs) {
    // 例如: "bij,bjk->bik" 表示批量矩阵乘法
    // 实现复杂度较高，但功能强大
}
```

#### 2.4.3 `tril` - 下三角矩阵（完善实现）
**用途**：生成或提取下三角矩阵
**当前状态**：已有框架但未完成实现
**Transformer应用**：
- 因果掩码生成
- 注意力掩码

**实现要点**：
```java
public Variable tril(int k) {
    // 返回下三角矩阵，k控制对角线偏移
    // 需要完成现有未实现的代码
}
```

### 2.5 数值和工具算子（中优先级）

#### 2.5.1 `roll` - 循环移位
**用途**：沿指定维度循环移动元素
**PyTorch对应**：`torch.roll()`
**Transformer应用**：
- RoPE（旋转位置编码）的实现
- 相对位置编码

**实现要点**：
```java
public Variable roll(int shifts, int dim) {
    // 沿dim维度循环移动shifts个位置
}
```

#### 2.5.2 `arange` - 生成序列
**用途**：生成等间隔的整数序列
**PyTorch对应**：`torch.arange()`
**Transformer应用**：
- 位置编码生成
- 索引生成

**实现要点**：
```java
public static Variable arange(int end) {
    // 生成 [0, 1, 2, ..., end-1]
}

public static Variable arange(int start, int end, int step) {
    // 生成 [start, start+step, ..., end)
}
```

#### 2.5.3 `linspace` - 线性间隔序列
**用途**：生成等间隔的浮点数序列
**PyTorch对应**：`torch.linspace()`
**Transformer应用**：
- RoPE频率计算
- 位置编码生成

#### 2.5.4 `onesLike` / `zerosLike` - 创建同形状张量
**用途**：创建与输入同形状的全1/全0张量
**PyTorch对应**：`torch.ones_like()`, `torch.zeros_like()`
**Transformer应用**：
- 掩码初始化
- 注意力权重初始化

**实现要点**：
```java
public Variable onesLike() {
    // 返回与当前Variable同形状的全1张量
}

public Variable zerosLike() {
    // 返回与当前Variable同形状的全0张量
}
```

#### 2.5.5 `detach` - 切断梯度
**用途**：从计算图中分离，停止梯度传播
**PyTorch对应**：`tensor.detach()`
**Transformer应用**：
- 冻结部分参数
- 辅助损失计算
- 对抗训练

**实现要点**：
```java
public Variable detach() {
    // 返回新Variable，requireGrad=false
    // 值相同但不在计算图中
}
```

#### 2.5.6 `clone` - 克隆张量
**用途**：深拷贝张量（包括梯度）
**PyTorch对应**：`tensor.clone()`
**Transformer应用**：
- 保存中间结果
- 避免意外的就地修改

**实现要点**：
```java
public Variable clone() {
    // 深拷贝value和grad
    // 不复制计算图（新Variable是叶子节点）
}
```

#### 2.5.7 `clamp` - 裁剪（增强版）
**用途**：将值限制在指定范围内
**当前状态**：已有`clip`，但可能需要更灵活的版本
**Transformer应用**：
- 梯度裁剪
- 数值稳定性
- 激活值限制

**实现要点**：
```java
public Variable clamp(float min, float max) {
    // 等价于clip，但命名更符合PyTorch习惯
}

public Variable clampMin(float min) {
    // 只限制最小值
}

public Variable clampMax(float max) {
    // 只限制最大值
}
```

### 2.6 损失函数算子（中优先级）

#### 2.6.1 `crossEntropy` - 交叉熵损失（增强版）
**用途**：带标签平滑的交叉熵损失
**当前状态**：已有`softmaxCrossEntropy`，但可能需要更灵活的版本
**Transformer应用**：
- 语言模型训练
- 标签平滑正则化

**实现要点**：
```java
public Variable crossEntropy(Variable target, float labelSmoothing) {
    // 支持标签平滑的交叉熵
    // labelSmoothing=0时等价于标准交叉熵
}
```

#### 2.6.2 `klDiv` - KL散度
**用途**：计算KL散度
**PyTorch对应**：`torch.nn.functional.kl_div()`
**Transformer应用**：
- 知识蒸馏
- 模型压缩
- 正则化

### 2.7 其他实用算子（低优先级）

#### 2.7.1 `flip` - 翻转
**用途**：沿指定维度翻转张量
**PyTorch对应**：`torch.flip()`
**Transformer应用**：
- 双向注意力
- 数据增强

#### 2.7.2 `rot90` - 旋转90度
**用途**：旋转矩阵
**Transformer应用**：
- 特殊的位置编码
- 数据预处理

#### 2.7.3 `meshgrid` - 网格生成
**用途**：生成坐标网格
**Transformer应用**：
- 位置编码生成
- 2D注意力机制

## 三、实现优先级

### 高优先级（必须实现，用于基础Transformer训练）

1. **维度操作**：`squeeze`, `expand`, `repeat`
2. **索引操作**：`scatter`, `indexSelect`, `where`
3. **归一化**：`rmsNorm`（现代LLM必需）
4. **注意力**：`bmm`, `tril`（完善实现）
5. **工具函数**：`detach`, `clone`, `onesLike`, `zerosLike`

### 中优先级（增强功能，提升训练效率）

1. **高级索引**：`advancedIndexing`
2. **数值函数**：`arange`, `linspace`, `roll`
3. **损失函数**：`crossEntropy`（带标签平滑）, `klDiv`
4. **裁剪**：`clamp`（增强版）

### 低优先级（可选，用于特殊场景）

1. **高级运算**：`einsum`（功能强大但实现复杂）
2. **其他工具**：`flip`, `rot90`, `meshgrid`, `tile`

## 四、实现策略

### 4.1 架构设计

所有算子应遵循现有的`Function`接口模式：

```java
public class NewOperator extends Function {
    @Override
    public NdArray forward(NdArray... inputs) {
        // 前向传播实现
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 反向传播实现
    }
    
    @Override
    public int requireInputNum() {
        // 返回所需输入数量
    }
}
```

### 4.2 在Variable中添加方法

```java
public class Variable {
    // 示例：添加squeeze方法
    public Variable squeeze() {
        Function function = new Squeeze();
        return function.call(this);
    }
    
    public Variable squeeze(int dim) {
        Function function = new Squeeze(dim);
        return function.call(this);
    }
}
```

### 4.3 实现注意事项

1. **数值稳定性**：
   - 使用epsilon防止除零
   - 注意浮点数精度问题
   - 对于exp/log等函数，注意溢出处理

2. **内存效率**：
   - 尽量使用view而非copy（如expand）
   - 对于需要copy的操作（如repeat），明确标注
   - 注意梯度计算的内存占用

3. **广播规则**：
   - 遵循NumPy/PyTorch的广播规则
   - 确保形状检查的准确性

4. **梯度计算**：
   - 确保所有算子的反向传播正确
   - 注意多输入算子的梯度分配
   - 对于不可导操作（如索引），返回null梯度

5. **测试覆盖**：
   - 单元测试覆盖前向和反向传播
   - 边界情况测试（空张量、单元素等）
   - 数值精度测试

## 五、具体实现建议

### 5.1 高优先级算子实现顺序

**第一阶段（核心算子）**：
1. `squeeze` - 简单，使用频率高
2. `expand` - 注意view语义
3. `bmm` - 注意力核心，性能关键
4. `tril` - 完善现有实现
5. `detach` - 简单但重要

**第二阶段（索引和条件）**：
1. `scatter` / `scatterAdd` - Embedding梯度必需
2. `indexSelect` - 常用索引操作
3. `where` - 条件选择，用途广泛

**第三阶段（归一化和工具）**：
1. `rmsNorm` - 现代LLM必需
2. `repeat` - 数据复制
3. `onesLike` / `zerosLike` - 工具函数
4. `clone` - 深拷贝

### 5.2 实现示例：squeeze算子

```java
package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 压缩维度算子
 * 移除大小为1的维度
 */
public class Squeeze extends Function {
    
    private final Integer targetDim; // null表示移除所有大小为1的维度
    private Shape inputShape;
    private Shape outputShape;
    
    public Squeeze() {
        this.targetDim = null;
    }
    
    public Squeeze(int dim) {
        this.targetDim = dim;
    }
    
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] dims = inputShape.getShapeDims();
        
        if (targetDim == null) {
            // 移除所有大小为1的维度
            List<Integer> newDims = new ArrayList<>();
            for (int d : dims) {
                if (d != 1) {
                    newDims.add(d);
                }
            }
            if (newDims.isEmpty()) {
                newDims.add(1); // 至少保留一个维度
            }
            outputShape = Shape.of(newDims.stream().mapToInt(i->i).toArray());
        } else {
            // 移除指定维度（如果大小为1）
            int target = targetDim < 0 ? dims.length + targetDim : targetDim;
            if (target < 0 || target >= dims.length) {
                throw new IllegalArgumentException("Dimension out of range");
            }
            if (dims[target] != 1) {
                throw new IllegalArgumentException("Can only squeeze dimension of size 1");
            }
            
            int[] newDims = new int[dims.length - 1];
            for (int i = 0, j = 0; i < dims.length; i++) {
                if (i != target) {
                    newDims[j++] = dims[i];
                }
            }
            outputShape = Shape.of(newDims);
        }
        
        return x.reshape(outputShape);
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 恢复原始形状
        return Collections.singletonList(yGrad.reshape(inputShape));
    }
    
    @Override
    public int requireInputNum() {
        return 1;
    }
}
```

### 5.3 实现示例：bmm算子

```java
package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 批量矩阵乘法
 * 输入: [batch, n, m] @ [batch, m, p] -> [batch, n, p]
 */
public class BMM extends Function {
    
    private int batchSize;
    private int n, m, p;
    
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray a = inputs[0];
        NdArray b = inputs[1];
        
        int[] aShape = a.getShape().getShapeDims();
        int[] bShape = b.getShape().getShapeDims();
        
        if (aShape.length != 3 || bShape.length != 3) {
            throw new IllegalArgumentException("BMM requires 3D tensors");
        }
        
        batchSize = aShape[0];
        n = aShape[1];
        m = aShape[2];
        p = bShape[2];
        
        if (aShape[0] != bShape[0] || aShape[2] != bShape[1]) {
            throw new IllegalArgumentException("Shape mismatch for BMM");
        }
        
        // 实现批量矩阵乘法
        // 可以循环调用matMul，或优化为批量计算
        NdArray result = NdArray.zeros(Shape.of(batchSize, n, p));
        
        for (int i = 0; i < batchSize; i++) {
            NdArray aBatch = a.subNdArray(i, i+1, 0, m).reshape(Shape.of(n, m));
            NdArray bBatch = b.subNdArray(i, i+1, 0, p).reshape(Shape.of(m, p));
            NdArray cBatch = aBatch.matMul(bBatch);
            // 将结果写入result
            result.setItem(new int[]{i}, null, cBatch.getArray());
        }
        
        return result;
    }
    
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度计算
        // dA = dY @ B^T
        // dB = A^T @ dY
        // 需要实现批量转置和批量矩阵乘法
        
        // 简化实现：循环计算每个batch的梯度
        // 实际应该优化为批量计算
        
        return Arrays.asList(
            computeGradA(yGrad, inputs[1]),
            computeGradB(inputs[0], yGrad)
        );
    }
    
    private NdArray computeGradA(NdArray dY, NdArray B) {
        // dA = dY @ B^T for each batch
        // 实现省略...
        return null;
    }
    
    private NdArray computeGradB(NdArray A, NdArray dY) {
        // dB = A^T @ dY for each batch
        // 实现省略...
        return null;
    }
    
    @Override
    public int requireInputNum() {
        return 2;
    }
}
```

## 六、测试策略

### 6.1 单元测试

每个算子需要包含：
1. **基础功能测试**：正常输入输出
2. **边界情况测试**：空张量、单元素、极端值
3. **形状测试**：各种形状组合
4. **梯度测试**：反向传播正确性
5. **数值精度测试**：与PyTorch结果对比

### 6.2 集成测试

1. **Transformer训练测试**：使用完整Transformer模型测试
2. **性能测试**：与PyTorch性能对比
3. **内存测试**：内存占用分析

## 七、参考实现

### 7.1 PyTorch参考

- PyTorch官方文档：https://pytorch.org/docs/stable/tensors.html
- PyTorch源码：https://github.com/pytorch/pytorch

### 7.2 相关论文

- Transformer: "Attention Is All You Need"
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- RMSNorm: "Root Mean Square Layer Normalization"

## 八、总结

本方案基于对现有Variable实现的分析，参考PyTorch的tensor API，针对Transformer/LLM训练需求，提出了需要补充的算子清单。

**核心建议**：
1. 优先实现高优先级算子，确保基础Transformer训练能力
2. 遵循现有架构模式，保持代码一致性
3. 注重数值稳定性和性能优化
4. 完善测试覆盖，确保正确性

**预期效果**：
完成高优先级算子后，TinyAI将能够支持完整的Transformer/LLM模型训练，包括：
- 标准Transformer（BERT、GPT等）
- 现代LLM（LLaMA、Qwen、DeepSeek等）
- 各种注意力变体（GQA、MQA等）
- 位置编码（绝对、相对、RoPE等）

