package io.leavesfly.tinyai.ndarr.integration;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray集成测试
 * 
 * 测试多个功能组合使用的场景，包括：
 * - 深度学习工作流
 * - 复杂数据处理流程
 * - 实际应用场景
 *
 * @author TinyAI
 */
public class NdArrayIntegrationTest {

    private static final float DELTA = 1e-5f;

    // =============================================================================
    // 神经网络前向传播场景
    // =============================================================================

    @Test
    public void testSimpleNeuralNetworkForward() {
        // 模拟简单的单层神经网络前向传播
        // 输入: batch_size=2, features=3
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });
        
        // 权重: features=3, hidden=2
        NdArray weights = NdArray.of(new float[][]{
            {0.1f, 0.2f},
            {0.3f, 0.4f},
            {0.5f, 0.6f}
        });
        
        // 偏置: hidden=2
        NdArray bias = NdArray.of(new float[][]{{0.1f, 0.2f}});
        
        // 前向传播: output = input.dot(weights) + bias
        NdArray output = input.dot(weights);
        NdArray broadcastBias = bias.broadcastTo(output.getShape());
        output = output.add(broadcastBias);
        
        // 激活函数: sigmoid
        output = output.sigmoid();
        
        // 验证输出形状
        assertEquals(Shape.of(2, 2), output.getShape());
        
        // 验证输出值在[0,1]之间
        for (float val : output.getArray()) {
            assertTrue(val >= 0f && val <= 1f);
        }
    }

    @Test
    public void testMultiLayerForwardPropagation() {
        // 多层神经网络前向传播
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        // 第一层: 3 -> 4
        NdArray w1 = NdArray.likeRandomN(Shape.of(3, 4));
        NdArray b1 = NdArray.zeros(Shape.of(1, 4));
        NdArray h1 = input.dot(w1).add(b1).tanh();
        
        // 第二层: 4 -> 2
        NdArray w2 = NdArray.likeRandomN(Shape.of(4, 2));
        NdArray b2 = NdArray.zeros(Shape.of(1, 2));
        NdArray output = h1.dot(w2).add(b2).softMax();
        
        // 验证
        assertEquals(Shape.of(1, 2), output.getShape());
        float sum = output.get(0, 0) + output.get(0, 1);
        assertEquals(1.0f, sum, DELTA);
    }

    // =============================================================================
    // 批量归一化场景
    // =============================================================================

    @Test
    public void testBatchNormalization() {
        // 批量归一化流程
        NdArray data = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        
        // 计算均值（按列）
        NdArray mean = data.mean(0);
        
        // 中心化
        NdArray centered = data.sub(mean.broadcastTo(data.getShape()));
        
        // 计算方差
        NdArray variance = centered.square().mean(0);
        
        // 标准化
        NdArray std = variance.sqrt().add(NdArray.like(variance.getShape(), 1e-5f));
        NdArray normalized = centered.div(std.broadcastTo(data.getShape()));
        
        // 验证均值接近0
        NdArray newMean = normalized.mean(0);
        float[][] meanMatrix = newMean.getMatrix();
        for (int i = 0; i < 3; i++) {
            assertEquals(0f, meanMatrix[0][i], 0.1f);
        }
    }

    // =============================================================================
    // Dropout模拟场景
    // =============================================================================

    @Test
    public void testDropoutSimulation() {
        // 模拟Dropout（使用mask）
        NdArray input = NdArray.ones(Shape.of(5, 5));
        
        // 创建随机mask（模拟dropout率0.5）
        NdArray mask = NdArray.likeRandom(0f, 1f, Shape.of(5, 5), 42);
        mask = mask.mask(0.5f); // >0.5为1，否则为0
        
        // 应用mask
        NdArray dropped = input.mul(mask);
        
        // 验证部分元素被设为0
        int zeroCount = 0;
        for (float val : dropped.getArray()) {
            if (val == 0f) zeroCount++;
        }
        assertTrue(zeroCount > 0);
    }

    // =============================================================================
    // 损失函数计算场景
    // =============================================================================

    @Test
    public void testCrossEntropyLoss() {
        // 交叉熵损失计算
        NdArray predictions = NdArray.of(new float[][]{
            {0.7f, 0.2f, 0.1f},
            {0.1f, 0.8f, 0.1f}
        });
        
        NdArray targets = NdArray.of(new float[][]{
            {1f, 0f, 0f},
            {0f, 1f, 0f}
        });
        
        // loss = -sum(targets * log(predictions))
        NdArray logPreds = predictions.log();
        NdArray loss = targets.mul(logPreds).neg().sum();
        
        // 验证损失为正数
        assertTrue(loss.getNumber().floatValue() > 0);
    }

    @Test
    public void testMeanSquaredError() {
        // 均方误差损失
        NdArray predictions = NdArray.of(new float[][]{{1.5f, 2.5f, 3.5f}});
        NdArray targets = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        // MSE = mean((predictions - targets)^2)
        NdArray diff = predictions.sub(targets);
        NdArray squared = diff.square();
        NdArray mse = squared.mean(1);
        
        // 验证MSE
        assertEquals(0.25f, mse.getNumber().floatValue(), DELTA);
    }

    // =============================================================================
    // 卷积操作模拟场景
    // =============================================================================

    @Test
    public void testConvolutionLikeOperation() {
        // 模拟简单的卷积操作（使用矩阵乘法）
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f, 4f},
            {5f, 6f, 7f, 8f},
            {9f, 10f, 11f, 12f},
            {13f, 14f, 15f, 16f}
        });
        
        // 提取3x3的patch
        NdArray patch = input.subNdArray(0, 3, 0, 3);
        assertEquals(Shape.of(3, 3), patch.getShape());
        
        // 应用简单的滤波器（求和）
        NdArray filtered = patch.sum();
        assertTrue(filtered.getNumber().floatValue() > 0);
    }

    // =============================================================================
    // 注意力机制模拟场景
    // =============================================================================

    @Test
    public void testAttentionMechanism() {
        // 模拟简化的注意力计算
        // Query, Key, Value
        NdArray query = NdArray.of(new float[][]{{1f, 2f}});
        NdArray key = NdArray.of(new float[][]{
            {1f, 0f},
            {0f, 1f}
        });
        NdArray value = NdArray.of(new float[][]{
            {1f, 2f},
            {3f, 4f}
        });
        
        // Attention scores = query.dot(key.T)
        NdArray scores = query.dot(key.transpose());
        
        // Attention weights = softmax(scores)
        NdArray weights = scores.softMax();
        
        // Output = weights.dot(value)
        NdArray output = weights.dot(value);
        
        assertEquals(Shape.of(1, 2), output.getShape());
    }

    // =============================================================================
    // 梯度下降优化场景
    // =============================================================================

    @Test
    public void testGradientDescentStep() {
        // 模拟简单的梯度下降步骤
        NdArray weights = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray gradients = NdArray.of(new float[][]{{0.1f, 0.2f, 0.3f}});
        float learningRate = 0.01f;
        
        // 更新: weights = weights - lr * gradients
        NdArray lrGrad = gradients.mulNum(learningRate);
        NdArray newWeights = weights.sub(lrGrad);
        
        // 验证权重已更新
        assertTrue(newWeights.get(0, 0) < weights.get(0, 0));
        assertTrue(newWeights.get(0, 1) < weights.get(0, 1));
        assertTrue(newWeights.get(0, 2) < weights.get(0, 2));
    }

    @Test
    public void testMomentumOptimization() {
        // 动量优化
        NdArray weights = NdArray.of(new float[][]{{1f, 2f}});
        NdArray velocity = NdArray.zeros(Shape.of(1, 2));
        NdArray gradients = NdArray.of(new float[][]{{0.1f, 0.2f}});
        
        float momentum = 0.9f;
        float learningRate = 0.01f;
        
        // velocity = momentum * velocity + gradients
        velocity = velocity.mulNum(momentum).add(gradients);
        
        // weights = weights - lr * velocity
        NdArray newWeights = weights.sub(velocity.mulNum(learningRate));
        
        assertNotNull(newWeights);
        assertEquals(Shape.of(1, 2), newWeights.getShape());
    }

    // =============================================================================
    // 数据增强场景
    // =============================================================================

    @Test
    public void testDataAugmentation() {
        // 数据增强：归一化 + 噪声
        NdArray data = NdArray.of(new float[][]{
            {10f, 20f, 30f},
            {40f, 50f, 60f}
        });
        
        // 归一化到[0,1]
        float min = 10f;
        float max = 60f;
        NdArray normalized = data.sub(NdArray.like(data.getShape(), min))
                                 .divNum(max - min);
        
        // 添加小噪声
        NdArray noise = NdArray.likeRandomN(data.getShape(), 123).mulNum(0.01f);
        NdArray augmented = normalized.add(noise);
        
        // 裁剪到[0,1]
        augmented = augmented.clip(0f, 1f);
        
        // 验证所有值在[0,1]之间
        for (float val : augmented.getArray()) {
            assertTrue(val >= 0f && val <= 1f);
        }
    }

    // =============================================================================
    // 特征提取场景
    // =============================================================================

    @Test
    public void testFeatureExtraction() {
        // 模拟特征提取流程
        NdArray rawData = NdArray.of(new float[][]{
            {1f, 2f, 3f, 4f},
            {5f, 6f, 7f, 8f},
            {9f, 10f, 11f, 12f}
        });
        
        // 1. 标准化
        NdArray mean = rawData.mean(0);
        NdArray centered = rawData.sub(mean.broadcastTo(rawData.getShape()));
        
        // 2. 降维（选择前两列）
        NdArray reduced = rawData.getItem(null, new int[]{0, 1});
        
        // 3. 激活
        NdArray activated = reduced.tanh();
        
        assertEquals(Shape.of(3, 2), activated.getShape());
    }

    // =============================================================================
    // 序列处理场景
    // =============================================================================

    @Test
    public void testSequenceProcessing() {
        // 模拟序列数据处理（如RNN的一个时间步）
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray hiddenState = NdArray.zeros(Shape.of(1, 4));
        
        // 输入权重和隐藏权重
        NdArray wxh = NdArray.likeRandomN(Shape.of(3, 4), 42);
        NdArray whh = NdArray.likeRandomN(Shape.of(4, 4), 43);
        
        // 新隐藏状态: tanh(input.dot(wxh) + hidden.dot(whh))
        NdArray newHidden = input.dot(wxh)
                                 .add(hiddenState.dot(whh))
                                 .tanh();
        
        assertEquals(Shape.of(1, 4), newHidden.getShape());
    }

    // =============================================================================
    // 批处理场景
    // =============================================================================

    @Test
    public void testBatchProcessing() {
        // 批处理多个样本
        int batchSize = 32;
        int features = 10;
        
        NdArray batch = NdArray.likeRandomN(Shape.of(batchSize, features), 100);
        NdArray weights = NdArray.likeRandomN(Shape.of(features, 5), 101);
        
        // 批量前向传播
        NdArray output = batch.dot(weights).sigmoid();
        
        assertEquals(Shape.of(batchSize, 5), output.getShape());
        
        // 批量损失计算
        NdArray targets = NdArray.ones(Shape.of(batchSize, 5));
        NdArray loss = targets.sub(output).square().sum();
        
        assertTrue(loss.getNumber().floatValue() >= 0);
    }

    // =============================================================================
    // 正则化场景
    // =============================================================================

    @Test
    public void testL2Regularization() {
        // L2正则化：loss + lambda * ||weights||^2
        NdArray weights = NdArray.of(new float[][]{{1f, 2f, 3f}});
        float lambda = 0.01f;
        
        // L2 penalty = lambda * sum(weights^2)
        NdArray l2Penalty = weights.square().sum().mulNum(lambda);
        
        assertTrue(l2Penalty.getNumber().floatValue() > 0);
    }

    // =============================================================================
    // 梯度裁剪场景
    // =============================================================================

    @Test
    public void testGradientClipping() {
        // 梯度裁剪
        NdArray gradients = NdArray.of(new float[][]{
            {10f, -20f, 5f},
            {-15f, 25f, 0f}
        });
        
        float clipValue = 10f;
        
        // 裁剪梯度到[-clipValue, clipValue]
        NdArray clipped = gradients.clip(-clipValue, clipValue);
        
        // 验证所有值在范围内
        for (float val : clipped.getArray()) {
            assertTrue(val >= -clipValue && val <= clipValue);
        }
        
        // 验证确实有值被裁剪
        assertNotEquals(gradients.get(0, 1), clipped.get(0, 1), DELTA);
    }

    // =============================================================================
    // 完整的训练迭代场景
    // =============================================================================

    @Test
    public void testCompleteTrainingIteration() {
        // 完整的训练迭代：前向传播 -> 损失计算 -> 反向传播（模拟） -> 参数更新
        
        // 1. 准备数据
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        NdArray target = NdArray.of(new float[][]{{0f, 1f}});
        
        // 2. 前向传播
        NdArray weights = NdArray.likeRandomN(Shape.of(2, 2), 200);
        NdArray output = input.dot(weights).softMax();
        
        // 3. 计算损失（交叉熵）
        NdArray loss = target.mul(output.log()).neg().sum();
        float initialLoss = loss.getNumber().floatValue();
        
        // 4. 模拟梯度（这里简化为随机梯度）
        NdArray gradients = NdArray.likeRandomN(weights.getShape(), 201).mulNum(0.1f);
        
        // 5. 参数更新
        NdArray newWeights = weights.sub(gradients.mulNum(0.01f));
        
        // 6. 再次前向传播
        NdArray newOutput = input.dot(newWeights).softMax();
        NdArray newLoss = target.mul(newOutput.log()).neg().sum();
        
        // 验证：参数已更新
        assertNotEquals(weights.get(0, 0), newWeights.get(0, 0), DELTA);
        
        // 验证：两次输出不同
        assertNotEquals(output.get(0, 0), newOutput.get(0, 0), DELTA);
    }

    // =============================================================================
    // 模型评估场景
    // =============================================================================

    @Test
    public void testModelEvaluation() {
        // 模型评估：准确率计算
        NdArray predictions = NdArray.of(new float[][]{
            {0.1f, 0.9f},
            {0.8f, 0.2f},
            {0.3f, 0.7f},
            {0.6f, 0.4f}
        });
        
        NdArray labels = NdArray.of(new float[][]{
            {0f, 1f},
            {1f, 0f},
            {0f, 1f},
            {1f, 0f}
        });
        
        // 找到最大值的索引（argmax）
        NdArray predClass = predictions.argMax(1);
        NdArray trueClass = labels.argMax(1);
        
        // 计算匹配数（这里简化处理）
        assertEquals(Shape.of(4, 1), predClass.getShape());
        assertEquals(Shape.of(4, 1), trueClass.getShape());
    }
}
