package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray广播机制测试
 * 
 * 测试广播机制的各种场景，包括：
 * - broadcastTo广播操作
 * - 不同维度的广播
 * - 广播规则验证
 * - 边界情况
 *
 * @author TinyAI
 */
public class NdArrayBroadcastTest {

    private static final float DELTA = 1e-6f;

    // =============================================================================
    // 基本广播测试
    // =============================================================================

    @Test
    public void testBroadcastTo2D() {
        // 测试简单的2D广播
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        NdArray result = input.broadcastTo(Shape.of(3, 2));
        
        assertEquals(Shape.of(3, 2), result.getShape());
        float[][] expected = {{1f, 2f}, {1f, 2f}, {1f, 2f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testBroadcastColumnVector() {
        // 测试列向量广播
        NdArray input = NdArray.of(new float[][]{{1f}, {2f}, {3f}});
        NdArray result = input.broadcastTo(Shape.of(3, 4));
        
        assertEquals(Shape.of(3, 4), result.getShape());
        float[][] expected = {
            {1f, 1f, 1f, 1f},
            {2f, 2f, 2f, 2f},
            {3f, 3f, 3f, 3f}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testBroadcastRowVector() {
        // 测试行向量广播
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray result = input.broadcastTo(Shape.of(4, 3));
        
        assertEquals(Shape.of(4, 3), result.getShape());
        float[][] expected = {
            {1f, 2f, 3f},
            {1f, 2f, 3f},
            {1f, 2f, 3f},
            {1f, 2f, 3f}
        };
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testBroadcastScalar() {
        // 测试标量广播
        NdArray scalar = NdArray.of(new float[][]{{5f}});
        NdArray result = scalar.broadcastTo(Shape.of(3, 4));
        
        assertEquals(Shape.of(3, 4), result.getShape());
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(5f, matrix[i][j], DELTA);
            }
        }
    }

    @Test
    public void testBroadcastToSameShape() {
        // 广播到相同形状应该返回副本
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.broadcastTo(Shape.of(2, 2));
        
        assertArrayEquals(input.getMatrix(), result.getMatrix());
    }

    // =============================================================================
    // 3D广播测试
    // =============================================================================

    @Test
    public void testBroadcast2DTo3D() {
        // 测试2D广播到3D
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        NdArray result = input.broadcastTo(Shape.of(2, 3, 2));
        
        assertEquals(Shape.of(2, 3, 2), result.getShape());
        
        // 验证广播结果
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, result.get(i, j, 0), DELTA);
                assertEquals(2f, result.get(i, j, 1), DELTA);
            }
        }
    }

    @Test
    public void testBroadcast3DFirstDim() {
        // 测试3D第一维广播
        float[][][] data = {{{1f, 2f}, {3f, 4f}}};
        NdArray input = NdArray.of(data);
        NdArray result = input.broadcastTo(Shape.of(3, 2, 2));
        
        assertEquals(Shape.of(3, 2, 2), result.getShape());
        
        // 每个批次应该是相同的
        for (int i = 0; i < 3; i++) {
            assertEquals(1f, result.get(i, 0, 0), DELTA);
            assertEquals(2f, result.get(i, 0, 1), DELTA);
            assertEquals(3f, result.get(i, 1, 0), DELTA);
            assertEquals(4f, result.get(i, 1, 1), DELTA);
        }
    }

    @Test
    public void testBroadcast3DMiddleDim() {
        // 测试3D中间维度广播
        float[][][] data = {{{1f, 2f}}};
        NdArray input = NdArray.of(data);  // Shape: [1,1,2]
        NdArray result = input.broadcastTo(Shape.of(1, 3, 2));
        
        assertEquals(Shape.of(1, 3, 2), result.getShape());
        
        for (int j = 0; j < 3; j++) {
            assertEquals(1f, result.get(0, j, 0), DELTA);
            assertEquals(2f, result.get(0, j, 1), DELTA);
        }
    }

    // =============================================================================
    // 广播规则验证
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastIncompatibleShape() {
        // 不兼容的形状应该抛出异常
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        input.broadcastTo(Shape.of(3, 3));  // [2,2] 不能广播到 [3,3]
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastToSmallerShape() {
        // 不能广播到更小的形状
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        input.broadcastTo(Shape.of(2, 2));
    }

    @Test
    public void testBroadcastOneDimension() {
        // 测试单维度为1的广播
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});  // Shape: [1,3]
        NdArray result = input.broadcastTo(Shape.of(5, 3));
        
        assertEquals(Shape.of(5, 3), result.getShape());
        for (int i = 0; i < 5; i++) {
            assertEquals(1f, result.get(i, 0), DELTA);
            assertEquals(2f, result.get(i, 1), DELTA);
            assertEquals(3f, result.get(i, 2), DELTA);
        }
    }

    // =============================================================================
    // 广播在运算中的应用
    // =============================================================================

    @Test
    public void testBroadcastInAddition() {
        // 测试广播在加法中的应用
        NdArray matrix = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray vector = NdArray.of(new float[][]{{10f, 20f, 30f}});
        
        NdArray broadcasted = vector.broadcastTo(matrix.getShape());
        NdArray result = matrix.add(broadcasted);
        
        float[][] expected = {{11f, 22f, 33f}, {14f, 25f, 36f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testBroadcastInMultiplication() {
        // 测试广播在乘法中的应用
        NdArray matrix = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}, {5f, 6f}});
        NdArray vector = NdArray.of(new float[][]{{2f, 3f}});
        
        NdArray broadcasted = vector.broadcastTo(matrix.getShape());
        NdArray result = matrix.mul(broadcasted);
        
        float[][] expected = {{2f, 6f}, {6f, 12f}, {10f, 18f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testBroadcastWithColumnVector() {
        // 测试列向量的广播乘法
        NdArray matrix = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray colVector = NdArray.of(new float[][]{{2f}, {3f}});
        
        NdArray broadcasted = colVector.broadcastTo(matrix.getShape());
        NdArray result = matrix.mul(broadcasted);
        
        float[][] expected = {{2f, 4f, 6f}, {12f, 15f, 18f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    // =============================================================================
    // 复杂广播场景
    // =============================================================================

    @Test
    public void testMultipleDimensionBroadcast() {
        // 测试多维度同时广播
        NdArray input = NdArray.of(new float[][]{{5f}});  // [1,1]
        NdArray result = input.broadcastTo(Shape.of(4, 3));
        
        assertEquals(Shape.of(4, 3), result.getShape());
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(5f, matrix[i][j], DELTA);
            }
        }
    }

    @Test
    public void testBroadcastWithZeros() {
        // 测试零向量的广播
        NdArray zeros = NdArray.of(new float[][]{{0f, 0f, 0f}});
        NdArray result = zeros.broadcastTo(Shape.of(3, 3));
        
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(0f, matrix[i][j], DELTA);
            }
        }
    }

    @Test
    public void testBroadcastLargeArray() {
        // 测试大数组广播
        NdArray small = NdArray.of(new float[][]{{1f}});
        NdArray large = small.broadcastTo(Shape.of(100, 100));
        
        assertEquals(Shape.of(100, 100), large.getShape());
        assertEquals(1f, large.get(0, 0), DELTA);
        assertEquals(1f, large.get(99, 99), DELTA);
        assertEquals(1f, large.get(50, 50), DELTA);
    }

    // =============================================================================
    // 广播与聚合的结合
    // =============================================================================

    @Test
    public void testBroadcastAfterMean() {
        // 测试均值后的广播（常见于归一化）
        NdArray data = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray mean = data.mean(0);  // 按列求平均: [2.5, 3.5, 4.5]
        
        NdArray broadcasted = mean.broadcastTo(data.getShape());
        NdArray centered = data.sub(broadcasted);  // 中心化
        
        // 验证每列的均值接近0
        NdArray newMean = centered.mean(0);
        float[][] meanMatrix = newMean.getMatrix();
        for (int i = 0; i < meanMatrix[0].length; i++) {
            assertEquals(0f, meanMatrix[0][i], 1e-5f);
        }
    }

    @Test
    public void testBroadcastForNormalization() {
        // 测试归一化场景的广播
        NdArray data = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}, {5f, 6f}});
        
        // 计算均值和标准差
        NdArray mean = data.mean(0);
        NdArray variance = data.var(0);
        NdArray std = variance.sqrt();
        
        // 广播并归一化
        NdArray meanBroadcast = mean.broadcastTo(data.getShape());
        NdArray stdBroadcast = std.broadcastTo(data.getShape());
        
        NdArray normalized = data.sub(meanBroadcast).div(stdBroadcast);
        
        // 归一化后的均值应该接近0
        NdArray normalizedMean = normalized.mean(0);
        float[][] meanMatrix = normalizedMean.getMatrix();
        for (int i = 0; i < meanMatrix[0].length; i++) {
            assertEquals(0f, meanMatrix[0][i], 1e-4f);
        }
    }

    // =============================================================================
    // 边界情况测试
    // =============================================================================

    @Test
    public void testBroadcastEmptyDimension() {
        // 测试包含0维度的广播
        NdArray input = NdArray.zeros(Shape.of(0, 3));
        NdArray result = input.broadcastTo(Shape.of(0, 3));
        
        assertEquals(Shape.of(0, 3), result.getShape());
    }

    @Test
    public void testBroadcastSingleElement() {
        // 测试单元素数组广播
        NdArray single = NdArray.of(new float[][]{{7f}});
        NdArray result = single.broadcastTo(Shape.of(2, 3));
        
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(7f, matrix[i][j], DELTA);
            }
        }
    }

    @Test
    public void testBroadcastPreservesValues() {
        // 测试广播不改变原始值
        NdArray original = NdArray.of(new float[][]{{1f, 2f, 3f}});
        float[] originalData = original.getArray().clone();
        
        NdArray broadcasted = original.broadcastTo(Shape.of(5, 3));
        
        // 原始数组不应该被修改
        assertArrayEquals(originalData, original.getArray(), DELTA);
    }

    @Test
    public void testChainedBroadcast() {
        // 测试链式广播
        NdArray start = NdArray.of(new float[][]{{1f}});
        NdArray step1 = start.broadcastTo(Shape.of(1, 3));
        NdArray step2 = step1.broadcastTo(Shape.of(2, 3));
        
        assertEquals(Shape.of(2, 3), step2.getShape());
        float[][] expected = {{1f, 1f, 1f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, step2.getMatrix());
    }

    // =============================================================================
    // 实际应用场景测试
    // =============================================================================

    @Test
    public void testBroadcastForBatchProcessing() {
        // 模拟批处理中的广播：为每个样本添加相同的偏置
        NdArray batch = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        NdArray bias = NdArray.of(new float[][]{{0.1f, 0.2f, 0.3f}});
        
        NdArray biasBroadcast = bias.broadcastTo(batch.getShape());
        NdArray result = batch.add(biasBroadcast);
        
        assertEquals(1.1f, result.get(0, 0), DELTA);
        assertEquals(2.2f, result.get(0, 1), DELTA);
        assertEquals(3.3f, result.get(0, 2), DELTA);
    }

    @Test
    public void testBroadcastForWeightScaling() {
        // 模拟权重缩放场景
        NdArray weights = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}, {5f, 6f}});
        NdArray scale = NdArray.of(new float[][]{{0.5f, 2f}});
        
        NdArray scaleBroadcast = scale.broadcastTo(weights.getShape());
        NdArray scaled = weights.mul(scaleBroadcast);
        
        float[][] expected = {{0.5f, 4f}, {1.5f, 8f}, {2.5f, 12f}};
        assertArrayEquals(expected, scaled.getMatrix());
    }
}
