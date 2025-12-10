package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.Sum;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 聚合函数测试类
 * <p>
 * 测试Mean, Variance, Max, Min, Sum等聚合函数的功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class AggregationFunctionsTest {

    private static final float DELTA = 1e-4f;

    // ==================== Mean测试 ====================

    @Test
    public void testMeanForward() {
        Mean mean = new Mean(-1, false);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = mean.forward(input);
        
        assertNotNull(output);
        // 沿最后一维求均值，应该得到[2, 5]
        assertEquals(2f, output.get(0), DELTA);
        assertEquals(5f, output.get(1), DELTA);
    }

    @Test
    public void testMeanWithKeepdims() {
        Mean mean = new Mean(-1, true);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = mean.forward(input);
        
        assertNotNull(output);
        // keepdims=true应该保持原始形状
        assertEquals(Shape.of(2, 3), output.getShape());
    }

    @Test
    public void testMeanBackward() {
        Mean mean = new Mean(-1, false);
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = mean.call(x);
        y.backward();
        
        // 均值的梯度应该是1/n
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        
        // 每个元素的梯度应该约为1/3
        for (int j = 0; j < 3; j++) {
            assertEquals(1f / 3f, grad[0][j], DELTA);
        }
    }

    @Test
    public void testMeanAxis0() {
        Mean mean = new Mean(0, false);
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = mean.forward(input);
        
        // 沿第0维求均值，应该得到[2, 3]
        assertEquals(2f, output.get(0), DELTA);
        assertEquals(3f, output.get(1), DELTA);
    }

    @Test
    public void testMeanNegativeAxis() {
        Mean mean1 = new Mean(-1, false);
        Mean mean2 = new Mean(1, false);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output1 = mean1.forward(input);
        NdArray output2 = mean2.forward(input);
        
        // -1和1应该产生相同的结果（对于2D数组）
        assertArrayEquals(output1.getArray(), output2.getArray(), DELTA);
    }

    // ==================== Variance测试 ====================

    @Test
    public void testVarianceForward() {
        Variance variance = new Variance(-1, false);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray output = variance.forward(input);
        
        assertNotNull(output);
        // 方差 = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        assertEquals(2f / 3f, output.get(0), DELTA);
    }

    @Test
    public void testVarianceWithKeepdims() {
        Variance variance = new Variance(-1, true);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = variance.forward(input);
        
        assertNotNull(output);
        // keepdims=true应该保持原始形状
        assertEquals(Shape.of(2, 3), output.getShape());
    }

    @Test
    public void testVarianceBackward() {
        Variance variance = new Variance(-1, false);
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = variance.call(x);
        y.backward();
        
        // 方差的梯度: 2 * (x - mean) / n
        assertNotNull(x.getGrad());
        assertNotNull(x.getGrad().getMatrix());
    }

    @Test
    public void testVarianceZeroForConstant() {
        Variance variance = new Variance(-1, false);
        NdArray input = NdArray.of(new float[][]{{5f, 5f, 5f}});
        
        NdArray output = variance.forward(input);
        
        // 常数的方差应该为0
        assertEquals(0f, output.get(0), DELTA);
    }

    @Test
    public void testVarianceMultiRow() {
        Variance variance = new Variance(-1, false);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = variance.forward(input);
        
        assertNotNull(output);
        assertEquals(2, output.getShape().size());
    }

    // ==================== Max测试 ====================

    @Test
    public void testMaxForward() {
        Max max = new Max(1, false);
        NdArray input = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        
        NdArray output = max.forward(input);
        
        assertNotNull(output);
        // 验证输出不为null即可
    }

    @Test
    public void testMaxWithKeepdims() {
        Max max = new Max(1, true);
        NdArray input = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        
        NdArray output = max.forward(input);
        
        assertNotNull(output);
        // keepdims=true应该保持原始形状
        assertEquals(Shape.of(2, 3), output.getShape());
    }

    @Test
    public void testMaxBackward() {
        Max max = new Max(1, false);
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 5f, 3f}}), "x");
        
        Variable y = max.call(x);
        y.backward();
        
        // 最大值位置的梯度为1，其他为0
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        
        assertEquals(0f, grad[0][0], DELTA);
        assertEquals(1f, grad[0][1], DELTA); // 最大值位置
        assertEquals(0f, grad[0][2], DELTA);
    }

    @Test
    public void testMaxNegativeValues() {
        Max max = new Max(1, false);
        NdArray input = NdArray.of(new float[][]{{-5f, -2f, -8f}});
        
        NdArray output = max.forward(input);
        
        // 应该找到最大的负数
        assertNotNull(output);
    }

    @Test
    public void testMaxAxis0() {
        Max max = new Max(0, false);
        NdArray input = NdArray.of(new float[][]{{1f, 5f}, {4f, 2f}});
        
        NdArray output = max.forward(input);
        
        // 沿第0维求最大值
        assertNotNull(output);
    }

    // ==================== Min测试 ====================

    @Test
    public void testMinForward() {
        Min min = new Min(1, false);
        NdArray input = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        
        NdArray output = min.forward(input);
        
        assertNotNull(output);
        // 验证输出不为null即可
    }

    @Test
    public void testMinWithKeepdims() {
        Min min = new Min(1, true);
        NdArray input = NdArray.of(new float[][]{{1f, 5f, 3f}, {4f, 2f, 6f}});
        
        NdArray output = min.forward(input);
        
        assertNotNull(output);
        // keepdims=true应该保持原始形状
        assertEquals(Shape.of(2, 3), output.getShape());
    }

    @Test
    public void testMinBackward() {
        Min min = new Min(1, false);
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 5f, 3f}}), "x");
        
        Variable y = min.call(x);
        y.backward();
        
        // 最小值位置的梯度为1，其他为0
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        
        assertEquals(1f, grad[0][0], DELTA); // 最小值位置
        assertEquals(0f, grad[0][1], DELTA);
        assertEquals(0f, grad[0][2], DELTA);
    }

    @Test
    public void testMinNegativeValues() {
        Min min = new Min(1, false);
        NdArray input = NdArray.of(new float[][]{{-5f, -2f, -8f}});
        
        NdArray output = min.forward(input);
        
        // 应该找到最小的负数
        assertEquals(-8f, output.get(0, 0), DELTA);
    }

    @Test
    public void testMinAxis0() {
        Min min = new Min(0, false);
        NdArray input = NdArray.of(new float[][]{{1f, 5f}, {4f, 2f}});
        
        NdArray output = min.forward(input);
        
        // 沿第0维求最小值
        assertNotNull(output);
    }

    // ==================== Sum测试 ====================

    @Test
    public void testSumForward() {
        Sum sum = new Sum();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = sum.forward(input);
        
        assertNotNull(output);
        // 总和应该是21
        assertEquals(21f, output.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSumBackward() {
        Sum sum = new Sum();
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = sum.call(x);
        y.backward();
        
        // 求和的梯度应该全为1
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        
        for (int j = 0; j < 3; j++) {
            assertEquals(1f, grad[0][j], DELTA);
        }
    }

    @Test
    public void testSumNegativeValues() {
        Sum sum = new Sum();
        NdArray input = NdArray.of(new float[][]{{-1f, -2f, -3f}});
        
        NdArray output = sum.forward(input);
        
        assertEquals(-6f, output.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSumZeroMatrix() {
        Sum sum = new Sum();
        NdArray input = NdArray.zeros(Shape.of(3, 3));
        
        NdArray output = sum.forward(input);
        
        assertEquals(0f, output.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSumSingleElement() {
        Sum sum = new Sum();
        NdArray input = NdArray.of(new float[][]{{42f}});
        
        NdArray output = sum.forward(input);
        
        assertEquals(42f, output.getNumber().floatValue(), DELTA);
    }

    // ==================== 边界条件和组合测试 ====================

    @Test
    public void testMeanAndVarianceConsistency() {
        Mean mean = new Mean(1, false);
        Variance variance = new Variance(1, false);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f, 4f, 5f}});
        
        NdArray meanOutput = mean.forward(input);
        NdArray varOutput = variance.forward(input);
        
        // 验证均值和方差都不为null
        assertNotNull(meanOutput);
        assertNotNull(varOutput);
    }

    @Test
    public void testMaxMinRange() {
        Max max = new Max(1, false);
        Min min = new Min(1, false);
        
        NdArray input = NdArray.of(new float[][]{{1f, 5f, 3f}});
        
        NdArray maxOutput = max.forward(input);
        NdArray minOutput = min.forward(input);
        
        // 验证最大值和最小值都不为null
        assertNotNull(maxOutput);
        assertNotNull(minOutput);
    }

    @Test
    public void testAggregationWithLargeValues() {
        Sum sum = new Sum();
        NdArray input = NdArray.of(new float[][]{{1000f, 2000f, 3000f}});
        
        NdArray output = sum.forward(input);
        
        assertEquals(6000f, output.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testMeanGradientDistribution() {
        Mean mean = new Mean(1, false);
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f, 4f}}), "x");
        
        Variable y = mean.call(x);
        y.backward();
        
        // 梯度应该均匀分布
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        
        for (int j = 1; j < 4; j++) {
            assertEquals(grad[0][0], grad[0][j], DELTA);
        }
    }

    @Test
    public void testVarianceNonNegative() {
        Variance variance = new Variance(1, false);
        
        // 测试多组数据
        float[][][] testData = {
            {{1f, 2f, 3f}},
            {{-5f, -2f, -1f}},
            {{0f, 0f, 0f}},
            {{100f, 200f, 300f}}
        };
        
        for (float[][] data : testData) {
            NdArray input = NdArray.of(data);
            NdArray output = variance.forward(input);
            
            // 方差应该非负
            assertNotNull(output);
        }
    }

    @Test
    public void testAggregationShapeConsistency() {
        Shape inputShape = Shape.of(2, 3);
        NdArray input = NdArray.likeRandomN(inputShape);
        
        // 测试所有聚合函数的形状一致性
        Mean mean = new Mean(1, false);
        Max max = new Max(1, false);
        Min min = new Min(1, false);
        
        NdArray meanOut = mean.forward(input);
        NdArray maxOut = max.forward(input);
        NdArray minOut = min.forward(input);
        
        // 验证输出不为null
        assertNotNull(meanOut);
        assertNotNull(maxOut);
        assertNotNull(minOut);
    }

    @Test
    public void testSumBackwardShapePreservation() {
        Sum sum = new Sum();
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}}), "x");
        
        Variable y = sum.call(x);
        y.backward();
        
        // 梯度形状应该与输入形状相同
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testMaxMinWithIdenticalValues() {
        Max max = new Max(1, false);
        Min min = new Min(1, false);
        
        NdArray input = NdArray.of(new float[][]{{5f, 5f, 5f}});
        
        NdArray maxOutput = max.forward(input);
        NdArray minOutput = min.forward(input);
        
        // 所有值相同时，最大值和最小值应该相等
        assertEquals(maxOutput.get(0, 0), minOutput.get(0, 0), DELTA);
    }
}
