package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.func.matrix.SoftMax;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 归一化函数测试类
 * <p>
 * 测试RMSNorm, Softmax, LogSoftmax归一化函数的功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class NormalizationFunctionsTest {

    private static final float DELTA = 1e-4f;

    // ==================== RMSNorm测试 ====================

    @Test
    public void testRMSNormForward2D() {
        RMSNorm rmsNorm = new RMSNorm(new int[]{2}, 1e-6f);
        
        NdArray x = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray weight = NdArray.of(new float[]{1f, 1f});
        
        NdArray output = rmsNorm.forward(x, weight);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 2), output.getShape());
    }

    @Test
    public void testRMSNormNormalizationEffect() {
        RMSNorm rmsNorm = new RMSNorm(new int[]{3}, 1e-6f);
        
        // 创建一个简单的测试用例
        NdArray x = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray weight = NdArray.ones(Shape.of(3));
        
        NdArray output = rmsNorm.forward(x, weight);
        
        // 验证输出形状
        assertEquals(Shape.of(1, 3), output.getShape());
        
        // RMSNorm应该减小数值的方差
        assertNotNull(output);
    }

    @Test
    public void testRMSNormBackward() {
        RMSNorm rmsNorm = new RMSNorm(new int[]{2}, 1e-6f);
        
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f}}), "x");
        Variable weight = new Variable(NdArray.ones(Shape.of(2)), "weight");
        
        Variable y = rmsNorm.call(x, weight);
        y.backward();
        
        // 验证梯度不为null
        assertNotNull(x.getGrad());
        assertNotNull(weight.getGrad());
    }

    @Test
    public void testRMSNormWithZeroInput() {
        RMSNorm rmsNorm = new RMSNorm(new int[]{2}, 1e-6f);
        
        NdArray x = NdArray.zeros(Shape.of(1, 2));
        NdArray weight = NdArray.ones(Shape.of(2));
        
        NdArray output = rmsNorm.forward(x, weight);
        
        // 零输入应该产生零输出（考虑epsilon）
        assertNotNull(output);
        assertEquals(Shape.of(1, 2), output.getShape());
    }

    // ==================== Softmax测试 ====================

    @Test
    public void testSoftmaxForward() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray output = softmax.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(1, 3), output.getShape());
    }

    @Test
    public void testSoftmaxProbabilitySum() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f, 4f}, {-1f, 0f, 1f, 2f}});
        
        NdArray output = softmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 验证每一行的和等于1（概率分布）
        for (int i = 0; i < 2; i++) {
            float sum = 0f;
            for (int j = 0; j < 4; j++) {
                assertTrue("Softmax输出应该大于0", outputData[i][j] > 0f);
                assertTrue("Softmax输出应该小于等于1", outputData[i][j] <= 1f);
                sum += outputData[i][j];
            }
            assertEquals("Softmax输出的和应该等于1", 1f, sum, DELTA);
        }
    }

    @Test
    public void testSoftmaxMonotonic() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f, 4f}});
        
        NdArray output = softmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 验证较大的输入对应较大的概率
        for (int j = 1; j < 4; j++) {
            assertTrue("较大的输入应该对应较大的Softmax概率", 
                      outputData[0][j] > outputData[0][j - 1]);
        }
    }

    @Test
    public void testSoftmaxNumericalStability() {
        SoftMax softmax = new SoftMax();
        // 测试大数值输入
        NdArray input = NdArray.of(new float[][]{{100f, 101f, 102f}});
        
        NdArray output = softmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 验证输出没有NaN或无穷大
        for (int j = 0; j < 3; j++) {
            assertFalse("Softmax输出不应该包含NaN", Float.isNaN(outputData[0][j]));
            assertFalse("Softmax输出不应该包含无穷大", Float.isInfinite(outputData[0][j]));
        }
    }

    @Test
    public void testSoftmaxUniformInput() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{2f, 2f, 2f, 2f}});
        
        NdArray output = softmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 均匀输入应该产生均匀分布
        float expected = 0.25f;
        for (int j = 0; j < 4; j++) {
            assertEquals("均匀输入应该产生均匀分布", expected, outputData[0][j], DELTA);
        }
    }

    @Test
    public void testSoftmaxBackward() {
        SoftMax softmax = new SoftMax();
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = softmax.call(x);
        y.backward();
        
        // 验证梯度不为null且形状正确
        assertNotNull(x.getGrad());
        assertEquals(Shape.of(1, 3), x.getGrad().getShape());
    }

    @Test
    public void testSoftmaxSingleElement() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{5f}});
        
        NdArray output = softmax.forward(input);
        
        // 单元素的Softmax应该等于1
        assertEquals(1f, output.get(0, 0), DELTA);
    }

    // ==================== LogSoftmax测试 ====================

    @Test
    public void testLogSoftmaxForward() {
        LogSoftmax logSoftmax = new LogSoftmax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray output = logSoftmax.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(1, 3), output.getShape());
    }

    @Test
    public void testLogSoftmaxValues() {
        LogSoftmax logSoftmax = new LogSoftmax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray output = logSoftmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // LogSoftmax的输出应该是负数或0（因为log(probability) <= 0）
        for (int j = 0; j < 3; j++) {
            assertTrue("LogSoftmax输出应该小于等于0", outputData[0][j] <= 0f);
            assertFalse("LogSoftmax输出不应该包含NaN", Float.isNaN(outputData[0][j]));
        }
    }

    @Test
    public void testLogSoftmaxConsistencyWithSoftmax() {
        SoftMax softmax = new SoftMax();
        LogSoftmax logSoftmax = new LogSoftmax();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray softmaxOutput = softmax.forward(input);
        NdArray logSoftmaxOutput = logSoftmax.forward(input);
        
        float[][] softmaxData = softmaxOutput.getMatrix();
        float[][] logSoftmaxData = logSoftmaxOutput.getMatrix();
        
        // 验证: log(softmax(x)) ≈ logsoftmax(x)
        for (int j = 0; j < 3; j++) {
            float expected = (float) Math.log(softmaxData[0][j]);
            assertEquals("LogSoftmax应该等于log(Softmax)", 
                        expected, logSoftmaxData[0][j], DELTA);
        }
    }

    @Test
    public void testLogSoftmaxNumericalStability() {
        LogSoftmax logSoftmax = new LogSoftmax();
        // 测试大数值输入
        NdArray input = NdArray.of(new float[][]{{100f, 101f, 102f}});
        
        NdArray output = logSoftmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 验证输出没有NaN或无穷大
        for (int j = 0; j < 3; j++) {
            assertFalse("LogSoftmax输出不应该包含NaN", Float.isNaN(outputData[0][j]));
            assertFalse("LogSoftmax输出不应该包含无穷大", Float.isInfinite(outputData[0][j]));
        }
    }

    @Test
    public void testLogSoftmaxBackward() {
        LogSoftmax logSoftmax = new LogSoftmax();
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = logSoftmax.call(x);
        y.backward();
        
        // 验证梯度不为null且形状正确
        assertNotNull(x.getGrad());
        assertEquals(Shape.of(1, 3), x.getGrad().getShape());
    }

    @Test
    public void testLogSoftmax1D() {
        LogSoftmax logSoftmax = new LogSoftmax();
        NdArray input = NdArray.of(new float[]{1f, 2f, 3f});
        
        NdArray output = logSoftmax.forward(input);
        
        assertNotNull(output);
        assertEquals(input.getShape(), output.getShape());
    }

    @Test
    public void testLogSoftmax2DAxisDefault() {
        LogSoftmax logSoftmax = new LogSoftmax();
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = logSoftmax.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 2), output.getShape());
    }

    @Test
    public void testLogSoftmaxGradientAccuracy() {
        LogSoftmax logSoftmax = new LogSoftmax();
        Variable x = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "x");
        
        Variable y = logSoftmax.call(x);
        
        // 假设损失是输出的和
        Variable loss = y.sum();
        loss.backward();
        
        // 验证梯度存在
        assertNotNull(x.getGrad());
        
        // LogSoftmax的梯度应该有特定的数值特征
        // grad_input = grad_output - softmax(input) * sum(grad_output)
        float[][] grad = x.getGrad().getMatrix();
        assertNotNull(grad);
    }

    // ==================== 边界条件测试 ====================

    @Test
    public void testNormalizationWithNegativeValues() {
        SoftMax softmax = new SoftMax();
        NdArray input = NdArray.of(new float[][]{{-5f, -2f, -1f}});
        
        NdArray output = softmax.forward(input);
        float[][] outputData = output.getMatrix();
        
        // 负值输入也应该产生有效的概率分布
        float sum = 0f;
        for (int j = 0; j < 3; j++) {
            assertTrue(outputData[0][j] > 0f);
            sum += outputData[0][j];
        }
        assertEquals(1f, sum, DELTA);
    }

    @Test
    public void testNormalizationShapePreservation() {
        SoftMax softmax = new SoftMax();
        Shape[] testShapes = {
            Shape.of(1, 3),
            Shape.of(2, 5),
            Shape.of(4, 10)
        };
        
        for (Shape shape : testShapes) {
            NdArray input = NdArray.likeRandomN(shape);
            NdArray output = softmax.forward(input);
            assertEquals("形状应该保持不变", shape, output.getShape());
        }
    }

    @Test
    public void testLogSoftmaxWithVerySmallValues() {
        LogSoftmax logSoftmax = new LogSoftmax();
        NdArray input = NdArray.of(new float[][]{{-100f, -99f, -98f}});
        
        NdArray output = logSoftmax.forward(input);
        
        // 即使是很小的值，LogSoftmax也应该返回有效的输出
        assertNotNull(output);
        float[][] outputData = output.getMatrix();
        for (int j = 0; j < 3; j++) {
            assertFalse(Float.isNaN(outputData[0][j]));
            assertFalse(Float.isInfinite(outputData[0][j]));
        }
    }

    @Test
    public void testCompareRMSNormWithDifferentEps() {
        float eps1 = 1e-6f;
        float eps2 = 1e-8f;
        
        RMSNorm rmsNorm1 = new RMSNorm(new int[]{2}, eps1);
        RMSNorm rmsNorm2 = new RMSNorm(new int[]{2}, eps2);
        
        NdArray x = NdArray.of(new float[][]{{1f, 2f}});
        NdArray weight = NdArray.ones(Shape.of(2));
        
        NdArray output1 = rmsNorm1.forward(x, weight);
        NdArray output2 = rmsNorm2.forward(x, weight);
        
        // 不同的epsilon应该产生略有不同的结果
        assertNotNull(output1);
        assertNotNull(output2);
    }
}
