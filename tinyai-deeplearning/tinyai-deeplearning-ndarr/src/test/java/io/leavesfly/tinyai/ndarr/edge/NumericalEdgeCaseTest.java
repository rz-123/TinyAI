package io.leavesfly.tinyai.ndarr.edge;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 数值精度和特殊值测试
 * 
 * 测试浮点数的特殊情况，包括：
 * - 极大和极小数值
 * - NaN和Infinity
 * - 精度边界
 * - 数值溢出
 *
 * @author TinyAI
 */
public class NumericalEdgeCaseTest {

    private static final float DELTA = 1e-6f;
    private static final float RELAXED_DELTA = 1e-4f;

    // =============================================================================
    // 极值测试
    // =============================================================================

    @Test
    public void testVeryLargeNumbers() {
        // 测试非常大的数值
        float largeValue = Float.MAX_VALUE / 10;
        NdArray array = NdArray.like(Shape.of(2, 2), largeValue);
        
        assertEquals(largeValue, array.get(0, 0), largeValue * DELTA);
    }

    @Test
    public void testVerySmallNumbers() {
        // 测试非常小的数值
        float smallValue = Float.MIN_VALUE * 1000;
        NdArray array = NdArray.like(Shape.of(2, 2), smallValue);
        
        assertTrue(array.get(0, 0) > 0);
    }

    @Test
    public void testVerySmallPositiveNumber() {
        // 测试接近零的正数
        float tiny = 1e-30f;
        NdArray array = NdArray.like(Shape.of(2, 2), tiny);
        
        NdArray result = array.add(array);
        assertEquals(tiny * 2, result.get(0, 0), tiny);
    }

    @Test
    public void testMaxFloatValue() {
        // 测试Float.MAX_VALUE
        NdArray array = NdArray.like(Shape.of(2, 2), Float.MAX_VALUE);
        assertEquals(Float.MAX_VALUE, array.get(0, 0), DELTA);
    }

    @Test
    public void testMinPositiveFloatValue() {
        // 测试Float.MIN_VALUE
        NdArray array = NdArray.like(Shape.of(2, 2), Float.MIN_VALUE);
        assertEquals(Float.MIN_VALUE, array.get(0, 0), 0);
    }

    // =============================================================================
    // 精度损失测试
    // =============================================================================

    @Test
    public void testPrecisionLossInSum() {
        // 测试求和时的精度损失
        NdArray array = NdArray.like(Shape.of(1000), 0.001f);
        NdArray sum = array.sum();
        
        // 1000 * 0.001 = 1.0，但可能有精度损失
        assertEquals(1.0f, sum.getNumber().floatValue(), 0.01f);
    }

    @Test
    public void testAddingVeryDifferentMagnitudes() {
        // 测试加法中数量级差异很大的数
        float large = 1e10f;
        float small = 1e-10f;
        
        NdArray largeArray = NdArray.like(Shape.of(2, 2), large);
        NdArray smallArray = NdArray.like(Shape.of(2, 2), small);
        
        NdArray result = largeArray.add(smallArray);
        // 小数会被大数"吞没"
        assertEquals(large, result.get(0, 0), large * DELTA);
    }

    @Test
    public void testMultiplicationPrecision() {
        // 测试乘法精度
        NdArray a = NdArray.like(Shape.of(2, 2), 0.1f);
        NdArray b = NdArray.like(Shape.of(2, 2), 0.2f);
        
        NdArray result = a.mul(b);
        assertEquals(0.02f, result.get(0, 0), RELAXED_DELTA);
    }

    @Test
    public void testDivisionPrecision() {
        // 测试除法精度
        NdArray a = NdArray.like(Shape.of(2, 2), 1.0f);
        NdArray b = NdArray.like(Shape.of(2, 2), 3.0f);
        
        NdArray result = a.div(b);
        assertEquals(1.0f / 3.0f, result.get(0, 0), RELAXED_DELTA);
    }

    // =============================================================================
    // 特殊值测试（NaN和Infinity）
    // =============================================================================

    @Test
    public void testInfinityValues() {
        // 测试无穷大值
        NdArray array = NdArray.like(Shape.of(2, 2), Float.POSITIVE_INFINITY);
        assertTrue(Float.isInfinite(array.get(0, 0)));
    }

    @Test
    public void testNegativeInfinityValues() {
        // 测试负无穷大值
        NdArray array = NdArray.like(Shape.of(2, 2), Float.NEGATIVE_INFINITY);
        assertTrue(Float.isInfinite(array.get(0, 0)));
        assertTrue(array.get(0, 0) < 0);
    }

    @Test
    public void testNaNValues() {
        // 测试NaN值
        NdArray array = NdArray.like(Shape.of(2, 2), Float.NaN);
        assertTrue(Float.isNaN(array.get(0, 0)));
    }

    @Test
    public void testOperationsWithInfinity() {
        // 测试包含无穷大的运算
        NdArray inf = NdArray.like(Shape.of(2, 2), Float.POSITIVE_INFINITY);
        NdArray normal = NdArray.like(Shape.of(2, 2), 5f);
        
        NdArray result = inf.add(normal);
        assertTrue(Float.isInfinite(result.get(0, 0)));
    }

    @Test
    public void testNaNPropagation() {
        // 测试NaN的传播
        NdArray nan = NdArray.like(Shape.of(2, 2), Float.NaN);
        NdArray normal = NdArray.like(Shape.of(2, 2), 5f);
        
        NdArray result = nan.add(normal);
        assertTrue(Float.isNaN(result.get(0, 0)));
    }

    // =============================================================================
    // 溢出测试
    // =============================================================================

    @Test
    public void testAdditionOverflow() {
        // 测试加法溢出
        float large = Float.MAX_VALUE / 2;
        NdArray array = NdArray.like(Shape.of(2, 2), large);
        
        NdArray result = array.add(array).add(array);
        // 可能溢出到Infinity
        float val = result.get(0, 0);
        assertTrue(Float.isInfinite(val) || val == Float.MAX_VALUE);
    }

    @Test
    public void testMultiplicationOverflow() {
        // 测试乘法溢出
        float large = (float) Math.sqrt(Float.MAX_VALUE);
        NdArray array = NdArray.like(Shape.of(2, 2), large);
        
        NdArray result = array.mul(array).mul(array);
        assertTrue(Float.isInfinite(result.get(0, 0)));
    }

    @Test
    public void testUnderflow() {
        // 测试下溢
        float tiny = Float.MIN_VALUE;
        NdArray array = NdArray.like(Shape.of(2, 2), tiny);
        
        NdArray result = array.divNum(1e10f);
        // 可能下溢到0
        assertEquals(0f, result.get(0, 0), 0);
    }

    // =============================================================================
    // 数学函数的数值边界测试
    // =============================================================================

    @Test
    public void testExpOfLargeNumber() {
        // exp(大数)会溢出
        NdArray array = NdArray.like(Shape.of(2, 2), 100f);
        NdArray result = array.exp();
        
        assertTrue(Float.isInfinite(result.get(0, 0)));
    }

    @Test
    public void testExpOfZero() {
        // exp(0) = 1
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        NdArray result = zeros.exp();
        
        assertEquals(1f, result.get(0, 0), DELTA);
    }

    @Test
    public void testLogOfVerySmallNumber() {
        // log(很小的数)会得到很大的负数
        NdArray array = NdArray.like(Shape.of(2, 2), 1e-30f);
        NdArray result = array.log();
        
        assertTrue(result.get(0, 0) < -60); // ln(1e-30) ≈ -69
    }

    @Test
    public void testSqrtOfZero() {
        // sqrt(0) = 0
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        NdArray result = zeros.sqrt();
        
        assertEquals(0f, result.get(0, 0), DELTA);
    }

    @Test
    public void testSqrtOfVeryLargeNumber() {
        // sqrt(大数)
        NdArray array = NdArray.like(Shape.of(2, 2), 1e20f);
        NdArray result = array.sqrt();
        
        assertEquals(1e10f, result.get(0, 0), 1e5f);
    }

    @Test
    public void testPowWithZeroExponent() {
        // 任何数的0次方都是1
        NdArray array = NdArray.of(new float[][]{{5f, -3f, 0f}});
        NdArray result = array.pow(0);
        
        for (int i = 0; i < 3; i++) {
            assertEquals(1f, result.get(0, i), DELTA);
        }
    }

    @Test
    public void testPowWithLargeExponent() {
        // 大指数次方
        NdArray array = NdArray.like(Shape.of(2, 2), 2f);
        NdArray result = array.pow(50);
        
        // 2^50 会溢出，结果应该非常大或无穷
        float val = result.get(0, 0);
        assertTrue(Float.isInfinite(val) || val > 1e14f);
    }

    // =============================================================================
    // 三角函数边界测试
    // =============================================================================

    @Test
    public void testSinOfZero() {
        // sin(0) = 0
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        NdArray result = zeros.sin();
        
        assertEquals(0f, result.get(0, 0), DELTA);
    }

    @Test
    public void testCosOfZero() {
        // cos(0) = 1
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        NdArray result = zeros.cos();
        
        assertEquals(1f, result.get(0, 0), DELTA);
    }

    @Test
    public void testSinOfVeryLargeNumber() {
        // sin(大数)，精度可能有问题
        NdArray array = NdArray.like(Shape.of(2, 2), 1e10f);
        NdArray result = array.sin();
        
        // sin的值应该在[-1, 1]之间
        float val = result.get(0, 0);
        assertTrue(val >= -1f && val <= 1f);
    }

    @Test
    public void testTanhSaturation() {
        // tanh在极值时饱和到±1
        NdArray large = NdArray.like(Shape.of(2, 2), 100f);
        NdArray result = large.tanh();
        
        assertEquals(1f, result.get(0, 0), RELAXED_DELTA);
    }

    // =============================================================================
    // Softmax数值稳定性测试
    // =============================================================================

    @Test
    public void testSoftmaxWithLargeValues() {
        // Softmax需要处理大值的数值稳定性
        NdArray array = NdArray.of(new float[][]{{1000f, 1001f, 999f}});
        NdArray result = array.softMax();
        
        // 所有值应该在[0,1]之间
        for (int i = 0; i < 3; i++) {
            float val = result.get(0, i);
            assertTrue(val >= 0f && val <= 1f);
            assertFalse(Float.isNaN(val));
        }
        
        // 和应该为1
        float sum = result.get(0, 0) + result.get(0, 1) + result.get(0, 2);
        assertEquals(1f, sum, RELAXED_DELTA);
    }

    @Test
    public void testSoftmaxWithNegativeValues() {
        // Softmax处理负值
        NdArray array = NdArray.of(new float[][]{{-1f, -2f, -3f}});
        NdArray result = array.softMax();
        
        float sum = result.get(0, 0) + result.get(0, 1) + result.get(0, 2);
        assertEquals(1f, sum, DELTA);
    }

    // =============================================================================
    // 聚合函数的数值稳定性测试
    // =============================================================================

    @Test
    public void testSumOfLargeArray() {
        // 大数组求和可能累积误差
        int size = 10000;
        NdArray array = NdArray.like(Shape.of(size), 0.1f);
        NdArray sum = array.sum();
        
        // 10000 * 0.1 = 1000，允许一定误差
        assertEquals(1000f, sum.getNumber().floatValue(), 1f);
    }

    @Test
    public void testMeanOfMixedValues() {
        // 包含正负值的平均
        NdArray array = NdArray.of(new float[][]{{1e10f, -1e10f, 5f}});
        NdArray mean = array.mean(1);
        
        // 平均值应该接近5/3
        float[][] meanMatrix = mean.getMatrix();
        assertEquals(5f / 3f, meanMatrix[0][0], RELAXED_DELTA);
    }

    @Test
    public void testVarianceOfConstantArray() {
        // 常数数组的方差应为0
        NdArray constant = NdArray.like(Shape.of(5, 5), 7f);
        NdArray variance = constant.var(0);
        
        for (float val : variance.getArray()) {
            assertEquals(0f, val, DELTA);
        }
    }

    @Test
    public void testVarOfLargeVariance() {
        // 大方差测试
        NdArray array = NdArray.of(new float[][]{{-1e6f, 1e6f}});
        NdArray variance = array.var(1);
        
        // 方差应该是正数且很大
        float[][] varMatrix = variance.getMatrix();
        assertTrue(varMatrix[0][0] > 1e11f);
    }

    // =============================================================================
    // 负零测试
    // =============================================================================

    @Test
    public void testNegativeZero() {
        // Java float支持-0.0
        NdArray array = NdArray.like(Shape.of(2, 2), -0.0f);
        
        // -0.0应该等于0.0
        assertEquals(0f, array.get(0, 0), 0);
    }

    @Test
    public void testNegativeZeroInOperations() {
        // -0.0参与运算
        NdArray negZero = NdArray.like(Shape.of(2, 2), -0.0f);
        NdArray posZero = NdArray.zeros(Shape.of(2, 2));
        
        NdArray result = negZero.add(posZero);
        assertEquals(0f, result.get(0, 0), 0);
    }

    // =============================================================================
    // 混合精度测试
    // =============================================================================

    @Test
    public void testMixedPrecisionAddition() {
        // 混合精度加法
        NdArray highPrecision = NdArray.like(Shape.of(2, 2), 1.23456789f);
        NdArray lowPrecision = NdArray.like(Shape.of(2, 2), 1.2f);
        
        NdArray result = highPrecision.add(lowPrecision);
        // 结果精度受限于float
        assertTrue(result.get(0, 0) > 2.4f && result.get(0, 0) < 2.5f);
    }

    @Test
    public void testRepeatedOperationsDrift() {
        // 测试重复操作的数值漂移
        NdArray array = NdArray.like(Shape.of(10), 1.0f);
        NdArray addVal = NdArray.like(Shape.of(10), 0.01f);
        NdArray subVal = NdArray.like(Shape.of(10), 0.01f);
        
        // 重复100次：加0.01再减0.01
        for (int i = 0; i < 100; i++) {
            array = array.add(addVal);
            array = array.sub(subVal);
        }
        
        // 理论上应该还是1.0，但可能有漂移
        float[] arrData = array.getArray();
        assertEquals(1.0f, arrData[0], 0.01f);
    }
}
