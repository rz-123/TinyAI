package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray算术运算测试
 * 
 * 测试基础算术运算，包括：
 * - 加减乘除运算
 * - 标量运算
 * - 广播运算
 * - 边界条件
 *
 * @author TinyAI
 */
public class NdArrayArithmeticTest {

    private NdArray a2x3;
    private NdArray b2x3;
    private NdArray scalar;

    @Before
    public void setUp() {
        a2x3 = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        b2x3 = NdArray.of(new float[][]{{2f, 3f, 4f}, {5f, 6f, 7f}});
        scalar = NdArray.of(2f);
    }

    // =============================================================================
    // 加法运算测试
    // =============================================================================

    @Test
    public void testAdd() {
        // 测试基本加法
        NdArray result = a2x3.add(b2x3);
        
        float[][] expected = {{3f, 5f, 7f}, {9f, 11f, 13f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testAddWithZero() {
        // 测试与零数组相加
        NdArray zeros = NdArray.zeros(a2x3.getShape());
        NdArray result = a2x3.add(zeros);
        
        assertArrayEquals(a2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testAddCommutative() {
        // 测试加法交换律 a + b = b + a
        NdArray result1 = a2x3.add(b2x3);
        NdArray result2 = b2x3.add(a2x3);
        
        assertArrayEquals(result1.getArray(), result2.getArray(), 1e-6f);
    }

    @Test
    public void testAddAssociative() {
        // 测试加法结合律 (a + b) + c = a + (b + c)
        NdArray c2x3 = NdArray.of(new float[][]{{1f, 1f, 1f}, {1f, 1f, 1f}});
        
        NdArray result1 = a2x3.add(b2x3).add(c2x3);
        NdArray result2 = a2x3.add(b2x3.add(c2x3));
        
        assertArrayEquals(result1.getArray(), result2.getArray(), 1e-5f);
    }

    @Test
    public void testAddNegativeNumbers() {
        // 测试负数加法
        NdArray negative = NdArray.of(new float[][]{{-1f, -2f, -3f}, {-4f, -5f, -6f}});
        NdArray result = a2x3.add(negative);
        
        float[][] expected = {{0f, 0f, 0f}, {0f, 0f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testAddIncompatibleShapes() {
        // 测试不兼容形状的加法
        NdArray incompatible = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        a2x3.add(incompatible);
    }

    // =============================================================================
    // 减法运算测试
    // =============================================================================

    @Test
    public void testSub() {
        // 测试基本减法
        NdArray result = b2x3.sub(a2x3);
        
        float[][] expected = {{1f, 1f, 1f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSubSelf() {
        // 测试自减运算 a - a = 0
        NdArray result = a2x3.sub(a2x3);
        
        float[][] expected = {{0f, 0f, 0f}, {0f, 0f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSubWithZero() {
        // 测试减零
        NdArray zeros = NdArray.zeros(a2x3.getShape());
        NdArray result = a2x3.sub(zeros);
        
        assertArrayEquals(a2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testSubNegativeResult() {
        // 测试产生负结果的减法
        NdArray result = a2x3.sub(b2x3);
        
        float[][] expected = {{-1f, -1f, -1f}, {-1f, -1f, -1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSubIncompatibleShapes() {
        // 测试不兼容形状的减法
        NdArray incompatible = NdArray.of(new float[]{1f, 2f, 3f});
        a2x3.sub(incompatible);
    }

    // =============================================================================
    // 乘法运算测试
    // =============================================================================

    @Test
    public void testMul() {
        // 测试基本元素级乘法
        NdArray result = a2x3.mul(b2x3);
        
        float[][] expected = {{2f, 6f, 12f}, {20f, 30f, 42f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMulWithZero() {
        // 测试与零相乘
        NdArray zeros = NdArray.zeros(a2x3.getShape());
        NdArray result = a2x3.mul(zeros);
        
        float[][] expected = {{0f, 0f, 0f}, {0f, 0f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMulWithOne() {
        // 测试与全一数组相乘
        NdArray ones = NdArray.ones(a2x3.getShape());
        NdArray result = a2x3.mul(ones);
        
        assertArrayEquals(a2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testMulCommutative() {
        // 测试乘法交换律 a * b = b * a
        NdArray result1 = a2x3.mul(b2x3);
        NdArray result2 = b2x3.mul(a2x3);
        
        assertArrayEquals(result1.getArray(), result2.getArray(), 1e-6f);
    }

    @Test
    public void testMulNum() {
        // 测试标量乘法
        NdArray result = a2x3.mulNum(2f);
        
        float[][] expected = {{2f, 4f, 6f}, {8f, 10f, 12f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMulNumWithZero() {
        // 测试乘以零标量
        NdArray result = a2x3.mulNum(0f);
        
        float[][] expected = {{0f, 0f, 0f}, {0f, 0f, 0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMulNumWithNegative() {
        // 测试乘以负数标量
        NdArray result = a2x3.mulNum(-1f);
        
        float[][] expected = {{-1f, -2f, -3f}, {-4f, -5f, -6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMulIncompatibleShapes() {
        // 测试不兼容形状的乘法
        NdArray incompatible = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}, {5f, 6f}});
        a2x3.mul(incompatible);
    }

    // =============================================================================
    // 除法运算测试
    // =============================================================================

    @Test
    public void testDiv() {
        // 测试基本除法
        NdArray a = NdArray.of(new float[][]{{6f, 8f, 10f}, {12f, 15f, 18f}});
        NdArray b = NdArray.of(new float[][]{{2f, 4f, 5f}, {3f, 5f, 6f}});
        NdArray result = a.div(b);
        
        float[][] expected = {{3f, 2f, 2f}, {4f, 3f, 3f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDivByOne() {
        // 测试除以1
        NdArray ones = NdArray.ones(a2x3.getShape());
        NdArray result = a2x3.div(ones);
        
        assertArrayEquals(a2x3.getMatrix(), result.getMatrix());
    }

    @Test
    public void testDivSelf() {
        // 测试自除 a / a = 1
        NdArray result = a2x3.div(a2x3);
        
        float[][] expected = {{1f, 1f, 1f}, {1f, 1f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDivNum() {
        // 测试标量除法
        NdArray a = NdArray.of(new float[][]{{2f, 4f, 6f}, {8f, 10f, 12f}});
        NdArray result = a.divNum(2f);
        
        float[][] expected = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDivNumWithFraction() {
        // 测试除以小数
        NdArray a = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = a.divNum(0.5f);
        
        float[][] expected = {{2f, 4f}, {6f, 8f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test(expected = ArithmeticException.class)
    public void testDivByZero() {
        // 测试除以零数组
        NdArray zeros = NdArray.zeros(a2x3.getShape());
        a2x3.div(zeros);
    }

    @Test(expected = ArithmeticException.class)
    public void testDivNumByZero() {
        // 测试标量除以零
        a2x3.divNum(0f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDivIncompatibleShapes() {
        // 测试不兼容形状的除法
        NdArray incompatible = NdArray.of(new float[]{1f, 2f});
        a2x3.div(incompatible);
    }

    // =============================================================================
    // 混合运算测试
    // =============================================================================

    @Test
    public void testChainedOperations() {
        // 测试链式运算 (a + b) * c - d
        NdArray c = NdArray.of(new float[][]{{2f, 2f, 2f}, {2f, 2f, 2f}});
        NdArray d = NdArray.of(new float[][]{{1f, 1f, 1f}, {1f, 1f, 1f}});
        
        NdArray result = a2x3.add(b2x3).mul(c).sub(d);
        
        // (a + b) = {{3, 5, 7}, {9, 11, 13}}
        // * c = {{6, 10, 14}, {18, 22, 26}}
        // - d = {{5, 9, 13}, {17, 21, 25}}
        float[][] expected = {{5f, 9f, 13f}, {17f, 21f, 25f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDistributiveLaw() {
        // 测试分配律 a * (b + c) = a*b + a*c
        NdArray c = NdArray.of(new float[][]{{1f, 1f, 1f}, {1f, 1f, 1f}});
        
        NdArray left = a2x3.mul(b2x3.add(c));
        NdArray right = a2x3.mul(b2x3).add(a2x3.mul(c));
        
        assertArrayEquals(left.getArray(), right.getArray(), 1e-5f);
    }

    @Test
    public void testOperationsWithVerySmallNumbers() {
        // 测试极小数运算
        NdArray small1 = NdArray.of(new float[][]{{1e-6f, 2e-6f}, {3e-6f, 4e-6f}});
        NdArray small2 = NdArray.of(new float[][]{{5e-6f, 6e-6f}, {7e-6f, 8e-6f}});
        
        NdArray result = small1.add(small2);
        assertNotNull(result);
        assertEquals(Shape.of(2, 2), result.getShape());
    }

    @Test
    public void testOperationsWithVeryLargeNumbers() {
        // 测试极大数运算
        NdArray large1 = NdArray.of(new float[][]{{1e6f, 2e6f}, {3e6f, 4e6f}});
        NdArray large2 = NdArray.of(new float[][]{{5e6f, 6e6f}, {7e6f, 8e6f}});
        
        NdArray result = large1.add(large2);
        float[][] expected = {{6e6f, 8e6f}, {10e6f, 12e6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    // =============================================================================
    // 特殊值测试
    // =============================================================================

    @Test
    public void testOperationsWithInfinity() {
        // 测试包含无穷大的运算
        NdArray withInf = NdArray.of(new float[][]{{Float.POSITIVE_INFINITY, 1f}, {2f, Float.NEGATIVE_INFINITY}});
        NdArray normal = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray result = withInf.add(normal);
        assertTrue(Float.isInfinite(result.get(0, 0)));
        assertTrue(Float.isInfinite(result.get(1, 1)));
    }

    @Test
    public void testMixedSignOperations() {
        // 测试正负混合运算
        NdArray positive = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray negative = NdArray.of(new float[][]{{-1f, -2f, -3f}, {-4f, -5f, -6f}});
        
        NdArray sum = positive.add(negative);
        float[][] expectedSum = {{0f, 0f, 0f}, {0f, 0f, 0f}};
        assertArrayEquals(expectedSum, sum.getMatrix());
        
        NdArray product = positive.mul(negative);
        float[][] expectedProduct = {{-1f, -4f, -9f}, {-16f, -25f, -36f}};
        assertArrayEquals(expectedProduct, product.getMatrix());
    }
}
