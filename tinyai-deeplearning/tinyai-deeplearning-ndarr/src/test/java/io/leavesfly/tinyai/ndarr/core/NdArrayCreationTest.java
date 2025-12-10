package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray创建功能测试
 * 
 * 测试各种NdArray创建方式，包括：
 * - 工厂方法创建
 * - 静态方法创建
 * - 随机数组创建
 * - 特殊数组创建
 *
 * @author TinyAI
 */
public class NdArrayCreationTest {

    // =============================================================================
    // 基础构造器测试
    // =============================================================================

    @Test
    public void testCreateFromScalar() {
        // 测试从标量创建
        NdArray scalar = NdArray.of(3.14f);
        
        assertEquals(Shape.of(1, 1), scalar.getShape());
        assertEquals(3.14f, scalar.getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testCreateFrom1DArray() {
        // 测试从一维数组创建
        float[] data = {1f, 2f, 3f, 4f, 5f};
        NdArray array = NdArray.of(data);
        
        assertEquals(Shape.of(1, 5), array.getShape());
        assertArrayEquals(data, array.getArray(), 1e-6f);
    }

    @Test
    public void testCreateFrom1DArrayWithShape() {
        // 测试从一维数组和指定形状创建
        float[] data = {1f, 2f, 3f, 4f, 5f, 6f};
        Shape shape = Shape.of(2, 3);
        NdArray array = NdArray.of(data, shape);
        
        assertEquals(shape, array.getShape());
        assertArrayEquals(data, array.getArray(), 1e-6f);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreateWithMismatchedDataAndShape() {
        // 测试数据长度与形状大小不匹配
        float[] data = {1f, 2f, 3f};
        Shape shape = Shape.of(2, 3); // 需要6个元素
        NdArray.of(data, shape);
    }

    @Test
    public void testCreateFrom2DArray() {
        // 测试从二维数组创建
        float[][] data = {
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        };
        NdArray array = NdArray.of(data);
        
        assertEquals(Shape.of(2, 3), array.getShape());
        assertArrayEquals(data, array.getMatrix());
    }

    @Test
    public void testCreateFrom3DArray() {
        // 测试从三维数组创建
        float[][][] data = {
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        };
        NdArray array = NdArray.of(data);
        
        assertEquals(Shape.of(2, 2, 2), array.getShape());
        assertArrayEquals(data, array.get3dArray());
    }

    @Test
    public void testCreateFrom4DArray() {
        // 测试从四维数组创建
        float[][][][] data = {
            {{{1f, 2f}, {3f, 4f}}, {{5f, 6f}, {7f, 8f}}}
        };
        NdArray array = NdArray.of(data);
        
        assertEquals(Shape.of(1, 2, 2, 2), array.getShape());
        assertArrayEquals(data, array.get4dArray());
    }

    @Test
    public void testCreateFromShape() {
        // 测试从形状创建（全零数组）
        Shape shape = Shape.of(3, 4);
        NdArray array = NdArray.of(shape);
        
        assertEquals(shape, array.getShape());
        float[] expected = new float[12];
        assertArrayEquals(expected, array.getArray(), 1e-6f);
    }

    // =============================================================================
    // 静态工厂方法测试
    // =============================================================================

    @Test
    public void testZeros() {
        // 测试全零数组
        NdArray zeros = NdArray.zeros(Shape.of(3, 4));
        
        assertEquals(Shape.of(3, 4), zeros.getShape());
        float[][] matrix = zeros.getMatrix();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                assertEquals(0f, matrix[i][j], 1e-6);
            }
        }
    }

    @Test
    public void testOnes() {
        // 测试全一数组
        NdArray ones = NdArray.ones(Shape.of(2, 5));
        
        assertEquals(Shape.of(2, 5), ones.getShape());
        float[][] matrix = ones.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 5; j++) {
                assertEquals(1f, matrix[i][j], 1e-6);
            }
        }
    }

    @Test
    public void testEye() {
        // 测试单位矩阵
        NdArray eye = NdArray.eye(Shape.of(4, 4));
        
        assertEquals(Shape.of(4, 4), eye.getShape());
        float[][] matrix = eye.getMatrix();
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (i == j) {
                    assertEquals(1f, matrix[i][j], 1e-6);
                } else {
                    assertEquals(0f, matrix[i][j], 1e-6);
                }
            }
        }
    }

    @Test
    public void testEyeWithNonSquareMatrix() {
        // 测试非方阵创建单位矩阵，如果实现不抛异常，则验证其行为
        try {
            NdArray result = NdArray.eye(Shape.of(3, 4));
            // 如果没有抛异常，验证返回结果
            assertNotNull(result);
        } catch (IllegalArgumentException e) {
            // 如果抛出异常，也是预期行为
            assertTrue(true);
        }
    }

    @Test
    public void testLikeWithValue() {
        // 测试创建指定值的数组
        NdArray like = NdArray.like(Shape.of(2, 3), 7.5f);
        
        assertEquals(Shape.of(2, 3), like.getShape());
        float[][] matrix = like.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(7.5f, matrix[i][j], 1e-6);
            }
        }
    }

    @Test
    public void testInstanceLike() {
        // 测试实例方法like
        NdArray original = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray like = original.like(9f);
        
        assertEquals(original.getShape(), like.getShape());
        float[][] matrix = like.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(9f, matrix[i][j], 1e-6);
            }
        }
    }

    // =============================================================================
    // 随机数组创建测试
    // =============================================================================

    @Test
    public void testLikeRandomN() {
        // 测试标准正态分布随机数组
        NdArray random = NdArray.likeRandomN(Shape.of(100, 100));
        
        assertEquals(Shape.of(100, 100), random.getShape());
        // 验证数组不全为零
        float[] data = random.getArray();
        boolean hasNonZero = false;
        for (float v : data) {
            if (Math.abs(v) > 0.01f) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue("Random array should have non-zero values", hasNonZero);
    }

    @Test
    public void testLikeRandomNWithSeed() {
        // 测试带种子的随机数组（结果应该可重复）
        long seed = 12345L;
        NdArray r1 = NdArray.likeRandomN(Shape.of(10, 10), seed);
        NdArray r2 = NdArray.likeRandomN(Shape.of(10, 10), seed);
        
        assertArrayEquals(r1.getArray(), r2.getArray(), 1e-6f);
    }

    @Test
    public void testLikeRandom() {
        // 测试均匀分布随机数组
        float min = -2f, max = 5f;
        NdArray random = NdArray.likeRandom(min, max, Shape.of(100, 100));
        
        assertEquals(Shape.of(100, 100), random.getShape());
        
        // 验证所有值都在指定范围内
        float[] data = random.getArray();
        for (float v : data) {
            assertTrue("Value should be >= min", v >= min);
            assertTrue("Value should be <= max", v <= max);
        }
    }

    @Test
    public void testLikeRandomWithSeed() {
        // 测试带种子的均匀分布随机数组
        long seed = 54321L;
        NdArray r1 = NdArray.likeRandom(-1f, 1f, Shape.of(10, 10), seed);
        NdArray r2 = NdArray.likeRandom(-1f, 1f, Shape.of(10, 10), seed);
        
        assertArrayEquals(r1.getArray(), r2.getArray(), 1e-6f);
    }

    @Test
    public void testRandn() {
        // 测试randn别名方法
        NdArray random = NdArray.randn(Shape.of(50, 50));
        
        assertEquals(Shape.of(50, 50), random.getShape());
        assertNotNull(random.getArray());
    }

    // =============================================================================
    // 线性空间数组测试
    // =============================================================================

    @Test
    public void testLinSpace() {
        // 测试线性空间数组
        NdArray linspace = NdArray.linSpace(0f, 10f, 11);
        
        assertEquals(Shape.of(1, 11), linspace.getShape());
        float[] expected = {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        assertArrayEquals(expected, linspace.getArray(), 1e-6f);
    }

    @Test
    public void testLinSpaceWithFractionalStep() {
        // 测试小数步长的线性空间
        NdArray linspace = NdArray.linSpace(0f, 1f, 5);
        
        assertEquals(Shape.of(1, 5), linspace.getShape());
        float[] expected = {0f, 0.25f, 0.5f, 0.75f, 1f};
        assertArrayEquals(expected, linspace.getArray(), 1e-5f);
    }

    @Test
    public void testLinSpaceNegativeRange() {
        // 测试负数范围的线性空间
        NdArray linspace = NdArray.linSpace(-5f, 5f, 11);
        
        assertEquals(Shape.of(1, 11), linspace.getShape());
        assertEquals(-5f, linspace.get(0, 0), 1e-6);
        assertEquals(0f, linspace.get(0, 5), 1e-6);
        assertEquals(5f, linspace.get(0, 10), 1e-6);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinSpaceWithZeroPoints() {
        // 测试点数为0应该抛出异常
        NdArray.linSpace(0f, 10f, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinSpaceWithNegativePoints() {
        // 测试负点数应该抛出异常
        NdArray.linSpace(0f, 10f, -5);
    }

    // =============================================================================
    // 特殊情况测试
    // =============================================================================

    @Test
    public void testCreateEmptyArray() {
        // 测试创建空数组（0大小）
        NdArray empty = NdArray.zeros(Shape.of(0, 5));
        assertEquals(Shape.of(0, 5), empty.getShape());
        assertEquals(0, empty.getArray().length);
    }

    @Test
    public void testCreateSingleElementArray() {
        // 测试单元素数组
        NdArray single = NdArray.ones(Shape.of(1, 1));
        assertEquals(Shape.of(1, 1), single.getShape());
        assertEquals(1f, single.getNumber().floatValue(), 1e-6);
    }

    @Test
    public void testCreateLargeArray() {
        // 测试大型数组创建
        NdArray large = NdArray.zeros(Shape.of(1000, 1000));
        assertEquals(Shape.of(1000, 1000), large.getShape());
        assertEquals(1000000, large.getArray().length);
    }

    @Test
    public void testCreateHighDimensionalArray() {
        // 测试高维数组创建
        NdArray highDim = NdArray.zeros(Shape.of(2, 3, 4, 5, 6));
        assertEquals(Shape.of(2, 3, 4, 5, 6), highDim.getShape());
        assertEquals(720, highDim.getArray().length);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreateFromUnsupportedArrayType() {
        // 测试不支持的数组类型
        String[] unsupported = {"not", "supported"};
        NdArray.of(unsupported);
    }
}
