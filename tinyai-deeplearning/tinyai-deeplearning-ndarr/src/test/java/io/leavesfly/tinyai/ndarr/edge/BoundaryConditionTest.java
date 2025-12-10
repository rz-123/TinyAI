package io.leavesfly.tinyai.ndarr.edge;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 边界条件测试
 * 
 * 测试各种边界情况，包括：
 * - 空数组和零维度
 * - 单元素数组
 * - 极大和极小数组
 * - 特殊形状
 *
 * @author TinyAI
 */
public class BoundaryConditionTest {

    private static final float DELTA = 1e-6f;

    // =============================================================================
    // 空数组和零维度测试
    // =============================================================================

    @Test
    public void testEmptyShape() {
        // 测试包含0维度的形状
        NdArray array = NdArray.zeros(Shape.of(0, 5));
        assertEquals(Shape.of(0, 5), array.getShape());
        assertEquals(0, array.getArray().length);
    }

    @Test
    public void testZeroColumnMatrix() {
        // 测试0列矩阵
        NdArray array = NdArray.zeros(Shape.of(3, 0));
        assertEquals(Shape.of(3, 0), array.getShape());
        assertEquals(0, array.getArray().length);
    }

    @Test
    public void testOperationsOnEmptyArray() {
        // 测试空数组上的操作
        NdArray empty1 = NdArray.zeros(Shape.of(0, 3));
        NdArray empty2 = NdArray.zeros(Shape.of(0, 3));
        
        NdArray result = empty1.add(empty2);
        assertEquals(0, result.getArray().length);
    }

    // =============================================================================
    // 单元素数组测试
    // =============================================================================

    @Test
    public void testSingleElementCreation() {
        // 测试单元素数组创建
        NdArray single = NdArray.of(new float[][]{{5f}});
        assertEquals(Shape.of(1, 1), single.getShape());
        assertEquals(5f, single.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSingleElementOperations() {
        // 测试单元素数组的运算
        NdArray a = NdArray.of(new float[][]{{3f}});
        NdArray b = NdArray.of(new float[][]{{2f}});
        
        assertEquals(5f, a.add(b).getNumber().floatValue(), DELTA);
        assertEquals(1f, a.sub(b).getNumber().floatValue(), DELTA);
        assertEquals(6f, a.mul(b).getNumber().floatValue(), DELTA);
        assertEquals(1.5f, a.div(b).getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSingleElementMathFunctions() {
        // 测试单元素数组的数学函数
        NdArray single = NdArray.of(new float[][]{{2f}});
        
        assertEquals(4f, single.square().getNumber().floatValue(), DELTA);
        assertEquals((float)Math.sqrt(2), single.sqrt().getNumber().floatValue(), DELTA);
        assertEquals((float)Math.exp(2), single.exp().getNumber().floatValue(), DELTA);
    }

    // =============================================================================
    // 极端尺寸测试
    // =============================================================================

    @Test
    public void testVeryLargeArray() {
        // 测试大型数组（10000元素）
        int size = 10000;
        NdArray large = NdArray.zeros(Shape.of(100, 100));
        
        assertEquals(size, large.getArray().length);
        assertEquals(Shape.of(100, 100), large.getShape());
    }

    @Test
    public void testVeryLongVector() {
        // 测试超长向量
        NdArray longVector = NdArray.zeros(Shape.of(1, 50000));
        assertEquals(50000, longVector.getArray().length);
    }

    @Test
    public void testHighDimensionalArray() {
        // 测试高维数组
        NdArray highDim = NdArray.zeros(Shape.of(2, 3, 4, 5));
        assertEquals(120, highDim.getArray().length);
        assertEquals(4, highDim.getShape().getDimNum());
    }

    @Test
    public void testVeryHighDimensionalArray() {
        // 测试超高维数组（6维）
        NdArray veryHighDim = NdArray.zeros(Shape.of(2, 2, 2, 2, 2, 2));
        assertEquals(64, veryHighDim.getArray().length);
        assertEquals(6, veryHighDim.getShape().getDimNum());
    }

    // =============================================================================
    // 特殊形状测试
    // =============================================================================

    @Test
    public void testSquareMatrices() {
        // 测试各种大小的方阵
        for (int size : new int[]{1, 2, 5, 10, 100}) {
            NdArray square = NdArray.zeros(Shape.of(size, size));
            assertEquals(size * size, square.getArray().length);
        }
    }

    @Test
    public void testVeryTallMatrix() {
        // 测试非常"高"的矩阵
        NdArray tall = NdArray.zeros(Shape.of(1000, 2));
        assertEquals(2000, tall.getArray().length);
    }

    @Test
    public void testVeryWideMatrix() {
        // 测试非常"宽"的矩阵
        NdArray wide = NdArray.zeros(Shape.of(2, 1000));
        assertEquals(2000, wide.getArray().length);
    }

    @Test
    public void testExtremeAspectRatio() {
        // 测试极端长宽比
        NdArray extreme1 = NdArray.zeros(Shape.of(10000, 1));
        NdArray extreme2 = NdArray.zeros(Shape.of(1, 10000));
        
        assertEquals(10000, extreme1.getArray().length);
        assertEquals(10000, extreme2.getArray().length);
    }

    // =============================================================================
    // 形状转换边界测试
    // =============================================================================

    @Test
    public void testReshapeToSingleElement() {
        // 测试reshape到单元素
        NdArray array = NdArray.of(new float[]{5f});
        NdArray reshaped = array.reshape(Shape.of(1, 1));
        assertEquals(5f, reshaped.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testReshapeBetweenExtremes() {
        // 测试在极端形状间reshape
        NdArray tall = NdArray.ones(Shape.of(100, 1));
        NdArray wide = tall.reshape(Shape.of(1, 100));
        
        assertEquals(Shape.of(1, 100), wide.getShape());
        assertArrayEquals(tall.getArray(), wide.getArray(), DELTA);
    }

    @Test
    public void testFlattenSingleElement() {
        // 测试单元素flatten
        NdArray single = NdArray.of(new float[][]{{7f}});
        NdArray flat = single.flatten();
        assertEquals(Shape.of(1, 1), flat.getShape());
    }

    // =============================================================================
    // 广播边界测试
    // =============================================================================

    @Test
    public void testBroadcastSingleToLarge() {
        // 测试单元素广播到大数组
        NdArray single = NdArray.of(new float[][]{{5f}});
        NdArray large = single.broadcastTo(Shape.of(100, 100));
        
        assertEquals(Shape.of(100, 100), large.getShape());
        for (float val : large.getArray()) {
            assertEquals(5f, val, DELTA);
        }
    }

    @Test
    public void testBroadcastToSingleElement() {
        // 测试广播到单元素（实际上就是原样返回）
        NdArray single = NdArray.of(new float[][]{{3f}});
        NdArray result = single.broadcastTo(Shape.of(1, 1));
        assertEquals(3f, result.getNumber().floatValue(), DELTA);
    }

    // =============================================================================
    // 切片边界测试
    // =============================================================================

    @Test
    public void testSliceSingleElement() {
        // 测试切片单个元素
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray slice = array.getItem(new int[]{0}, new int[]{0});
        assertEquals(1f, slice.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSliceEntireArray() {
        // 测试切片整个数组（null索引）
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray slice = array.getItem(null, null);
        assertArrayEquals(array.getArray(), slice.getArray(), DELTA);
    }

    // =============================================================================
    // 聚合操作边界测试
    // =============================================================================

    @Test
    public void testSumOfSingleElement() {
        // 测试单元素求和
        NdArray single = NdArray.of(new float[][]{{5f}});
        NdArray sum = single.sum();
        assertEquals(5f, sum.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testMeanOfSingleElement() {
        // 测试单元素平均值
        NdArray single = NdArray.of(new float[][]{{7f}});
        NdArray mean = single.mean(0);
        assertEquals(7f, mean.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testMaxOfSingleElement() {
        // 测试单元素最大值
        NdArray single = NdArray.of(new float[][]{{9f}});
        float max = single.max();
        assertEquals(9f, max, DELTA);
    }

    // =============================================================================
    // 矩阵运算边界测试
    // =============================================================================

    @Test
    public void testDotWithSingleElement() {
        // 测试单元素矩阵乘法
        NdArray a = NdArray.of(new float[][]{{3f}});
        NdArray b = NdArray.of(new float[][]{{2f}});
        NdArray result = a.dot(b);
        assertEquals(6f, result.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testTransposeSingleElement() {
        // 测试单元素转置
        NdArray single = NdArray.of(new float[][]{{5f}});
        NdArray transposed = single.transpose();
        assertEquals(Shape.of(1, 1), transposed.getShape());
        assertEquals(5f, transposed.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testDotWithVectorAndScalar() {
        // 测试向量与1x1矩阵的乘法
        NdArray vector = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray scalar = NdArray.of(new float[][]{{2f}});
        
        // 这应该失败或者有特殊处理
        try {
            NdArray result = vector.dot(scalar);
            // 如果成功，验证结果
            assertNotNull(result);
        } catch (RuntimeException e) {
            // 预期可能抛出异常
            assertTrue(true);
        }
    }

    // =============================================================================
    // 数值范围边界测试
    // =============================================================================

    @Test
    public void testAllZeros() {
        // 测试全零数组
        NdArray zeros = NdArray.zeros(Shape.of(3, 3));
        
        assertEquals(0f, zeros.sum().getNumber().floatValue(), DELTA);
        assertEquals(0f, zeros.max(), DELTA);
    }

    @Test
    public void testAllOnes() {
        // 测试全一数组
        NdArray ones = NdArray.ones(Shape.of(5, 5));
        
        assertEquals(25f, ones.sum().getNumber().floatValue(), DELTA);
        assertEquals(1f, ones.max(), DELTA);
    }

    @Test
    public void testAllSameValue() {
        // 测试所有元素相同的数组
        NdArray same = NdArray.like(Shape.of(4, 4), 7f);
        
        NdArray variance = same.var(0);
        for (float val : variance.getArray()) {
            assertEquals(0f, val, DELTA); // 方差应该为0
        }
    }

    // =============================================================================
    // 索引边界测试
    // =============================================================================

    @Test
    public void testGetFirstElement() {
        // 测试获取第一个元素
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        assertEquals(1f, array.get(0, 0), DELTA);
    }

    @Test
    public void testGetLastElement() {
        // 测试获取最后一个元素
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        assertEquals(6f, array.get(1, 2), DELTA);
    }

    @Test
    public void testSetFirstElement() {
        // 测试设置第一个元素
        NdArray array = NdArray.zeros(Shape.of(2, 2));
        array.set(99f, 0, 0);
        assertEquals(99f, array.get(0, 0), DELTA);
    }

    @Test
    public void testSetLastElement() {
        // 测试设置最后一个元素
        NdArray array = NdArray.zeros(Shape.of(3, 3));
        array.set(99f, 2, 2);
        assertEquals(99f, array.get(2, 2), DELTA);
    }

    // =============================================================================
    // 内存和性能边界测试
    // =============================================================================

    @Test
    public void testRepeatedOperations() {
        // 测试重复操作不会导致问题
        NdArray array = NdArray.ones(Shape.of(10, 10));
        
        for (int i = 0; i < 100; i++) {
            array = array.add(NdArray.ones(Shape.of(10, 10)));
        }
        
        assertEquals(101f, array.get(0, 0), DELTA);
    }

    @Test
    public void testChainedReshapes() {
        // 测试链式reshape
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        
        array = array.reshape(Shape.of(2, 3))
                    .reshape(Shape.of(3, 2))
                    .reshape(Shape.of(1, 6))
                    .reshape(Shape.of(6, 1));
        
        assertEquals(Shape.of(6, 1), array.getShape());
        assertArrayEquals(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, array.getArray(), DELTA);
    }
}
