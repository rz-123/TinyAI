package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray聚合统计测试
 * 
 * 测试各种聚合和统计操作，包括：
 * - sum, mean, var等统计函数
 * - argmax, argmin等索引函数
 * - max, min等极值函数
 * - 按轴聚合操作
 *
 * @author TinyAI
 */
public class NdArrayAggregationTest {

    private static final float DELTA = 1e-5f;

    // =============================================================================
    // Sum测试
    // =============================================================================

    @Test
    public void testSumAll() {
        // 测试全局求和
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.sum();
        
        assertEquals(21f, result.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSumAxis0() {
        // 测试按列求和（axis=0沿行方向聚合）
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.sum(0);
        
        // 实际返回形状是 [3]（一维）而非 [1,3]
        float[][] expected = {{5f, 7f, 9f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSumAxis1() {
        // 测试按行求和（axis=1沿列方向聚合）
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.sum(1);
        
        float[][] expected = {{6f, 15f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSumWithNegatives() {
        // 测试包含负数的求和
        NdArray array = NdArray.of(new float[][]{{-1f, 2f, -3f}, {4f, -5f, 6f}});
        NdArray result = array.sum();
        
        assertEquals(3f, result.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testSumZeros() {
        // 测试全零数组求和
        NdArray zeros = NdArray.zeros(Shape.of(3, 4));
        NdArray result = zeros.sum();
        
        assertEquals(0f, result.getNumber().floatValue(), DELTA);
    }

    // =============================================================================
    // Mean测试
    // =============================================================================

    @Test
    public void testMeanAxis0() {
        // 测试按列求平均
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.mean(0);
        
        float[][] expected = {{2.5f, 3.5f, 4.5f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMeanAxis1() {
        // 测试按行求平均
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.mean(1);
        
        float[][] expected = {{2f, 5f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMeanWithFloats() {
        // 测试小数平均值
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 5f}});
        NdArray result = array.mean(0);
        
        float[][] expected = {{2f, 3.5f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    // =============================================================================
    // Variance测试
    // =============================================================================

    @Test
    public void testVarAxis0() {
        // 测试按列计算方差
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.var(0);
        
        // 第一列: [1,4] 均值=2.5, 方差=((1-2.5)^2+(4-2.5)^2)/2=2.25
        float[][] expected = {{2.25f, 2.25f, 2.25f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testVarAxis1() {
        // 测试按行计算方差
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.var(1);
        
        // 第一行: [1,2,3] 均值=2, 方差=((1-2)^2+(2-2)^2+(3-2)^2)/3=2/3
        float[][] expected = {{2f/3f, 2f/3f}};
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], result.getMatrix()[i][j], DELTA);
            }
        }
    }

    @Test
    public void testVarOfConstants() {
        // 常数的方差应该为0
        NdArray constants = NdArray.like(Shape.of(2, 5), 3f);
        NdArray result = constants.var(1);
        
        float[][] matrix = result.getMatrix();
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                assertEquals(0f, matrix[i][j], DELTA);
            }
        }
    }

    // =============================================================================
    // Max测试
    // =============================================================================

    @Test
    public void testMaxGlobal() {
        // 测试全局最大值
        NdArray array = NdArray.of(new float[][]{{1f, 5f, 3f}, {2f, 9f, 4f}});
        float result = array.max();
        
        assertEquals(9f, result, DELTA);
    }

    @Test
    public void testMaxAxis0() {
        // 测试按列找最大值
        NdArray array = NdArray.of(new float[][]{{1f, 8f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.max(0);
        
        float[][] expected = {{4f, 8f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMaxAxis1() {
        // 测试按行找最大值
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {6f, 5f, 4f}});
        NdArray result = array.max(1);
        
        float[][] expected = {{3f}, {6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMaxWithNegatives() {
        // 测试全是负数的最大值
        NdArray array = NdArray.of(new float[][]{{-5f, -2f}, {-3f, -4f}});
        float result = array.max();
        
        assertEquals(-2f, result, DELTA);
    }

    // =============================================================================
    // Min测试
    // =============================================================================

    @Test
    public void testMinAxis0() {
        // 测试按列找最小值
        NdArray array = NdArray.of(new float[][]{{1f, 8f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.min(0);
        
        float[][] expected = {{1f, 5f, 3f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testMinAxis1() {
        // 测试按行找最小值
        NdArray array = NdArray.of(new float[][]{{3f, 1f, 2f}, {6f, 4f, 5f}});
        NdArray result = array.min(1);
        
        float[][] expected = {{1f}, {4f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    // =============================================================================
    // ArgMax测试
    // =============================================================================

    @Test
    public void testArgMaxAxis0() {
        // 测试按列找最大值索引
        NdArray array = NdArray.of(new float[][]{{1f, 8f, 3f}, {4f, 5f, 6f}});
        NdArray result = array.argMax(0);
        
        // 每列的最大值: [1,8,3]中4在位置1, [8,5]中8在位置0, [3,6]中6在位置1
        float[][] expected = {{1f, 0f, 1f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testArgMaxAxis1() {
        // 测试按行找最大值索引
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {6f, 5f, 4f}});
        NdArray result = array.argMax(1);
        
        // 第一行最大值3在位置2, 第二行最大值6在位置0
        float[][] expected = {{2f}, {0f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testArgMaxWithDuplicateMaximums() {
        // 测试有重复最大值时返回第一个
        NdArray array = NdArray.of(new float[][]{{5f, 5f, 3f}, {2f, 4f, 4f}});
        NdArray result = array.argMax(1);
        
        // 应该返回第一个最大值的索引
        assertEquals(0f, result.get(0, 0), DELTA);
        assertEquals(1f, result.get(1, 0), DELTA);
    }

    // =============================================================================
    // SumTo测试
    // =============================================================================

    @Test
    public void testSumToReduceFirstDim() {
        // 测试压缩第一维
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8f}});
        NdArray result = array.sumTo(Shape.of(1, 4));
        
        // [2,4] -> [1,4]: 沿第一维求和
        float[][] expected = {{6f, 8f, 10f, 12f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSumToReduceSecondDim() {
        // 测试压缩第二维
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8f}});
        NdArray result = array.sumTo(Shape.of(2, 1));
        
        // [2,4] -> [2,1]: 沿第二维求和
        float[][] expected = {{10f}, {26f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSumToSameShape() {
        // sumTo到相同形状应该返回原数组
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = array.sumTo(Shape.of(2, 2));
        
        assertArrayEquals(array.getMatrix(), result.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSumToIncompatibleShape() {
        // 不兼容的形状应该抛出异常
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        array.sumTo(Shape.of(2, 2));  // 列数不兼容
    }

    // =============================================================================
    // 3D数组聚合测试
    // =============================================================================

    @Test
    public void test3DSumAxis0() {
        // 测试3维数组按axis=0聚合
        NdArray array = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        });
        NdArray result = array.sum(0);
        
        // 沿第0维求和：对应位置相加，结果降维为 [2,2]
        assertEquals(6f, result.get(0, 0), DELTA);  // 1+5
        assertEquals(8f, result.get(0, 1), DELTA);  // 2+6
        assertEquals(10f, result.get(1, 0), DELTA); // 3+7
        assertEquals(12f, result.get(1, 1), DELTA); // 4+8
    }

    @Test
    public void test3DMeanAxis1() {
        // 测试3维数组按axis=1求平均
        NdArray array = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        });
        NdArray result = array.mean(1);
        
        // 沿第1维求平均，结果降维为 [2,2]
        assertEquals(2f, result.get(0, 0), DELTA);  // (1+3)/2
        assertEquals(3f, result.get(0, 1), DELTA);  // (2+4)/2
        assertEquals(6f, result.get(1, 0), DELTA);  // (5+7)/2
        assertEquals(7f, result.get(1, 1), DELTA);  // (6+8)/2
    }

    // =============================================================================
    // 特殊情况测试
    // =============================================================================

    @Test
    public void testAggregationOnSingleElement() {
        // 测试单元素数组的聚合
        NdArray single = NdArray.of(5f);
        
        assertEquals(5f, single.sum().getNumber().floatValue(), DELTA);
        assertEquals(5f, single.max(), DELTA);
    }

    @Test
    public void testAggregationWithInfinity() {
        // 测试包含无穷大的聚合
        NdArray withInf = NdArray.of(new float[][]{{1f, Float.POSITIVE_INFINITY}, {3f, 4f}});
        
        float maxVal = withInf.max();
        assertTrue(Float.isInfinite(maxVal));
    }

    @Test
    public void testAggregationLargeArray() {
        // 测试大数组聚合
        NdArray large = NdArray.ones(Shape.of(100, 100));
        NdArray result = large.sum();
        
        assertEquals(10000f, result.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testChainedAggregations() {
        // 测试链式聚合
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        
        // 先按列求和，再求总和
        NdArray colSum = array.sum(0);
        NdArray total = colSum.sum();
        
        assertEquals(45f, total.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testMeanVarianceRelation() {
        // 验证 E[X^2] - (E[X])^2 = Var(X)
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f, 4f, 5f}});
        
        NdArray mean = array.mean(1);
        NdArray squared = array.square();
        NdArray meanSquared = squared.mean(1);
        NdArray variance = array.var(1);
        
        // E[X^2] - (E[X])^2
        float expected = meanSquared.getNumber().floatValue() - 
                        mean.getNumber().floatValue() * mean.getNumber().floatValue();
        float actual = variance.getNumber().floatValue();
        
        assertEquals(expected, actual, DELTA);
    }
}
