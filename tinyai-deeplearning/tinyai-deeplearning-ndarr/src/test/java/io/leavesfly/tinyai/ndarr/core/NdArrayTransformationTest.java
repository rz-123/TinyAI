package io.leavesfly.tinyai.ndarr.core;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * NdArray形状变换测试
 * 
 * 测试各种形状变换操作，包括：
 * - reshape重塑
 * - transpose转置
 * - flatten展平
 * - 切片操作
 *
 * @author TinyAI
 */
public class NdArrayTransformationTest {

    private static final float DELTA = 1e-6f;

    // =============================================================================
    // Reshape测试
    // =============================================================================

    @Test
    public void testReshape() {
        // 测试基本reshape
        NdArray input = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        NdArray result = input.reshape(Shape.of(2, 3));
        
        assertEquals(Shape.of(2, 3), result.getShape());
        float[][] expected = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testReshapeTo3D() {
        // 测试reshape到3维
        NdArray input = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f});
        NdArray result = input.reshape(Shape.of(2, 2, 2));
        
        assertEquals(Shape.of(2, 2, 2), result.getShape());
        assertEquals(1f, result.get(0, 0, 0), DELTA);
        assertEquals(8f, result.get(1, 1, 1), DELTA);
    }

    @Test
    public void testReshapePreservesData() {
        // 测试reshape不改变数据
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.reshape(Shape.of(3, 2));
        
        assertArrayEquals(input.getArray(), result.getArray(), DELTA);
    }

    @Test(expected = RuntimeException.class)
    public void testReshapeWithIncompatibleSize() {
        // 测试大小不匹配的reshape
        NdArray input = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        input.reshape(Shape.of(2, 2)); // 需要4个元素，但有6个
    }

    @Test
    public void testReshapeToSameShape() {
        // 测试reshape到相同形状
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.reshape(Shape.of(2, 2));
        
        assertArrayEquals(input.getMatrix(), result.getMatrix());
    }

    // =============================================================================
    // Transpose测试
    // =============================================================================

    @Test
    public void testTranspose2D() {
        // 测试2维转置
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.transpose();
        
        assertEquals(Shape.of(3, 2), result.getShape());
        float[][] expected = {{1f, 4f}, {2f, 5f}, {3f, 6f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testTransposeSquareMatrix() {
        // 测试方阵转置
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        NdArray result = input.transpose();
        
        assertEquals(Shape.of(3, 3), result.getShape());
        float[][] expected = {{1f, 4f, 7f}, {2f, 5f, 8f}, {3f, 6f, 9f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testDoubleTranspose() {
        // 两次转置应该回到原始状态
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.transpose().transpose();
        
        assertArrayEquals(input.getMatrix(), result.getMatrix());
    }

    @Test
    public void testTransposeWithOrder3D() {
        // 测试3维指定顺序转置
        NdArray input = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        });
        
        // 转置顺序: (0,1,2) -> (2,1,0)
        NdArray result = input.transpose(2, 1, 0);
        
        assertEquals(Shape.of(2, 2, 2), result.getShape());
        assertEquals(1f, result.get(0, 0, 0), DELTA);
        assertEquals(5f, result.get(0, 0, 1), DELTA);
    }

    // =============================================================================
    // Flatten测试
    // =============================================================================

    @Test
    public void testFlatten2D() {
        // 测试2维展平
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray result = input.flatten();
        
        assertEquals(Shape.of(1, 6), result.getShape());
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f};
        assertArrayEquals(expected, result.getArray(), DELTA);
    }

    @Test
    public void testFlatten3D() {
        // 测试3维展平
        NdArray input = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        });
        NdArray result = input.flatten();
        
        assertEquals(Shape.of(1, 8), result.getShape());
        float[] expected = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f};
        assertArrayEquals(expected, result.getArray(), DELTA);
    }

    @Test
    public void testFlattenAlreadyFlat() {
        // 已经是一维的数组flatten后应该不变
        NdArray input = NdArray.of(new float[]{1f, 2f, 3f, 4f});
        NdArray result = input.flatten();
        
        assertEquals(Shape.of(1, 4), result.getShape());
        assertArrayEquals(input.getArray(), result.getArray(), DELTA);
    }

    // =============================================================================
    // 切片操作测试
    // =============================================================================

    @Test
    public void testGetItemSingleElement() {
        // 测试获取单个元素
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        NdArray result = input.getItem(new int[]{1}, new int[]{2});
        
        assertEquals(6f, result.getNumber().floatValue(), DELTA);
    }

    @Test
    public void testGetItemRowSlice() {
        // 测试行切片
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        NdArray result = input.getItem(new int[]{0, 2}, null);
        
        assertEquals(Shape.of(2, 3), result.getShape());
        float[][] expected = {{1f, 2f, 3f}, {7f, 8f, 9f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testGetItemColumnSlice() {
        // 测试列切片
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        NdArray result = input.getItem(null, new int[]{0, 2});
        
        assertEquals(Shape.of(3, 2), result.getShape());
        float[][] expected = {{1f, 3f}, {4f, 6f}, {7f, 9f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testGetItemBothSlices() {
        // 测试同时行列切片
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f, 4f}, {5f, 6f, 7f, 8f}, {9f, 10f, 11f, 12f}});
        NdArray result = input.getItem(new int[]{0, 2}, new int[]{1, 3});
        
        assertEquals(Shape.of(1, 2), result.getShape());
    }

    @Test
    public void testSubNdArray() {
        // 测试子数组提取
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f, 4f},
            {5f, 6f, 7f, 8f},
            {9f, 10f, 11f, 12f}
        });
        NdArray result = input.subNdArray(1, 3, 1, 3);
        
        assertEquals(Shape.of(2, 2), result.getShape());
        float[][] expected = {{6f, 7f}, {10f, 11f}};
        assertArrayEquals(expected, result.getMatrix());
    }

    @Test
    public void testSubNdArrayFullRange() {
        // 测试完整范围的子数组（应该等于原数组）
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray result = input.subNdArray(0, 2, 0, 2);
        
        assertArrayEquals(input.getMatrix(), result.getMatrix());
    }

    // =============================================================================
    // SetItem测试
    // =============================================================================

    @Test
    public void testSetItem() {
        // 测试设置特定位置的值
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        float[] newData = {99f, 88f};
        array.setItem(new int[]{0, 2}, new int[]{1, 1}, newData);
        
        assertEquals(99f, array.get(0, 1), DELTA);
        assertEquals(88f, array.get(2, 1), DELTA);
        // 其他位置不变
        assertEquals(1f, array.get(0, 0), DELTA);
        assertEquals(9f, array.get(2, 2), DELTA);
    }

    // =============================================================================
    // Get/Set测试
    // =============================================================================

    @Test
    public void testGetAndSet() {
        // 测试get和set方法
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        assertEquals(3f, array.get(1, 0), DELTA);
        
        array.set(99f, 1, 0);
        assertEquals(99f, array.get(1, 0), DELTA);
    }

    @Test
    public void testGet3D() {
        // 测试3维数组的get
        NdArray array = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        });
        
        assertEquals(1f, array.get(0, 0, 0), DELTA);
        assertEquals(8f, array.get(1, 1, 1), DELTA);
        assertEquals(6f, array.get(1, 0, 1), DELTA);
    }

    @Test
    public void testSet3D() {
        // 测试3维数组的set
        NdArray array = NdArray.zeros(Shape.of(2, 2, 2));
        
        array.set(5f, 0, 0, 0);
        array.set(10f, 1, 1, 1);
        
        assertEquals(5f, array.get(0, 0, 0), DELTA);
        assertEquals(10f, array.get(1, 1, 1), DELTA);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWithInvalidIndices() {
        // 测试索引越界
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get(2, 0);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWithInvalidIndices() {
        // 测试设置时索引越界
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.set(99f, 0, 3);
    }

    // =============================================================================
    // 数组转换测试
    // =============================================================================

    @Test
    public void testGetMatrix() {
        // 测试获取矩阵
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        float[][] matrix = array.getMatrix();
        
        assertEquals(2, matrix.length);
        assertEquals(2, matrix[0].length);
        assertEquals(1f, matrix[0][0], DELTA);
        assertEquals(4f, matrix[1][1], DELTA);
    }

    @Test
    public void testGet3dArray() {
        // 测试获取3维数组
        float[][][] original = {
            {{1f, 2f}, {3f, 4f}},
            {{5f, 6f}, {7f, 8f}}
        };
        NdArray array = NdArray.of(original);
        float[][][] result = array.get3dArray();
        
        assertArrayEquals(original, result);
    }

    @Test
    public void testGet4dArray() {
        // 测试获取4维数组
        float[][][][] original = {
            {{{1f, 2f}, {3f, 4f}}, {{5f, 6f}, {7f, 8f}}}
        };
        NdArray array = NdArray.of(original);
        float[][][][] result = array.get4dArray();
        
        assertArrayEquals(original, result);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGetMatrixFrom3D() {
        // 3维数组不能直接转为矩阵
        NdArray array = NdArray.zeros(Shape.of(2, 3, 4));
        array.getMatrix();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGet3dArrayFrom2D() {
        // 2维数组不能转为3维数组
        NdArray array = NdArray.zeros(Shape.of(2, 3));
        array.get3dArray();
    }

    // =============================================================================
    // AddTo和AddAt测试
    // =============================================================================

    @Test
    public void testAddTo() {
        // 测试在指定位置累加
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray delta = NdArray.of(new float[][]{{10f, 20f}});
        
        array.addTo(0, 1, delta);
        
        assertEquals(1f, array.get(0, 0), DELTA);
        assertEquals(12f, array.get(0, 1), DELTA);  // 2 + 10
        assertEquals(23f, array.get(0, 2), DELTA);  // 3 + 20
        assertEquals(4f, array.get(1, 0), DELTA);
    }

    @Test
    public void testAddAt() {
        // 测试在指定索引位置累加
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}, {7f, 8f, 9f}});
        NdArray delta = NdArray.of(new float[][]{{10f}, {20f}});
        
        NdArray result = array.addAt(new int[]{0, 2}, new int[]{1, 1}, delta);
        
        assertEquals(12f, result.get(0, 1), DELTA);  // 2 + 10
        assertEquals(28f, result.get(2, 1), DELTA);  // 8 + 20
        // 其他位置不变
        assertEquals(1f, result.get(0, 0), DELTA);
        assertEquals(9f, result.get(2, 2), DELTA);
    }

    // =============================================================================
    // 形状操作测试
    // =============================================================================

    @Test
    public void testGetShape() {
        // 测试获取形状
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        Shape shape = array.getShape();
        
        assertEquals(Shape.of(2, 3), shape);
        assertEquals(2, shape.getRow());
        assertEquals(3, shape.getColumn());
    }

    @Test
    public void testSetShape() {
        // 测试设置形状（大小必须匹配）
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        array.setShape(Shape.of(2, 3));
        
        assertEquals(Shape.of(2, 3), array.getShape());
        float[][] expected = {{1f, 2f, 3f}, {4f, 5f, 6f}};
        assertArrayEquals(expected, array.getMatrix());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetShapeWithMismatchedSize() {
        // 设置不匹配大小的形状应该抛出异常
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        array.setShape(Shape.of(2, 2));  // 需要4个元素，但有6个
    }
}
