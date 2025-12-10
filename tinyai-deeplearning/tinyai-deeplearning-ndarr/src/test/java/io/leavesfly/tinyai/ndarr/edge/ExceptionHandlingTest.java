package io.leavesfly.tinyai.ndarr.edge;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 异常处理测试
 * 
 * 测试各种异常情况的处理，包括：
 * - 参数验证异常
 * - 形状不匹配异常
 * - 索引越界异常
 * - 数学运算异常
 *
 * @author TinyAI
 */
public class ExceptionHandlingTest {

    // =============================================================================
    // 创建时的异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testCreateWithMismatchedDataAndShape() {
        // 数据长度与形状不匹配
        float[] data = {1f, 2f, 3f};
        NdArray.of(data, Shape.of(2, 2));
    }

    @Test(expected = IllegalArgumentException.class)
    public void testCreateWithInvalidArrayType() {
        // 不支持的数组类型
        String[] invalid = {"not", "a", "number"};
        NdArray.of(invalid);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinSpaceWithZeroPoints() {
        // linSpace点数为0
        NdArray.linSpace(0f, 10f, 0);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testLinSpaceWithNegativePoints() {
        // linSpace点数为负数
        NdArray.linSpace(0f, 10f, -5);
    }

    // =============================================================================
    // 形状操作异常测试
    // =============================================================================

    @Test(expected = RuntimeException.class)
    public void testReshapeWithIncompatibleSize() {
        // reshape到不兼容的大小
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f, 4f, 5f, 6f});
        array.reshape(Shape.of(2, 2)); // 需要4个元素，但有6个
    }

    @Test(expected = RuntimeException.class)
    public void testReshapeToLargerSize() {
        // reshape到更大的大小
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f});
        array.reshape(Shape.of(2, 3)); // 需要6个元素，只有3个
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetShapeWithMismatchedSize() {
        // 设置不匹配的形状
        NdArray array = NdArray.of(new float[]{1f, 2f, 3f, 4f});
        array.setShape(Shape.of(2, 3)); // 需要6个元素，只有4个
    }

    // =============================================================================
    // 索引越界异常测试
    // =============================================================================

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWithRowIndexOutOfBounds() {
        // 行索引越界
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get(2, 0); // 最大行索引是1
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWithColumnIndexOutOfBounds() {
        // 列索引越界
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get(0, 2); // 最大列索引是1
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWithIndexOutOfBounds() {
        // set索引越界
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.set(99f, 3, 3);
    }

    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWithNegativeIndex() {
        // 负索引
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get(-1, 0);
    }

    // =============================================================================
    // 运算异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testAddWithIncompatibleShapes() {
        // 形状不兼容的加法
        NdArray a = NdArray.of(new float[][]{{1f, 2f}});
        NdArray b = NdArray.of(new float[][]{{1f}, {2f}});
        a.add(b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSubWithIncompatibleShapes() {
        // 形状不兼容的减法
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f}});
        a.sub(b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMulWithIncompatibleShapes() {
        // 形状不兼容的乘法
        NdArray a = NdArray.of(new float[][]{{1f, 2f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f, 3f}});
        a.mul(b);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testDivWithIncompatibleShapes() {
        // 形状不兼容的除法
        NdArray a = NdArray.of(new float[][]{{1f, 2f}});
        NdArray b = NdArray.of(new float[]{1f, 2f, 3f});
        a.div(b);
    }

    @Test(expected = ArithmeticException.class)
    public void testDivideByZero() {
        // 除以零
        NdArray a = NdArray.of(new float[][]{{1f, 2f}});
        NdArray b = NdArray.of(new float[][]{{0f, 1f}});
        a.div(b);
    }

    @Test(expected = ArithmeticException.class)
    public void testDivideScalarByZero() {
        // 标量除以零
        NdArray a = NdArray.of(new float[][]{{1f, 2f}});
        a.divNum(0f);
    }

    // =============================================================================
    // 数学函数异常测试
    // =============================================================================

    @Test(expected = ArithmeticException.class)
    public void testLogOfZero() {
        // log(0)应该抛出异常
        NdArray zeros = NdArray.zeros(Shape.of(2, 2));
        zeros.log();
    }

    @Test(expected = ArithmeticException.class)
    public void testLogOfNegative() {
        // log(负数)应该抛出异常
        NdArray negative = NdArray.of(new float[][]{{-1f, -2f}});
        negative.log();
    }

    // =============================================================================
    // 矩阵运算异常测试
    // =============================================================================

    @Test(expected = RuntimeException.class)
    public void testDotWithIncompatibleDimensions() {
        // 矩阵乘法维度不匹配
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}}); // 1x3
        NdArray b = NdArray.of(new float[][]{{1f, 2f}}); // 1x2
        a.dot(b); // 1x3 × 1x2 不兼容
    }

    @Test(expected = RuntimeException.class)
    public void testDotWithIncompatibleMatrices() {
        // 矩阵乘法尺寸不匹配
        NdArray a = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}}); // 2x2
        NdArray b = NdArray.of(new float[][]{{1f, 2f, 3f}}); // 1x3
        a.dot(b); // 2x2 × 1x3 不兼容
    }

    // =============================================================================
    // 转置异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testTransposeWithInvalidOrder() {
        // 转置顺序无效
        NdArray array = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}}
        });
        array.transpose(0, 1, 5); // 维度5不存在
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTransposeWithDuplicateDimensions() {
        // 转置顺序重复
        NdArray array = NdArray.of(new float[][][]{
            {{1f, 2f}, {3f, 4f}}
        });
        array.transpose(0, 1, 1); // 维度1重复
    }

    // =============================================================================
    // 广播异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastToIncompatibleShape() {
        // 广播到不兼容的形状
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.broadcastTo(Shape.of(3, 3)); // [2,2]不能广播到[3,3]
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBroadcastToSmallerShape() {
        // 广播到更小的形状
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}});
        array.broadcastTo(Shape.of(1, 2)); // 不能缩小维度
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSumToIncompatibleShape() {
        // sumTo到不兼容的形状
        NdArray array = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        array.sumTo(Shape.of(2, 2)); // 列数不兼容
    }

    // =============================================================================
    // 聚合操作异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testSumWithInvalidAxis() {
        // 无效的axis参数
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.sum(5); // axis=5超出范围
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMeanWithNegativeAxis() {
        // 负的axis参数（某些实现可能支持，某些不支持）
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.mean(-3); // axis=-3超出范围
    }

    @Test(expected = IllegalArgumentException.class)
    public void testMaxWithInvalidAxis() {
        // 无效的axis参数
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.max(2); // 只有0和1两个轴
    }

    // =============================================================================
    // 类型转换异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testGet3dArrayFrom2D() {
        // 2D数组不能转为3D
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get3dArray();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGet4dArrayFrom3D() {
        // 3D数组不能转为4D
        NdArray array = NdArray.of(new float[][][]{{{1f, 2f}}});
        array.get4dArray();
    }

    @Test(expected = IllegalArgumentException.class)
    public void testGetMatrixFromHighDim() {
        // 高维数组不能直接转为矩阵
        NdArray array = NdArray.zeros(Shape.of(2, 3, 4));
        array.getMatrix();
    }

    // =============================================================================
    // 切片异常测试
    // =============================================================================

    // =============================================================================
    // Clip异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testClipWithInvalidRange() {
        // clip的最小值大于最大值
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.clip(10f, 5f); // min > max
    }

    // =============================================================================
    // 多维度异常测试
    // =============================================================================

    @Test(expected = IllegalArgumentException.class)
    public void testGetWithWrongDimensionCount() {
        // get方法维度数量不匹配
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.get(0, 0, 0); // 2D数组用3个索引
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSetWithWrongDimensionCount() {
        // set方法维度数量不匹配
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        array.set(99f, 0); // 2D数组只用1个索引
    }

    // =============================================================================
    // 空值和null测试
    // =============================================================================

    @Test(expected = NullPointerException.class)
    public void testAddWithNull() {
        // 与null相加
        NdArray array = NdArray.of(new float[][]{{1f, 2f}});
        array.add(null);
    }

    @Test(expected = NullPointerException.class)
    public void testDotWithNull() {
        // 与null进行矩阵乘法
        NdArray array = NdArray.of(new float[][]{{1f, 2f}});
        array.dot(null);
    }

    // =============================================================================
    // 综合异常场景测试
    // =============================================================================

    @Test
    public void testMultipleExceptionsInSequence() {
        // 测试连续多个操作中的异常处理
        NdArray array = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        // 第一个异常
        try {
            array.get(5, 5);
            fail("应该抛出IndexOutOfBoundsException");
        } catch (IndexOutOfBoundsException e) {
            // 预期异常
        }
        
        // 第二个异常
        try {
            array.reshape(Shape.of(3, 3));
            fail("应该抛出RuntimeException");
        } catch (RuntimeException e) {
            // 预期异常
        }
        
        // 正常操作应该仍然工作
        NdArray result = array.add(array);
        assertEquals(2f, result.get(0, 0), 1e-6f);
    }
}
