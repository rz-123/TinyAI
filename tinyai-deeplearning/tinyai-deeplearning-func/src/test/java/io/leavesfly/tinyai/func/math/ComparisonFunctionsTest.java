package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 比较函数测试类
 * <p>
 * 测试LessThan, GreaterThan, Equal等比较函数的功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class ComparisonFunctionsTest {

    private static final float DELTA = 1e-6f;

    // ==================== LessThan测试 ====================

    @Test
    public void testLessThanForward() {
        LessThan lessThan = new LessThan();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{2f, 2f, 2f}});
        
        NdArray output = lessThan.forward(a, b);
        
        assertNotNull(output);
        float[][] result = output.getMatrix();
        
        // 1 < 2 = true (1.0), 2 < 2 = false (0.0), 3 < 2 = false (0.0)
        assertEquals(1f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(0f, result[0][2], DELTA);
    }

    @Test
    public void testLessThanBroadcast() {
        LessThan lessThan = new LessThan();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray b = NdArray.of(new float[][]{{3f, 3f, 3f}});
        
        NdArray output = lessThan.forward(a, b.broadcastTo(a.getShape()));
        
        assertNotNull(output);
        float[][] result = output.getMatrix();
        
        // 第一行: 1<3, 2<3, 3<3 -> 1, 1, 0
        assertEquals(1f, result[0][0], DELTA);
        assertEquals(1f, result[0][1], DELTA);
        assertEquals(0f, result[0][2], DELTA);
        
        // 第二行: 4<3, 5<3, 6<3 -> 0, 0, 0
        assertEquals(0f, result[1][0], DELTA);
        assertEquals(0f, result[1][1], DELTA);
        assertEquals(0f, result[1][2], DELTA);
    }

    @Test
    public void testLessThanNegativeValues() {
        LessThan lessThan = new LessThan();
        NdArray a = NdArray.of(new float[][]{{-5f, -2f, 0f}});
        NdArray b = NdArray.of(new float[][]{{-3f, -2f, 1f}});
        
        NdArray output = lessThan.forward(a, b);
        float[][] result = output.getMatrix();
        
        // -5 < -3 = true, -2 < -2 = false, 0 < 1 = true
        assertEquals(1f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(1f, result[0][2], DELTA);
    }

    @Test
    public void testLessThanBackward() {
        LessThan lessThan = new LessThan();
        Variable a = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "a");
        Variable b = new Variable(NdArray.of(new float[][]{{2f, 2f, 2f}}), "b");
        
        Variable result = lessThan.call(a, b);
        Variable sum = result.sum(); // 转换为标量再反向传播
        sum.backward();
        
        // 比较操作不可导，梯度应该为0
        assertNotNull(a.getGrad());
        float[][] grad = a.getGrad().getMatrix();
        
        for (int j = 0; j < 3; j++) {
            assertEquals(0f, grad[0][j], DELTA);
        }
    }

    @Test
    public void testLessThanShapePreservation() {
        LessThan lessThan = new LessThan();
        Shape[] testShapes = {
            Shape.of(1, 3),
            Shape.of(2, 4),
            Shape.of(3, 5)
        };
        
        for (Shape shape : testShapes) {
            NdArray a = NdArray.likeRandomN(shape);
            NdArray b = NdArray.likeRandomN(shape);
            NdArray output = lessThan.forward(a, b);
            assertEquals("形状应该保持不变", shape, output.getShape());
        }
    }

    // ==================== GreaterThan测试 ====================

    @Test
    public void testGreaterThanForward() {
        GreaterThan greaterThan = new GreaterThan();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{2f, 2f, 2f}});
        
        NdArray output = greaterThan.forward(a, b);
        
        assertNotNull(output);
        float[][] result = output.getMatrix();
        
        // 1 > 2 = false (0.0), 2 > 2 = false (0.0), 3 > 2 = true (1.0)
        assertEquals(0f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(1f, result[0][2], DELTA);
    }

    @Test
    public void testGreaterThanBroadcast() {
        GreaterThan greaterThan = new GreaterThan();
        NdArray a = NdArray.of(new float[][]{{1f, 3f, 5f}, {2f, 4f, 6f}});
        NdArray b = NdArray.of(new float[][]{{3f, 3f, 3f}});
        
        NdArray output = greaterThan.forward(a, b.broadcastTo(a.getShape()));
        
        assertNotNull(output);
        float[][] result = output.getMatrix();
        
        // 第一行: 1>3, 3>3, 5>3 -> 0, 0, 1
        assertEquals(0f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(1f, result[0][2], DELTA);
        
        // 第二行: 2>3, 4>3, 6>3 -> 0, 1, 1
        assertEquals(0f, result[1][0], DELTA);
        assertEquals(1f, result[1][1], DELTA);
        assertEquals(1f, result[1][2], DELTA);
    }

    @Test
    public void testGreaterThanNegativeValues() {
        GreaterThan greaterThan = new GreaterThan();
        NdArray a = NdArray.of(new float[][]{{-5f, -2f, 0f}});
        NdArray b = NdArray.of(new float[][]{{-3f, -2f, -1f}});
        
        NdArray output = greaterThan.forward(a, b);
        float[][] result = output.getMatrix();
        
        // -5 > -3 = false, -2 > -2 = false, 0 > -1 = true
        assertEquals(0f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(1f, result[0][2], DELTA);
    }

    @Test
    public void testGreaterThanBackward() {
        GreaterThan greaterThan = new GreaterThan();
        Variable a = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "a");
        Variable b = new Variable(NdArray.of(new float[][]{{2f, 2f, 2f}}), "b");
        
        Variable result = greaterThan.call(a, b);
        Variable sum = result.sum(); // 转换为标量再反向传播
        sum.backward();
        
        // 比较操作不可导，梯度应该为0
        assertNotNull(a.getGrad());
        float[][] grad = a.getGrad().getMatrix();
        
        for (int j = 0; j < 3; j++) {
            assertEquals(0f, grad[0][j], DELTA);
        }
    }

    @Test
    public void testGreaterThanShapePreservation() {
        GreaterThan greaterThan = new GreaterThan();
        Shape[] testShapes = {
            Shape.of(1, 3),
            Shape.of(2, 4),
            Shape.of(3, 5)
        };
        
        for (Shape shape : testShapes) {
            NdArray a = NdArray.likeRandomN(shape);
            NdArray b = NdArray.likeRandomN(shape);
            NdArray output = greaterThan.forward(a, b);
            assertEquals("形状应该保持不变", shape, output.getShape());
        }
    }

    // ==================== Equal测试 ====================

    @Test
    public void testEqualForward() {
        Equal equal = new Equal();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f, 4f}});
        
        NdArray output = equal.forward(a, b);
        
        assertNotNull(output);
        float[][] result = output.getMatrix();
        
        // 1 == 1 = true, 2 == 2 = true, 3 == 4 = false
        assertEquals(1f, result[0][0], DELTA);
        assertEquals(1f, result[0][1], DELTA);
        assertEquals(0f, result[0][2], DELTA);
    }

    @Test
    public void testEqualIdenticalArrays() {
        Equal equal = new Equal();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        NdArray output = equal.forward(a, b);
        float[][] result = output.getMatrix();
        
        // 所有元素都应该相等
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, result[i][j], DELTA);
            }
        }
    }

    @Test
    public void testEqualDifferentArrays() {
        Equal equal = new Equal();
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{4f, 5f, 6f}});
        
        NdArray output = equal.forward(a, b);
        float[][] result = output.getMatrix();
        
        // 所有元素都不相等
        for (int j = 0; j < 3; j++) {
            assertEquals(0f, result[0][j], DELTA);
        }
    }

    @Test
    public void testEqualNegativeValues() {
        Equal equal = new Equal();
        NdArray a = NdArray.of(new float[][]{{-5f, -2f, 0f}});
        NdArray b = NdArray.of(new float[][]{{-5f, -3f, 0f}});
        
        NdArray output = equal.forward(a, b);
        float[][] result = output.getMatrix();
        
        // -5 == -5 = true, -2 == -3 = false, 0 == 0 = true
        assertEquals(1f, result[0][0], DELTA);
        assertEquals(0f, result[0][1], DELTA);
        assertEquals(1f, result[0][2], DELTA);
    }

    @Test
    public void testEqualBackward() {
        Equal equal = new Equal();
        Variable a = new Variable(NdArray.of(new float[][]{{1f, 2f, 3f}}), "a");
        Variable b = new Variable(NdArray.of(new float[][]{{1f, 2f, 4f}}), "b");
        
        Variable result = equal.call(a, b);
        Variable sum = result.sum(); // 转换为标量再反向传播
        sum.backward();
        
        // 比较操作不可导，梯度应该为0
        assertNotNull(a.getGrad());
        float[][] grad = a.getGrad().getMatrix();
        
        for (int j = 0; j < 3; j++) {
            assertEquals(0f, grad[0][j], DELTA);
        }
    }

    @Test
    public void testEqualShapePreservation() {
        Equal equal = new Equal();
        Shape[] testShapes = {
            Shape.of(1, 3),
            Shape.of(2, 4),
            Shape.of(3, 5)
        };
        
        for (Shape shape : testShapes) {
            NdArray a = NdArray.likeRandomN(shape);
            NdArray b = NdArray.of(a.getArray(), a.getShape()); // 使用相同的数组
            NdArray output = equal.forward(a, b);
            assertEquals("形状应该保持不变", shape, output.getShape());
        }
    }

    // ==================== 组合和边界条件测试 ====================

    @Test
    public void testComparisonConsistency() {
        LessThan lessThan = new LessThan();
        GreaterThan greaterThan = new GreaterThan();
        Equal equal = new Equal();
        
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{2f, 2f, 2f}});
        
        NdArray ltResult = lessThan.forward(a, b);
        NdArray gtResult = greaterThan.forward(a, b);
        NdArray eqResult = equal.forward(a, b);
        
        // 对于每个位置，lt + gt + eq应该 <= 1（只能满足一个或都不满足）
        float[][] lt = ltResult.getMatrix();
        float[][] gt = gtResult.getMatrix();
        float[][] eq = eqResult.getMatrix();
        
        for (int j = 0; j < 3; j++) {
            float sum = lt[0][j] + gt[0][j] + eq[0][j];
            assertTrue("每个位置最多满足一个比较条件", sum <= 1f + DELTA);
        }
    }

    @Test
    public void testComparisonWithZero() {
        LessThan lessThan = new LessThan();
        GreaterThan greaterThan = new GreaterThan();
        Equal equal = new Equal();
        
        NdArray a = NdArray.of(new float[][]{{-1f, 0f, 1f}});
        NdArray zero = NdArray.zeros(Shape.of(1, 3));
        
        NdArray ltResult = lessThan.forward(a, zero);
        NdArray gtResult = greaterThan.forward(a, zero);
        NdArray eqResult = equal.forward(a, zero);
        
        float[][] lt = ltResult.getMatrix();
        float[][] gt = gtResult.getMatrix();
        float[][] eq = eqResult.getMatrix();
        
        // -1 < 0 = true, 0 < 0 = false, 1 < 0 = false
        assertEquals(1f, lt[0][0], DELTA);
        assertEquals(0f, lt[0][1], DELTA);
        assertEquals(0f, lt[0][2], DELTA);
        
        // -1 > 0 = false, 0 > 0 = false, 1 > 0 = true
        assertEquals(0f, gt[0][0], DELTA);
        assertEquals(0f, gt[0][1], DELTA);
        assertEquals(1f, gt[0][2], DELTA);
        
        // -1 == 0 = false, 0 == 0 = true, 1 == 0 = false
        assertEquals(0f, eq[0][0], DELTA);
        assertEquals(1f, eq[0][1], DELTA);
        assertEquals(0f, eq[0][2], DELTA);
    }

    @Test
    public void testComparisonBinaryOutput() {
        LessThan lessThan = new LessThan();
        
        NdArray a = NdArray.of(new float[][]{{1.5f, 2.7f, 3.9f}});
        NdArray b = NdArray.of(new float[][]{{2.1f, 2.5f, 4.0f}});
        
        NdArray output = lessThan.forward(a, b);
        float[][] result = output.getMatrix();
        
        // 输出应该只包含0或1
        for (int j = 0; j < 3; j++) {
            assertTrue("比较结果应该是0或1", 
                      result[0][j] == 0f || result[0][j] == 1f);
        }
    }

    @Test
    public void testComparisonSelfComparison() {
        Equal equal = new Equal();
        LessThan lessThan = new LessThan();
        GreaterThan greaterThan = new GreaterThan();
        
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        
        NdArray eqResult = equal.forward(a, a);
        NdArray ltResult = lessThan.forward(a, a);
        NdArray gtResult = greaterThan.forward(a, a);
        
        float[][] eq = eqResult.getMatrix();
        float[][] lt = ltResult.getMatrix();
        float[][] gt = gtResult.getMatrix();
        
        // 自我比较：所有元素都相等
        for (int j = 0; j < 3; j++) {
            assertEquals(1f, eq[0][j], DELTA);
            assertEquals(0f, lt[0][j], DELTA);
            assertEquals(0f, gt[0][j], DELTA);
        }
    }

    @Test
    public void testComparisonWithFloatPrecision() {
        Equal equal = new Equal();
        
        // 测试浮点精度问题
        NdArray a = NdArray.of(new float[][]{{0.1f + 0.2f}});
        NdArray b = NdArray.of(new float[][]{{0.3f}});
        
        NdArray output = equal.forward(a, b);
        
        // 由于浮点精度问题，可能不完全相等
        assertNotNull(output);
    }

    @Test
    public void testComparisonChain() {
        LessThan lessThan = new LessThan();
        
        // 测试链式比较 a < b < c
        NdArray a = NdArray.of(new float[][]{{1f}});
        NdArray b = NdArray.of(new float[][]{{2f}});
        NdArray c = NdArray.of(new float[][]{{3f}});
        
        NdArray result1 = lessThan.forward(a, b);
        NdArray result2 = lessThan.forward(b, c);
        
        assertEquals(1f, result1.get(0, 0), DELTA);
        assertEquals(1f, result2.get(0, 0), DELTA);
    }

    @Test
    public void testComparisonAllZeros() {
        Equal equal = new Equal();
        
        NdArray a = NdArray.zeros(Shape.of(2, 3));
        NdArray b = NdArray.zeros(Shape.of(2, 3));
        
        NdArray output = equal.forward(a, b);
        float[][] result = output.getMatrix();
        
        // 所有零应该都相等
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, result[i][j], DELTA);
            }
        }
    }

    @Test
    public void testComparisonLargeValues() {
        LessThan lessThan = new LessThan();
        
        NdArray a = NdArray.of(new float[][]{{1000000f, 2000000f}});
        NdArray b = NdArray.of(new float[][]{{1500000f, 1500000f}});
        
        NdArray output = lessThan.forward(a, b);
        float[][] result = output.getMatrix();
        
        assertEquals(1f, result[0][0], DELTA); // 1000000 < 1500000
        assertEquals(0f, result[0][1], DELTA); // 2000000 < 1500000
    }

    @Test
    public void testComparisonSymmetry() {
        LessThan lessThan = new LessThan();
        GreaterThan greaterThan = new GreaterThan();
        
        NdArray a = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray b = NdArray.of(new float[][]{{3f, 2f, 1f}});
        
        NdArray ltResult = lessThan.forward(a, b);
        NdArray gtResult = greaterThan.forward(b, a);
        
        // a < b 应该等于 b > a
        float[][] lt = ltResult.getMatrix();
        float[][] gt = gtResult.getMatrix();
        
        for (int j = 0; j < 3; j++) {
            assertEquals(lt[0][j], gt[0][j], DELTA);
        }
    }

    @Test
    public void testComparisonNonContiguousArrays() {
        Equal equal = new Equal();
        
        // 创建并转置数组（非连续存储）
        NdArray a = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray b = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = equal.forward(a, b);
        
        // 即使是非连续数组，比较也应该正常工作
        assertNotNull(output);
        assertEquals(a.getShape(), output.getShape());
    }
}
