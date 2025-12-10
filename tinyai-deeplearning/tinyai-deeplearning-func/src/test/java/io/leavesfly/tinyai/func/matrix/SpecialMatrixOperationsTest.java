package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 特殊矩阵操作函数的单元测试
 * 包括: MaskedFill, Tril, TopK, Where
 *
 * @author TinyAI
 */
public class SpecialMatrixOperationsTest {

    private static final float DELTA = 1e-4f;
    private boolean originalTrainMode;

    @Before
    public void setUp() {
        originalTrainMode = Config.train;
        Config.train = true;
    }

    @After
    public void tearDown() {
        Config.train = originalTrainMode;
    }

    // ==================== MaskedFill Tests ====================

    @Test
    public void testMaskedFillBasic() {
        MaskedFill maskedFill = new MaskedFill(-Float.MAX_VALUE);
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f}
        });
        
        NdArray mask = NdArray.of(new float[][]{
            {0f, 1f, 0f},
            {1f, 0f, 0f}
        });
        
        NdArray output = maskedFill.forward(input, mask);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 3), output.getShape());
        
        // mask=0的位置保持原值，mask=1的位置填充-Float.MAX_VALUE
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(-Float.MAX_VALUE, output.get(0, 1), DELTA);
        assertEquals(3f, output.get(0, 2), DELTA);
        assertEquals(-Float.MAX_VALUE, output.get(1, 0), DELTA);
        assertEquals(5f, output.get(1, 1), DELTA);
    }

    @Test
    public void testMaskedFillZero() {
        MaskedFill maskedFill = new MaskedFill(0f);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray mask = NdArray.of(new float[][]{{1f, 0f}, {0f, 1f}});
        
        NdArray output = maskedFill.forward(input, mask);
        
        assertEquals(0f, output.get(0, 0), DELTA);
        assertEquals(2f, output.get(0, 1), DELTA);
        assertEquals(3f, output.get(1, 0), DELTA);
        assertEquals(0f, output.get(1, 1), DELTA);
    }

    @Test
    public void testMaskedFillAllMasked() {
        MaskedFill maskedFill = new MaskedFill(99f);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray mask = NdArray.of(new float[][]{{1f, 1f}, {1f, 1f}});
        
        NdArray output = maskedFill.forward(input, mask);
        
        // 全部被填充为99
        assertEquals(99f, output.get(0, 0), DELTA);
        assertEquals(99f, output.get(0, 1), DELTA);
        assertEquals(99f, output.get(1, 0), DELTA);
        assertEquals(99f, output.get(1, 1), DELTA);
    }

    @Test
    public void testMaskedFillNoneMasked() {
        MaskedFill maskedFill = new MaskedFill(99f);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray mask = NdArray.of(new float[][]{{0f, 0f}, {0f, 0f}});
        
        NdArray output = maskedFill.forward(input, mask);
        
        // 全部保持原值
        assertArrayEquals(input.getMatrix(), output.getMatrix());
    }

    @Test
    public void testMaskedFillBackward() {
        MaskedFill maskedFill = new MaskedFill(0f);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray mask = NdArray.of(new float[][]{{1f, 0f}, {0f, 1f}});
        
        Variable x = new Variable(input, "x");
        Variable m = new Variable(mask, "m");
        Variable y = maskedFill.call(x, m);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // 被mask掉的位置梯度为0
        assertEquals(0f, x.getGrad().get(0, 0), DELTA);
        assertEquals(1f, x.getGrad().get(0, 1), DELTA);
        assertEquals(1f, x.getGrad().get(1, 0), DELTA);
        assertEquals(0f, x.getGrad().get(1, 1), DELTA);
    }

    // ==================== Tril Tests ====================

    @Test
    public void testTrilBasic() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        
        NdArray output = tril.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 3), output.getShape());
        
        // 下三角矩阵（包括对角线）
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(0, 2), DELTA);
        
        assertEquals(4f, output.get(1, 0), DELTA);
        assertEquals(5f, output.get(1, 1), DELTA);
        assertEquals(0f, output.get(1, 2), DELTA);
        
        assertEquals(7f, output.get(2, 0), DELTA);
        assertEquals(8f, output.get(2, 1), DELTA);
        assertEquals(9f, output.get(2, 2), DELTA);
    }

    @Test
    public void testTrilPositiveOffset() {
        Tril tril = new Tril(1);
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        
        NdArray output = tril.forward(input);
        
        // k=1: 主对角线上方一条也保留
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(2f, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(0, 2), DELTA);
        
        assertEquals(4f, output.get(1, 0), DELTA);
        assertEquals(5f, output.get(1, 1), DELTA);
        assertEquals(6f, output.get(1, 2), DELTA);
    }

    @Test
    public void testTrilNegativeOffset() {
        Tril tril = new Tril(-1);
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f},
            {4f, 5f, 6f},
            {7f, 8f, 9f}
        });
        
        NdArray output = tril.forward(input);
        
        // k=-1: 主对角线被置0
        assertEquals(0f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(0, 2), DELTA);
        
        assertEquals(4f, output.get(1, 0), DELTA);
        assertEquals(0f, output.get(1, 1), DELTA);
        assertEquals(0f, output.get(1, 2), DELTA);
        
        assertEquals(7f, output.get(2, 0), DELTA);
        assertEquals(8f, output.get(2, 1), DELTA);
        assertEquals(0f, output.get(2, 2), DELTA);
    }

    @Test
    public void testTrilRectangular() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 2f, 3f, 4f},
            {5f, 6f, 7f, 8f}
        });
        
        NdArray output = tril.forward(input);
        
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
        assertEquals(5f, output.get(1, 0), DELTA);
        assertEquals(6f, output.get(1, 1), DELTA);
        assertEquals(0f, output.get(1, 2), DELTA);
    }

    @Test
    public void testTrilBackward() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = tril.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // 上三角位置梯度为0，下三角为1
        assertEquals(1f, x.getGrad().get(0, 0), DELTA);
        assertEquals(0f, x.getGrad().get(0, 1), DELTA);
        assertEquals(1f, x.getGrad().get(1, 0), DELTA);
        assertEquals(1f, x.getGrad().get(1, 1), DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTril3DNotSupported() {
        Tril tril = new Tril();
        NdArray input = NdArray.of(Shape.of(2, 3, 3));
        tril.forward(input);
    }

    // ==================== TopK Tests ====================

    @Test(expected = UnsupportedOperationException.class)
    public void testTopKNotImplemented() {
        TopK topk = new TopK(2, 1, true, true);
        NdArray input = NdArray.of(new float[][]{{3f, 1f, 4f, 1f, 5f}});
        // TopK未完全实现，应该抛出异常
        topk.forward(input);
    }

    @Test
    public void testTopKRequireInputNum() {
        TopK topk = new TopK(2, 1, true, true);
        assertEquals(1, topk.requireInputNum());
    }

    @Test
    public void testTopKBasicAxis1() {
        TopK topk = new TopK(2, 1, true, true);
        
        // 输入: [[3, 1, 4, 1, 5],
        //       [2, 7, 1, 8, 2]]
        NdArray input = NdArray.of(new float[][]{
            {3f, 1f, 4f, 1f, 5f},
            {2f, 7f, 1f, 8f, 2f}
        });
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertNotNull(values);
        assertNotNull(indices);
        assertEquals(Shape.of(2, 2), values.getShape());
        assertEquals(Shape.of(2, 2), indices.getShape());
        
        // 第一行的top-2: 5, 4 (索引4, 2)
        assertEquals(5f, values.get(0, 0), DELTA);
        assertEquals(4f, values.get(0, 1), DELTA);
        assertEquals(4, (int)indices.get(0, 0));
        assertEquals(2, (int)indices.get(0, 1));
        
        // 第二行的top-2: 8, 7 (索引3, 1)
        assertEquals(8f, values.get(1, 0), DELTA);
        assertEquals(7f, values.get(1, 1), DELTA);
        assertEquals(3, (int)indices.get(1, 0));
        assertEquals(1, (int)indices.get(1, 1));
    }

    @Test
    public void testTopKBasicAxis0() {
        TopK topk = new TopK(2, 0, true, true);
        
        // 输入: [[1, 5, 3],
        //       [4, 2, 6]]
        NdArray input = NdArray.of(new float[][]{
            {1f, 5f, 3f},
            {4f, 2f, 6f}
        });
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertNotNull(values);
        assertNotNull(indices);
        assertEquals(Shape.of(2, 3), values.getShape());
        
        // 第一列: [1, 4] -> top-2: 4, 1
        assertEquals(4f, values.get(0, 0), DELTA);
        assertEquals(1f, values.get(1, 0), DELTA);
        
        // 第二列: [5, 2] -> top-2: 5, 2
        assertEquals(5f, values.get(0, 1), DELTA);
        assertEquals(2f, values.get(1, 1), DELTA);
        
        // 第三列: [3, 6] -> top-2: 6, 3
        assertEquals(6f, values.get(0, 2), DELTA);
        assertEquals(3f, values.get(1, 2), DELTA);
    }

    @Test
    public void testTopKSmallest() {
        TopK topk = new TopK(2, 1, false, true);
        
        // 输入: [[5, 2, 8, 1, 3]]
        NdArray input = NdArray.of(new float[][]{{5f, 2f, 8f, 1f, 3f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertNotNull(values);
        assertEquals(Shape.of(1, 2), values.getShape());
        
        // 最小的2个: 1, 2 (索引3, 1)
        assertEquals(1f, values.get(0, 0), DELTA);
        assertEquals(2f, values.get(0, 1), DELTA);
        assertEquals(3, (int)indices.get(0, 0));
        assertEquals(1, (int)indices.get(0, 1));
    }

    @Test
    public void testTopKNegativeAxis() {
        TopK topk = new TopK(3, -1, true, true);
        
        NdArray input = NdArray.of(new float[][]{
            {9f, 3f, 7f, 2f, 5f},
            {1f, 8f, 4f, 6f, 0f}
        });
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertEquals(Shape.of(2, 3), values.getShape());
        
        // 第一行top-3: 9, 7, 5
        assertEquals(9f, values.get(0, 0), DELTA);
        assertEquals(7f, values.get(0, 1), DELTA);
        assertEquals(5f, values.get(0, 2), DELTA);
        
        // 第二行top-3: 8, 6, 4
        assertEquals(8f, values.get(1, 0), DELTA);
        assertEquals(6f, values.get(1, 1), DELTA);
        assertEquals(4f, values.get(1, 2), DELTA);
    }

    @Test
    public void testTopKUnsorted() {
        TopK topk = new TopK(3, 1, true, false);
        
        NdArray input = NdArray.of(new float[][]{{5f, 1f, 8f, 3f, 9f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertNotNull(values);
        assertEquals(Shape.of(1, 3), values.getShape());
        
        // 值应该是9, 8, 5，但因为sorted=false，可能按索引排序
        // 索引应该是2, 4, 0 (按原始索引顺序)
        float[] actualValues = new float[3];
        int[] actualIndices = new int[3];
        for (int i = 0; i < 3; i++) {
            actualValues[i] = values.get(0, i);
            actualIndices[i] = (int)indices.get(0, i);
        }
        
        // 验证值都在top-3中
        assertTrue(actualValues[0] >= 5f);
        assertTrue(actualValues[1] >= 5f);
        assertTrue(actualValues[2] >= 5f);
    }

    @Test
    public void testTopKKGreaterThanSize() {
        TopK topk = new TopK(10, 1, true, true);
        
        NdArray input = NdArray.of(new float[][]{{3f, 1f, 4f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        // 实际返回的k应该是min(k, size) = 3
        assertEquals(Shape.of(1, 3), values.getShape());
        assertEquals(4f, values.get(0, 0), DELTA);
        assertEquals(3f, values.get(0, 1), DELTA);
        assertEquals(1f, values.get(0, 2), DELTA);
    }

    @Test
    public void testTopKSingleElement() {
        TopK topk = new TopK(1, 1, true, true);
        
        NdArray input = NdArray.of(new float[][]{{5f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertEquals(Shape.of(1, 1), values.getShape());
        assertEquals(5f, values.get(0, 0), DELTA);
        assertEquals(0, (int)indices.get(0, 0));
    }

    @Test
    public void testTopKWithNegativeValues() {
        TopK topk = new TopK(2, 1, true, true);
        
        NdArray input = NdArray.of(new float[][]{{-5f, -1f, -8f, -3f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        // 最大的2个负数: -1, -3
        assertEquals(-1f, values.get(0, 0), DELTA);
        assertEquals(-3f, values.get(0, 1), DELTA);
        assertEquals(1, (int)indices.get(0, 0));
        assertEquals(3, (int)indices.get(0, 1));
    }

    @Test
    public void testTopKWithDuplicateValues() {
        TopK topk = new TopK(3, 1, true, true);
        
        NdArray input = NdArray.of(new float[][]{{5f, 5f, 3f, 5f, 2f}});
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        // top-3应该是: 5, 5, 5
        assertEquals(5f, values.get(0, 0), DELTA);
        assertEquals(5f, values.get(0, 1), DELTA);
        assertEquals(5f, values.get(0, 2), DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testTopKInvalidAxis() {
        TopK topk = new TopK(2, 5, true, true);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        topk.forwardMulti(input);
    }

    @Test
    public void testTopKLargeMatrix() {
        TopK topk = new TopK(3, 1, true, true);
        
        NdArray input = NdArray.of(new float[][]{
            {9f, 2f, 7f, 4f, 1f, 8f, 3f},
            {5f, 0f, 6f, 3f, 8f, 2f, 1f}
        });
        
        NdArray[] result = topk.forwardMulti(input);
        NdArray values = result[0];
        NdArray indices = result[1];
        
        assertEquals(Shape.of(2, 3), values.getShape());
        
        // 第一行top-3: 9, 8, 7
        assertEquals(9f, values.get(0, 0), DELTA);
        assertEquals(8f, values.get(0, 1), DELTA);
        assertEquals(7f, values.get(0, 2), DELTA);
        
        // 第二行top-3: 8, 6, 5
        assertEquals(8f, values.get(1, 0), DELTA);
        assertEquals(6f, values.get(1, 1), DELTA);
        assertEquals(5f, values.get(1, 2), DELTA);
    }

    // ==================== Where Tests ====================

    @Test
    public void testWhereBasic() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{1f, 0f}, {0f, 1f}});
        NdArray x = NdArray.of(new float[][]{{10f, 20f}, {30f, 40f}});
        NdArray y = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = where.forward(condition, x, y);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 2), output.getShape());
        
        // condition=1选x，condition=0选y
        assertEquals(10f, output.get(0, 0), DELTA);
        assertEquals(2f, output.get(0, 1), DELTA);
        assertEquals(3f, output.get(1, 0), DELTA);
        assertEquals(40f, output.get(1, 1), DELTA);
    }

    @Test
    public void testWhereAllTrue() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{1f, 1f}, {1f, 1f}});
        NdArray x = NdArray.of(new float[][]{{10f, 20f}, {30f, 40f}});
        NdArray y = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = where.forward(condition, x, y);
        
        // 全部选x
        assertArrayEquals(x.getMatrix(), output.getMatrix());
    }

    @Test
    public void testWhereAllFalse() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{0f, 0f}, {0f, 0f}});
        NdArray x = NdArray.of(new float[][]{{10f, 20f}, {30f, 40f}});
        NdArray y = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        NdArray output = where.forward(condition, x, y);
        
        // 全部选y
        assertArrayEquals(y.getMatrix(), output.getMatrix());
    }

    @Test
    public void testWhereBackward() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{1f, 0f}});
        NdArray x = NdArray.of(new float[][]{{10f, 20f}});
        NdArray y = NdArray.of(new float[][]{{1f, 2f}});
        
        Variable cond = new Variable(condition, "cond");
        Variable xVar = new Variable(x, "x");
        Variable yVar = new Variable(y, "y");
        Variable output = where.call(cond, xVar, yVar);
        
        Variable sum = output.sum();
        sum.backward();
        
        assertNotNull(xVar.getGrad());
        assertNotNull(yVar.getGrad());
        
        // x的梯度：condition=1的位置为1，否则为0
        assertEquals(1f, xVar.getGrad().get(0, 0), DELTA);
        assertEquals(0f, xVar.getGrad().get(0, 1), DELTA);
        
        // y的梯度：condition=0的位置为1，否则为0
        assertEquals(0f, yVar.getGrad().get(0, 0), DELTA);
        assertEquals(1f, yVar.getGrad().get(0, 1), DELTA);
    }

    // ==================== Edge Cases ====================

    @Test
    public void testMaskedFillSingleElement() {
        MaskedFill maskedFill = new MaskedFill(0f);
        
        NdArray input = NdArray.of(new float[][]{{5f}});
        NdArray mask = NdArray.of(new float[][]{{1f}});
        
        NdArray output = maskedFill.forward(input, mask);
        
        assertEquals(0f, output.get(0, 0), DELTA);
    }

    @Test
    public void testTrilIdentityMatrix() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{
            {1f, 0f, 0f},
            {0f, 1f, 0f},
            {0f, 0f, 1f}
        });
        
        NdArray output = tril.forward(input);
        
        // 对角矩阵的下三角仍是对角矩阵
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(1f, output.get(1, 1), DELTA);
        assertEquals(1f, output.get(2, 2), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
    }

    @Test
    public void testWhereSameValues() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{1f, 0f}});
        NdArray x = NdArray.of(new float[][]{{5f, 5f}});
        NdArray y = NdArray.of(new float[][]{{5f, 5f}});
        
        NdArray output = where.forward(condition, x, y);
        
        // x和y相同，输出也应该相同
        assertEquals(5f, output.get(0, 0), DELTA);
        assertEquals(5f, output.get(0, 1), DELTA);
    }

    @Test
    public void testTrilSingleRow() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray output = tril.forward(input);
        
        // 单行矩阵的下三角只保留第一个元素
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(0, 2), DELTA);
    }

    @Test
    public void testTrilSingleColumn() {
        Tril tril = new Tril();
        
        NdArray input = NdArray.of(new float[][]{{1f}, {2f}, {3f}});
        NdArray output = tril.forward(input);
        
        // 单列矩阵的下三角保留全部
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(2f, output.get(1, 0), DELTA);
        assertEquals(3f, output.get(2, 0), DELTA);
    }

    @Test
    public void testRequireInputNum() {
        assertEquals(2, new MaskedFill(0f).requireInputNum());
        assertEquals(1, new Tril().requireInputNum());
        assertEquals(1, new Tril(1).requireInputNum());
        assertEquals(1, new TopK(2, 1, true, true).requireInputNum());
        assertEquals(3, new Where().requireInputNum());
    }

    @Test
    public void testMaskedFillNegativeValues() {
        MaskedFill maskedFill = new MaskedFill(-1f);
        
        NdArray input = NdArray.of(new float[][]{{-5f, -10f}, {15f, 20f}});
        NdArray mask = NdArray.of(new float[][]{{1f, 0f}, {0f, 1f}});
        
        NdArray output = maskedFill.forward(input, mask);
        
        assertEquals(-1f, output.get(0, 0), DELTA);
        assertEquals(-10f, output.get(0, 1), DELTA);
        assertEquals(15f, output.get(1, 0), DELTA);
        assertEquals(-1f, output.get(1, 1), DELTA);
    }

    @Test
    public void testWhereWithNegativeValues() {
        Where where = new Where();
        
        NdArray condition = NdArray.of(new float[][]{{1f, 0f}});
        NdArray x = NdArray.of(new float[][]{{-10f, -20f}});
        NdArray y = NdArray.of(new float[][]{{10f, 20f}});
        
        NdArray output = where.forward(condition, x, y);
        
        assertEquals(-10f, output.get(0, 0), DELTA);
        assertEquals(20f, output.get(0, 1), DELTA);
    }
}
