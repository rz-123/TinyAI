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
 * 形状操作函数的单元测试
 * 包括: Permute, Squeeze, Unsqueeze, Expand, Repeat
 *
 * @author TinyAI
 */
public class ShapeManipulationTest {

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

    // ==================== Permute Tests ====================

    @Test
    public void testPermuteForward2D() {
        Permute permute = new Permute(1, 0);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray output = permute.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 2), output.getShape());
        
        // 转置后: [1,4; 2,5; 3,6]
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(4f, output.get(0, 1), DELTA);
        assertEquals(2f, output.get(1, 0), DELTA);
        assertEquals(5f, output.get(1, 1), DELTA);
    }

    @Test
    public void testPermuteForward3D() {
        Permute permute = new Permute(2, 0, 1);
        
        // [2, 3, 4] -> [4, 2, 3]
        NdArray input = NdArray.of(Shape.of(2, 3, 4));
        NdArray output = permute.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(4, 2, 3), output.getShape());
    }

    @Test
    public void testPermuteBackward() {
        Permute permute = new Permute(1, 0);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = permute.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPermuteInvalidOrder() {
        Permute permute = new Permute(0, 0); // 重复索引
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        permute.forward(input);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testPermuteOrderLengthMismatch() {
        Permute permute = new Permute(0, 1, 2); // 3维顺序用于2维张量
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        permute.forward(input);
    }

    // ==================== Squeeze Tests ====================

    @Test
    public void testSqueezeAllDims() {
        Squeeze squeeze = new Squeeze();
        
        NdArray input = NdArray.of(Shape.of(1, 3, 1, 2, 1));
        NdArray output = squeeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 2), output.getShape());
    }

    @Test
    public void testSqueezeSpecificDim() {
        Squeeze squeeze = new Squeeze(0);
        
        NdArray input = NdArray.of(Shape.of(1, 3, 2));
        NdArray output = squeeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 2), output.getShape());
    }

    @Test
    public void testSqueezeNegativeDim() {
        Squeeze squeeze = new Squeeze(-1);
        
        NdArray input = NdArray.of(Shape.of(3, 2, 1));
        NdArray output = squeeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 2), output.getShape());
    }

    @Test
    public void testSqueezeBackward() {
        Squeeze squeeze = new Squeeze(0);
        
        NdArray input = NdArray.of(Shape.of(1, 2, 3));
        Variable x = new Variable(input, "x");
        Variable y = squeeze.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testSqueezeNonSingletonDim() {
        Squeeze squeeze = new Squeeze(0);
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}}); // 第0维大小为2
        squeeze.forward(input);
    }

    @Test
    public void testSqueezePreservesSingleDim() {
        Squeeze squeeze = new Squeeze();
        
        NdArray input = NdArray.of(Shape.of(1, 1, 1));
        NdArray output = squeeze.forward(input);
        
        // 至少保留一个维度
        assertEquals(1, output.getShape().size());
    }

    // ==================== Unsqueeze Tests ====================

    @Test
    public void testUnsqueezeAtBeginning() {
        Unsqueeze unsqueeze = new Unsqueeze(0);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = unsqueeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(1, 2, 2), output.getShape());
    }

    @Test
    public void testUnsqueezeAtMiddle() {
        Unsqueeze unsqueeze = new Unsqueeze(1);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = unsqueeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 1, 2), output.getShape());
    }

    @Test
    public void testUnsqueezeAtEnd() {
        Unsqueeze unsqueeze = new Unsqueeze(2);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = unsqueeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 2, 1), output.getShape());
    }

    @Test
    public void testUnsqueezeNegativeIndex() {
        Unsqueeze unsqueeze = new Unsqueeze(-1);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = unsqueeze.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 2, 1), output.getShape());
    }

    @Test
    public void testUnsqueezeBackward() {
        Unsqueeze unsqueeze = new Unsqueeze(0);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = unsqueeze.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    // ==================== Expand Tests ====================

    @Test
    public void testExpandSingleDimension() {
        Expand expand = new Expand(Shape.of(3, 2));
        
        NdArray input = NdArray.of(Shape.of(1, 2));
        NdArray output = expand.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 2), output.getShape());
    }

    @Test
    public void testExpandMultipleDimensions() {
        Expand expand = new Expand(Shape.of(3, 4, 5));
        
        NdArray input = NdArray.of(Shape.of(1, 1, 5));
        NdArray output = expand.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 4, 5), output.getShape());
    }

    @Test
    public void testExpandBackward() {
        Expand expand = new Expand(Shape.of(3, 2));
        
        NdArray input = NdArray.of(Shape.of(1, 2));
        Variable x = new Variable(input, "x");
        Variable y = expand.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExpandInvalidDimension() {
        Expand expand = new Expand(Shape.of(3, 2));
        NdArray input = NdArray.of(Shape.of(2, 2)); // 第0维不是1，不能扩展
        expand.forward(input);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testExpandDimensionMismatch() {
        Expand expand = new Expand(Shape.of(3, 4, 5));
        NdArray input = NdArray.of(Shape.of(1, 2)); // 维度数量不匹配
        expand.forward(input);
    }

    // ==================== Repeat Tests ====================

    @Test
    public void testRepeatSingleDimension() {
        Repeat repeat = new Repeat(2, 1);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = repeat.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(4, 2), output.getShape());
        
        // 验证重复
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(1f, output.get(2, 0), DELTA); // 重复的第一行
    }

    @Test
    public void testRepeatMultipleDimensions() {
        Repeat repeat = new Repeat(2, 3);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        NdArray output = repeat.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 6), output.getShape());
        
        // 验证形状正确即可
    }

    @Test
    public void testRepeatBackward() {
        Repeat repeat = new Repeat(2, 2);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        Variable x = new Variable(input, "x");
        Variable y = repeat.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // 重复的梯度应该累加
        // 每个元素被重复了2x2=4次，所以梯度应该是4
        assertEquals(4f, x.getGrad().get(0, 0), DELTA);
        assertEquals(4f, x.getGrad().get(0, 1), DELTA);
    }

    @Test(expected = IllegalArgumentException.class)
    public void testRepeatLengthMismatch() {
        Repeat repeat = new Repeat(2, 3, 4); // 3个重复次数
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}}); // 2维
        repeat.forward(input);
    }

    // ==================== Combined Operations ====================

    @Test
    public void testSqueezeUnsqueeze() {
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        Unsqueeze unsqueeze = new Unsqueeze(0);
        Squeeze squeeze = new Squeeze(0);
        
        Variable x = new Variable(input, "x");
        Variable y1 = unsqueeze.call(x);
        Variable y2 = squeeze.call(y1);
        
        // 形状应该恢复原样
        assertEquals(x.getValue().getShape(), y2.getValue().getShape());
        
        Variable sum = y2.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
    }

    @Test
    public void testPermuteInverse() {
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        
        Permute permute1 = new Permute(1, 0);
        Permute permute2 = new Permute(1, 0);
        
        Variable x = new Variable(input, "x");
        Variable y1 = permute1.call(x);
        Variable y2 = permute2.call(y1);
        
        // 两次转置应该恢复原样
        assertEquals(x.getValue().getShape(), y2.getValue().getShape());
        assertArrayEquals(input.getMatrix(), y2.getValue().getMatrix());
    }

    @Test
    public void testExpandAndRepeatDifference() {
        // Expand和Repeat的区别：Expand是view（不复制），Repeat是复制
        NdArray input = NdArray.of(Shape.of(1, 2));
        
        // Expand: 扩展大小为1的维度
        Expand expand = new Expand(Shape.of(3, 2));
        NdArray expandOut = expand.forward(input);
        assertEquals(Shape.of(3, 2), expandOut.getShape());
        
        // Repeat: 沿各维度重复
        Repeat repeat = new Repeat(3, 1);
        NdArray repeatOut = repeat.forward(input);
        assertEquals(Shape.of(3, 2), repeatOut.getShape());
    }

    // ==================== Edge Cases ====================

    // Permute转置测试：验证二维矩阵的转置
    // input[1,3]经permute(1,0)后应该是[3,1]
    // order=(1,0)表示新维度0来自旧维度1，新维度1来自旧维度0
    @Test
    public void testPermute1D() {
        Permute permute = new Permute(1, 0);  // 修正：应该是(1,0)而不是(0,1)
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray output = permute.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(3, 1), output.getShape());
        
        // 验证数据转置正确
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(2f, output.get(1, 0), DELTA);
        assertEquals(3f, output.get(2, 0), DELTA);
    }

    @Test
    public void testSqueezeNoSingletonDims() {
        Squeeze squeeze = new Squeeze();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = squeeze.forward(input);
        
        // 没有大小为1的维度，形状不变
        assertEquals(input.getShape(), output.getShape());
    }

    @Test
    public void testRepeatNoRepeat() {
        Repeat repeat = new Repeat(1, 1);
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = repeat.forward(input);
        
        // 重复次数都是1，形状不变
        assertEquals(input.getShape(), output.getShape());
        assertArrayEquals(input.getMatrix(), output.getMatrix());
    }

    @Test
    public void testRequireInputNum() {
        assertEquals(1, new Permute(0, 1).requireInputNum());
        assertEquals(1, new Squeeze().requireInputNum());
        assertEquals(1, new Squeeze(0).requireInputNum());
        assertEquals(1, new Unsqueeze(0).requireInputNum());
        assertEquals(1, new Expand(Shape.of(2, 3)).requireInputNum());
        assertEquals(1, new Repeat(2, 3).requireInputNum());
    }
}
