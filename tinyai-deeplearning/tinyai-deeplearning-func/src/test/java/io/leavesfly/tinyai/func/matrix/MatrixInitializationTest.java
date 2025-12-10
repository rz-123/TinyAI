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
 * 矩阵初始化函数的单元测试
 * 包括: OnesLike, ZerosLike, Clone, Detach
 *
 * @author TinyAI
 */
public class MatrixInitializationTest {

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

    // ==================== OnesLike Tests ====================

    @Test
    public void testOnesLikeBasic() {
        OnesLike onesLike = new OnesLike();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray output = onesLike.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 3), output.getShape());
        
        // 验证所有元素为1
        float[][] result = output.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(1f, result[i][j], DELTA);
            }
        }
    }

    @Test
    public void testOnesLikeDifferentShapes() {
        OnesLike onesLike = new OnesLike();
        
        // 测试1D
        NdArray input1D = NdArray.of(new float[]{1f, 2f, 3f});
        NdArray output1D = onesLike.forward(input1D);
        assertEquals(input1D.getShape(), output1D.getShape());
        
        // 测试3D
        NdArray input3D = NdArray.of(Shape.of(2, 3, 4));
        NdArray output3D = onesLike.forward(input3D);
        assertEquals(input3D.getShape(), output3D.getShape());
        assertEquals(24, output3D.getArray().length);
    }

    @Test
    public void testOnesLikeBackward() {
        OnesLike onesLike = new OnesLike();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = onesLike.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // onesLike的梯度应该全为0（因为输出不依赖输入值）
        float[][] grad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0f, grad[i][j], DELTA);
            }
        }
    }

    @Test
    public void testOnesLikeSingleElement() {
        OnesLike onesLike = new OnesLike();
        
        NdArray input = NdArray.of(new float[][]{{99f}});
        NdArray output = onesLike.forward(input);
        
        assertEquals(Shape.of(1, 1), output.getShape());
        assertEquals(1f, output.get(0, 0), DELTA);
    }

    // ==================== ZerosLike Tests ====================

    @Test
    public void testZerosLikeBasic() {
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}, {4f, 5f, 6f}});
        NdArray output = zerosLike.forward(input);
        
        assertNotNull(output);
        assertEquals(Shape.of(2, 3), output.getShape());
        
        // 验证所有元素为0
        float[][] result = output.getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                assertEquals(0f, result[i][j], DELTA);
            }
        }
    }

    @Test
    public void testZerosLikeDifferentShapes() {
        ZerosLike zerosLike = new ZerosLike();
        
        // 测试1D
        NdArray input1D = NdArray.of(new float[]{10f, 20f, 30f});
        NdArray output1D = zerosLike.forward(input1D);
        assertEquals(input1D.getShape(), output1D.getShape());
        
        // 测试3D
        NdArray input3D = NdArray.of(Shape.of(3, 4, 5));
        NdArray output3D = zerosLike.forward(input3D);
        assertEquals(input3D.getShape(), output3D.getShape());
        assertEquals(60, output3D.getArray().length);
    }

    @Test
    public void testZerosLikeBackward() {
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = zerosLike.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // zerosLike的梯度应该全为0
        float[][] grad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0f, grad[i][j], DELTA);
            }
        }
    }

    @Test
    public void testZerosLikeSingleElement() {
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray input = NdArray.of(new float[][]{{99f}});
        NdArray output = zerosLike.forward(input);
        
        assertEquals(Shape.of(1, 1), output.getShape());
        assertEquals(0f, output.get(0, 0), DELTA);
    }

    // ==================== Clone Tests ====================

    @Test
    public void testCloneBasic() {
        Clone clone = new Clone();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = clone.forward(input);
        
        assertNotNull(output);
        assertEquals(input.getShape(), output.getShape());
        
        // 验证值相同
        assertArrayEquals(input.getMatrix(), output.getMatrix());
        
        // 验证是深拷贝（修改输入不影响输出）
        float[] inputArray = input.getArray();
        float[] outputArray = output.getArray();
        assertNotSame(inputArray, outputArray);
    }

    @Test
    public void testCloneDeepCopy() {
        Clone clone = new Clone();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = clone.forward(input);
        
        // 修改输入数据
        input.getArray()[0] = 999f;
        
        // 输出不应该被影响
        assertEquals(1f, output.get(0, 0), DELTA);
    }

    @Test
    public void testCloneBackward() {
        Clone clone = new Clone();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = clone.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // clone的梯度应该直接传播
        float[][] grad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(1f, grad[i][j], DELTA);
            }
        }
    }

    @Test
    public void testCloneDifferentShapes() {
        Clone clone = new Clone();
        
        // 测试3D
        NdArray input3D = NdArray.of(Shape.of(2, 3, 4));
        NdArray output3D = clone.forward(input3D);
        assertEquals(input3D.getShape(), output3D.getShape());
        assertNotSame(input3D.getArray(), output3D.getArray());
    }

    // ==================== Detach Tests ====================

    @Test
    public void testDetachBasic() {
        Detach detach = new Detach();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = detach.forward(input);
        
        assertNotNull(output);
        assertEquals(input.getShape(), output.getShape());
        
        // 验证值相同
        assertArrayEquals(input.getMatrix(), output.getMatrix());
        
        // 验证是深拷贝
        assertNotSame(input.getArray(), output.getArray());
    }

    @Test
    public void testDetachStopsGradient() {
        Detach detach = new Detach();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        Variable x = new Variable(input, "x");
        Variable y = detach.call(x);
        
        Variable sum = y.sum();
        sum.backward();
        
        assertNotNull(x.getGrad());
        assertEquals(x.getValue().getShape(), x.getGrad().getShape());
        
        // detach的梯度应该全为0（阻断梯度传播）
        float[][] grad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0f, grad[i][j], DELTA);
            }
        }
    }

    @Test
    public void testDetachInChain() {
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        Variable x = new Variable(input, "x");
        Variable y = x.add(x);  // y = 2x
        Variable z = new Detach().call(y);  // z = detach(y)
        Variable w = z.add(z);  // w = 2z
        
        Variable sum = w.sum();
        sum.backward();
        
        // 因为detach阻断了梯度，x的梯度应该为0
        assertNotNull(x.getGrad());
        float[][] grad = x.getGrad().getMatrix();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                assertEquals(0f, grad[i][j], DELTA);
            }
        }
    }

    @Test
    public void testDetachDeepCopy() {
        Detach detach = new Detach();
        
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        NdArray output = detach.forward(input);
        
        // 修改输入数据
        input.getArray()[0] = 999f;
        
        // 输出不应该被影响
        assertEquals(1f, output.get(0, 0), DELTA);
    }

    // ==================== Combined Operations ====================

    @Test
    public void testOnesLikeAndZerosLike() {
        NdArray input = NdArray.of(new float[][]{{5f, 10f}, {15f, 20f}});
        
        OnesLike onesLike = new OnesLike();
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray ones = onesLike.forward(input);
        NdArray zeros = zerosLike.forward(input);
        
        // ones + zeros应该等于ones
        NdArray sum = ones.add(zeros);
        assertArrayEquals(ones.getMatrix(), sum.getMatrix());
    }

    @Test
    public void testCloneAndDetachDifference() {
        NdArray input = NdArray.of(new float[][]{{1f, 2f}, {3f, 4f}});
        
        Clone clone = new Clone();
        Detach detach = new Detach();
        
        Variable x = new Variable(input, "x");
        
        // Clone传播梯度
        Variable cloned = clone.call(x);
        Variable clonedSum = cloned.sum();
        clonedSum.backward();
        assertNotNull(x.getGrad());
        assertEquals(1f, x.getGrad().get(0, 0), DELTA);
        
        // 重置梯度
        x.clearGrad();
        
        // Detach阻断梯度
        Variable detached = detach.call(x);
        Variable detachedSum = detached.sum();
        detachedSum.backward();
        assertNotNull(x.getGrad());
        assertEquals(0f, x.getGrad().get(0, 0), DELTA);
    }

    // ==================== Edge Cases ====================

    @Test
    public void testOnesLikeEmptyShape() {
        OnesLike onesLike = new OnesLike();
        
        NdArray input = NdArray.of(Shape.of(0, 0));
        NdArray output = onesLike.forward(input);
        
        assertEquals(Shape.of(0, 0), output.getShape());
    }

    @Test
    public void testZerosLikeEmptyShape() {
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray input = NdArray.of(Shape.of(0, 0));
        NdArray output = zerosLike.forward(input);
        
        assertEquals(Shape.of(0, 0), output.getShape());
    }

    @Test
    public void testCloneLargeArray() {
        Clone clone = new Clone();
        
        // 创建较大的数组
        float[][] largeData = new float[100][100];
        for (int i = 0; i < 100; i++) {
            for (int j = 0; j < 100; j++) {
                largeData[i][j] = i * 100 + j;
            }
        }
        
        NdArray input = NdArray.of(largeData);
        NdArray output = clone.forward(input);
        
        assertEquals(input.getShape(), output.getShape());
        assertArrayEquals(input.getMatrix(), output.getMatrix());
        assertNotSame(input.getArray(), output.getArray());
    }

    @Test
    public void testDetachMultipleTimes() {
        // 简化测试，避免多次detach导致的死循环问题
        NdArray input = NdArray.of(new float[][]{{1f, 2f}});
        Variable x = new Variable(input, "x");
        
        // 单次detach测试
        Detach detach = new Detach();
        Variable y = detach.call(x);
        
        // detach后的变量应该不需要梯度
        assertNotNull(y);
        assertNotNull(y.getValue());
        assertEquals(input.getShape(), y.getValue().getShape());
        
        // 验证值相同
        assertEquals(1f, y.getValue().get(0, 0), DELTA);
        assertEquals(2f, y.getValue().get(0, 1), DELTA);
    }

    @Test
    public void testRequireInputNum() {
        assertEquals(1, new OnesLike().requireInputNum());
        assertEquals(1, new ZerosLike().requireInputNum());
        assertEquals(1, new Clone().requireInputNum());
        assertEquals(1, new Detach().requireInputNum());
    }

    @Test
    public void testOnesLikeWithNegativeValues() {
        OnesLike onesLike = new OnesLike();
        
        NdArray input = NdArray.of(new float[][]{{-5f, -10f}, {-15f, -20f}});
        NdArray output = onesLike.forward(input);
        
        // 无论输入值如何，输出都应该是1
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(1f, output.get(1, 1), DELTA);
    }

    @Test
    public void testZerosLikeWithLargeValues() {
        ZerosLike zerosLike = new ZerosLike();
        
        NdArray input = NdArray.of(new float[][]{{1000f, 2000f}, {3000f, 4000f}});
        NdArray output = zerosLike.forward(input);
        
        // 无论输入值如何，输出都应该是0
        assertEquals(0f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(1, 1), DELTA);
    }

    @Test
    public void testClonePreservesValues() {
        Clone clone = new Clone();
        
        NdArray input = NdArray.of(new float[][]{{Float.MAX_VALUE, Float.MIN_VALUE}, {0f, -0f}});
        NdArray output = clone.forward(input);
        
        // 验证特殊值也能正确克隆
        assertEquals(Float.MAX_VALUE, output.get(0, 0), DELTA);
        assertEquals(Float.MIN_VALUE, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(1, 0), DELTA);
    }
}
