package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * 基础数学函数测试
 * 
 * @author TinyAI
 */
public class BasicMathFunctionsTest {
    
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
    
    // =============================================================================
    // Exp测试
    // =============================================================================
    
    @Test
    public void testExpForward() {
        Exp exp = new Exp();
        NdArray input = NdArray.of(new float[][]{{0f, 1f, 2f}});
        NdArray output = exp.forward(input);
        
        float[] expected = {1f, (float)Math.exp(1), (float)Math.exp(2)};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    @Test
    public void testExpBackward() {
        Exp exp = new Exp();
        Variable x = new Variable(NdArray.of(new float[][]{{1f}}), "x");
        Variable y = exp.call(x);
        y.backward();
        
        // d(e^x)/dx = e^x
        assertEquals((float)Math.exp(1), x.getGrad().get(0, 0), DELTA);
    }
    
    // =============================================================================
    // Log测试
    // =============================================================================
    
    @Test
    public void testLogForward() {
        Log log = new Log();
        NdArray input = NdArray.of(new float[][]{{1f, (float)Math.E, 10f}});
        NdArray output = log.forward(input);
        
        assertEquals(0f, output.get(0, 0), DELTA);
        assertEquals(1f, output.get(0, 1), DELTA);
        assertTrue(output.get(0, 2) > 2.3f && output.get(0, 2) < 2.4f);
    }
    
    // =============================================================================
    // Sqrt测试
    // =============================================================================
    
    @Test
    public void testSqrtForward() {
        Sqrt sqrt = new Sqrt();
        NdArray input = NdArray.of(new float[][]{{1f, 4f, 9f, 16f}});
        NdArray output = sqrt.forward(input);
        
        float[] expected = {1f, 2f, 3f, 4f};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    // =============================================================================
    // Pow测试
    // =============================================================================
    
    @Test
    public void testPowForward() {
        Pow pow = new Pow(2);
        NdArray input = NdArray.of(new float[][]{{1f, 2f, 3f}});
        NdArray output = pow.forward(input);
        
        float[] expected = {1f, 4f, 9f};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    // =============================================================================
    // Squ (Square)测试
    // =============================================================================
    
    @Test
    public void testSquForward() {
        Squ squ = new Squ();
        NdArray input = NdArray.of(new float[][]{{2f, 3f, -4f}});
        NdArray output = squ.forward(input);
        
        float[] expected = {4f, 9f, 16f};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    @Test
    public void testSquBackward() {
        Squ squ = new Squ();
        Variable x = new Variable(NdArray.of(new float[][]{{3f}}), "x");
        Variable y = squ.call(x);
        y.backward();
        
        // d(x^2)/dx = 2x
        assertEquals(6f, x.getGrad().get(0, 0), DELTA);
    }
    
    // =============================================================================
    // Sin/Cos测试
    // =============================================================================
    
    @Test
    public void testSinForward() {
        Sin sin = new Sin();
        NdArray input = NdArray.of(new float[][]{{0f, (float)Math.PI/2, (float)Math.PI}});
        NdArray output = sin.forward(input);
        
        assertEquals(0f, output.get(0, 0), DELTA);
        assertEquals(1f, output.get(0, 1), DELTA);
        assertEquals(0f, output.get(0, 2), DELTA);
    }
    
    @Test
    public void testCosForward() {
        Cos cos = new Cos();
        NdArray input = NdArray.of(new float[][]{{0f, (float)Math.PI/2, (float)Math.PI}});
        NdArray output = cos.forward(input);
        
        assertEquals(1f, output.get(0, 0), DELTA);
        assertEquals(0f, output.get(0, 1), DELTA);
        assertEquals(-1f, output.get(0, 2), DELTA);
    }
    
    // =============================================================================
    // 边界条件测试
    // =============================================================================
    
    @Test
    public void testExpWithZero() {
        Exp exp = new Exp();
        NdArray input = NdArray.of(new float[][]{{0f}});
        NdArray output = exp.forward(input);
        assertEquals(1f, output.get(0, 0), DELTA);
    }
    
    @Test
    public void testSqrtWithOne() {
        Sqrt sqrt = new Sqrt();
        NdArray input = NdArray.of(new float[][]{{1f}});
        NdArray output = sqrt.forward(input);
        assertEquals(1f, output.get(0, 0), DELTA);
    }
    
    @Test
    public void testPowWithZeroExponent() {
        Pow pow = new Pow(0);
        NdArray input = NdArray.of(new float[][]{{5f, 10f}});
        NdArray output = pow.forward(input);
        
        // 任何数的0次方都是1
        float[] expected = {1f, 1f};
        assertArrayEquals(expected, output.getArray(), DELTA);
    }
    
    @Test
    public void testChainedMathOperations() {
        // 测试链式数学操作: exp(log(x)) = x
        Variable x = new Variable(NdArray.of(new float[][]{{5f}}), "x");
        
        Log log = new Log();
        Exp exp = new Exp();
        
        Variable y1 = log.call(x);
        Variable y2 = exp.call(y1);
        
        // 应该恢复原值
        assertEquals(5f, y2.getValue().get(0, 0), 0.01f);
    }
    
    @Test
    public void testGradientAccuracy() {
        // 测试梯度精度
        Squ squ = new Squ();
        Variable x = new Variable(NdArray.of(new float[][]{{2f}}), "x");
        Variable y = squ.call(x);
        y.backward();
        
        // 手动计算: f(x) = x^2, f'(x) = 2x = 4
        assertEquals(4f, x.getGrad().get(0, 0), DELTA);
    }
}
