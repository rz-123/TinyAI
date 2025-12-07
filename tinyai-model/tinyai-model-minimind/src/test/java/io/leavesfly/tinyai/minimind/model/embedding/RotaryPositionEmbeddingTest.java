package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * RotaryPositionEmbedding单元测试
 * 
 * @author leavesfly
 */
public class RotaryPositionEmbeddingTest {
    
    private RotaryPositionEmbedding rope;
    private final int dimModel = 64;
    private final int maxSeqLen = 128;
    private final float theta = 10000.0f;
    
    @BeforeEach
    public void setUp() {
        rope = new RotaryPositionEmbedding(dimModel, maxSeqLen, theta);
    }
    
    @Test
    public void testRoPECreation() {
        assertNotNull(rope, "RoPE不应为null");
    }
    
    @Test
    public void testComputeFrequencies() {
        // RoPE应该预计算频率
        // 这是内部实现,我们通过forward测试间接验证
        assertNotNull(rope, "RoPE创建后应正常");
    }
    
    @Test
    public void testForwardSinglePosition() {
        // 输入: [batch=1, seq_len=1, dim=64]
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = (float) i / dimModel; // 简单初始化
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        assertNotNull(output, "RoPE输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(1, shape[0], "batch维度应保持");
        assertEquals(1, shape[1], "seq_len维度应保持");
        assertEquals(dimModel, shape[2], "dim维度应保持");
    }
    
    @Test
    public void testForwardMultiplePositions() {
        // 输入: [batch=2, seq_len=4, dim=64]
        int batchSize = 2;
        int seqLen = 4;
        
        float[] data = new float[batchSize * seqLen * dimModel];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) i / data.length;
        }
        
        NdArray input = NdArray.of(data, Shape.of(batchSize, seqLen, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        assertNotNull(output, "RoPE输出不应为null");
        
        int[] shape = output.getValue().getShape().getShapeDims();
        assertEquals(batchSize, shape[0], "batch维度应为" + batchSize);
        assertEquals(seqLen, shape[1], "seq_len维度应为" + seqLen);
        assertEquals(dimModel, shape[2], "dim维度应为" + dimModel);
    }
    
    @Test
    public void testPositionOffset() {
        // 测试不同position offset的影响
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = 1.0f;
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        
        Variable pos1 = new Variable(NdArray.of(new float[]{0}, Shape.of(1)));
        Variable pos2 = new Variable(NdArray.of(new float[]{10}, Shape.of(1)));
        Variable output1 = rope.forward(new Variable(input), pos1);
        Variable output2 = rope.forward(new Variable(input), pos2);
        
        // 不同位置应产生不同的输出
        float[] result1 = output1.getValue().getArray();
        float[] result2 = output2.getValue().getArray();
        
        boolean different = false;
        for (int i = 0; i < result1.length; i++) {
            if (Math.abs(result1[i] - result2[i]) > 1e-6f) {
                different = true;
                break;
            }
        }
        assertTrue(different, "不同位置应产生不同的RoPE编码");
    }
    
    @Test
    public void testRoPERotation() {
        // RoPE应该对输入进行旋转变换
        float[] data = new float[dimModel];
        for (int i = 0; i < dimModel; i++) {
            data[i] = 1.0f; // 全1向量
        }
        
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        Variable inputVar = new Variable(input);
        
        Variable posVar = new Variable(NdArray.of(new float[]{5}, Shape.of(1)));
        Variable output = rope.forward(inputVar, posVar);
        
        float[] result = output.getValue().getArray();
        
        // 验证输出已经被修改(不再是全1)
        boolean modified = false;
        for (int i = 0; i < result.length; i++) {
            if (Math.abs(result[i] - 1.0f) > 1e-3f) {
                modified = true;
                break;
            }
        }
        assertTrue(modified, "RoPE应该修改输入向量");
    }
    
    @Test
    public void testMaxSeqLenBoundary() {
        // 测试最大序列长度边界
        float[] data = new float[dimModel];
        NdArray input = NdArray.of(data, Shape.of(1, 1, dimModel));
        
        // 在最大长度内应该正常工作
        Variable posVar = new Variable(NdArray.of(new float[]{maxSeqLen - 1}, Shape.of(1)));
        Variable output = rope.forward(new Variable(input), posVar);
        assertNotNull(output, "在最大序列长度内应正常工作");
    }
}
