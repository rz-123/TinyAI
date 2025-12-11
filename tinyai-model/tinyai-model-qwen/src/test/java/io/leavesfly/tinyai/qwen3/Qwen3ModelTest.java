package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Qwen3Model单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3ModelTest {
    
    @Test
    public void testModelCreation() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test-model", config);
        
        assertNotNull(model);
        assertEquals("test-model", model.getName());
        assertEquals(config, model.getConfig());
    }
    
    @Test
    public void testCreateSmallModel() {
        Qwen3Model model = Qwen3Model.createSmallModel("small");
        
        assertNotNull(model);
        assertEquals("small", model.getName());
        
        Qwen3Config config = model.getConfig();
        assertEquals(10000, config.getVocabSize());
        assertEquals(512, config.getHiddenSize());
        assertEquals(4, config.getNumHiddenLayers());
    }
    
    @Test
    public void testCreateDemoModel() {
        Qwen3Model model = Qwen3Model.createDemoModel("demo");
        
        assertNotNull(model);
        Qwen3Config config = model.getConfig();
        assertEquals(6, config.getNumHiddenLayers());
    }
    
    @Test
    public void testForwardPass_SingleSequence() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test", config);
        
        // 准备输入 [1, seq_len]
        int seqLen = 8;
        float[][] inputData = new float[1][seqLen];
        for (int i = 0; i < seqLen; i++) {
            inputData[0][i] = i % config.getVocabSize();
        }
        
        NdArray inputIds = NdArray.of(inputData);
        Variable input = new Variable(inputIds);
        
        // 前向传播
        Variable output = model.forward(input);
        
        // 验证输出形状 [1, seq_len, vocab_size]
        Shape outputShape = output.getShape();
        assertEquals(3, outputShape.getDimNum());
        assertEquals(1, outputShape.getDimension(0));
        assertEquals(seqLen, outputShape.getDimension(1));
        assertEquals(config.getVocabSize(), outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPass_BatchSequences() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test", config);
        
        // 准备批量输入 [batch_size, seq_len]
        int batchSize = 2;
        int seqLen = 8;
        float[][] inputData = new float[batchSize][seqLen];
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                inputData[b][s] = (b * 10 + s) % config.getVocabSize();
            }
        }
        
        NdArray inputIds = NdArray.of(inputData);
        Variable input = new Variable(inputIds);
        
        // 前向传播
        Variable output = model.forward(input);
        
        // 验证输出形状
        Shape outputShape = output.getShape();
        assertEquals(batchSize, outputShape.getDimension(0));
        assertEquals(seqLen, outputShape.getDimension(1));
        assertEquals(config.getVocabSize(), outputShape.getDimension(2));
    }
    
    @Test
    public void testPredict() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test", config);
        
        int seqLen = 5;
        float[][] inputData = new float[1][seqLen];
        for (int i = 0; i < seqLen; i++) {
            inputData[0][i] = i;
        }
        
        NdArray inputIds = NdArray.of(inputData);
        Variable input = new Variable(inputIds);
        
        // 使用predict方法
        Variable output = model.predict(input);
        
        assertNotNull(output);
        assertEquals(config.getVocabSize(), output.getShape().getDimension(2));
    }
    
    @Test
    public void testModelInfo() {
        Qwen3Model model = Qwen3Model.createSmallModel("info-test");
        
        // 测试printModelInfo不抛出异常
        model.printModelInfo();
        
        // 测试getConfigSummary
        String summary = model.getConfigSummary();
        assertNotNull(summary);
        assertTrue(summary.contains("词汇表大小"));
        assertTrue(summary.contains("隐藏维度"));
    }
    
    @Test
    public void testToString() {
        Qwen3Model model = Qwen3Model.createSmallModel("toString-test");
        String str = model.toString();
        
        assertNotNull(str);
        assertTrue(str.contains("Qwen3Model"));
        assertTrue(str.contains("toString-test"));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testForwardPass_EmptyInput() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test", config);
        
        // 空输入应该抛出异常
        model.forward();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testForwardPass_ExceedMaxLength() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        Qwen3Model model = new Qwen3Model("test", config);
        
        // 超过最大序列长度
        int tooLongSeq = config.getMaxPositionEmbeddings() + 10;
        float[][] inputData = new float[1][tooLongSeq];
        for (int i = 0; i < tooLongSeq; i++) {
            inputData[0][i] = i % config.getVocabSize();
        }
        
        NdArray inputIds = NdArray.of(inputData);
        Variable input = new Variable(inputIds);
        
        // 应该抛出异常
        model.forward(input);
    }
}
