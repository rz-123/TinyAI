package io.leavesfly.tinyai.qwen3;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Qwen3Config单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3ConfigTest {
    
    @Test
    public void testDefaultConfig() {
        Qwen3Config config = new Qwen3Config();
        
        // 验证默认值
        assertEquals(32000, config.getVocabSize());
        assertEquals(2048, config.getHiddenSize());
        assertEquals(24, config.getNumHiddenLayers());
        assertEquals(16, config.getNumAttentionHeads());
        assertEquals(16, config.getNumKeyValueHeads());
        assertEquals(5632, config.getIntermediateSize());
        assertEquals(2048, config.getMaxPositionEmbeddings());
        assertEquals(1e-6, config.getRmsNormEps(), 1e-9);
        assertEquals(10000.0, config.getRopeTheta(), 0.01);
    }
    
    @Test
    public void testSmallConfig() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        
        // 验证小型配置
        assertEquals(10000, config.getVocabSize());
        assertEquals(512, config.getHiddenSize());
        assertEquals(4, config.getNumHiddenLayers());
        assertEquals(8, config.getNumAttentionHeads());
        assertEquals(8, config.getNumKeyValueHeads());
        assertEquals(1408, config.getIntermediateSize());
        
        // 验证参数量在合理范围内（约16M）
        long params = config.estimateParameterCount();
        assertTrue("参数量应在10M-30M之间", params > 10_000_000 && params < 30_000_000);
    }
    
    @Test
    public void testDemoConfig() {
        Qwen3Config config = Qwen3Config.createDemoConfig();
        
        // 验证演示配置
        assertEquals(32000, config.getVocabSize());
        assertEquals(512, config.getHiddenSize());
        assertEquals(6, config.getNumHiddenLayers());
        
        // 验证参数量在合理范围内（约62M）
        long params = config.estimateParameterCount();
        assertTrue("参数量应在40M-80M之间", params > 40_000_000 && params < 80_000_000);
    }
    
    @Test
    public void testHeadDimCalculation() {
        Qwen3Config config = new Qwen3Config();
        config.setHiddenSize(512);
        config.setNumAttentionHeads(8);
        
        assertEquals(64, config.getHeadDim());
    }
    
    @Test
    public void testNumKeyValueGroups() {
        Qwen3Config config = new Qwen3Config();
        
        // 测试标准配置（MHA）
        config.setNumAttentionHeads(16);
        config.setNumKeyValueHeads(16);
        assertEquals(1, config.getNumKeyValueGroups());
        
        // 测试GQA配置
        config.setNumAttentionHeads(16);
        config.setNumKeyValueHeads(8);
        assertEquals(2, config.getNumKeyValueGroups());
    }
    
    @Test
    public void testValidation_Success() {
        Qwen3Config config = new Qwen3Config();
        config.setVocabSize(10000);
        config.setHiddenSize(512);
        config.setNumHiddenLayers(4);
        config.setNumAttentionHeads(8);
        config.setNumKeyValueHeads(8);
        config.setIntermediateSize(1408);
        
        // 应该不抛出异常
        config.validate();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testValidation_NegativeVocabSize() {
        Qwen3Config config = new Qwen3Config();
        config.setVocabSize(-1);
        config.validate();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testValidation_InvalidHeadDim() {
        Qwen3Config config = new Qwen3Config();
        config.setHiddenSize(513);  // 不能被头数整除
        config.setNumAttentionHeads(8);
        config.validate();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testValidation_InvalidKeyValueHeads() {
        Qwen3Config config = new Qwen3Config();
        config.setNumAttentionHeads(16);
        config.setNumKeyValueHeads(5);  // 16不能被5整除
        config.validate();
    }
    
    @Test
    public void testParameterEstimation() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        long params = config.estimateParameterCount();
        
        // 验证参数估算包含所有主要组件
        // Embedding + Transformer层 + LM Head
        assertTrue("参数量应大于0", params > 0);
        
        // 手动计算并验证（粗略）
        long vocabSize = config.getVocabSize();
        long hiddenSize = config.getHiddenSize();
        long numLayers = config.getNumHiddenLayers();
        long intermediateSize = config.getIntermediateSize();
        
        // Token embedding
        long embeddingParams = vocabSize * hiddenSize;
        
        // 每层的参数（简化估算）
        long perLayerParams = hiddenSize * hiddenSize * 4  // QKV + O
                            + hiddenSize * intermediateSize * 3  // Gate, Up, Down
                            + hiddenSize * 4;  // RMSNorm权重
        
        long expectedParams = embeddingParams + perLayerParams * numLayers + vocabSize * hiddenSize;
        
        // 允许20%的误差
        double ratio = (double) params / expectedParams;
        assertTrue("参数估算应在合理范围内", ratio > 0.8 && ratio < 1.2);
    }
    
    @Test
    public void testToString() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        String str = config.toString();
        
        // 验证toString包含关键信息
        assertTrue(str.contains("vocabSize"));
        assertTrue(str.contains("hiddenSize"));
        assertTrue(str.contains("numHiddenLayers"));
        assertTrue(str.contains("estimatedParams"));
    }
    
    @Test
    public void testSettersAndGetters() {
        Qwen3Config config = new Qwen3Config();
        
        config.setVocabSize(50000);
        assertEquals(50000, config.getVocabSize());
        
        config.setHiddenSize(1024);
        assertEquals(1024, config.getHiddenSize());
        
        config.setNumHiddenLayers(12);
        assertEquals(12, config.getNumHiddenLayers());
        
        config.setNumAttentionHeads(16);
        assertEquals(16, config.getNumAttentionHeads());
        
        config.setRmsNormEps(1e-5);
        assertEquals(1e-5, config.getRmsNormEps(), 1e-9);
        
        config.setRopeTheta(100000.0);
        assertEquals(100000.0, config.getRopeTheta(), 0.01);
    }
}
