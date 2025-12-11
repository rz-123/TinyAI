package io.leavesfly.tinyai.gpt1;

import org.junit.Test;
import static org.junit.Assert.*;

/**
 * GPT1Config 单元测试
 * 
 * 测试覆盖：
 * 1. 默认配置创建
 * 2. 预设配置工厂方法（Standard、Tiny、Small）
 * 3. 自定义配置
 * 4. 配置验证
 * 5. 参数估算
 * 6. 边界条件
 * 7. 异常处理
 * 
 * @author TinyAI
 */
public class GPT1ConfigTest {
    
    // ==================== 基础配置测试 ====================
    
    @Test
    public void testDefaultConfig() {
        GPT1Config config = new GPT1Config();
        
        // 验证默认值（GPT-1标准配置）
        assertEquals("默认词汇表大小应为40478", 40478, config.getVocabSize());
        assertEquals("默认序列长度应为512", 512, config.getNPositions());
        assertEquals("默认嵌入维度应为768", 768, config.getNEmbd());
        assertEquals("默认层数应为12", 12, config.getNLayer());
        assertEquals("默认注意力头数应为12", 12, config.getNHead());
        assertEquals("默认FFN维度应为3072", 3072, config.getNInner());
        assertEquals("默认激活函数应为gelu", "gelu", config.getActivationFunction());
        
        // 验证dropout配置
        assertEquals("默认残差dropout应为0.1", 0.1, config.getResidPdrop(), 0.001);
        assertEquals("默认嵌入dropout应为0.1", 0.1, config.getEmbdPdrop(), 0.001);
        assertEquals("默认注意力dropout应为0.1", 0.1, config.getAttnPdrop(), 0.001);
        
        // 验证初始化配置
        assertEquals("默认LayerNorm epsilon应为1e-5", 1e-5, config.getLayerNormEpsilon(), 1e-6);
        assertEquals("默认初始化范围应为0.02", 0.02, config.getInitializerRange(), 0.001);
    }
    
    @Test
    public void testStandardConfig() {
        GPT1Config config = GPT1Config.createStandardConfig();
        
        // 验证标准配置（117M参数）
        assertEquals(40478, config.getVocabSize());
        assertEquals(768, config.getNEmbd());
        assertEquals(12, config.getNLayer());
        assertEquals(12, config.getNHead());
        assertEquals(3072, config.getNInner());
        assertEquals(512, config.getNPositions());
        
        // 验证配置有效性
        try {
            config.validate();
        } catch (Exception e) {
            fail("标准配置应该有效");
        }
    }
    
    @Test
    public void testTinyConfig() {
        GPT1Config config = GPT1Config.createTinyConfig();
        
        // 验证微型配置
        assertEquals(10000, config.getVocabSize());
        assertEquals(256, config.getNEmbd());
        assertEquals(6, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(1024, config.getNInner());
        assertEquals(128, config.getNPositions());
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("微型配置应该有效");
        }
    }
    
    @Test
    public void testSmallConfig() {
        GPT1Config config = GPT1Config.createSmallConfig();
        
        // 验证小型配置
        assertEquals(20000, config.getVocabSize());
        assertEquals(512, config.getNEmbd());
        assertEquals(8, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(2048, config.getNInner());
        assertEquals(256, config.getNPositions());
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("小型配置应该有效");
        }
    }
    
    // ==================== 配置修改测试 ====================
    
    @Test
    public void testCustomConfig() {
        GPT1Config config = new GPT1Config();
        
        // 修改基础配置
        config.setVocabSize(32000);
        config.setNPositions(1024);
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setActivationFunction("gelu");
        
        // 验证修改生效
        assertEquals(32000, config.getVocabSize());
        assertEquals(1024, config.getNPositions());
        assertEquals(512, config.getNEmbd());
        assertEquals(8, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(2048, config.getNInner());
        assertEquals("gelu", config.getActivationFunction());
    }
    
    @Test
    public void testDropoutConfiguration() {
        GPT1Config config = new GPT1Config();
        
        config.setResidPdrop(0.2);
        config.setEmbdPdrop(0.15);
        config.setAttnPdrop(0.05);
        
        assertEquals(0.2, config.getResidPdrop(), 0.001);
        assertEquals(0.15, config.getEmbdPdrop(), 0.001);
        assertEquals(0.05, config.getAttnPdrop(), 0.001);
    }
    
    // ==================== 配置验证测试 ====================
    
    @Test
    public void testInvalidVocabSize() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(0);
        
        try {
            config.validate();
            fail("词汇表大小为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("词汇表大小"));
        }
    }
    
    @Test
    public void testInvalidNegativeVocabSize() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(-100);
        
        try {
            config.validate();
            fail("词汇表大小为负数应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("词汇表大小"));
        }
    }
    
    @Test
    public void testInvalidPositions() {
        GPT1Config config = new GPT1Config();
        config.setNPositions(0);
        
        try {
            config.validate();
            fail("位置数为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("最大位置数"));
        }
    }
    
    @Test
    public void testInvalidNegativePositions() {
        GPT1Config config = new GPT1Config();
        config.setNPositions(-512);
        
        try {
            config.validate();
            fail("位置数为负数应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("最大位置数"));
        }
    }
    
    @Test
    public void testInvalidEmbedDim() {
        GPT1Config config = new GPT1Config();
        config.setNEmbd(0);
        
        try {
            config.validate();
            fail("嵌入维度为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("嵌入维度"));
        }
    }
    
    @Test
    public void testInvalidLayer() {
        GPT1Config config = new GPT1Config();
        config.setNLayer(0);
        
        try {
            config.validate();
            fail("层数为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("层数"));
        }
    }
    
    @Test
    public void testInvalidNHead() {
        GPT1Config config = new GPT1Config();
        config.setNHead(0);
        
        try {
            config.validate();
            fail("注意力头数为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("注意力头数"));
        }
    }
    
    @Test
    public void testEmbedDimNotDivisibleByHeads() {
        GPT1Config config = new GPT1Config();
        config.setNEmbd(100);
        config.setNHead(7);
        
        try {
            config.validate();
            fail("嵌入维度不能被头数整除应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("整除"));
        }
    }
    
    @Test
    public void testInvalidFFNDim() {
        GPT1Config config = new GPT1Config();
        config.setNInner(0);
        
        try {
            config.validate();
            fail("前馈网络维度为0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("前馈网络维度"));
        }
    }
    
    @Test
    public void testInvalidDropoutNegative() {
        GPT1Config config = new GPT1Config();
        config.setResidPdrop(-0.1);
        
        try {
            config.validate();
            fail("负数dropout应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropout"));
        }
    }
    
    @Test
    public void testInvalidDropoutTooLarge() {
        GPT1Config config = new GPT1Config();
        config.setEmbdPdrop(1.0);
        
        try {
            config.validate();
            fail("dropout=1.0应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropout"));
        }
    }
    
    @Test
    public void testInvalidAttnDropoutOutOfRange() {
        GPT1Config config = new GPT1Config();
        config.setAttnPdrop(1.5);
        
        try {
            config.validate();
            fail("dropout>1应抛出异常");
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("dropout"));
        }
    }
    
    // ==================== 参数估算测试 ====================
    
    @Test
    public void testStandardModelParameterEstimation() {
        GPT1Config config = GPT1Config.createStandardConfig();
        long paramCount = config.estimateParameterCount();
        
        // 标准GPT-1模型约117M参数
        assertTrue("标准模型参数应在100M-130M之间，实际: " + paramCount,
            paramCount > 100_000_000 && paramCount < 130_000_000);
    }
    
    @Test
    public void testTinyModelParameterEstimation() {
        GPT1Config config = GPT1Config.createTinyConfig();
        long paramCount = config.estimateParameterCount();
        
        // 微型模型参数较少
        assertTrue("微型模型参数应在5M-15M之间，实际: " + paramCount,
            paramCount > 5_000_000 && paramCount < 15_000_000);
    }
    
    @Test
    public void testSmallModelParameterEstimation() {
        GPT1Config config = GPT1Config.createSmallConfig();
        long paramCount = config.estimateParameterCount();
        
        // 小型模型参数
        assertTrue("小型模型参数应在30M-60M之间，实际: " + paramCount,
            paramCount > 30_000_000 && paramCount < 60_000_000);
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testMinimalValidConfig() {
        GPT1Config config = new GPT1Config();
        config.setVocabSize(100);
        config.setNPositions(32);
        config.setNEmbd(64);
        config.setNLayer(2);
        config.setNHead(4);
        config.setNInner(256);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("最小有效配置应通过验证");
        }
    }
    
    @Test
    public void testZeroDropout() {
        GPT1Config config = new GPT1Config();
        config.setResidPdrop(0.0);
        config.setEmbdPdrop(0.0);
        config.setAttnPdrop(0.0);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("0 dropout应该有效");
        }
    }
    
    @Test
    public void testMaxDropout() {
        GPT1Config config = new GPT1Config();
        config.setResidPdrop(0.99);
        config.setEmbdPdrop(0.99);
        config.setAttnPdrop(0.99);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("0.99 dropout应该有效");
        }
    }
    
    @Test
    public void testSingleHead() {
        GPT1Config config = new GPT1Config();
        config.setNEmbd(64);
        config.setNHead(1);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("单头注意力应该有效");
        }
    }
    
    @Test
    public void testSingleLayer() {
        GPT1Config config = new GPT1Config();
        config.setNLayer(1);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("单层Transformer应该有效");
        }
    }
    
    // ==================== 字符串表示测试 ====================
    
    @Test
    public void testToString() {
        GPT1Config config = GPT1Config.createTinyConfig();
        String str = config.toString();
        
        assertNotNull("toString不应返回null", str);
        assertTrue("toString应包含vocabSize", str.contains("vocabSize"));
        assertTrue("toString应包含nEmbd", str.contains("nEmbd"));
        assertTrue("toString应包含nLayer", str.contains("nLayer"));
        assertTrue("toString应包含nHead", str.contains("nHead"));
    }
    
    @Test
    public void testParameterCountFormatting() {
        // 测试参数量格式化显示
        GPT1Config config = GPT1Config.createStandardConfig();
        String str = config.toString();
        
        // 应该包含格式化的参数量（例如 "117.00M"）
        assertTrue("toString应包含estimatedParams", str.contains("estimatedParams"));
    }
    
    // ==================== 完整构造函数测试 ====================
    
    @Test
    public void testFullConstructor() {
        GPT1Config config = new GPT1Config(
            30000,    // vocabSize
            256,      // nPositions
            384,      // nEmbd
            6,        // nLayer
            6,        // nHead
            1536,     // nInner
            "gelu",   // activationFunction
            0.15,     // residPdrop
            0.15,     // embdPdrop
            0.15,     // attnPdrop
            1e-5,     // layerNormEpsilon
            0.02      // initializerRange
        );
        
        assertEquals(30000, config.getVocabSize());
        assertEquals(256, config.getNPositions());
        assertEquals(384, config.getNEmbd());
        assertEquals(6, config.getNLayer());
        assertEquals(6, config.getNHead());
        assertEquals(1536, config.getNInner());
        assertEquals("gelu", config.getActivationFunction());
        assertEquals(0.15, config.getResidPdrop(), 0.001);
        assertEquals(0.15, config.getEmbdPdrop(), 0.001);
        assertEquals(0.15, config.getAttnPdrop(), 0.001);
        assertEquals(1e-5, config.getLayerNormEpsilon(), 1e-6);
        assertEquals(0.02, config.getInitializerRange(), 0.001);
        
        try {
            config.validate();
        } catch (Exception e) {
            fail("完整构造的配置应该有效");
        }
    }
}
