package io.leavesfly.tinyai.deepseek.v3;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeekV3Config单元测试
 * 
 * 测试范围：
 * 1. 默认配置创建
 * 2. 预设配置工厂方法（Tiny、Small、Standard、Large）
 * 3. 配置参数验证
 * 4. 参数量估算
 * 5. 激活率计算
 * 6. 配置setter/getter
 * 
 * @author leavesfly
 */
public class DeepSeekV3ConfigTest {
    
    @Test
    public void testDefaultConfig() {
        // 测试默认配置创建
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 验证基础配置默认值
        assertEquals(50257, config.getVocabSize(), "默认词汇表大小应为50257");
        assertEquals(2048, config.getNPositions(), "默认序列长度应为2048");
        assertEquals(768, config.getNEmbd(), "默认嵌入维度应为768");
        assertEquals(12, config.getNLayer(), "默认层数应为12");
        assertEquals(12, config.getNHead(), "默认注意力头数应为12");
        assertEquals(3072, config.getNInner(), "默认前馈网络维度应为3072");
        
        // 验证MoE配置默认值
        assertEquals(8, config.getNumExperts(), "默认专家数量应为8");
        assertEquals(2, config.getTopK(), "默认Top-K应为2");
        assertEquals(3072, config.getExpertHiddenDim(), "默认专家隐藏层维度应为3072");
        
        // 验证任务感知配置
        assertTrue(config.isEnableTaskAwareRouting(), "默认应启用任务感知路由");
        assertEquals(5, config.getNumTaskTypes(), "默认支持5种任务类型");
    }
    
    @Test
    public void testTinyConfig() {
        // 测试微型配置
        DeepSeekV3Config config = DeepSeekV3Config.createTinyConfig();
        
        assertEquals(10000, config.getVocabSize(), "Tiny配置词汇表大小应为10000");
        assertEquals(256, config.getNEmbd(), "Tiny配置嵌入维度应为256");
        assertEquals(6, config.getNLayer(), "Tiny配置层数应为6");
        assertEquals(8, config.getNHead(), "Tiny配置注意力头数应为8");
        assertEquals(512, config.getNPositions(), "Tiny配置序列长度应为512");
        assertEquals(4, config.getNumExperts(), "Tiny配置专家数量应为4");
        assertEquals(2, config.getTopK(), "Tiny配置Top-K应为2");
    }
    
    @Test
    public void testSmallConfig() {
        // 测试小型配置
        DeepSeekV3Config config = DeepSeekV3Config.createSmallConfig();
        
        assertEquals(30000, config.getVocabSize(), "Small配置词汇表大小应为30000");
        assertEquals(512, config.getNEmbd(), "Small配置嵌入维度应为512");
        assertEquals(8, config.getNLayer(), "Small配置层数应为8");
        assertEquals(8, config.getNHead(), "Small配置注意力头数应为8");
        assertEquals(1024, config.getNPositions(), "Small配置序列长度应为1024");
        assertEquals(4, config.getNumExperts(), "Small配置专家数量应为4");
    }
    
    @Test
    public void testStandardConfig() {
        // 测试标准配置
        DeepSeekV3Config config = DeepSeekV3Config.createStandardConfig();
        
        assertEquals(50257, config.getVocabSize(), "Standard配置词汇表大小应为50257");
        assertEquals(768, config.getNEmbd(), "Standard配置嵌入维度应为768");
        assertEquals(12, config.getNLayer(), "Standard配置层数应为12");
        assertEquals(12, config.getNHead(), "Standard配置注意力头数应为12");
        assertEquals(2048, config.getNPositions(), "Standard配置序列长度应为2048");
        assertEquals(8, config.getNumExperts(), "Standard配置专家数量应为8");
    }
    
    @Test
    public void testCustomConfig() {
        // 测试自定义配置
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 设置自定义参数
        config.setVocabSize(50257);
        config.setNEmbd(1024);
        config.setNLayer(24);
        config.setNHead(16);
        config.setNPositions(2048);
        config.setNumExperts(8);
        config.setTopK(2);
        
        assertEquals(50257, config.getVocabSize(), "词汇表大小应为50257");
        assertEquals(1024, config.getNEmbd(), "嵌入维度应为1024");
        assertEquals(24, config.getNLayer(), "层数应为24");
        assertEquals(16, config.getNHead(), "注意力头数应为16");
        assertEquals(2048, config.getNPositions(), "序列长度应为2048");
        assertEquals(8, config.getNumExperts(), "专家数量应为8");
    }
    
    @Test
    public void testConfigValidation() {
        // 测试配置验证
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 默认配置应该通过验证
        assertDoesNotThrow(() -> config.validate(), "默认配置应该有效");
        
        // 测试无效的词汇表大小
        config.setVocabSize(-1);
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "负数词汇表大小应该抛出异常");
        
        // 恢复有效值
        config.setVocabSize(50257);
        
        // 测试无效的嵌入维度
        config.setNEmbd(0);
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "零嵌入维度应该抛出异常");
        
        // 恢复有效值
        config.setNEmbd(768);
        
        // 测试无效的注意力头数（嵌入维度不能被头数整除）
        config.setNHead(7); // 768不能被7整除
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "嵌入维度不能被注意力头数整除时应该抛出异常");
        
        // 恢复有效值
        config.setNHead(12);
        
        // 测试无效的TopK（大于专家数量）
        config.setTopK(10); // 大于默认的8个专家
        assertThrows(IllegalArgumentException.class, () -> config.validate(), 
                    "TopK大于专家数量时应该抛出异常");
    }
    
    @Test
    public void testParameterCountEstimation() {
        // 测试参数量估算
        DeepSeekV3Config tinyConfig = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Config standardConfig = DeepSeekV3Config.createStandardConfig();
        
        long tinyParams = tinyConfig.estimateParameterCount();
        long standardParams = standardConfig.estimateParameterCount();
        
        // Tiny配置参数量应该更小
        assertTrue(tinyParams < standardParams, 
                  "Tiny配置的参数量应该小于Standard配置");
        
        // 参数量应该大于0
        assertTrue(tinyParams > 0, "Tiny配置参数量应大于0");
        assertTrue(standardParams > 0, "Standard配置参数量应大于0");
        
        // 打印参数量信息（用于调试）
        System.out.println("Tiny配置参数量: " + formatParamCount(tinyParams));
        System.out.println("Standard配置参数量: " + formatParamCount(standardParams));
    }
    
    @Test
    public void testActiveParameterCountEstimation() {
        // 测试激活参数量估算
        DeepSeekV3Config config = DeepSeekV3Config.createStandardConfig();
        
        long totalParams = config.estimateParameterCount();
        long activeParams = config.estimateActiveParameterCount();
        
        // 激活参数应该小于总参数（MoE只激活部分专家）
        assertTrue(activeParams < totalParams, 
                  "激活参数量应该小于总参数量");
        
        // 验证激活率约为25%（Top-2 / 8专家）
        double activationRatio = config.getActivationRatio();
        assertTrue(activationRatio > 20 && activationRatio < 30, 
                  "激活率应该在20-30%之间，实际: " + activationRatio);
        
        System.out.println("总参数量: " + formatParamCount(totalParams));
        System.out.println("激活参数量: " + formatParamCount(activeParams));
        System.out.println("激活率: " + String.format("%.2f%%", activationRatio));
    }
    
    @Test
    public void testSettersAndGetters() {
        // 测试所有setter和getter方法
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 基础配置
        config.setVocabSize(100000);
        assertEquals(100000, config.getVocabSize());
        
        config.setNEmbd(1024);
        assertEquals(1024, config.getNEmbd());
        
        config.setNLayer(16);
        assertEquals(16, config.getNLayer());
        
        config.setNHead(16);
        assertEquals(16, config.getNHead());
        
        // MoE配置
        config.setNumExperts(16);
        assertEquals(16, config.getNumExperts());
        
        config.setTopK(4);
        assertEquals(4, config.getTopK());
        
        config.setLoadBalanceLossWeight(0.02);
        assertEquals(0.02, config.getLoadBalanceLossWeight(), 0.0001);
        
        // 任务感知配置
        config.setEnableTaskAwareRouting(false);
        assertFalse(config.isEnableTaskAwareRouting());
        
        config.setNumTaskTypes(10);
        assertEquals(10, config.getNumTaskTypes());
        
        // Dropout配置
        config.setResidPdrop(0.2);
        assertEquals(0.2, config.getResidPdrop(), 0.0001);
        
        config.setEmbdPdrop(0.15);
        assertEquals(0.15, config.getEmbdPdrop(), 0.0001);
        
        config.setAttnPdrop(0.1);
        assertEquals(0.1, config.getAttnPdrop(), 0.0001);
    }
    
    @Test
    public void testMoEConfiguration() {
        // 测试MoE特定配置
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        // 设置8个专家，Top-2路由
        config.setNumExperts(8);
        config.setTopK(2);
        config.setExpertHiddenDim(3072);
        config.setExpertDropout(0.1);
        
        assertEquals(8, config.getNumExperts(), "专家数量应为8");
        assertEquals(2, config.getTopK(), "Top-K应为2");
        assertEquals(3072, config.getExpertHiddenDim(), "专家隐藏层维度应为3072");
        assertEquals(0.1, config.getExpertDropout(), 0.0001, "专家dropout应为0.1");
        
        // 激活率应该是 TopK / NumExperts
        double expectedRatio = (2.0 / 8.0) * 100;
        assertEquals(expectedRatio, config.getActivationRatio(), 0.1, 
                    "激活率应为25%");
    }
    
    @Test
    public void testTaskAwareConfiguration() {
        // 测试任务感知配置
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        config.setEnableTaskAwareRouting(true);
        config.setNumTaskTypes(5);
        config.setTaskEmbedDim(128);
        config.setTaskClassifierHiddenDim(256);
        
        assertTrue(config.isEnableTaskAwareRouting(), "应启用任务感知路由");
        assertEquals(5, config.getNumTaskTypes(), "任务类型数量应为5");
        assertEquals(128, config.getTaskEmbedDim(), "任务嵌入维度应为128");
        assertEquals(256, config.getTaskClassifierHiddenDim(), 
                    "任务识别器隐藏层维度应为256");
    }
    
    @Test
    public void testCodeGenerationConfiguration() {
        // 测试代码生成配置
        DeepSeekV3Config config = new DeepSeekV3Config();
        
        config.setCodeQualityDim(4);
        config.setNumProgrammingLanguages(10);
        config.setCodeAnalysisHiddenDim(512);
        config.setSyntaxValidatorHiddenDim(256);
        
        assertEquals(4, config.getCodeQualityDim(), "代码质量评估维度应为4");
        assertEquals(10, config.getNumProgrammingLanguages(), 
                    "支持的编程语言数量应为10");
        assertEquals(512, config.getCodeAnalysisHiddenDim(), 
                    "代码分析隐藏层维度应为512");
        assertEquals(256, config.getSyntaxValidatorHiddenDim(), 
                    "语法验证器隐藏层维度应为256");
    }
    
    @Test
    public void testToString() {
        // 测试toString方法
        DeepSeekV3Config config = DeepSeekV3Config.createTinyConfig();
        String str = config.toString();
        
        assertNotNull(str, "toString应返回非空字符串");
        assertTrue(str.contains("DeepSeekV3Config"), "toString应包含类名");
        assertTrue(str.contains("vocabSize"), "toString应包含vocabSize");
        assertTrue(str.contains("nEmbd"), "toString应包含nEmbd");
        assertTrue(str.contains("numExperts"), "toString应包含numExperts");
        
        System.out.println("配置信息:\n" + str);
    }
    
    // ==================== 辅助方法 ====================
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
}
