package io.leavesfly.tinyai.minimind.model;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

/**
 * MiniMindConfig 测试类
 *
 * @author TinyAI Team
 */
public class MiniMindConfigTest {

    @Test
    public void testSmallConfig() {
        MiniMindConfig config = MiniMindConfig.createSmallConfig();

        assertEquals(6400, config.getVocabSize());
        assertEquals(512, config.getMaxSeqLen());
        assertEquals(512, config.getHiddenSize());
        assertEquals(8, config.getNumLayers());
        assertEquals(16, config.getNumHeads());
        assertEquals(1024, config.getFfnHiddenSize());
        assertEquals("silu", config.getActivationFunction());
        assertTrue(config.isUseRoPE());
        assertTrue(config.isPreLayerNorm());
        assertFalse(config.isUseMoE());

        // 验证配置有效性
        assertDoesNotThrow(() -> config.validate());

        // 验证头维度计算
        assertEquals(32, config.getHeadDim());

        // 验证模型规模
        assertEquals("Small-26M", config.getModelSize());

        // 验证参数量估算(应该接近 26M)
        long params = config.estimateParameters();
        assertTrue(params > 20_000_000 && params < 30_000_000,
                "Estimated parameters should be around 26M, but got: " + params);
    }

    @Test
    public void testMediumConfig() {
        MiniMindConfig config = MiniMindConfig.createMediumConfig();

        assertEquals(768, config.getHiddenSize());
        assertEquals(16, config.getNumLayers());
        assertEquals(2048, config.getFfnHiddenSize());

        // 验证配置有效性
        assertDoesNotThrow(() -> config.validate());

        // 验证头维度
        assertEquals(48, config.getHeadDim());

        // 验证模型规模
        assertEquals("Medium-108M", config.getModelSize());

        // 验证参数量(应该接近 108M)
        long params = config.estimateParameters();
        assertTrue(params > 100_000_000 && params < 120_000_000,
                "Estimated parameters should be around 108M, but got: " + params);
    }

    @Test
    public void testMoEConfig() {
        MiniMindConfig config = MiniMindConfig.createMoEConfig();

        assertTrue(config.isUseMoE());
        assertEquals(4, config.getNumExperts());
        assertEquals(2, config.getNumExpertsPerToken());
        assertEquals(0.01f, config.getMoeLoadBalanceWeight(), 0.001f);

        // 验证配置有效性
        assertDoesNotThrow(() -> config.validate());

        // 验证模型规模描述
        assertTrue(config.getModelSize().contains("MoE"));
        assertTrue(config.getModelSize().contains("4 Experts"));

        // 验证参数量(应该接近 145M)
        long params = config.estimateParameters();
        assertTrue(params > 130_000_000 && params < 160_000_000,
                "Estimated parameters should be around 145M, but got: " + params);
    }

    @Test
    public void testConfigValidation() {
        MiniMindConfig config = new MiniMindConfig();

        // 测试无效的 hiddenSize / numHeads
        config.setHiddenSize(513);
        config.setNumHeads(16);
        assertThrows(IllegalStateException.class, () -> config.validate());

        // 修复配置
        config.setHiddenSize(512);
        assertDoesNotThrow(() -> config.validate());

        // 测试无效的 dropout
        config.setDropout(1.5f);
        assertThrows(IllegalStateException.class, () -> config.validate());

        config.setDropout(0.1f);
        assertDoesNotThrow(() -> config.validate());

        // 测试 MoE 配置验证
        config.setUseMoE(true);
        config.setNumExperts(0);
        assertThrows(IllegalStateException.class, () -> config.validate());

        config.setNumExperts(4);
        config.setNumExpertsPerToken(5);  // 超过专家数
        assertThrows(IllegalStateException.class, () -> config.validate());

        config.setNumExpertsPerToken(2);
        assertDoesNotThrow(() -> config.validate());
    }

    @Test
    public void testHeadDimCalculation() {
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        assertEquals(32, config.getHeadDim());

        config.setHiddenSize(768);
        assertEquals(48, config.getHeadDim());

        // 测试不能整除的情况
        config.setHiddenSize(513);
        assertThrows(IllegalStateException.class, () -> config.getHeadDim());
    }

    @Test
    public void testToString() {
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        String str = config.toString();

        assertTrue(str.contains("Small-26M"));
        assertTrue(str.contains("vocabSize=6400"));
        assertTrue(str.contains("hiddenSize=512"));
        assertTrue(str.contains("numLayers=8"));
        assertTrue(str.contains("useRoPE=true"));
    }
}
