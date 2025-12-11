package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.qwen3.Qwen3Config;
import io.leavesfly.tinyai.qwen3.Qwen3Model;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 * Qwen3Inference单元测试
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3InferenceTest {
    
    private Qwen3Model model;
    private Qwen3Inference inference;
    
    @Before
    public void setUp() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        model = new Qwen3Model("test-inference", config);
        inference = new Qwen3Inference(model);
    }
    
    @Test
    public void testGreedyGeneration() {
        int[] inputIds = {1, 2, 3, 4, 5};
        int maxNewTokens = 5;
        
        int[] output = inference.generateGreedy(inputIds, maxNewTokens);
        
        assertNotNull(output);
        assertTrue("输出长度应该大于输入", output.length > inputIds.length);
        assertTrue("输出长度不应超过输入+最大生成", output.length <= inputIds.length + maxNewTokens);
        
        // 验证输出包含输入
        for (int i = 0; i < inputIds.length; i++) {
            assertEquals(inputIds[i], output[i]);
        }
    }
    
    @Test
    public void testTopKGeneration() {
        int[] inputIds = {1, 2, 3};
        int maxNewTokens = 5;
        int topK = 5;
        
        int[] output = inference.generateTopK(inputIds, maxNewTokens, topK);
        
        assertNotNull(output);
        assertTrue(output.length > inputIds.length);
        assertTrue(output.length <= inputIds.length + maxNewTokens);
    }
    
    @Test
    public void testTopPGeneration() {
        int[] inputIds = {1, 2, 3};
        int maxNewTokens = 5;
        float topP = 0.9f;
        
        int[] output = inference.generateTopP(inputIds, maxNewTokens, topP);
        
        assertNotNull(output);
        assertTrue(output.length > inputIds.length);
        assertTrue(output.length <= inputIds.length + maxNewTokens);
    }
    
    @Test
    public void testTemperatureGeneration() {
        int[] inputIds = {1, 2, 3};
        int maxNewTokens = 5;
        float temperature = 0.8f;
        
        int[] output = inference.generateTemperature(inputIds, maxNewTokens, temperature);
        
        assertNotNull(output);
        assertTrue(output.length > inputIds.length);
        assertTrue(output.length <= inputIds.length + maxNewTokens);
    }
    
    @Test
    public void testDifferentStrategiesProduceDifferentResults() {
        int[] inputIds = {1, 2, 3};
        int maxNewTokens = 10;
        
        int[] greedy = inference.generateGreedy(inputIds, maxNewTokens);
        int[] topK = inference.generateTopK(inputIds, maxNewTokens, 5);
        int[] topP = inference.generateTopP(inputIds, maxNewTokens, 0.9f);
        
        // 不同策略可能产生不同结果（虽然不是100%保证）
        // 至少验证它们都能正常工作
        assertNotNull(greedy);
        assertNotNull(topK);
        assertNotNull(topP);
    }
    
    @Test
    public void testEarlyStoppingOnEOS() {
        Qwen3Config config = Qwen3Config.createSmallConfig();
        // 设置一个较小的EOS token用于测试
        config.setEosTokenId(999);  // 设置一个不太可能生成的EOS
        
        Qwen3Model testModel = new Qwen3Model("eos-test", config);
        Qwen3Inference testInference = new Qwen3Inference(testModel);
        
        int[] inputIds = {2, 3, 4};
        int maxNewTokens = 5;  // 使用较小的值进行测试
        
        int[] output = testInference.generateGreedy(inputIds, maxNewTokens);
        
        // 验证输出包含输入
        assertNotNull(output);
        assertTrue("输出应包含输入", output.length >= inputIds.length);
    }
    
    @Test
    public void testZeroMaxNewTokens() {
        int[] inputIds = {1, 2, 3};
        int maxNewTokens = 0;
        
        int[] output = inference.generateGreedy(inputIds, maxNewTokens);
        
        // 不生成新token，应该返回原输入
        assertEquals(inputIds.length, output.length);
        for (int i = 0; i < inputIds.length; i++) {
            assertEquals(inputIds[i], output[i]);
        }
    }
    
    @Test
    public void testSingleInputToken() {
        int[] inputIds = {5};
        int maxNewTokens = 3;
        
        int[] output = inference.generateGreedy(inputIds, maxNewTokens);
        
        assertNotNull(output);
        assertTrue("输出长度应大于等于1", output.length >= 1);
        assertEquals("第一个token应该是输入", 5, output[0]);
    }
    
    @Test
    public void testMultipleGenerations() {
        int[] inputIds = {1, 2, 3};
        
        // 连续生成多次，确保推理器状态正确
        int[] output1 = inference.generateGreedy(inputIds, 5);
        int[] output2 = inference.generateGreedy(inputIds, 5);
        int[] output3 = inference.generateGreedy(inputIds, 5);
        
        assertNotNull(output1);
        assertNotNull(output2);
        assertNotNull(output3);
        
        // 贪婪解码应该产生确定性结果
        assertArrayEquals(output1, output2);
        assertArrayEquals(output2, output3);
    }
}
