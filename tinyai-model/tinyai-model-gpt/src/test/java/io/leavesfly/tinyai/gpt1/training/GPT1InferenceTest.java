package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

/**
 * GPT1Inference 单元测试
 * 
 * 测试覆盖：
 * 1. 推理引擎初始化
 * 2. 贪婪解码生成
 * 3. Temperature采样生成
 * 4. Top-K采样生成
 * 5. Top-P采样生成
 * 6. Beam Search生成
 * 7. 边界条件
 * 8. 生成质量验证
 * 
 * @author TinyAI
 */
public class GPT1InferenceTest {
    
    private GPT1Model model;
    private GPT1Inference inference;
    
    @Before
    public void setUp() {
        // 使用Tiny模型以节省内存
        model = GPT1Model.createTinyModel("test-inference");
        inference = new GPT1Inference(model);
    }
    
    // ==================== 初始化测试 ====================
    
    @Test
    public void testInferenceCreation() {
        GPT1Inference inf = new GPT1Inference(model);
        assertNotNull("推理引擎不应为null", inf);
    }
    
    // ==================== 贪婪解码测试 ====================
    
    @Test
    public void testGreedyGeneration() {
        int[] promptIds = {1, 2, 3};
        int maxNewTokens = 10;
        
        int[] generated = inference.generateGreedy(promptIds, maxNewTokens);
        
        assertNotNull("生成结果不应为null", generated);
        assertTrue("生成长度应>=原始长度", generated.length >= promptIds.length);
        assertTrue("生成长度应<=原始+maxNewTokens", 
            generated.length <= promptIds.length + maxNewTokens);
    }
    
    @Test
    public void testGreedyGenerationZeroTokens() {
        int[] promptIds = {1, 2, 3, 4, 5};
        
        int[] generated = inference.generateGreedy(promptIds, 0);
        
        assertNotNull(generated);
        assertEquals("生成0个token应返回原始序列", promptIds.length, generated.length);
    }
    
    @Test
    public void testGreedyGenerationLongPrompt() {
        int[] promptIds = new int[50];
        for (int i = 0; i < 50; i++) {
            promptIds[i] = i % 100;
        }
        
        int[] generated = inference.generateGreedy(promptIds, 20);
        
        assertNotNull(generated);
        assertTrue("生成结果应包含原始prompt", generated.length >= 50);
    }
    
    @Test
    public void testGreedyDeterministic() {
        int[] promptIds = {10, 20, 30};
        
        int[] generated1 = inference.generateGreedy(promptIds, 5);
        int[] generated2 = inference.generateGreedy(promptIds, 5);
        
        // 贪婪解码应该是确定性的
        assertArrayEquals("相同输入的贪婪解码应产生相同结果", generated1, generated2);
    }
    
    // ==================== Temperature采样测试 ====================
    
    @Test
    public void testTemperatureGeneration() {
        int[] promptIds = {1, 2, 3};
        float temperature = 0.8f;
        
        int[] generated = inference.generateWithTemperature(promptIds, 10, temperature);
        
        assertNotNull("Temperature生成结果不应为null", generated);
        assertTrue("生成长度应合理", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTemperatureZero() {
        // temperature接近0应该类似贪婪解码
        int[] promptIds = {5, 10, 15};
        
        int[] generated = inference.generateWithTemperature(promptIds, 8, 0.01f);
        
        assertNotNull(generated);
        assertTrue("低温采样应产生结果", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTemperatureHigh() {
        // 高temperature应该增加随机性
        int[] promptIds = {1, 2, 3};
        
        int[] generated = inference.generateWithTemperature(promptIds, 10, 2.0f);
        
        assertNotNull(generated);
        assertTrue("高温采样应产生结果", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTemperatureRandomness() {
        // 使用相同输入，temperature采样应产生不同结果（通常）
        int[] promptIds = {1, 2, 3};
        
        int[] generated1 = inference.generateWithTemperature(promptIds, 10, 1.0f);
        int[] generated2 = inference.generateWithTemperature(promptIds, 10, 1.0f);
        
        assertNotNull(generated1);
        assertNotNull(generated2);
        
        // 注意：由于随机性，不保证一定不同，但概率很高
        // 这里只验证都能成功生成
        assertTrue("两次生成都应成功", 
            generated1.length > 0 && generated2.length > 0);
    }
    
    // ==================== Top-K采样测试 ====================
    
    @Test
    public void testTopKGeneration() {
        int[] promptIds = {1, 2, 3};
        int topK = 10;
        float temperature = 1.0f;
        
        int[] generated = inference.generateTopK(promptIds, 10, topK, temperature);
        
        assertNotNull("Top-K生成结果不应为null", generated);
        assertTrue("生成长度应合理", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopKSmallK() {
        int[] promptIds = {5, 10};
        
        int[] generated = inference.generateTopK(promptIds, 5, 3, 1.0f);
        
        assertNotNull(generated);
        assertTrue("小K值应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopKLargeK() {
        int[] promptIds = {1, 2, 3};
        
        // K值接近词汇表大小
        int[] generated = inference.generateTopK(promptIds, 8, 100, 1.0f);
        
        assertNotNull(generated);
        assertTrue("大K值应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopKWithDifferentTemperatures() {
        int[] promptIds = {10, 20};
        
        int[] low = inference.generateTopK(promptIds, 5, 10, 0.5f);
        int[] high = inference.generateTopK(promptIds, 5, 10, 2.0f);
        
        assertNotNull("低温Top-K应成功", low);
        assertNotNull("高温Top-K应成功", high);
    }
    
    // ==================== Top-P采样测试 ====================
    
    @Test
    public void testTopPGeneration() {
        int[] promptIds = {1, 2, 3};
        float topP = 0.9f;
        float temperature = 1.0f;
        
        int[] generated = inference.generateTopP(promptIds, 10, topP, temperature);
        
        assertNotNull("Top-P生成结果不应为null", generated);
        assertTrue("生成长度应合理", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopPSmallP() {
        int[] promptIds = {5, 10};
        
        // 小p值更保守
        int[] generated = inference.generateTopP(promptIds, 5, 0.5f, 1.0f);
        
        assertNotNull(generated);
        assertTrue("小p值应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopPLargeP() {
        int[] promptIds = {1, 2, 3};
        
        // 大p值更随机
        int[] generated = inference.generateTopP(promptIds, 8, 0.95f, 1.0f);
        
        assertNotNull(generated);
        assertTrue("大p值应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testTopPBoundary() {
        int[] promptIds = {10};
        
        // p=1.0应该包含所有token
        int[] generated = inference.generateTopP(promptIds, 3, 1.0f, 1.0f);
        
        assertNotNull(generated);
        assertTrue("p=1.0应能生成", generated.length > promptIds.length);
    }
    
    // ==================== Beam Search测试 ====================
    
    @Test
    public void testBeamSearchGeneration() {
        int[] promptIds = {1, 2, 3};
        int beamSize = 3;
        
        int[] generated = inference.generateBeamSearch(promptIds, 10, beamSize);
        
        assertNotNull("Beam Search生成结果不应为null", generated);
        assertTrue("生成长度应合理", generated.length >= promptIds.length);
    }
    
    @Test
    public void testBeamSearchSingleBeam() {
        int[] promptIds = {5, 10};
        
        // beamSize=1应该类似贪婪搜索
        int[] generated = inference.generateBeamSearch(promptIds, 5, 1);
        
        assertNotNull(generated);
        assertTrue("单beam应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testBeamSearchLargeBeam() {
        int[] promptIds = {1, 2};
        
        int[] generated = inference.generateBeamSearch(promptIds, 8, 5);
        
        assertNotNull(generated);
        assertTrue("大beam应能生成", generated.length >= promptIds.length);
    }
    
    @Test
    public void testBeamSearchDeterministic() {
        int[] promptIds = {10, 20, 30};
        
        int[] generated1 = inference.generateBeamSearch(promptIds, 5, 3);
        int[] generated2 = inference.generateBeamSearch(promptIds, 5, 3);
        
        // Beam Search应该是确定性的
        assertArrayEquals("相同输入的Beam Search应产生相同结果", generated1, generated2);
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testSingleTokenPrompt() {
        int[] promptIds = {42};
        
        int[] greedy = inference.generateGreedy(promptIds, 5);
        int[] temp = inference.generateWithTemperature(promptIds, 5, 1.0f);
        int[] topk = inference.generateTopK(promptIds, 5, 10, 1.0f);
        int[] topp = inference.generateTopP(promptIds, 5, 0.9f, 1.0f);
        int[] beam = inference.generateBeamSearch(promptIds, 5, 2);
        
        assertTrue("单token prompt - Greedy应能生成", greedy.length > 1);
        assertTrue("单token prompt - Temperature应能生成", temp.length > 1);
        assertTrue("单token prompt - Top-K应能生成", topk.length > 1);
        assertTrue("单token prompt - Top-P应能生成", topp.length > 1);
        assertTrue("单token prompt - Beam Search应能生成", beam.length > 1);
    }
    
    @Test
    public void testLongPrompt() {
        // 测试较长的prompt
        int[] promptIds = new int[60];
        for (int i = 0; i < 60; i++) {
            promptIds[i] = i % 50 + 1;
        }
        
        int[] generated = inference.generateGreedy(promptIds, 10);
        
        assertNotNull(generated);
        assertTrue("长prompt应能生成", generated.length >= 60);
    }
    
    @Test
    public void testMaxSeqLenLimit() {
        // 测试生成接近maxSeqLen的限制
        GPT1Config config = model.getConfig();
        int maxSeqLen = config.getNPositions();
        
        int[] promptIds = new int[10];
        for (int i = 0; i < 10; i++) {
            promptIds[i] = i + 1;
        }
        
        // 尝试生成超过maxSeqLen的token
        int[] generated = inference.generateGreedy(promptIds, maxSeqLen + 100);
        
        assertNotNull(generated);
        // 应该被限制在maxSeqLen
        assertTrue("生成长度不应超过maxSeqLen", generated.length <= maxSeqLen);
    }
    
    // ==================== 生成质量验证测试 ====================
    
    @Test
    public void testGeneratedTokensInVocabRange() {
        int[] promptIds = {1, 2, 3};
        int vocabSize = model.getConfig().getVocabSize();
        
        int[] generated = inference.generateGreedy(promptIds, 10);
        
        // 验证所有token都在词汇表范围内
        for (int token : generated) {
            assertTrue("Token应在词汇表范围内: " + token, 
                token >= 0 && token < vocabSize);
        }
    }
    
    @Test
    public void testPromptPreservation() {
        int[] promptIds = {10, 20, 30, 40};
        
        int[] generated = inference.generateGreedy(promptIds, 5);
        
        // 验证生成序列包含原始prompt（前几个token）
        assertTrue("生成序列应>=prompt长度", generated.length >= promptIds.length);
        for (int i = 0; i < promptIds.length; i++) {
            assertEquals("Prompt token应被保留", promptIds[i], generated[i]);
        }
    }
    
    @Test
    public void testMultipleGenerationsConsistency() {
        int[] promptIds = {5, 10, 15};
        
        // 测试多次生成（贪婪应一致）
        int[] gen1 = inference.generateGreedy(promptIds, 8);
        int[] gen2 = inference.generateGreedy(promptIds, 8);
        int[] gen3 = inference.generateGreedy(promptIds, 8);
        
        assertArrayEquals("多次贪婪生成应一致", gen1, gen2);
        assertArrayEquals("多次贪婪生成应一致", gen2, gen3);
    }
    
    @Test
    public void testDifferentStrategiesProduceResults() {
        int[] promptIds = {1, 2, 3, 4};
        int maxNew = 6;
        
        int[] greedy = inference.generateGreedy(promptIds, maxNew);
        int[] temp = inference.generateWithTemperature(promptIds, maxNew, 1.0f);
        int[] topk = inference.generateTopK(promptIds, maxNew, 10, 1.0f);
        int[] topp = inference.generateTopP(promptIds, maxNew, 0.9f, 1.0f);
        int[] beam = inference.generateBeamSearch(promptIds, maxNew, 2);
        
        // 所有策略都应产生有效结果
        assertTrue("Greedy应生成", greedy.length >= promptIds.length);
        assertTrue("Temperature应生成", temp.length >= promptIds.length);
        assertTrue("Top-K应生成", topk.length >= promptIds.length);
        assertTrue("Top-P应生成", topp.length >= promptIds.length);
        assertTrue("Beam Search应生成", beam.length >= promptIds.length);
    }
    
    // ==================== 性能基准测试 ====================
    
    @Test
    public void testGenerationSpeed() {
        int[] promptIds = {1, 2, 3};
        
        long start = System.currentTimeMillis();
        int[] generated = inference.generateGreedy(promptIds, 20);
        long elapsed = System.currentTimeMillis() - start;
        
        assertNotNull(generated);
        // 简单验证生成在合理时间内完成（这里只是确保没有超时）
        assertTrue("生成应在合理时间内完成", elapsed < 30000); // 30秒
    }
    
    @Test
    public void testBatchInference() {
        // 测试多个prompt的推理
        int[][] prompts = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9}
        };
        
        for (int[] prompt : prompts) {
            int[] generated = inference.generateGreedy(prompt, 5);
            assertNotNull("批次推理应成功", generated);
            assertTrue("每个prompt都应生成结果", generated.length >= prompt.length);
        }
    }
}
