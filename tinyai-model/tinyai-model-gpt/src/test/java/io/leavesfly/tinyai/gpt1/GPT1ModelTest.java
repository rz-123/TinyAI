package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.Test;
import org.junit.Before;
import static org.junit.Assert.*;

/**
 * GPT1Model 单元测试
 * 
 * 测试覆盖：
 * 1. 模型创建（工厂方法）
 * 2. 前向传播
 * 3. 文本生成
 * 4. 模型信息输出
 * 5. 边界条件测试
 * 6. 异常处理
 * 
 * 注意事项：
 * - 使用Tiny/Small模型进行测试以避免OOM
 * - 序列长度限制在合理范围内
 * - Standard模型仅测试配置，不实例化
 * 
 * @author TinyAI
 */
public class GPT1ModelTest {
    
    private GPT1Model tinyModel;
    
    @Before
    public void setUp() {
        // 每个测试前创建微型模型（内存友好）
        tinyModel = GPT1Model.createTinyModel("test-gpt1-tiny");
    }
    
    // ==================== 模型创建测试 ====================
    
    @Test
    public void testCreateTinyModel() {
        GPT1Model model = GPT1Model.createTinyModel("gpt1-tiny");
        assertNotNull("微型模型不应为null", model);
        assertEquals("gpt1-tiny", model.getName());
        
        GPT1Config config = model.getConfig();
        assertEquals(256, config.getNEmbd());
        assertEquals(6, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(128, config.getNPositions());
    }
    
    @Test
    public void testCreateSmallModel() {
        GPT1Model model = GPT1Model.createSmallModel("gpt1-small");
        assertNotNull("小型模型不应为null", model);
        assertEquals("gpt1-small", model.getName());
        
        GPT1Config config = model.getConfig();
        assertEquals(512, config.getNEmbd());
        assertEquals(8, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(256, config.getNPositions());
    }
    
    @Test
    public void testStandardModelConfig() {
        // 注意：由于Standard模型（117M参数）可能导致OOM，仅测试配置而不实例化模型
        GPT1Config config = GPT1Config.createStandardConfig();
        assertNotNull("标准配置不应为null", config);
        assertEquals(768, config.getNEmbd());
        assertEquals(12, config.getNHead());
        assertEquals(12, config.getNLayer());
        assertEquals(512, config.getNPositions());
        
        // 验证参数估算在合理范围内
        long paramCount = config.estimateParameterCount();
        assertTrue("标准模型参数应在100M-130M之间", 
            paramCount > 100_000_000L && paramCount < 130_000_000L);
    }
    
    @Test
    public void testCustomModelCreation() {
        GPT1Config customConfig = new GPT1Config();
        customConfig.setVocabSize(5000);
        customConfig.setNEmbd(128);
        customConfig.setNLayer(4);
        customConfig.setNHead(4);
        customConfig.setNInner(512);
        customConfig.setNPositions(64);
        
        GPT1Model model = new GPT1Model("custom-gpt1", customConfig);
        assertNotNull(model);
        assertEquals("custom-gpt1", model.getName());
        assertEquals(128, model.getConfig().getNEmbd());
    }
    
    // ==================== 前向传播测试 ====================
    
    @Test
    public void testForwardPassSingleBatch() {
        // 创建输入: batch_size=1, seq_len=10
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        // 填充一些token IDs
        for (int i = 0; i < 10; i++) {
            tokenIds.set(i * 10, 0, i);
        }
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull("输出不应为null", output);
        assertNotNull("输出值不应为null", output.getValue());
        
        // 验证输出形状: (batch_size, seq_len, vocab_size)
        Shape outputShape = output.getValue().getShape();
        assertEquals("输出批次大小应为1", 1, outputShape.getDimension(0));
        assertEquals("输出序列长度应为10", 10, outputShape.getDimension(1));
        assertEquals("输出词汇表大小应为10000", 10000, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassMultipleBatches() {
        // 创建输入: batch_size=4, seq_len=20
        NdArray tokenIds = NdArray.of(Shape.of(4, 20));
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        
        // 验证输出形状
        Shape outputShape = output.getValue().getShape();
        assertEquals(4, outputShape.getDimension(0));
        assertEquals(20, outputShape.getDimension(1));
        assertEquals(10000, outputShape.getDimension(2));
    }
    
    @Test
    public void testForwardPassLongSequence() {
        // 测试接近最大序列长度（Tiny模型maxSeqLen=128）
        NdArray tokenIds = NdArray.of(Shape.of(2, 64));
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        assertEquals(64, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testPredictMethod() {
        // 测试predict方法（是forward的别名）
        NdArray tokenIds = NdArray.of(Shape.of(1, 15));
        Variable input = new Variable(tokenIds);
        
        Variable output = tinyModel.predict(input);
        
        assertNotNull(output);
        assertEquals(15, output.getValue().getShape().getDimension(1));
    }
    
    // ==================== 文本生成测试 ====================
    
    @Test
    public void testGenerateSequenceBasic() {
        // 创建prompt: batch_size=1, seq_len=5
        NdArray promptIds = NdArray.of(Shape.of(1, 5));
        
        // 生成10个新token
        NdArray generated = tinyModel.generateSequence(promptIds, 10);
        
        assertNotNull("生成结果不应为null", generated);
        
        // 验证生成序列长度: 原始5 + 新生成10 = 15
        assertEquals("生成序列长度应为15", 15, generated.getShape().getDimension(1));
    }
    
    @Test
    public void testGenerateSequenceZeroTokens() {
        NdArray promptIds = NdArray.of(Shape.of(1, 10));
        
        // 生成0个新token（应该返回原始prompt）
        NdArray generated = tinyModel.generateSequence(promptIds, 0);
        
        assertNotNull(generated);
        assertEquals(10, generated.getShape().getDimension(1));
    }
    
    @Test
    public void testGenerateSequenceMultipleBatches() {
        // 测试批量生成
        NdArray promptIds = NdArray.of(Shape.of(3, 8));
        
        NdArray generated = tinyModel.generateSequence(promptIds, 12);
        
        assertNotNull(generated);
        assertEquals("批次大小应保持为3", 3, generated.getShape().getDimension(0));
        assertEquals("生成序列长度应为20", 20, generated.getShape().getDimension(1));
    }
    
    @Test
    public void testGenerateSequenceWithMaxSeqLen() {
        // 测试生成到达最大序列长度
        NdArray promptIds = NdArray.of(Shape.of(1, 10));
        
        // 尝试生成超过maxSeqLen的token（Tiny模型maxSeqLen=128）
        NdArray generated = tinyModel.generateSequence(promptIds, 200);
        
        assertNotNull(generated);
        // 应该被限制在maxSeqLen
        assertTrue("生成长度不应超过maxSeqLen", 
            generated.getShape().getDimension(1) <= 128);
    }
    
    // ==================== 边界条件测试 ====================
    
    @Test
    public void testMinimalSequenceLength() {
        // 测试最小序列长度1
        NdArray tokenIds = NdArray.of(Shape.of(1, 1));
        tokenIds.set(100, 0, 0);
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        assertEquals(1, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testMaxSequenceLength() {
        // 测试最大序列长度（Tiny模型：128）
        NdArray tokenIds = NdArray.of(Shape.of(1, 128));
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        assertEquals(128, output.getValue().getShape().getDimension(1));
    }
    
    @Test
    public void testSingleBatch() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        assertEquals(1, output.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testLargeBatch() {
        // 测试较大批次（降低序列长度以避免OOM）
        NdArray tokenIds = NdArray.of(Shape.of(8, 16));
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        assertNotNull(output);
        assertEquals(8, output.getValue().getShape().getDimension(0));
    }
    
    // ==================== 异常处理测试 ====================
    
    @Test(expected = IllegalArgumentException.class)
    public void testInvalidInputDimensions() {
        // 创建1维输入（应该是2维）
        NdArray tokenIds = NdArray.of(Shape.of(10));
        Variable input = new Variable(tokenIds);
        
        // 应该抛出异常
        tinyModel.forward(input);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSequenceTooLong() {
        // 创建超过最大长度的序列（128+1 for Tiny model）
        NdArray tokenIds = NdArray.of(Shape.of(1, 129));
        Variable input = new Variable(tokenIds);
        
        // 应该抛出异常
        tinyModel.forward(input);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNullInput() {
        // 传入null应抛出异常
        tinyModel.forward((Variable[]) null);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testEmptyInput() {
        // 传入空数组应抛出异常
        tinyModel.forward(new Variable[0]);
    }
    
    // ==================== 模型信息测试 ====================
    
    @Test
    public void testModelName() {
        assertEquals("test-gpt1-tiny", tinyModel.getName());
    }
    
    @Test
    public void testModelConfig() {
        GPT1Config config = tinyModel.getConfig();
        assertNotNull("配置不应为null", config);
        
        // 验证微型模型配置
        assertEquals(256, config.getNEmbd());
        assertEquals(6, config.getNLayer());
        assertEquals(8, config.getNHead());
        assertEquals(1024, config.getNInner());
    }
    
    @Test
    public void testGetGPT1Block() {
        GPT1MainBlock block = tinyModel.getGPT1Block();
        assertNotNull("GPT1Block不应为null", block);
    }
    
    @Test
    public void testPrintModelInfo() {
        // 测试打印模型信息不抛出异常
        try {
            tinyModel.printModelInfo();
        } catch (Exception e) {
            fail("打印模型信息不应抛出异常: " + e.getMessage());
        }
    }
    
    @Test
    public void testGetConfigSummary() {
        String summary = tinyModel.getConfigSummary();
        
        assertNotNull("配置摘要不应为null", summary);
        assertTrue("摘要应包含词汇表大小", summary.contains("词汇表大小"));
        assertTrue("摘要应包含嵌入维度", summary.contains("嵌入维度"));
        assertTrue("摘要应包含层数", summary.contains("层数"));
        assertTrue("摘要应包含参数量", summary.contains("参数量"));
    }
    
    @Test
    public void testToString() {
        String str = tinyModel.toString();
        
        assertNotNull("toString不应返回null", str);
        assertTrue("toString应包含模型名", str.contains("test-gpt1-tiny"));
        assertTrue("toString应包含GPT1Model", str.contains("GPT1Model"));
    }
    
    // ==================== 模型一致性测试 ====================
    
    @Test
    public void testConsecutiveForwardPasses() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 10));
        Variable input = new Variable(tokenIds);
        
        // 执行两次前向传播
        Variable output1 = tinyModel.forward(input);
        Variable output2 = tinyModel.forward(input);
        
        // 两次输出形状应该相同
        assertEquals(output1.getValue().getShape().toString(), 
                    output2.getValue().getShape().toString());
    }
    
    @Test
    public void testDifferentBatchSizes() {
        // 测试相同模型可以处理不同批次大小
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(4, 10));
        NdArray input3 = NdArray.of(Shape.of(8, 10));
        
        Variable output1 = tinyModel.forward(new Variable(input1));
        Variable output2 = tinyModel.forward(new Variable(input2));
        Variable output3 = tinyModel.forward(new Variable(input3));
        
        assertEquals(1, output1.getValue().getShape().getDimension(0));
        assertEquals(4, output2.getValue().getShape().getDimension(0));
        assertEquals(8, output3.getValue().getShape().getDimension(0));
    }
    
    @Test
    public void testDifferentSequenceLengths() {
        // 测试相同模型可以处理不同序列长度
        NdArray input1 = NdArray.of(Shape.of(1, 10));
        NdArray input2 = NdArray.of(Shape.of(1, 32));
        NdArray input3 = NdArray.of(Shape.of(1, 64));
        
        Variable output1 = tinyModel.forward(new Variable(input1));
        Variable output2 = tinyModel.forward(new Variable(input2));
        Variable output3 = tinyModel.forward(new Variable(input3));
        
        assertEquals(10, output1.getValue().getShape().getDimension(1));
        assertEquals(32, output2.getValue().getShape().getDimension(1));
        assertEquals(64, output3.getValue().getShape().getDimension(1));
    }
    
    // ==================== 输出有效性测试 ====================
    
    @Test
    public void testOutputNotAllZeros() {
        NdArray tokenIds = NdArray.of(Shape.of(1, 5));
        for (int i = 0; i < 5; i++) {
            tokenIds.set(i + 1, 0, i);
        }
        
        Variable input = new Variable(tokenIds);
        Variable output = tinyModel.forward(input);
        
        NdArray outputArray = output.getValue();
        
        // 检查输出不全为0（模型已初始化）
        boolean hasNonZero = false;
        int vocabSize = outputArray.getShape().getDimension(2);
        for (int i = 0; i < Math.min(10, vocabSize); i++) {
            if (Math.abs(outputArray.get(0, 0, i)) > 1e-6) {
                hasNonZero = true;
                break;
            }
        }
        
        assertTrue("输出不应全为0", hasNonZero);
    }
    
    @Test
    public void testGeneratedTokensInVocabRange() {
        NdArray promptIds = NdArray.of(Shape.of(1, 3));
        promptIds.set(1, 0, 0);
        promptIds.set(2, 0, 1);
        promptIds.set(3, 0, 2);
        
        NdArray generated = tinyModel.generateSequence(promptIds, 5);
        
        int vocabSize = tinyModel.getConfig().getVocabSize();
        
        // 检查生成的token在词汇表范围内
        for (int i = 0; i < generated.getShape().getDimension(1); i++) {
            int tokenId = (int) generated.get(0, i);
            assertTrue("Token ID应在词汇表范围内", tokenId >= 0 && tokenId < vocabSize);
        }
    }
}
