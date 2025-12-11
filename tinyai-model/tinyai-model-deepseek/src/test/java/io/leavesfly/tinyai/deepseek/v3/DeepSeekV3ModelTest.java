package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * DeepSeekV3Model单元测试
 * 
 * 测试范围：
 * 1. 模型创建（工厂方法）
 * 2. 前向传播（基础predict）
 * 3. 任务感知推理
 * 4. 代码生成功能
 * 5. 序列生成
 * 6. 模型信息
 * 
 * @author leavesfly
 */
public class DeepSeekV3ModelTest {
    
    private DeepSeekV3Config tinyConfig;
    
    @BeforeEach
    public void setUp() {
        // 使用Tiny配置加快测试速度
        tinyConfig = DeepSeekV3Config.createTinyConfig();
    }
    
    @Test
    public void testCreateStandardModel() {
        // 测试创建标准模型
        DeepSeekV3Model model = DeepSeekV3Model.createStandardModel("test-standard");
        
        assertNotNull(model, "模型应创建成功");
        assertEquals("test-standard", model.getName(), "模型名称应匹配");
        String summary = model.getConfigSummary();
        assertNotNull(summary, "配置摘要应存在");
        assertTrue(summary.contains("DeepSeek-V3"), 
                  "摘要应包含DeepSeek-V3");
    }
    
    @Test
    public void testCreateTinyModel() {
        // 测试创建微型模型
        DeepSeekV3Model model = DeepSeekV3Model.createTinyModel("test-tiny");
        
        assertNotNull(model, "微型模型应创建成功");
        assertEquals("test-tiny", model.getName(), "模型名称应匹配");
    }
    
    @Test
    public void testCreateSmallModel() {
        // 测试创建小型模型
        DeepSeekV3Model model = DeepSeekV3Model.createSmallModel("test-small");
        
        assertNotNull(model, "小型模型应创建成功");
        assertEquals("test-small", model.getName(), "模型名称应匹配");
    }
    
    @Test
    public void testBasicPredict() {
        // 测试基础预测功能
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        // 创建输入: [batch=2, seq=4]
        float[][] inputData = {
            {1, 15, 23, 42},
            {2, 16, 24, 43}
        };
        Variable input = new Variable(NdArray.of(inputData));
        
        // 执行预测
        Variable output = model.predict(input);
        
        assertNotNull(output, "输出应存在");
        assertNotNull(output.getValue(), "输出值应存在");
        
        // 验证输出形状: [batch, seq, vocab_size]
        NdArray outputArray = output.getValue();
        assertEquals(3, outputArray.getShape().getDimNum(), 
                    "输出应为3维张量");
        assertEquals(2, outputArray.getShape().getDimension(0), 
                    "batch维度应为2");
        assertEquals(4, outputArray.getShape().getDimension(1), 
                    "序列维度应为4");
        assertEquals(tinyConfig.getVocabSize(), 
                    outputArray.getShape().getDimension(2), 
                    "词汇表维度应匹配");
    }
    
    @Test
    public void testPredictWithDifferentBatchSizes() {
        // 测试不同batch大小
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        // Batch size = 1
        float[][] input1 = {{1, 2, 3}};
        Variable output1 = model.predict(new Variable(NdArray.of(input1)));
        assertEquals(1, output1.getValue().getShape().getDimension(0), 
                    "batch=1的输出应正确");
        
        // Batch size = 4
        float[][] input4 = {
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        };
        Variable output4 = model.predict(new Variable(NdArray.of(input4)));
        assertEquals(4, output4.getValue().getShape().getDimension(0), 
                    "batch=4的输出应正确");
    }
    
    @Test
    public void testPredictWithDetails() {
        // 测试带详细信息的预测
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] inputData = {{1, 2, 3, 4, 5}};
        Variable input = new Variable(NdArray.of(inputData));
        
        // 使用REASONING任务类型
        DeepSeekV3Block.DetailedForwardResult result = 
            model.predictWithDetails(input, TaskType.REASONING);
        
        assertNotNull(result, "详细结果应存在");
        assertNotNull(result.logits, "logits应存在");
        assertNotNull(result.reasoningResult, "推理结果应存在");
    }
    
    @Test
    public void testCodeGeneration() {
        // 测试代码生成功能
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] codePrompt = {{1, 10, 20, 30}};
        Variable input = new Variable(NdArray.of(codePrompt));
        
        DeepSeekV3Model.CodeGenerationResult result = model.generateCode(input);
        
        assertNotNull(result, "代码生成结果应存在");
        assertNotNull(result.logits, "logits应存在");
        assertNotNull(result.detectedLanguage, "检测到的语言应存在");
        
        System.out.println("检测到的编程语言: " + result.detectedLanguage);
    }
    
    @Test
    public void testReasoningTask() {
        // 测试推理任务
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] reasoningPrompt = {{1, 5, 10, 15, 20}};
        Variable input = new Variable(NdArray.of(reasoningPrompt));
        
        DeepSeekV3Model.ReasoningResult result = model.performReasoning(input);
        
        assertNotNull(result, "推理结果应存在");
        assertNotNull(result.logits, "logits应存在");
        assertTrue(result.confidence >= 0 && result.confidence <= 1, 
                  "置信度应在[0,1]范围内，实际: " + result.confidence);
        
        System.out.println("推理置信度: " + result.confidence);
        System.out.println("任务类型: " + result.taskType);
    }
    
    @Test
    public void testMathTask() {
        // 测试数学任务
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] mathPrompt = {{1, 3, 7, 11}};
        Variable input = new Variable(NdArray.of(mathPrompt));
        
        DeepSeekV3Model.MathResult result = model.solveMath(input);
        
        assertNotNull(result, "数学结果应存在");
        assertNotNull(result.logits, "logits应存在");
        assertTrue(result.confidence >= 0 && result.confidence <= 1, 
                  "数学置信度应在[0,1]范围内");
        
        System.out.println("数学置信度: " + result.confidence);
    }
    
    @Test
    public void testGenerateSequence() {
        // 测试序列生成
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] prompt = {{1, 2, 3}};
        NdArray promptIds = NdArray.of(prompt);
        
        int maxNewTokens = 5;
        NdArray generated = model.generateSequence(promptIds, maxNewTokens, TaskType.GENERAL);
        
        assertNotNull(generated, "生成序列应存在");
        assertEquals(2, generated.getShape().getDimNum(), 
                    "生成序列应为2维");
        assertEquals(1, generated.getShape().getDimension(0), 
                    "batch维度应为1");
        assertEquals(3 + maxNewTokens, generated.getShape().getDimension(1), 
                    "序列长度应为prompt长度+新生成token数");
    }
    
    @Test
    public void testGenerateSequenceWithDifferentTaskTypes() {
        // 测试不同任务类型的序列生成
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] prompt = {{1, 2}};
        NdArray promptIds = NdArray.of(prompt);
        int maxTokens = 3;
        
        // 测试不同任务类型
        TaskType[] taskTypes = {
            TaskType.REASONING,
            TaskType.CODING,
            TaskType.MATH,
            TaskType.GENERAL
        };
        
        for (TaskType taskType : taskTypes) {
            NdArray generated = model.generateSequence(promptIds, maxTokens, taskType);
            assertNotNull(generated, 
                         "任务类型 " + taskType + " 的生成应成功");
            assertEquals(2 + maxTokens, generated.getShape().getDimension(1), 
                        "生成长度应正确");
        }
    }
    
    @Test
    public void testModelInfo() {
        // 测试模型信息输出
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        // 打印模型信息（不应抛出异常）
        assertDoesNotThrow(() -> model.printModelInfo(), 
                          "打印模型信息不应抛出异常");
        
        // 验证配置摘要包含关键字
        String summary = model.getConfigSummary();
        assertTrue(summary.contains("DeepSeek-V3"), "摘要应包含模型名称");
        assertTrue(summary.contains("参数"), "摘要应包含参数量信息");
        assertTrue(summary.contains("专家"), "摘要应包含专家信息");
    }
    
    @Test
    public void testModelWithDifferentConfigs() {
        // 测试使用不同配置创建模型
        DeepSeekV3Config config1 = DeepSeekV3Config.createTinyConfig();
        DeepSeekV3Config config2 = DeepSeekV3Config.createSmallConfig();
        
        DeepSeekV3Model model1 = new DeepSeekV3Model("model1", config1);
        DeepSeekV3Model model2 = new DeepSeekV3Model("model2", config2);
        
        assertNotNull(model1, "Tiny配置模型应创建成功");
        assertNotNull(model2, "Small配置模型应创建成功");
        
        // 验证不同配置的模型参数量不同
        String desc1 = model1.getConfigSummary();
        String desc2 = model2.getConfigSummary();
        assertNotEquals(desc1, desc2, "不同配置的模型摘要应不同");
    }
    
    @Test
    public void testForwardMethod() {
        // 测试Model基类的forward方法
        DeepSeekV3Model model = new DeepSeekV3Model("test-model", tinyConfig);
        
        float[][] inputData = {{1, 2, 3}};
        Variable input = new Variable(NdArray.of(inputData));
        
        // 调用forward方法（predict内部调用forward）
        Variable output = model.forward(input);
        
        assertNotNull(output, "forward输出应存在");
        assertNotNull(output.getValue(), "forward输出值应存在");
    }
    
    @Test
    public void testTaskTypeEnum() {
        // 测试TaskType枚举
        TaskType[] taskTypes = TaskType.values();
        
        assertTrue(taskTypes.length >= 5, 
                  "应至少有5种任务类型");
        
        // 验证所有任务类型
        assertNotNull(TaskType.REASONING, "REASONING任务类型应存在");
        assertNotNull(TaskType.CODING, "CODING任务类型应存在");
        assertNotNull(TaskType.MATH, "MATH任务类型应存在");
        assertNotNull(TaskType.GENERAL, "GENERAL任务类型应存在");
        assertNotNull(TaskType.MULTIMODAL, "MULTIMODAL任务类型应存在");
    }
}
