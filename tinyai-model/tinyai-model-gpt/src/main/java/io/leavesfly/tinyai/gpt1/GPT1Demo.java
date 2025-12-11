package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * GPT-1模型演示程序
 */
public class GPT1Demo {
    
    public static void main(String[] args) {
        System.out.println("\n" + "=".repeat(80));
        System.out.println("GPT-1 模型演示");
        System.out.println("=".repeat(80) + "\n");
        
        // 演示1: 创建Tiny模型（快速测试）
        demoTinyModel();
        
        // 演示2: 创建Small模型
        demoSmallModel();
        
        // 演示3: 创建Standard模型
        demoStandardModel();
        
        // 演示4: 自定义配置
        demoCustomConfig();
    }
    
    private static void demoTinyModel() {
        System.out.println("\n>>> 演示1: Tiny GPT-1模型 (快速测试)");
        System.out.println("-".repeat(80));
        
        GPT1Model model = GPT1Model.createTinyModel("gpt1-tiny");
        model.printModelInfo();
        
        testForwardPass(model, 1, 16);
    }
    
    private static void demoSmallModel() {
        System.out.println("\n>>> 演示2: Small GPT-1模型");
        System.out.println("-".repeat(80));
        
        GPT1Model model = GPT1Model.createSmallModel("gpt1-small");
        System.out.println(model.getConfigSummary());
        System.out.println();
        
        testForwardPass(model, 2, 32);
    }
    
    private static void demoStandardModel() {
        System.out.println("\n>>> 演示3: Standard GPT-1模型 (117M参数)");
        System.out.println("-".repeat(80));
        
        GPT1Model model = GPT1Model.createStandardModel("gpt1-117m");
        System.out.println(model.getConfigSummary());
        System.out.println();
        
        testForwardPass(model, 1, 10);
    }
    
    private static void demoCustomConfig() {
        System.out.println("\n>>> 演示4: 自定义配置");
        System.out.println("-".repeat(80));
        
        GPT1Config config = new GPT1Config();
        config.setVocabSize(5000);
        config.setNEmbd(384);
        config.setNLayer(6);
        config.setNHead(6);
        config.setNInner(1536);
        config.setNPositions(256);
        
        GPT1Model model = new GPT1Model("gpt1-custom", config);
        System.out.println("自定义模型: " + model);
        System.out.println(model.getConfigSummary());
        System.out.println();
        
        testForwardPass(model, 1, 20);
    }
    
    private static void testForwardPass(GPT1Model model, int batchSize, int seqLen) {
        System.out.println("\n前向传播测试:");
        System.out.printf("  输入形状: (batch_size=%d, seq_len=%d)\n", batchSize, seqLen);
        
        NdArray tokenIds = NdArray.of(Shape.of(batchSize, seqLen));
        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                tokenIds.set(s % 100, b, s);
            }
        }
        
        Variable input = new Variable(tokenIds);
        long startTime = System.currentTimeMillis();
        Variable output = model.forward(input);
        long endTime = System.currentTimeMillis();
        
        Shape outputShape = output.getValue().getShape();
        System.out.printf("  输出形状: (batch_size=%d, seq_len=%d, vocab_size=%d)\n",
            outputShape.getDimension(0),
            outputShape.getDimension(1),
            outputShape.getDimension(2));
        System.out.printf("  推理耗时: %d ms\n", endTime - startTime);
        System.out.println("  ✓ 前向传播成功！");
    }
    
    public static void runQuickDemo() {
        System.out.println("\n快速演示: GPT-1 Tiny模型");
        System.out.println("=".repeat(60));
        
        GPT1Model model = GPT1Model.createTinyModel("gpt1-demo");
        System.out.println(model);
        
        NdArray input = NdArray.of(Shape.of(1, 10));
        for (int i = 0; i < 10; i++) {
            input.set(i, 0, i);
        }
        
        Variable output = model.forward(new Variable(input));
        System.out.println("\n输出形状: " + output.getValue().getShape());
        System.out.println("模型运行正常！");
    }
}
