package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;

import java.util.List;

/**
 * 示例06: 文本生成策略
 * 
 * 本示例演示:
 * 1. Greedy Search (贪心搜索)
 * 2. Temperature采样
 * 3. Top-K采样
 * 4. Top-P (Nucleus)采样
 * 5. 不同生成策略对比
 * 
 * @author leavesfly
 */
public class Example06_TextGenerationStrategies {
    
    public static void main(String[] args) {
        System.out.println("=== 文本生成策略示例 ===\n");
        
        // 1. 创建小型模型用于演示
        MiniMindConfig config = createDemoConfig();
        MiniMindModel model = new MiniMindModel("minimind-demo", config);
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), config.getMaxSeqLen()
        );
        
        System.out.println("模型初始化完成");
        System.out.println("参数量: ~" + (config.estimateParameters() / 1000) + "K\n");
        
        // 2. 准备输入
        String prompt = "深度学习";
        List<Integer> promptIds = tokenizer.encode(prompt, false, false);
        int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
        
        System.out.println("输入提示: " + prompt);
        System.out.println("=".repeat(60) + "\n");
        
        // 3. 策略1: Greedy Search (temperature=0)
        System.out.println("策略1: Greedy Search (贪心搜索)");
        System.out.println("特点: 总是选择概率最高的token,结果确定");
        generateWithStrategy(model, tokenizer, promptArray, 
            "Greedy", 0.0f, 0, 0.0f);
        
        // 4. 策略2: 低温采样 (temperature=0.5)
        System.out.println("\n策略2: 低温采样 (Temperature=0.5)");
        System.out.println("特点: 较为保守,输出相对确定");
        generateWithStrategy(model, tokenizer, promptArray, 
            "Low Temp", 0.5f, 0, 0.0f);
        
        // 5. 策略3: 高温采样 (temperature=1.0)
        System.out.println("\n策略3: 高温采样 (Temperature=1.0)");
        System.out.println("特点: 增加随机性,输出更多样");
        generateWithStrategy(model, tokenizer, promptArray, 
            "High Temp", 1.0f, 0, 0.0f);
        
        // 6. 策略4: Top-K采样
        System.out.println("\n策略4: Top-K采样 (K=5)");
        System.out.println("特点: 只从前K个最可能的token中采样");
        generateWithStrategy(model, tokenizer, promptArray, 
            "Top-K", 1.0f, 5, 0.0f);
        
        // 7. 策略5: Top-P (Nucleus)采样
        System.out.println("\n策略5: Top-P采样 (P=0.9)");
        System.out.println("特点: 从累积概率达到P的token集合中采样");
        generateWithStrategy(model, tokenizer, promptArray, 
            "Top-P", 1.0f, 0, 0.9f);
        
        // 8. 策略6: 组合策略
        System.out.println("\n策略6: 组合策略 (Temp=0.8, Top-K=10, Top-P=0.95)");
        System.out.println("特点: 结合多种策略,平衡质量和多样性");
        generateWithStrategy(model, tokenizer, promptArray, 
            "Combined", 0.8f, 10, 0.95f);
        
        System.out.println("\n" + "=".repeat(60));
        System.out.println("策略选择建议:");
        System.out.println("  - 需要确定性输出: Greedy Search");
        System.out.println("  - 创意写作: 高温度 + Top-P");
        System.out.println("  - 代码生成: 低温度 + Top-K");
        System.out.println("  - 通用场景: Temperature=0.7-0.9 + Top-P=0.9");
        
        System.out.println("\n=== 示例完成 ===");
    }
    
    /**
     * 使用指定策略生成文本
     */
    private static void generateWithStrategy(
        MiniMindModel model,
        MiniMindTokenizer tokenizer,
        int[] promptArray,
        String strategyName,
        float temperature,
        int topK,
        float topP
    ) {
        try {
            // 生成
            int maxNewTokens = 10;  // 生成10个新token
            int[] generated = model.generate(
                promptArray,
                maxNewTokens,
                temperature,
                topK,
                topP
            );
            
            // 解码
            List<Integer> genIds = new java.util.ArrayList<>();
            for (int id : generated) {
                genIds.add(id);
            }
            String result = tokenizer.decode(genIds, true);
            
            System.out.println("  生成结果: " + result);
            System.out.println("  参数: temp=" + temperature + ", topK=" + topK + ", topP=" + topP);
            
        } catch (Exception e) {
            System.out.println("  [注意] 生成过程需要训练后的模型,当前使用随机初始化");
            System.out.println("  策略配置: temp=" + temperature + ", topK=" + topK + ", topP=" + topP);
        }
    }
    
    /**
     * 创建演示用配置
     */
    private static MiniMindConfig createDemoConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(512);
        config.setMaxSeqLen(128);
        config.setHiddenSize(64);
        config.setNumLayers(2);
        config.setNumHeads(2);
        config.setFfnHiddenSize(128);
        config.setDropout(0.0f);
        return config;
    }
}
