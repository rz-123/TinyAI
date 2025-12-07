package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.lora.LoRALinear;

/**
 * 示例04: LoRA微调
 * 
 * 本示例演示:
 * 1. 创建基础模型
 * 2. 应用LoRA适配器
 * 3. 只训练LoRA参数
 * 4. 合并LoRA权重
 * 
 * @author leavesfly
 */
public class Example04_LoRAFineTuning {
    
    public static void main(String[] args) {
        System.out.println("=== LoRA微调示例 ===\n");
        
        // 1. 创建基础模型
        System.out.println("1. 创建基础模型");
        MiniMindConfig config = createConfig();
        MiniMindModel model = new MiniMindModel("minimind", config);
        
        long totalParams = config.estimateParameters();
        System.out.println("基础模型参数量: " + totalParams);
        
        // 2. LoRA配置
        System.out.println("\n2. LoRA配置");
        int rank = 8;  // LoRA秩
        float alpha = 16.0f;  // 缩放因子
        
        System.out.println("LoRA秩(rank): " + rank);
        System.out.println("缩放因子(alpha): " + alpha);
        
        // 3. 创建LoRA层示例
        System.out.println("\n3. 创建LoRA层");
        int inputDim = config.getHiddenSize();
        int outputDim = config.getHiddenSize();
        
        LoRALinear loraLayer = new LoRALinear(
            "lora_example",
            inputDim,
            outputDim,
            false,  // 不使用bias
            rank,
            alpha
        );
        
        System.out.println("LoRA层创建成功");
        System.out.println("输入维度: " + inputDim);
        System.out.println("输出维度: " + outputDim);
        
        // 4. 参数效率分析
        System.out.println("\n4. 参数效率分析");
        long originalParams = inputDim * outputDim;
        long loraParams = (inputDim + outputDim) * rank;
        double reduction = 100.0 * (1.0 - (double)loraParams / originalParams);
        
        System.out.println("原始参数量: " + originalParams);
        System.out.println("LoRA参数量: " + loraParams);
        System.out.println("参数减少: " + String.format("%.1f%%", reduction));
        
        // 5. 应用LoRA到模型
        System.out.println("\n5. LoRA应用策略");
        System.out.println("推荐应用位置:");
        System.out.println("  - Query投影层 (wq)");
        System.out.println("  - Value投影层 (wv)");
        System.out.println("  - 可选: Key投影层 (wk)");
        System.out.println("  - 可选: Output投影层 (wo)");
        
        // 6. 训练流程说明
        System.out.println("\n6. LoRA训练流程");
        System.out.println("步骤:");
        System.out.println("  1. 冻结基础模型参数");
        System.out.println("  2. 只训练LoRA参数 (A和B矩阵)");
        System.out.println("  3. 前向传播: output = W*x + (B*A)*x");
        System.out.println("  4. 反向传播只更新A和B");
        
        // 7. 权重合并
        System.out.println("\n7. 推理时权重合并");
        System.out.println("合并公式: W' = W + B*A*α/r");
        System.out.println("优势: 推理时无额外计算开销");
        
        // 8. 使用建议
        System.out.println("\n8. 使用建议");
        System.out.println("Rank选择:");
        System.out.println("  - 简单任务: rank=4~8");
        System.out.println("  - 复杂任务: rank=16~32");
        System.out.println("  - 通用推荐: rank=8");
        System.out.println("\nAlpha选择:");
        System.out.println("  - 通常设置为 alpha = 2 * rank");
        System.out.println("  - 较大alpha增强LoRA影响");
        
        System.out.println("\n=== 示例完成 ===");
    }
    
    private static MiniMindConfig createConfig() {
        MiniMindConfig config = new MiniMindConfig();
        config.setVocabSize(512);
        config.setHiddenSize(64);
        config.setNumLayers(2);
        config.setNumHeads(2);
        config.setFfnHiddenSize(128);
        return config;
    }
}
