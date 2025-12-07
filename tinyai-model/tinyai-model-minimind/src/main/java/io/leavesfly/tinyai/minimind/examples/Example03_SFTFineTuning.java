package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * 示例03: 监督微调(SFT)示例
 * 
 * 本示例展示如何在预训练模型基础上进行监督微调(Supervised Fine-Tuning)
 * 
 * @author leavesfly
 * @since 2024
 */
public class Example03_SFTFineTuning {
    
    public static void main(String[] args) {
        System.out.println("=== 监督微调(SFT)示例 ===\n");
        
        // 1. 加载预训练模型
        System.out.println("1. 加载预训练模型");
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        MiniMindModel model = new MiniMindModel("minimind-sft", config);
        
        System.out.println("模型: " + config.getModelSize());
        System.out.println("参数量: " + config.estimateParameters() + " (~" + 
                          (config.estimateParameters() / 1_000_000) + "M)");
        
        // 2. 准备SFT数据集
        System.out.println("\n2. 准备SFT数据集");
        List<String> sftData = prepareSFTDataset();
        System.out.println("SFT样本数: " + sftData.size());
        
        // 3. 创建Tokenizer
        System.out.println("\n3. 创建Tokenizer");
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), 
            config.getMaxSeqLen()
        );
        
        // 4. 数据预处理
        System.out.println("\n4. 数据预处理");
        preprocessData(tokenizer, sftData);
        
        // 5. 训练配置
        System.out.println("\n5. 训练配置");
        printTrainingConfig();
        
        // 6. 模拟训练过程
        System.out.println("\n6. 训练过程演示");
        demonstrateTrainingStep(model, tokenizer, sftData.get(0));
        
        System.out.println("\n=== SFT微调示例完成 ===");
    }
    
    /**
     * 准备SFT数据集
     * SFT数据通常是问答对、指令-响应对等
     */
    private static List<String> prepareSFTDataset() {
        List<String> dataset = new ArrayList<>();
        
        // 示例: 问答对格式
        dataset.add("问:什么是深度学习? 答:深度学习是机器学习的一个分支");
        dataset.add("问:什么是神经网络? 答:神经网络是由多层节点组成的计算模型");
        dataset.add("问:什么是梯度下降? 答:梯度下降是一种优化算法");
        
        // 示例: 指令格式
        dataset.add("指令:解释Transformer架构 响应:Transformer是基于自注意力机制的模型");
        dataset.add("指令:描述反向传播 响应:反向传播是训练神经网络的核心算法");
        
        return dataset;
    }
    
    /**
     * 数据预处理
     */
    private static void preprocessData(MiniMindTokenizer tokenizer, List<String> data) {
        System.out.println("预处理SFT数据...");
        
        int maxLen = 0;
        int totalTokens = 0;
        
        for (String text : data) {
            List<Integer> tokens = tokenizer.encode(text, false, false);
            maxLen = Math.max(maxLen, tokens.size());
            totalTokens += tokens.size();
        }
        
        System.out.println("  - 样本数: " + data.size());
        System.out.println("  - 最大序列长度: " + maxLen);
        System.out.println("  - 平均序列长度: " + (totalTokens / data.size()));
        System.out.println("  - 总token数: " + totalTokens);
    }
    
    /**
     * 打印训练配置
     */
    private static void printTrainingConfig() {
        System.out.println("SFT训练参数:");
        System.out.println("  - 学习率: 1e-4 (较预训练更小)");
        System.out.println("  - Batch大小: 8");
        System.out.println("  - 训练轮数: 3-5");
        System.out.println("  - 优化器: AdamW");
        System.out.println("  - 学习率调度: Cosine");
        System.out.println("  - 梯度裁剪: 1.0");
        System.out.println("  - 权重衰减: 0.01");
        
        System.out.println("\nSFT特殊配置:");
        System.out.println("  - 仅微调部分层: 可选");
        System.out.println("  - 使用LoRA: 推荐");
        System.out.println("  - 冻结Embedding: 可选");
    }
    
    /**
     * 演示单步训练
     */
    private static void demonstrateTrainingStep(MiniMindModel model, 
                                                MiniMindTokenizer tokenizer,
                                                String sample) {
        System.out.println("训练样本: " + sample);
        
        // 1. 编码
        List<Integer> tokenIds = tokenizer.encode(sample, false, false);
        System.out.println("Token IDs: " + tokenIds.subList(0, Math.min(10, tokenIds.size())) + "...");
        
        // 2. 创建输入
        int[] ids = tokenIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));
        
        // 3. 前向传播
        Variable inputVar = new Variable(inputArray);
        model.setTraining(true);  // 切换到训练模式
        Variable output = model.forward(inputVar);
        
        // 4. 输出信息
        int[] outputShape = output.getValue().getShape().getShapeDims();
        System.out.println("输出形状: [" + outputShape[0] + ", " + 
                          outputShape[1] + ", " + outputShape[2] + "]");
        
        // 5. 注意事项
        System.out.println("\n训练注意事项:");
        System.out.println("  1. 使用较小的学习率避免灾难性遗忘");
        System.out.println("  2. 监控验证集性能防止过拟合");
        System.out.println("  3. 保存多个检查点便于选择");
        System.out.println("  4. 考虑使用LoRA减少参数量");
        
        model.setTraining(false);  // 切回评估模式
    }
    
    /**
     * SFT最佳实践
     */
    private static void printBestPractices() {
        System.out.println("\n=== SFT最佳实践 ===");
        
        System.out.println("\n1. 数据准备:");
        System.out.println("  - 高质量标注数据");
        System.out.println("  - 数据格式统一");
        System.out.println("  - 适当的数据增强");
        
        System.out.println("\n2. 训练策略:");
        System.out.println("  - 逐层解冻微调");
        System.out.println("  - 差异化学习率");
        System.out.println("  - 早停策略");
        
        System.out.println("\n3. 评估指标:");
        System.out.println("  - 困惑度(Perplexity)");
        System.out.println("  - 任务相关指标");
        System.out.println("  - 人工评估");
        
        System.out.println("\n4. 常见问题:");
        System.out.println("  - 灾难性遗忘: 降低学习率、使用正则化");
        System.out.println("  - 过拟合: 增加数据、使用Dropout");
        System.out.println("  - 训练不稳定: 梯度裁剪、调整batch size");
    }
}
