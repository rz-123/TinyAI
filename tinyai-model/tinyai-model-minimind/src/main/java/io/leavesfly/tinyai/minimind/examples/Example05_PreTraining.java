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
 * 示例05: 预训练示例
 * 
 * 本示例展示如何从头开始预训练MiniMind模型
 * 
 * @author leavesfly
 * @since 2024
 */
public class Example05_PreTraining {
    
    public static void main(String[] args) {
        System.out.println("=== 预训练示例 ===\n");
        
        // 1. 创建模型配置
        System.out.println("1. 创建模型配置");
        MiniMindConfig config = createPreTrainingConfig();
        printConfig(config);
        
        // 2. 创建模型
        System.out.println("\n2. 创建模型");
        MiniMindModel model = new MiniMindModel("minimind-pretrain", config);
        System.out.println("模型创建成功");
        System.out.println("参数量: " + config.estimateParameters() + 
                          " (~" + (config.estimateParameters() / 1_000_000) + "M)");
        
        // 3. 准备训练数据
        System.out.println("\n3. 准备训练数据");
        preparePreTrainingData();
        
        // 4. 训练配置
        System.out.println("\n4. 预训练配置");
        printPreTrainingConfig();
        
        // 5. 演示训练流程
        System.out.println("\n5. 训练流程演示");
        demonstrateTrainingProcess(model, config);
        
        System.out.println("\n=== 预训练示例完成 ===");
    }
    
    /**
     * 创建预训练配置
     */
    private static MiniMindConfig createPreTrainingConfig() {
        MiniMindConfig config = new MiniMindConfig();
        
        // 模型架构
        config.setVocabSize(6400);      // 词汇表大小
        config.setMaxSeqLen(512);       // 最大序列长度
        config.setHiddenSize(512);      // 隐藏维度
        config.setNumLayers(8);         // Transformer层数
        config.setNumHeads(8);          // 注意力头数
        config.setFfnHiddenSize(2048);  // FFN隐藏维度
        config.setDropout(0.1f);        // Dropout率
        
        // 位置编码
        config.setUseRoPE(true);        // 使用RoPE
        
        // 激活函数
        config.setActivationFunction("gelu");
        
        return config;
    }
    
    /**
     * 打印配置信息
     */
    private static void printConfig(MiniMindConfig config) {
        System.out.println("预训练模型配置:");
        System.out.println("  - 词汇表: " + config.getVocabSize());
        System.out.println("  - 序列长度: " + config.getMaxSeqLen());
        System.out.println("  - 隐藏维度: " + config.getHiddenSize());
        System.out.println("  - 层数: " + config.getNumLayers());
        System.out.println("  - 注意力头: " + config.getNumHeads());
        System.out.println("  - 头维度: " + (config.getHiddenSize() / config.getNumHeads()));
        System.out.println("  - FFN维度: " + config.getFfnHiddenSize());
        System.out.println("  - Dropout: " + config.getDropout());
    }
    
    /**
     * 准备预训练数据
     */
    private static void preparePreTrainingData() {
        System.out.println("预训练数据准备:");
        
        // 数据来源
        System.out.println("\n数据来源:");
        System.out.println("  1. 大规模文本语料");
        System.out.println("  2. 书籍、文章、网页等");
        System.out.println("  3. 代码库(如需要)");
        
        // 数据处理
        System.out.println("\n数据处理流程:");
        System.out.println("  1. 文本清洗(去除噪声、标准化)");
        System.out.println("  2. Tokenization(BPE/WordPiece)");
        System.out.println("  3. 序列切分(固定长度)");
        System.out.println("  4. 数据打乱(Shuffle)");
        
        // 数据统计
        System.out.println("\n数据规模示例:");
        System.out.println("  - 文本量: 10GB+ 原始文本");
        System.out.println("  - Token数: 约2B tokens");
        System.out.println("  - 训练样本: 约4M 序列");
    }
    
    /**
     * 打印预训练配置
     */
    private static void printPreTrainingConfig() {
        System.out.println("预训练超参数:");
        System.out.println("  - 学习率: 1e-3 (warm-up后)");
        System.out.println("  - Warm-up步数: 2000");
        System.out.println("  - Batch大小: 32-64");
        System.out.println("  - 累积梯度: 4-8步");
        System.out.println("  - 有效Batch: 128-512");
        System.out.println("  - 训练步数: 100K-500K");
        System.out.println("  - 优化器: AdamW");
        System.out.println("  - Beta1: 0.9, Beta2: 0.95");
        System.out.println("  - 权重衰减: 0.1");
        System.out.println("  - 梯度裁剪: 1.0");
        
        System.out.println("\n学习率调度:");
        System.out.println("  - Warm-up: Linear (0 -> max_lr)");
        System.out.println("  - Decay: Cosine (max_lr -> min_lr)");
        System.out.println("  - 最小学习率: 1e-5");
    }
    
    /**
     * 演示训练过程
     */
    private static void demonstrateTrainingProcess(MiniMindModel model, MiniMindConfig config) {
        // 创建tokenizer
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), 
            config.getMaxSeqLen()
        );
        
        // 模拟训练样本
        String sample = "深度学习是机器学习的一个分支,使用神经网络进行特征学习";
        List<Integer> tokenIds = tokenizer.encode(sample, false, false);
        
        System.out.println("训练样本: " + sample);
        System.out.println("Token数量: " + tokenIds.size());
        
        // 创建输入
        int[] ids = tokenIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));
        
        // 前向传播
        Variable inputVar = new Variable(inputArray);
        model.setTraining(true);
        Variable output = model.forward(inputVar);
        
        int[] outputShape = output.getValue().getShape().getShapeDims();
        System.out.println("输出形状: [batch=" + outputShape[0] + 
                          ", seq_len=" + outputShape[1] + 
                          ", vocab_size=" + outputShape[2] + "]");
        
        model.setTraining(false);
        
        // 训练步骤说明
        System.out.println("\n训练步骤:");
        System.out.println("  1. 加载批次数据");
        System.out.println("  2. 前向传播计算logits");
        System.out.println("  3. 计算交叉熵损失");
        System.out.println("  4. 反向传播计算梯度");
        System.out.println("  5. 梯度裁剪");
        System.out.println("  6. 优化器更新参数");
        System.out.println("  7. 学习率调度");
        System.out.println("  8. 记录日志和指标");
    }
    
    /**
     * 训练监控
     */
    private static void printTrainingMonitoring() {
        System.out.println("\n=== 训练监控 ===");
        
        System.out.println("\n关键指标:");
        System.out.println("  - 训练损失(Loss)");
        System.out.println("  - 验证损失");
        System.out.println("  - 困惑度(Perplexity)");
        System.out.println("  - 梯度范数");
        System.out.println("  - 学习率");
        
        System.out.println("\n检查点策略:");
        System.out.println("  - 每N步保存一次");
        System.out.println("  - 保留最近K个检查点");
        System.out.println("  - 保存最佳验证loss模型");
        
        System.out.println("\n评估策略:");
        System.out.println("  - 每N步在验证集评估");
        System.out.println("  - 计算困惑度");
        System.out.println("  - 生成样本文本检查");
    }
    
    /**
     * 优化技巧
     */
    private static void printOptimizationTips() {
        System.out.println("\n=== 优化技巧 ===");
        
        System.out.println("\n1. 混合精度训练:");
        System.out.println("  - 使用FP16加速");
        System.out.println("  - 减少显存占用");
        System.out.println("  - 注意数值稳定性");
        
        System.out.println("\n2. 梯度累积:");
        System.out.println("  - 模拟大batch训练");
        System.out.println("  - 降低显存需求");
        System.out.println("  - 提升训练稳定性");
        
        System.out.println("\n3. 梯度检查点:");
        System.out.println("  - 减少显存占用");
        System.out.println("  - 允许更大模型");
        System.out.println("  - 略微降低速度");
        
        System.out.println("\n4. 数据并行:");
        System.out.println("  - 多GPU训练");
        System.out.println("  - 线性加速");
        System.out.println("  - 同步梯度更新");
    }
}
