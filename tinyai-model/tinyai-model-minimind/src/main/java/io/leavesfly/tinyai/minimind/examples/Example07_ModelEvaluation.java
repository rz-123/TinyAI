package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;

/**
 * 示例07: 模型评估示例
 * 
 * 本示例展示如何评估MiniMind模型的性能
 * 
 * @author leavesfly
 * @since 2024
 */
public class Example07_ModelEvaluation {
    
    public static void main(String[] args) {
        System.out.println("=== 模型评估示例 ===\n");
        
        // 1. 加载模型
        System.out.println("1. 加载模型");
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        MiniMindModel model = new MiniMindModel("minimind-eval", config);
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), 
            config.getMaxSeqLen()
        );
        
        System.out.println("模型: " + config.getModelSize());
        System.out.println("参数量: ~" + (config.estimateParameters() / 1_000_000) + "M");
        
        // 2. 评估指标概述
        System.out.println("\n2. 评估指标");
        printEvaluationMetrics();
        
        // 3. 困惑度计算
        System.out.println("\n3. 困惑度(Perplexity)计算");
        demonstratePerplexity(model, tokenizer);
        
        // 4. 生成质量评估
        System.out.println("\n4. 生成质量评估");
        evaluateGenerationQuality();
        
        // 5. 性能基准测试
        System.out.println("\n5. 性能基准测试");
        benchmarkPerformance(model, tokenizer, config);
        
        System.out.println("\n=== 模型评估完成 ===");
    }
    
    /**
     * 打印评估指标
     */
    private static void printEvaluationMetrics() {
        System.out.println("常用评估指标:");
        
        System.out.println("\n1. 困惑度(Perplexity)");
        System.out.println("  - 定义: exp(avg_loss)");
        System.out.println("  - 含义: 模型预测的不确定性");
        System.out.println("  - 越低越好");
        
        System.out.println("\n2. 准确率(Accuracy)");
        System.out.println("  - 定义: 正确预测token比例");
        System.out.println("  - Top-1准确率");
        System.out.println("  - Top-K准确率");
        
        System.out.println("\n3. 生成质量");
        System.out.println("  - 流畅性");
        System.out.println("  - 连贯性");
        System.out.println("  - 相关性");
        System.out.println("  - 多样性");
        
        System.out.println("\n4. 任务特定指标");
        System.out.println("  - BLEU (翻译)");
        System.out.println("  - ROUGE (摘要)");
        System.out.println("  - F1-Score (分类)");
    }
    
    /**
     * 演示困惑度计算
     */
    private static void demonstratePerplexity(MiniMindModel model, MiniMindTokenizer tokenizer) {
        // 测试文本
        String testText = "深度学习是机器学习的一个重要分支";
        
        System.out.println("测试文本: " + testText);
        
        // 编码
        List<Integer> tokenIds = tokenizer.encode(testText, false, false);
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
        model.setTraining(false);
        Variable output = model.predict(inputVar);
        
        // 困惑度计算说明
        System.out.println("\n困惑度计算步骤:");
        System.out.println("  1. 计算每个token的交叉熵损失");
        System.out.println("  2. 求平均损失: avg_loss = sum(loss) / N");
        System.out.println("  3. 计算困惑度: PPL = exp(avg_loss)");
        System.out.println("  4. 示例: avg_loss=3.0 -> PPL=20.09");
        
        System.out.println("\n困惑度参考值:");
        System.out.println("  - PPL < 20: 优秀");
        System.out.println("  - PPL 20-50: 良好");
        System.out.println("  - PPL 50-100: 一般");
        System.out.println("  - PPL > 100: 需要改进");
    }
    
    /**
     * 评估生成质量
     */
    private static void evaluateGenerationQuality() {
        System.out.println("生成质量评估维度:");
        
        System.out.println("\n1. 流畅性(Fluency)");
        System.out.println("  - 语法正确性");
        System.out.println("  - 语句通顺性");
        System.out.println("  - 评估方法: 人工评分/语言模型打分");
        
        System.out.println("\n2. 连贯性(Coherence)");
        System.out.println("  - 上下文一致");
        System.out.println("  - 逻辑连贯");
        System.out.println("  - 评估方法: 人工判断/连贯性模型");
        
        System.out.println("\n3. 相关性(Relevance)");
        System.out.println("  - 与输入提示相关");
        System.out.println("  - 符合任务要求");
        System.out.println("  - 评估方法: 相似度计算");
        
        System.out.println("\n4. 多样性(Diversity)");
        System.out.println("  - 词汇丰富度");
        System.out.println("  - 句式变化");
        System.out.println("  - 评估方法: Distinct-N指标");
        
        System.out.println("\n5. 安全性(Safety)");
        System.out.println("  - 无有害内容");
        System.out.println("  - 无偏见歧视");
        System.out.println("  - 评估方法: 安全分类器");
    }
    
    /**
     * 性能基准测试
     */
    private static void benchmarkPerformance(MiniMindModel model, 
                                            MiniMindTokenizer tokenizer,
                                            MiniMindConfig config) {
        System.out.println("性能指标:");
        
        // 测试数据
        String testText = "人工智能";
        List<Integer> tokenIds = tokenizer.encode(testText, false, false);
        
        // 创建输入
        int[] ids = tokenIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));
        Variable inputVar = new Variable(inputArray);
        
        // 推理延迟测试
        model.setTraining(false);
        int warmupRuns = 5;
        int testRuns = 10;
        
        // 预热
        for (int i = 0; i < warmupRuns; i++) {
            model.predict(inputVar);
        }
        
        // 测试
        long startTime = System.nanoTime();
        for (int i = 0; i < testRuns; i++) {
            model.predict(inputVar);
        }
        long endTime = System.nanoTime();
        
        double avgLatency = (endTime - startTime) / (testRuns * 1_000_000.0);
        
        System.out.println("\n推理性能:");
        System.out.println("  - 序列长度: " + ids.length);
        System.out.println("  - 平均延迟: " + String.format("%.2f", avgLatency) + " ms");
        System.out.println("  - 吞吐量: " + String.format("%.2f", 1000.0 / avgLatency) + " 序列/秒");
        
        // 内存占用
        System.out.println("\n内存占用估算:");
        long params = config.estimateParameters();
        long modelSize = params * 4 / (1024 * 1024);  // float32, MB
        System.out.println("  - 参数量: " + params);
        System.out.println("  - 模型大小: ~" + modelSize + " MB (FP32)");
        System.out.println("  - 模型大小: ~" + (modelSize / 2) + " MB (FP16)");
        
        // 性能建议
        System.out.println("\n性能优化建议:");
        System.out.println("  - 使用批处理提高吞吐量");
        System.out.println("  - 使用KV-Cache加速生成");
        System.out.println("  - 考虑量化降低内存");
        System.out.println("  - 启用混合精度加速计算");
    }
    
    /**
     * 评估最佳实践
     */
    private static void printEvaluationBestPractices() {
        System.out.println("\n=== 评估最佳实践 ===");
        
        System.out.println("\n1. 评估数据集:");
        System.out.println("  - 使用独立测试集");
        System.out.println("  - 覆盖多种场景");
        System.out.println("  - 数据分布与应用一致");
        
        System.out.println("\n2. 评估频率:");
        System.out.println("  - 训练中定期评估");
        System.out.println("  - 使用验证集调参");
        System.out.println("  - 最终用测试集报告");
        
        System.out.println("\n3. 多维度评估:");
        System.out.println("  - 自动指标(PPL、BLEU等)");
        System.out.println("  - 人工评估(质量、安全性)");
        System.out.println("  - A/B测试(实际应用)");
        
        System.out.println("\n4. 基准对比:");
        System.out.println("  - 与baseline模型对比");
        System.out.println("  - 与SOTA模型对比");
        System.out.println("  - 与人类水平对比");
    }
    
    /**
     * 常见问题诊断
     */
    private static void printTroubleshooting() {
        System.out.println("\n=== 常见问题诊断 ===");
        
        System.out.println("\n1. 高困惑度:");
        System.out.println("  - 可能原因: 训练不足、数据不匹配");
        System.out.println("  - 解决方案: 增加训练、检查数据");
        
        System.out.println("\n2. 过拟合:");
        System.out.println("  - 表现: 训练集好、测试集差");
        System.out.println("  - 解决方案: 正则化、增加数据");
        
        System.out.println("\n3. 欠拟合:");
        System.out.println("  - 表现: 训练集和测试集都差");
        System.out.println("  - 解决方案: 增大模型、更多训练");
        
        System.out.println("\n4. 生成质量差:");
        System.out.println("  - 可能原因: 解码策略不当");
        System.out.println("  - 解决方案: 调整温度、top-k/p参数");
    }
}
