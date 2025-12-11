package io.leavesfly.tinyai.gpt1.training;

import io.leavesfly.tinyai.gpt1.GPT1Config;
import io.leavesfly.tinyai.gpt1.GPT1Model;

import java.util.Arrays;
import java.util.List;

/**
 * GPT-1训练和推理完整演示
 * 
 * 展示完整的训练流程:
 * 1. 预训练(Pretrain)
 * 2. 微调(Finetune/Posttrain)
 * 3. 推理(Inference)
 * 
 * @author TinyAI
 * @since 2024
 */
public class GPT1TrainDemo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(70));
        System.out.println("GPT-1 完整训练与推理演示");
        System.out.println("=".repeat(70));
        
        // 演示1: 预训练
        demoPretraining();
        
        // 演示2: 微调
        demoFinetuning();
        
        // 演示3: 推理
        demoInference();
        
        System.out.println("\n演示完成!");
    }
    
    /**
     * 演示1: 预训练流程
     */
    private static void demoPretraining() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("演示1: GPT-1预训练 (Pretrain)");
        System.out.println("=".repeat(70));
        
        // 1. 创建模型(使用Tiny配置用于演示)
        System.out.println("\n步骤1: 创建模型");
        GPT1Config config = GPT1Config.createTinyConfig();
        GPT1Model model = new GPT1Model("gpt1-pretrain-demo", config);
        System.out.println("✓ 模型创建成功");
        System.out.println("  - 配置: Tiny");
        System.out.println("  - 参数量: ~" + config.estimateParameterCount() / 1_000_000 + "M");
        
        // 2. 准备数据集
        System.out.println("\n步骤2: 准备预训练数据");
        GPT1Dataset.SimpleTokenizer tokenizer = new GPT1Dataset.SimpleTokenizer();
        
        // 示例文本数据
        List<String> texts = Arrays.asList(
            "Deep learning is a subset of machine learning that uses neural networks",
            "Artificial intelligence is transforming the world",
            "Natural language processing enables computers to understand human language",
            "Transformer architecture revolutionized the field of NLP",
            "Large language models can generate coherent text"
        );
        
        GPT1Dataset dataset = new GPT1Dataset(
            config.getNPositions(),  // maxSeqLen
            2,                       // batchSize(小批次用于演示)
            tokenizer.getVocabSize()
        );
        
        dataset.loadFromTexts(texts, tokenizer);
        System.out.println("✓ 数据加载完成");
        System.out.println("  - 样本数: " + dataset.getSampleCount());
        System.out.println("  - 批次数: " + dataset.getBatchCount());
        
        // 3. 配置并开始预训练
        System.out.println("\n步骤3: 开始预训练");
        GPT1Pretrain trainer = new GPT1Pretrain(model, dataset);
        trainer.configure(
            2,        // maxEpochs(演示用小epoch)
            1e-3f,    // learningRate
            100,      // warmupSteps
            1.0f      // maxGradNorm
        ).setCheckpoint("./checkpoints/pretrain_demo", 500);
        
        // 注意: 实际训练注释掉,仅演示流程
        // trainer.train();
        
        System.out.println("✓ 预训练配置完成(实际训练已跳过)");
        System.out.println("\n预训练阶段说明:");
        System.out.println("  - 目标: 学习语言的通用模式");
        System.out.println("  - 数据: 大规模无标注文本");
        System.out.println("  - 损失: 因果语言建模损失");
        System.out.println("  - 学习率: 2.5e-4 (warmup + cosine decay)");
    }
    
    /**
     * 演示2: 微调流程
     */
    private static void demoFinetuning() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("演示2: GPT-1微调 (Finetune/Posttrain)");
        System.out.println("=".repeat(70));
        
        // 1. 加载预训练模型
        System.out.println("\n步骤1: 加载预训练模型");
        GPT1Config config = GPT1Config.createTinyConfig();
        GPT1Model model = new GPT1Model("gpt1-finetune-demo", config);
        System.out.println("✓ 预训练模型加载完成");
        
        // 2. 准备微调数据
        System.out.println("\n步骤2: 准备微调数据");
        GPT1Dataset.SimpleTokenizer tokenizer = new GPT1Dataset.SimpleTokenizer();
        
        // 训练集
        List<String> trainTexts = Arrays.asList(
            "Question: What is deep learning? Answer: Deep learning is a type of machine learning",
            "Question: What is NLP? Answer: NLP stands for natural language processing"
        );
        
        // 验证集
        List<String> valTexts = Arrays.asList(
            "Question: What is AI? Answer: AI is artificial intelligence"
        );
        
        GPT1Dataset trainDataset = new GPT1Dataset(
            config.getNPositions(), 2, tokenizer.getVocabSize()
        );
        trainDataset.loadFromTexts(trainTexts, tokenizer);
        
        GPT1Dataset valDataset = new GPT1Dataset(
            config.getNPositions(), 1, tokenizer.getVocabSize()
        );
        valDataset.loadFromTexts(valTexts, tokenizer);
        
        System.out.println("✓ 微调数据准备完成");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        
        // 3. 配置并开始微调
        System.out.println("\n步骤3: 开始微调");
        GPT1Finetune finetuner = new GPT1Finetune(model, trainDataset, valDataset);
        finetuner.configure(
            3,        // maxEpochs
            1e-4f,    // learningRate(比预训练小10倍)
            2         // patience
        ).setCheckpoint("./checkpoints/finetune_demo", 50);
        
        // 注意: 实际训练注释掉,仅演示流程
        // finetuner.train();
        
        System.out.println("✓ 微调配置完成(实际训练已跳过)");
        System.out.println("\n微调阶段说明:");
        System.out.println("  - 目标: 适应特定任务");
        System.out.println("  - 数据: 任务相关的标注数据");
        System.out.println("  - 损失: 任务特定损失");
        System.out.println("  - 学习率: 2.5e-5 (比预训练小10倍)");
        System.out.println("  - 技巧: 早停机制防止过拟合");
    }
    
    /**
     * 演示3: 推理流程
     */
    private static void demoInference() {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("演示3: GPT-1推理与文本生成");
        System.out.println("=".repeat(70));
        
        // 1. 加载训练好的模型
        System.out.println("\n步骤1: 加载模型");
        GPT1Config config = GPT1Config.createTinyConfig();
        GPT1Model model = new GPT1Model("gpt1-inference-demo", config);
        GPT1Inference inference = new GPT1Inference(model);
        System.out.println("✓ 模型加载完成");
        
        // 2. 准备提示词
        System.out.println("\n步骤2: 准备提示词");
        int[] promptIds = {1, 2, 3, 4, 5};  // 示例token序列
        System.out.println("✓ 提示词: " + Arrays.toString(promptIds));
        
        // 3. 展示不同的生成策略
        System.out.println("\n步骤3: 多种生成策略演示\n");
        
        // 策略1: 贪婪解码
        System.out.println("策略1: 贪婪解码 (Greedy Decoding)");
        System.out.println("  - 特点: 始终选择概率最高的token");
        System.out.println("  - 优点: 确定性输出,适合需要一致性的任务");
        System.out.println("  - 缺点: 可能陷入重复模式");
        // int[] greedyResult = inference.generateGreedy(promptIds, 10);
        System.out.println("  - 示例: [已跳过实际生成]");
        
        // 策略2: Temperature采样
        System.out.println("\n策略2: Temperature采样");
        System.out.println("  - 参数: temperature=0.8");
        System.out.println("  - 特点: 控制输出的随机性");
        System.out.println("  - temperature<1: 更确定");
        System.out.println("  - temperature>1: 更随机");
        // int[] tempResult = inference.generateWithTemperature(promptIds, 10, 0.8f);
        System.out.println("  - 示例: [已跳过实际生成]");
        
        // 策略3: Top-K采样
        System.out.println("\n策略3: Top-K采样");
        System.out.println("  - 参数: k=40, temperature=1.0");
        System.out.println("  - 特点: 只从概率最高的K个token中采样");
        System.out.println("  - 优点: 避免采样到低概率token");
        // int[] topKResult = inference.generateTopK(promptIds, 10, 40, 1.0f);
        System.out.println("  - 示例: [已跳过实际生成]");
        
        // 策略4: Top-P采样
        System.out.println("\n策略4: Top-P (Nucleus) 采样");
        System.out.println("  - 参数: p=0.9, temperature=1.0");
        System.out.println("  - 特点: 从累积概率达到p的最小token集合中采样");
        System.out.println("  - 优点: 动态调整候选集大小");
        // int[] topPResult = inference.generateTopP(promptIds, 10, 0.9f, 1.0f);
        System.out.println("  - 示例: [已跳过实际生成]");
        
        // 策略5: Beam Search
        System.out.println("\n策略5: Beam Search");
        System.out.println("  - 参数: beamSize=5");
        System.out.println("  - 特点: 维护多个候选序列,选择全局最优");
        System.out.println("  - 优点: 生成质量高");
        System.out.println("  - 缺点: 计算开销大");
        // int[] beamResult = inference.generateBeamSearch(promptIds, 10, 5);
        System.out.println("  - 示例: [已跳过实际生成]");
        
        System.out.println("\n推理阶段说明:");
        System.out.println("  - 输入: 提示词token序列");
        System.out.println("  - 输出: 生成的token序列");
        System.out.println("  - 策略选择:");
        System.out.println("    * 需要确定性: 贪婪解码");
        System.out.println("    * 平衡质量与多样性: Top-P采样");
        System.out.println("    * 最高质量: Beam Search");
        System.out.println("    * 创造性任务: 高temperature的采样");
    }
    
    /**
     * 完整流程演示
     */
    public static void runCompleteWorkflow() {
        System.out.println("=".repeat(70));
        System.out.println("GPT-1 完整训练流程");
        System.out.println("=".repeat(70));
        
        // 阶段1: 预训练
        System.out.println("\n阶段1: 预训练 (Pretrain)");
        System.out.println("  目标: 学习语言的通用表示");
        System.out.println("  数据: BooksCorpus (7000本书籍)");
        System.out.println("  任务: 因果语言建模 (预测下一个词)");
        System.out.println("  耗时: 约30天 (8个GPU)");
        
        // 阶段2: 微调
        System.out.println("\n阶段2: 微调 (Finetune/Posttrain)");
        System.out.println("  目标: 适应下游任务");
        System.out.println("  数据: 任务特定数据集");
        System.out.println("  任务: 文本分类/问答/文本蕴含等");
        System.out.println("  耗时: 约3个epoch");
        
        // 阶段3: 推理
        System.out.println("\n阶段3: 推理 (Inference)");
        System.out.println("  输入: 提示词");
        System.out.println("  处理: 自回归生成");
        System.out.println("  输出: 生成文本");
        System.out.println("  速度: 毫秒级 (CPU推理)");
        
        System.out.println("\n训练提示:");
        System.out.println("  1. 预训练需要大量计算资源");
        System.out.println("  2. 微调可以在单卡上完成");
        System.out.println("  3. 使用梯度累积可以模拟更大的batch");
        System.out.println("  4. 定期保存检查点防止训练中断");
        System.out.println("  5. 监控验证集损失防止过拟合");
    }
}
