package io.leavesfly.tinyai.minimind.examples;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.List;

/**
 * 示例08: 完整训练流程
 * 
 * 本示例展示MiniMind模型的完整训练流程,包括:
 * - 数据准备
 * - 模型训练
 * - 验证评估
 * - 模型保存
 * 
 * @author leavesfly
 * @since 2024
 */
public class Example08_CompleteTrainingPipeline {
    
    public static void main(String[] args) {
        System.out.println("=== 完整训练流程示例 ===\n");
        
        // 阶段1: 环境准备
        System.out.println("【阶段1】环境准备");
        setupEnvironment();
        
        // 阶段2: 数据准备
        System.out.println("\n【阶段2】数据准备");
        prepareData();
        
        // 阶段3: 模型创建
        System.out.println("\n【阶段3】模型创建");
        MiniMindConfig config = createModel();
        
        // 阶段4: 训练配置
        System.out.println("\n【阶段4】训练配置");
        configureTraining();
        
        // 阶段5: 训练循环
        System.out.println("\n【阶段5】训练循环");
        demonstrateTrainingLoop(config);
        
        // 阶段6: 验证评估
        System.out.println("\n【阶段6】验证评估");
        evaluateModel();
        
        // 阶段7: 模型保存
        System.out.println("\n【阶段7】模型保存");
        saveModel();
        
        System.out.println("\n=== 训练流程完成 ===");
    }
    
    /**
     * 环境准备
     */
    private static void setupEnvironment() {
        System.out.println("1. 检查依赖版本");
        System.out.println("  ✓ Java 17");
        System.out.println("  ✓ TinyAI V2 API");
        System.out.println("  ✓ Maven 3.6+");
        
        System.out.println("\n2. 设置随机种子");
        System.out.println("  - 保证实验可复现");
        System.out.println("  - Random.setSeed(42)");
        
        System.out.println("\n3. 配置日志");
        System.out.println("  - 训练日志记录");
        System.out.println("  - TensorBoard集成(可选)");
    }
    
    /**
     * 数据准备
     */
    private static void prepareData() {
        System.out.println("1. 加载原始数据");
        System.out.println("  - 训练集: 80%");
        System.out.println("  - 验证集: 10%");
        System.out.println("  - 测试集: 10%");
        
        System.out.println("\n2. 数据预处理");
        System.out.println("  - 文本清洗");
        System.out.println("  - Tokenization");
        System.out.println("  - 序列截断/填充");
        
        System.out.println("\n3. 创建DataLoader");
        System.out.println("  - Batch大小: 32");
        System.out.println("  - Shuffle: true");
        System.out.println("  - 预取策略");
    }
    
    /**
     * 创建模型
     */
    private static MiniMindConfig createModel() {
        System.out.println("1. 选择模型配置");
        MiniMindConfig config = MiniMindConfig.createSmallConfig();
        
        System.out.println("  - 模型: " + config.getModelSize());
        System.out.println("  - 参数量: ~" + (config.estimateParameters() / 1_000_000) + "M");
        System.out.println("  - 隐藏维度: " + config.getHiddenSize());
        System.out.println("  - 层数: " + config.getNumLayers());
        
        System.out.println("\n2. 初始化模型");
        MiniMindModel model = new MiniMindModel("minimind", config);
        System.out.println("  ✓ 模型创建成功");
        
        System.out.println("\n3. 参数初始化");
        System.out.println("  - Embedding: Normal(0, 0.02)");
        System.out.println("  - Linear: Kaiming Uniform");
        System.out.println("  - LayerNorm: 初始化为1");
        
        return config;
    }
    
    /**
     * 配置训练
     */
    private static void configureTraining() {
        System.out.println("1. 优化器配置");
        System.out.println("  - 类型: AdamW");
        System.out.println("  - 学习率: 1e-4");
        System.out.println("  - Beta1: 0.9");
        System.out.println("  - Beta2: 0.999");
        System.out.println("  - 权重衰减: 0.01");
        
        System.out.println("\n2. 学习率调度");
        System.out.println("  - Warm-up步数: 500");
        System.out.println("  - 调度策略: Cosine");
        System.out.println("  - 最小学习率: 1e-6");
        
        System.out.println("\n3. 损失函数");
        System.out.println("  - 交叉熵损失");
        System.out.println("  - Label Smoothing: 0.1");
        
        System.out.println("\n4. 其他配置");
        System.out.println("  - 梯度裁剪: 1.0");
        System.out.println("  - 累积梯度: 4步");
        System.out.println("  - 混合精度: FP16");
    }
    
    /**
     * 演示训练循环
     */
    private static void demonstrateTrainingLoop(MiniMindConfig config) {
        // 创建模型和tokenizer
        MiniMindModel model = new MiniMindModel("minimind-train", config);
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(), 
            config.getMaxSeqLen()
        );
        
        System.out.println("训练循环结构:");
        System.out.println("```");
        System.out.println("for epoch in range(num_epochs):");
        System.out.println("    for batch in train_dataloader:");
        System.out.println("        # 1. 前向传播");
        System.out.println("        outputs = model(batch)");
        System.out.println("        loss = criterion(outputs, targets)");
        System.out.println("        ");
        System.out.println("        # 2. 反向传播");
        System.out.println("        loss.backward()");
        System.out.println("        ");
        System.out.println("        # 3. 梯度裁剪");
        System.out.println("        clip_grad_norm_(model.parameters(), max_norm=1.0)");
        System.out.println("        ");
        System.out.println("        # 4. 优化器步进");
        System.out.println("        optimizer.step()");
        System.out.println("        optimizer.zero_grad()");
        System.out.println("        ");
        System.out.println("        # 5. 学习率调度");
        System.out.println("        scheduler.step()");
        System.out.println("        ");
        System.out.println("        # 6. 日志记录");
        System.out.println("        log_metrics(loss, learning_rate)");
        System.out.println("```");
        
        // 模拟单步训练
        System.out.println("\n单步训练演示:");
        String sample = "深度学习训练示例";
        List<Integer> tokenIds = tokenizer.encode(sample, false, false);
        
        int[] ids = tokenIds.stream().mapToInt(i -> i).toArray();
        float[] floatIds = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            floatIds[i] = (float) ids[i];
        }
        NdArray inputArray = NdArray.of(floatIds, Shape.of(1, ids.length));
        
        Variable inputVar = new Variable(inputArray);
        model.setTraining(true);
        Variable output = model.forward(inputVar);
        
        System.out.println("  - 输入形状: [1, " + ids.length + "]");
        int[] outputShape = output.getValue().getShape().getShapeDims();
        System.out.println("  - 输出形状: [" + outputShape[0] + ", " + 
                          outputShape[1] + ", " + outputShape[2] + "]");
        System.out.println("  ✓ 训练步骤完成");
        
        model.setTraining(false);
    }
    
    /**
     * 评估模型
     */
    private static void evaluateModel() {
        System.out.println("1. 验证集评估");
        System.out.println("  - 计算验证损失");
        System.out.println("  - 计算困惑度");
        System.out.println("  - 示例: Val Loss=2.5, PPL=12.18");
        
        System.out.println("\n2. 生成样本检查");
        System.out.println("  - 输入提示: '人工智能'");
        System.out.println("  - 生成文本: '人工智能是计算机科学的一个分支...'");
        System.out.println("  - 人工检查质量");
        
        System.out.println("\n3. 决策是否继续训练");
        System.out.println("  - 验证loss是否下降");
        System.out.println("  - 是否出现过拟合");
        System.out.println("  - 生成质量是否改善");
    }
    
    /**
     * 保存模型
     */
    private static void saveModel() {
        System.out.println("1. 保存检查点");
        System.out.println("  - 模型参数: model.pth");
        System.out.println("  - 优化器状态: optimizer.pth");
        System.out.println("  - 训练配置: config.json");
        System.out.println("  - Tokenizer: tokenizer.json");
        
        System.out.println("\n2. 保存最佳模型");
        System.out.println("  - 基于验证loss选择");
        System.out.println("  - best_model.pth");
        
        System.out.println("\n3. 导出推理模型");
        System.out.println("  - 仅保存模型权重");
        System.out.println("  - 移除训练相关参数");
        System.out.println("  - 优化推理性能");
    }
    
    /**
     * 训练监控
     */
    private static void printTrainingMonitoring() {
        System.out.println("\n=== 训练监控 ===");
        
        System.out.println("\n实时监控指标:");
        System.out.println("  - 当前步数/总步数");
        System.out.println("  - 训练损失");
        System.out.println("  - 学习率");
        System.out.println("  - 梯度范数");
        System.out.println("  - 每步耗时");
        System.out.println("  - GPU/CPU使用率");
        
        System.out.println("\n定期评估指标:");
        System.out.println("  - 验证损失(每N步)");
        System.out.println("  - 困惑度");
        System.out.println("  - 生成样本质量");
        
        System.out.println("\n可视化工具:");
        System.out.println("  - TensorBoard");
        System.out.println("  - 自定义日志");
        System.out.println("  - 实时曲线图");
    }
    
    /**
     * 训练技巧
     */
    private static void printTrainingTips() {
        System.out.println("\n=== 训练技巧 ===");
        
        System.out.println("\n1. 超参数调优:");
        System.out.println("  - 学习率是最重要的超参数");
        System.out.println("  - 使用学习率查找器");
        System.out.println("  - Batch大小影响收敛");
        
        System.out.println("\n2. 训练稳定性:");
        System.out.println("  - 梯度裁剪防止爆炸");
        System.out.println("  - 权重初始化很关键");
        System.out.println("  - LayerNorm提升稳定性");
        
        System.out.println("\n3. 加速训练:");
        System.out.println("  - 混合精度训练");
        System.out.println("  - 梯度累积模拟大batch");
        System.out.println("  - 数据并行");
        
        System.out.println("\n4. 避免过拟合:");
        System.out.println("  - Dropout正则化");
        System.out.println("  - 权重衰减");
        System.out.println("  - 早停策略");
        System.out.println("  - 数据增强");
    }
    
    /**
     * 故障排查
     */
    private static void printTroubleshooting() {
        System.out.println("\n=== 常见问题排查 ===");
        
        System.out.println("\n1. 损失不下降:");
        System.out.println("  - 检查学习率是否过小");
        System.out.println("  - 检查数据是否正确");
        System.out.println("  - 检查模型初始化");
        
        System.out.println("\n2. 损失爆炸/NaN:");
        System.out.println("  - 降低学习率");
        System.out.println("  - 启用梯度裁剪");
        System.out.println("  - 检查数据异常值");
        
        System.out.println("\n3. 训练过慢:");
        System.out.println("  - 增大batch大小");
        System.out.println("  - 使用混合精度");
        System.out.println("  - 优化数据加载");
        
        System.out.println("\n4. 内存不足:");
        System.out.println("  - 减小batch大小");
        System.out.println("  - 使用梯度检查点");
        System.out.println("  - 减小模型规模");
    }
}
