package io.leavesfly.tinyai.deepseek.v3.training;

import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Block;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Config;
import io.leavesfly.tinyai.deepseek.v3.DeepSeekV3Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-V3预训练器
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练,
 * 特别优化MoE负载均衡和任务感知能力
 * 
 * 关键特性：
 * 1. MoE负载均衡损失 - 确保专家均匀使用
 * 2. 任务感知训练 - 提升任务路由准确性
 * 3. Warmup + Cosine衰减学习率
 * 4. 梯度裁剪防止爆炸
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Pretrain {
    
    private final DeepSeekV3Model model;
    private final DeepSeekV3Config config;
    private final DeepSeekV3Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    // 训练超参数
    private int maxEpochs;
    private float initialLearningRate;
    private float minLearningRate;
    private int warmupSteps;
    private float maxGradNorm;
    private float moeLoadBalanceWeight;  // MoE负载均衡权重(V3特有)
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private float currentLearningRate;
    private List<Float> lossHistory;
    private List<Float> moeLossHistory;      // MoE损失历史(V3特有)
    private List<Float> confidenceHistory;
    
    /**
     * 构造函数
     */
    public DeepSeekV3Pretrain(DeepSeekV3Model model, DeepSeekV3Dataset dataset) {
        this.model = model;
        this.config = model.getConfig();
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认超参数
        this.maxEpochs = 10;
        this.initialLearningRate = 2.5e-4f;
        this.minLearningRate = 1e-5f;
        this.warmupSteps = 2000;
        this.maxGradNorm = 1.0f;
        this.moeLoadBalanceWeight = (float) config.getLoadBalanceLossWeight();
        this.logInterval = 100;
        this.saveInterval = 5000;
        this.checkpointDir = "./checkpoints/deepseek_v3_pretrain";
        
        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
        this.moeLossHistory = new ArrayList<>();
        this.confidenceHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DeepSeekV3Pretrain configure(int maxEpochs, float learningRate,
                                         int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        return this;
    }
    
    /**
     * 配置MoE参数
     */
    public DeepSeekV3Pretrain configureMoE(float moeLoadBalanceWeight) {
        this.moeLoadBalanceWeight = moeLoadBalanceWeight;
        return this;
    }
    
    /**
     * 设置检查点配置
     */
    public DeepSeekV3Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(80));
        System.out.println("DeepSeek-V3 预训练（含MoE负载均衡）");
        System.out.println("=".repeat(80));
        System.out.println("模型参数:");
        System.out.println("  - 嵌入维度: " + config.getNEmbd());
        System.out.println("  - Transformer层数: " + config.getNLayer());
        System.out.println("  - 注意力头数: " + config.getNHead());
        System.out.println("  - 专家数量: " + config.getNumExperts());
        System.out.println("  - Top-K选择: " + config.getTopK());
        System.out.println("  - 总参数量: " + formatParamCount(config.estimateParameterCount()));
        System.out.println("  - 激活参数: " + formatParamCount(config.estimateActiveParameterCount()) + 
                          " (" + String.format("%.1f%%", config.getActivationRatio()) + ")");
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("  - MoE负载均衡权重: " + moeLoadBalanceWeight);
        System.out.println("=".repeat(80));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        System.out.println("\n训练完成!");
        System.out.println("最终语言模型损失: " + getAverage(lossHistory, 100));
        System.out.println("最终MoE负载损失: " + getAverage(moeLossHistory, 100));
        System.out.println("平均推理置信度: " + getAverage(confidenceHistory, 100));
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochLoss = 0.0;
        double epochMoeLoss = 0.0;
        double epochConfidence = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNext()) {
            DeepSeekV3Dataset.Batch batch = dataset.nextBatch();
            
            // 训练一步
            StepResult stepResult = trainStep(batch);
            
            epochLoss += stepResult.languageModelLoss;
            epochMoeLoss += stepResult.moeLoss;
            epochConfidence += stepResult.confidence;
            batchCount++;
            globalStep++;
            
            // 记录
            lossHistory.add(stepResult.languageModelLoss);
            moeLossHistory.add(stepResult.moeLoss);
            confidenceHistory.add(stepResult.confidence);
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverage(lossHistory, logInterval);
                float avgMoeLoss = getAverage(moeLossHistory, logInterval);
                float avgConf = getAverage(confidenceHistory, logInterval);
                System.out.printf("Epoch %d/%d | Step %d | LM Loss: %.4f | MoE Loss: %.6f | " +
                                 "Confidence: %.4f | LR: %.6f%n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss, avgMoeLoss, 
                    avgConf, currentLearningRate);
            }
            
            // 保存检查点
            if (globalStep % saveInterval == 0) {
                saveCheckpoint("step_" + globalStep);
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochMoeLoss = batchCount > 0 ? epochMoeLoss / batchCount : 0.0;
        double avgEpochConf = batchCount > 0 ? epochConfidence / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均LM损失: %.4f | 平均MoE损失: %.6f | " +
                         "平均置信度: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, avgEpochMoeLoss, avgEpochConf, 
            epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练单步（含MoE负载均衡）
     */
    private StepResult trainStep(DeepSeekV3Dataset.Batch batch) {
        // 更新学习率
        updateLearningRate();
        
        // 准备输入
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播(带详细信息,包含MoE损失)
        DeepSeekV3Block.DetailedForwardResult result = 
            model.predictWithDetails(inputVar, batch.getMajorityTaskType());
        Variable logits = result.logits;
        
        // 计算语言模型损失
        // SoftmaxCE只接受2维输入，需要将3D张量reshape为2D
        int batchSize = inputIds.getShape().getDimension(0);
        int seqLen = inputIds.getShape().getDimension(1);
        int vocabSize = model.getConfig().getVocabSize();
        
        // logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        
        // targets: [batch_size, seq_len] -> [batch_size * seq_len, 1]
        Variable targetVar = new Variable(targetIds);
        Variable targets2D = targetVar.reshape(Shape.of(batchSize * seqLen, 1));
        
        Variable lmLoss = lossFunction.loss(targets2D, logits2D);
        
        float lmLossValue = lmLoss.getValue().getNumber().floatValue();
        float moeLossValue = (float) result.avgMoELoss;
        float confidence = (float) result.reasoningResult.confidence;
        
        // 总损失 = 语言模型损失 + MoE负载均衡损失
        Variable totalLoss = lmLoss;
        if (moeLoadBalanceWeight > 0) {
            // 创建标量MoE损失，使用一维数组
            float[] moeLossData = new float[]{moeLossValue * moeLoadBalanceWeight};
            Variable moeLossVar = new Variable(NdArray.of(moeLossData));
            totalLoss = totalLoss.add(moeLossVar);
        }
        
        // 清空梯度
        model.clearGrads();
        
        // 反向传播
        totalLoss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 参数更新
        optimizer.update();
        
        // 断开计算图
        totalLoss.unChainBackward();
        
        return new StepResult(lmLossValue, moeLossValue, confidence);
    }
    
    /**
     * 更新学习率(warmup + cosine衰减)
     */
    private void updateLearningRate() {
        if (globalStep < warmupSteps) {
            // 线性warmup
            currentLearningRate = initialLearningRate * ((float) globalStep / warmupSteps);
        } else {
            // 余弦退火
            int totalSteps = maxEpochs * dataset.getBatchCount();
            int decaySteps = totalSteps - warmupSteps;
            int currentDecayStep = globalStep - warmupSteps;
            
            double cosineDecay = 0.5 * (1 + Math.cos(Math.PI * currentDecayStep / decaySteps));
            float decayedLR = (initialLearningRate - minLearningRate) * (float) cosineDecay + minLearningRate;
            currentLearningRate = Math.max(decayedLR, minLearningRate);
        }
        
        optimizer.setLearningRate(currentLearningRate);
    }
    
    /**
     * 梯度裁剪（使用V2 Parameter）
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        // 计算梯度范数
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        for (Parameter param : params.values()) {
            if (param.requiresGrad() && param.grad() != null) {
                NdArray grad = param.grad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 裁剪梯度
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.requiresGrad() && param.grad() != null) {
                    NdArray clippedGrad = param.grad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint(String name) {
        String path = checkpointDir + "/" + name + ".ckpt";
        System.out.println("保存检查点: " + path);
        // 实际保存逻辑（这里简化）
    }
    
    /**
     * 计算平均值
     */
    private float getAverage(List<Float> values, int last) {
        if (values.isEmpty()) return 0.0f;
        
        int start = Math.max(0, values.size() - last);
        float sum = 0.0f;
        for (int i = start; i < values.size(); i++) {
            sum += values.get(i);
        }
        return sum / (values.size() - start);
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    /**
     * 训练步骤结果类
     */
    private static class StepResult {
        final float languageModelLoss;  // 语言模型损失
        final float moeLoss;            // MoE负载均衡损失
        final float confidence;         // 推理置信度
        
        StepResult(float languageModelLoss, float moeLoss, float confidence) {
            this.languageModelLoss = languageModelLoss;
            this.moeLoss = moeLoss;
            this.confidence = confidence;
        }
    }
    
    /**
     * 训练统计信息
     */
    public static class TrainingStats {
        public final int totalSteps;
        public final double avgLoss;
        public final double avgMoeLoss;
        public final double avgConfidence;
        
        public TrainingStats(int totalSteps, double avgLoss, 
                           double avgMoeLoss, double avgConfidence) {
            this.totalSteps = totalSteps;
            this.avgLoss = avgLoss;
            this.avgMoeLoss = avgMoeLoss;
            this.avgConfidence = avgConfidence;
        }
        
        @Override
        public String toString() {
            return String.format("TrainingStats[steps=%d, loss=%.4f, moeLoss=%.6f, conf=%.4f]",
                totalSteps, avgLoss, avgMoeLoss, avgConfidence);
        }
    }
    
    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        double avgLoss = lossHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        double avgMoeLoss = moeLossHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        double avgConf = confidenceHistory.stream().mapToDouble(f -> f).average().orElse(0.0);
        return new TrainingStats(globalStep, avgLoss, avgMoeLoss, avgConf);
    }
}
