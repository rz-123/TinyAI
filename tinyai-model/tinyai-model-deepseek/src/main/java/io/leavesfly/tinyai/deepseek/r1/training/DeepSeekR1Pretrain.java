package io.leavesfly.tinyai.deepseek.r1.training;

import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Config;
import io.leavesfly.tinyai.deepseek.r1.DeepSeekR1Model;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * DeepSeek-R1预训练器
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练,
 * 同时训练推理和反思能力
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekR1Pretrain {
    
    private final DeepSeekR1Model model;
    private final DeepSeekR1Config config;
    private final DeepSeekR1Dataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    // 训练超参数
    private int maxEpochs;
    private float initialLearningRate;
    private float minLearningRate;
    private int warmupSteps;
    private float maxGradNorm;
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private float currentLearningRate;
    private List<Float> lossHistory;
    private List<Float> reasoningConfidenceHistory;
    
    /**
     * 构造函数
     */
    public DeepSeekR1Pretrain(DeepSeekR1Model model, DeepSeekR1Dataset dataset) {
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
        this.logInterval = 100;
        this.saveInterval = 5000;
        this.checkpointDir = "./checkpoints/deepseek_r1_pretrain";
        
        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
        this.reasoningConfidenceHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public DeepSeekR1Pretrain configure(int maxEpochs, float learningRate,
                                         int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        return this;
    }
    
    /**
     * 设置检查点配置
     */
    public DeepSeekR1Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 设置日志输出间隔
     */
    public DeepSeekR1Pretrain setLogInterval(int logInterval) {
        this.logInterval = logInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("DeepSeek-R1 预训练");
        System.out.println("=".repeat(70));
        System.out.println("模型参数:");
        System.out.println("  - 嵌入维度: " + config.getNEmbd());
        System.out.println("  - Transformer层数: " + config.getNLayer());
        System.out.println("  - 注意力头数: " + config.getNHead());
        System.out.println("  - 最大推理步骤: " + config.getMaxReasoningSteps());
        System.out.println("  - 质量评分维度: " + config.getQualityScoreDim());
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("=".repeat(70));
        
        // 创建检查点目录
        createCheckpointDir();
        
        // 训练循环
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        // 保存最终模型
        saveCheckpoint("final");
        
        System.out.println("\n训练完成!");
        System.out.println("最终损失: " + getAverageLoss(lossHistory, 100));
        System.out.println("平均推理置信度: " + getAverageLoss(reasoningConfidenceHistory, 100));
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochLoss = 0.0;
        double epochConfidence = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNext()) {
            DeepSeekR1Dataset.Batch batch = dataset.nextBatch();
            
            // 训练一步
            StepResult stepResult = trainStep(batch);
            
            epochLoss += stepResult.loss;
            epochConfidence += stepResult.confidence;
            batchCount++;
            globalStep++;
            
            // 记录
            lossHistory.add(stepResult.loss);
            reasoningConfidenceHistory.add(stepResult.confidence);
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverageLoss(lossHistory, logInterval);
                float avgConf = getAverageLoss(reasoningConfidenceHistory, logInterval);
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f | Confidence: %.4f | LR: %.6f%n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss, avgConf, currentLearningRate);
            }
            
            // 保存检查点
            if (globalStep % saveInterval == 0) {
                saveCheckpoint("step_" + globalStep);
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        double avgEpochConf = batchCount > 0 ? epochConfidence / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 平均置信度: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, avgEpochConf, epochEndTime - epochStartTime);
        
        dataset.reset();
        
        // Epoch结束后主动触发GC，帮助释放内存
        System.gc();
    }
    
    /**
     * 训练单步
     */
    private StepResult trainStep(DeepSeekR1Dataset.Batch batch) {
        // 更新学习率
        updateLearningRate();
        
        // 准备输入
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播(带推理和反思)
        DeepSeekR1Model.ReasoningOutput result = model.performReasoning(inputVar);
        Variable logits = result.logits;
        
        // 计算损失 - SoftmaxCE只支持2D输入，需要reshape
        // logits: [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
        // targets: [batch_size, seq_len] -> [batch_size * seq_len, 1]
        int[] logitsShape = logits.getValue().getShape().getShapeDims();
        int batchSize = logitsShape[0];
        int seqLen = logitsShape[1];
        int vocabSize = logitsShape[2];
        
        Variable logits2D = logits.reshape(Shape.of(batchSize * seqLen, vocabSize));
        Variable targetVar = new Variable(targetIds.reshape(Shape.of(batchSize * seqLen, 1)));
        Variable loss = lossFunction.loss(targetVar, logits2D);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        float confidence = (float) result.averageConfidence;
        
        // 清空梯度
        model.clearGrads();
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 参数更新
        optimizer.update();
        
        // 彻底断开计算图，释放内存
        loss.unChainBackward();
        logits.unChainBackward();
        inputVar.unChainBackward();
        
        return new StepResult(lossValue, confidence);
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
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        // 计算梯度范数(使用V2 Parameter)
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        for (Parameter param : params.values()) {
            if (param.grad() != null) {
                NdArray grad = param.grad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        // 裁剪
        if (totalNorm > maxGradNorm) {
            float scale = (float) (maxGradNorm / totalNorm);
            for (Parameter param : params.values()) {
                if (param.grad() != null) {
                    NdArray clippedGrad = param.grad().mulNum(scale);
                    param.setGrad(clippedGrad);
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint(String suffix) {
        try {
            String filename = String.format("deepseek_r1_pretrain_%s.model", suffix);
            String filepath = checkpointDir + File.separator + filename;
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Files.createDirectories(Paths.get(checkpointDir));
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    /**
     * 获取平均损失
     */
    private float getAverageLoss(List<Float> history, int n) {
        if (history.isEmpty()) {
            return 0.0f;
        }
        
        int start = Math.max(0, history.size() - n);
        float sum = 0.0f;
        for (int i = start; i < history.size(); i++) {
            sum += history.get(i);
        }
        return sum / (history.size() - start);
    }
    
    /**
     * 单步训练结果
     */
    private static class StepResult {
        final float loss;
        final float confidence;
        
        StepResult(float loss, float confidence) {
            this.loss = loss;
            this.confidence = confidence;
        }
    }
    
    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        return new TrainingStats(
            currentEpoch,
            globalStep,
            currentLearningRate,
            lossHistory.isEmpty() ? 0.0f : lossHistory.get(lossHistory.size() - 1),
            getAverageLoss(lossHistory, 100),
            getAverageLoss(reasoningConfidenceHistory, 100)
        );
    }
    
    /**
     * 训练统计信息
     */
    public static class TrainingStats {
        public final int epoch;
        public final int step;
        public final float learningRate;
        public final float currentLoss;
        public final float avgLoss;
        public final float avgConfidence;
        
        public TrainingStats(int epoch, int step, float learningRate,
                           float currentLoss, float avgLoss, float avgConfidence) {
            this.epoch = epoch;
            this.step = step;
            this.learningRate = learningRate;
            this.currentLoss = currentLoss;
            this.avgLoss = avgLoss;
            this.avgConfidence = avgConfidence;
        }
        
        @Override
        public String toString() {
            return String.format(
                "TrainingStats{epoch=%d, step=%d, lr=%.6f, loss=%.4f, avgLoss=%.4f, avgConf=%.4f}",
                epoch, step, learningRate, currentLoss, avgLoss, avgConfidence
            );
        }
    }
}
