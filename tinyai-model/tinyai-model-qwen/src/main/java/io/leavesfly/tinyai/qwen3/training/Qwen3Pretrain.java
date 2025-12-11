package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.qwen3.Qwen3Config;
import io.leavesfly.tinyai.qwen3.Qwen3Model;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Qwen3预训练器
 * 
 * 实现因果语言建模(Causal Language Modeling)预训练
 * 支持学习率调度、梯度裁剪、检查点保存等功能
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Pretrain {
    
    private final Qwen3Model model;
    private final Qwen3Config config;
    private final Qwen3Dataset dataset;
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
    
    /**
     * 构造函数
     */
    public Qwen3Pretrain(Qwen3Model model, Qwen3Dataset dataset) {
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
        this.checkpointDir = "./checkpoints/qwen3_pretrain";
        
        // 创建优化器
        this.optimizer = new Adam(model, initialLearningRate, 0.9f, 0.999f, 1e-8f);
        
        // 初始化状态
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.currentLearningRate = 0.0f;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public Qwen3Pretrain configure(int maxEpochs, float learningRate,
                                    int warmupSteps, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.initialLearningRate = learningRate;
        this.warmupSteps = warmupSteps;
        this.maxGradNorm = maxGradNorm;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 设置检查点配置
     */
    public Qwen3Pretrain setCheckpoint(String checkpointDir, int saveInterval) {
        this.checkpointDir = checkpointDir;
        this.saveInterval = saveInterval;
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("Qwen3 预训练");
        System.out.println("=".repeat(70));
        System.out.println("模型参数:");
        System.out.println("  - 隐藏维度: " + config.getHiddenSize());
        System.out.println("  - Transformer层数: " + config.getNumHiddenLayers());
        System.out.println("  - 注意力头数: " + config.getNumAttentionHeads());
        System.out.println("  - 词表大小: " + config.getVocabSize());
        System.out.println("  - 估算参数量: " + formatParamCount(config.estimateParameterCount()));
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + dataset.getSampleCount());
        System.out.println("  - 批次数量: " + dataset.getBatchCount());
        System.out.println("  - 最大轮次: " + maxEpochs);
        System.out.println("  - 初始学习率: " + initialLearningRate);
        System.out.println("  - Warmup步数: " + warmupSteps);
        System.out.println("  - 梯度裁剪: " + maxGradNorm);
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
        System.out.println("最终损失: " + getAverage(lossHistory, 100));
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNext()) {
            Qwen3Dataset.Batch batch = dataset.nextBatch();
            
            // 训练一步
            float loss = trainStep(batch);
            
            epochLoss += loss;
            batchCount++;
            globalStep++;
            
            // 记录
            lossHistory.add(loss);
            
            // 打印日志
            if (globalStep % logInterval == 0) {
                float avgLoss = getAverage(lossHistory, logInterval);
                System.out.printf("[Epoch %d/%d] [Step %d] Loss: %.4f | LR: %.6f\n",
                    currentEpoch + 1, maxEpochs, globalStep, avgLoss, currentLearningRate);
            }
            
            // 保存检查点
            if (globalStep % saveInterval == 0) {
                saveCheckpoint("step_" + globalStep);
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        
        System.out.printf("\nEpoch %d 完成 | 平均损失: %.4f | 耗时: %d ms\n",
            currentEpoch + 1, epochLoss / batchCount, epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练单步
     */
    private float trainStep(Qwen3Dataset.Batch batch) {
        // 更新学习率
        updateLearningRate();
        
        // 准备输入
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播
        Variable logits = model.forward(inputVar);
        
        // 计算损失
        Variable targetVar = new Variable(targetIds);
        Variable loss = lossFunction.loss(targetVar, logits);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        // 清空梯度
        model.clearGrads();
        
        // 反向传播
        loss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 参数更新
        optimizer.update();
        
        // 断开计算图
        loss.unChainBackward();
        
        return lossValue;
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
            
            float cosineDecay = (float) (0.5 * (1.0 + Math.cos(Math.PI * currentDecayStep / decaySteps)));
            currentLearningRate = minLearningRate + 
                (initialLearningRate - minLearningRate) * cosineDecay;
        }
        
        optimizer.setLearningRate(currentLearningRate);
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
        String filepath = checkpointDir + "/qwen3_" + name + ".ckpt";
        try {
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 计算平均值
     */
    private float getAverage(List<Float> values, int window) {
        if (values.isEmpty()) return 0.0f;
        
        int start = Math.max(0, values.size() - window);
        int end = values.size();
        
        float sum = 0.0f;
        for (int i = start; i < end; i++) {
            sum += values.get(i);
        }
        
        return sum / (end - start);
    }
    
    /**
     * 格式化参数数量
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
    
    /**
     * 获取训练统计信息
     */
    public TrainingStats getStats() {
        return new TrainingStats(globalStep, getAverage(lossHistory, lossHistory.size()));
    }
    
    /**
     * 训练统计类
     */
    public static class TrainingStats {
        public final int totalSteps;
        public final float avgLoss;
        
        public TrainingStats(int totalSteps, float avgLoss) {
            this.totalSteps = totalSteps;
            this.avgLoss = avgLoss;
        }
    }
}
