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
 * Qwen3后训练/微调器
 * 
 * 实现指令微调(Instruction Tuning)和任务适配
 * 学习率比预训练小,更保守的训练策略
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Posttrain {
    
    private final Qwen3Model model;
    private final Qwen3Config config;
    private final Qwen3Dataset trainDataset;
    private final Qwen3Dataset valDataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    // 训练超参数
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int evalInterval;
    private int patience;
    private String checkpointDir;
    
    // 训练状态
    private int currentEpoch;
    private int globalStep;
    private float bestValLoss;
    private int stepsWithoutImprovement;
    private List<Float> trainLossHistory;
    private List<Float> valLossHistory;
    
    /**
     * 构造函数
     */
    public Qwen3Posttrain(Qwen3Model model, Qwen3Dataset trainDataset, Qwen3Dataset valDataset) {
        this.model = model;
        this.config = model.getConfig();
        this.trainDataset = trainDataset;
        this.valDataset = valDataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 后训练学习率比预训练小10倍
        this.maxEpochs = 5;
        this.learningRate = 2.5e-5f;
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.evalInterval = 100;
        this.patience = 3;
        this.checkpointDir = "./checkpoints/qwen3_posttrain";
        
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.currentEpoch = 0;
        this.globalStep = 0;
        this.trainLossHistory = new ArrayList<>();
        this.valLossHistory = new ArrayList<>();
        this.bestValLoss = Float.MAX_VALUE;
        this.stepsWithoutImprovement = 0;
    }
    
    /**
     * 配置训练参数
     */
    public Qwen3Posttrain configure(int maxEpochs, float learningRate, int patience) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.patience = patience;
        this.optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(70));
        System.out.println("Qwen3 后训练/微调 (Posttrain)");
        System.out.println("=".repeat(70));
        System.out.println("模型配置: " + model.getName());
        System.out.println("  - 隐藏维度: " + config.getHiddenSize());
        System.out.println("  - 层数: " + config.getNumHiddenLayers());
        System.out.println("训练配置:");
        System.out.println("  - 训练样本: " + trainDataset.getSampleCount());
        System.out.println("  - 验证样本: " + valDataset.getSampleCount());
        System.out.println("  - 学习率: " + learningRate + " (预训练的1/10)");
        System.out.println("  - 早停耐心: " + patience + " epochs");
        System.out.println("=".repeat(70));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
            
            // 验证
            float valLoss = validate();
            valLossHistory.add(valLoss);
            
            // 早停检查
            if (valLoss < bestValLoss) {
                bestValLoss = valLoss;
                stepsWithoutImprovement = 0;
                saveCheckpoint("best");
            } else {
                stepsWithoutImprovement++;
                if (stepsWithoutImprovement >= patience) {
                    System.out.println("\n早停触发！验证损失未改善 " + patience + " 轮");
                    break;
                }
            }
        }
        
        System.out.println("\n微调完成!");
        System.out.println("最佳验证损失: " + bestValLoss);
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        trainDataset.prepare(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        while (trainDataset.hasNext()) {
            Qwen3Dataset.Batch batch = trainDataset.nextBatch();
            
            float loss = trainStep(batch);
            
            epochLoss += loss;
            batchCount++;
            globalStep++;
            trainLossHistory.add(loss);
            
            if (globalStep % logInterval == 0) {
                System.out.printf("[Epoch %d/%d] [Step %d] Loss: %.4f\n",
                    currentEpoch + 1, maxEpochs, globalStep, loss);
            }
        }
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f\n",
            currentEpoch + 1, epochLoss / batchCount);
        
        trainDataset.reset();
    }
    
    /**
     * 训练单步
     */
    private float trainStep(Qwen3Dataset.Batch batch) {
        NdArray inputIds = batch.getInputIds();
        NdArray targetIds = batch.getTargetIds();
        
        Variable inputVar = new Variable(inputIds);
        Variable logits = model.forward(inputVar);
        
        Variable targetVar = new Variable(targetIds);
        Variable loss = lossFunction.loss(targetVar, logits);
        
        float lossValue = loss.getValue().getNumber().floatValue();
        
        model.clearGrads();
        loss.backward();
        clipGradients();
        optimizer.update();
        loss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 验证
     */
    private float validate() {
        valDataset.prepare(false);
        
        double totalLoss = 0.0;
        int batchCount = 0;
        
        while (valDataset.hasNext()) {
            Qwen3Dataset.Batch batch = valDataset.nextBatch();
            
            NdArray inputIds = batch.getInputIds();
            NdArray targetIds = batch.getTargetIds();
            
            Variable inputVar = new Variable(inputIds);
            Variable logits = model.forward(inputVar);
            
            Variable targetVar = new Variable(targetIds);
            Variable loss = lossFunction.loss(targetVar, logits);
            
            totalLoss += loss.getValue().getNumber().floatValue();
            batchCount++;
        }
        
        valDataset.reset();
        
        float avgLoss = (float) (totalLoss / batchCount);
        System.out.printf("验证 | Epoch %d | 验证损失: %.4f\n", currentEpoch + 1, avgLoss);
        
        return avgLoss;
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        Map<String, Parameter> params = model.getModule().namedParameters("", true);
        for (Parameter param : params.values()) {
            if (param.grad() != null) {
                NdArray grad = param.grad();
                double norm = grad.mul(grad).sum().getNumber().doubleValue();
                totalNorm += norm;
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
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
    
    private void createCheckpointDir() {
        File dir = new File(checkpointDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }
    }
    
    private void saveCheckpoint(String name) {
        String filepath = checkpointDir + "/qwen3_posttrain_" + name + ".ckpt";
        try {
            model.saveModel(filepath);
            System.out.println("检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
}
