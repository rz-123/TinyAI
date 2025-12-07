package io.leavesfly.tinyai.minimind.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.training.dataset.SFTDataset;
import io.leavesfly.tinyai.ml.loss.SoftmaxCrossEntropy;
import io.leavesfly.tinyai.ml.optimize.Adam;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.io.File;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * SFT(Supervised Fine-Tuning)训练器
 * 
 * 实现指令微调,仅计算模型输出部分的损失
 * 
 * @author leavesfly
 * @since 2024
 */
public class SFTTrainer {
    
    private final MiniMindModel model;
    private final SFTDataset dataset;
    private final SoftmaxCrossEntropy lossFunction;
    private final Adam optimizer;
    
    private int maxEpochs;
    private float learningRate;
    private float maxGradNorm;
    private int logInterval;
    private int saveInterval;
    private String checkpointDir;
    
    private int currentEpoch;
    private int currentStep;
    private List<Float> lossHistory;
    
    /**
     * 构造函数
     */
    public SFTTrainer(MiniMindModel model, SFTDataset dataset) {
        this.model = model;
        this.dataset = dataset;
        this.lossFunction = new SoftmaxCrossEntropy();
        
        // 默认配置(较小学习率,避免灾难性遗忘)
        this.maxEpochs = 3;
        this.learningRate = 5e-5f;
        this.maxGradNorm = 1.0f;
        this.logInterval = 50;
        this.saveInterval = 500;
        this.checkpointDir = "./sft_checkpoints";
        
        // 创建优化器
        this.optimizer = new Adam(model, learningRate, 0.9f, 0.999f, 1e-8f);
        
        this.currentEpoch = 0;
        this.currentStep = 0;
        this.lossHistory = new ArrayList<>();
    }
    
    /**
     * 配置训练参数
     */
    public SFTTrainer configure(int maxEpochs, float learningRate, float maxGradNorm) {
        this.maxEpochs = maxEpochs;
        this.learningRate = learningRate;
        this.maxGradNorm = maxGradNorm;
        
        optimizer.setLearningRate(learningRate);
        return this;
    }
    
    /**
     * 开始训练
     */
    public void train() {
        System.out.println("=".repeat(60));
        System.out.println("开始SFT微调");
        System.out.println("=".repeat(60));
        System.out.println("训练样本数: " + dataset.getSampleCount());
        System.out.println("批次数量: " + dataset.getBatchCount());
        System.out.println("最大轮次: " + maxEpochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("=".repeat(60));
        
        createCheckpointDir();
        
        for (currentEpoch = 0; currentEpoch < maxEpochs; currentEpoch++) {
            trainOneEpoch();
        }
        
        System.out.println("SFT微调完成!");
    }
    
    /**
     * 训练一个epoch
     */
    private void trainOneEpoch() {
        dataset.prepare(true);
        model.setTraining(true);
        
        double epochLoss = 0.0;
        int batchCount = 0;
        
        long epochStartTime = System.currentTimeMillis();
        
        while (dataset.hasNextBatch()) {
            SFTDataset.Batch batch = dataset.getNextBatch();
            float stepLoss = trainStep(batch);
            
            epochLoss += stepLoss;
            batchCount++;
            currentStep++;
            
            lossHistory.add(stepLoss);
            
            if (currentStep % logInterval == 0) {
                double avgLoss = lossHistory.stream()
                    .skip(Math.max(0, lossHistory.size() - logInterval))
                    .mapToDouble(Float::doubleValue)
                    .average()
                    .orElse(0.0);
                
                System.out.printf("Epoch %d/%d | Step %d | Loss: %.4f%n",
                    currentEpoch + 1, maxEpochs, currentStep, avgLoss);
            }
            
            if (currentStep % saveInterval == 0) {
                saveCheckpoint();
            }
        }
        
        long epochEndTime = System.currentTimeMillis();
        double avgEpochLoss = batchCount > 0 ? epochLoss / batchCount : 0.0;
        
        System.out.printf("Epoch %d 完成 | 平均损失: %.4f | 耗时: %d ms%n",
            currentEpoch + 1, avgEpochLoss, epochEndTime - epochStartTime);
        
        dataset.reset();
    }
    
    /**
     * 训练一步
     */
    private float trainStep(SFTDataset.Batch batch) {
        NdArray inputArray = batch.getInput();
        NdArray labelArray = batch.getLabels();
        NdArray maskArray = batch.getLossMask();
        
        Variable input = new Variable(inputArray);
        Variable labels = new Variable(labelArray);
        
        // 前向传播
        Variable logits = model.predict(input);
        
        // 计算损失(应用掩码)
        Variable loss = lossFunction.loss(labels, logits);
        
        // 应用损失掩码
        Variable maskedLoss = applyLossMask(loss, maskArray);
        
        float lossValue = maskedLoss.getValue().getNumber().floatValue();
        
        // 清除梯度
        model.clearGrads();
        
        // 反向传播
        maskedLoss.backward();
        
        // 梯度裁剪
        clipGradients();
        
        // 更新参数
        optimizer.update();
        
        // 断开计算图
        maskedLoss.unChainBackward();
        
        return lossValue;
    }
    
    /**
     * 应用损失掩码
     */
    private Variable applyLossMask(Variable loss, NdArray mask) {
        // 将损失与掩码相乘,然后取平均
        Variable maskVar = new Variable(mask);
        Variable maskedLoss = loss.mul(maskVar);
        
        // 计算掩码中1的数量,用于归一化
        float[] maskData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) mask).buffer;
        float maskSum = 0;
        for (float m : maskData) {
            maskSum += m;
        }
        
        if (maskSum > 0) {
            return maskedLoss.div(new Variable(NdArray.of(maskSum)));
        }
        
        return maskedLoss;
    }
    
    /**
     * 梯度裁剪
     */
    private void clipGradients() {
        double totalNorm = 0.0;
        
        for (var param : model.getAllParams().values()) {
            if (param.getGrad() != null) {
                NdArray grad = param.getGrad();
                float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                
                for (float g : gradData) {
                    totalNorm += g * g;
                }
            }
        }
        
        totalNorm = Math.sqrt(totalNorm);
        
        if (totalNorm > maxGradNorm) {
            float clipCoef = maxGradNorm / (float) totalNorm;
            
            for (var param : model.getAllParams().values()) {
                if (param.getGrad() != null) {
                    NdArray grad = param.getGrad();
                    float[] gradData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) grad).buffer;
                    
                    for (int i = 0; i < gradData.length; i++) {
                        gradData[i] *= clipCoef;
                    }
                }
            }
        }
    }
    
    /**
     * 保存检查点
     */
    private void saveCheckpoint() {
        String filename = String.format("sft_checkpoint_epoch%d_step%d.model", 
            currentEpoch, currentStep);
        String filepath = Paths.get(checkpointDir, filename).toString();
        
        try {
            model.save(new File(filepath));
            System.out.println("SFT检查点已保存: " + filepath);
        } catch (Exception e) {
            System.err.println("保存检查点失败: " + e.getMessage());
        }
    }
    
    /**
     * 创建检查点目录
     */
    private void createCheckpointDir() {
        try {
            Path path = Paths.get(checkpointDir);
            if (!Files.exists(path)) {
                Files.createDirectories(path);
            }
        } catch (IOException e) {
            System.err.println("创建检查点目录失败: " + e.getMessage());
        }
    }
    
    public List<Float> getLossHistory() {
        return new ArrayList<>(lossHistory);
    }
    
    public void setCheckpointDir(String checkpointDir) {
        this.checkpointDir = checkpointDir;
    }
}
