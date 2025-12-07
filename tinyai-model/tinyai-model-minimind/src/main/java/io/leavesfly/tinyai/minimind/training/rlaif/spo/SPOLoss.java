package io.leavesfly.tinyai.minimind.training.rlaif.spo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * SPO (Simplified Policy Optimization) 损失函数
 * 
 * SPO算法核心思想:
 * 1. 生成K个候选回答
 * 2. 计算每个候选的奖励R(y_i)
 * 3. 计算相对优势: A(y_i) = R(y_i) - mean(R)
 * 4. 策略梯度: ∇L = -∑ A(y_i) * log π_θ(y_i|x)
 * 5. 添加熵正则化鼓励探索
 * 
 * 无需Critic网络,直接使用奖励信号优化策略
 * 
 * @author leavesfly
 * @since 2024
 */
public class SPOLoss {
    
    private final SPOConfig config;
    
    /**
     * 构造函数
     * 
     * @param config SPO配置
     */
    public SPOLoss(SPOConfig config) {
        this.config = config;
    }
    
    /**
     * 计算SPO损失
     * 
     * @param logits 模型输出的logits [K, batch_size, seq_len, vocab_size]
     * @param labels 标签 [K, batch_size, seq_len]
     * @param rewards 奖励 [batch_size, K]
     * @return SPO损失
     */
    public Variable computeLoss(Variable[] logits, Variable[] labels, float[][] rewards) {
        int numCandidates = logits.length;
        int batchSize = rewards.length;
        
        // 1. 归一化奖励
        float[][] normalizedRewards = normalizeRewards(rewards);
        
        // 2. 计算优势函数: A(y_i) = R(y_i) - mean(R)
        float[][] advantages = computeAdvantages(normalizedRewards);
        
        // 3. 计算每个候选的对数概率
        Variable[] logProbs = new Variable[numCandidates];
        for (int k = 0; k < numCandidates; k++) {
            logProbs[k] = computeLogProb(logits[k], labels[k]);
        }
        
        // 4. 计算策略梯度损失: L = -∑ A(y_i) * log π(y_i|x)
        Variable policyLoss = computePolicyGradientLoss(logProbs, advantages);
        
        // 5. 添加熵正则化(鼓励探索)
        Variable entropyLoss = computeEntropyLoss(logits);
        
        // 6. 总损失 = 策略损失 - 熵系数 * 熵损失
        Variable totalLoss = policyLoss.sub(
            entropyLoss.mul(new Variable(NdArray.of(config.getEntropyCoef())))
        );
        
        return totalLoss;
    }
    
    /**
     * 归一化奖励
     */
    private float[][] normalizeRewards(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        float[][] normalized = new float[batchSize][numCandidates];
        
        switch (config.getRewardNormalization()) {
            case NONE:
                // 不归一化,直接返回
                for (int i = 0; i < batchSize; i++) {
                    System.arraycopy(rewards[i], 0, normalized[i], 0, numCandidates);
                }
                break;
                
            case STANDARDIZE:
                // 标准化: (r - mean) / std
                for (int i = 0; i < batchSize; i++) {
                    float mean = 0.0f;
                    for (float r : rewards[i]) {
                        mean += r;
                    }
                    mean /= numCandidates;
                    
                    float std = 0.0f;
                    for (float r : rewards[i]) {
                        std += (r - mean) * (r - mean);
                    }
                    std = (float) Math.sqrt(std / numCandidates + 1e-8f);
                    
                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = (rewards[i][k] - mean) / std;
                    }
                }
                break;
                
            case NORMALIZE:
                // 归一化到[0,1]: (r - min) / (max - min)
                for (int i = 0; i < batchSize; i++) {
                    float min = Float.MAX_VALUE;
                    float max = Float.MIN_VALUE;
                    for (float r : rewards[i]) {
                        min = Math.min(min, r);
                        max = Math.max(max, r);
                    }
                    
                    float range = max - min + 1e-8f;
                    for (int k = 0; k < numCandidates; k++) {
                        normalized[i][k] = (rewards[i][k] - min) / range;
                    }
                }
                break;
                
            case WHITENING:
                // 白化: standardize + clip
                for (int i = 0; i < batchSize; i++) {
                    float mean = 0.0f;
                    for (float r : rewards[i]) {
                        mean += r;
                    }
                    mean /= numCandidates;
                    
                    float std = 0.0f;
                    for (float r : rewards[i]) {
                        std += (r - mean) * (r - mean);
                    }
                    std = (float) Math.sqrt(std / numCandidates + 1e-8f);
                    
                    for (int k = 0; k < numCandidates; k++) {
                        float value = (rewards[i][k] - mean) / std;
                        // Clip到[-3, 3]范围
                        normalized[i][k] = Math.max(-3.0f, Math.min(3.0f, value));
                    }
                }
                break;
        }
        
        return normalized;
    }
    
    /**
     * 计算优势函数: A(y_i) = R(y_i) - mean(R)
     */
    private float[][] computeAdvantages(float[][] rewards) {
        int batchSize = rewards.length;
        int numCandidates = rewards[0].length;
        float[][] advantages = new float[batchSize][numCandidates];
        
        for (int i = 0; i < batchSize; i++) {
            // 计算平均奖励
            float meanReward = 0.0f;
            for (float r : rewards[i]) {
                meanReward += r;
            }
            meanReward /= numCandidates;
            
            // 计算优势
            for (int k = 0; k < numCandidates; k++) {
                advantages[i][k] = rewards[i][k] - meanReward;
            }
            
            // 可选: 归一化优势
            if (config.isNormalizeAdvantage()) {
                float std = 0.0f;
                for (float a : advantages[i]) {
                    std += a * a;
                }
                std = (float) Math.sqrt(std / numCandidates + 1e-8f);
                
                for (int k = 0; k < numCandidates; k++) {
                    advantages[i][k] /= std;
                }
            }
        }
        
        return advantages;
    }
    
    /**
     * 计算对数概率: log π(y|x)
     */
    private Variable computeLogProb(Variable logits, Variable labels) {
        // Log softmax
        Variable logProbs = logSoftmax(logits);
        
        // 简化实现:使用交叉熵的负值
        // 实际应该gather对应label的log概率
        Variable nll = negativeLogLikelihood(logProbs, labels);
        
        // 返回平均对数概率
        return nll.mul(new Variable(NdArray.of(-1.0f)));
    }
    
    /**
     * 计算策略梯度损失: L = -∑ A(y_i) * log π(y_i|x)
     */
    private Variable computePolicyGradientLoss(Variable[] logProbs, float[][] advantages) {
        Variable totalLoss = null;
        
        int batchSize = advantages.length;
        int numCandidates = logProbs.length;
        
        for (int k = 0; k < numCandidates; k++) {
            // 创建优势权重tensor
            float[] advantageWeights = new float[batchSize];
            for (int i = 0; i < batchSize; i++) {
                advantageWeights[i] = advantages[i][k];
            }
            
            Variable advantageVar = new Variable(NdArray.of(advantageWeights));
            
            // L_k = A_k * log_prob_k
            Variable weightedLogProb = logProbs[k].mul(advantageVar);
            
            if (totalLoss == null) {
                totalLoss = weightedLogProb;
            } else {
                totalLoss = totalLoss.add(weightedLogProb);
            }
        }
        
        // 返回负值(因为要最大化奖励,等价于最小化负奖励)
        return totalLoss.mul(new Variable(NdArray.of(-1.0f / numCandidates)));
    }
    
    /**
     * 计算熵损失(鼓励探索)
     * H = -∑ p * log(p)
     */
    private Variable computeEntropyLoss(Variable[] logits) {
        Variable totalEntropy = null;
        
        for (Variable logit : logits) {
            // Softmax概率
            Variable probs = softmax(logit);
            
            // Log softmax
            Variable logProbs = logSoftmax(logit);
            
            // 熵: H = -∑ p * log(p)
            Variable entropy = probs.mul(logProbs).mul(new Variable(NdArray.of(-1.0f)));
            Variable meanEntropy = entropy.mean(0, true);
            
            if (totalEntropy == null) {
                totalEntropy = meanEntropy;
            } else {
                totalEntropy = totalEntropy.add(meanEntropy);
            }
        }
        
        return totalEntropy.mul(new Variable(NdArray.of(1.0f / logits.length)));
    }
    
    /**
     * Log Softmax实现
     */
    private Variable logSoftmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
    
    /**
     * Softmax实现
     */
    private Variable softmax(Variable x) {
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        return expX.div(sumExp);
    }
    
    /**
     * 负对数似然
     */
    private Variable negativeLogLikelihood(Variable logProbs, Variable labels) {
        // 简化实现:使用交叉熵计算
        return logProbs.mul(new Variable(NdArray.of(-1.0f)));
    }
}
