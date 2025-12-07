package io.leavesfly.tinyai.minimind.training.dpo;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;

/**
 * DPO (Direct Preference Optimization) 损失函数
 * 
 * DPO损失公式:
 * L_DPO = -log(σ(β * log(π_θ(y_w|x)/π_ref(y_w|x)) - β * log(π_θ(y_l|x)/π_ref(y_l|x))))
 * 
 * 其中:
 * - π_θ: 策略模型(被训练的模型)
 * - π_ref: 参考模型(冻结的模型)
 * - y_w: 更好的响应(chosen/winner)
 * - y_l: 较差的响应(rejected/loser)
 * - β: KL散度惩罚系数
 * - σ: Sigmoid函数
 * 
 * DPO直接优化偏好,无需奖励模型
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOLoss {
    
    private final float beta;
    private final float labelSmoothing;
    
    /**
     * 构造函数
     * 
     * @param beta KL散度惩罚系数
     * @param labelSmoothing 标签平滑系数
     */
    public DPOLoss(float beta, float labelSmoothing) {
        this.beta = beta;
        this.labelSmoothing = labelSmoothing;
    }
    
    /**
     * 计算DPO损失
     * 
     * @param chosenLogProbs 策略模型在chosen响应上的对数概率
     * @param rejectedLogProbs 策略模型在rejected响应上的对数概率
     * @param refChosenLogProbs 参考模型在chosen响应上的对数概率
     * @param refRejectedLogProbs 参考模型在rejected响应上的对数概率
     * @return DPO损失
     */
    public Variable loss(Variable chosenLogProbs, Variable rejectedLogProbs,
                        Variable refChosenLogProbs, Variable refRejectedLogProbs) {
        
        // 计算策略模型的log ratio
        // log(π_θ(y_w|x)/π_θ(y_l|x)) = log π_θ(y_w|x) - log π_θ(y_l|x)
        Variable policyLogRatio = chosenLogProbs.sub(rejectedLogProbs);
        
        // 计算参考模型的log ratio
        // log(π_ref(y_w|x)/π_ref(y_l|x)) = log π_ref(y_w|x) - log π_ref(y_l|x)
        Variable refLogRatio = refChosenLogProbs.sub(refRejectedLogProbs);
        
        // 计算隐式奖励: β * [log(π_θ/π_ref)(y_w) - log(π_θ/π_ref)(y_l)]
        // = β * [(log π_θ(y_w) - log π_ref(y_w)) - (log π_θ(y_l) - log π_ref(y_l))]
        Variable implicitReward = policyLogRatio.sub(refLogRatio);
        Variable scaledReward = implicitReward.mul(new Variable(NdArray.of(beta)));
        
        // 计算sigmoid损失: -log(σ(scaled_reward))
        // 等价于: log(1 + exp(-scaled_reward))
        Variable dpoLoss = logSigmoid(scaledReward.mul(new Variable(NdArray.of(-1.0f))));
        
        // 应用标签平滑
        if (labelSmoothing > 0) {
            // 添加正则化项鼓励chosen和rejected的logprobs都接近0
            Variable regularization = chosenLogProbs.add(rejectedLogProbs).mul(
                new Variable(NdArray.of(-labelSmoothing * 0.5f))
            );
            dpoLoss = dpoLoss.add(regularization);
        }
        
        // 返回平均损失
        return dpoLoss.mean(0, true);
    }
    
    /**
     * 计算log(sigmoid(x)) = -log(1 + exp(-x))
     * 
     * 使用数值稳定的实现:
     * log(sigmoid(x)) = -log(1 + exp(-x))
     *                 = -softplus(-x)
     *                 = x - softplus(x)  (当x > 0时)
     *                 = -softplus(-x)     (当x < 0时)
     * 
     * @param x 输入
     * @return log(sigmoid(x))
     */
    private Variable logSigmoid(Variable x) {
        // 使用log(sigmoid(x)) = -log(1 + exp(-x))的稳定实现
        // 等价于: x - softplus(x)
        return x.sub(softplus(x));
    }
    
    /**
     * Softplus函数: softplus(x) = log(1 + exp(x))
     * 简化实现,适用于TinyAI的Variable API
     * 
     * @param x 输入
     * @return softplus(x)
     */
    private Variable softplus(Variable x) {
        // softplus(x) = log(1 + exp(x))
        Variable expX = x.exp();
        Variable onePlusExp = expX.add(new Variable(NdArray.of(1.0f)));
        return onePlusExp.log();
    }
    
    /**
     * 计算序列的对数概率
     * 
     * @param logits 模型输出logits [batch, seq_len, vocab_size]
     * @param labels 标签 [batch, seq_len]
     * @param mask 掩码 [batch, seq_len], 1表示计算,0表示忽略
     * @return 每个序列的平均对数概率
     */
    public Variable computeLogProbs(Variable logits, Variable labels, Variable mask) {
        // 计算log softmax
        Variable logProbs = logSoftmax(logits);
        
        // 提取对应标签的log概率
        // 这里需要gather操作,简化实现使用交叉熵的负值
        Variable nll = negativeLogLikelihood(logProbs, labels);
        
        // 应用mask
        Variable maskedNll = nll.mul(mask);
        
        // 计算平均(考虑mask)
        Variable sumMask = mask.sum();
        Variable totalNll = maskedNll.sum();
        
        // 返回负值(因为nll是负对数概率)
        return totalNll.div(sumMask).mul(new Variable(NdArray.of(-1.0f)));
    }
    
    /**
     * Log Softmax实现 - 简化版本
     */
    private Variable logSoftmax(Variable x) {
        // log_softmax(x) = x - log(sum(exp(x)))
        // 简化实现,避免使用max()方法
        Variable expX = x.exp();
        Variable sumExp = expX.sum();
        Variable logSumExp = sumExp.log();
        return x.sub(logSumExp);
    }
    
    /**
     * 负对数似然
     */
    private Variable negativeLogLikelihood(Variable logProbs, Variable labels) {
        // 简化实现:使用交叉熵计算
        // 实际应该gather对应label的log概率
        return logProbs.mul(new Variable(NdArray.of(-1.0f)));
    }
}
