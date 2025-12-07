package io.leavesfly.tinyai.minimind.training.rlaif.spo;

/**
 * SPO (Simplified Policy Optimization) 配置类
 * 
 * SPO是简化版的PPO算法,主要特点:
 * 1. 无需Critic价值网络
 * 2. 使用相对优势函数: A(y) = R(y) - mean(R)
 * 3. 直接策略梯度优化
 * 4. 计算效率高,适合资源受限场景
 * 
 * @author leavesfly
 * @since 2024
 */
public class SPOConfig {
    
    /**
     * 候选回答数量K(每个prompt生成K个候选)
     */
    private int numCandidates;
    
    /**
     * 学习率
     */
    private float learningRate;
    
    /**
     * 奖励归一化方式
     */
    private RewardNormalization rewardNormalization;
    
    /**
     * 优势函数归一化
     */
    private boolean normalizeAdvantage;
    
    /**
     * 熵正则化系数(鼓励探索)
     */
    private float entropyCoef;
    
    /**
     * 梯度裁剪阈值
     */
    private float maxGradNorm;
    
    /**
     * 温度参数(采样温度)
     */
    private float temperature;
    
    /**
     * 奖励裁剪范围
     */
    private float rewardClipMin;
    private float rewardClipMax;
    
    /**
     * 奖励归一化方式枚举
     */
    public enum RewardNormalization {
        NONE,           // 不归一化
        STANDARDIZE,    // 标准化: (r - mean) / std
        NORMALIZE,      // 归一化到[0,1]: (r - min) / (max - min)
        WHITENING      // 白化: standardize + clip
    }
    
    /**
     * 默认构造函数
     */
    public SPOConfig() {
        this.numCandidates = 4;
        this.learningRate = 1e-5f;
        this.rewardNormalization = RewardNormalization.STANDARDIZE;
        this.normalizeAdvantage = true;
        this.entropyCoef = 0.01f;
        this.maxGradNorm = 1.0f;
        this.temperature = 1.0f;
        this.rewardClipMin = -10.0f;
        this.rewardClipMax = 10.0f;
    }
    
    /**
     * 完整构造函数
     */
    public SPOConfig(int numCandidates, float learningRate, RewardNormalization rewardNormalization,
                    boolean normalizeAdvantage, float entropyCoef, float maxGradNorm,
                    float temperature, float rewardClipMin, float rewardClipMax) {
        this.numCandidates = numCandidates;
        this.learningRate = learningRate;
        this.rewardNormalization = rewardNormalization;
        this.normalizeAdvantage = normalizeAdvantage;
        this.entropyCoef = entropyCoef;
        this.maxGradNorm = maxGradNorm;
        this.temperature = temperature;
        this.rewardClipMin = rewardClipMin;
        this.rewardClipMax = rewardClipMax;
    }
    
    // Getters and Setters
    
    public int getNumCandidates() {
        return numCandidates;
    }
    
    public void setNumCandidates(int numCandidates) {
        if (numCandidates < 2) {
            throw new IllegalArgumentException("numCandidates must be >= 2");
        }
        this.numCandidates = numCandidates;
    }
    
    public float getLearningRate() {
        return learningRate;
    }
    
    public void setLearningRate(float learningRate) {
        this.learningRate = learningRate;
    }
    
    public RewardNormalization getRewardNormalization() {
        return rewardNormalization;
    }
    
    public void setRewardNormalization(RewardNormalization rewardNormalization) {
        this.rewardNormalization = rewardNormalization;
    }
    
    public boolean isNormalizeAdvantage() {
        return normalizeAdvantage;
    }
    
    public void setNormalizeAdvantage(boolean normalizeAdvantage) {
        this.normalizeAdvantage = normalizeAdvantage;
    }
    
    public float getEntropyCoef() {
        return entropyCoef;
    }
    
    public void setEntropyCoef(float entropyCoef) {
        this.entropyCoef = entropyCoef;
    }
    
    public float getMaxGradNorm() {
        return maxGradNorm;
    }
    
    public void setMaxGradNorm(float maxGradNorm) {
        this.maxGradNorm = maxGradNorm;
    }
    
    public float getTemperature() {
        return temperature;
    }
    
    public void setTemperature(float temperature) {
        this.temperature = temperature;
    }
    
    public float getRewardClipMin() {
        return rewardClipMin;
    }
    
    public void setRewardClipMin(float rewardClipMin) {
        this.rewardClipMin = rewardClipMin;
    }
    
    public float getRewardClipMax() {
        return rewardClipMax;
    }
    
    public void setRewardClipMax(float rewardClipMax) {
        this.rewardClipMax = rewardClipMax;
    }
    
    /**
     * 创建默认配置
     */
    public static SPOConfig createDefault() {
        return new SPOConfig();
    }
    
    /**
     * 创建保守配置(更稳定,适合初期训练)
     */
    public static SPOConfig createConservative() {
        SPOConfig config = new SPOConfig();
        config.setLearningRate(5e-6f);
        config.setEntropyCoef(0.02f);
        config.setMaxGradNorm(0.5f);
        config.setRewardClipMin(-5.0f);
        config.setRewardClipMax(5.0f);
        return config;
    }
    
    /**
     * 创建激进配置(更快收敛,适合后期微调)
     */
    public static SPOConfig createAggressive() {
        SPOConfig config = new SPOConfig();
        config.setLearningRate(2e-5f);
        config.setEntropyCoef(0.005f);
        config.setMaxGradNorm(2.0f);
        config.setRewardNormalization(RewardNormalization.WHITENING);
        return config;
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (numCandidates < 2) {
            throw new IllegalArgumentException("numCandidates must be >= 2");
        }
        if (learningRate <= 0) {
            throw new IllegalArgumentException("learningRate must be positive");
        }
        if (entropyCoef < 0) {
            throw new IllegalArgumentException("entropyCoef must be non-negative");
        }
        if (maxGradNorm <= 0) {
            throw new IllegalArgumentException("maxGradNorm must be positive");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("temperature must be positive");
        }
        if (rewardClipMin >= rewardClipMax) {
            throw new IllegalArgumentException("rewardClipMin must be < rewardClipMax");
        }
    }
    
    @Override
    public String toString() {
        return String.format("SPOConfig{candidates=%d, lr=%.2e, norm=%s, entropy=%.3f, gradClip=%.1f, temp=%.1f}",
            numCandidates, learningRate, rewardNormalization, entropyCoef, maxGradNorm, temperature);
    }
}
