package io.leavesfly.tinyai.minimind.training.rlaif.grpo;

/**
 * GRPO (Group Relative Policy Optimization) 配置类
 * 
 * GRPO是PPO的变体,主要特点:
 * 1. 组相对优势:将候选分组,计算组内相对优势
 * 2. 仍使用Clip机制防止策略更新过大
 * 3. 更适合大规模候选场景(K>>2)
 * 4. 减少奖励估计方差
 * 
 * 核心思想:
 * - 将K个候选分为G组,每组g_size个候选
 * - 组内计算相对优势: A_relative = R(y_i) - mean_group(R)
 * - 组间也可以有相对关系
 * 
 * @author leavesfly
 * @since 2024
 */
public class GRPOConfig {
    
    /**
     * 候选回答数量K
     */
    private int numCandidates;
    
    /**
     * 组大小(每组包含多少个候选)
     */
    private int groupSize;
    
    /**
     * Actor学习率
     */
    private float actorLearningRate;
    
    /**
     * Critic学习率
     */
    private float criticLearningRate;
    
    /**
     * Clip范围ε
     */
    private float clipEpsilon;
    
    /**
     * 价值损失系数
     */
    private float valueLossCoef;
    
    /**
     * 熵正则化系数
     */
    private float entropyCoef;
    
    /**
     * 梯度裁剪阈值
     */
    private float maxGradNorm;
    
    /**
     * GRPO更新轮数
     */
    private int grpoEpochs;
    
    /**
     * 是否使用优势归一化
     */
    private boolean normalizeAdvantage;
    
    /**
     * 是否使用组间对比
     * 如果true,还会考虑组间的相对优势
     */
    private boolean useGroupContrast;
    
    /**
     * 温度参数
     */
    private float temperature;
    
    /**
     * 奖励归一化方式
     */
    private RewardNormalization rewardNormalization;
    
    /**
     * 奖励归一化方式枚举
     */
    public enum RewardNormalization {
        NONE,           // 不归一化
        STANDARDIZE,    // 标准化: (r - mean) / std
        NORMALIZE,      // 归一化到[0,1]
        WHITENING       // 白化: standardize + clip
    }
    
    /**
     * 默认构造函数
     */
    public GRPOConfig() {
        this.numCandidates = 8;
        this.groupSize = 4;
        this.actorLearningRate = 1e-5f;
        this.criticLearningRate = 3e-5f;
        this.clipEpsilon = 0.2f;
        this.valueLossCoef = 0.5f;
        this.entropyCoef = 0.01f;
        this.maxGradNorm = 0.5f;
        this.grpoEpochs = 4;
        this.normalizeAdvantage = true;
        this.useGroupContrast = true;
        this.temperature = 1.0f;
        this.rewardNormalization = RewardNormalization.STANDARDIZE;
    }
    
    /**
     * 完整构造函数
     */
    public GRPOConfig(int numCandidates, int groupSize, float actorLearningRate,
                     float criticLearningRate, float clipEpsilon, float valueLossCoef,
                     float entropyCoef, float maxGradNorm, int grpoEpochs,
                     boolean normalizeAdvantage, boolean useGroupContrast,
                     float temperature, RewardNormalization rewardNormalization) {
        this.numCandidates = numCandidates;
        this.groupSize = groupSize;
        this.actorLearningRate = actorLearningRate;
        this.criticLearningRate = criticLearningRate;
        this.clipEpsilon = clipEpsilon;
        this.valueLossCoef = valueLossCoef;
        this.entropyCoef = entropyCoef;
        this.maxGradNorm = maxGradNorm;
        this.grpoEpochs = grpoEpochs;
        this.normalizeAdvantage = normalizeAdvantage;
        this.useGroupContrast = useGroupContrast;
        this.temperature = temperature;
        this.rewardNormalization = rewardNormalization;
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
    
    public int getGroupSize() {
        return groupSize;
    }
    
    public void setGroupSize(int groupSize) {
        if (groupSize < 2) {
            throw new IllegalArgumentException("groupSize must be >= 2");
        }
        this.groupSize = groupSize;
    }
    
    public float getActorLearningRate() {
        return actorLearningRate;
    }
    
    public void setActorLearningRate(float actorLearningRate) {
        this.actorLearningRate = actorLearningRate;
    }
    
    public float getCriticLearningRate() {
        return criticLearningRate;
    }
    
    public void setCriticLearningRate(float criticLearningRate) {
        this.criticLearningRate = criticLearningRate;
    }
    
    public float getClipEpsilon() {
        return clipEpsilon;
    }
    
    public void setClipEpsilon(float clipEpsilon) {
        this.clipEpsilon = clipEpsilon;
    }
    
    public float getValueLossCoef() {
        return valueLossCoef;
    }
    
    public void setValueLossCoef(float valueLossCoef) {
        this.valueLossCoef = valueLossCoef;
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
    
    public int getGrpoEpochs() {
        return grpoEpochs;
    }
    
    public void setGrpoEpochs(int grpoEpochs) {
        this.grpoEpochs = grpoEpochs;
    }
    
    public boolean isNormalizeAdvantage() {
        return normalizeAdvantage;
    }
    
    public void setNormalizeAdvantage(boolean normalizeAdvantage) {
        this.normalizeAdvantage = normalizeAdvantage;
    }
    
    public boolean isUseGroupContrast() {
        return useGroupContrast;
    }
    
    public void setUseGroupContrast(boolean useGroupContrast) {
        this.useGroupContrast = useGroupContrast;
    }
    
    public float getTemperature() {
        return temperature;
    }
    
    public void setTemperature(float temperature) {
        this.temperature = temperature;
    }
    
    public RewardNormalization getRewardNormalization() {
        return rewardNormalization;
    }
    
    public void setRewardNormalization(RewardNormalization rewardNormalization) {
        this.rewardNormalization = rewardNormalization;
    }
    
    /**
     * 获取组数量
     */
    public int getNumGroups() {
        return (numCandidates + groupSize - 1) / groupSize;
    }
    
    /**
     * 创建默认配置
     */
    public static GRPOConfig createDefault() {
        return new GRPOConfig();
    }
    
    /**
     * 创建保守配置(更稳定)
     */
    public static GRPOConfig createConservative() {
        GRPOConfig config = new GRPOConfig();
        config.setActorLearningRate(5e-6f);
        config.setCriticLearningRate(1e-5f);
        config.setClipEpsilon(0.1f);
        config.setMaxGradNorm(0.3f);
        config.setGrpoEpochs(3);
        config.setUseGroupContrast(false);
        return config;
    }
    
    /**
     * 创建激进配置(更快收敛)
     */
    public static GRPOConfig createAggressive() {
        GRPOConfig config = new GRPOConfig();
        config.setActorLearningRate(2e-5f);
        config.setCriticLearningRate(5e-5f);
        config.setClipEpsilon(0.3f);
        config.setEntropyCoef(0.005f);
        config.setGrpoEpochs(6);
        config.setGroupSize(2);
        config.setUseGroupContrast(true);
        return config;
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (numCandidates < 2) {
            throw new IllegalArgumentException("numCandidates must be >= 2");
        }
        if (groupSize < 2) {
            throw new IllegalArgumentException("groupSize must be >= 2");
        }
        if (groupSize > numCandidates) {
            throw new IllegalArgumentException("groupSize must be <= numCandidates");
        }
        if (actorLearningRate <= 0) {
            throw new IllegalArgumentException("actorLearningRate must be positive");
        }
        if (criticLearningRate <= 0) {
            throw new IllegalArgumentException("criticLearningRate must be positive");
        }
        if (clipEpsilon <= 0 || clipEpsilon >= 1) {
            throw new IllegalArgumentException("clipEpsilon must be in (0, 1)");
        }
        if (valueLossCoef < 0) {
            throw new IllegalArgumentException("valueLossCoef must be non-negative");
        }
        if (entropyCoef < 0) {
            throw new IllegalArgumentException("entropyCoef must be non-negative");
        }
        if (maxGradNorm <= 0) {
            throw new IllegalArgumentException("maxGradNorm must be positive");
        }
        if (grpoEpochs < 1) {
            throw new IllegalArgumentException("grpoEpochs must be >= 1");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("temperature must be positive");
        }
    }
    
    @Override
    public String toString() {
        return String.format("GRPOConfig{candidates=%d, groupSize=%d, groups=%d, " +
                "actorLR=%.2e, criticLR=%.2e, clip=%.2f, epochs=%d, contrast=%b}",
            numCandidates, groupSize, getNumGroups(),
            actorLearningRate, criticLearningRate, clipEpsilon, 
            grpoEpochs, useGroupContrast);
    }
}
