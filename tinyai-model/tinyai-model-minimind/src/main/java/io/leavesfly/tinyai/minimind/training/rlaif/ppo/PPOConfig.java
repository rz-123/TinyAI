package io.leavesfly.tinyai.minimind.training.rlaif.ppo;

/**
 * PPO (Proximal Policy Optimization) 配置类
 * 
 * PPO是当前最流行的强化学习算法之一,主要特点:
 * 1. 使用Critic价值网络评估状态价值
 * 2. Clipped Surrogate Objective防止策略更新过大
 * 3. GAE (Generalized Advantage Estimation)
 * 4. Actor-Critic架构
 * 
 * 核心公式:
 * L^{CLIP}(θ) = min(r_t(θ)*A_t, clip(r_t(θ), 1-ε, 1+ε)*A_t)
 * 其中 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
 * 
 * @author leavesfly
 * @since 2024
 */
public class PPOConfig {
    
    /**
     * 候选回答数量K(每个prompt生成K个候选)
     */
    private int numCandidates;
    
    /**
     * Actor学习率(策略网络)
     */
    private float actorLearningRate;
    
    /**
     * Critic学习率(价值网络)
     */
    private float criticLearningRate;
    
    /**
     * Clip范围ε (通常0.1-0.3)
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
     * GAE lambda参数
     */
    private float gaeLambda;
    
    /**
     * 折扣因子γ
     */
    private float gamma;
    
    /**
     * 梯度裁剪阈值
     */
    private float maxGradNorm;
    
    /**
     * PPO更新轮数(多次更新同一批数据)
     */
    private int ppoEpochs;
    
    /**
     * Mini-batch大小(从经验池中采样)
     */
    private int miniBatchSize;
    
    /**
     * 是否使用优势归一化
     */
    private boolean normalizeAdvantage;
    
    /**
     * 是否使用价值函数裁剪
     */
    private boolean clipValueLoss;
    
    /**
     * 温度参数
     */
    private float temperature;
    
    /**
     * 默认构造函数
     */
    public PPOConfig() {
        this.numCandidates = 4;
        this.actorLearningRate = 1e-5f;
        this.criticLearningRate = 3e-5f;
        this.clipEpsilon = 0.2f;
        this.valueLossCoef = 0.5f;
        this.entropyCoef = 0.01f;
        this.gaeLambda = 0.95f;
        this.gamma = 0.99f;
        this.maxGradNorm = 0.5f;
        this.ppoEpochs = 4;
        this.miniBatchSize = 64;
        this.normalizeAdvantage = true;
        this.clipValueLoss = true;
        this.temperature = 1.0f;
    }
    
    /**
     * 完整构造函数
     */
    public PPOConfig(int numCandidates, float actorLearningRate, float criticLearningRate,
                    float clipEpsilon, float valueLossCoef, float entropyCoef,
                    float gaeLambda, float gamma, float maxGradNorm,
                    int ppoEpochs, int miniBatchSize, boolean normalizeAdvantage,
                    boolean clipValueLoss, float temperature) {
        this.numCandidates = numCandidates;
        this.actorLearningRate = actorLearningRate;
        this.criticLearningRate = criticLearningRate;
        this.clipEpsilon = clipEpsilon;
        this.valueLossCoef = valueLossCoef;
        this.entropyCoef = entropyCoef;
        this.gaeLambda = gaeLambda;
        this.gamma = gamma;
        this.maxGradNorm = maxGradNorm;
        this.ppoEpochs = ppoEpochs;
        this.miniBatchSize = miniBatchSize;
        this.normalizeAdvantage = normalizeAdvantage;
        this.clipValueLoss = clipValueLoss;
        this.temperature = temperature;
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
    
    public float getGaeLambda() {
        return gaeLambda;
    }
    
    public void setGaeLambda(float gaeLambda) {
        this.gaeLambda = gaeLambda;
    }
    
    public float getGamma() {
        return gamma;
    }
    
    public void setGamma(float gamma) {
        this.gamma = gamma;
    }
    
    public float getMaxGradNorm() {
        return maxGradNorm;
    }
    
    public void setMaxGradNorm(float maxGradNorm) {
        this.maxGradNorm = maxGradNorm;
    }
    
    public int getPpoEpochs() {
        return ppoEpochs;
    }
    
    public void setPpoEpochs(int ppoEpochs) {
        this.ppoEpochs = ppoEpochs;
    }
    
    public int getMiniBatchSize() {
        return miniBatchSize;
    }
    
    public void setMiniBatchSize(int miniBatchSize) {
        this.miniBatchSize = miniBatchSize;
    }
    
    public boolean isNormalizeAdvantage() {
        return normalizeAdvantage;
    }
    
    public void setNormalizeAdvantage(boolean normalizeAdvantage) {
        this.normalizeAdvantage = normalizeAdvantage;
    }
    
    public boolean isClipValueLoss() {
        return clipValueLoss;
    }
    
    public void setClipValueLoss(boolean clipValueLoss) {
        this.clipValueLoss = clipValueLoss;
    }
    
    public float getTemperature() {
        return temperature;
    }
    
    public void setTemperature(float temperature) {
        this.temperature = temperature;
    }
    
    /**
     * 创建默认配置
     */
    public static PPOConfig createDefault() {
        return new PPOConfig();
    }
    
    /**
     * 创建保守配置(更稳定,适合初期训练)
     */
    public static PPOConfig createConservative() {
        PPOConfig config = new PPOConfig();
        config.setActorLearningRate(5e-6f);
        config.setCriticLearningRate(1e-5f);
        config.setClipEpsilon(0.1f);
        config.setMaxGradNorm(0.3f);
        config.setPpoEpochs(3);
        return config;
    }
    
    /**
     * 创建激进配置(更快收敛,适合后期微调)
     */
    public static PPOConfig createAggressive() {
        PPOConfig config = new PPOConfig();
        config.setActorLearningRate(2e-5f);
        config.setCriticLearningRate(5e-5f);
        config.setClipEpsilon(0.3f);
        config.setEntropyCoef(0.005f);
        config.setPpoEpochs(6);
        return config;
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (numCandidates < 2) {
            throw new IllegalArgumentException("numCandidates must be >= 2");
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
        if (gaeLambda < 0 || gaeLambda > 1) {
            throw new IllegalArgumentException("gaeLambda must be in [0, 1]");
        }
        if (gamma < 0 || gamma > 1) {
            throw new IllegalArgumentException("gamma must be in [0, 1]");
        }
        if (maxGradNorm <= 0) {
            throw new IllegalArgumentException("maxGradNorm must be positive");
        }
        if (ppoEpochs < 1) {
            throw new IllegalArgumentException("ppoEpochs must be >= 1");
        }
        if (miniBatchSize < 1) {
            throw new IllegalArgumentException("miniBatchSize must be >= 1");
        }
        if (temperature <= 0) {
            throw new IllegalArgumentException("temperature must be positive");
        }
    }
    
    @Override
    public String toString() {
        return String.format("PPOConfig{candidates=%d, actorLR=%.2e, criticLR=%.2e, clip=%.2f, " +
                "valueCoef=%.2f, entropy=%.3f, lambda=%.2f, gamma=%.2f, epochs=%d}",
            numCandidates, actorLearningRate, criticLearningRate, clipEpsilon,
            valueLossCoef, entropyCoef, gaeLambda, gamma, ppoEpochs);
    }
}
