package io.leavesfly.tinyai.minimind.training.dpo;

/**
 * DPO (Direct Preference Optimization) 配置类
 * 
 * 控制DPO训练的关键超参数
 * 
 * @author leavesfly
 * @since 2024
 */
public class DPOConfig {
    
    /**
     * β参数 - KL散度惩罚系数
     * 
     * 控制策略模型与参考模型的偏离程度:
     * - 较大的β: 更保守,更接近参考模型
     * - 较小的β: 更激进,更关注偏好优化
     * 
     * 典型值: 0.1 - 0.5
     */
    private float beta;
    
    /**
     * 参考模型策略
     * 
     * - USE_INITIAL_MODEL: 使用训练开始时的模型副本作为参考
     * - USE_SFT_MODEL: 使用预训练的SFT模型作为参考
     */
    public enum ReferenceModelStrategy {
        USE_INITIAL_MODEL,  // 使用初始模型
        USE_SFT_MODEL      // 使用SFT模型
    }
    
    private ReferenceModelStrategy referenceStrategy;
    
    /**
     * 标签平滑系数
     * 
     * 减少过拟合,提升泛化能力
     * 典型值: 0.0 - 0.1
     */
    private float labelSmoothing;
    
    /**
     * 是否使用长度归一化
     * 
     * 对不同长度的响应进行公平比较
     */
    private boolean useLengthNormalization;
    
    /**
     * 是否只对response部分计算损失
     * 
     * true: 只计算response的DPO损失(推荐)
     * false: 计算整个序列的DPO损失
     */
    private boolean responseOnlyLoss;
    
    /**
     * 默认构造函数
     */
    public DPOConfig() {
        this.beta = 0.1f;
        this.referenceStrategy = ReferenceModelStrategy.USE_INITIAL_MODEL;
        this.labelSmoothing = 0.0f;
        this.useLengthNormalization = false;
        this.responseOnlyLoss = true;
    }
    
    /**
     * 完整构造函数
     */
    public DPOConfig(float beta, ReferenceModelStrategy referenceStrategy,
                     float labelSmoothing, boolean useLengthNormalization,
                     boolean responseOnlyLoss) {
        this.beta = beta;
        this.referenceStrategy = referenceStrategy;
        this.labelSmoothing = labelSmoothing;
        this.useLengthNormalization = useLengthNormalization;
        this.responseOnlyLoss = responseOnlyLoss;
    }
    
    /**
     * 验证配置有效性
     */
    public void validate() {
        if (beta <= 0) {
            throw new IllegalArgumentException("Beta must be positive, got: " + beta);
        }
        if (labelSmoothing < 0 || labelSmoothing > 1) {
            throw new IllegalArgumentException("Label smoothing must be in [0, 1], got: " + labelSmoothing);
        }
    }
    
    // Getters and Setters
    
    public float getBeta() {
        return beta;
    }
    
    public void setBeta(float beta) {
        this.beta = beta;
    }
    
    public ReferenceModelStrategy getReferenceStrategy() {
        return referenceStrategy;
    }
    
    public void setReferenceStrategy(ReferenceModelStrategy referenceStrategy) {
        this.referenceStrategy = referenceStrategy;
    }
    
    public float getLabelSmoothing() {
        return labelSmoothing;
    }
    
    public void setLabelSmoothing(float labelSmoothing) {
        this.labelSmoothing = labelSmoothing;
    }
    
    public boolean isUseLengthNormalization() {
        return useLengthNormalization;
    }
    
    public void setUseLengthNormalization(boolean useLengthNormalization) {
        this.useLengthNormalization = useLengthNormalization;
    }
    
    public boolean isResponseOnlyLoss() {
        return responseOnlyLoss;
    }
    
    public void setResponseOnlyLoss(boolean responseOnlyLoss) {
        this.responseOnlyLoss = responseOnlyLoss;
    }
    
    @Override
    public String toString() {
        return "DPOConfig{" +
                "beta=" + beta +
                ", referenceStrategy=" + referenceStrategy +
                ", labelSmoothing=" + labelSmoothing +
                ", useLengthNormalization=" + useLengthNormalization +
                ", responseOnlyLoss=" + responseOnlyLoss +
                '}';
    }
    
    /**
     * 创建默认配置
     */
    public static DPOConfig createDefault() {
        return new DPOConfig();
    }
    
    /**
     * 创建保守配置(较大β,更接近参考模型)
     */
    public static DPOConfig createConservative() {
        DPOConfig config = new DPOConfig();
        config.setBeta(0.5f);
        config.setLabelSmoothing(0.1f);
        config.setUseLengthNormalization(true);
        return config;
    }
    
    /**
     * 创建激进配置(较小β,更关注偏好优化)
     */
    public static DPOConfig createAggressive() {
        DPOConfig config = new DPOConfig();
        config.setBeta(0.05f);
        config.setLabelSmoothing(0.0f);
        config.setUseLengthNormalization(false);
        return config;
    }
}
