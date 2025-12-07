package io.leavesfly.tinyai.minimind.training.lora;

import java.io.Serializable;

/**
 * LoRA配置类
 * 
 * 低秩适配(Low-Rank Adaptation)配置
 * LoRA通过在原始权重矩阵旁添加低秩分解矩阵来实现参数高效微调
 * 
 * @author leavesfly
 * @since 2024
 */
public class LoRAConfig implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    // LoRA秩(r) - 低秩分解的秩
    private int rank;
    
    // LoRA缩放因子(alpha) - 控制LoRA权重的影响程度
    private float alpha;
    
    // LoRA Dropout
    private float dropout;
    
    // 目标模块名称(应用LoRA的层)
    private String[] targetModules;
    
    // 是否冻结原始模型参数
    private boolean freezeOriginal;
    
    /**
     * 默认构造函数
     */
    public LoRAConfig() {
        this.rank = 8;
        this.alpha = 16.0f;
        this.dropout = 0.1f;
        this.targetModules = new String[]{"queryProj", "valueProj"};
        this.freezeOriginal = true;
    }
    
    /**
     * 完整构造函数
     */
    public LoRAConfig(int rank, float alpha, float dropout, 
                      String[] targetModules, boolean freezeOriginal) {
        this.rank = rank;
        this.alpha = alpha;
        this.dropout = dropout;
        this.targetModules = targetModules;
        this.freezeOriginal = freezeOriginal;
    }
    
    /**
     * 创建默认配置(仅Q、V投影)
     */
    public static LoRAConfig createDefault() {
        return new LoRAConfig();
    }
    
    /**
     * 创建全量配置(Q、K、V、O投影)
     */
    public static LoRAConfig createFullAttention() {
        return new LoRAConfig(
            8, 16.0f, 0.1f,
            new String[]{"queryProj", "keyProj", "valueProj", "outputProj"},
            true
        );
    }
    
    /**
     * 获取缩放系数
     * scaling = alpha / r
     */
    public float getScaling() {
        return alpha / rank;
    }
    
    /**
     * 估算额外参数量
     * 
     * @param inFeatures 输入特征数
     * @param outFeatures 输出特征数
     * @return 额外参数量
     */
    public int estimateExtraParams(int inFeatures, int outFeatures) {
        // LoRA参数量 = (inFeatures * rank + rank * outFeatures) * 目标模块数
        int paramsPerModule = inFeatures * rank + rank * outFeatures;
        return paramsPerModule * targetModules.length;
    }
    
    /**
     * 验证配置
     */
    public void validate() {
        if (rank <= 0) {
            throw new IllegalArgumentException("LoRA rank必须大于0");
        }
        if (alpha <= 0) {
            throw new IllegalArgumentException("LoRA alpha必须大于0");
        }
        if (dropout < 0 || dropout >= 1) {
            throw new IllegalArgumentException("LoRA dropout必须在[0, 1)范围内");
        }
        if (targetModules == null || targetModules.length == 0) {
            throw new IllegalArgumentException("必须指定至少一个目标模块");
        }
    }
    
    // Getters and Setters
    
    public int getRank() {
        return rank;
    }
    
    public void setRank(int rank) {
        this.rank = rank;
    }
    
    public float getAlpha() {
        return alpha;
    }
    
    public void setAlpha(float alpha) {
        this.alpha = alpha;
    }
    
    public float getDropout() {
        return dropout;
    }
    
    public void setDropout(float dropout) {
        this.dropout = dropout;
    }
    
    public String[] getTargetModules() {
        return targetModules;
    }
    
    public void setTargetModules(String[] targetModules) {
        this.targetModules = targetModules;
    }
    
    public boolean isFreezeOriginal() {
        return freezeOriginal;
    }
    
    public void setFreezeOriginal(boolean freezeOriginal) {
        this.freezeOriginal = freezeOriginal;
    }
    
    @Override
    public String toString() {
        return String.format(
            "LoRAConfig{rank=%d, alpha=%.1f, dropout=%.2f, scaling=%.2f, targets=%d}",
            rank, alpha, dropout, getScaling(), targetModules.length
        );
    }
}
