package io.leavesfly.tinyai.minimind.moe;

/**
 * MoE Configuration - MoE配置类
 * 
 * 管理MoE架构的所有超参数和配置项
 * 
 * @author leavesfly
 * @since 2024
 */
public class MoEConfig {
    
    // ========== 基础配置 ==========
    private int inputDim;           // 输入维度
    private int hiddenDim;          // 专家隐藏层维度
    private int outputDim;          // 输出维度
    
    // ========== 专家配置 ==========
    private int numExperts;         // 专家数量
    private int topK;               // Top-K选择数量
    private float noiseFactor;      // 路由噪声因子
    
    // ========== 负载均衡配置 ==========
    private float importanceCoef;   // 重要性损失系数
    private float loadCoef;         // 负载损失系数
    private boolean enableLoadBalance; // 是否启用负载均衡
    
    // ========== 训练配置 ==========
    private boolean sharedExperts;  // 是否使用共享专家
    private float dropoutRate;      // Dropout比率
    
    /**
     * 私有构造函数(使用Builder创建)
     */
    private MoEConfig() {}
    
    /**
     * 创建Builder
     */
    public static Builder builder() {
        return new Builder();
    }
    
    // ========== 预设配置 ==========
    
    /**
     * 小型MoE配置(4专家,适合145M参数)
     * - 4个专家
     * - Top-2选择
     * - 启用负载均衡
     */
    public static MoEConfig small() {
        return builder()
            .inputDim(512)
            .hiddenDim(2048)
            .outputDim(512)
            .numExperts(4)
            .topK(2)
            .noiseFactor(0.1f)
            .enableLoadBalance(true)
            .importanceCoef(0.01f)
            .loadCoef(0.01f)
            .build();
    }
    
    /**
     * 中型MoE配置(8专家)
     */
    public static MoEConfig medium() {
        return builder()
            .inputDim(512)
            .hiddenDim(2048)
            .outputDim(512)
            .numExperts(8)
            .topK(2)
            .noiseFactor(0.1f)
            .enableLoadBalance(true)
            .importanceCoef(0.01f)
            .loadCoef(0.01f)
            .build();
    }
    
    /**
     * 大型MoE配置(16专家)
     */
    public static MoEConfig large() {
        return builder()
            .inputDim(512)
            .hiddenDim(2048)
            .outputDim(512)
            .numExperts(16)
            .topK(2)
            .noiseFactor(0.1f)
            .enableLoadBalance(true)
            .importanceCoef(0.01f)
            .loadCoef(0.01f)
            .build();
    }
    
    // ========== Getters ==========
    
    public int getInputDim() {
        return inputDim;
    }
    
    public int getHiddenDim() {
        return hiddenDim;
    }
    
    public int getOutputDim() {
        return outputDim;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public int getTopK() {
        return topK;
    }
    
    public float getNoiseFactor() {
        return noiseFactor;
    }
    
    public float getImportanceCoef() {
        return importanceCoef;
    }
    
    public float getLoadCoef() {
        return loadCoef;
    }
    
    public boolean isEnableLoadBalance() {
        return enableLoadBalance;
    }
    
    public boolean isSharedExperts() {
        return sharedExperts;
    }
    
    public float getDropoutRate() {
        return dropoutRate;
    }
    
    /**
     * 计算参数数量
     */
    public long calculateParameters() {
        // Router参数: input_dim * num_experts
        long routerParams = (long) inputDim * numExperts;
        
        // 单个专家参数: (input_dim + 1) * hidden_dim + (hidden_dim + 1) * output_dim
        long expertParams = (long) (inputDim + 1) * hiddenDim + (long) (hiddenDim + 1) * outputDim;
        
        // 总参数 = Router + 专家总参数
        long totalParams = routerParams + (long) numExperts * expertParams;
        
        return totalParams;
    }
    
    /**
     * 计算激活参数数量(稀疏激活)
     */
    public long calculateActiveParameters() {
        // Router参数: input_dim * num_experts
        long routerParams = (long) inputDim * numExperts;
        
        // 单个专家参数
        long expertParams = (long) (inputDim + 1) * hiddenDim + (long) (hiddenDim + 1) * outputDim;
        
        // 激活参数 = Router + Top-K个专家参数
        long activeParams = routerParams + (long) topK * expertParams;
        
        return activeParams;
    }
    
    @Override
    public String toString() {
        return String.format("MoEConfig{\n" +
            "  inputDim=%d, hiddenDim=%d, outputDim=%d\n" +
            "  numExperts=%d, topK=%d, noiseFactor=%.3f\n" +
            "  loadBalance=%s, importanceCoef=%.4f, loadCoef=%.4f\n" +
            "  totalParams=%d, activeParams=%d (%.1f%%)\n" +
            "}",
            inputDim, hiddenDim, outputDim,
            numExperts, topK, noiseFactor,
            enableLoadBalance, importanceCoef, loadCoef,
            calculateParameters(), calculateActiveParameters(),
            100.0 * calculateActiveParameters() / calculateParameters());
    }
    
    /**
     * Builder类
     */
    public static class Builder {
        private final MoEConfig config;
        
        public Builder() {
            this.config = new MoEConfig();
            // 默认值
            config.inputDim = 512;
            config.hiddenDim = 2048;
            config.outputDim = 512;
            config.numExperts = 4;
            config.topK = 2;
            config.noiseFactor = 0.1f;
            config.importanceCoef = 0.01f;
            config.loadCoef = 0.01f;
            config.enableLoadBalance = true;
            config.sharedExperts = false;
            config.dropoutRate = 0.0f;
        }
        
        public Builder inputDim(int inputDim) {
            config.inputDim = inputDim;
            return this;
        }
        
        public Builder hiddenDim(int hiddenDim) {
            config.hiddenDim = hiddenDim;
            return this;
        }
        
        public Builder outputDim(int outputDim) {
            config.outputDim = outputDim;
            return this;
        }
        
        public Builder numExperts(int numExperts) {
            config.numExperts = numExperts;
            return this;
        }
        
        public Builder topK(int topK) {
            config.topK = topK;
            return this;
        }
        
        public Builder noiseFactor(float noiseFactor) {
            config.noiseFactor = noiseFactor;
            return this;
        }
        
        public Builder importanceCoef(float importanceCoef) {
            config.importanceCoef = importanceCoef;
            return this;
        }
        
        public Builder loadCoef(float loadCoef) {
            config.loadCoef = loadCoef;
            return this;
        }
        
        public Builder enableLoadBalance(boolean enableLoadBalance) {
            config.enableLoadBalance = enableLoadBalance;
            return this;
        }
        
        public Builder sharedExperts(boolean sharedExperts) {
            config.sharedExperts = sharedExperts;
            return this;
        }
        
        public Builder dropoutRate(float dropoutRate) {
            config.dropoutRate = dropoutRate;
            return this;
        }
        
        public MoEConfig build() {
            validate();
            return config;
        }
        
        private void validate() {
            if (config.inputDim <= 0) {
                throw new IllegalArgumentException("inputDim must be > 0");
            }
            if (config.hiddenDim <= 0) {
                throw new IllegalArgumentException("hiddenDim must be > 0");
            }
            if (config.outputDim <= 0) {
                throw new IllegalArgumentException("outputDim must be > 0");
            }
            if (config.numExperts < 2) {
                throw new IllegalArgumentException("numExperts must be >= 2");
            }
            if (config.topK < 1 || config.topK > config.numExperts) {
                throw new IllegalArgumentException("topK must be in [1, numExperts]");
            }
            if (config.noiseFactor < 0) {
                throw new IllegalArgumentException("noiseFactor must be >= 0");
            }
        }
    }
}
