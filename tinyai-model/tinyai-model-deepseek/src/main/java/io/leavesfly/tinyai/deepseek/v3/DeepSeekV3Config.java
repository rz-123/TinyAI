package io.leavesfly.tinyai.deepseek.v3;

/**
 * DeepSeek-V3模型配置类
 * 
 * DeepSeek-V3是一个基于混合专家模型(MoE)的大语言模型，
 * 通过任务感知的专家路由实现高效的多任务处理和代码生成优化。
 * 
 * 核心特点：
 * 1. 混合专家模型(MoE) - 8个专家网络，Top-2路由选择
 * 2. 任务感知路由 - 支持推理、代码、数学、通用、多模态5种任务类型
 * 3. 代码生成优化 - 支持10种主流编程语言的识别和质量评估
 * 4. Pre-LayerNorm架构 - 提升训练稳定性
 * 5. 参数高效 - 每次仅激活约25%的参数
 * 
 * 本实现完全基于TinyAI框架的V2 API，不依赖任何V1组件。
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3Config {
    
    // ==================== 基础模型配置 ====================
    
    /** 词汇表大小，默认50257 */
    private int vocabSize = 50257;
    
    /** 最大位置数（序列长度），默认2048 */
    private int nPositions = 2048;
    
    /** 嵌入维度，默认768 */
    private int nEmbd = 768;
    
    /** Transformer层数，默认12 */
    private int nLayer = 12;
    
    /** 注意力头数，默认12 */
    private int nHead = 12;
    
    /** 前馈网络中间层维度，默认4倍嵌入维度 */
    private int nInner = 3072;
    
    /** 激活函数类型，默认"gelu" */
    private String activationFunction = "gelu";
    
    // ==================== MoE配置 ====================
    
    /** 专家数量，默认8个专家 */
    private int numExperts = 8;
    
    /** Top-K专家选择数量，默认选择2个专家 */
    private int topK = 2;
    
    /** 每个专家的隐藏层维度，默认与nInner相同 */
    private int expertHiddenDim = 3072;
    
    /** 负载均衡损失权重，默认0.01 */
    private double loadBalanceLossWeight = 0.01;
    
    /** 专家dropout概率，默认0.1 */
    private double expertDropout = 0.1;
    
    // ==================== 任务感知配置 ====================
    
    /** 是否启用任务感知路由，默认启用 */
    private boolean enableTaskAwareRouting = true;
    
    /** 任务类型嵌入维度，默认128 */
    private int taskEmbedDim = 128;
    
    /** 任务识别器隐藏层维度，默认256 */
    private int taskClassifierHiddenDim = 256;
    
    /** 支持的任务类型数量，默认5种：推理、代码、数学、通用、多模态 */
    private int numTaskTypes = 5;
    
    // ==================== 推理配置 ====================
    
    /** 推理隐藏层维度，默认为嵌入维度的2倍 */
    private int reasoningHiddenDim = 1536;
    
    /** 推理置信度阈值，默认0.75（V3比R1更严格） */
    private double confidenceThreshold = 0.75;
    
    /** 是否启用自我纠错机制，默认启用 */
    private boolean enableSelfCorrection = true;
    
    // ==================== 代码生成配置 ====================
    
    /** 代码质量评估维度数量，默认4个维度（语法、结构、可读性、性能） */
    private int codeQualityDim = 4;
    
    /** 支持的编程语言数量，默认10种 */
    private int numProgrammingLanguages = 10;
    
    /** 代码分析隐藏层维度，默认512 */
    private int codeAnalysisHiddenDim = 512;
    
    /** 语法验证器隐藏层维度，默认256 */
    private int syntaxValidatorHiddenDim = 256;
    
    // ==================== Dropout配置 ====================
    
    /** 残差dropout概率，默认0.1 */
    private double residPdrop = 0.1;
    
    /** 嵌入dropout概率，默认0.1 */
    private double embdPdrop = 0.1;
    
    /** 注意力dropout概率，默认0.1 */
    private double attnPdrop = 0.1;
    
    // ==================== 初始化配置 ====================
    
    /** 层归一化epsilon，默认1e-5 */
    private double layerNormEpsilon = 1e-5;
    
    /** 权重初始化范围，默认0.02 */
    private double initializerRange = 0.02;
    
    /**
     * 默认构造函数，创建标准DeepSeek-V3配置
     */
    public DeepSeekV3Config() {
        // 使用默认值
    }
    
    /**
     * 完整配置构造函数
     */
    public DeepSeekV3Config(int vocabSize, int nPositions, int nEmbd, int nLayer,
                           int nHead, int nInner, String activationFunction,
                           int numExperts, int topK, int expertHiddenDim,
                           double loadBalanceLossWeight, double expertDropout,
                           boolean enableTaskAwareRouting, int taskEmbedDim,
                           int taskClassifierHiddenDim, int numTaskTypes,
                           int reasoningHiddenDim, double confidenceThreshold,
                           boolean enableSelfCorrection, int codeQualityDim,
                           int numProgrammingLanguages, int codeAnalysisHiddenDim,
                           int syntaxValidatorHiddenDim,
                           double residPdrop, double embdPdrop, double attnPdrop,
                           double layerNormEpsilon, double initializerRange) {
        this.vocabSize = vocabSize;
        this.nPositions = nPositions;
        this.nEmbd = nEmbd;
        this.nLayer = nLayer;
        this.nHead = nHead;
        this.nInner = nInner;
        this.activationFunction = activationFunction;
        this.numExperts = numExperts;
        this.topK = topK;
        this.expertHiddenDim = expertHiddenDim;
        this.loadBalanceLossWeight = loadBalanceLossWeight;
        this.expertDropout = expertDropout;
        this.enableTaskAwareRouting = enableTaskAwareRouting;
        this.taskEmbedDim = taskEmbedDim;
        this.taskClassifierHiddenDim = taskClassifierHiddenDim;
        this.numTaskTypes = numTaskTypes;
        this.reasoningHiddenDim = reasoningHiddenDim;
        this.confidenceThreshold = confidenceThreshold;
        this.enableSelfCorrection = enableSelfCorrection;
        this.codeQualityDim = codeQualityDim;
        this.numProgrammingLanguages = numProgrammingLanguages;
        this.codeAnalysisHiddenDim = codeAnalysisHiddenDim;
        this.syntaxValidatorHiddenDim = syntaxValidatorHiddenDim;
        this.residPdrop = residPdrop;
        this.embdPdrop = embdPdrop;
        this.attnPdrop = attnPdrop;
        this.layerNormEpsilon = layerNormEpsilon;
        this.initializerRange = initializerRange;
    }
    
    // ==================== 预设配置工厂方法 ====================
    
    /**
     * 创建标准DeepSeek-V3配置
     * 配置：768维, 12层, 12头, 8专家, Top-2路由, 序列长度2048
     */
    public static DeepSeekV3Config createStandardConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setVocabSize(50257);
        config.setNEmbd(768);
        config.setNLayer(12);
        config.setNHead(12);
        config.setNInner(3072);
        config.setNPositions(2048);
        config.setNumExperts(8);
        config.setTopK(2);
        config.setExpertHiddenDim(3072);
        config.setReasoningHiddenDim(1536);
        config.setCodeAnalysisHiddenDim(512);
        return config;
    }
    
    /**
     * 创建微型DeepSeek-V3配置（用于快速测试）
     * 配置：256维, 6层, 8头, 4专家, Top-2路由, 序列长度512
     */
    public static DeepSeekV3Config createTinyConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setVocabSize(10000);
        config.setNEmbd(256);
        config.setNLayer(6);
        config.setNHead(8);
        config.setNInner(1024);
        config.setNPositions(512);
        config.setNumExperts(4);
        config.setTopK(2);
        config.setExpertHiddenDim(1024);
        config.setReasoningHiddenDim(512);
        config.setCodeAnalysisHiddenDim(256);
        return config;
    }
    
    /**
     * 创建小型DeepSeek-V3配置（用于学习和实验）
     * 配置：512维, 8层, 8头, 6专家, Top-2路由, 序列长度1024
     */
    public static DeepSeekV3Config createSmallConfig() {
        DeepSeekV3Config config = new DeepSeekV3Config();
        config.setVocabSize(30000);
        config.setNEmbd(512);
        config.setNLayer(8);
        config.setNHead(8);
        config.setNInner(2048);
        config.setNPositions(1024);
        config.setNumExperts(6);
        config.setTopK(2);
        config.setExpertHiddenDim(2048);
        config.setReasoningHiddenDim(1024);
        config.setCodeAnalysisHiddenDim(384);
        return config;
    }
    
    // ==================== 验证方法 ====================
    
    /**
     * 验证配置的有效性
     * 
     * @throws IllegalArgumentException 如果配置无效
     */
    public void validate() {
        if (vocabSize <= 0) {
            throw new IllegalArgumentException("词汇表大小必须大于0，实际: " + vocabSize);
        }
        if (nPositions <= 0) {
            throw new IllegalArgumentException("最大位置数必须大于0，实际: " + nPositions);
        }
        if (nEmbd <= 0) {
            throw new IllegalArgumentException("嵌入维度必须大于0，实际: " + nEmbd);
        }
        if (nLayer <= 0) {
            throw new IllegalArgumentException("层数必须大于0，实际: " + nLayer);
        }
        if (nHead <= 0) {
            throw new IllegalArgumentException("注意力头数必须大于0，实际: " + nHead);
        }
        if (nEmbd % nHead != 0) {
            throw new IllegalArgumentException(
                String.format("嵌入维度(%d)必须能被注意力头数(%d)整除", nEmbd, nHead)
            );
        }
        if (nInner <= 0) {
            throw new IllegalArgumentException("前馈网络维度必须大于0，实际: " + nInner);
        }
        if (numExperts <= 0) {
            throw new IllegalArgumentException("专家数量必须大于0，实际: " + numExperts);
        }
        if (topK <= 0 || topK > numExperts) {
            throw new IllegalArgumentException(
                String.format("Top-K值(%d)必须在[1, %d]范围内", topK, numExperts)
            );
        }
        if (confidenceThreshold < 0 || confidenceThreshold > 1) {
            throw new IllegalArgumentException("置信度阈值必须在[0,1]范围内，实际: " + confidenceThreshold);
        }
        if (loadBalanceLossWeight < 0) {
            throw new IllegalArgumentException("负载均衡损失权重必须非负，实际: " + loadBalanceLossWeight);
        }
        if (residPdrop < 0 || residPdrop >= 1) {
            throw new IllegalArgumentException("残差dropout概率必须在[0,1)范围内，实际: " + residPdrop);
        }
        if (embdPdrop < 0 || embdPdrop >= 1) {
            throw new IllegalArgumentException("嵌入dropout概率必须在[0,1)范围内，实际: " + embdPdrop);
        }
        if (attnPdrop < 0 || attnPdrop >= 1) {
            throw new IllegalArgumentException("注意力dropout概率必须在[0,1)范围内，实际: " + attnPdrop);
        }
    }
    
    /**
     * 估算模型参数数量
     * 
     * @return 估算的参数数量
     */
    public long estimateParameterCount() {
        // Token嵌入: vocabSize * nEmbd
        long tokenEmbed = (long) vocabSize * nEmbd;
        
        // 位置嵌入: nPositions * nEmbd
        long posEmbed = (long) nPositions * nEmbd;
        
        // 每个Transformer块的参数
        // 注意力: QKV投影(3 * nEmbd * nEmbd) + 输出投影(nEmbd * nEmbd) + 偏置(4 * nEmbd)
        long attnParams = 4L * nEmbd * nEmbd + 4L * nEmbd;
        
        // LayerNorm1: gamma(nEmbd) + beta(nEmbd)
        long ln1Params = 2L * nEmbd;
        
        // MoE层参数
        // 门控网络: nEmbd * numExperts
        long gatingParams = (long) nEmbd * numExperts;
        
        // 每个专家的FFN: fc1(nEmbd * expertHiddenDim + expertHiddenDim) + fc2(expertHiddenDim * nEmbd + nEmbd)
        long paramsPerExpert = (long) nEmbd * expertHiddenDim + expertHiddenDim +
                                (long) expertHiddenDim * nEmbd + nEmbd;
        
        // 所有专家的参数
        long allExpertsParams = paramsPerExpert * numExperts;
        
        // MoE总参数
        long moeParams = gatingParams + allExpertsParams;
        
        // LayerNorm2: gamma(nEmbd) + beta(nEmbd)
        long ln2Params = 2L * nEmbd;
        
        // 每层总参数
        long paramsPerLayer = attnParams + ln1Params + moeParams + ln2Params;
        
        // 所有Transformer层的参数
        long allLayersParams = paramsPerLayer * nLayer;
        
        // 任务感知路由参数
        long taskRoutingParams = 0;
        if (enableTaskAwareRouting) {
            // 任务分类器: nEmbd * taskClassifierHiddenDim + taskClassifierHiddenDim +
            //             taskClassifierHiddenDim * numTaskTypes + numTaskTypes
            taskRoutingParams = (long) nEmbd * taskClassifierHiddenDim + taskClassifierHiddenDim +
                                (long) taskClassifierHiddenDim * numTaskTypes + numTaskTypes;
        }
        
        // 推理模块参数
        // 推理投影: nEmbd * reasoningHiddenDim + reasoningHiddenDim
        // 推理输出: reasoningHiddenDim * nEmbd + nEmbd
        // 置信度评估: reasoningHiddenDim * 1 + 1
        long reasoningParams = (long) nEmbd * reasoningHiddenDim + reasoningHiddenDim +
                               (long) reasoningHiddenDim * nEmbd + nEmbd +
                               reasoningHiddenDim + 1;
        
        // 代码生成模块参数
        // 语言识别: nEmbd * numProgrammingLanguages + numProgrammingLanguages
        // 代码分析: nEmbd * codeAnalysisHiddenDim + codeAnalysisHiddenDim
        // 质量评估: codeAnalysisHiddenDim * codeQualityDim + codeQualityDim
        long codeParams = (long) nEmbd * numProgrammingLanguages + numProgrammingLanguages +
                          (long) nEmbd * codeAnalysisHiddenDim + codeAnalysisHiddenDim +
                          (long) codeAnalysisHiddenDim * codeQualityDim + codeQualityDim;
        
        // 最终LayerNorm: gamma(nEmbd) + beta(nEmbd)
        long finalLnParams = 2L * nEmbd;
        
        // 输出投影: nEmbd * vocabSize（通常与token嵌入权重共享）
        // 这里不重复计算
        
        // 总参数 = 嵌入 + 所有Transformer层 + 任务路由 + 推理模块 + 代码模块 + 最终LN
        return tokenEmbed + posEmbed + allLayersParams + taskRoutingParams + 
               reasoningParams + codeParams + finalLnParams;
    }
    
    /**
     * 计算激活参数数量（仅激活Top-K个专家）
     * 
     * @return 激活的参数数量
     */
    public long estimateActiveParameterCount() {
        // 基础参数（非MoE部分）
        long baseParams = estimateParameterCount() - 
            (long) nLayer * numExperts * 
            ((long) nEmbd * expertHiddenDim + expertHiddenDim + 
             (long) expertHiddenDim * nEmbd + nEmbd);
        
        // 激活的专家参数（每层激活topK个专家）
        long activeExpertParams = (long) nLayer * topK * 
            ((long) nEmbd * expertHiddenDim + expertHiddenDim + 
             (long) expertHiddenDim * nEmbd + nEmbd);
        
        return baseParams + activeExpertParams;
    }
    
    /**
     * 计算参数激活率
     * 
     * @return 激活率（百分比）
     */
    public double getActivationRatio() {
        return (double) estimateActiveParameterCount() / estimateParameterCount() * 100.0;
    }
    
    // ==================== Getter和Setter方法 ====================
    
    public int getVocabSize() {
        return vocabSize;
    }
    
    public void setVocabSize(int vocabSize) {
        this.vocabSize = vocabSize;
    }
    
    public int getNPositions() {
        return nPositions;
    }
    
    public void setNPositions(int nPositions) {
        this.nPositions = nPositions;
    }
    
    public int getNEmbd() {
        return nEmbd;
    }
    
    public void setNEmbd(int nEmbd) {
        this.nEmbd = nEmbd;
    }
    
    public int getNLayer() {
        return nLayer;
    }
    
    public void setNLayer(int nLayer) {
        this.nLayer = nLayer;
    }
    
    public int getNHead() {
        return nHead;
    }
    
    public void setNHead(int nHead) {
        this.nHead = nHead;
    }
    
    public int getNInner() {
        return nInner;
    }
    
    public void setNInner(int nInner) {
        this.nInner = nInner;
    }
    
    public String getActivationFunction() {
        return activationFunction;
    }
    
    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }
    
    public int getNumExperts() {
        return numExperts;
    }
    
    public void setNumExperts(int numExperts) {
        this.numExperts = numExperts;
    }
    
    public int getTopK() {
        return topK;
    }
    
    public void setTopK(int topK) {
        this.topK = topK;
    }
    
    public int getExpertHiddenDim() {
        return expertHiddenDim;
    }
    
    public void setExpertHiddenDim(int expertHiddenDim) {
        this.expertHiddenDim = expertHiddenDim;
    }
    
    public double getLoadBalanceLossWeight() {
        return loadBalanceLossWeight;
    }
    
    public void setLoadBalanceLossWeight(double loadBalanceLossWeight) {
        this.loadBalanceLossWeight = loadBalanceLossWeight;
    }
    
    public double getExpertDropout() {
        return expertDropout;
    }
    
    public void setExpertDropout(double expertDropout) {
        this.expertDropout = expertDropout;
    }
    
    public boolean isEnableTaskAwareRouting() {
        return enableTaskAwareRouting;
    }
    
    public void setEnableTaskAwareRouting(boolean enableTaskAwareRouting) {
        this.enableTaskAwareRouting = enableTaskAwareRouting;
    }
    
    public int getTaskEmbedDim() {
        return taskEmbedDim;
    }
    
    public void setTaskEmbedDim(int taskEmbedDim) {
        this.taskEmbedDim = taskEmbedDim;
    }
    
    public int getTaskClassifierHiddenDim() {
        return taskClassifierHiddenDim;
    }
    
    public void setTaskClassifierHiddenDim(int taskClassifierHiddenDim) {
        this.taskClassifierHiddenDim = taskClassifierHiddenDim;
    }
    
    public int getNumTaskTypes() {
        return numTaskTypes;
    }
    
    public void setNumTaskTypes(int numTaskTypes) {
        this.numTaskTypes = numTaskTypes;
    }
    
    public int getReasoningHiddenDim() {
        return reasoningHiddenDim;
    }
    
    public void setReasoningHiddenDim(int reasoningHiddenDim) {
        this.reasoningHiddenDim = reasoningHiddenDim;
    }
    
    public double getConfidenceThreshold() {
        return confidenceThreshold;
    }
    
    public void setConfidenceThreshold(double confidenceThreshold) {
        this.confidenceThreshold = confidenceThreshold;
    }
    
    public boolean isEnableSelfCorrection() {
        return enableSelfCorrection;
    }
    
    public void setEnableSelfCorrection(boolean enableSelfCorrection) {
        this.enableSelfCorrection = enableSelfCorrection;
    }
    
    public int getCodeQualityDim() {
        return codeQualityDim;
    }
    
    public void setCodeQualityDim(int codeQualityDim) {
        this.codeQualityDim = codeQualityDim;
    }
    
    public int getNumProgrammingLanguages() {
        return numProgrammingLanguages;
    }
    
    public void setNumProgrammingLanguages(int numProgrammingLanguages) {
        this.numProgrammingLanguages = numProgrammingLanguages;
    }
    
    public int getCodeAnalysisHiddenDim() {
        return codeAnalysisHiddenDim;
    }
    
    public void setCodeAnalysisHiddenDim(int codeAnalysisHiddenDim) {
        this.codeAnalysisHiddenDim = codeAnalysisHiddenDim;
    }
    
    public int getSyntaxValidatorHiddenDim() {
        return syntaxValidatorHiddenDim;
    }
    
    public void setSyntaxValidatorHiddenDim(int syntaxValidatorHiddenDim) {
        this.syntaxValidatorHiddenDim = syntaxValidatorHiddenDim;
    }
    
    public double getResidPdrop() {
        return residPdrop;
    }
    
    public void setResidPdrop(double residPdrop) {
        this.residPdrop = residPdrop;
    }
    
    public double getEmbdPdrop() {
        return embdPdrop;
    }
    
    public void setEmbdPdrop(double embdPdrop) {
        this.embdPdrop = embdPdrop;
    }
    
    public double getAttnPdrop() {
        return attnPdrop;
    }
    
    public void setAttnPdrop(double attnPdrop) {
        this.attnPdrop = attnPdrop;
    }
    
    public double getLayerNormEpsilon() {
        return layerNormEpsilon;
    }
    
    public void setLayerNormEpsilon(double layerNormEpsilon) {
        this.layerNormEpsilon = layerNormEpsilon;
    }
    
    public double getInitializerRange() {
        return initializerRange;
    }
    
    public void setInitializerRange(double initializerRange) {
        this.initializerRange = initializerRange;
    }
    
    @Override
    public String toString() {
        return String.format(
            "DeepSeekV3Config{\n" +
            "  vocabSize=%d,\n" +
            "  nPositions=%d,\n" +
            "  nEmbd=%d,\n" +
            "  nLayer=%d,\n" +
            "  nHead=%d,\n" +
            "  nInner=%d,\n" +
            "  activationFunction='%s',\n" +
            "  numExperts=%d,\n" +
            "  topK=%d,\n" +
            "  expertHiddenDim=%d,\n" +
            "  loadBalanceLossWeight=%.4f,\n" +
            "  enableTaskAwareRouting=%b,\n" +
            "  numTaskTypes=%d,\n" +
            "  reasoningHiddenDim=%d,\n" +
            "  confidenceThreshold=%.2f,\n" +
            "  enableSelfCorrection=%b,\n" +
            "  codeQualityDim=%d,\n" +
            "  numProgrammingLanguages=%d,\n" +
            "  residPdrop=%.3f,\n" +
            "  embdPdrop=%.3f,\n" +
            "  attnPdrop=%.3f,\n" +
            "  layerNormEpsilon=%.1e,\n" +
            "  initializerRange=%.3f,\n" +
            "  estimatedParams=%s,\n" +
            "  activeParams=%s,\n" +
            "  activationRatio=%.2f%%\n" +
            "}",
            vocabSize, nPositions, nEmbd, nLayer, nHead, nInner,
            activationFunction, numExperts, topK, expertHiddenDim,
            loadBalanceLossWeight, enableTaskAwareRouting, numTaskTypes,
            reasoningHiddenDim, confidenceThreshold, enableSelfCorrection,
            codeQualityDim, numProgrammingLanguages,
            residPdrop, embdPdrop, attnPdrop,
            layerNormEpsilon, initializerRange,
            formatParamCount(estimateParameterCount()),
            formatParamCount(estimateActiveParameterCount()),
            getActivationRatio()
        );
    }
    
    /**
     * 格式化参数数量显示
     */
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2fB", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2fM", count / 1_000_000.0);
        } else if (count >= 1_000) {
            return String.format("%.2fK", count / 1_000.0);
        } else {
            return String.format("%d", count);
        }
    }
}
