package io.leavesfly.tinyai.deepseek.v3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * DeepSeek-V3代码生成专门模块
 * 
 * V3的核心优势之一是对代码生成任务的深度优化。
 * 
 * 核心功能：
 * 1. 编程语言识别（10种主流语言）
 * 2. 代码结构分析
 * 3. 代码质量评估（4个维度）
 * 
 * 支持的编程语言：
 * Java, Python, JavaScript, C++, C, Go, Rust, TypeScript, Kotlin, Swift
 * 
 * @author leavesfly
 * @version 1.0
 */
public class DeepSeekV3CodeBlock extends Module {
    
    private final DeepSeekV3Config config;
    
    /** 支持的编程语言列表 */
    public static final String[] SUPPORTED_LANGUAGES = {
        "Java", "Python", "JavaScript", "C++", "C",
        "Go", "Rust", "TypeScript", "Kotlin", "Swift"
    };
    
    // 语言识别器
    private Linear languageClassifier;
    
    // 代码分析器
    private Linear codeAnalyzer;
    
    // 质量评估器
    private Linear qualityEvaluator;
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config V3配置对象
     */
    public DeepSeekV3CodeBlock(String name, DeepSeekV3Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    /**
     * 初始化组件
     */
    private void initializeComponents() {
        int nEmbd = config.getNEmbd();
        int codeAnalysisHiddenDim = config.getCodeAnalysisHiddenDim();
        
        // 语言识别器: nEmbd -> numProgrammingLanguages
        languageClassifier = new Linear(
            name + "_lang_classifier",
            nEmbd,
            config.getNumProgrammingLanguages(),
            true
        );
        registerModule("lang_classifier", languageClassifier);
        
        // 代码分析器: nEmbd -> codeAnalysisHiddenDim
        codeAnalyzer = new Linear(
            name + "_code_analyzer",
            nEmbd,
            codeAnalysisHiddenDim,
            true
        );
        registerModule("code_analyzer", codeAnalyzer);
        
        // 质量评估器: codeAnalysisHiddenDim -> codeQualityDim
        qualityEvaluator = new Linear(
            name + "_quality_evaluator",
            codeAnalysisHiddenDim,
            config.getCodeQualityDim(),
            true
        );
        registerModule("quality_evaluator", qualityEvaluator);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入张量 [batch_size, seq_len, nEmbd]
     * @return 代码分析输出 [batch_size, seq_len, nEmbd]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable input = inputs[0];
        
        // 简单地返回输入（代码分析主要通过analyzeCode方法）
        return input;
    }
    
    /**
     * 分析代码
     * 
     * @param input 输入张量 [batch_size, seq_len, nEmbd]
     * @return 代码分析结果
     */
    public CodeAnalysisResult analyzeCode(Variable input) {
        // 1. 语言识别
        String detectedLanguage = detectLanguage(input);
        
        // 2. 代码分析
        Variable codeFeatures = codeAnalyzer.forward(input);
        
        // 3. 质量评估
        CodeQualityScore qualityScore = evaluateQuality(codeFeatures);
        
        return new CodeAnalysisResult(detectedLanguage, qualityScore);
    }
    
    /**
     * 检测编程语言
     */
    private String detectLanguage(Variable input) {
        Variable langLogits = languageClassifier.forward(input);
        NdArray logitsArray = langLogits.getValue();
        
        int batchSize = logitsArray.getShape().getDimension(0);
        int seqLen = logitsArray.getShape().getDimension(1);
        int numLangs = logitsArray.getShape().getDimension(2);
        
        // 统计每种语言的得分
        float[] langScores = new float[numLangs];
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int lang = 0; lang < numLangs; lang++) {
                    langScores[lang] += logitsArray.get(b, t, lang);
                }
            }
        }
        
        // 选择得分最高的语言
        int maxLangId = 0;
        float maxScore = langScores[0];
        for (int i = 1; i < numLangs; i++) {
            if (langScores[i] > maxScore) {
                maxScore = langScores[i];
                maxLangId = i;
            }
        }
        
        return maxLangId < SUPPORTED_LANGUAGES.length ? 
               SUPPORTED_LANGUAGES[maxLangId] : "Unknown";
    }
    
    /**
     * 评估代码质量
     */
    private CodeQualityScore evaluateQuality(Variable codeFeatures) {
        Variable qualityLogits = qualityEvaluator.forward(codeFeatures);
        NdArray logitsArray = qualityLogits.getValue();
        
        int batchSize = logitsArray.getShape().getDimension(0);
        int seqLen = logitsArray.getShape().getDimension(1);
        int qualityDim = logitsArray.getShape().getDimension(2);
        
        // 对每个质量维度计算平均分
        float[] qualityScores = new float[qualityDim];
        int totalTokens = batchSize * seqLen;
        
        for (int b = 0; b < batchSize; b++) {
            for (int t = 0; t < seqLen; t++) {
                for (int q = 0; q < qualityDim; q++) {
                    float logit = logitsArray.get(b, t, q);
                    // Sigmoid激活转换为[0,1]分数
                    float score = (float) (1.0 / (1.0 + Math.exp(-logit)));
                    qualityScores[q] += score;
                }
            }
        }
        
        for (int q = 0; q < qualityDim; q++) {
            qualityScores[q] /= totalTokens;
        }
        
        // 质量维度：语法、结构、可读性、性能
        return new CodeQualityScore(
            qualityScores.length > 0 ? qualityScores[0] : 0.0f,  // 语法正确性
            qualityScores.length > 1 ? qualityScores[1] : 0.0f,  // 代码结构
            qualityScores.length > 2 ? qualityScores[2] : 0.0f,  // 可读性
            qualityScores.length > 3 ? qualityScores[3] : 0.0f   // 性能
        );
    }
    
    /**
     * 代码分析结果类
     */
    public static class CodeAnalysisResult {
        /** 检测到的编程语言 */
        public final String detectedLanguage;
        /** 代码质量评分 */
        public final CodeQualityScore qualityScore;
        
        public CodeAnalysisResult(String detectedLanguage, CodeQualityScore qualityScore) {
            this.detectedLanguage = detectedLanguage;
            this.qualityScore = qualityScore;
        }
        
        @Override
        public String toString() {
            return String.format(
                "CodeAnalysisResult{language='%s', %s}",
                detectedLanguage,
                qualityScore
            );
        }
    }
    
    /**
     * 代码质量评分类
     */
    public static class CodeQualityScore {
        /** 语法正确性得分 [0,1] */
        public final float syntaxScore;
        /** 代码结构得分 [0,1] */
        public final float structureScore;
        /** 可读性得分 [0,1] */
        public final float readabilityScore;
        /** 性能得分 [0,1] */
        public final float performanceScore;
        
        public CodeQualityScore(float syntaxScore, float structureScore,
                               float readabilityScore, float performanceScore) {
            this.syntaxScore = syntaxScore;
            this.structureScore = structureScore;
            this.readabilityScore = readabilityScore;
            this.performanceScore = performanceScore;
        }
        
        /**
         * 获取总体质量得分
         */
        public float getOverallScore() {
            return (syntaxScore + structureScore + readabilityScore + performanceScore) / 4.0f;
        }
        
        @Override
        public String toString() {
            return String.format(
                "CodeQualityScore{语法=%.2f, 结构=%.2f, 可读性=%.2f, 性能=%.2f, 总分=%.2f}",
                syntaxScore, structureScore, readabilityScore, 
                performanceScore, getOverallScore()
            );
        }
    }
}
