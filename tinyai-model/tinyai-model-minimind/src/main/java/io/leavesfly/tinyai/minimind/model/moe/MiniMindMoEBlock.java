package io.leavesfly.tinyai.minimind.model.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.attention.KVCache;
import io.leavesfly.tinyai.minimind.model.embedding.TokenEmbedding;
import io.leavesfly.tinyai.minimind.moe.MoEConfig;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * MiniMind MoE 模型块
 * <p>
 * 集成 MoE (Mixture of Experts) 的 MiniMind 模型主体结构
 * 
 * 架构:
 * 1. Token Embedding 层
 * 2. N 个 MoE Transformer 层 (注意力 + MoE FFN)
 * 3. Final LayerNorm
 * 4. Language Model Head
 * 
 * 相比标准 MiniMindBlock:
 * - 用 MoE 层替换标准 FFN
 * - 支持负载均衡损失计算
 * - 专家使用统计
 * 
 * @author leavesfly
 * @version 1.0
 */
public class MiniMindMoEBlock extends Module {

    /**
     * 模型配置
     */
    private final MiniMindConfig config;

    /**
     * MoE 配置
     */
    private final MoEConfig moeConfig;

    /**
     * Token 嵌入层
     */
    private final TokenEmbedding tokenEmbedding;

    /**
     * MoE Transformer 层列表
     */
    private final List<MiniMindMoETransformerLayer> layers;

    /**
     * 最终归一化层
     */
    private final LayerNorm finalNorm;

    /**
     * 语言模型头 (LM Head)
     */
    private final Linear lmHead;

    /**
     * 是否处于训练模式
     */
    private boolean training = true;

    /**
     * 累积的负载均衡损失
     */
    private float totalBalanceLoss = 0.0f;

    /**
     * 构造 MiniMindMoEBlock
     *
     * @param config 模型配置
     */
    public MiniMindMoEBlock(MiniMindConfig config) {
        super("MiniMindMoEBlock");
        this.config = config;

        // 创建 MoE 配置
        this.moeConfig = createMoEConfig(config);

        // 1. 创建 Token Embedding 层
        this.tokenEmbedding = new TokenEmbedding(config.getVocabSize(), config.getHiddenSize());
        registerModule("token_embedding", tokenEmbedding);

        // 2. 创建 MoE Transformer 层列表
        this.layers = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
            MiniMindMoETransformerLayer layer = new MiniMindMoETransformerLayer(
                "moe_layer_" + i,
                config,
                moeConfig
            );
            layers.add(layer);
            registerModule("moe_layer_" + i, layer);
        }

        // 3. 创建最终归一化层
        this.finalNorm = new LayerNorm("final_norm", config.getHiddenSize(), config.getEpsilon());
        registerModule("final_norm", finalNorm);

        // 4. 创建 LM Head
        this.lmHead = new Linear("lm_head", config.getHiddenSize(), config.getVocabSize(), false);
        registerModule("lm_head", lmHead);

        // 初始化参数
        init();
    }

    /**
     * 从 MiniMindConfig 创建 MoEConfig
     */
    private MoEConfig createMoEConfig(MiniMindConfig config) {
        return MoEConfig.builder()
            .inputDim(config.getHiddenSize())
            .hiddenDim(config.getFfnHiddenSize())
            .outputDim(config.getHiddenSize())
            .numExperts(config.getNumExperts())
            .topK(config.getNumExpertsPerToken())
            .noiseFactor(0.1f)
            .enableLoadBalance(true)
            .importanceCoef(0.01f)
            .loadCoef(0.01f)
            .build();
    }

    /**
     * 前向传播（不使用 KV-Cache）
     *
     * @param inputs 输入 Variable 数组,inputs[0] 为 token IDs
     * @return 输出 Variable,形状 [batch_size, seq_len, vocab_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        Variable tokenIds = inputs[0];
        return forwardWithCache(tokenIds, null, 0).getOutput();
    }

    /**
     * 带 KV-Cache 和负载均衡损失的前向传播
     *
     * @param tokenIds  Token IDs,形状 [batch_size, seq_len]
     * @param kvCaches  KV-Cache 列表（每层一个）,可为 null
     * @param startPos  起始位置（用于 RoPE 和因果掩码）
     * @return MoE 输出结果
     */
    public MoEOutput forwardWithCache(Variable tokenIds, List<KVCache> kvCaches, int startPos) {
        // 重置负载均衡损失
        totalBalanceLoss = 0.0f;

        // 1. Token Embedding: [batch, seq_len] -> [batch, seq_len, hidden_size]
        Variable x = tokenEmbedding.forward(tokenIds);

        // 2. 通过所有 MoE Transformer 层
        for (int i = 0; i < layers.size(); i++) {
            MiniMindMoETransformerLayer layer = layers.get(i);
            KVCache kvCache = (kvCaches != null && i < kvCaches.size()) ? kvCaches.get(i) : null;
            
            MiniMindMoETransformerLayer.LayerOutput layerOutput = 
                layer.forwardWithCache(x, kvCache, startPos);
            
            x = layerOutput.getOutput();
            totalBalanceLoss += layerOutput.getBalanceLoss();
        }

        // 3. 最终归一化
        x = finalNorm.forward(x);

        // 4. LM Head: [batch, seq_len, hidden_size] -> [batch, seq_len, vocab_size]
        Variable logits = lmHead.forward(x);

        return new MoEOutput(logits, totalBalanceLoss);
    }

    /**
     * 生成时的前向传播（使用 KV-Cache 优化）
     *
     * @param tokenId  当前 token ID,形状 [batch_size, 1]
     * @param kvCaches KV-Cache 列表（每层一个）
     * @param position 当前位置
     * @return MoE 输出结果
     */
    public MoEOutput forwardGeneration(Variable tokenId, List<KVCache> kvCaches, int position) {
        return forwardWithCache(tokenId, kvCaches, position);
    }

    /**
     * 创建 KV-Cache 列表
     *
     * @param batchSize 批次大小
     * @return KV-Cache 列表
     */
    public List<KVCache> createKVCaches(int batchSize) {
        List<KVCache> kvCaches = new ArrayList<>();
        for (int i = 0; i < config.getNumLayers(); i++) {
            KVCache cache = new KVCache(
                batchSize,
                config.getNumHeads(),
                config.getHiddenSize() / config.getNumHeads(),
                config.getMaxSeqLen()
            );
            kvCaches.add(cache);
        }
        return kvCaches;
    }

    /**
     * 清空所有 KV-Cache
     *
     * @param kvCaches KV-Cache 列表
     */
    public void clearKVCaches(List<KVCache> kvCaches) {
        if (kvCaches != null) {
            for (KVCache cache : kvCaches) {
                cache.clear();
            }
        }
    }

    /**
     * 设置训练模式
     *
     * @param training 是否为训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
        for (MiniMindMoETransformerLayer layer : layers) {
            layer.setTraining(training);
        }
    }

    /**
     * 是否为训练模式
     */
    public boolean isTraining() {
        return training;
    }

    /**
     * 获取配置
     */
    public MiniMindConfig getConfig() {
        return config;
    }

    /**
     * 获取 MoE 配置
     */
    public MoEConfig getMoEConfig() {
        return moeConfig;
    }

    /**
     * 获取所有 MoE 层
     */
    public List<MiniMindMoETransformerLayer> getLayers() {
        return layers;
    }

    /**
     * 获取总的负载均衡损失
     */
    public float getTotalBalanceLoss() {
        return totalBalanceLoss;
    }

    /**
     * 获取专家使用统计
     */
    public String getExpertUsageStats() {
        StringBuilder sb = new StringBuilder();
        sb.append("Expert Usage Statistics:\n");
        for (int i = 0; i < layers.size(); i++) {
            sb.append(String.format("Layer %d: %s\n", 
                i, layers.get(i).getUsageStats()));
        }
        return sb.toString();
    }

    /**
     * 重置所有层的统计信息
     */
    public void resetStats() {
        for (MiniMindMoETransformerLayer layer : layers) {
            layer.resetStats();
        }
        totalBalanceLoss = 0.0f;
    }

    /**
     * 获取模型信息
     */
    public String getModelInfo() {
        long params = estimateParameters();
        return String.format(
            "MiniMindMoEBlock:\n" +
            "  - Vocab Size: %d\n" +
            "  - Hidden Size: %d\n" +
            "  - Num Layers: %d\n" +
            "  - Num Heads: %d\n" +
            "  - FFN Hidden: %d\n" +
            "  - Num Experts: %d\n" +
            "  - Experts Per Token: %d\n" +
            "  - Max Seq Len: %d\n" +
            "  - Parameters: ~%dM\n" +
            "  - Mode: %s",
            config.getVocabSize(),
            config.getHiddenSize(),
            config.getNumLayers(),
            config.getNumHeads(),
            config.getFfnHiddenSize(),
            config.getNumExperts(),
            config.getNumExpertsPerToken(),
            config.getMaxSeqLen(),
            params / 1_000_000,
            training ? "Training" : "Eval"
        );
    }

    /**
     * 估算参数量
     */
    private long estimateParameters() {
        long params = 0;
        
        // Token Embedding
        params += (long) config.getVocabSize() * config.getHiddenSize();
        
        // 每层参数 (MoE 版本)
        int hiddenSize = config.getHiddenSize();
        int numHeads = config.getNumHeads();
        int ffnHidden = config.getFfnHiddenSize();
        int numExperts = config.getNumExperts();
        
        for (int i = 0; i < config.getNumLayers(); i++) {
            // Attention (QKV + Output)
            params += (long) hiddenSize * hiddenSize * 4;
            
            // MoE FFN (每个专家)
            long expertParams = (long) hiddenSize * ffnHidden + (long) ffnHidden * hiddenSize;
            params += expertParams * numExperts;
            
            // Router
            params += (long) hiddenSize * numExperts;
            
            // LayerNorm (2个)
            params += (long) hiddenSize * 2;
        }
        
        // Final LayerNorm
        params += config.getHiddenSize();
        
        // LM Head
        params += (long) config.getHiddenSize() * config.getVocabSize();
        
        return params;
    }

    @Override
    public String toString() {
        return getModelInfo();
    }

    /**
     * MoE 输出结果
     */
    public static class MoEOutput {
        private final Variable output;
        private final float balanceLoss;

        public MoEOutput(Variable output, float balanceLoss) {
            this.output = output;
            this.balanceLoss = balanceLoss;
        }

        public Variable getOutput() {
            return output;
        }

        public float getBalanceLoss() {
            return balanceLoss;
        }

        @Override
        public String toString() {
            return String.format("MoEOutput(shape=%s, balance_loss=%.6f)",
                output.getShape(), balanceLoss);
        }
    }
}
