package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.embedding.Embedding;

import java.util.ArrayList;
import java.util.List;

/**
 * Qwen3主体块
 * 
 * 整合所有Qwen3组件，构建完整的模型架构：
 * 1. Token嵌入层
 * 2. Transformer解码器层堆叠
 * 3. 最终RMSNorm
 * 4. 输出投影层（语言模型头）
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Block extends Module {
    
    private final Qwen3Config config;
    private final boolean includeLMHead;
    
    private Embedding embedTokens;
    private List<Qwen3TransformerBlock> layers;
    private RMSNormLayer norm;
    private Linear lmHead;  // 可选的语言模型头
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3配置
     * @param includeLMHead 是否包含语言模型头
     */
    public Qwen3Block(String name, Qwen3Config config, boolean includeLMHead) {
        super(name);
        this.config = config;
        this.includeLMHead = includeLMHead;
        initializeComponents();
    }
    
    /**
     * 初始化所有组件
     */
    private void initializeComponents() {
        // 1. Token嵌入层
        embedTokens = new Embedding(
            name + "_embed_tokens",
            config.getVocabSize(),
            config.getHiddenSize()
        );
        registerModule("embed_tokens", embedTokens);
        
        // 2. Transformer解码器层堆叠
        layers = new ArrayList<>();
        for (int i = 0; i < config.getNumHiddenLayers(); i++) {
            Qwen3TransformerBlock layer = new Qwen3TransformerBlock(
                name + "_layer_" + i,
                config
            );
            layers.add(layer);
            registerModule("layer_" + i, layer);
        }
        
        // 3. 最终归一化层
        norm = new RMSNormLayer(
            name + "_norm",
            config.getHiddenSize(),
            config.getRmsNormEps()
        );
        registerModule("norm", norm);
        
        // 4. 语言模型头（可选）
        if (includeLMHead) {
            lmHead = new Linear(
                name + "_lm_head",
                config.getHiddenSize(),
                config.getVocabSize(),
                false  // 不使用偏置
            );
            registerModule("lm_head", lmHead);
        }
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为token ID序列 [batch_size, seq_len]
     * @return 如果includeLMHead为true，返回logits [batch_size, seq_len, vocab_size]
     *         否则返回隐藏状态 [batch_size, seq_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable inputIds = inputs[0];
        validateInput(inputIds);
        
        // 1. Token嵌入
        Variable hiddenStates = embedTokens.forward(inputIds);
        
        // 2. 通过所有Transformer层
        for (Qwen3TransformerBlock layer : layers) {
            hiddenStates = layer.forward(hiddenStates);
        }
        
        // 3. 最终归一化
        hiddenStates = norm.forward(hiddenStates);
        
        // 4. 输出投影（如果包含LM头）
        if (includeLMHead && lmHead != null) {
            return lmHead.forward(hiddenStates);
        }
        
        return hiddenStates;
    }
    
    /**
     * 验证输入
     */
    private void validateInput(Variable inputIds) {
        NdArray data = inputIds.getValue();
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape())
            );
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getMaxPositionEmbeddings()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", 
                    seqLen, config.getMaxPositionEmbeddings())
            );
        }
    }
    
    /**
     * 获取配置
     */
    public Qwen3Config getConfig() {
        return config;
    }
    
    /**
     * 获取Transformer层列表
     */
    public List<Qwen3TransformerBlock> getLayers() {
        return layers;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3Block{name='%s', numLayers=%d, hiddenSize=%d, includeLMHead=%b}",
            name, layers.size(), config.getHiddenSize(), includeLMHead
        );
    }
}
