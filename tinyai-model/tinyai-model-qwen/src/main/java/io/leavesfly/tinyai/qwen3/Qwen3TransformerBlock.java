package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;

/**
 * Qwen3 Transformer解码器块
 * 
 * Pre-LayerNorm架构：
 * 1. RMSNorm -> Self-Attention -> 残差连接
 * 2. RMSNorm -> MLP -> 残差连接
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3TransformerBlock extends Module {
    
    private final Qwen3Config config;
    
    private RMSNormLayer inputLayerNorm;        // 注意力前归一化
    private Qwen3AttentionBlock selfAttention;  // 自注意力
    private RMSNormLayer postAttentionLayerNorm; // MLP前归一化
    private Qwen3MLPBlock mlp;                  // 前馈网络
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3配置
     */
    public Qwen3TransformerBlock(String name, Qwen3Config config) {
        super(name);
        this.config = config;
        initializeLayers();
    }
    
    /**
     * 初始化各层
     */
    private void initializeLayers() {
        // 输入归一化层
        inputLayerNorm = new RMSNormLayer(
            name + "_input_layernorm",
            config.getHiddenSize(),
            config.getRmsNormEps()
        );
        registerModule("input_layernorm", inputLayerNorm);
        
        // 自注意力块
        selfAttention = new Qwen3AttentionBlock(
            name + "_self_attn",
            config
        );
        registerModule("self_attn", selfAttention);
        
        // 注意力后归一化层
        postAttentionLayerNorm = new RMSNormLayer(
            name + "_post_attention_layernorm",
            config.getHiddenSize(),
            config.getRmsNormEps()
        );
        registerModule("post_attention_layernorm", postAttentionLayerNorm);
        
        // MLP块
        mlp = new Qwen3MLPBlock(name + "_mlp", config);
        registerModule("mlp", mlp);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入隐藏状态 [batch_size, seq_len, hidden_size]
     * @return 输出隐藏状态 [batch_size, seq_len, hidden_size]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("Transformer块输入不能为空");
        }
        
        Variable hiddenStates = inputs[0];
        
        // 1. 自注意力子层：LayerNorm -> SelfAttention -> Residual
        Variable normed1 = inputLayerNorm.forward(hiddenStates);
        Variable attnOutput = selfAttention.forward(normed1);
        Variable residual1 = hiddenStates.add(attnOutput);
        
        // 2. MLP子层：LayerNorm -> MLP -> Residual
        Variable normed2 = postAttentionLayerNorm.forward(residual1);
        Variable mlpOutput = mlp.forward(normed2);
        Variable output = residual1.add(mlpOutput);
        
        return output;
    }
    
    @Override
    public String toString() {
        return String.format("Qwen3TransformerBlock{name='%s'}", name);
    }
}
