package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.activation.GELU;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;
import io.leavesfly.tinyai.nnet.v2.layer.transformer.MultiHeadAttention;

/**
 * GPT-1 Transformer块（Post-LayerNorm架构）
 * 
 * Post-LN架构: SubLayer -> Dropout -> Add -> LayerNorm
 */
public class GPT1TransformerBlock extends Module {
    
    private final GPT1Config config;
    private final MultiHeadAttention attention;
    private final Dropout attnDropout;
    private final LayerNorm layerNorm1;
    private final Linear ffnLinear1;
    private final GELU activation;
    private final Linear ffnLinear2;
    private final Dropout mlpDropout;
    private final LayerNorm layerNorm2;
    
    public GPT1TransformerBlock(String name, GPT1Config config) {
        super(name);
        this.config = config;
        
        int dModel = config.getNEmbd();
        int numHeads = config.getNHead();
        int dFF = config.getNInner();
        float dropout = (float) config.getResidPdrop();
        float attnDropoutRate = (float) config.getAttnPdrop();
        
        this.attention = new MultiHeadAttention("attn", dModel, numHeads, attnDropoutRate);
        this.attnDropout = new Dropout("attn_dropout", dropout);
        this.layerNorm1 = new LayerNorm("ln1", dModel, (float) config.getLayerNormEpsilon());
        this.ffnLinear1 = new Linear("ffn_fc1", dModel, dFF, true);
        this.activation = new GELU("gelu");
        this.ffnLinear2 = new Linear("ffn_fc2", dFF, dModel, true);
        this.mlpDropout = new Dropout("mlp_dropout", dropout);
        this.layerNorm2 = new LayerNorm("ln2", dModel, (float) config.getLayerNormEpsilon());
        
        registerModule("attn", attention);
        registerModule("attn_dropout", attnDropout);
        registerModule("ln1", layerNorm1);
        registerModule("ffn_fc1", ffnLinear1);
        registerModule("gelu", activation);
        registerModule("ffn_fc2", ffnLinear2);
        registerModule("mlp_dropout", mlpDropout);
        registerModule("ln2", layerNorm2);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        Variable x = inputs[0];
        int seqLen = x.getValue().getShape().getDimension(1);
        Variable causalMask = MultiHeadAttention.generateCausalMaskBatched(seqLen);
        
        // Post-LayerNorm: Attention -> Add -> LN -> FFN -> Add -> LN
        Variable attnOutput = attention.forward(x, x, x, causalMask, null);
        attnOutput = attnDropout.forward(attnOutput);
        Variable residual1 = layerNorm1.forward(x.add(attnOutput));
        
        Variable mlpOutput = ffnLinear1.forward(residual1);
        mlpOutput = activation.forward(mlpOutput);
        mlpOutput = ffnLinear2.forward(mlpOutput);
        mlpOutput = mlpDropout.forward(mlpOutput);
        Variable output = layerNorm2.forward(residual1.add(mlpOutput));
        
        return output;
    }
    
    public GPT1Config getConfig() { return config; }
}
