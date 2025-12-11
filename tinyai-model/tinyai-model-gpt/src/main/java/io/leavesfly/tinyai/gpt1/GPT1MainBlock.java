package io.leavesfly.tinyai.gpt1;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;
import io.leavesfly.tinyai.nnet.v2.layer.norm.LayerNorm;

import java.util.ArrayList;
import java.util.List;

/**
 * GPT-1主体块实现（基于V2 Module）
 */
public class GPT1MainBlock extends Module {
    
    private final GPT1Config config;
    private GPT1TokenEmbedding tokenEmbedding;
    private List<GPT1TransformerBlock> transformerBlocks;
    private LayerNorm finalLayerNorm;
    private Linear outputProjection;
    
    public GPT1MainBlock(String name, GPT1Config config) {
        super(name);
        this.config = config;
        initializeComponents();
    }
    
    private void initializeComponents() {
        tokenEmbedding = new GPT1TokenEmbedding(name + "_token_embedding", config);
        registerModule("token_embedding", tokenEmbedding);
        
        transformerBlocks = new ArrayList<>();
        for (int i = 0; i < config.getNLayer(); i++) {
            GPT1TransformerBlock block = new GPT1TransformerBlock(
                name + "_transformer_" + i, config);
            transformerBlocks.add(block);
            registerModule("transformer_" + i, block);
        }
        
        finalLayerNorm = new LayerNorm(
            name + "_final_ln", 
            config.getNEmbd(),
            (float) config.getLayerNormEpsilon());
        registerModule("final_ln", finalLayerNorm);
        
        outputProjection = new Linear(
            name + "_output_proj",
            config.getNEmbd(),
            config.getVocabSize(),
            false);
        registerModule("output_proj", outputProjection);
    }
    
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("输入不能为空");
        }
        
        Variable tokenIds = inputs[0];
        validateInput(tokenIds);
        
        Variable x = tokenEmbedding.forward(tokenIds);
        
        for (GPT1TransformerBlock block : transformerBlocks) {
            x = block.forward(x);
        }
        
        x = finalLayerNorm.forward(x);
        Variable logits = outputProjection.forward(x);
        
        return logits;
    }
    
    private void validateInput(Variable tokenIds) {
        NdArray data = tokenIds.getValue();
        if (data.getShape().getDimNum() != 2) {
            throw new IllegalArgumentException(
                String.format("输入必须是2维张量 (batch_size, seq_len)，实际: %s", 
                    data.getShape()));
        }
        
        int seqLen = data.getShape().getDimension(1);
        if (seqLen > config.getNPositions()) {
            throw new IllegalArgumentException(
                String.format("序列长度(%d)超过最大位置数(%d)", seqLen, config.getNPositions()));
        }
    }
    
    public long getParameterCount() {
        return config.estimateParameterCount();
    }
    
    public void printArchitecture() {
        System.out.println("=".repeat(60));
        System.out.println("GPT-1 主体块架构");
        System.out.println("=".repeat(60));
        System.out.printf("配置: %s\n", config);
        System.out.println("-".repeat(60));
        System.out.printf("Token嵌入层: %s\n", tokenEmbedding.getClass().getSimpleName());
        System.out.printf("Transformer块数量: %d\n", transformerBlocks.size());
        System.out.printf("架构模式: Post-LayerNorm\n");
        System.out.printf("估算参数数量: %s\n", formatParamCount(getParameterCount()));
        System.out.println("=".repeat(60));
    }
    
    private String formatParamCount(long count) {
        if (count >= 1_000_000_000) {
            return String.format("%.2f B", count / 1_000_000_000.0);
        } else if (count >= 1_000_000) {
            return String.format("%.2f M", count / 1_000_000.0);
        } else {
            return String.format("%,d", count);
        }
    }
    
    public GPT1Config getConfig() { return config; }
    public List<GPT1TransformerBlock> getTransformerBlocks() { return transformerBlocks; }
}
