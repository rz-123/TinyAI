package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * Qwen3 MLP块（前馈网络）
 * 
 * 使用SwiGLU激活函数的前馈网络结构：
 * 1. Gate投影：hidden_size -> intermediate_size
 * 2. Up投影：hidden_size -> intermediate_size
 * 3. SwiGLU激活：swish(gate) ⊙ up
 * 4. Down投影：intermediate_size -> hidden_size
 * 
 * SwiGLU公式：
 * FFN_SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up)W_down
 * 其中 Swish(x) = x * sigmoid(x)
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3MLPBlock extends Module {
    
    private final Qwen3Config config;
    
    private Linear gateProj;   // 门控投影
    private Linear upProj;     // 上投影  
    private Linear downProj;   // 下投影
    
    /**
     * 构造函数
     * 
     * @param name 模块名称
     * @param config Qwen3配置
     */
    public Qwen3MLPBlock(String name, Qwen3Config config) {
        super(name);
        this.config = config;
        initializeLayers();
    }
    
    /**
     * 初始化各层
     */
    private void initializeLayers() {
        // Gate投影层：hidden_size -> intermediate_size
        gateProj = new Linear(
            name + "_gate_proj",
            config.getHiddenSize(),
            config.getIntermediateSize(),
            false  // 不使用偏置
        );
        registerModule("gate_proj", gateProj);
        
        // Up投影层：hidden_size -> intermediate_size
        upProj = new Linear(
            name + "_up_proj",
            config.getHiddenSize(),
            config.getIntermediateSize(),
            false  // 不使用偏置
        );
        registerModule("up_proj", upProj);
        
        // Down投影层：intermediate_size -> hidden_size
        downProj = new Linear(
            name + "_down_proj",
            config.getIntermediateSize(),
            config.getHiddenSize(),
            false  // 不使用偏置
        );
        registerModule("down_proj", downProj);
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
            throw new IllegalArgumentException("MLP输入不能为空");
        }
        
        Variable hiddenStates = inputs[0];
        
        // 1. Gate投影和Up投影
        Variable gateOutput = gateProj.forward(hiddenStates);
        Variable upOutput = upProj.forward(hiddenStates);
        
        // 2. 应用SwiGLU激活函数：swish(gate) ⊙ up
        Variable swiGLUOutput = applySwiGLU(gateOutput, upOutput);
        
        // 3. Down投影
        Variable output = downProj.forward(swiGLUOutput);
        
        return output;
    }
    
    /**
     * 应用SwiGLU激活函数
     * 
     * SwiGLU(gate, up) = Swish(gate) ⊙ up
     * 其中 Swish(x) = x * sigmoid(x)
     * 
     * @param gate 门控输出
     * @param up 上投影输出
     * @return SwiGLU激活后的输出
     */
    private Variable applySwiGLU(Variable gate, Variable up) {
        // Swish(x) = x * sigmoid(x)
        Variable swishGate = applySwish(gate);
        
        // SwiGLU = swish(gate) ⊙ up（逐元素乘法）
        return swishGate.mul(up);
    }
    
    /**
     * 应用Swish激活函数
     * 
     * Swish(x) = x * sigmoid(x)
     * 
     * @param x 输入
     * @return Swish激活后的输出
     */
    private Variable applySwish(Variable x) {
        // sigmoid(x) = 1 / (1 + exp(-x))
        Variable negX = x.mul(new Variable(-1.0f));
        Variable expNegX = negX.exp();
        Variable onePlusExp = expNegX.add(new Variable(1.0f));
        Variable sigmoid = new Variable(1.0f).div(onePlusExp);
        
        // swish(x) = x * sigmoid(x)
        return x.mul(sigmoid);
    }
    
    /**
     * 获取Gate投影层
     */
    public Linear getGateProj() {
        return gateProj;
    }
    
    /**
     * 获取Up投影层
     */
    public Linear getUpProj() {
        return upProj;
    }
    
    /**
     * 获取Down投影层
     */
    public Linear getDownProj() {
        return downProj;
    }
    
    @Override
    public String toString() {
        return String.format(
            "Qwen3MLPBlock{name='%s', hiddenSize=%d, intermediateSize=%d}",
            name, config.getHiddenSize(), config.getIntermediateSize()
        );
    }
}
