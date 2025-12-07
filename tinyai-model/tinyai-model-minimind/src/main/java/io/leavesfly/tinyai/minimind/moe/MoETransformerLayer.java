package io.leavesfly.tinyai.minimind.moe;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.layer.norm.RMSNorm;

/**
 * MoE Transformer Layer - MoE Transformer层
 * 
 * 将MoE集成到Transformer架构中,替换传统的FFN层
 * 
 * 标准Transformer层结构:
 * 1. Multi-Head Self-Attention + Residual
 * 2. RMSNorm
 * 3. MoE Layer + Residual
 * 4. RMSNorm
 * 
 * MoE替换FFN的优势:
 * - 参数容量增加N倍(N=专家数量)
 * - 计算成本只增加Top-K倍
 * - 稀疏激活,提高效率
 * 
 * @author leavesfly
 * @since 2024
 */
public class MoETransformerLayer extends Module {
    
    private final int hiddenDim;
    private final MoEConfig config;
    
    // 注意:此处简化实现,仅展示MoE层集成
    // 完整实现需要包含Attention层
    private final RMSNorm norm1;     // 第一个归一化层
    private final MoELayer moeLayer; // MoE层(替换FFN)
    private final RMSNorm norm2;     // 第二个归一化层
    
    private final LoadBalanceLoss loadBalanceLoss; // 负载均衡损失
    
    /**
     * 构造函数
     * 
     * @param hiddenDim 隐藏层维度
     * @param config MoE配置
     */
    public MoETransformerLayer(int hiddenDim, MoEConfig config) {
        super("moe_transformer_layer");
        
        this.hiddenDim = hiddenDim;
        this.config = config;
        
        // 创建归一化层
        this.norm1 = new RMSNorm("norm1", hiddenDim, 1e-6f);
        this.norm2 = new RMSNorm("norm2", hiddenDim, 1e-6f);
        
        // 创建MoE层
        this.moeLayer = new MoELayer(
            config.getInputDim(),
            config.getHiddenDim(),
            config.getOutputDim(),
            config.getNumExperts(),
            config.getTopK(),
            config.getNoiseFactor()
        );
        
        // 创建负载均衡损失
        this.loadBalanceLoss = new LoadBalanceLoss(
            config.getImportanceCoef(),
            config.getLoadCoef()
        );
        
        // 注册子模块
        registerModule("norm1", norm1);
        registerModule("moe", moeLayer);
        registerModule("norm2", norm2);
    }
    
    /**
     * 前向传播(Variable版本,Module接口要求)
     */
    @Override
    public Variable forward(Variable... inputs) {
        return forwardVar(inputs[0]);
    }
    
    /**
     * 前向传播(Function接口)
     * 
     * @param inputs 输入数组
     * @return 输出NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        Variable input = new Variable(inputs[0]);
        return forwardVar(input).getValue();
    }
    
    /**
     * 前向传播(Variable版本)
     * 
     * 完整流程应包含:
     * 1. x = x + Attention(norm1(x))
     * 2. x = x + MoE(norm2(x))
     * 
     * 此处简化为:
     * x = x + MoE(norm(x))
     * 
     * @param input 输入 [batch_size, seq_len, hidden_dim]
     * @return 输出 [batch_size, seq_len, hidden_dim]
     */
    public Variable forwardVar(Variable input) {
        // 注意:简化实现,假设input已经过Attention层
        
        // 1. Pre-Norm
        Variable normalized = norm1.forward(input);
        
        // 2. MoE Layer
        Variable moeOutput = moeLayer.forwardVar(normalized);
        
        // 3. Residual Connection
        Variable output = input.add(moeOutput);
        
        // 4. Post-Norm(可选)
        // output = norm2.forward(output);
        
        return output;
    }
    
    /**
     * 前向传播(带负载均衡损失)
     * 
     * @param input 输入
     * @return MoE输出结果(包含负载均衡损失)
     */
    public MoEOutput forwardWithLoss(Variable input) {
        // Pre-Norm
        Variable normalized = norm1.forward(input);
        
        // Router计算
        ExpertRouter router = moeLayer.getRouter();
        ExpertRouter.RouterOutput routerOutput = router.forwardRouter(normalized);
        
        // MoE前向传播
        Variable moeOutput = moeLayer.forwardVar(normalized);
        
        // Residual
        Variable output = input.add(moeOutput);
        
        // 计算负载均衡损失
        float balanceLoss = 0.0f;
        if (config.isEnableLoadBalance()) {
            MoELayer.LoadBalanceStats stats = moeLayer.getLoadBalanceStats(routerOutput);
            balanceLoss = loadBalanceLoss.computeLoss(stats, config.getNumExperts());
        }
        
        return new MoEOutput(output, balanceLoss);
    }
    
    /**
     * 获取MoE层
     */
    public MoELayer getMoELayer() {
        return moeLayer;
    }
    
    /**
     * 获取负载均衡损失计算器
     */
    public LoadBalanceLoss getLoadBalanceLoss() {
        return loadBalanceLoss;
    }
    
    /**
     * 获取配置
     */
    public MoEConfig getConfig() {
        return config;
    }
    
    /**
     * 获取专家使用统计
     */
    public MoELayer.ExpertUsageStats getUsageStats() {
        return moeLayer.getUsageStats();
    }
    
    /**
     * 重置统计信息
     */
    public void resetStats() {
        moeLayer.resetStats();
    }
    
    @Override
    public String toString() {
        return String.format("MoETransformerLayer(hidden=%d, experts=%d, topK=%d)",
            hiddenDim, config.getNumExperts(), config.getTopK());
    }
    
    /**
     * MoE输出结果(包含负载均衡损失)
     */
    public static class MoEOutput {
        private final Variable output;        // 输出变量
        private final float balanceLoss;      // 负载均衡损失
        
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
            return String.format("MoEOutput(output_shape=%s, balance_loss=%.6f)",
                output.getShape(), balanceLoss);
        }
    }
}
