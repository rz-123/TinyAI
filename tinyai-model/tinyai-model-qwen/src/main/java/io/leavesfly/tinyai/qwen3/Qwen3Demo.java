package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * Qwen3模型演示程序
 * 
 * 展示Qwen3模型的基本使用方法，包括：
 * 1. 创建不同规模的模型
 * 2. 前向传播推理
 * 3. 模型信息查看
 * 4. 配置管理
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Demo {
    
    public static void main(String[] args) {
        System.out.println("=".repeat(80));
        System.out.println("Qwen3 模型演示程序");
        System.out.println("=".repeat(80));
        System.out.println();
        
        // 演示1: 创建小型模型
        demo1_CreateSmallModel();
        System.out.println();
        
        // 演示2: 模型前向传播
        demo2_ForwardPass();
        System.out.println();
        
        // 演示3: 配置管理
        demo3_ConfigManagement();
        System.out.println();
        
        // 演示4: 模型信息
        demo4_ModelInfo();
        System.out.println();
        
        System.out.println("=".repeat(80));
        System.out.println("演示完成！");
        System.out.println("=".repeat(80));
    }
    
    /**
     * 演示1: 创建不同规模的模型
     */
    private static void demo1_CreateSmallModel() {
        System.out.println("【演示1】创建不同规模的Qwen3模型");
        System.out.println("-".repeat(80));
        
        try {
            // 创建小型模型（约16M参数）
            System.out.println("1. 创建小型模型...");
            Qwen3Model smallModel = Qwen3Model.createSmallModel("qwen3-small");
            System.out.println("   ✓ 小型模型创建成功");
            System.out.println("   - 模型名称: " + smallModel.getName());
            System.out.println("   - 参数量: " + formatParamCount(smallModel.getConfig().estimateParameterCount()));
            
            // 创建演示模型（约62M参数）
            System.out.println("\n2. 创建演示模型...");
            Qwen3Model demoModel = Qwen3Model.createDemoModel("qwen3-demo");
            System.out.println("   ✓ 演示模型创建成功");
            System.out.println("   - 模型名称: " + demoModel.getName());
            System.out.println("   - 参数量: " + formatParamCount(demoModel.getConfig().estimateParameterCount()));
            
            // 创建自定义配置模型
            System.out.println("\n3. 创建自定义配置模型...");
            Qwen3Config customConfig = new Qwen3Config();
            customConfig.setHiddenSize(256);
            customConfig.setNumHiddenLayers(2);
            customConfig.setNumAttentionHeads(4);
            customConfig.setNumKeyValueHeads(4);  // 必须能被注意力头数整除
            customConfig.setIntermediateSize(704);
            customConfig.validate();
            
            Qwen3Model customModel = new Qwen3Model("qwen3-custom", customConfig);
            System.out.println("   ✓ 自定义模型创建成功");
            System.out.println("   - 隐藏维度: " + customConfig.getHiddenSize());
            System.out.println("   - 层数: " + customConfig.getNumHiddenLayers());
            System.out.println("   - 注意力头数: " + customConfig.getNumAttentionHeads());
            
        } catch (Exception e) {
            System.err.println("   ✗ 创建模型失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示2: 模型前向传播
     */
    private static void demo2_ForwardPass() {
        System.out.println("【演示2】模型前向传播");
        System.out.println("-".repeat(80));
        
        try {
            // 创建小型模型
            Qwen3Model model = Qwen3Model.createSmallModel("qwen3-small");
            
            // 准备输入数据 [batch_size=2, seq_len=8]
            System.out.println("1. 准备输入数据...");
            int batchSize = 2;
            int seqLen = 8;
            float[][] inputData = new float[batchSize][seqLen];
            
            // 随机初始化token IDs (0到vocab_size-1)
            for (int b = 0; b < batchSize; b++) {
                for (int t = 0; t < seqLen; t++) {
                    inputData[b][t] = (float) (Math.random() * model.getConfig().getVocabSize());
                }
            }
            
            NdArray inputIds = NdArray.of(inputData);
            System.out.println("   ✓ 输入形状: " + inputIds.getShape());
            System.out.println("   - batch_size: " + batchSize);
            System.out.println("   - seq_len: " + seqLen);
            
            // 前向传播
            System.out.println("\n2. 执行前向传播...");
            long startTime = System.currentTimeMillis();
            Variable output = model.forward(new Variable(inputIds));
            long endTime = System.currentTimeMillis();
            
            System.out.println("   ✓ 前向传播成功");
            System.out.println("   - 输出形状: " + output.getValue().getShape());
            System.out.println("   - 耗时: " + (endTime - startTime) + " ms");
            
            // 验证输出形状
            Shape expectedShape = Shape.of(batchSize, seqLen, model.getConfig().getVocabSize());
            if (output.getValue().getShape().equals(expectedShape)) {
                System.out.println("   ✓ 输出形状验证通过");
            } else {
                System.out.println("   ✗ 输出形状不匹配");
                System.out.println("     期望: " + expectedShape);
                System.out.println("     实际: " + output.getValue().getShape());
            }
            
        } catch (Exception e) {
            System.err.println("   ✗ 前向传播失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示3: 配置管理
     */
    private static void demo3_ConfigManagement() {
        System.out.println("【演示3】配置管理");
        System.out.println("-".repeat(80));
        
        try {
            // 创建并验证配置
            System.out.println("1. 创建自定义配置...");
            Qwen3Config config = new Qwen3Config();
            
            // 基础配置
            config.setVocabSize(32000);
            config.setHiddenSize(512);
            config.setIntermediateSize(1408);
            config.setNumHiddenLayers(6);
            config.setNumAttentionHeads(8);
            config.setNumKeyValueHeads(8);
            config.setMaxPositionEmbeddings(2048);
            
            // RMSNorm配置
            config.setRmsNormEps(1e-6);
            
            // RoPE配置
            config.setRopeTheta(10000.0);
            
            // 特殊标记配置
            config.setPadTokenId(0);
            config.setBosTokenId(1);
            config.setEosTokenId(2);
            
            System.out.println("   ✓ 配置创建完成");
            
            // 验证配置
            System.out.println("\n2. 验证配置有效性...");
            config.validate();
            System.out.println("   ✓ 配置验证通过");
            
            // 显示配置信息
            System.out.println("\n3. 配置详情:");
            System.out.println(config);
            
            // 显示计算属性
            System.out.println("\n4. 计算属性:");
            System.out.println("   - 头维度: " + config.getHeadDim());
            System.out.println("   - 键值组数: " + config.getNumKeyValueGroups());
            System.out.println("   - 估算参数量: " + formatParamCount(config.estimateParameterCount()));
            
        } catch (Exception e) {
            System.err.println("   ✗ 配置管理失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 演示4: 模型信息
     */
    private static void demo4_ModelInfo() {
        System.out.println("【演示4】模型信息查看");
        System.out.println("-".repeat(80));
        
        try {
            // 创建模型
            Qwen3Model model = Qwen3Model.createDemoModel("qwen3-demo");
            
            // 打印模型信息
            System.out.println("1. 模型基本信息:");
            model.printModelInfo();
            
            // 打印配置摘要
            System.out.println("\n2. 配置摘要:");
            System.out.println(model.getConfigSummary());
            
            // 打印模型字符串表示
            System.out.println("\n3. 模型toString:");
            System.out.println(model);
            
        } catch (Exception e) {
            System.err.println("   ✗ 查看模型信息失败: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * 格式化参数数量
     */
    private static String formatParamCount(long count) {
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
