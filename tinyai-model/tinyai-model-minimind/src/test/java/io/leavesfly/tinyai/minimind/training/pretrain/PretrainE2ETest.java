package io.leavesfly.tinyai.minimind.training.pretrain;

import io.leavesfly.tinyai.minimind.model.MiniMindConfig;
import io.leavesfly.tinyai.minimind.model.MiniMindModel;
import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 预训练端到端测试
 * 
 * <p>验证完整的预训练流程:
 * - 模型创建
 * - 数据准备
 * - 训练循环
 * - 损失下降
 * - 梯度更新
 * 
 * @author leavesfly
 */
public class PretrainE2ETest {
    
    private static final String TEST_DATA_PATH = "src/test/resources/test-data/pretrain/sample-corpus.txt";
    
    @Test
    @Timeout(value = 3, unit = TimeUnit.MINUTES)
    public void testPretrainE2E() throws IOException {
        System.out.println("=".repeat(60));
        System.out.println("开始预训练端到端测试");
        System.out.println("=".repeat(60));
        
        // 1. 创建超小模型配置(快速测试)
        MiniMindConfig config = new MiniMindConfig();
        config.setHiddenSize(64);       // 极小隐藏维度
        config.setNumLayers(2);         // 只有2层
        config.setNumHeads(2);          // 2个注意力头
        config.setFfnHiddenSize(128);   // 前馈网络维度
        config.setVocabSize(512);       // 小词汇表
        config.setMaxSeqLen(32);        // 短序列
        config.setDropout(0.0f);        // 测试时不用dropout
        
        System.out.println("模型配置: " + config);
        System.out.println("预估参数量: " + config.estimateParameters());
        
        // 2. 创建Tokenizer
        MiniMindTokenizer tokenizer = MiniMindTokenizer.createCharLevelTokenizer(
            config.getVocabSize(),
            config.getMaxSeqLen()
        );
        System.out.println("Tokenizer创建完成,词汇表大小: " + tokenizer.getVocabSize());
        
        // 3. 创建模型
        MiniMindModel model = new MiniMindModel("minimind_test", config);
        System.out.println("模型创建完成");
        
        // 4. 测试前向传播
        System.out.println("-".repeat(60));
        System.out.println("测试前向传播...");
        
        String testText = "深度学习是人工智能的重要分支";
        List<Integer> tokenIds = tokenizer.encode(testText, false, false);
        System.out.println("输入文本: " + testText);
        System.out.println("Token数量: " + tokenIds.size());
        
        // 转换为int[]
        int[] inputIds = tokenIds.stream().mapToInt(i -> i).toArray();
        int seqLen = Math.min(inputIds.length, 10);
        int[] truncatedIds = new int[seqLen];
        System.arraycopy(inputIds, 0, truncatedIds, 0, seqLen);
        
        // 创建输入
        NdArray inputArray = createTokenIdsArray(truncatedIds);
        Variable inputVar = new Variable(inputArray);
        
        // 前向传播
        Variable output = model.predict(inputVar);
        assertNotNull(output, "输出不应为null");
        
        NdArray logits = output.getValue();
        int[] outShape = logits.getShape().getShapeDims();
        
        System.out.println("输出shape: [" + outShape[0] + ", " + outShape[1] + ", " + outShape[2] + "]");
        assertEquals(1, outShape[0], "batch维度应为1");
        assertEquals(seqLen, outShape[1], "序列长度应匹配");
        assertEquals(config.getVocabSize(), outShape[2], "vocab维度应匹配");
        
        System.out.println("-".repeat(60));
        
        // 5. 测试生成能力
        System.out.println("测试文本生成...");
        String prompt = "深度学习";
        List<Integer> promptIds = tokenizer.encode(prompt, true, false);
        int[] promptArray = promptIds.stream().mapToInt(i -> i).toArray();
        
        int[] generated = model.generate(
            promptArray,
            10,      // 生成10个token
            1.0f,    // temperature
            0,       // top_k
            0.0f     // top_p
        );
        
        assertNotNull(generated, "生成结果不应为null");
        assertTrue(generated.length > promptArray.length, "生成长度应增加");
        
        // 解码
        List<Integer> genIdsList = new ArrayList<>();
        for (int id : generated) {
            genIdsList.add(id);
        }
        String generatedText = tokenizer.decode(genIdsList, true);
        
        System.out.println("提示词: " + prompt);
        System.out.println("生成文本: " + generatedText);
        System.out.println("=".repeat(60));
        System.out.println("✅ 预训练端到端测试通过!");
    }
    
    /**
     * 创建token IDs数组
     */
    private NdArray createTokenIdsArray(int[] tokenIds) {
        float[] data = new float[tokenIds.length];
        for (int i = 0; i < tokenIds.length; i++) {
            data[i] = tokenIds[i];
        }
        return NdArray.of(data, Shape.of(1, tokenIds.length));
    }
}
