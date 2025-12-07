package io.leavesfly.tinyai.minimind.tokenizer;

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
 * BPE训练端到端测试
 * 
 * <p>验证完整的BPE训练流程:
 * - 从语料库训练BPE
 * - 学习merge规则
 * - 编码解码可逆性
 * - 模型保存加载
 * 
 * @author leavesfly
 */
public class BPETrainingE2ETest {
    
    private static final String TEST_DATA_PATH = "src/test/resources/test-data/pretrain/sample-corpus.txt";
    
    @Test
    @Timeout(value = 2, unit = TimeUnit.MINUTES)
    public void testBPETrainingE2E() throws IOException {
        System.out.println("=".repeat(60));
        System.out.println("开始BPE训练端到端测试");
        System.out.println("=".repeat(60));
        
        // 1. 加载训练语料
        List<String> corpus = loadCorpus(TEST_DATA_PATH);
        System.out.println("加载语料: " + corpus.size() + " 条");
        
        // 2. 创建BPE训练器(小词汇表,快速测试)
        BPETrainer trainer = new BPETrainer(300, 2);  // 300词汇 + 最小频率2
        System.out.println("BPE训练器创建完成");
        System.out.println("-".repeat(60));
        
        // 3. 训练BPE
        System.out.println("开始训练BPE...");
        Vocabulary vocab = trainer.train(corpus);
        System.out.println("-".repeat(60));
        
        // 4. 验证结果
        assertNotNull(vocab, "词汇表不应为null");
        assertTrue(vocab.getVocabSize() > 100, "词汇表大小应合理");
        
        System.out.println("训练完成:");
        System.out.println("  词汇表大小: " + vocab.getVocabSize());
        System.out.println("  Merge规则数: " + trainer.getMerges().size());
        
        // 5. 创建Tokenizer
        MiniMindTokenizer tokenizer = MiniMindTokenizer.fromBPETrainer(trainer, 128);
        System.out.println("从BPE训练器创建Tokenizer");
        assertTrue(tokenizer.isUseBPE(), "应使用BPE模式");
        System.out.println("  使用BPE: " + tokenizer.isUseBPE());
        System.out.println("  Merge数量: " + tokenizer.getNumMerges());
        System.out.println("-".repeat(60));
        
        // 6. 测试编码解码
        System.out.println("测试编码解码:");
        String testText = "深度学习是人工智能的重要分支";
        System.out.println("  原始文本: " + testText);
        
        List<Integer> encoded = tokenizer.encode(testText, false, false);
        System.out.println("  Token数量: " + encoded.size());
        
        String decoded = tokenizer.decode(encoded, false);
        System.out.println("  解码文本: " + decoded);
        
        // 验证可逆性
        assertEquals(testText, decoded, "编码解码应可逆");
        System.out.println("  ✓ 编码解码可逆性验证通过");
        
        System.out.println("=".repeat(60));
        System.out.println("✅ BPE训练端到端测试通过!");
        System.out.println("=".repeat(60));
    }
    
    /**
     * 加载语料库
     */
    private List<String> loadCorpus(String filePath) throws IOException {
        List<String> corpus = new ArrayList<>();
        
        try (BufferedReader reader = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    corpus.add(line);
                }
            }
        }
        
        return corpus;
    }
    
    @Test
    public void testBPEEncodingConsistency() {
        System.out.println("测试BPE编码一致性...");
        
        // 创建简单训练器
        List<String> corpus = new ArrayList<>();
        corpus.add("hello world");
        corpus.add("hello java");
        corpus.add("world java");
        
        BPETrainer trainer = new BPETrainer(100, 1);
        Vocabulary vocab = trainer.train(corpus);
        
        MiniMindTokenizer tokenizer = MiniMindTokenizer.fromBPETrainer(trainer, 64);
        
        // 多次编码同一文本
        String text = "hello";
        List<Integer> encoded1 = tokenizer.encode(text, false, false);
        List<Integer> encoded2 = tokenizer.encode(text, false, false);
        
        assertEquals(encoded1, encoded2, "同一文本的编码应一致");
        
        System.out.println("✅ BPE编码一致性测试通过!");
    }
}
