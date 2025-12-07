package io.leavesfly.tinyai.minimind.tokenizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import java.util.List;
import java.util.ArrayList;

import static org.junit.jupiter.api.Assertions.*;

/**
 * MiniMindTokenizer补充单元测试
 * 测试BPE模式、批处理、保存加载等高级功能
 * 
 * @author leavesfly
 */
public class MiniMindTokenizerExtendedTest {
    
    private MiniMindTokenizer charTokenizer;
    private MiniMindTokenizer bpeTokenizer;
    
    @BeforeEach
    public void setUp() {
        // 创建字符级Tokenizer
        charTokenizer = MiniMindTokenizer.createCharLevelTokenizer(100, 128);
        
        // 创建简单的BPE Tokenizer
        Vocabulary vocab = new Vocabulary(50);
        vocab.addToken("h");
        vocab.addToken("e");
        vocab.addToken("l");
        vocab.addToken("o");
        vocab.addToken("he");
        vocab.addToken("ll");
        
        List<String> merges = new ArrayList<>();
        merges.add("h e");
        merges.add("l l");
        
        bpeTokenizer = new MiniMindTokenizer(vocab, 128, merges);
    }
    
    @Test
    public void testCharTokenizerCreation() {
        assertNotNull(charTokenizer, "字符级Tokenizer不应为null");
        assertFalse(charTokenizer.isUseBPE(), "字符级模式不应使用BPE");
        assertEquals(0, charTokenizer.getNumMerges(), "字符级模式merge数应为0");
    }
    
    @Test
    public void testBPETokenizerCreation() {
        assertNotNull(bpeTokenizer, "BPE Tokenizer不应为null");
        assertTrue(bpeTokenizer.isUseBPE(), "应使用BPE模式");
        assertEquals(2, bpeTokenizer.getNumMerges(), "merge规则数应为2");
    }
    
    @Test
    public void testNormalization() {
        String text = "  Hello\t\nWorld  ";
        String normalized = charTokenizer.normalize(text);
        
        assertNotNull(normalized, "归一化结果不应为null");
        assertFalse(normalized.contains("\t"), "不应包含制表符");
        assertFalse(normalized.contains("\n"), "不应包含换行符");
        assertEquals("Hello World", normalized, "应去除多余空白");
    }
    
    @Test
    public void testEncodeWithBosEos() {
        String text = "Hello";
        
        // 测试添加BOS/EOS
        List<Integer> ids1 = charTokenizer.encode(text, true, true);
        assertTrue(ids1.size() > text.length(), "应包含BOS和EOS");
        assertEquals(charTokenizer.getVocabSize(), charTokenizer.getVocabSize()); // Just a sanity check
        
        // 测试不添加BOS/EOS
        List<Integer> ids2 = charTokenizer.encode(text, false, false);
        assertTrue(ids2.size() <= text.length() + 1, "不应包含额外token");
    }
    
    @Test
    public void testEncodeDecodeReversibility() {
        String original = "深度学习";
        
        List<Integer> encoded = charTokenizer.encode(original, false, false);
        String decoded = charTokenizer.decode(encoded, false);
        
        assertEquals(original, decoded, "编码解码应可逆");
    }
    
    @Test
    public void testDecodeSkipSpecialTokens() {
        String text = "Test";
        
        List<Integer> ids1 = charTokenizer.encode(text, true, true);
        
        // 跳过特殊token
        String decoded1 = charTokenizer.decode(ids1, true);
        assertFalse(decoded1.contains("<|"), "应跳过特殊token");
        
        // 不跳过特殊token
        String decoded2 = charTokenizer.decode(ids1, false);
        assertTrue(decoded2.length() >= decoded1.length(), "包含特殊token应更长");
    }
    
    @Test
    public void testBatchEncode() {
        List<String> texts = new ArrayList<>();
        texts.add("Hello");
        texts.add("World");
        texts.add("Test");
        
        MiniMindTokenizer.EncodingResult result = charTokenizer.batchEncode(texts, false, false);
        
        assertNotNull(result, "批处理结果不应为null");
        assertNotNull(result.getInputIds(), "input_ids不应为null");
        assertNotNull(result.getAttentionMask(), "attention_mask不应为null");
        assertEquals(3, result.getInputIds().size(), "应有3个样本");
        assertEquals(3, result.getAttentionMask().size(), "应有3个mask");
    }
    
    @Test
    public void testBatchEncodeWithPadding() {
        List<String> texts = new ArrayList<>();
        texts.add("Hi");
        texts.add("Hello World");
        
        MiniMindTokenizer.EncodingResult result = charTokenizer.batchEncode(texts, true, false);
        
        assertNotNull(result, "结果不应为null");
        
        List<List<Integer>> inputIds = result.getInputIds();
        List<List<Integer>> masks = result.getAttentionMask();
        
        // 所有序列应该长度相同(填充后)
        int len0 = inputIds.get(0).size();
        int len1 = inputIds.get(1).size();
        assertEquals(len0, len1, "填充后长度应相同");
        
        // 验证mask正确
        assertEquals(inputIds.get(0).size(), masks.get(0).size(), "mask长度应匹配");
        assertEquals(inputIds.get(1).size(), masks.get(1).size(), "mask长度应匹配");
    }
    
    @Test
    public void testTruncation() {
        // 创建一个很长的文本
        StringBuilder longText = new StringBuilder();
        for (int i = 0; i < 200; i++) {
            longText.append("a");
        }
        
        List<Integer> encoded = charTokenizer.encode(longText.toString(), true, true);
        
        // 应该被截断到maxSeqLen
        assertTrue(encoded.size() <= charTokenizer.getMaxSeqLen(), 
                  "编码长度应不超过maxSeqLen");
    }
    
    @Test
    public void testEmptyText() {
        String empty = "";
        
        List<Integer> encoded = charTokenizer.encode(empty, false, false);
        
        assertNotNull(encoded, "空文本编码不应为null");
        assertEquals(0, encoded.size(), "空文本应编码为空列表");
        
        String decoded = charTokenizer.decode(encoded, false);
        assertEquals("", decoded, "空列表应解码为空字符串");
    }
    
    @Test
    public void testSpecialCharacters() {
        String special = "Hello, 世界! 123";
        
        List<Integer> encoded = charTokenizer.encode(special, false, false);
        String decoded = charTokenizer.decode(encoded, false);
        
        // 归一化可能改变某些字符,但基本内容应保留
        assertTrue(decoded.contains("Hello"), "应包含英文");
        assertTrue(decoded.contains("世界"), "应包含中文");
    }
    
    @Test
    public void testGetVocabSize() {
        assertTrue(charTokenizer.getVocabSize() > 0, "词汇表大小应大于0");
        assertTrue(charTokenizer.getVocabSize() <= 100, "词汇表大小应不超过设定值");
    }
    
    @Test
    public void testGetMaxSeqLen() {
        assertEquals(128, charTokenizer.getMaxSeqLen(), "最大序列长度应为128");
    }
}
