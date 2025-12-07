package io.leavesfly.tinyai.minimind.tokenizer;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Vocabulary单元测试
 * 
 * @author leavesfly
 */
public class VocabularyTest {
    
    private Vocabulary vocab;
    
    @BeforeEach
    public void setUp() {
        vocab = new Vocabulary(100);
    }
    
    @Test
    public void testVocabularyCreation() {
        assertNotNull(vocab, "词汇表不应为null");
        assertTrue(vocab.getVocabSize() >= 4, "词汇表应包含特殊tokens");
    }
    
    @Test
    public void testSpecialTokens() {
        // 验证特殊token IDs
        assertEquals(0, vocab.getPadTokenId(), "PAD token ID应为0");
        assertEquals(1, vocab.getUnkTokenId(), "UNK token ID应为1");
        assertEquals(2, vocab.getBosTokenId(), "BOS token ID应为2");
        assertEquals(3, vocab.getEosTokenId(), "EOS token ID应为3");
        
        // 验证特殊token可以查询
        assertEquals(Vocabulary.PAD_TOKEN, vocab.getToken(0));
        assertEquals(Vocabulary.UNK_TOKEN, vocab.getToken(1));
        assertEquals(Vocabulary.BOS_TOKEN, vocab.getToken(2));
        assertEquals(Vocabulary.EOS_TOKEN, vocab.getToken(3));
    }
    
    @Test
    public void testAddToken() {
        String newToken = "test_token";
        int id = vocab.addToken(newToken);
        
        assertTrue(id >= 0, "Token ID应该有效");
        assertEquals(newToken, vocab.getToken(id), "应能通过ID获取token");
        assertEquals(id, vocab.getTokenId(newToken), "应能通过token获取ID");
    }
    
    @Test
    public void testAddDuplicateToken() {
        String token = "duplicate";
        int id1 = vocab.addToken(token);
        int id2 = vocab.addToken(token);
        
        assertEquals(id1, id2, "重复添加应返回相同ID");
    }
    
    @Test
    public void testGetUnknownToken() {
        String unknownToken = "unknown_token_xyz";
        int id = vocab.getTokenId(unknownToken);
        
        assertEquals(vocab.getUnkTokenId(), id, "未知token应返回UNK ID");
    }
    
    @Test
    public void testGetUnknownId() {
        int unknownId = 9999;
        String token = vocab.getToken(unknownId);
        
        assertEquals(Vocabulary.UNK_TOKEN, token, "未知ID应返回UNK token");
    }
    
    @Test
    public void testContainsToken() {
        String token = "test";
        vocab.addToken(token);
        
        assertTrue(vocab.containsToken(token), "应包含已添加的token");
        assertFalse(vocab.containsToken("not_exists"), "不应包含未添加的token");
    }
    
    @Test
    public void testContainsId() {
        assertTrue(vocab.containsId(0), "应包含PAD token ID");
        assertTrue(vocab.containsId(1), "应包含UNK token ID");
        assertFalse(vocab.containsId(9999), "不应包含不存在的ID");
    }
    
    @Test
    public void testGetAllTokens() {
        var tokens = vocab.getAllTokens();
        
        assertNotNull(tokens, "所有tokens不应为null");
        assertTrue(tokens.size() >= 4, "应至少包含4个特殊tokens");
        assertTrue(tokens.contains(Vocabulary.PAD_TOKEN));
        assertTrue(tokens.contains(Vocabulary.UNK_TOKEN));
        assertTrue(tokens.contains(Vocabulary.BOS_TOKEN));
        assertTrue(tokens.contains(Vocabulary.EOS_TOKEN));
    }
    
    @Test
    public void testVocabularySize() {
        int initialSize = vocab.getVocabSize();
        
        vocab.addToken("token1");
        vocab.addToken("token2");
        
        assertEquals(initialSize + 2, vocab.getVocabSize(), "添加token后大小应增加");
    }
    
    @Test
    public void testToString() {
        String str = vocab.toString();
        
        assertNotNull(str, "toString不应为null");
        assertTrue(str.contains("Vocabulary"), "应包含类名");
        assertTrue(str.contains("vocabSize"), "应包含词汇表大小");
    }
}
