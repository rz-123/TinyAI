package io.leavesfly.tinyai.minimind.tokenizer;

import java.io.*;
import java.text.Normalizer;
import java.util.*;
import java.util.regex.Pattern;

/**
 * MiniMind Tokenizer - 完整 BPE 分词器
 * <p>
 * 功能：
 * - 文本归一化
 * - BPE 分词（支持完整 BPE 训练）
 * - 批量编码/解码
 * - 特殊 Token 处理
 * <p>
 * 支持两种模式：
 * 1. 字符级模式：简化实现，用于演示和测试
 * 2. BPE模式：完整BPE算法，从语料库学习merge规则
 *
 * @author leavesfly
 * @version 2.0
 */
public class MiniMindTokenizer implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * 词汇表
     */
    private final Vocabulary vocabulary;

    /**
     * 最大序列长度
     */
    private final int maxSeqLen;
    
    /**
     * BPE merge规则列表
     * 格式: ["a b", "c d", ...] 表示将"a b"合并为"ab"
     */
    private final List<String> bpeMerges;
    
    /**
     * 是否使用BPE模式
     */
    private final boolean useBPE;

    /**
     * 预分词正则表达式
     */
    private static final Pattern PRE_TOKENIZE_PATTERN = 
        Pattern.compile("\\s+|[a-zA-Z]+|[0-9]+|[^\\s\\w]+");

    /**
     * 构造 Tokenizer
     *
     * @param vocabulary 词汇表
     * @param maxSeqLen  最大序列长度
     * @param bpeMerges  BPE merge规则
     */
    public MiniMindTokenizer(Vocabulary vocabulary, int maxSeqLen, List<String> bpeMerges) {
        this.vocabulary = vocabulary;
        this.maxSeqLen = maxSeqLen;
        this.bpeMerges = bpeMerges != null ? new ArrayList<>(bpeMerges) : new ArrayList<>();
        this.useBPE = !this.bpeMerges.isEmpty();
    }

    /**
     * 创建默认 Tokenizer（字符级）
     *
     * @param vocabSize  词汇表大小
     * @param maxSeqLen  最大序列长度
     * @return Tokenizer 实例
     */
    public static MiniMindTokenizer createCharLevelTokenizer(int vocabSize, int maxSeqLen) {
        Vocabulary vocab = new Vocabulary(vocabSize);
        
        // 添加常用字符到词汇表
        // ASCII 可打印字符 (32-126)
        for (int i = 32; i < 127 && vocab.getVocabSize() < vocabSize; i++) {
            vocab.addToken(String.valueOf((char) i));
        }
        
        // 添加常用中文字符（简化）
        String commonChars = "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开要手知道";
        for (char c : commonChars.toCharArray()) {
            if (vocab.getVocabSize() >= vocabSize) break;
            vocab.addToken(String.valueOf(c));
        }
        
        return new MiniMindTokenizer(vocab, maxSeqLen, null);
    }
    
    /**
     * 从 BPE 训练器创建 Tokenizer
     * 
     * @param trainer BPE训练器
     * @param maxSeqLen 最大序列长度
     * @return Tokenizer实例
     */
    public static MiniMindTokenizer fromBPETrainer(BPETrainer trainer, int maxSeqLen) {
        Vocabulary vocab = new Vocabulary(trainer.getVocab());
        List<String> merges = trainer.getMerges();
        return new MiniMindTokenizer(vocab, maxSeqLen, merges);
    }

    /**
     * 文本归一化
     *
     * @param text 原始文本
     * @return 归一化后的文本
     */
    public String normalize(String text) {
        if (text == null || text.isEmpty()) {
            return "";
        }
        
        // Unicode 规范化 (NFC)
        text = Normalizer.normalize(text, Normalizer.Form.NFC);
        
        // 移除控制字符
        text = text.replaceAll("\\p{Cntrl}", " ");
        
        // 统一空白字符
        text = text.replaceAll("\\s+", " ");
        
        return text.trim();
    }

    /**
     * 预分词（按空格、字母、数字、标点分割）
     *
     * @param text 文本
     * @return 预分词结果
     */
    private List<String> preTokenize(String text) {
        List<String> tokens = new ArrayList<>();
        var matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        
        while (matcher.find()) {
            String token = matcher.group();
            if (!token.trim().isEmpty()) {
                tokens.add(token);
            }
        }
        
        return tokens;
    }

    /**
     * 字符级编码（简化版 BPE）
     *
     * @param text 文本
     * @return Token IDs
     */
    public List<Integer> encode(String text) {
        return encode(text, true, true);
    }

    /**
     * 编码文本
     *
     * @param text         文本
     * @param addBos       是否添加 BOS token
     * @param addEos       是否添加 EOS token
     * @return Token IDs
     */
    public List<Integer> encode(String text, boolean addBos, boolean addEos) {
        List<Integer> tokenIds = new ArrayList<>();
        
        // 添加 BOS token
        if (addBos) {
            tokenIds.add(vocabulary.getBosTokenId());
        }
        
        // 归一化
        text = normalize(text);
        
        if (useBPE) {
            // 使用BPE编码
            List<Integer> bpeIds = encodeBPE(text);
            tokenIds.addAll(bpeIds);
        } else {
            // 字符级编码
            for (char c : text.toCharArray()) {
                String token = String.valueOf(c);
                tokenIds.add(vocabulary.getTokenId(token));
            }
        }
        
        // 添加 EOS token
        if (addEos) {
            tokenIds.add(vocabulary.getEosTokenId());
        }
        
        // 截断到最大长度
        if (tokenIds.size() > maxSeqLen) {
            tokenIds = tokenIds.subList(0, maxSeqLen);
        }
        
        return tokenIds;
    }
    
    /**
     * BPE编码
     * 
     * @param text 文本
     * @return Token IDs
     */
    private List<Integer> encodeBPE(String text) {
        // 1. 初始化: 将文本拆分为字符列表
        List<String> tokens = new ArrayList<>();
        for (char c : text.toCharArray()) {
            tokens.add(String.valueOf(c));
        }
        
        // 2. 应用BPE merge规则
        for (String merge : bpeMerges) {
            String[] pair = merge.split(" ");
            if (pair.length != 2) continue;
            
            String token1 = pair[0];
            String token2 = pair[1];
            String mergedToken = token1 + token2;
            
            // 合并token pair
            tokens = mergePair(tokens, token1, token2, mergedToken);
        }
        
        // 3. 将tokens转换为IDs
        List<Integer> tokenIds = new ArrayList<>();
        for (String token : tokens) {
            tokenIds.add(vocabulary.getTokenId(token));
        }
        
        return tokenIds;
    }
    
    /**
     * 合并token列表中的指定 pair
     */
    private List<String> mergePair(List<String> tokens, String token1, String token2, String mergedToken) {
        List<String> newTokens = new ArrayList<>();
        
        int i = 0;
        while (i < tokens.size()) {
            // 检查是否匹配pair
            if (i < tokens.size() - 1 && 
                tokens.get(i).equals(token1) && 
                tokens.get(i + 1).equals(token2)) {
                newTokens.add(mergedToken);
                i += 2;
            } else {
                newTokens.add(tokens.get(i));
                i += 1;
            }
        }
        
        return newTokens;
    }

    /**
     * 解码 Token IDs
     *
     * @param tokenIds Token IDs
     * @return 解码后的文本
     */
    public String decode(List<Integer> tokenIds) {
        return decode(tokenIds, true);
    }

    /**
     * 解码 Token IDs
     *
     * @param tokenIds       Token IDs
     * @param skipSpecial    是否跳过特殊 Token
     * @return 解码后的文本
     */
    public String decode(List<Integer> tokenIds, boolean skipSpecial) {
        StringBuilder sb = new StringBuilder();
        
        Set<Integer> specialIds = new HashSet<>();
        if (skipSpecial) {
            specialIds.add(vocabulary.getPadTokenId());
            specialIds.add(vocabulary.getBosTokenId());
            specialIds.add(vocabulary.getEosTokenId());
        }
        
        for (int id : tokenIds) {
            if (skipSpecial && specialIds.contains(id)) {
                continue;
            }
            
            String token = vocabulary.getToken(id);
            sb.append(token);
        }
        
        return sb.toString();
    }

    /**
     * 批量编码
     *
     * @param texts        文本列表
     * @param padding      是否填充
     * @param truncation   是否截断
     * @return 编码结果（包含 input_ids 和 attention_mask）
     */
    public EncodingResult batchEncode(List<String> texts, boolean padding, boolean truncation) {
        List<List<Integer>> allInputIds = new ArrayList<>();
        List<List<Integer>> allAttentionMasks = new ArrayList<>();
        
        int maxLen = 0;
        
        // 编码所有文本
        for (String text : texts) {
            List<Integer> inputIds = encode(text);
            
            // 截断
            if (truncation && inputIds.size() > maxSeqLen) {
                inputIds = inputIds.subList(0, maxSeqLen);
            }
            
            maxLen = Math.max(maxLen, inputIds.size());
            allInputIds.add(inputIds);
        }
        
        // 填充
        if (padding) {
            for (List<Integer> inputIds : allInputIds) {
                List<Integer> attentionMask = new ArrayList<>();
                
                // 原始部分的 mask 为 1
                for (int i = 0; i < inputIds.size(); i++) {
                    attentionMask.add(1);
                }
                
                // 填充部分
                int padLen = maxLen - inputIds.size();
                for (int i = 0; i < padLen; i++) {
                    inputIds.add(vocabulary.getPadTokenId());
                    attentionMask.add(0);
                }
                
                allAttentionMasks.add(attentionMask);
            }
        } else {
            // 不填充时，attention mask 全为 1
            for (List<Integer> inputIds : allInputIds) {
                List<Integer> attentionMask = new ArrayList<>();
                for (int i = 0; i < inputIds.size(); i++) {
                    attentionMask.add(1);
                }
                allAttentionMasks.add(attentionMask);
            }
        }
        
        return new EncodingResult(allInputIds, allAttentionMasks);
    }

    /**
     * 获取词汇表
     */
    public Vocabulary getVocabulary() {
        return vocabulary;
    }

    /**
     * 获取词汇表大小
     */
    public int getVocabSize() {
        return vocabulary.getVocabSize();
    }

    /**
     * 获取最大序列长度
     */
    public int getMaxSeqLen() {
        return maxSeqLen;
    }
    
    /**
     * 是否使用BPE模式
     */
    public boolean isUseBPE() {
        return useBPE;
    }
    
    /**
     * 获取BPE merge规则数量
     */
    public int getNumMerges() {
        return bpeMerges.size();
    }

    /**
     * 保存 Tokenizer
     *
     * @param dirPath 保存目录
     */
    public void save(String dirPath) throws IOException {
        File dir = new File(dirPath);
        if (!dir.exists()) {
            dir.mkdirs();
        }
        
        // 保存词汇表
        vocabulary.save(dirPath + "/vocab.txt");
        
        // 保存配置
        try (PrintWriter writer = new PrintWriter(dirPath + "/config.txt")) {
            writer.println("max_seq_len=" + maxSeqLen);
            writer.println("vocab_size=" + vocabulary.getVocabSize());
            writer.println("use_bpe=" + useBPE);
        }
        
        // 保存BPE merge规则
        if (useBPE) {
            try (PrintWriter writer = new PrintWriter(dirPath + "/merges.txt")) {
                for (String merge : bpeMerges) {
                    writer.println(merge);
                }
            }
        }
    }

    /**
     * 加载 Tokenizer
     *
     * @param dirPath 加载目录
     * @return Tokenizer 实例
     */
    public static MiniMindTokenizer load(String dirPath) throws IOException {
        // 加载词汇表
        Vocabulary vocab = Vocabulary.load(dirPath + "/vocab.txt");
        
        // 加载配置
        int maxSeqLen = 512;
        boolean useBPE = false;
        try (BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/config.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("max_seq_len=")) {
                    maxSeqLen = Integer.parseInt(line.split("=")[1]);
                } else if (line.startsWith("use_bpe=")) {
                    useBPE = Boolean.parseBoolean(line.split("=")[1]);
                }
            }
        }
        
        // 加载BPE merge规则
        List<String> merges = null;
        if (useBPE) {
            merges = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(dirPath + "/merges.txt"))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    merges.add(line);
                }
            }
        }
        
        return new MiniMindTokenizer(vocab, maxSeqLen, merges);
    }

    /**
     * 编码结果类
     */
    public static class EncodingResult {
        private final List<List<Integer>> inputIds;
        private final List<List<Integer>> attentionMask;

        public EncodingResult(List<List<Integer>> inputIds, List<List<Integer>> attentionMask) {
            this.inputIds = inputIds;
            this.attentionMask = attentionMask;
        }

        public List<List<Integer>> getInputIds() {
            return inputIds;
        }

        public List<List<Integer>> getAttentionMask() {
            return attentionMask;
        }
    }
}
