package io.leavesfly.tinyai.minimind.training.dataset;

import io.leavesfly.tinyai.minimind.tokenizer.MiniMindTokenizer;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 预训练数据集
 * 
 * 负责加载文本数据并转换为模型训练所需的Token序列
 * 支持因果语言建模(Causal Language Modeling)任务
 * 
 * @author leavesfly
 * @since 2024
 */
public class PretrainDataset implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    
    // 存储所有训练样本
    private List<int[]> samples;
    
    // 批次数据
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * 构造函数
     * 
     * @param tokenizer 分词器
     * @param maxSeqLen 最大序列长度
     * @param batchSize 批次大小
     */
    public PretrainDataset(MiniMindTokenizer tokenizer, int maxSeqLen, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从文本文件加载数据
     * 
     * @param filePath 文件路径
     * @throws IOException IO异常
     */
    public void loadFromFile(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        loadFromTexts(lines);
    }
    
    /**
     * 从文本列表加载数据
     * 
     * @param texts 文本列表
     */
    public void loadFromTexts(List<String> texts) {
        samples.clear();
        
        for (String text : texts) {
            if (text == null || text.trim().isEmpty()) {
                continue;
            }
            
            // 对文本进行分词编码
            List<Integer> tokenIds = tokenizer.encode(text, true, true);
            
            // 将长文本切分为多个固定长度的序列
            splitIntoSequences(tokenIds);
        }
        
        System.out.println("数据加载完成,共 " + samples.size() + " 个训练样本");
    }
    
    /**
     * 将Token序列切分为固定长度的训练样本
     * 
     * @param tokenIds Token ID列表
     */
    private void splitIntoSequences(List<Integer> tokenIds) {
        int totalLen = tokenIds.size();
        
        // 如果序列太短,跳过
        if (totalLen < 2) {
            return;
        }
        
        // 滑动窗口切分
        for (int i = 0; i + maxSeqLen + 1 <= totalLen; i += maxSeqLen) {
            int[] sample = new int[maxSeqLen + 1];
            for (int j = 0; j <= maxSeqLen; j++) {
                sample[j] = tokenIds.get(i + j);
            }
            samples.add(sample);
        }
        
        // 处理剩余部分(如果足够长)
        int remaining = totalLen % maxSeqLen;
        if (remaining > 10) { // 至少保留10个token
            int startIdx = totalLen - remaining;
            int[] sample = new int[remaining];
            for (int j = 0; j < remaining; j++) {
                sample[j] = tokenIds.get(startIdx + j);
            }
            samples.add(sample);
        }
    }
    
    /**
     * 准备批次数据
     * 
     * @param shuffle 是否打乱数据
     */
    public void prepare(boolean shuffle) {
        if (samples.isEmpty()) {
            throw new IllegalStateException("数据集为空,请先加载数据");
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        // 打乱样本
        List<int[]> workingSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(workingSamples);
        }
        
        // 创建批次
        for (int i = 0; i < workingSamples.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, workingSamples.size());
            List<int[]> batchSamples = workingSamples.subList(i, endIdx);
            batches.add(createBatch(batchSamples));
        }
        
        System.out.println("批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 创建单个批次
     * 
     * @param batchSamples 批次样本列表
     * @return 批次对象
     */
    private Batch createBatch(List<int[]> batchSamples) {
        int actualBatchSize = batchSamples.size();
        
        // 找出批次中最长的序列
        int maxLen = batchSamples.stream()
            .mapToInt(sample -> sample.length)
            .max()
            .orElse(maxSeqLen);
        
        // 输入数据: [batchSize, seqLen]
        int[][] inputData = new int[actualBatchSize][maxLen - 1];
        // 目标数据: [batchSize, seqLen]
        int[][] targetData = new int[actualBatchSize][maxLen - 1];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int[] sample = batchSamples.get(i);
            int seqLen = sample.length - 1;
            
            // 输入是前n个token
            for (int j = 0; j < seqLen; j++) {
                inputData[i][j] = sample[j];
            }
            
            // 目标是后n个token(向右偏移1位)
            for (int j = 0; j < seqLen; j++) {
                targetData[i][j] = sample[j + 1];
            }
            
            // 填充到maxLen-1
            int padTokenId = tokenizer.getVocabulary().getPadTokenId();
            for (int j = seqLen; j < maxLen - 1; j++) {
                inputData[i][j] = padTokenId;
                targetData[i][j] = padTokenId;
            }
        }
        
        // 转换为NdArray
        NdArray inputArray = createNdArray(inputData, actualBatchSize, maxLen - 1);
        NdArray targetArray = createNdArray(targetData, actualBatchSize, maxLen - 1);
        
        return new Batch(inputArray, targetArray, actualBatchSize, maxLen - 1);
    }
    
    /**
     * 创建NdArray
     * 
     * @param data 二维整数数组
     * @param batchSize 批次大小
     * @param seqLen 序列长度
     * @return NdArray对象
     */
    private NdArray createNdArray(int[][] data, int batchSize, int seqLen) {
        float[] flatData = new float[batchSize * seqLen];
        int idx = 0;
        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < seqLen; j++) {
                flatData[idx++] = (float) data[i][j];
            }
        }
        return NdArray.of(flatData, Shape.of(batchSize, seqLen));
    }
    
    /**
     * 是否还有下一个批次
     * 
     * @return true如果还有批次
     */
    public boolean hasNextBatch() {
        return currentBatchIndex < batches.size();
    }
    
    /**
     * 获取下一个批次
     * 
     * @return 批次对象
     */
    public Batch getNextBatch() {
        if (!hasNextBatch()) {
            throw new NoSuchElementException("没有更多批次数据");
        }
        return batches.get(currentBatchIndex++);
    }
    
    /**
     * 重置批次索引
     */
    public void reset() {
        currentBatchIndex = 0;
    }
    
    /**
     * 获取批次总数
     * 
     * @return 批次数量
     */
    public int getBatchCount() {
        return batches.size();
    }
    
    /**
     * 获取样本总数
     * 
     * @return 样本数量
     */
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray input;      // [batchSize, seqLen]
        private final NdArray target;     // [batchSize, seqLen]
        private final int batchSize;
        private final int seqLen;
        
        public Batch(NdArray input, NdArray target, int batchSize, int seqLen) {
            this.input = input;
            this.target = target;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
        
        public NdArray getInput() {
            return input;
        }
        
        public NdArray getTarget() {
            return target;
        }
        
        public int getBatchSize() {
            return batchSize;
        }
        
        public int getSeqLen() {
            return seqLen;
        }
    }
}
