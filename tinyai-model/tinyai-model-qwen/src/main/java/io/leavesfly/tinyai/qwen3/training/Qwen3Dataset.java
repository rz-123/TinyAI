package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Qwen3数据集类
 * 
 * 支持预训练和后训练（微调）两种模式的数据加载
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Dataset {
    
    private final List<int[]> sequences;      // 完整序列
    private final List<String> prompts;       // 提示词（微调用）
    private final List<String> responses;     // 响应（微调用）
    private final int maxSeqLength;
    private final int batchSize;
    private final boolean shuffle;
    
    private int currentIndex;
    private List<Integer> indices;
    
    /**
     * 构造函数（预训练模式）
     * 
     * @param sequences token序列列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public Qwen3Dataset(List<int[]> sequences, int maxSeqLength, 
                        int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.prompts = new ArrayList<>();
        this.responses = new ArrayList<>();
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 构造函数（后训练/微调模式）
     * 
     * @param sequences token序列列表
     * @param prompts 提示词列表
     * @param responses 响应列表
     * @param maxSeqLength 最大序列长度
     * @param batchSize 批次大小
     * @param shuffle 是否打乱数据
     */
    public Qwen3Dataset(List<int[]> sequences, List<String> prompts,
                        List<String> responses, int maxSeqLength,
                        int batchSize, boolean shuffle) {
        this.sequences = sequences;
        this.prompts = prompts;
        this.responses = responses;
        this.maxSeqLength = maxSeqLength;
        this.batchSize = batchSize;
        this.shuffle = shuffle;
        this.currentIndex = 0;
        initIndices();
    }
    
    /**
     * 初始化索引
     */
    private void initIndices() {
        indices = new ArrayList<>();
        for (int i = 0; i < sequences.size(); i++) {
            indices.add(i);
        }
    }
    
    /**
     * 准备数据集（打乱或重置）
     * 
     * @param shouldShuffle 是否打乱
     */
    public void prepare(boolean shouldShuffle) {
        if (shouldShuffle && shuffle) {
            Collections.shuffle(indices, new Random());
        }
        currentIndex = 0;
    }
    
    /**
     * 是否还有下一批数据
     */
    public boolean hasNext() {
        return currentIndex < sequences.size();
    }
    
    /**
     * 获取下一批数据
     * 
     * @return 批次数据
     */
    public Batch nextBatch() {
        int endIndex = Math.min(currentIndex + batchSize, sequences.size());
        int actualBatchSize = endIndex - currentIndex;
        
        // 准备输入和目标
        float[][] inputData = new float[actualBatchSize][maxSeqLength];
        float[][] targetData = new float[actualBatchSize][maxSeqLength];
        String[] promptTexts = new String[actualBatchSize];
        String[] responseTexts = new String[actualBatchSize];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int dataIndex = indices.get(currentIndex + i);
            int[] sequence = sequences.get(dataIndex);
            
            // 填充或截断序列
            int seqLen = Math.min(sequence.length, maxSeqLength);
            
            // 输入：序列的前n-1个token
            for (int j = 0; j < seqLen - 1; j++) {
                inputData[i][j] = sequence[j];
            }
            
            // 目标：序列的后n-1个token（用于语言建模）
            for (int j = 1; j < seqLen; j++) {
                targetData[i][j - 1] = sequence[j];
            }
            
            // 微调数据
            if (!prompts.isEmpty() && dataIndex < prompts.size()) {
                promptTexts[i] = prompts.get(dataIndex);
            }
            if (!responses.isEmpty() && dataIndex < responses.size()) {
                responseTexts[i] = responses.get(dataIndex);
            }
        }
        
        currentIndex = endIndex;
        
        NdArray inputIds = NdArray.of(inputData);
        NdArray targetIds = NdArray.of(targetData);
        
        return new Batch(inputIds, targetIds, promptTexts, responseTexts);
    }
    
    /**
     * 重置数据集
     */
    public void reset() {
        currentIndex = 0;
    }
    
    /**
     * 获取样本数量
     */
    public int getSampleCount() {
        return sequences.size();
    }
    
    /**
     * 获取批次数量
     */
    public int getBatchCount() {
        return (sequences.size() + batchSize - 1) / batchSize;
    }
    
    /**
     * 创建演示数据集（用于测试）
     */
    public static Qwen3Dataset createDemoDataset(int vocabSize, int numSamples, 
                                                  int maxSeqLength, int batchSize) {
        List<int[]> sequences = new ArrayList<>();
        Random random = new Random(42);
        
        for (int i = 0; i < numSamples; i++) {
            int seqLen = random.nextInt(maxSeqLength - 10) + 10;
            int[] sequence = new int[seqLen];
            for (int j = 0; j < seqLen; j++) {
                sequence[j] = random.nextInt(vocabSize);
            }
            sequences.add(sequence);
        }
        
        return new Qwen3Dataset(sequences, maxSeqLength, batchSize, true);
    }
    
    /**
     * 批次数据类
     */
    public static class Batch {
        private final NdArray inputIds;
        private final NdArray targetIds;
        private final String[] prompts;
        private final String[] responses;
        
        public Batch(NdArray inputIds, NdArray targetIds, 
                    String[] prompts, String[] responses) {
            this.inputIds = inputIds;
            this.targetIds = targetIds;
            this.prompts = prompts;
            this.responses = responses;
        }
        
        public NdArray getInputIds() {
            return inputIds;
        }
        
        public NdArray getTargetIds() {
            return targetIds;
        }
        
        public String[] getPrompts() {
            return prompts;
        }
        
        public String[] getResponses() {
            return responses;
        }
        
        public int getBatchSize() {
            return inputIds.getShape().getDimension(0);
        }
        
        public int getSeqLength() {
            return inputIds.getShape().getDimension(1);
        }
    }
}
