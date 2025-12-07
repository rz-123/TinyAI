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
import org.json.*;

/**
 * SFT(Supervised Fine-Tuning)数据集
 * 
 * 支持指令微调数据格式:
 * {
 *   "instruction": "用户指令",
 *   "input": "可选的输入",
 *   "output": "期望的输出"
 * }
 * 
 * @author leavesfly
 * @since 2024
 */
public class SFTDataset implements Serializable {
    
    private static final long serialVersionUID = 1L;
    
    private final MiniMindTokenizer tokenizer;
    private final int maxSeqLen;
    private final int batchSize;
    
    // 对话模板
    private static final String CHAT_TEMPLATE = 
        "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s<|im_end|>";
    
    // 训练样本
    private List<SFTSample> samples;
    private List<Batch> batches;
    private int currentBatchIndex;
    
    /**
     * SFT样本
     */
    public static class SFTSample {
        public final String instruction;
        public final String input;
        public final String output;
        
        public SFTSample(String instruction, String input, String output) {
            this.instruction = instruction;
            this.input = input;
            this.output = output;
        }
        
        public String formatPrompt() {
            if (input != null && !input.isEmpty()) {
                return instruction + "\n" + input;
            }
            return instruction;
        }
    }
    
    /**
     * 构造函数
     */
    public SFTDataset(MiniMindTokenizer tokenizer, int maxSeqLen, int batchSize) {
        this.tokenizer = tokenizer;
        this.maxSeqLen = maxSeqLen;
        this.batchSize = batchSize;
        this.samples = new ArrayList<>();
        this.batches = new ArrayList<>();
        this.currentBatchIndex = 0;
    }
    
    /**
     * 从JSONL文件加载数据
     * 
     * @param filePath 文件路径
     * @throws IOException IO异常
     */
    public void loadFromJsonl(String filePath) throws IOException {
        Path path = Paths.get(filePath);
        if (!Files.exists(path)) {
            throw new FileNotFoundException("数据文件不存在: " + filePath);
        }
        
        samples.clear();
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        
        for (String line : lines) {
            if (line.trim().isEmpty()) {
                continue;
            }
            
            try {
                JSONObject json = new JSONObject(line);
                String instruction = json.optString("instruction", "");
                String input = json.optString("input", "");
                String output = json.optString("output", "");
                
                if (!instruction.isEmpty() && !output.isEmpty()) {
                    samples.add(new SFTSample(instruction, input, output));
                }
            } catch (Exception e) {
                System.err.println("解析JSON失败: " + line);
            }
        }
        
        System.out.println("SFT数据加载完成,共 " + samples.size() + " 个样本");
    }
    
    /**
     * 添加单个样本
     */
    public void addSample(String instruction, String input, String output) {
        samples.add(new SFTSample(instruction, input, output));
    }
    
    /**
     * 准备批次数据
     */
    public void prepare(boolean shuffle) {
        if (samples.isEmpty()) {
            throw new IllegalStateException("数据集为空");
        }
        
        batches.clear();
        currentBatchIndex = 0;
        
        List<SFTSample> workingSamples = new ArrayList<>(samples);
        if (shuffle) {
            Collections.shuffle(workingSamples);
        }
        
        // 创建批次
        for (int i = 0; i < workingSamples.size(); i += batchSize) {
            int endIdx = Math.min(i + batchSize, workingSamples.size());
            List<SFTSample> batchSamples = workingSamples.subList(i, endIdx);
            batches.add(createBatch(batchSamples));
        }
        
        System.out.println("SFT批次准备完成,共 " + batches.size() + " 个批次");
    }
    
    /**
     * 创建批次
     */
    private Batch createBatch(List<SFTSample> batchSamples) {
        int actualBatchSize = batchSamples.size();
        
        List<int[]> inputIds = new ArrayList<>();
        List<int[]> labelIds = new ArrayList<>();
        List<int[]> lossMasks = new ArrayList<>();
        
        int maxLen = 0;
        
        for (SFTSample sample : batchSamples) {
            // 构建对话文本
            String prompt = sample.formatPrompt();
            String response = sample.output;
            String fullText = String.format(CHAT_TEMPLATE, prompt, response);
            
            // 编码整个对话
            List<Integer> fullTokenIds = tokenizer.encode(fullText, false, false);
            
            // 编码用户部分(用于确定掩码位置)
            String userPart = String.format("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", prompt);
            List<Integer> userTokenIds = tokenizer.encode(userPart, false, false);
            int promptLen = userTokenIds.size();
            
            // 截断到最大长度
            if (fullTokenIds.size() > maxSeqLen) {
                fullTokenIds = fullTokenIds.subList(0, maxSeqLen);
            }
            
            int seqLen = fullTokenIds.size();
            maxLen = Math.max(maxLen, seqLen);
            
            // 输入: 前n-1个token
            int[] input = new int[seqLen - 1];
            for (int i = 0; i < seqLen - 1; i++) {
                input[i] = fullTokenIds.get(i);
            }
            
            // 标签: 后n-1个token
            int[] labels = new int[seqLen - 1];
            for (int i = 0; i < seqLen - 1; i++) {
                labels[i] = fullTokenIds.get(i + 1);
            }
            
            // 损失掩码: 只计算assistant部分的损失
            int[] mask = new int[seqLen - 1];
            for (int i = 0; i < seqLen - 1; i++) {
                if (i >= promptLen - 1) {
                    mask[i] = 1;  // 计算损失
                } else {
                    mask[i] = 0;  // 忽略损失
                }
            }
            
            inputIds.add(input);
            labelIds.add(labels);
            lossMasks.add(mask);
        }
        
        // 填充到maxLen-1
        int padTokenId = tokenizer.getVocabulary().getPadTokenId();
        int paddedLen = maxLen - 1;
        
        int[][] inputData = new int[actualBatchSize][paddedLen];
        int[][] labelData = new int[actualBatchSize][paddedLen];
        int[][] maskData = new int[actualBatchSize][paddedLen];
        
        for (int i = 0; i < actualBatchSize; i++) {
            int[] input = inputIds.get(i);
            int[] labels = labelIds.get(i);
            int[] mask = lossMasks.get(i);
            
            System.arraycopy(input, 0, inputData[i], 0, input.length);
            System.arraycopy(labels, 0, labelData[i], 0, labels.length);
            System.arraycopy(mask, 0, maskData[i], 0, mask.length);
            
            // 填充部分
            for (int j = input.length; j < paddedLen; j++) {
                inputData[i][j] = padTokenId;
                labelData[i][j] = padTokenId;
                maskData[i][j] = 0;
            }
        }
        
        // 转换为NdArray
        NdArray inputArray = createNdArray(inputData, actualBatchSize, paddedLen);
        NdArray labelArray = createNdArray(labelData, actualBatchSize, paddedLen);
        NdArray maskArray = createNdArray(maskData, actualBatchSize, paddedLen);
        
        return new Batch(inputArray, labelArray, maskArray, actualBatchSize, paddedLen);
    }
    
    /**
     * 创建NdArray
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
    
    public boolean hasNextBatch() {
        return currentBatchIndex < batches.size();
    }
    
    public Batch getNextBatch() {
        if (!hasNextBatch()) {
            throw new NoSuchElementException("没有更多批次数据");
        }
        return batches.get(currentBatchIndex++);
    }
    
    public void reset() {
        currentBatchIndex = 0;
    }
    
    public int getBatchCount() {
        return batches.size();
    }
    
    public int getSampleCount() {
        return samples.size();
    }
    
    /**
     * SFT批次数据
     */
    public static class Batch {
        private final NdArray input;
        private final NdArray labels;
        private final NdArray lossMask;  // 损失掩码,1表示计算损失,0表示忽略
        private final int batchSize;
        private final int seqLen;
        
        public Batch(NdArray input, NdArray labels, NdArray lossMask,
                     int batchSize, int seqLen) {
            this.input = input;
            this.labels = labels;
            this.lossMask = lossMask;
            this.batchSize = batchSize;
            this.seqLen = seqLen;
        }
        
        public NdArray getInput() {
            return input;
        }
        
        public NdArray getLabels() {
            return labels;
        }
        
        public NdArray getLossMask() {
            return lossMask;
        }
        
        public int getBatchSize() {
            return batchSize;
        }
        
        public int getSeqLen() {
            return seqLen;
        }
    }
}
