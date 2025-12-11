package io.leavesfly.tinyai.qwen3.training;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.qwen3.Qwen3Model;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Qwen3推理器
 * 
 * 提供多种文本生成策略：
 * - Greedy：贪婪解码
 * - Top-K：Top-K采样
 * - Top-P：核采样(Nucleus Sampling)
 * - Temperature：温度采样
 * 
 * @author leavesfly
 * @version 1.0
 */
public class Qwen3Inference {
    
    private final Qwen3Model model;
    private final Random random;
    
    /**
     * 生成策略
     */
    public enum Strategy {
        GREEDY,      // 贪婪解码
        TOP_K,       // Top-K采样
        TOP_P,       // 核采样
        TEMPERATURE  // 温度采样
    }
    
    public Qwen3Inference(Qwen3Model model) {
        this.model = model;
        this.random = new Random(42);
    }
    
    /**
     * 贪婪解码生成
     * 
     * @param inputIds 输入token IDs
     * @param maxNewTokens 最大生成token数
     * @return 生成的token IDs
     */
    public int[] generateGreedy(int[] inputIds, int maxNewTokens) {
        List<Integer> tokens = new ArrayList<>();
        for (int id : inputIds) {
            tokens.add(id);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int nextToken = predictNext(tokens, Strategy.GREEDY, 0.0f, 0, 0.0f);
            if (nextToken == getEosToken()) {
                break;
            }
            tokens.add(nextToken);
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * Top-K采样生成
     * 
     * @param inputIds 输入token IDs
     * @param maxNewTokens 最大生成token数
     * @param topK Top-K值
     * @return 生成的token IDs
     */
    public int[] generateTopK(int[] inputIds, int maxNewTokens, int topK) {
        List<Integer> tokens = new ArrayList<>();
        for (int id : inputIds) {
            tokens.add(id);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int nextToken = predictNext(tokens, Strategy.TOP_K, 0.0f, topK, 0.0f);
            if (nextToken == getEosToken()) {
                break;
            }
            tokens.add(nextToken);
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * Top-P (Nucleus)采样生成
     * 
     * @param inputIds 输入token IDs
     * @param maxNewTokens 最大生成token数
     * @param topP Top-P值(累积概率阈值)
     * @return 生成的token IDs
     */
    public int[] generateTopP(int[] inputIds, int maxNewTokens, float topP) {
        List<Integer> tokens = new ArrayList<>();
        for (int id : inputIds) {
            tokens.add(id);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int nextToken = predictNext(tokens, Strategy.TOP_P, 0.0f, 0, topP);
            if (nextToken == getEosToken()) {
                break;
            }
            tokens.add(nextToken);
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * Temperature采样生成
     * 
     * @param inputIds 输入token IDs
     * @param maxNewTokens 最大生成token数
     * @param temperature 温度参数
     * @return 生成的token IDs
     */
    public int[] generateTemperature(int[] inputIds, int maxNewTokens, float temperature) {
        List<Integer> tokens = new ArrayList<>();
        for (int id : inputIds) {
            tokens.add(id);
        }
        
        for (int i = 0; i < maxNewTokens; i++) {
            int nextToken = predictNext(tokens, Strategy.TEMPERATURE, temperature, 0, 0.0f);
            if (nextToken == getEosToken()) {
                break;
            }
            tokens.add(nextToken);
        }
        
        return tokens.stream().mapToInt(Integer::intValue).toArray();
    }
    
    /**
     * 预测下一个token
     */
    private int predictNext(List<Integer> tokens, Strategy strategy, 
                           float temperature, int topK, float topP) {
        // 准备输入
        float[][] inputData = new float[1][tokens.size()];
        for (int i = 0; i < tokens.size(); i++) {
            inputData[0][i] = tokens.get(i);
        }
        
        NdArray inputIds = NdArray.of(inputData);
        Variable inputVar = new Variable(inputIds);
        
        // 前向传播
        Variable logits = model.forward(inputVar);
        
        // 获取最后一个位置的logits
        NdArray logitsArray = logits.getValue();
        int vocabSize = logitsArray.getShape().getDimension(2);
        float[] lastLogits = new float[vocabSize];
        
        for (int i = 0; i < vocabSize; i++) {
            lastLogits[i] = logitsArray.get(0, tokens.size() - 1, i);
        }
        
        // 根据策略选择token
        switch (strategy) {
            case GREEDY:
                return argmax(lastLogits);
            case TOP_K:
                return sampleTopK(lastLogits, topK);
            case TOP_P:
                return sampleTopP(lastLogits, topP);
            case TEMPERATURE:
                return sampleTemperature(lastLogits, temperature);
            default:
                return argmax(lastLogits);
        }
    }
    
    /**
     * 找到最大值的索引
     */
    private int argmax(float[] logits) {
        int maxIdx = 0;
        float maxVal = logits[0];
        for (int i = 1; i < logits.length; i++) {
            if (logits[i] > maxVal) {
                maxVal = logits[i];
                maxIdx = i;
            }
        }
        return maxIdx;
    }
    
    /**
     * Top-K采样
     */
    private int sampleTopK(float[] logits, int k) {
        // 简化实现：softmax后采样
        float[] probs = softmax(logits);
        
        // 找到Top-K
        int[] topKIndices = getTopKIndices(probs, k);
        
        // 从Top-K中采样
        float sum = 0.0f;
        for (int idx : topKIndices) {
            sum += probs[idx];
        }
        
        float rand = random.nextFloat() * sum;
        float cumSum = 0.0f;
        for (int idx : topKIndices) {
            cumSum += probs[idx];
            if (cumSum >= rand) {
                return idx;
            }
        }
        
        return topKIndices[0];
    }
    
    /**
     * Top-P采样
     */
    private int sampleTopP(float[] logits, float p) {
        float[] probs = softmax(logits);
        
        // 按概率排序
        Integer[] indices = new Integer[probs.length];
        for (int i = 0; i < probs.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
        
        // 累积到p
        float cumSum = 0.0f;
        List<Integer> selectedIndices = new ArrayList<>();
        for (int idx : indices) {
            cumSum += probs[idx];
            selectedIndices.add(idx);
            if (cumSum >= p) {
                break;
            }
        }
        
        // 从选中的采样
        float sum = 0.0f;
        for (int idx : selectedIndices) {
            sum += probs[idx];
        }
        
        float rand = random.nextFloat() * sum;
        cumSum = 0.0f;
        for (int idx : selectedIndices) {
            cumSum += probs[idx];
            if (cumSum >= rand) {
                return idx;
            }
        }
        
        return selectedIndices.get(0);
    }
    
    /**
     * 温度采样
     */
    private int sampleTemperature(float[] logits, float temperature) {
        // 应用温度
        float[] scaledLogits = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            scaledLogits[i] = logits[i] / temperature;
        }
        
        float[] probs = softmax(scaledLogits);
        
        // 采样
        float rand = random.nextFloat();
        float cumSum = 0.0f;
        for (int i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (cumSum >= rand) {
                return i;
            }
        }
        
        return probs.length - 1;
    }
    
    /**
     * Softmax
     */
    private float[] softmax(float[] logits) {
        float max = logits[0];
        for (float logit : logits) {
            if (logit > max) {
                max = logit;
            }
        }
        
        float sum = 0.0f;
        float[] probs = new float[logits.length];
        for (int i = 0; i < logits.length; i++) {
            probs[i] = (float) Math.exp(logits[i] - max);
            sum += probs[i];
        }
        
        for (int i = 0; i < probs.length; i++) {
            probs[i] /= sum;
        }
        
        return probs;
    }
    
    /**
     * 获取Top-K索引
     */
    private int[] getTopKIndices(float[] probs, int k) {
        Integer[] indices = new Integer[probs.length];
        for (int i = 0; i < probs.length; i++) {
            indices[i] = i;
        }
        java.util.Arrays.sort(indices, (a, b) -> Float.compare(probs[b], probs[a]));
        
        int[] topK = new int[Math.min(k, probs.length)];
        for (int i = 0; i < topK.length; i++) {
            topK[i] = indices[i];
        }
        
        return topK;
    }
    
    /**
     * 获取EOS token ID
     */
    private int getEosToken() {
        return model.getConfig().getEosTokenId();
    }
}
