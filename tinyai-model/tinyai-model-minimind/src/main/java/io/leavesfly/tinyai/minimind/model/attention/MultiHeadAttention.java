package io.leavesfly.tinyai.minimind.model.attention;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.minimind.model.embedding.RotaryPositionEmbedding;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Linear;

/**
 * 多头注意力机制（Multi-Head Attention）
 * <p>
 * 实现功能：
 * - Q、K、V 投影使用 V2 Linear 层
 * - 集成 RoPE 旋转位置编码
 * - 支持因果掩码(Causal Mask)
 * - 支持 KV-Cache 增量推理
 * - Scaled Dot-Product Attention
 * <p>
 * 计算流程：
 * 1. Q = X @ W_Q, K = X @ W_K, V = X @ W_V
 * 2. 应用 RoPE 位置编码到 Q、K
 * 3. 多头分割：reshape 为 [batch, numHeads, seqLen, headDim]
 * 4. 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
 * 5. 应用因果掩码
 * 6. Softmax 归一化
 * 7. 应用注意力权重：output = scores @ V
 * 8. 多头合并：reshape 为 [batch, seqLen, hiddenSize]
 * 9. 输出投影：output @ W_O
 *
 * @author leavesfly
 * @version 1.0
 */
public class MultiHeadAttention extends Module {

    /**
     * 隐藏层维度
     */
    private final int hiddenSize;

    /**
     * 注意力头数
     */
    private final int numHeads;

    /**
     * 每个头的维度
     */
    private final int headDim;

    /**
     * Query 投影层
     */
    private final Linear queryProj;

    /**
     * Key 投影层
     */
    private final Linear keyProj;

    /**
     * Value 投影层
     */
    private final Linear valueProj;

    /**
     * 输出投影层
     */
    private final Linear outputProj;

    /**
     * RoPE 位置编码
     */
    private final RotaryPositionEmbedding rope;

    /**
     * Dropout 比例
     */
    private final float dropoutRate;

    /**
     * 是否处于训练模式
     */
    private boolean training = true;

    /**
     * 构造多头注意力层
     *
     * @param name        层名称
     * @param hiddenSize  隐藏层维度
     * @param numHeads    注意力头数
     * @param maxSeqLen   最大序列长度
     * @param dropoutRate Dropout 比例
     */
    public MultiHeadAttention(String name, int hiddenSize, int numHeads, int maxSeqLen, float dropoutRate) {
        super(name);

        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException("hiddenSize must be divisible by numHeads");
        }

        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.headDim = hiddenSize / numHeads;
        this.dropoutRate = dropoutRate;

        // 创建投影层（使用 V2 Linear）
        this.queryProj = new Linear("query_proj", hiddenSize, hiddenSize, false);
        this.keyProj = new Linear("key_proj", hiddenSize, hiddenSize, false);
        this.valueProj = new Linear("value_proj", hiddenSize, hiddenSize, false);
        this.outputProj = new Linear("output_proj", hiddenSize, hiddenSize, false);

        // 注册子模块
        registerModule("query_proj", queryProj);
        registerModule("key_proj", keyProj);
        registerModule("value_proj", valueProj);
        registerModule("output_proj", outputProj);

        // 创建 RoPE 位置编码
        this.rope = new RotaryPositionEmbedding(headDim, maxSeqLen);
        registerModule("rope", rope);

        // 初始化参数
        init();
    }

    @Override
    public Variable forward(Variable... inputs) {
        // 默认不使用 KV-Cache
        return forwardWithCache(inputs[0], null, 0);
    }

    /**
     * 带 KV-Cache 的前向传播
     *
     * @param x        输入 Variable
     * @param kvCache  KV-Cache 对象（可为 null）
     * @param startPos 起始位置（用于 RoPE 和因果掩码）
     * @return 输出 Variable
     */
    public Variable forwardWithCache(Variable x, KVCache kvCache, int startPos) {

        // 1. Q、K、V 投影
        Variable Q = queryProj.forward(x);
        Variable K = keyProj.forward(x);
        Variable V = valueProj.forward(x);

        // 获取输入形状
        int[] qShape = Q.getValue().getShape().getShapeDims();
        int batchSize = qShape[0];
        int seqLen = qShape[1];

        // 2. 应用 RoPE 位置编码
        Q = rope.forward(Q, new Variable(NdArray.of(new float[]{startPos})));
        K = rope.forward(K, new Variable(NdArray.of(new float[]{startPos})));

        // 3. 多头分割：[batch, seqLen, hiddenSize] -> [batch, numHeads, seqLen, headDim]
        NdArray qSplit = reshapeMultiHead(Q.getValue(), batchSize, seqLen);
        NdArray kSplit = reshapeMultiHead(K.getValue(), batchSize, seqLen);
        NdArray vSplit = reshapeMultiHead(V.getValue(), batchSize, seqLen);

        // 4. KV-Cache 处理
        if (kvCache != null) {
            NdArray[] updated = kvCache.update(kSplit, vSplit);
            kSplit = updated[0];
            vSplit = updated[1];
        }

        int kvSeqLen = kSplit.getShape().getShapeDims()[2];

        // 5. 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
        NdArray scores = computeAttentionScores(qSplit, kSplit, batchSize, seqLen, kvSeqLen);

        // 6. 应用因果掩码
        if (training || kvCache == null) {
            scores = applyCausalMask(scores, batchSize, seqLen, kvSeqLen, startPos);
        }

        // 7. Softmax 归一化
        NdArray attnWeights = softmax(scores, batchSize, numHeads, seqLen, kvSeqLen);

        // 8. Dropout（训练时）
        if (training && dropoutRate > 0) {
            attnWeights = applyDropout(attnWeights, dropoutRate);
        }

        // 9. 应用注意力权重：output = attnWeights @ V
        NdArray attended = applyAttentionWeights(attnWeights, vSplit, batchSize, seqLen);

        // 10. 多头合并：[batch, numHeads, seqLen, headDim] -> [batch, seqLen, hiddenSize]
        NdArray merged = mergeMultiHead(attended, batchSize, seqLen);

        // 11. 输出投影
        Variable output = outputProj.forward(new Variable(merged));

        return output;
    }

    /**
     * 多头分割：[batch, seqLen, hiddenSize] -> [batch, numHeads, seqLen, headDim]
     */
    private NdArray reshapeMultiHead(NdArray input, int batchSize, int seqLen) {
        float[] data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) input).buffer;
        float[] result = new float[batchSize * numHeads * seqLen * headDim];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        int srcIdx = (b * seqLen + s) * hiddenSize + h * headDim + d;
                        int dstIdx = ((b * numHeads + h) * seqLen + s) * headDim + d;
                        result[dstIdx] = data[srcIdx];
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batchSize, numHeads, seqLen, headDim));
    }

    /**
     * 计算注意力分数：scores = (Q @ K^T) / sqrt(headDim)
     */
    private NdArray computeAttentionScores(NdArray Q, NdArray K, int batchSize, int qSeqLen, int kvSeqLen) {
        float[] qData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) Q).buffer;
        float[] kData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) K).buffer;
        float[] scores = new float[batchSize * numHeads * qSeqLen * kvSeqLen];

        float scale = (float) (1.0 / Math.sqrt(headDim));

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < qSeqLen; i++) {
                    for (int j = 0; j < kvSeqLen; j++) {
                        float sum = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            int qIdx = ((b * numHeads + h) * qSeqLen + i) * headDim + d;
                            int kIdx = ((b * numHeads + h) * kvSeqLen + j) * headDim + d;
                            sum += qData[qIdx] * kData[kIdx];
                        }
                        int scoreIdx = ((b * numHeads + h) * qSeqLen + i) * kvSeqLen + j;
                        scores[scoreIdx] = sum * scale;
                    }
                }
            }
        }

        return NdArray.of(scores, Shape.of(batchSize, numHeads, qSeqLen, kvSeqLen));
    }

    /**
     * 应用因果掩码（禁止看到未来 token）
     */
    private NdArray applyCausalMask(NdArray scores, int batchSize, int qSeqLen, int kvSeqLen, int startPos) {
        float[] scoresData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) scores).buffer;
        float[] result = new float[scoresData.length];
        System.arraycopy(scoresData, 0, result, 0, scoresData.length);

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < qSeqLen; i++) {
                    for (int j = 0; j < kvSeqLen; j++) {
                        // 因果掩码：当前位置只能看到之前的位置
                        int currentPos = startPos + i;
                        if (j > currentPos) {
                            int idx = ((b * numHeads + h) * qSeqLen + i) * kvSeqLen + j;
                            result[idx] = Float.NEGATIVE_INFINITY;
                        }
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batchSize, numHeads, qSeqLen, kvSeqLen));
    }

    /**
     * Softmax 归一化（在最后一个维度上）
     */
    private NdArray softmax(NdArray input, int batchSize, int numHeads, int qSeqLen, int kvSeqLen) {
        float[] data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) input).buffer;
        float[] result = new float[data.length];

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < qSeqLen; i++) {
                    int offset = ((b * numHeads + h) * qSeqLen + i) * kvSeqLen;

                    // 找最大值（数值稳定性）
                    float maxVal = Float.NEGATIVE_INFINITY;
                    for (int j = 0; j < kvSeqLen; j++) {
                        maxVal = Math.max(maxVal, data[offset + j]);
                    }

                    // 计算 exp 和 sum
                    float sum = 0.0f;
                    for (int j = 0; j < kvSeqLen; j++) {
                        result[offset + j] = (float) Math.exp(data[offset + j] - maxVal);
                        sum += result[offset + j];
                    }

                    // 归一化
                    for (int j = 0; j < kvSeqLen; j++) {
                        result[offset + j] /= sum;
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batchSize, numHeads, qSeqLen, kvSeqLen));
    }

    /**
     * 应用 Dropout
     */
    private NdArray applyDropout(NdArray input, float rate) {
        float[] data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) input).buffer;
        float[] result = new float[data.length];
        float scale = 1.0f / (1.0f - rate);

        for (int i = 0; i < data.length; i++) {
            if (Math.random() > rate) {
                result[i] = data[i] * scale;
            } else {
                result[i] = 0.0f;
            }
        }

        return NdArray.of(result, input.getShape());
    }

    /**
     * 应用注意力权重：output = attnWeights @ V
     */
    private NdArray applyAttentionWeights(NdArray attnWeights, NdArray V, int batchSize, int seqLen) {
        float[] weightsData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) attnWeights).buffer;
        float[] vData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) V).buffer;
        int kvSeqLen = V.getShape().getShapeDims()[2];

        float[] result = new float[batchSize * numHeads * seqLen * headDim];

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int i = 0; i < seqLen; i++) {
                    for (int d = 0; d < headDim; d++) {
                        float sum = 0.0f;
                        for (int j = 0; j < kvSeqLen; j++) {
                            int weightIdx = ((b * numHeads + h) * seqLen + i) * kvSeqLen + j;
                            int vIdx = ((b * numHeads + h) * kvSeqLen + j) * headDim + d;
                            sum += weightsData[weightIdx] * vData[vIdx];
                        }
                        int outIdx = ((b * numHeads + h) * seqLen + i) * headDim + d;
                        result[outIdx] = sum;
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batchSize, numHeads, seqLen, headDim));
    }

    /**
     * 多头合并：[batch, numHeads, seqLen, headDim] -> [batch, seqLen, hiddenSize]
     */
    private NdArray mergeMultiHead(NdArray input, int batchSize, int seqLen) {
        float[] data = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) input).buffer;
        float[] result = new float[batchSize * seqLen * hiddenSize];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int d = 0; d < headDim; d++) {
                        int srcIdx = ((b * numHeads + h) * seqLen + s) * headDim + d;
                        int dstIdx = (b * seqLen + s) * hiddenSize + h * headDim + d;
                        result[dstIdx] = data[srcIdx];
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batchSize, seqLen, hiddenSize));
    }

    /**
     * 设置训练模式
     */
    public void setTraining(boolean training) {
        this.training = training;
    }

    /**
     * 获取注意力头数
     */
    public int getNumHeads() {
        return numHeads;
    }

    /**
     * 获取每个头的维度
     */
    public int getHeadDim() {
        return headDim;
    }
}
