package io.leavesfly.tinyai.minimind.model.attention;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

/**
 * KV-Cache 增量推理缓存管理
 * <p>
 * 功能：
 * - 缓存历史 Key、Value 向量，避免重复计算
 * - 支持增量添加新 token 的 K、V
 * - 动态维护缓存序列长度
 * <p>
 * 应用场景：
 * - 自回归文本生成
 * - 减少重复的注意力计算开销
 *
 * @author leavesfly
 * @version 1.0
 */
public class KVCache {

    /**
     * 缓存的 Key 向量
     * Shape: [batchSize, numHeads, seqLen, headDim]
     */
    private NdArray cachedK;

    /**
     * 缓存的 Value 向量
     * Shape: [batchSize, numHeads, seqLen, headDim]
     */
    private NdArray cachedV;

    /**
     * 当前缓存的序列长度
     */
    private int currentSeqLen;

    /**
     * 批次大小
     */
    private final int batchSize;

    /**
     * 注意力头数
     */
    private final int numHeads;

    /**
     * 每个头的维度
     */
    private final int headDim;

    /**
     * 最大缓存长度
     */
    private final int maxCacheLen;

    /**
     * 构造 KVCache
     *
     * @param batchSize   批次大小
     * @param numHeads    注意力头数
     * @param headDim     每个头的维度
     * @param maxCacheLen 最大缓存序列长度
     */
    public KVCache(int batchSize, int numHeads, int headDim, int maxCacheLen) {
        this.batchSize = batchSize;
        this.numHeads = numHeads;
        this.headDim = headDim;
        this.maxCacheLen = maxCacheLen;
        this.currentSeqLen = 0;
        this.cachedK = null;
        this.cachedV = null;
    }

    /**
     * 更新缓存：添加新的 K、V
     *
     * @param newK 新的 Key 向量，Shape: [batchSize, numHeads, newSeqLen, headDim]
     * @param newV 新的 Value 向量，Shape: [batchSize, numHeads, newSeqLen, headDim]
     * @return 更新后的完整 K、V 数组
     */
    public NdArray[] update(NdArray newK, NdArray newV) {
        if (cachedK == null || cachedV == null) {
            // 首次初始化缓存
            cachedK = newK;
            cachedV = newV;
            currentSeqLen = newK.getShape().getShapeDims()[2];
        } else {
            // 拼接新的 K、V 到缓存
            cachedK = concatenateSeqDim(cachedK, newK);
            cachedV = concatenateSeqDim(cachedV, newV);
            currentSeqLen += newK.getShape().getShapeDims()[2];

            // 如果超出最大长度，截断旧数据
            if (currentSeqLen > maxCacheLen) {
                int excessLen = currentSeqLen - maxCacheLen;
                cachedK = sliceSeqDim(cachedK, excessLen, maxCacheLen);
                cachedV = sliceSeqDim(cachedV, excessLen, maxCacheLen);
                currentSeqLen = maxCacheLen;
            }
        }

        return new NdArray[]{cachedK, cachedV};
    }

    /**
     * 在序列维度(dim=2)上拼接两个 NdArray
     *
     * @param cached 已缓存的数据
     * @param newData 新数据
     * @return 拼接后的数组
     */
    private NdArray concatenateSeqDim(NdArray cached, NdArray newData) {
        int[] cachedShape = cached.getShape().getShapeDims();
        int[] newShape = newData.getShape().getShapeDims();

        int batch = cachedShape[0];
        int heads = cachedShape[1];
        int oldSeqLen = cachedShape[2];
        int newSeqLen = newShape[2];
        int dim = cachedShape[3];

        int totalSeqLen = oldSeqLen + newSeqLen;

        // 创建拼接后的数组
        float[] result = new float[batch * heads * totalSeqLen * dim];
        float[] cachedData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) cached).buffer;
        float[] newDataArr = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) newData).buffer;

        // 按序列维度拼接
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int s = 0; s < oldSeqLen; s++) {
                    for (int d = 0; d < dim; d++) {
                        int srcIdx = ((b * heads + h) * oldSeqLen + s) * dim + d;
                        int dstIdx = ((b * heads + h) * totalSeqLen + s) * dim + d;
                        result[dstIdx] = cachedData[srcIdx];
                    }
                }
                for (int s = 0; s < newSeqLen; s++) {
                    for (int d = 0; d < dim; d++) {
                        int srcIdx = ((b * heads + h) * newSeqLen + s) * dim + d;
                        int dstIdx = ((b * heads + h) * totalSeqLen + (oldSeqLen + s)) * dim + d;
                        result[dstIdx] = newDataArr[srcIdx];
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batch, heads, totalSeqLen, dim));
    }

    /**
     * 在序列维度上切片
     *
     * @param data 原始数据
     * @param start 起始位置
     * @param end 结束位置
     * @return 切片后的数组
     */
    private NdArray sliceSeqDim(NdArray data, int start, int end) {
        int[] shape = data.getShape().getShapeDims();
        int batch = shape[0];
        int heads = shape[1];
        int seqLen = shape[2];
        int dim = shape[3];

        int newSeqLen = end - start;
        float[] result = new float[batch * heads * newSeqLen * dim];
        float[] srcData = ((io.leavesfly.tinyai.ndarr.cpu.NdArrayCpu) data).buffer;

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < heads; h++) {
                for (int s = start; s < end; s++) {
                    for (int d = 0; d < dim; d++) {
                        int srcIdx = ((b * heads + h) * seqLen + s) * dim + d;
                        int dstIdx = ((b * heads + h) * newSeqLen + (s - start)) * dim + d;
                        result[dstIdx] = srcData[srcIdx];
                    }
                }
            }
        }

        return NdArray.of(result, Shape.of(batch, heads, newSeqLen, dim));
    }

    /**
     * 清空缓存
     */
    public void clear() {
        cachedK = null;
        cachedV = null;
        currentSeqLen = 0;
    }

    /**
     * 获取当前缓存的序列长度
     */
    public int getCurrentSeqLen() {
        return currentSeqLen;
    }

    /**
     * 获取缓存的 Key
     */
    public NdArray getCachedK() {
        return cachedK;
    }

    /**
     * 获取缓存的 Value
     */
    public NdArray getCachedV() {
        return cachedV;
    }

    /**
     * 判断缓存是否为空
     */
    public boolean isEmpty() {
        return cachedK == null || cachedV == null;
    }
}
