package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;

/**
 * 旋转位置编码 (Rotary Position Embedding, RoPE)
 * <p>
 * RoPE 通过旋转矩阵对 Q、K 向量进行位置编码，使模型能够捕捉相对位置信息。
 * 相比传统的绝对位置编码，RoPE 具有更好的长度外推能力。
 * </p>
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class RotaryPositionEmbedding extends Module {

    private final int dim;           // 特征维度(必须是偶数)
    private final int maxSeqLen;     // 最大序列长度
    private final float theta;       // 频率基数(默认 10000)
    
    // 预计算的 cos 和 sin 缓存
    private NdArray cosCache;
    private NdArray sinCache;

    /**
     * 构造 RoPE 位置编码
     *
     * @param dim        特征维度(必须是偶数)
     * @param maxSeqLen  最大序列长度
     * @param theta      频率基数(默认 10000)
     */
    public RotaryPositionEmbedding(int dim, int maxSeqLen, float theta) {
        super("RoPE");
        
        if (dim % 2 != 0) {
            throw new IllegalArgumentException("dim must be even, got: " + dim);
        }
        
        this.dim = dim;
        this.maxSeqLen = maxSeqLen;
        this.theta = theta;
        
        // 预计算 cos 和 sin 值
        precomputeFreqsCis();
    }

    /**
     * 构造 RoPE 位置编码(使用默认 theta=10000)
     *
     * @param dim        特征维度
     * @param maxSeqLen  最大序列长度
     */
    public RotaryPositionEmbedding(int dim, int maxSeqLen) {
        this(dim, maxSeqLen, 10000.0f);
    }

    /**
     * 预计算频率的 cos 和 sin 值
     * <p>
     * 计算公式:
     * freq[i] = 1.0 / (theta ^ (2i / dim)), i = 0, 1, ..., dim/2-1
     * 对于每个位置 pos:
     *   cos_cache[pos, i] = cos(pos * freq[i])
     *   sin_cache[pos, i] = sin(pos * freq[i])
     * </p>
     */
    private void precomputeFreqsCis() {
        int halfDim = dim / 2;
        
        // 计算频率: freq[i] = 1.0 / (theta ^ (2i / dim))
        float[] freqs = new float[halfDim];
        for (int i = 0; i < halfDim; i++) {
            freqs[i] = (float) (1.0 / Math.pow(theta, (2.0 * i) / dim));
        }
        
        // 为每个位置计算 cos 和 sin
        float[] cosData = new float[maxSeqLen * halfDim];
        float[] sinData = new float[maxSeqLen * halfDim];
        
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < halfDim; i++) {
                float angle = pos * freqs[i];
                cosData[pos * halfDim + i] = (float) Math.cos(angle);
                sinData[pos * halfDim + i] = (float) Math.sin(angle);
            }
        }
        
        // 缓存为 NdArray: [maxSeqLen, halfDim]
        this.cosCache = NdArray.of(cosData, Shape.of(maxSeqLen, halfDim));
        this.sinCache = NdArray.of(sinData, Shape.of(maxSeqLen, halfDim));
        
        // 注册为缓冲区(不参与训练)
        registerBuffer("cos_cache", this.cosCache);
        registerBuffer("sin_cache", this.sinCache);
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("RoPE requires at least one input");
        }
        
        Variable x = inputs[0];
        int startPos = inputs.length > 1 ? (int) inputs[1].getValue().getArray()[0] : 0;
        
        return new Variable(applyRotaryEmbedding(x.getValue(), startPos));
    }

    /**
     * 应用旋转位置编码
     *
     * @param x        输入张量 [batch_size, seq_len, dim] 或 [batch_size, num_heads, seq_len, head_dim]
     * @param startPos 起始位置(用于增量推理)
     * @return 应用 RoPE 后的张量
     */
    private NdArray applyRotaryEmbedding(NdArray x, int startPos) {
        int[] shape = x.getShape().getShapeDims();
        
        // 支持两种输入形状
        boolean is3D = (shape.length == 3);  // [B, L, D]
        boolean is4D = (shape.length == 4);  // [B, H, L, D]
        
        if (!is3D && !is4D) {
            throw new IllegalArgumentException(
                "Input must be 3D [B,L,D] or 4D [B,H,L,D], got: " + x.getShape()
            );
        }
        
        int batchSize, seqLen, featureDim, numHeads = 1;
        
        if (is3D) {
            batchSize = shape[0];
            seqLen = shape[1];
            featureDim = shape[2];
        } else {  // is4D
            batchSize = shape[0];
            numHeads = shape[1];
            seqLen = shape[2];
            featureDim = shape[3];
        }
        
        if (featureDim != dim) {
            throw new IllegalArgumentException(
                "Feature dim mismatch: expected " + dim + ", got " + featureDim
            );
        }
        
        if (startPos + seqLen > maxSeqLen) {
            throw new IllegalArgumentException(
                "Sequence too long: startPos=" + startPos + ", seqLen=" + seqLen + 
                ", maxSeqLen=" + maxSeqLen
            );
        }
        
        // 提取对应位置的 cos 和 sin
        float[] xData = x.getArray();
        float[] cosData = cosCache.getArray();
        float[] sinData = sinCache.getArray();
        
        int halfDim = dim / 2;
        float[] output = new float[xData.length];
        
        // 应用旋转变换
        if (is3D) {
            for (int b = 0; b < batchSize; b++) {
                for (int s = 0; s < seqLen; s++) {
                    int pos = startPos + s;
                    applyRotation(xData, output, b * seqLen * dim + s * dim,
                                cosData, sinData, pos * halfDim, halfDim);
                }
            }
            return NdArray.of(output, Shape.of(batchSize, seqLen, dim));
        } else {  // is4D
            for (int b = 0; b < batchSize; b++) {
                for (int h = 0; h < numHeads; h++) {
                    for (int s = 0; s < seqLen; s++) {
                        int pos = startPos + s;
                        int offset = ((b * numHeads + h) * seqLen + s) * dim;
                        applyRotation(xData, output, offset,
                                    cosData, sinData, pos * halfDim, halfDim);
                    }
                }
            }
            return NdArray.of(output, Shape.of(batchSize, numHeads, seqLen, dim));
        }
    }

    /**
     * 对单个向量应用旋转
     * <p>
     * 旋转公式:
     * x_rotated[2i]   = x[2i] * cos - x[2i+1] * sin
     * x_rotated[2i+1] = x[2i] * sin + x[2i+1] * cos
     * </p>
     *
     * @param xData     输入数据
     * @param output    输出数据
     * @param xOffset   输入偏移
     * @param cosData   cos 数据
     * @param sinData   sin 数据
     * @param freqOffset 频率偏移
     * @param halfDim   半维度
     */
    private void applyRotation(float[] xData, float[] output, int xOffset,
                              float[] cosData, float[] sinData, int freqOffset, int halfDim) {
        for (int i = 0; i < halfDim; i++) {
            float x0 = xData[xOffset + 2 * i];
            float x1 = xData[xOffset + 2 * i + 1];
            float cos = cosData[freqOffset + i];
            float sin = sinData[freqOffset + i];
            
            output[xOffset + 2 * i]     = x0 * cos - x1 * sin;
            output[xOffset + 2 * i + 1] = x0 * sin + x1 * cos;
        }
    }

    @Override
    public String extraRepr() {
        return String.format("dim=%d, maxSeqLen=%d, theta=%.1f", dim, maxSeqLen, theta);
    }
}
