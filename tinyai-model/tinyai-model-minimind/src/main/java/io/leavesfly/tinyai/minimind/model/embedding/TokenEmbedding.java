package io.leavesfly.tinyai.minimind.model.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * Token 嵌入层
 * <p>
 * 将 Token IDs 转换为密集的向量表示。
 * 实现词汇表查找功能,支持权重共享(与 LM Head 共享嵌入矩阵)。
 * </p>
 *
 * @author TinyAI Team
 * @version 1.0
 */
public class TokenEmbedding extends Module {

    private final int vocabSize;
    private final int embeddingDim;
    private Parameter weight;

    /**
     * 构造 Token 嵌入层
     *
     * @param vocabSize    词汇表大小
     * @param embeddingDim 嵌入维度
     */
    public TokenEmbedding(int vocabSize, int embeddingDim) {
        super("TokenEmbedding");
        this.vocabSize = vocabSize;
        this.embeddingDim = embeddingDim;

        // 注册嵌入矩阵参数: [vocabSize, embeddingDim]
        NdArray embeddingMatrix = NdArray.likeRandomN(Shape.of(vocabSize, embeddingDim));
        // 缩放到合适的标准差 0.02
        embeddingMatrix = embeddingMatrix.mulNum(0.02f);
        this.weight = registerParameter("weight", new Parameter(embeddingMatrix, true));
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("TokenEmbedding requires at least one input");
        }

        Variable tokenIds = inputs[0];
        NdArray ids = tokenIds.getValue();

        // tokenIds shape: [batch_size, seq_len] 或 [batch_size, seq_len, 1]
        int[] shape = ids.getShape().getShapeDims();
        int batchSize = shape[0];
        int seqLen = shape.length > 1 ? shape[1] : 1;

        // 获取嵌入权重
        NdArray embeddingWeight = weight.data();

        // 执行嵌入查找
        NdArray embedded = embeddingLookup(embeddingWeight, ids, batchSize, seqLen);

        return new Variable(embedded);
    }

    /**
     * 嵌入查找实现
     *
     * @param embeddingWeight 嵌入矩阵 [vocabSize, embeddingDim]
     * @param tokenIds        Token IDs [batch_size, seq_len]
     * @param batchSize       批次大小
     * @param seqLen          序列长度
     * @return 嵌入向量 [batch_size, seq_len, embeddingDim]
     */
    private NdArray embeddingLookup(NdArray embeddingWeight, NdArray tokenIds, int batchSize, int seqLen) {
        float[] embeddingData = embeddingWeight.getArray();
        float[] idsData = tokenIds.getArray();

        // 输出形状: [batch_size, seq_len, embeddingDim]
        float[] output = new float[batchSize * seqLen * embeddingDim];

        for (int b = 0; b < batchSize; b++) {
            for (int s = 0; s < seqLen; s++) {
                int idxPos = b * seqLen + s;
                int tokenId = (int) idsData[idxPos];

                // 检查索引有效性
                if (tokenId < 0 || tokenId >= vocabSize) {
                    throw new IndexOutOfBoundsException(
                            "Token ID " + tokenId + " out of range [0, " + vocabSize + ")"
                    );
                }

                // 复制嵌入向量
                int outputOffset = (b * seqLen + s) * embeddingDim;
                int embeddingOffset = tokenId * embeddingDim;
                System.arraycopy(embeddingData, embeddingOffset, output, outputOffset, embeddingDim);
            }
        }

        return NdArray.of(output, Shape.of(batchSize, seqLen, embeddingDim));
    }

    /**
     * 获取嵌入权重参数(用于权重共享)
     *
     * @return 嵌入权重参数
     */
    public Parameter getWeight() {
        return weight;
    }

    @Override
    public String extraRepr() {
        return String.format("vocabSize=%d, embeddingDim=%d", vocabSize, embeddingDim);
    }
}
