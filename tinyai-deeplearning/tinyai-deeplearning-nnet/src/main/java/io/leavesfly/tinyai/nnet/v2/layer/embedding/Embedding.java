package io.leavesfly.tinyai.nnet.v2.layer.embedding;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;
import io.leavesfly.tinyai.nnet.v2.init.Initializers;

/**
 * 词嵌入层：将离散索引映射为连续向量表示。
 * <p>
 * 仅支持1D/2D索引输入：
 * - (seq_len,)
 * - (batch_size, seq_len)
 * 输出形状：
 * - 1D输入 -> (seq_len, embedding_dim)
 * - 2D输入 -> (batch_size, seq_len, embedding_dim)，若 seq_len==1 则压缩为 (batch_size, embedding_dim)
 */
public class Embedding extends Module {

    private final int numEmbeddings;
    private final int embeddingDim;
    private final Parameter weight;

    public Embedding(String name, int numEmbeddings, int embeddingDim) {
        super(name);
        this.numEmbeddings = numEmbeddings;
        this.embeddingDim = embeddingDim;

        NdArray weightData = NdArray.likeRandomN(Shape.of(numEmbeddings, embeddingDim));
        this.weight = registerParameter("weight", new Parameter(weightData));

        init();
    }

    @Override
    public void resetParameters() {
        // 使用较小方差的正态分布初始化嵌入
        Initializers.normal(weight.data(), 0f, 0.01f);
    }

    @Override
    public Variable forward(Variable... inputs) {
        if (inputs.length == 0) {
            throw new IllegalArgumentException("Embedding requires one input indices Variable");
        }
        Variable indices = inputs[0];
        NdArray idxValue = indices.getValue();
        int dim = idxValue.getShape().getDimNum();

        if (dim == 1) {
            int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[0]);
            return weight.getItem(slices, null);
        } else if (dim == 2) {
            int batchSize = idxValue.getShape().getRow();
            int seqLen = idxValue.getShape().getColumn();

            NdArray result = NdArray.zeros(Shape.of(batchSize, seqLen, embeddingDim));
            for (int i = 0; i < batchSize; i++) {
                int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[i]);
                Variable embRow = weight.getItem(slices, null);
                NdArray embVal = embRow.getValue();
                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        result.set(embVal.get(j, k), i, j, k);
                    }
                }
            }

            if (seqLen == 1) {
                result = result.reshape(Shape.of(batchSize, embeddingDim));
            }
            return new Variable(result);
        } else {
            throw new IllegalArgumentException("Embedding only supports 1D or 2D index tensors, got shape: " + idxValue.getShape());
        }
    }

    public Parameter getWeight() {
        return weight;
    }

    public int getNumEmbeddings() {
        return numEmbeddings;
    }

    public int getEmbeddingDim() {
        return embeddingDim;
    }

    @Override
    public String toString() {
        return "Embedding{" +
                "name='" + name + '\'' +
                ", numEmbeddings=" + numEmbeddings +
                ", embeddingDim=" + embeddingDim +
                '}';
    }
}

