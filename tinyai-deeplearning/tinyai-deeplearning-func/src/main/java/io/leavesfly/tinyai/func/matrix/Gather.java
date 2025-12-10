package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * Gather 操作 (Embedding Lookup)
 * <p>
 * forward: y = x[indices]
 * backward: dx[indices] += dy (Scatter Add)
 */
public class Gather extends Function {

    private int[] flattenedSlices;
    private Shape inputIndicesShape;
    private Shape inputWeightShape;

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray weight = inputs[0];
        NdArray indices = inputs[1];

        inputWeightShape = weight.getShape();
        inputIndicesShape = indices.getShape();

        // 解析索引
        float[] idsData = indices.getArray();
        int total = idsData.length;
        flattenedSlices = new int[total];
        int vocabSize = weight.getShape().getDimension(0);

        for (int i = 0; i < total; i++) {
            int tokenId = (int) idsData[i];
            if (tokenId < 0 || tokenId >= vocabSize) {
                throw new IndexOutOfBoundsException(
                        "Token ID " + tokenId + " out of range [0, " + vocabSize + ")"
                );
            }
            flattenedSlices[i] = tokenId;
        }

        // 执行 gather (使用 getItem 选取行)
        // 注意：NdArray.getItem 接受 int[] rowSlices, int[] colSlices
        // 这里我们只选行，colSlices 为 null 表示选取该行的所有列
        NdArray gathered = weight.getItem(flattenedSlices, null);

        // Reshape 结果
        // output shape: indices.shape + [embedding_dim]
        int embeddingDim = weight.getShape().getDimension(1);
        int[] indexDims = indices.getShape().getShapeDims();
        int[] outDims = new int[indexDims.length + 1];
        System.arraycopy(indexDims, 0, outDims, 0, indexDims.length);
        outDims[indexDims.length] = embeddingDim;

        return gathered.reshape(Shape.of(outDims));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // yGrad shape: [batch, seq, dim]
        // 需要 reshape 成 [batch*seq, dim] 以匹配 flattenedSlices
        int embeddingDim = inputWeightShape.getDimension(1);
        int totalIndices = inputIndicesShape.size();

        NdArray flatGrad = yGrad.reshape(Shape.of(totalIndices, embeddingDim));

        // 创建 weight 的梯度累加器
        NdArray weightGrad = NdArray.zeros(inputWeightShape);

        // Scatter Add: 将梯度累加回对应的行
        weightGrad.addAt(flattenedSlices, null, flatGrad);

        // indices 是整数索引，不可导，返回 null
        return Arrays.asList(weightGrad, null);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

