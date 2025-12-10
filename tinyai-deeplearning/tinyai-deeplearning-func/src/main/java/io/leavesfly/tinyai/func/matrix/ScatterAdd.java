package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 分散累加算子
 * <p>
 * 将源张量的值根据索引分散到目标张量的指定位置，并累加
 * 用于Embedding层的梯度回传
 */
public class ScatterAdd extends Function {

    private final int dim;
    private int[] indices;
    private Shape inputShape;
    private Shape indexShape;

    public ScatterAdd(int dim) {
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray input = inputs[0];
        NdArray index = inputs[1];
        NdArray src = inputs[2];

        inputShape = input.getShape();
        indexShape = index.getShape();
        int[] inputDims = inputShape.getShapeDims();

        // 处理负数维度
        int actualDim = dim < 0 ? inputDims.length + dim : dim;
        if (actualDim < 0 || actualDim >= inputDims.length) {
            throw new IllegalArgumentException("ScatterAdd: dimension out of range: " + dim);
        }

        // 解析索引
        float[] indexData = index.getArray();
        int totalIndices = indexData.length;
        indices = new int[totalIndices];
        int dimSize = inputDims[actualDim];

        for (int i = 0; i < totalIndices; i++) {
            int idx = (int) indexData[i];
            if (idx < 0 || idx >= dimSize) {
                throw new IndexOutOfBoundsException(
                    "ScatterAdd: index " + idx + " out of range [0, " + dimSize + ")"
                );
            }
            indices[i] = idx;
        }

        // 复制输入并执行分散累加
        float[] inputData = input.getArray();
        float[] resultData = new float[inputData.length];
        System.arraycopy(inputData, 0, resultData, 0, inputData.length);

        float[] srcData = src.getArray();
        int[] inputStrides = computeStrides(inputDims);
        int[] srcDims = src.getShape().getShapeDims();
        int[] srcStrides = computeStrides(srcDims);

        // 执行分散累加
        scatterAddElements(resultData, srcData, actualDim, inputDims, inputStrides, srcDims, srcStrides);

        return NdArray.of(resultData, inputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 反向传播：
        // dInput = dY (直接传播)
        // dSrc = dY[index] (使用IndexSelect)
        // dIndex = null (索引不可导)

        // dInput
        NdArray gradInput = yGrad;

        // dSrc: 需要从yGrad中选择对应的元素
        // 简化实现：使用IndexSelect的逻辑
        NdArray gradSrc = selectFromGrad(yGrad);

        return java.util.Arrays.asList(
            gradInput,
            null, // index不可导
            gradSrc
        );
    }

    /**
     * 执行分散累加
     */
    private void scatterAddElements(float[] result, float[] src, int dim,
                                     int[] inputDims, int[] inputStrides,
                                     int[] srcDims, int[] srcStrides) {
        // 遍历src的所有元素
        int[] srcIdx = new int[srcDims.length];
        for (int i = 0; i < src.length; i++) {
            // 计算src的索引
            int pos = i;
            for (int d = srcDims.length - 1; d >= 0; d--) {
                srcIdx[d] = pos % srcDims[d];
                pos /= srcDims[d];
            }

            // 计算对应的input索引
            int[] inputIdx = new int[inputDims.length];
            int flatSrcIdx = 0;
            for (int d = 0; d < srcDims.length; d++) {
                if (d < dim) {
                    inputIdx[d] = srcIdx[d];
                } else if (d > dim) {
                    inputIdx[d] = srcIdx[d - 1]; // 跳过dim维度
                }
                flatSrcIdx += srcIdx[d] * srcStrides[d];
            }
            inputIdx[dim] = indices[srcIdx[0]]; // 假设索引在第一个维度

            // 计算input位置
            int inputPos = 0;
            for (int d = 0; d < inputDims.length; d++) {
                inputPos += inputIdx[d] * inputStrides[d];
            }

            // 累加
            result[inputPos] += src[flatSrcIdx];
        }
    }

    /**
     * 从梯度中选择元素（用于dSrc）
     */
    private NdArray selectFromGrad(NdArray yGrad) {
        // 简化实现：返回与src相同形状的梯度
        // 完整实现需要根据索引选择
        Shape srcShape = inputs[2].getValue().getShape();
        return yGrad.broadcastTo(srcShape);
    }

    /**
     * 计算步长
     */
    private int[] computeStrides(int[] dims) {
        int[] strides = new int[dims.length];
        strides[dims.length - 1] = 1;
        for (int i = dims.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        return strides;
    }

    @Override
    public int requireInputNum() {
        return 3;
    }
}

