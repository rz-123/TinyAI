package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 索引选择算子
 * <p>
 * 沿指定维度选择索引对应的元素
 * 输入: [..., N, ...] 和索引 [M] -> 输出: [..., M, ...]
 */
public class IndexSelect extends Function {

    private final int dim;
    private Shape inputShape;
    private int[] indices;
    private int indexSize;

    public IndexSelect(int dim) {
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        NdArray index = inputs[1];

        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();

        // 处理负数维度
        int actualDim = dim < 0 ? inputDims.length + dim : dim;
        if (actualDim < 0 || actualDim >= inputDims.length) {
            throw new IllegalArgumentException("IndexSelect: dimension out of range: " + dim);
        }

        // 解析索引
        float[] indexData = index.getArray();
        indexSize = indexData.length;
        indices = new int[indexSize];
        int dimSize = inputDims[actualDim];

        for (int i = 0; i < indexSize; i++) {
            int idx = (int) indexData[i];
            if (idx < 0 || idx >= dimSize) {
                throw new IndexOutOfBoundsException(
                    "IndexSelect: index " + idx + " out of range [0, " + dimSize + ")"
                );
            }
            indices[i] = idx;
        }

        // 计算输出形状
        int[] outputDims = inputDims.clone();
        outputDims[actualDim] = indexSize;
        Shape outputShape = Shape.of(outputDims);

        // 执行索引选择
        return selectElements(x, actualDim, outputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度需要scatter回原始形状
        // 对于每个被选中的索引位置，将梯度散射回去
        // 如果索引有重复，需要累加梯度
        
        int[] inputDims = inputShape.getShapeDims();
        float[] gradData = new float[inputShape.size()];
        float[] yGradData = yGrad.getArray();
        int[] yGradDims = yGrad.getShape().getShapeDims();
        
        // 处理负数维度
        int actualDim = dim < 0 ? inputDims.length + dim : dim;
        
        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] yGradStrides = computeStrides(yGradDims);
        
        // 遍历yGrad的所有元素，scatter回输入形状
        int[] yGradIdx = new int[yGradDims.length];
        for (int i = 0; i < yGradData.length; i++) {
            // 计算yGrad索引
            int pos = i;
            for (int d = yGradDims.length - 1; d >= 0; d--) {
                yGradIdx[d] = pos % yGradDims[d];
                pos /= yGradDims[d];
            }
            
            // 计算对应的输入索引
            int[] inputIdx = yGradIdx.clone();
            inputIdx[actualDim] = indices[yGradIdx[actualDim]];
            
            // 计算输入位置
            int inputPos = 0;
            for (int d = 0; d < inputDims.length; d++) {
                inputPos += inputIdx[d] * inputStrides[d];
            }
            
            // 累加梯度（如果索引重复，会自动累加）
            gradData[inputPos] += yGradData[i];
        }
        
        // 返回两个梯度：第一个对应input，第二个对应index（不可导）
        return java.util.Arrays.asList(
            NdArray.of(gradData, inputShape),
            null  // index不可导
        );
    }

    /**
     * 执行元素选择
     */
    private NdArray selectElements(NdArray x, int dim, Shape outputShape) {
        float[] xData = x.getArray();
        float[] outputData = new float[outputShape.size()];
        int[] inputDims = inputShape.getShapeDims();
        int[] outputDims = outputShape.getShapeDims();

        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] outputStrides = computeStrides(outputDims);

        // 遍历所有输出位置
        int[] outputIdx = new int[outputDims.length];
        for (int i = 0; i < outputData.length; i++) {
            // 计算输出索引
            int pos = i;
            for (int d = outputDims.length - 1; d >= 0; d--) {
                outputIdx[d] = pos % outputDims[d];
                pos /= outputDims[d];
            }

            // 计算输入索引
            int[] inputIdx = outputIdx.clone();
            inputIdx[dim] = indices[outputIdx[dim]];

            // 计算输入位置
            int inputPos = 0;
            for (int d = 0; d < inputDims.length; d++) {
                inputPos += inputIdx[d] * inputStrides[d];
            }

            outputData[i] = xData[inputPos];
        }

        return NdArray.of(outputData, outputShape);
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
        return 2;
    }
}

