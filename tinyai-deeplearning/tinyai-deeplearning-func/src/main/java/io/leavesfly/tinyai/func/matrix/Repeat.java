package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 重复张量算子
 * <p>
 * 沿指定维度重复张量（复制数据）。
 * repeats[i] 表示第i维重复的次数。
 */
public class Repeat extends Function {

    private final int[] repeats;
    private Shape inputShape;
    private Shape outputShape;

    public Repeat(int... repeats) {
        this.repeats = repeats;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();

        // 验证repeats长度
        if (repeats.length != inputDims.length) {
            throw new IllegalArgumentException(
                "Repeat: repeats length must match input dimensions. " +
                "Input dims: " + inputDims.length + ", repeats: " + repeats.length
            );
        }

        // 计算输出形状
        int[] outputDims = new int[inputDims.length];
        for (int i = 0; i < inputDims.length; i++) {
            outputDims[i] = inputDims[i] * repeats[i];
        }
        outputShape = Shape.of(outputDims);

        // 实现重复：通过tile操作
        // 由于NdArray可能没有直接的tile方法，我们使用循环复制
        return tileArray(x, repeats);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度需要sum回原始形状
        // 对于每个被重复的维度，需要将梯度sum回原始大小
        // 手动实现而不是使用sumToOptimized，避免兼容性问题
        
        int[] inputDims = inputShape.getShapeDims();
        int[] outputDims = outputShape.getShapeDims();
        float[] yGradData = yGrad.getArray();
        float[] gradData = new float[inputShape.size()];
        
        // 计算步长
        int[] inputStrides = computeStrides(inputDims);
        int[] outputStrides = computeStrides(outputDims);
        
        // 遍历yGrad的所有元素，累加回输入位置
        int[] outputIdx = new int[outputDims.length];
        int[] inputIdx = new int[inputDims.length];
        
        for (int i = 0; i < yGradData.length; i++) {
            // 计算输出索引
            int pos = i;
            for (int dim = outputDims.length - 1; dim >= 0; dim--) {
                outputIdx[dim] = pos % outputDims[dim];
                pos /= outputDims[dim];
            }
            
            // 计算对应的输入索引（取模）
            for (int dim = 0; dim < inputDims.length; dim++) {
                inputIdx[dim] = outputIdx[dim] % inputDims[dim];
            }
            
            // 计算输入位置
            int inputPos = 0;
            for (int dim = 0; dim < inputDims.length; dim++) {
                inputPos += inputIdx[dim] * inputStrides[dim];
            }
            
            // 累加梯度
            gradData[inputPos] += yGradData[i];
        }
        
        return Collections.singletonList(NdArray.of(gradData, inputShape));
    }

    /**
     * 平铺数组（复制数据）
     */
    private NdArray tileArray(NdArray x, int[] repeats) {
        int[] inputDims = x.getShape().getShapeDims();
        int[] outputDims = new int[inputDims.length];
        for (int i = 0; i < inputDims.length; i++) {
            outputDims[i] = inputDims[i] * repeats[i];
        }
        
        float[] inputData = x.getArray();
        float[] outputData = new float[Shape.of(outputDims).size()];
        
        // 计算每个输出位置对应的输入位置
        int[] inputStrides = computeStrides(inputDims);
        int[] outputStrides = computeStrides(outputDims);
        
        int[] inputIdx = new int[inputDims.length];
        int[] outputIdx = new int[outputDims.length];
        
        // 遍历所有输出位置
        for (int i = 0; i < outputData.length; i++) {
            // 计算输出索引
            int pos = i;
            for (int dim = outputDims.length - 1; dim >= 0; dim--) {
                outputIdx[dim] = pos % outputDims[dim];
                pos /= outputDims[dim];
            }
            
            // 计算对应的输入索引（取模）
            for (int dim = 0; dim < inputDims.length; dim++) {
                inputIdx[dim] = outputIdx[dim] % inputDims[dim];
            }
            
            // 计算输入位置
            int inputPos = 0;
            for (int dim = 0; dim < inputDims.length; dim++) {
                inputPos += inputIdx[dim] * inputStrides[dim];
            }
            
            outputData[i] = inputData[inputPos];
        }
        
        return NdArray.of(outputData, Shape.of(outputDims));
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
        return 1;
    }
}

