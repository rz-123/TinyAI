package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 扩展维度算子（广播语义）
 * <p>
 * 将大小为1的维度扩展到指定大小，不复制数据（view语义）。
 * 只能扩展大小为1的维度，其他维度必须匹配。
 */
public class Expand extends Function {

    private final Shape targetShape;
    private Shape inputShape;

    public Expand(Shape targetShape) {
        this.targetShape = targetShape;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] inputDims = inputShape.getShapeDims();
        int[] targetDims = targetShape.getShapeDims();

        // 验证形状兼容性
        if (inputDims.length != targetDims.length) {
            throw new IllegalArgumentException(
                "Expand: input and target must have same number of dimensions. " +
                "Input: " + inputShape + ", Target: " + targetShape
            );
        }

        // 验证每个维度：要么大小相同，要么输入大小为1
        for (int i = 0; i < inputDims.length; i++) {
            if (inputDims[i] != targetDims[i] && inputDims[i] != 1) {
                throw new IllegalArgumentException(
                    "Expand: can only expand dimension of size 1. " +
                    "Dimension " + i + ": input=" + inputDims[i] + ", target=" + targetDims[i]
                );
            }
        }

        // 使用新的broadcastReshape API（支持广播语义）
        return x.broadcastReshape(targetShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度需要sum回原始形状（因为扩展是广播）
        // 对于扩展的维度，需要求和
        int[] inputDims = inputShape.getShapeDims();
        int[] targetDims = targetShape.getShapeDims();
        
        NdArray grad = yGrad;
        
        // 对于每个被扩展的维度（从1扩展到更大），需要求和
        // 使用SumTo将梯度sum回原始形状
        grad = grad.sumTo(inputShape);
        
        // 移除大小为1的维度（如果原始输入在该维度大小为1）
        return Collections.singletonList(grad.reshape(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

