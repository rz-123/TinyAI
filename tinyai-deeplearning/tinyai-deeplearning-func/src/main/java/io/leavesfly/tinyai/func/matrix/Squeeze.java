package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 压缩维度算子
 * <p>
 * 移除大小为1的维度。如果指定了维度，则只移除该维度（如果大小为1）。
 * 如果未指定维度，则移除所有大小为1的维度。
 */
public class Squeeze extends Function {

    private final Integer targetDim; // null表示移除所有大小为1的维度
    private Shape inputShape;
    private Shape outputShape;

    /**
     * 构造函数：移除所有大小为1的维度
     */
    public Squeeze() {
        this.targetDim = null;
    }

    /**
     * 构造函数：移除指定维度（如果大小为1）
     *
     * @param dim 要移除的维度索引（支持负数索引）
     */
    public Squeeze(int dim) {
        this.targetDim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        inputShape = x.getShape();
        int[] dims = inputShape.getShapeDims();

        if (targetDim == null) {
            // 移除所有大小为1的维度
            List<Integer> newDims = new ArrayList<>();
            for (int d : dims) {
                if (d != 1) {
                    newDims.add(d);
                }
            }
            if (newDims.isEmpty()) {
                newDims.add(1); // 至少保留一个维度
            }
            outputShape = Shape.of(newDims.stream().mapToInt(i -> i).toArray());
        } else {
            // 移除指定维度（如果大小为1）
            int target = targetDim < 0 ? dims.length + targetDim : targetDim;
            if (target < 0 || target >= dims.length) {
                throw new IllegalArgumentException("Dimension out of range: " + targetDim + " for shape " + inputShape);
            }
            if (dims[target] != 1) {
                throw new IllegalArgumentException("Can only squeeze dimension of size 1, but got size " + dims[target] + " at dimension " + target);
            }

            int[] newDims = new int[dims.length - 1];
            for (int i = 0, j = 0; i < dims.length; i++) {
                if (i != target) {
                    newDims[j++] = dims[i];
                }
            }
            outputShape = Shape.of(newDims);
        }

        return x.reshape(outputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 恢复原始形状
        return Collections.singletonList(yGrad.reshape(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

