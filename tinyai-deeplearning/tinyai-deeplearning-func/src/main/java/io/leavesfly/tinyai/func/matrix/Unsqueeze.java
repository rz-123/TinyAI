package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 增加维度
 * <p>
 * 在指定位置插入大小为1的维度。
 */
public class Unsqueeze extends Function {

    private final int dim;
    private Shape inputShape;

    public Unsqueeze(int dim) {
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        int[] dims = inputShape.getShapeDims();
        int[] newDims = new int[dims.length + 1];
        
        int targetDim = dim;
        if (targetDim < 0) targetDim += newDims.length;
        
        for (int i = 0, j = 0; i < newDims.length; i++) {
            if (i == targetDim) {
                newDims[i] = 1;
            } else {
                newDims[i] = dims[j++];
            }
        }
        
        return inputs[0].reshape(Shape.of(newDims));
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 恢复原状
        return Collections.singletonList(yGrad.reshape(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

