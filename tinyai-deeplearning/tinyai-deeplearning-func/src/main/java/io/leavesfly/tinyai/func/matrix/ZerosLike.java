package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 创建同形状全0张量算子
 * <p>
 * 返回与输入张量相同形状的全0张量
 */
public class ZerosLike extends Function {

    private Shape inputShape;

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        return NdArray.zeros(inputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // zerosLike不依赖输入，梯度为0
        return Collections.singletonList(NdArray.zeros(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

