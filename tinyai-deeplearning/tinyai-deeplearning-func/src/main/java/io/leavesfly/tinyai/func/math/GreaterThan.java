package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 大于比较
 * <p>
 * 返回 0/1 掩码。不可导（梯度为0）。
 */
public class GreaterThan extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].gt(inputs[1]);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 比较操作不可导，梯度为0
        NdArray grad0 = NdArray.zeros(inputs[0].getValue().getShape());
        NdArray grad1 = NdArray.zeros(inputs[1].getValue().getShape());
        return java.util.Arrays.asList(grad0, grad1);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

