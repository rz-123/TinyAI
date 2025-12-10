package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 绝对值运算
 * <p>
 * forward: y = |x|
 * backward: dy/dx = sgn(x) * gy
 */
public class Abs extends Function {

    @Override
    public NdArray forward(NdArray... inputs) {
        return inputs[0].abs();
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // dx = gy * sgn(x)
        // sgn(x) = x / |x| (当 x != 0), 0 (当 x = 0)
        // 实际上 NdArray 可能没有 sgn 函数，但我们可以用 gt/lt 组合或者 div
        NdArray x = inputs[0].getValue();
        
        // 简单实现 sgn: mask(x > 0) - mask(x < 0)
        // 或者使用 x.div(x.abs()) 注意除零问题
        
        // 由于 NdArray 接口限制，这里可能需要一种更高效的方式。
        // 目前 NdArray 没有 sgn。
        // 我们可以用:
        // mask_pos = x.gt(0) -> 1 if >0 else 0
        // mask_neg = x.lt(0) -> 1 if <0 else 0
        // sgn = mask_pos - mask_neg
        
        NdArray zeros = NdArray.zeros(x.getShape());
        NdArray maskPos = x.gt(zeros);
        NdArray maskNeg = x.lt(zeros);
        NdArray sgn = maskPos.sub(maskNeg);
        
        return Collections.singletonList(yGrad.mul(sgn));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

