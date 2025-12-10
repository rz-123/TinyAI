package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * 掩码填充
 * <p>
 * forward: y = x, but elements where mask == 1 are replaced by value.
 * backward: dx = dy, but elements where mask == 1 are set to 0.
 */
public class MaskedFill extends Function {

    private final float fillValue;
    private NdArray mask;

    public MaskedFill(float fillValue) {
        this.fillValue = fillValue;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        this.mask = inputs[1]; // 0/1 mask

        // y = x * (1 - mask) + value * mask
        // 但为了效率和数值稳定性，底层 NdArray 最好有专门的 maskedFill
        // 这里模拟实现:
        // mask 是 0/1。
        // out = x.mul(mask.neg().add(1)); // 保留 mask为0 的部分
        // out = out.add(mask.mul(fillValue)); // 填充 mask为1 的部分

        // 这种实现当 x 中对应位置是 NaN 或 Inf 时可能出问题，
        // 更好的方式是: if (mask[i]) out[i] = value; else out[i] = x[i];
        
        // 暂用 NdArray.mask 辅助实现或者循环实现
        // 假设 mask 也是 float 类型的 0.0/1.0
        
        NdArray ones = NdArray.ones(mask.getShape());
        NdArray invMask = ones.sub(mask); // 1 - mask (保留原值的位置为1)
        
        NdArray keptPart = x.mul(invMask);
        NdArray filledPart = mask.mulNum(fillValue);
        
        return keptPart.add(filledPart);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // dx = dy * (1 - mask)
        // 填充位置的梯度被阻断
        NdArray ones = NdArray.ones(mask.getShape());
        NdArray invMask = ones.sub(mask);
        
        // 返回两个梯度：第一个对应input，第二个对应mask（不可导）
        return java.util.Arrays.asList(
            yGrad.mul(invMask),
            null  // mask不可导
        );
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

