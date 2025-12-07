package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * SiLU激活函数（Sigmoid Linear Unit，又称Swish）
 * <p>
 * SiLU是一种自门控激活函数，由Google在2017年提出。
 * 它结合了线性和非线性的特性，在许多深度学习任务中表现优异。
 * <p>
 * 公式：SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
 * <p>
 * 特性：
 * - 平滑且非单调
 * - 具有自门控特性
 * - 梯度比ReLU更稳定
 * - 广泛应用于EfficientNet、YOLOv5等模型
 *
 * @author leavesfly
 * @version 1.0
 */
public class SiLU extends Function {

    /**
     * 前向传播计算SiLU
     * <p>
     * 计算公式：SiLU(x) = x * sigmoid(x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return SiLU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];

        // sigmoid(x) = 1 / (1 + exp(-x))
        NdArray denominator = x.neg().exp().add(NdArray.ones(x.getShape()));
        NdArray sigmoid = NdArray.ones(x.getShape()).div(denominator);

        // SiLU(x) = x * sigmoid(x)
        return x.mul(sigmoid);
    }

    /**
     * 反向传播计算梯度
     * <p>
     * SiLU'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     *          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
     *          = sigmoid(x) * (1 + x - x * sigmoid(x))
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();

        // sigmoid(x)
        NdArray denominator = x.neg().exp().add(NdArray.ones(x.getShape()));
        NdArray sigmoid = NdArray.ones(x.getShape()).div(denominator);

        // 1 - sigmoid(x)
        NdArray oneMinusSigmoid = NdArray.ones(x.getShape()).sub(sigmoid);

        // x * (1 - sigmoid(x))
        NdArray xTimesOneMinusSigmoid = x.mul(oneMinusSigmoid);

        // 1 + x * (1 - sigmoid(x))
        NdArray onePlusXTimes = NdArray.ones(x.getShape()).add(xTimesOneMinusSigmoid);

        // sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        NdArray grad = sigmoid.mul(onePlusXTimes);

        return Collections.singletonList(yGrad.mul(grad));
    }

    /**
     * 获取所需输入参数个数
     *
     * @return 输入参数个数，固定为1
     */
    @Override
    public int requireInputNum() {
        return 1;
    }
}

