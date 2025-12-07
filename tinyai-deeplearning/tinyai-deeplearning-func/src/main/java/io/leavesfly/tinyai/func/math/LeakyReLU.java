package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * LeakyReLU激活函数（Leaky Rectified Linear Unit）
 * <p>
 * LeakyReLU是ReLU的改进版本，解决了ReLU的"神经元死亡"问题。
 * 它允许负值以一个小的斜率通过，从而保持梯度流动。
 * <p>
 * 公式：
 * LeakyReLU(x) = x,           if x >= 0
 *              = negative_slope * x,  if x < 0
 * <p>
 * 或简写为：LeakyReLU(x) = max(negative_slope * x, x)
 * <p>
 * 特性：
 * - 解决ReLU的神经元死亡问题
 * - 负值区域有小梯度，保持梯度流动
 * - 默认negative_slope = 0.01
 *
 * @author leavesfly
 * @version 1.0
 */
public class LeakyReLU extends Function {

    /**
     * 负斜率参数
     * 控制负值区域的斜率，默认为0.01
     */
    private final float negativeSlope;

    /**
     * 构造函数
     *
     * @param negativeSlope 负斜率参数
     */
    public LeakyReLU(float negativeSlope) {
        this.negativeSlope = negativeSlope;
    }

    /**
     * 默认构造函数（negativeSlope = 0.01）
     */
    public LeakyReLU() {
        this(0.01f);
    }

    /**
     * 前向传播计算LeakyReLU
     * <p>
     * 计算公式：max(negative_slope * x, x)
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return LeakyReLU函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        float[] data = x.getArray();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] >= 0 ? data[i] : negativeSlope * data[i];
        }

        return NdArray.of(result, x.getShape());
    }

    /**
     * 反向传播计算梯度
     * <p>
     * LeakyReLU'(x) = 1,              if x >= 0
     *               = negative_slope, if x < 0
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        float[] data = x.getArray();
        float[] gradData = yGrad.getArray();
        float[] result = new float[data.length];

        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] >= 0 ? gradData[i] : negativeSlope * gradData[i];
        }

        return Collections.singletonList(NdArray.of(result, x.getShape()));
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

    /**
     * 获取负斜率参数
     *
     * @return 负斜率参数值
     */
    public float getNegativeSlope() {
        return negativeSlope;
    }
}

