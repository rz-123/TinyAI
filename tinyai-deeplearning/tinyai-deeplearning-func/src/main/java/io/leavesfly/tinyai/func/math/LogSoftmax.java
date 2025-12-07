package io.leavesfly.tinyai.func.math;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Collections;
import java.util.List;

/**
 * LogSoftmax激活函数
 * <p>
 * LogSoftmax是Softmax的对数形式，常用于NLLLoss组合。
 * LogSoftmax(x) = log(softmax(x)) = x - log(sum(exp(x)))
 * <p>
 * 使用log-sum-exp技巧保证数值稳定性：
 * LogSoftmax(x) = x - max(x) - log(sum(exp(x - max(x))))
 * <p>
 * 特性：
 * - 数值稳定，避免溢出
 * - 常与NLLLoss配合使用
 * - 默认在最后一维进行计算
 *
 * @author leavesfly
 * @version 1.0
 */
public class LogSoftmax extends Function {

    /**
     * 计算轴
     * -1 表示最后一维
     */
    private final int axis;

    /**
     * 构造函数
     *
     * @param axis 计算轴
     */
    public LogSoftmax(int axis) {
        this.axis = axis;
    }

    /**
     * 默认构造函数（axis = -1，最后一维）
     */
    public LogSoftmax() {
        this(-1);
    }

    /**
     * 前向传播计算LogSoftmax
     * <p>
     * 使用数值稳定的log-sum-exp技巧
     *
     * @param inputs 输入的NdArray数组，长度为1
     * @return LogSoftmax函数值的NdArray
     */
    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        int[] dims = x.getShape().getShapeDims();
        int actualAxis = axis < 0 ? dims.length + axis : axis;

        if (dims.length == 1) {
            // 1D情况
            return computeLogSoftmax1D(x);
        } else if (dims.length == 2) {
            // 2D情况
            return computeLogSoftmax2D(x, actualAxis);
        } else {
            // 多维情况，默认最后一维
            return computeLogSoftmaxND(x, actualAxis);
        }
    }

    private NdArray computeLogSoftmax1D(NdArray x) {
        float[] data = x.getArray();
        float[] result = new float[data.length];

        // 找最大值（数值稳定性）
        float maxVal = Float.NEGATIVE_INFINITY;
        for (float v : data) {
            maxVal = Math.max(maxVal, v);
        }

        // 计算 log(sum(exp(x - max)))
        float sumExp = 0;
        for (float v : data) {
            sumExp += (float) Math.exp(v - maxVal);
        }
        float logSumExp = (float) Math.log(sumExp) + maxVal;

        // log_softmax = x - log_sum_exp
        for (int i = 0; i < data.length; i++) {
            result[i] = data[i] - logSumExp;
        }

        return NdArray.of(result, x.getShape());
    }

    private NdArray computeLogSoftmax2D(NdArray x, int actualAxis) {
        float[][] matrix = x.getMatrix();
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[][] result = new float[rows][cols];

        if (actualAxis == 1) {
            // 按行计算（每行是一个样本）
            for (int i = 0; i < rows; i++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int j = 0; j < cols; j++) {
                    maxVal = Math.max(maxVal, matrix[i][j]);
                }

                float sumExp = 0;
                for (int j = 0; j < cols; j++) {
                    sumExp += (float) Math.exp(matrix[i][j] - maxVal);
                }
                float logSumExp = (float) Math.log(sumExp) + maxVal;

                for (int j = 0; j < cols; j++) {
                    result[i][j] = matrix[i][j] - logSumExp;
                }
            }
        } else {
            // 按列计算
            for (int j = 0; j < cols; j++) {
                float maxVal = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < rows; i++) {
                    maxVal = Math.max(maxVal, matrix[i][j]);
                }

                float sumExp = 0;
                for (int i = 0; i < rows; i++) {
                    sumExp += (float) Math.exp(matrix[i][j] - maxVal);
                }
                float logSumExp = (float) Math.log(sumExp) + maxVal;

                for (int i = 0; i < rows; i++) {
                    result[i][j] = matrix[i][j] - logSumExp;
                }
            }
        }

        return NdArray.of(result);
    }

    private NdArray computeLogSoftmaxND(NdArray x, int actualAxis) {
        // 对于多维情况，先简化为2D处理
        // 这里采用简化实现，将最后一维作为softmax维度
        int[] dims = x.getShape().getShapeDims();
        int lastDim = dims[dims.length - 1];
        int batchSize = x.getShape().size() / lastDim;

        float[] data = x.getArray();
        float[] result = new float[data.length];

        for (int b = 0; b < batchSize; b++) {
            int offset = b * lastDim;

            // 找最大值
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < lastDim; i++) {
                maxVal = Math.max(maxVal, data[offset + i]);
            }

            // 计算 log(sum(exp))
            float sumExp = 0;
            for (int i = 0; i < lastDim; i++) {
                sumExp += (float) Math.exp(data[offset + i] - maxVal);
            }
            float logSumExp = (float) Math.log(sumExp) + maxVal;

            // log_softmax
            for (int i = 0; i < lastDim; i++) {
                result[offset + i] = data[offset + i] - logSumExp;
            }
        }

        return NdArray.of(result, x.getShape());
    }

    /**
     * 反向传播计算梯度
     * <p>
     * LogSoftmax的梯度：
     * grad_input = grad_output - softmax(input) * sum(grad_output)
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    @Override
    public List<NdArray> backward(NdArray yGrad) {
        NdArray x = inputs[0].getValue();
        NdArray output = this.output.getValue();  // log_softmax的输出

        // softmax = exp(log_softmax)
        NdArray softmax = output.exp();

        int[] dims = x.getShape().getShapeDims();
        int lastDim = dims[dims.length - 1];
        int batchSize = x.getShape().size() / lastDim;

        float[] gradData = yGrad.getArray();
        float[] softmaxData = softmax.getArray();
        float[] result = new float[gradData.length];

        for (int b = 0; b < batchSize; b++) {
            int offset = b * lastDim;

            // sum(grad_output)
            float sumGrad = 0;
            for (int i = 0; i < lastDim; i++) {
                sumGrad += gradData[offset + i];
            }

            // grad_input = grad_output - softmax * sum(grad_output)
            for (int i = 0; i < lastDim; i++) {
                result[offset + i] = gradData[offset + i] - softmaxData[offset + i] * sumGrad;
            }
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
     * 获取计算轴
     *
     * @return 计算轴
     */
    public int getAxis() {
        return axis;
    }
}

