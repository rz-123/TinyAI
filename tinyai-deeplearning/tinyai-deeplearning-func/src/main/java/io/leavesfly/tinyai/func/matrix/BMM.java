package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Arrays;
import java.util.List;

/**
 * 批量矩阵乘法算子
 * <p>
 * 输入: [batch, n, m] @ [batch, m, p] -> [batch, n, p]
 * 比循环调用matMul更高效
 */
public class BMM extends Function {

    private int batchSize;
    private int n, m, p;
    private Shape aShape;
    private Shape bShape;

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray a = inputs[0];
        NdArray b = inputs[1];

        aShape = a.getShape();
        bShape = b.getShape();
        int[] aDims = aShape.getShapeDims();
        int[] bDims = bShape.getShapeDims();

        if (aDims.length != 3 || bDims.length != 3) {
            throw new IllegalArgumentException(
                "BMM requires 3D tensors. Got: a=" + aShape + ", b=" + bShape
            );
        }

        batchSize = aDims[0];
        n = aDims[1];
        m = aDims[2];
        p = bDims[2];

        if (aDims[0] != bDims[0] || aDims[2] != bDims[1]) {
            throw new IllegalArgumentException(
                "BMM shape mismatch: a=" + aShape + ", b=" + bShape
            );
        }

        // 实现批量矩阵乘法
        // 可以循环调用matMul，或优化为批量计算
        NdArray result = NdArray.zeros(Shape.of(batchSize, n, p));
        float[] aData = a.getArray();
        float[] bData = b.getArray();
        float[] resultData = result.getArray();

        // 批量矩阵乘法：对每个batch执行矩阵乘法
        for (int batch = 0; batch < batchSize; batch++) {
            int aOffset = batch * n * m;
            int bOffset = batch * m * p;
            int resultOffset = batch * n * p;

            // 矩阵乘法: result[b] = a[b] @ b[b]
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < m; k++) {
                        sum += aData[aOffset + i * m + k] * bData[bOffset + k * p + j];
                    }
                    resultData[resultOffset + i * p + j] = sum;
                }
            }
        }

        return result;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度计算
        // dA = dY @ B^T  (每个batch)
        // dB = A^T @ dY  (每个batch)

        float[] yGradData = yGrad.getArray();
        float[] aData = inputs[0].getValue().getArray();
        float[] bData = inputs[1].getValue().getArray();

        NdArray gradA = NdArray.zeros(aShape);
        NdArray gradB = NdArray.zeros(bShape);
        float[] gradAData = gradA.getArray();
        float[] gradBData = gradB.getArray();

        // 计算每个batch的梯度
        for (int batch = 0; batch < batchSize; batch++) {
            int aOffset = batch * n * m;
            int bOffset = batch * m * p;
            int yGradOffset = batch * n * p;

            // dA = dY @ B^T
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < m; k++) {
                    float sum = 0.0f;
                    for (int j = 0; j < p; j++) {
                        sum += yGradData[yGradOffset + i * p + j] * bData[bOffset + k * p + j];
                    }
                    gradAData[aOffset + i * m + k] = sum;
                }
            }

            // dB = A^T @ dY
            for (int k = 0; k < m; k++) {
                for (int j = 0; j < p; j++) {
                    float sum = 0.0f;
                    for (int i = 0; i < n; i++) {
                        sum += aData[aOffset + i * m + k] * yGradData[yGradOffset + i * p + j];
                    }
                    gradBData[bOffset + k * p + j] = sum;
                }
            }
        }

        return Arrays.asList(gradA, gradB);
    }

    @Override
    public int requireInputNum() {
        return 2;
    }
}

