package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 下三角矩阵算子
 * <p>
 * 返回输入矩阵的下三角部分，上三角部分置为0。
 * k参数控制对角线的偏移：
 * - k=0: 主对角线及以下
 * - k>0: 主对角线上方k条对角线也保留
 * - k<0: 主对角线下方|k|条对角线被置0
 */
public class Tril extends Function {

    private final int k;
    private NdArray mask;

    public Tril(int k) {
        this.k = k;
    }

    public Tril() {
        this(0);
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        Shape shape = x.getShape();
        
        // 生成下三角掩码
        mask = generateTrilMask(shape, k);
        
        // y = x * mask
        return x.mul(mask);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // dy/dx = mask
        return Collections.singletonList(yGrad.mul(mask));
    }

    /**
     * 生成下三角掩码矩阵
     *
     * @param shape 输入形状
     * @param k     对角线偏移
     * @return 掩码矩阵（1表示保留，0表示置0）
     */
    private NdArray generateTrilMask(Shape shape, int k) {
        int[] dims = shape.getShapeDims();
        
        // 目前只支持2D矩阵
        if (dims.length != 2) {
            throw new IllegalArgumentException("Tril currently only supports 2D matrices, got shape: " + shape);
        }
        
        int rows = dims[0];
        int cols = dims[1];
        float[] maskData = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // 下三角：j <= i + k
                if (j <= i + k) {
                    maskData[i * cols + j] = 1.0f;
                } else {
                    maskData[i * cols + j] = 0.0f;
                }
            }
        }
        
        return NdArray.of(maskData, shape);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

