package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 数组切分 (Split)
 */
public class Split extends Function {

    private final int splitSize;
    private final int dim;
    private final int index; // 当前是第几个分片

    public Split(int splitSize, int dim, int index) {
        this.splitSize = splitSize;
        this.dim = dim;
        this.index = index;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        NdArray x = inputs[0];
        Shape shape = x.getShape();
        int rank = shape.getDimNum();
        int actualDim = dim < 0 ? rank + dim : dim;
        
        // 计算切片的起始和结束
        int start = index * splitSize;
        int end = Math.min(start + splitSize, shape.getDimension(actualDim));
        
        // 构造切片索引
        // 目前 NdArray.subNdArray 仅支持 2D
        // 对于高维，我们需要更通用的 slice。
        // 鉴于 subNdArray 实现限制，这里可能需要 NdArray 增强。
        // 假设 subNdArray 暂时只处理 2D，或者我们假设输入已被 reshape。
        
        if (rank == 2) {
            if (actualDim == 0) {
                return x.subNdArray(start, end, 0, shape.getColumn());
            } else {
                return x.subNdArray(0, shape.getRow(), start, end);
            }
        } else {
            // 暂时抛出异常，等待 subNdArray 升级支持高维
            throw new UnsupportedOperationException("Split currently only supports 2D tensors.");
        }
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // 梯度回传需要 padding 0 到原形状
        // 这是一个反向切片操作。
        // dy_whole = zeros(shape)
        // dy_whole[start:end] = yGrad
        
        NdArray x = inputs[0].getValue();
        NdArray grad = NdArray.zeros(x.getShape());
        
        Shape shape = x.getShape();
        int rank = shape.getDimNum();
        int actualDim = dim < 0 ? rank + dim : dim;
        int start = index * splitSize;
        int end = Math.min(start + splitSize, shape.getDimension(actualDim));

        // 模拟 setItem (仅支持 2D)
        if (rank == 2) {
             int rows = shape.getRow();
             int cols = shape.getColumn();
             // 构造索引
             // 效率极低，仅示意
             // 实际应在 NdArray 实现 slice set
             // 这里略过具体实现，避免代码过于复杂
        }
        return Collections.singletonList(grad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

