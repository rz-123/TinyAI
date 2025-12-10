package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.ArrayList;
import java.util.List;

/**
 * 拼接算子 (Concatenate)
 * <p>
 * forward: 将多个 Variable 沿指定维度拼接
 * backward: 将梯度沿指定维度切分回传
 */
public class Concat extends Function {

    private final int dim;
    private int[] splitSizes;
    private Shape[] inputShapes;

    public Concat(int dim) {
        this.dim = dim;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("Concat inputs cannot be empty");
        }
        
        // 记录输入形状以便反向传播
        inputShapes = new Shape[inputs.length];
        splitSizes = new int[inputs.length];
        
        int totalSizeInDim = 0;
        Shape baseShape = inputs[0].getShape();
        int targetDim = dim < 0 ? baseShape.getDimNum() + dim : dim;
        
        for (int i = 0; i < inputs.length; i++) {
            inputShapes[i] = inputs[i].getShape();
            splitSizes[i] = inputShapes[i].getDimension(targetDim);
            totalSizeInDim += splitSizes[i];
            
            // 校验其他维度是否一致
            if (i > 0) {
                // 简单校验维度数
                if (inputShapes[i].getDimNum() != baseShape.getDimNum()) {
                    throw new IllegalArgumentException("Input shapes must have same rank");
                }
            }
        }
        
        // 计算目标形状
        int[] newDims = baseShape.getShapeDims().clone();
        newDims[targetDim] = totalSizeInDim;
        Shape targetShape = Shape.of(newDims);
        
        // 创建结果数组
        NdArray result = NdArray.zeros(targetShape);
        
        int currentOffset = 0;
        
        // 使用新的高性能API代替逐点赋值
        if (baseShape.getDimNum() == 2) {
            for (int i = 0; i < inputs.length; i++) {
                NdArray input = inputs[i];
                int rows = input.getShape().getRow();
                int cols = input.getShape().getColumn();
                
                // 使用setBlock高效赋值
                if (targetDim == 0) { // 垂直拼接
                    result.setBlock(currentOffset, currentOffset + rows, 0, cols, input.getArray());
                    currentOffset += rows;
                } else { // 水平拼接
                    result.setBlock(0, rows, currentOffset, currentOffset + cols, input.getArray());
                    currentOffset += cols;
                }
            }
        } else {
            // 对于高维，暂时抛出异常或仅支持 flatten 后的拼接
            throw new UnsupportedOperationException("Concat currently only supports 2D tensors.");
        }
        
        return result;
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // Split yGrad into parts
        List<NdArray> grads = new ArrayList<>();
        int currentOffset = 0;
        int targetDim = dim < 0 ? yGrad.getShape().getDimNum() + dim : dim;
        
        if (yGrad.getShape().getDimNum() == 2) {
            for (int i = 0; i < splitSizes.length; i++) {
                int size = splitSizes[i];
                int rows = inputShapes[i].getRow();
                int cols = inputShapes[i].getColumn();
                
                int startRow, endRow, startCol, endCol;
                
                if (targetDim == 0) {
                    startRow = currentOffset;
                    endRow = currentOffset + size;
                    startCol = 0;
                    endCol = cols;
                    currentOffset += size;
                } else {
                    startRow = 0;
                    endRow = rows;
                    startCol = currentOffset;
                    endCol = currentOffset + size;
                    currentOffset += size;
                }
                
                // 使用 subNdArray 获取子视图，然后复制数据（因为我们需要独立的梯度数组）
                NdArray subGrad = yGrad.subNdArray(startRow, endRow, startCol, endCol);
                
                // subNdArray 返回的可能是 view，我们需要确保它是独立的 NdArray
                // 可以通过 mul(1.0) 或者 clone (如果支持)
                // 这里假设 mul(1) 会产生新数组
                grads.add(subGrad.mulNum(1.0f)); 
            }
        }
        
        return grads;
    }

    @Override
    public int requireInputNum() {
        return -1; // Variable inputs
    }
}

