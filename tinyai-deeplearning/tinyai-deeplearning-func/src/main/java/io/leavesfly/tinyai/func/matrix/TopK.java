package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.*;

/**
 * TopK 算子
 * <p>
 * 返回沿指定轴最大的 k 个值及其索引。
 * forward: returns [values, indices]
 * backward: 仅对 values 回传梯度，Scatter 到原位置。
 */
public class TopK extends Function {

    private final int k;
    private final int axis;
    private final boolean largest;
    private final boolean sorted;

    private Shape inputShape; // 保存输入 shape 用于 backward 初始化
    private NdArray cachedIndices; // 缓存索引用于反向传播

    public TopK(int k, int axis, boolean largest, boolean sorted) {
        this.k = k;
        this.axis = axis;
        this.largest = largest;
        this.sorted = sorted;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        throw new UnsupportedOperationException("TopK is a multi-output function, use forwardMulti instead.");
    }

    @Override
    public NdArray[] forwardMulti(NdArray... inputs) {
        NdArray x = inputs[0];
        this.inputShape = x.getShape();

        // 实现 topk 逻辑
        int[] dims = inputShape.getShapeDims();
        int ndim = dims.length;
        
        // 标准化 axis（支持负数索引）
        int normAxis = axis < 0 ? ndim + axis : axis;
        if (normAxis < 0 || normAxis >= ndim) {
            throw new IllegalArgumentException("axis out of bounds: " + axis);
        }

        // 计算输出形状
        int[] outDims = dims.clone();
        outDims[normAxis] = Math.min(k, dims[normAxis]);
        Shape outShape = Shape.of(outDims);

        // 创建输出数组
        NdArray values = NdArray.zeros(outShape);
        NdArray indices = NdArray.zeros(outShape);

        // 执行 topk 操作
        performTopK(x, values, indices, normAxis);

        // 缓存索引用于反向传播
        this.cachedIndices = indices;

        return new NdArray[]{values, indices};
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        throw new UnsupportedOperationException("TopK is a multi-output function, use backwardMulti instead.");
    }

    @Override
    public List<NdArray> backwardMulti(List<NdArray> yGrads) {
        // yGrads[0] 是 values 的梯度
        // yGrads[1] 是 indices 的梯度 (通常为 null 或不可导)
        
        NdArray gradValues = yGrads.get(0);
        
        // 如果 values 梯度为空，则无需反向传播
        if (gradValues == null) {
            return Collections.singletonList(null);
        }

        // 获取 indices (从输出 Variable 中获取)
        // this.outputs[1] 对应 indices Variable
        NdArray indices = this.outputs[1].getValue();

        // 实现 scatter 操作
        // 我们需要把 gradValues scatter 回到 inputShape 的形状中。
        // 创建一个新的全 0 梯度张量
        NdArray gradInput = NdArray.zeros(inputShape);
        
        // 使用 scatter 将梯度填回对应位置
        scatterGradients(gradInput, cachedIndices, gradValues, axis);
        
        return Collections.singletonList(gradInput); 
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    /**
     * 执行 TopK 操作
     */
    private void performTopK(NdArray input, NdArray values, NdArray indices, int normAxis) {
        int[] inDims = input.getShape().getShapeDims();
        int[] outDims = values.getShape().getShapeDims();
        
        // 针对2维情况的优化实现
        if (inDims.length == 2) {
            performTopK2D(input, values, indices, normAxis);
            return;
        }
        
        // 通用多维实现（简化版，仅支持最后一维）
        if (normAxis != inDims.length - 1) {
            throw new UnsupportedOperationException(
                "Currently only support topk on the last axis for multi-dimensional arrays"
            );
        }
        
        performTopKLastAxis(input, values, indices);
    }

    /**
     * 2维数组的 TopK 实现
     */
    private void performTopK2D(NdArray input, NdArray values, NdArray indices, int axis) {
        int rows = input.getShape().getDimension(0);
        int cols = input.getShape().getDimension(1);
        
        if (axis == 1) {
            // 沿列方向（每行取 top-k）
            for (int i = 0; i < rows; i++) {
                float[] rowData = new float[cols];
                for (int j = 0; j < cols; j++) {
                    rowData[j] = input.get(i, j);
                }
                
                IndexValue[] topKResults = findTopK(rowData, k, largest, sorted);
                
                for (int j = 0; j < topKResults.length; j++) {
                    values.set(topKResults[j].value, i, j);
                    indices.set(topKResults[j].index, i, j);
                }
            }
        } else if (axis == 0) {
            // 沿行方向（每列取 top-k）
            for (int j = 0; j < cols; j++) {
                float[] colData = new float[rows];
                for (int i = 0; i < rows; i++) {
                    colData[i] = input.get(i, j);
                }
                
                IndexValue[] topKResults = findTopK(colData, k, largest, sorted);
                
                for (int i = 0; i < topKResults.length; i++) {
                    values.set(topKResults[i].value, i, j);
                    indices.set(topKResults[i].index, i, j);
                }
            }
        }
    }

    /**
     * 沿最后一维的 TopK 实现（通用多维）
     */
    private void performTopKLastAxis(NdArray input, NdArray values, NdArray indices) {
        int[] inDims = input.getShape().getShapeDims();
        int lastDim = inDims[inDims.length - 1];
        
        // 计算前面维度的总数
        int outerSize = 1;
        for (int i = 0; i < inDims.length - 1; i++) {
            outerSize *= inDims[i];
        }
        
        // 对每个外部位置执行 topk
        for (int outer = 0; outer < outerSize; outer++) {
            float[] data = new float[lastDim];
            
            // 提取数据
            for (int i = 0; i < lastDim; i++) {
                int[] coords = computeCoordinates(outer, i, inDims);
                data[i] = input.get(coords);
            }
            
            // 找到 top-k
            IndexValue[] topKResults = findTopK(data, k, largest, sorted);
            
            // 写回结果
            for (int i = 0; i < topKResults.length; i++) {
                int[] coords = computeCoordinates(outer, i, values.getShape().getShapeDims());
                values.set(topKResults[i].value, coords);
                indices.set(topKResults[i].index, coords);
            }
        }
    }

    /**
     * 找到数组中的 top-k 元素
     */
    private IndexValue[] findTopK(float[] data, int k, boolean largest, boolean sorted) {
        List<IndexValue> indexValues = new ArrayList<>();
        for (int i = 0; i < data.length; i++) {
            indexValues.add(new IndexValue(i, data[i]));
        }
        
        // 排序
        if (largest) {
            indexValues.sort((a, b) -> Float.compare(b.value, a.value)); // 降序
        } else {
            indexValues.sort((a, b) -> Float.compare(a.value, b.value)); // 升序
        }
        
        // 取前 k 个
        int actualK = Math.min(k, data.length);
        IndexValue[] result = new IndexValue[actualK];
        for (int i = 0; i < actualK; i++) {
            result[i] = indexValues.get(i);
        }
        
        // 如果需要排序，结果已经是排序的；否则可以按索引重新排序
        if (!sorted && actualK > 0) {
            Arrays.sort(result, Comparator.comparingInt(iv -> iv.index));
        }
        
        return result;
    }

    /**
     * 将梯度 scatter 回输入位置
     */
    private void scatterGradients(NdArray gradInput, NdArray indices, NdArray gradValues, int axis) {
        int[] inDims = gradInput.getShape().getShapeDims();
        int[] outDims = gradValues.getShape().getShapeDims();
        
        // 针对2维情况的优化实现
        if (inDims.length == 2) {
            scatterGradients2D(gradInput, indices, gradValues, axis);
            return;
        }
        
        // 通用多维实现（简化版）
        scatterGradientsND(gradInput, indices, gradValues);
    }

    /**
     * 2维梯度 scatter
     */
    private void scatterGradients2D(NdArray gradInput, NdArray indices, NdArray gradValues, int axis) {
        int rows = gradInput.getShape().getDimension(0);
        int cols = gradInput.getShape().getDimension(1);
        
        if (axis == 1) {
            // 沿列方向
            for (int i = 0; i < gradValues.getShape().getDimension(0); i++) {
                for (int j = 0; j < gradValues.getShape().getDimension(1); j++) {
                    int idx = (int) indices.get(i, j);
                    float grad = gradValues.get(i, j);
                    float currentGrad = gradInput.get(i, idx);
                    gradInput.set(currentGrad + grad, i, idx);
                }
            }
        } else if (axis == 0) {
            // 沿行方向
            for (int i = 0; i < gradValues.getShape().getDimension(0); i++) {
                for (int j = 0; j < gradValues.getShape().getDimension(1); j++) {
                    int idx = (int) indices.get(i, j);
                    float grad = gradValues.get(i, j);
                    float currentGrad = gradInput.get(idx, j);
                    gradInput.set(currentGrad + grad, idx, j);
                }
            }
        }
    }

    /**
     * 多维梯度 scatter（通用实现）
     */
    private void scatterGradientsND(NdArray gradInput, NdArray indices, NdArray gradValues) {
        int[] outDims = gradValues.getShape().getShapeDims();
        int lastDim = outDims[outDims.length - 1];
        
        // 计算前面维度的总数
        int outerSize = 1;
        for (int i = 0; i < outDims.length - 1; i++) {
            outerSize *= outDims[i];
        }
        
        // 对每个外部位置执行 scatter
        for (int outer = 0; outer < outerSize; outer++) {
            for (int i = 0; i < lastDim; i++) {
                int[] outCoords = computeCoordinates(outer, i, outDims);
                int idx = (int) indices.get(outCoords);
                float grad = gradValues.get(outCoords);
                
                int[] inCoords = computeCoordinates(outer, idx, gradInput.getShape().getShapeDims());
                float currentGrad = gradInput.get(inCoords);
                gradInput.set(currentGrad + grad, inCoords);
            }
        }
    }

    /**
     * 计算多维坐标
     */
    private int[] computeCoordinates(int outer, int inner, int[] dims) {
        int[] coords = new int[dims.length];
        coords[dims.length - 1] = inner;
        
        int temp = outer;
        for (int i = dims.length - 2; i >= 0; i--) {
            coords[i] = temp % dims[i];
            temp /= dims[i];
        }
        
        return coords;
    }

    /**
     * 索引-值对
     */
    private static class IndexValue {
        final int index;
        final float value;

        IndexValue(int index, float value) {
            this.index = index;
            this.value = value;
        }
    }
}
