package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 克隆张量算子
 * <p>
 * 深拷贝张量的值和梯度，返回新的叶子节点（不在计算图中）
 */
public class Clone extends Function {

    private Shape inputShape;

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        // 深拷贝数组数据
        float[] data = inputs[0].getArray();
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return NdArray.of(newData, inputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // clone操作传播梯度
        return Collections.singletonList(yGrad);
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

