package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Collections;
import java.util.List;

/**
 * 切断梯度算子
 * <p>
 * 从计算图中分离变量，停止梯度传播。
 * 返回一个新变量，值与输入相同，但requireGrad=false，不在计算图中。
 */
public class Detach extends Function {

    private Shape inputShape;

    @Override
    public NdArray forward(NdArray... inputs) {
        inputShape = inputs[0].getShape();
        // 复制数组数据
        float[] data = inputs[0].getArray();
        float[] newData = new float[data.length];
        System.arraycopy(data, 0, newData, 0, data.length);
        return NdArray.of(newData, inputShape);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        // detach操作不传播梯度，返回全零梯度阻断梯度传播
        return Collections.singletonList(NdArray.zeros(inputShape));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }
}

