package io.leavesfly.tinyai.func.matrix;

import io.leavesfly.tinyai.func.Function;
import io.leavesfly.tinyai.ndarr.NdArray;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * 维度置换操作，支持任意阶张量的轴重排。
 * <p>
 * forward: x.transpose(order)
 * backward: 将梯度按逆置换恢复。
 */
public class Permute extends Function {

    private final int[] order;

    public Permute(int... order) {
        this.order = order;
    }

    @Override
    public NdArray forward(NdArray... inputs) {
        validateOrder(inputs[0].getShape().getDimNum());
        return inputs[0].transpose(order);
    }

    @Override
    public List<NdArray> backward(NdArray yGrad) {
        int[] inverse = buildInverseOrder(order);
        return Collections.singletonList(yGrad.transpose(inverse));
    }

    @Override
    public int requireInputNum() {
        return 1;
    }

    private void validateOrder(int dimNum) {
        if (order == null || order.length != dimNum) {
            throw new IllegalArgumentException(
                    String.format("Invalid permute order, expected %d dims but got %s", dimNum, Arrays.toString(order)));
        }
        boolean[] seen = new boolean[dimNum];
        for (int idx : order) {
            if (idx < 0 || idx >= dimNum) {
                throw new IllegalArgumentException("Permute index out of range: " + idx);
            }
            if (seen[idx]) {
                throw new IllegalArgumentException("Duplicate axis in permute order: " + idx);
            }
            seen[idx] = true;
        }
    }

    private int[] buildInverseOrder(int[] order) {
        int[] inverse = new int[order.length];
        for (int i = 0; i < order.length; i++) {
            inverse[order[i]] = i;
        }
        return inverse;
    }
}

