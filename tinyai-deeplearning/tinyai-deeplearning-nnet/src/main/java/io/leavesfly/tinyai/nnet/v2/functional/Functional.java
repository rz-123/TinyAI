package io.leavesfly.tinyai.nnet.v2.functional;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.NdArrayUtil;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.layer.dnn.Dropout;
import io.leavesfly.tinyai.util.Config;

import java.util.Objects;

/**
 * A lightweight, stateless functional API mirroring常用层的前向计算。
 * <p>
 * 设计目标：
 * <ul>
 *   <li>与PyTorch的 {@code torch.nn.functional} 类似，提供直接的函数式调用</li>
 *   <li>不持有内部状态，参数/缓冲由调用方显式传入</li>
 *   <li>复用已有的自动微分体系（Variable/Function），梯度可自动回传到传入的参数</li>
 * </ul>
 *
 * 注意：本类仅封装核心高频算子，更多算子可按需补充。
 */
public final class Functional {

    private Functional() {
        // 工具类不允许实例化
    }

    /* --------------------------------- 激活函数 --------------------------------- */

    public static Variable relu(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.relu();
    }

    public static Variable sigmoid(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.sigmoid();
    }

    public static Variable tanh(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.tanh();
    }

    public static Variable softmax(Variable input) {
        Objects.requireNonNull(input, "input cannot be null");
        return input.softMax();
    }

    /* --------------------------------- 线性与正则 --------------------------------- */

    public static Variable linear(Variable input, Variable weight) {
        return linear(input, weight, null);
    }

    /**
     * 线性映射：y = x W^T + b
     */
    public static Variable linear(Variable input, Variable weight, Variable bias) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(weight, "weight cannot be null");
        return input.linear(weight, bias);
    }

    /**
     * Dropout（功能等价于Dropout Module）。
     *
     * @param input    输入
     * @param p        丢弃概率，取值 [0,1)
     * @param training 是否处于训练模式（推理模式将直接返回输入）
     */
    public static Variable dropout(Variable input, float p, boolean training) {
        Objects.requireNonNull(input, "input cannot be null");
        Dropout dropout = new Dropout("functional_dropout", p);
        dropout.train(training);
        return dropout.forward(input);
    }

    /**
     * Dropout，训练模式默认取决于全局Config.train。
     */
    public static Variable dropout(Variable input, float p) {
        return dropout(input, p, Boolean.TRUE.equals(Config.train));
    }

    /* --------------------------------- 归一化 --------------------------------- */

    /**
     * LayerNorm 前向：y = gamma * (x - mean) / sqrt(var + eps) + beta
     * gamma/beta 由调用方提供，可传入可学习的Parameter或常量Variable。
     */
    public static Variable layerNorm(Variable input, Variable gamma, Variable beta, float eps) {
        Objects.requireNonNull(input, "input cannot be null");
        Objects.requireNonNull(gamma, "gamma cannot be null");
        Objects.requireNonNull(beta, "beta cannot be null");

        Variable mean = input.mean(-1, true);
        Variable variance = input.var(-1, true);
        Variable normalized = input.sub(mean).div(variance.add(new Variable(eps)).sqrt());
        return normalized.mul(gamma).add(beta);
    }

    public static Variable layerNorm(Variable input, Variable gamma, Variable beta) {
        return layerNorm(input, gamma, beta, 1e-5f);
    }

    /**
     * BatchNorm1d 前向。支持训练/推理，两种模式下的行为与 BatchNorm1d Module 保持一致。
     *
     * @param input       输入 (batch, features[, length])
     * @param gamma       缩放参数，可为空（等价于不使用affine）
     * @param beta        平移参数，可为空（等价于不使用affine）
     * @param runningMean 运行时均值缓冲，可为空（不跟踪统计量时）
     * @param runningVar  运行时方差缓冲，可为空（不跟踪统计量时）
     * @param training    是否训练模式
     * @param momentum    运行时统计量动量
     * @param eps         数值稳定项
     */
    public static Variable batchNorm1d(Variable input,
                                       Variable gamma,
                                       Variable beta,
                                       NdArray runningMean,
                                       NdArray runningVar,
                                       boolean training,
                                       float momentum,
                                       float eps) {
        Objects.requireNonNull(input, "input cannot be null");

        int[] dims = input.getValue().getShape().getShapeDims();
        if (dims.length < 2) {
            throw new IllegalArgumentException("BatchNorm1d expects input with at least 2 dimensions");
        }

        Variable mean;
        Variable var;

        if (training) {
            mean = input.mean(0, false);
            var = input.var(0, false);

            if (runningMean != null && runningVar != null) {
                float[] rm = runningMean.getArray();
                float[] rv = runningVar.getArray();
                float[] bm = mean.getValue().getArray();
                float[] bv = var.getValue().getArray();

                if (rm.length != bm.length || rv.length != bv.length) {
                    throw new IllegalArgumentException("runningMean/runningVar size mismatch with input features");
                }

                for (int i = 0; i < rm.length; i++) {
                    rm[i] = (1 - momentum) * rm[i] + momentum * bm[i];
                    rv[i] = (1 - momentum) * rv[i] + momentum * bv[i];
                }
            }
        } else {
            if (runningMean == null || runningVar == null) {
                throw new IllegalArgumentException("runningMean and runningVar are required in eval mode");
            }
            mean = new Variable(runningMean);
            var = new Variable(runningVar);
        }

        Variable normalized = input.sub(mean).div(var.add(new Variable(eps)).sqrt());
        if (gamma != null) {
            normalized = normalized.mul(gamma);
        }
        if (beta != null) {
            normalized = normalized.add(beta);
        }
        return normalized;
    }

    public static Variable batchNorm1d(Variable input,
                                       Variable gamma,
                                       Variable beta,
                                       NdArray runningMean,
                                       NdArray runningVar,
                                       boolean training) {
        return batchNorm1d(input, gamma, beta, runningMean, runningVar, training, 0.1f, 1e-5f);
    }

    /* --------------------------------- 嵌入 --------------------------------- */

    /**
     * 词嵌入查找，等价于 Embedding Module 的前向。
     *
     * @param indices 索引张量，支持1D或2D
     * @param weight  形状为 (num_embeddings, embedding_dim) 的嵌入矩阵
     */
    public static Variable embedding(Variable indices, Variable weight) {
        Objects.requireNonNull(indices, "indices cannot be null");
        Objects.requireNonNull(weight, "weight cannot be null");

        NdArray idxValue = indices.getValue();
        int dim = idxValue.getShape().getDimNum();
        NdArray weightData = weight.getValue();
        int embeddingDim = weightData.getShape().getColumn();

        if (dim == 1) {
            int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[0]);
            return weight.getItem(slices, null);
        } else if (dim == 2) {
            int batchSize = idxValue.getShape().getRow();
            int seqLen = idxValue.getShape().getColumn();

            NdArray result = NdArray.zeros(Shape.of(batchSize, seqLen, embeddingDim));
            for (int i = 0; i < batchSize; i++) {
                int[] slices = NdArrayUtil.toInt(idxValue.getMatrix()[i]);
                Variable embRow = weight.getItem(slices, null);
                NdArray embVal = embRow.getValue();
                for (int j = 0; j < seqLen; j++) {
                    for (int k = 0; k < embeddingDim; k++) {
                        result.set(embVal.get(j, k), i, j, k);
                    }
                }
            }

            if (seqLen == 1) {
                result = result.reshape(Shape.of(batchSize, embeddingDim));
            }
            return new Variable(result);
        } else {
            throw new IllegalArgumentException("Embedding only supports 1D or 2D index tensors, got shape: " + idxValue.getShape());
        }
    }
}

