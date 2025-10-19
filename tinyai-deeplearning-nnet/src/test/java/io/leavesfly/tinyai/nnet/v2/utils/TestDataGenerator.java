package io.leavesfly.tinyai.nnet.v2.utils;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import java.util.Random;

/**
 * 测试数据生成器
 * <p>
 * 提供各种测试数据生成功能：
 * - 随机数组（多种分布）
 * - 分类任务数据集
 * - 回归任务数据集
 *
 * @author leavesfly
 * @version 2.0
 */
public class TestDataGenerator {

    private static final Random RANDOM = new Random(42); // 固定种子确保可复现

    /**
     * 设置随机种子
     *
     * @param seed 随机种子
     */
    public static void setSeed(long seed) {
        RANDOM.setSeed(seed);
    }

    /**
     * 生成随机NdArray（均匀分布）
     *
     * @param shape 数组形状
     * @param min   最小值
     * @param max   最大值
     * @return 随机数组
     */
    public static NdArray randomUniform(Shape shape, double min, double max) {
        int total = shape.getTotal();
        double[] data = new double[total];
        for (int i = 0; i < total; i++) {
            data[i] = min + (max - min) * RANDOM.nextDouble();
        }
        return NdArray.of(data, shape);
    }

    /**
     * 生成随机NdArray（均匀分布[0, 1)）
     *
     * @param shape 数组形状
     * @return 随机数组
     */
    public static NdArray randomUniform(Shape shape) {
        return randomUniform(shape, 0.0, 1.0);
    }

    /**
     * 生成随机NdArray（正态分布）
     *
     * @param shape 数组形状
     * @param mean  均值
     * @param std   标准差
     * @return 随机数组
     */
    public static NdArray randomNormal(Shape shape, double mean, double std) {
        int total = shape.getTotal();
        double[] data = new double[total];
        for (int i = 0; i < total; i++) {
            data[i] = mean + std * RANDOM.nextGaussian();
        }
        return NdArray.of(data, shape);
    }

    /**
     * 生成随机NdArray（标准正态分布）
     *
     * @param shape 数组形状
     * @return 随机数组
     */
    public static NdArray randomNormal(Shape shape) {
        return randomNormal(shape, 0.0, 1.0);
    }

    /**
     * 生成全零数组
     *
     * @param shape 数组形状
     * @return 全零数组
     */
    public static NdArray zeros(Shape shape) {
        return NdArray.of(new double[shape.getTotal()], shape);
    }

    /**
     * 生成全一数组
     *
     * @param shape 数组形状
     * @return 全一数组
     */
    public static NdArray ones(Shape shape) {
        int total = shape.getTotal();
        double[] data = new double[total];
        for (int i = 0; i < total; i++) {
            data[i] = 1.0;
        }
        return NdArray.of(data, shape);
    }

    /**
     * 生成常量数组
     *
     * @param shape 数组形状
     * @param value 常量值
     * @return 常量数组
     */
    public static NdArray constant(Shape shape, double value) {
        int total = shape.getTotal();
        double[] data = new double[total];
        for (int i = 0; i < total; i++) {
            data[i] = value;
        }
        return NdArray.of(data, shape);
    }

    /**
     * 生成线性序列数组
     *
     * @param start 起始值
     * @param end   结束值（不含）
     * @param step  步长
     * @return 序列数组
     */
    public static NdArray arange(double start, double end, double step) {
        int size = (int) Math.ceil((end - start) / step);
        double[] data = new double[size];
        for (int i = 0; i < size; i++) {
            data[i] = start + i * step;
        }
        return NdArray.of(data, new Shape(size));
    }

    /**
     * 生成线性序列数组（步长为1）
     *
     * @param start 起始值
     * @param end   结束值（不含）
     * @return 序列数组
     */
    public static NdArray arange(double start, double end) {
        return arange(start, end, 1.0);
    }

    /**
     * 生成合成二分类数据集
     * <p>
     * 生成两个高斯分布的点云，用于分类任务测试
     *
     * @param samplesPerClass 每类的样本数
     * @param features        特征维度
     * @param separation      类别间的分离度（越大越容易分类）
     * @return ClassificationDataset对象
     */
    public static ClassificationDataset syntheticBinaryClassification(
            int samplesPerClass, int features, double separation) {

        int totalSamples = samplesPerClass * 2;
        double[][] X = new double[totalSamples][features];
        double[][] y = new double[totalSamples][1];

        // 类别0：中心在 [-separation, -separation, ...]
        for (int i = 0; i < samplesPerClass; i++) {
            for (int j = 0; j < features; j++) {
                X[i][j] = -separation + RANDOM.nextGaussian();
            }
            y[i][0] = 0.0;
        }

        // 类别1：中心在 [+separation, +separation, ...]
        for (int i = 0; i < samplesPerClass; i++) {
            int idx = samplesPerClass + i;
            for (int j = 0; j < features; j++) {
                X[idx][j] = separation + RANDOM.nextGaussian();
            }
            y[idx][0] = 1.0;
        }

        // 打乱数据
        shuffleDataset(X, y);

        return new ClassificationDataset(NdArray.of(X), NdArray.of(y), 2);
    }

    /**
     * 生成合成多分类数据集
     *
     * @param samplesPerClass 每类的样本数
     * @param features        特征维度
     * @param numClasses      类别数
     * @param separation      类别间的分离度
     * @return ClassificationDataset对象
     */
    public static ClassificationDataset syntheticMultiClassification(
            int samplesPerClass, int features, int numClasses, double separation) {

        int totalSamples = samplesPerClass * numClasses;
        double[][] X = new double[totalSamples][features];
        double[][] y = new double[totalSamples][1];

        for (int cls = 0; cls < numClasses; cls++) {
            // 为每个类别生成一个随机中心
            double[] center = new double[features];
            for (int j = 0; j < features; j++) {
                center[j] = separation * (RANDOM.nextDouble() * 2 - 1);
            }

            // 围绕中心生成样本
            for (int i = 0; i < samplesPerClass; i++) {
                int idx = cls * samplesPerClass + i;
                for (int j = 0; j < features; j++) {
                    X[idx][j] = center[j] + RANDOM.nextGaussian();
                }
                y[idx][0] = cls;
            }
        }

        // 打乱数据
        shuffleDataset(X, y);

        return new ClassificationDataset(NdArray.of(X), NdArray.of(y), numClasses);
    }

    /**
     * 生成合成回归数据集
     * <p>
     * y = Xw + b + noise
     *
     * @param samples  样本数
     * @param features 特征维度
     * @param noise    噪声标准差
     * @return RegressionDataset对象
     */
    public static RegressionDataset syntheticLinearRegression(
            int samples, int features, double noise) {

        // 生成随机权重和偏置
        double[] trueWeights = new double[features];
        for (int i = 0; i < features; i++) {
            trueWeights[i] = RANDOM.nextGaussian();
        }
        double trueBias = RANDOM.nextGaussian();

        // 生成输入数据
        double[][] X = new double[samples][features];
        double[][] y = new double[samples][1];

        for (int i = 0; i < samples; i++) {
            // 生成输入
            for (int j = 0; j < features; j++) {
                X[i][j] = RANDOM.nextGaussian();
            }

            // 计算目标值：y = Xw + b + noise
            double target = trueBias;
            for (int j = 0; j < features; j++) {
                target += X[i][j] * trueWeights[j];
            }
            target += noise * RANDOM.nextGaussian();

            y[i][0] = target;
        }

        return new RegressionDataset(
                NdArray.of(X),
                NdArray.of(y),
                NdArray.of(trueWeights, new Shape(features)),
                trueBias
        );
    }

    /**
     * 生成合成非线性回归数据集
     * <p>
     * y = sin(x) + noise
     *
     * @param samples 样本数
     * @param noise   噪声标准差
     * @return RegressionDataset对象
     */
    public static RegressionDataset syntheticNonLinearRegression(int samples, double noise) {
        double[][] X = new double[samples][1];
        double[][] y = new double[samples][1];

        for (int i = 0; i < samples; i++) {
            // 输入范围 [-π, π]
            X[i][0] = -Math.PI + 2 * Math.PI * RANDOM.nextDouble();
            // 目标值：sin(x) + noise
            y[i][0] = Math.sin(X[i][0]) + noise * RANDOM.nextGaussian();
        }

        return new RegressionDataset(NdArray.of(X), NdArray.of(y), null, 0.0);
    }

    /**
     * 打乱数据集
     *
     * @param X 输入数据
     * @param y 标签数据
     */
    private static void shuffleDataset(double[][] X, double[][] y) {
        for (int i = X.length - 1; i > 0; i--) {
            int j = RANDOM.nextInt(i + 1);
            // 交换X[i]和X[j]
            double[] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;
            // 交换y[i]和y[j]
            double[] tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;
        }
    }

    /**
     * 分类数据集
     */
    public static class ClassificationDataset {
        private final NdArray X;          // 输入特征 (samples, features)
        private final NdArray y;          // 标签 (samples, 1)
        private final int numClasses;     // 类别数

        public ClassificationDataset(NdArray X, NdArray y, int numClasses) {
            this.X = X;
            this.y = y;
            this.numClasses = numClasses;
        }

        public NdArray getX() {
            return X;
        }

        public NdArray getY() {
            return y;
        }

        public int getNumClasses() {
            return numClasses;
        }

        public int getNumSamples() {
            return X.getShape().getDims()[0];
        }

        public int getNumFeatures() {
            return X.getShape().getDims()[1];
        }
    }

    /**
     * 回归数据集
     */
    public static class RegressionDataset {
        private final NdArray X;            // 输入特征 (samples, features)
        private final NdArray y;            // 目标值 (samples, 1)
        private final NdArray trueWeights;  // 真实权重（如果有）
        private final double trueBias;      // 真实偏置（如果有）

        public RegressionDataset(NdArray X, NdArray y, NdArray trueWeights, double trueBias) {
            this.X = X;
            this.y = y;
            this.trueWeights = trueWeights;
            this.trueBias = trueBias;
        }

        public NdArray getX() {
            return X;
        }

        public NdArray getY() {
            return y;
        }

        public NdArray getTrueWeights() {
            return trueWeights;
        }

        public double getTrueBias() {
            return trueBias;
        }

        public int getNumSamples() {
            return X.getShape().getDims()[0];
        }

        public int getNumFeatures() {
            return X.getShape().getDims()[1];
        }
    }
}
