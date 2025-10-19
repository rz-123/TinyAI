package io.leavesfly.tinyai.nnet.v2.layer.norm;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;
import io.leavesfly.tinyai.nnet.v2.utils.AssertHelper;
import io.leavesfly.tinyai.nnet.v2.utils.TestDataGenerator;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * BatchNorm1d层的单元测试
 *
 * @author leavesfly
 * @version 2.0
 */
class BatchNorm1dTest {

    private BatchNorm1d bn;
    private static final int NUM_FEATURES = 4;
    private static final int BATCH_SIZE = 8;

    @BeforeEach
    void setUp() {
        TestDataGenerator.setSeed(42);
        bn = new BatchNorm1d("test_bn", NUM_FEATURES);
    }

    @Test
    void testConstruction() {
        assertNotNull(bn);
        assertEquals(NUM_FEATURES, bn.getGamma().data().getShape().getTotal());
        assertEquals(NUM_FEATURES, bn.getBeta().data().getShape().getTotal());
        assertNotNull(bn.getRunningMean());
        assertNotNull(bn.getRunningVar());
    }

    @Test
    void testParameterInitialization() {
        // gamma应该初始化为1
        AssertHelper.assertAllEquals(1.0, bn.getGamma().data(), "Gamma initialization");

        // beta应该初始化为0
        AssertHelper.assertAllZeros(bn.getBeta().data(), "Beta initialization");

        // running_mean应该初始化为0
        AssertHelper.assertAllZeros(bn.getRunningMean(), "Running mean initialization");

        // running_var应该初始化为1
        AssertHelper.assertAllEquals(1.0, bn.getRunningVar(), "Running var initialization");
    }

    @Test
    void testTrainingModeNormalization() {
        bn.train();

        // 生成测试数据：均值=5, 标准差=2
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
        Variable x = new Variable(input);

        // 前向传播
        Variable output = bn.forward(x);
        NdArray outputData = output.getValue();

        // 验证输出形状
        AssertHelper.assertShapeEquals(input.getShape(), outputData.getShape());

        // 验证输出满足归一化（均值≈0，方差≈1）
        // 注意：由于应用了gamma和beta，需要考虑它们的影响
        // gamma=1, beta=0时，输出应该是归一化的
        NdArray featureMean = computeFeatureMean(outputData);
        NdArray featureVar = computeFeatureVar(outputData);

        AssertHelper.assertAllEquals(0.0, featureMean, 0.2, "Output mean after normalization");
        AssertHelper.assertAllEquals(1.0, featureVar, 0.3, "Output variance after normalization");
    }

    @Test
    void testEvalModeUsesRunningStats() {
        // 先在训练模式下处理一些数据
        bn.train();

        for (int i = 0; i < 10; i++) {
            NdArray input = TestDataGenerator.randomNormal(
                    Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
            Variable x = new Variable(input);
            bn.forward(x);
        }

        // 保存训练后的running stats
        NdArray trainedRunningMean = NdArray.of(
                bn.getRunningMean().getArray().clone(),
                bn.getRunningMean().getShape()
        );
        NdArray trainedRunningVar = NdArray.of(
                bn.getRunningVar().getArray().clone(),
                bn.getRunningVar().getShape()
        );

        // 切换到推理模式
        bn.eval();

        // 在推理模式下前向传播
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
        Variable x = new Variable(input);
        Variable output = bn.forward(x);

        // 验证running stats没有变化
        AssertHelper.assertArrayClose(trainedRunningMean, bn.getRunningMean(),
                "Running mean should not change in eval mode");
        AssertHelper.assertArrayClose(trainedRunningVar, bn.getRunningVar(),
                "Running var should not change in eval mode");

        // 验证输出形状正确
        AssertHelper.assertShapeEquals(input.getShape(), output.getValue().getShape());
    }

    @Test
    void testRunningStatsUpdate() {
        bn.train();

        // 初始running stats
        NdArray initialMean = NdArray.of(
                bn.getRunningMean().getArray().clone(),
                bn.getRunningMean().getShape()
        );
        NdArray initialVar = NdArray.of(
                bn.getRunningVar().getArray().clone(),
                bn.getRunningVar().getShape()
        );

        // 前向传播
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
        Variable x = new Variable(input);
        bn.forward(x);

        // 验证running stats已更新
        boolean meanChanged = false;
        boolean varChanged = false;

        double[] newMean = bn.getRunningMean().getArray();
        double[] oldMean = initialMean.getArray();
        for (int i = 0; i < NUM_FEATURES; i++) {
            if (Math.abs(newMean[i] - oldMean[i]) > 1e-6) {
                meanChanged = true;
                break;
            }
        }

        double[] newVar = bn.getRunningVar().getArray();
        double[] oldVar = initialVar.getArray();
        for (int i = 0; i < NUM_FEATURES; i++) {
            if (Math.abs(newVar[i] - oldVar[i]) > 1e-6) {
                varChanged = true;
                break;
            }
        }

        assertTrue(meanChanged, "Running mean should be updated");
        assertTrue(varChanged, "Running var should be updated");
    }

    @Test
    void testGammaBetaEffect() {
        bn.train();

        // 设置gamma=2, beta=3
        double[] gammaData = bn.getGamma().data().getArray();
        double[] betaData = bn.getBeta().data().getArray();
        for (int i = 0; i < NUM_FEATURES; i++) {
            gammaData[i] = 2.0;
            betaData[i] = 3.0;
        }

        // 生成标准化输入（均值=0, 标准差=1）
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 0.0, 1.0);
        Variable x = new Variable(input);

        // 前向传播
        Variable output = bn.forward(x);
        NdArray outputData = output.getValue();

        // 归一化后的数据应该满足：y = 2 * x_normalized + 3
        // 因此输出的均值应该接近3，标准差接近2
        NdArray featureMean = computeFeatureMean(outputData);
        NdArray featureStd = computeFeatureStd(outputData);

        AssertHelper.assertAllEquals(3.0, featureMean, 0.5, "Mean with beta=3");
        AssertHelper.assertAllEquals(2.0, featureStd, 0.5, "Std with gamma=2");
    }

    @Test
    void testBatchSizeOne() {
        bn.train();

        // 批次大小为1时的特殊情况
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(1, NUM_FEATURES), 5.0, 2.0);
        Variable x = new Variable(input);

        // 应该不抛出异常
        assertDoesNotThrow(() -> bn.forward(x));

        Variable output = bn.forward(x);

        // 验证输出形状
        AssertHelper.assertShapeEquals(input.getShape(), output.getValue().getShape());

        // 验证没有NaN或Inf
        AssertHelper.assertFinite(output.getValue(), "Output with batch_size=1");
    }

    @Test
    void testResetRunningStats() {
        bn.train();

        // 更新running stats
        for (int i = 0; i < 5; i++) {
            NdArray input = TestDataGenerator.randomNormal(
                    Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
            Variable x = new Variable(input);
            bn.forward(x);
        }

        // 重置
        bn.resetRunningStats();

        // 验证running stats已重置
        AssertHelper.assertAllZeros(bn.getRunningMean(), "Running mean after reset");
        AssertHelper.assertAllEquals(1.0, bn.getRunningVar(), "Running var after reset");
        assertEquals(0, bn.getNumBatchesTracked(), "Num batches tracked after reset");
    }

    @Test
    void testInvalidInputShape() {
        // 错误的特征维度
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES + 1));
        Variable x = new Variable(input);

        assertThrows(IllegalArgumentException.class, () -> bn.forward(x),
                "Should throw exception for invalid input shape");
    }

    @Test
    void testWithoutAffine() {
        // 创建不带可学习参数的BatchNorm
        BatchNorm1d bnNoAffine = new BatchNorm1d(
                "bn_no_affine", NUM_FEATURES, 1e-5f, 0.1f, false, true);
        bnNoAffine.train();

        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);
        Variable x = new Variable(input);

        Variable output = bnNoAffine.forward(x);
        NdArray outputData = output.getValue();

        // 不带affine，输出应该是纯归一化的（均值≈0，方差≈1）
        NdArray featureMean = computeFeatureMean(outputData);
        NdArray featureVar = computeFeatureVar(outputData);

        AssertHelper.assertAllEquals(0.0, featureMean, 0.2, "Mean without affine");
        AssertHelper.assertAllEquals(1.0, featureVar, 0.3, "Variance without affine");
    }

    @Test
    void testConsistencyAcrossMultipleBatches() {
        bn.eval();

        // 在推理模式下，相同的输入应该产生相同的输出
        NdArray input = TestDataGenerator.randomNormal(
                Shape.of(BATCH_SIZE, NUM_FEATURES), 5.0, 2.0);

        Variable x1 = new Variable(input);
        Variable output1 = bn.forward(x1);

        Variable x2 = new Variable(input);
        Variable output2 = bn.forward(x2);

        AssertHelper.assertArrayClose(output1.getValue(), output2.getValue(),
                "Outputs should be identical for same input in eval mode");
    }

    // 辅助方法：计算特征维度的均值
    private NdArray computeFeatureMean(NdArray data) {
        int[] dims = data.getShape().getDims();
        int batchSize = dims[0];
        int features = dims[1];

        double[] mean = new double[features];
        double[][] matrix = data.getMatrix();

        for (int j = 0; j < features; j++) {
            double sum = 0.0;
            for (int i = 0; i < batchSize; i++) {
                sum += matrix[i][j];
            }
            mean[j] = sum / batchSize;
        }

        return NdArray.of(mean, Shape.of(features));
    }

    // 辅助方法：计算特征维度的方差
    private NdArray computeFeatureVar(NdArray data) {
        int[] dims = data.getShape().getDims();
        int batchSize = dims[0];
        int features = dims[1];

        double[][] matrix = data.getMatrix();
        NdArray meanArray = computeFeatureMean(data);
        double[] mean = meanArray.getArray();

        double[] variance = new double[features];

        for (int j = 0; j < features; j++) {
            double sumSq = 0.0;
            for (int i = 0; i < batchSize; i++) {
                double diff = matrix[i][j] - mean[j];
                sumSq += diff * diff;
            }
            variance[j] = sumSq / batchSize;
        }

        return NdArray.of(variance, Shape.of(features));
    }

    // 辅助方法：计算特征维度的标准差
    private NdArray computeFeatureStd(NdArray data) {
        NdArray var = computeFeatureVar(data);
        double[] variance = var.getArray();
        double[] std = new double[variance.length];

        for (int i = 0; i < variance.length; i++) {
            std[i] = Math.sqrt(variance[i]);
        }

        return NdArray.of(std, var.getShape());
    }
}
