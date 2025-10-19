package io.leavesfly.tinyai.nnet.v2.utils;

import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.ndarr.Shape;

import static org.junit.jupiter.api.Assertions.*;

/**
 * 测试断言辅助工具类
 * <p>
 * 提供浮点数比较、数组比较、形状比较等专用断言方法，
 * 简化神经网络模块的单元测试编写。
 *
 * @author leavesfly
 * @version 2.0
 */
public class AssertHelper {

    /**
     * 默认浮点数容差
     */
    public static final double DEFAULT_TOLERANCE = 1e-5;

    /**
     * 宽松浮点数容差（用于统计检验等）
     */
    public static final double LOOSE_TOLERANCE = 0.1;

    /**
     * 严格浮点数容差（用于精确计算验证）
     */
    public static final double STRICT_TOLERANCE = 1e-7;

    /**
     * 断言两个浮点数近似相等
     *
     * @param expected  期望值
     * @param actual    实际值
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertClose(double expected, double actual, double tolerance, String message) {
        double diff = Math.abs(expected - actual);
        assertTrue(diff <= tolerance,
                String.format("%s: expected=%.6f, actual=%.6f, diff=%.6f, tolerance=%.6f",
                        message, expected, actual, diff, tolerance));
    }

    /**
     * 断言两个浮点数近似相等（使用默认容差）
     *
     * @param expected 期望值
     * @param actual   实际值
     * @param message  错误消息
     */
    public static void assertClose(double expected, double actual, String message) {
        assertClose(expected, actual, DEFAULT_TOLERANCE, message);
    }

    /**
     * 断言两个NdArray的数据近似相等
     *
     * @param expected  期望数组
     * @param actual    实际数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertArrayClose(NdArray expected, NdArray actual, double tolerance, String message) {
        // 首先检查形状
        assertShapeEquals(expected.getShape(), actual.getShape(), message + " (shape mismatch)");

        // 逐元素比较
        double[] expectedData = expected.getArray();
        double[] actualData = actual.getArray();

        for (int i = 0; i < expectedData.length; i++) {
            assertClose(expectedData[i], actualData[i], tolerance,
                    String.format("%s [index=%d]", message, i));
        }
    }

    /**
     * 断言两个NdArray的数据近似相等（使用默认容差）
     *
     * @param expected 期望数组
     * @param actual   实际数组
     * @param message  错误消息
     */
    public static void assertArrayClose(NdArray expected, NdArray actual, String message) {
        assertArrayClose(expected, actual, DEFAULT_TOLERANCE, message);
    }

    /**
     * 断言两个double数组近似相等
     *
     * @param expected  期望数组
     * @param actual    实际数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertArrayClose(double[] expected, double[] actual, double tolerance, String message) {
        assertEquals(expected.length, actual.length, message + " (length mismatch)");

        for (int i = 0; i < expected.length; i++) {
            assertClose(expected[i], actual[i], tolerance,
                    String.format("%s [index=%d]", message, i));
        }
    }

    /**
     * 断言两个double数组近似相等（使用默认容差）
     *
     * @param expected 期望数组
     * @param actual   实际数组
     * @param message  错误消息
     */
    public static void assertArrayClose(double[] expected, double[] actual, String message) {
        assertArrayClose(expected, actual, DEFAULT_TOLERANCE, message);
    }

    /**
     * 断言两个Shape相等
     *
     * @param expected 期望形状
     * @param actual   实际形状
     * @param message  错误消息
     */
    public static void assertShapeEquals(Shape expected, Shape actual, String message) {
        assertEquals(expected.getTotal(), actual.getTotal(),
                message + String.format(" (total: expected=%d, actual=%d)",
                        expected.getTotal(), actual.getTotal()));

        int[] expectedDims = expected.getDims();
        int[] actualDims = actual.getDims();

        assertEquals(expectedDims.length, actualDims.length,
                message + String.format(" (rank: expected=%d, actual=%d)",
                        expectedDims.length, actualDims.length));

        for (int i = 0; i < expectedDims.length; i++) {
            assertEquals(expectedDims[i], actualDims[i],
                    message + String.format(" (dim[%d]: expected=%d, actual=%d)",
                            i, expectedDims[i], actualDims[i]));
        }
    }

    /**
     * 断言两个Shape相等（简化版）
     *
     * @param expected 期望形状
     * @param actual   实际形状
     */
    public static void assertShapeEquals(Shape expected, Shape actual) {
        assertShapeEquals(expected, actual, "Shape mismatch");
    }

    /**
     * 断言NdArray的所有元素都为零
     *
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertAllZeros(NdArray array, double tolerance, String message) {
        double[] data = array.getArray();
        for (int i = 0; i < data.length; i++) {
            assertClose(0.0, data[i], tolerance,
                    String.format("%s [index=%d]", message, i));
        }
    }

    /**
     * 断言NdArray的所有元素都为零（使用默认容差）
     *
     * @param array   待检查数组
     * @param message 错误消息
     */
    public static void assertAllZeros(NdArray array, String message) {
        assertAllZeros(array, DEFAULT_TOLERANCE, message);
    }

    /**
     * 断言NdArray的所有元素都等于指定值
     *
     * @param expected  期望值
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertAllEquals(double expected, NdArray array, double tolerance, String message) {
        double[] data = array.getArray();
        for (int i = 0; i < data.length; i++) {
            assertClose(expected, data[i], tolerance,
                    String.format("%s [index=%d]", message, i));
        }
    }

    /**
     * 断言NdArray的所有元素都等于指定值（使用默认容差）
     *
     * @param expected 期望值
     * @param array    待检查数组
     * @param message  错误消息
     */
    public static void assertAllEquals(double expected, NdArray array, String message) {
        assertAllEquals(expected, array, DEFAULT_TOLERANCE, message);
    }

    /**
     * 断言NdArray的所有元素都在指定范围内
     *
     * @param array   待检查数组
     * @param min     最小值（含）
     * @param max     最大值（含）
     * @param message 错误消息
     */
    public static void assertInRange(NdArray array, double min, double max, String message) {
        double[] data = array.getArray();
        for (int i = 0; i < data.length; i++) {
            double value = data[i];
            assertTrue(value >= min && value <= max,
                    String.format("%s [index=%d]: value=%.6f not in [%.6f, %.6f]",
                            message, i, value, min, max));
        }
    }

    /**
     * 断言NdArray中没有NaN值
     *
     * @param array   待检查数组
     * @param message 错误消息
     */
    public static void assertNoNaN(NdArray array, String message) {
        double[] data = array.getArray();
        for (int i = 0; i < data.length; i++) {
            assertFalse(Double.isNaN(data[i]),
                    String.format("%s [index=%d]: NaN detected", message, i));
        }
    }

    /**
     * 断言NdArray中没有无穷值
     *
     * @param array   待检查数组
     * @param message 错误消息
     */
    public static void assertNoInf(NdArray array, String message) {
        double[] data = array.getArray();
        for (int i = 0; i < data.length; i++) {
            assertFalse(Double.isInfinite(data[i]),
                    String.format("%s [index=%d]: Inf detected", message, i));
        }
    }

    /**
     * 断言NdArray是有限的（无NaN和Inf）
     *
     * @param array   待检查数组
     * @param message 错误消息
     */
    public static void assertFinite(NdArray array, String message) {
        assertNoNaN(array, message);
        assertNoInf(array, message);
    }

    /**
     * 计算NdArray的均值
     *
     * @param array 输入数组
     * @return 均值
     */
    public static double mean(NdArray array) {
        double[] data = array.getArray();
        double sum = 0.0;
        for (double v : data) {
            sum += v;
        }
        return sum / data.length;
    }

    /**
     * 计算NdArray的方差（总体方差）
     *
     * @param array 输入数组
     * @return 方差
     */
    public static double variance(NdArray array) {
        double[] data = array.getArray();
        double mean = mean(array);
        double sumSquaredDiff = 0.0;
        for (double v : data) {
            double diff = v - mean;
            sumSquaredDiff += diff * diff;
        }
        return sumSquaredDiff / data.length;
    }

    /**
     * 计算NdArray的标准差
     *
     * @param array 输入数组
     * @return 标准差
     */
    public static double std(NdArray array) {
        return Math.sqrt(variance(array));
    }

    /**
     * 断言均值接近指定值
     *
     * @param expected  期望均值
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertMeanClose(double expected, NdArray array, double tolerance, String message) {
        double actualMean = mean(array);
        assertClose(expected, actualMean, tolerance,
                String.format("%s (mean)", message));
    }

    /**
     * 断言方差接近指定值
     *
     * @param expected  期望方差
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertVarianceClose(double expected, NdArray array, double tolerance, String message) {
        double actualVar = variance(array);
        assertClose(expected, actualVar, tolerance,
                String.format("%s (variance)", message));
    }

    /**
     * 断言标准差接近指定值
     *
     * @param expected  期望标准差
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertStdClose(double expected, NdArray array, double tolerance, String message) {
        double actualStd = std(array);
        assertClose(expected, actualStd, tolerance,
                String.format("%s (std)", message));
    }

    /**
     * 断言数组满足归一化条件（均值≈0，方差≈1）
     *
     * @param array     待检查数组
     * @param tolerance 容差
     * @param message   错误消息
     */
    public static void assertNormalized(NdArray array, double tolerance, String message) {
        assertMeanClose(0.0, array, tolerance, message);
        assertVarianceClose(1.0, array, tolerance, message);
    }

    /**
     * 断言数组满足归一化条件（使用宽松容差）
     *
     * @param array   待检查数组
     * @param message 错误消息
     */
    public static void assertNormalized(NdArray array, String message) {
        assertNormalized(array, LOOSE_TOLERANCE, message);
    }

    /**
     * 计算数组中非零元素的比例
     *
     * @param array     输入数组
     * @param tolerance 判断为零的容差
     * @return 非零元素比例（0.0-1.0）
     */
    public static double nonZeroRatio(NdArray array, double tolerance) {
        double[] data = array.getArray();
        int nonZeroCount = 0;
        for (double v : data) {
            if (Math.abs(v) > tolerance) {
                nonZeroCount++;
            }
        }
        return (double) nonZeroCount / data.length;
    }

    /**
     * 计算数组中非零元素的比例（使用默认容差）
     *
     * @param array 输入数组
     * @return 非零元素比例（0.0-1.0）
     */
    public static double nonZeroRatio(NdArray array) {
        return nonZeroRatio(array, DEFAULT_TOLERANCE);
    }

    /**
     * 断言数组的非零比例在指定范围内
     *
     * @param array       待检查数组
     * @param expectedMin 期望最小比例
     * @param expectedMax 期望最大比例
     * @param message     错误消息
     */
    public static void assertNonZeroRatioInRange(NdArray array, double expectedMin, double expectedMax, String message) {
        double ratio = nonZeroRatio(array);
        assertTrue(ratio >= expectedMin && ratio <= expectedMax,
                String.format("%s: nonZeroRatio=%.4f not in [%.4f, %.4f]",
                        message, ratio, expectedMin, expectedMax));
    }
}
