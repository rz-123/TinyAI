package io.leavesfly.tinyai.nnet.v2.utils;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

import java.util.Map;
import java.util.function.Function;

/**
 * 梯度验证工具类
 * <p>
 * 使用数值梯度方法验证自动微分计算的梯度正确性。
 * 数值梯度公式：grad ≈ [f(x+ε) - f(x-ε)] / (2ε)
 *
 * @author leavesfly
 * @version 2.0
 */
public class GradientChecker {

    /**
     * 默认扰动大小
     */
    public static final double DEFAULT_EPSILON = 1e-5;

    /**
     * 默认梯度误差容差
     */
    public static final double DEFAULT_TOLERANCE = 1e-4;

    /**
     * 严格梯度误差容差
     */
    public static final double STRICT_TOLERANCE = 1e-5;

    /**
     * 宽松梯度误差容差（用于复杂计算）
     */
    public static final double LOOSE_TOLERANCE = 1e-3;

    /**
     * 验证单个变量的梯度
     *
     * @param function  待验证的函数（输入Variable，输出标量Variable）
     * @param input     输入变量
     * @param epsilon   扰动大小
     * @param tolerance 梯度误差容差
     * @return 梯度检验结果
     */
    public static GradientCheckResult checkGradient(
            Function<Variable, Variable> function,
            Variable input,
            double epsilon,
            double tolerance) {

        // 1. 计算解析梯度
        Variable output = function.apply(input);
        output.backward();
        NdArray analyticGrad = input.getGrad();

        if (analyticGrad == null) {
            throw new IllegalStateException("Input has no gradient after backward()");
        }

        // 2. 计算数值梯度
        NdArray numericalGrad = computeNumericalGradient(function, input, epsilon);

        // 3. 比较梯度
        return compareGradients(analyticGrad, numericalGrad, tolerance);
    }

    /**
     * 验证单个变量的梯度（使用默认参数）
     *
     * @param function 待验证的函数
     * @param input    输入变量
     * @return 梯度检验结果
     */
    public static GradientCheckResult checkGradient(
            Function<Variable, Variable> function,
            Variable input) {
        return checkGradient(function, input, DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    /**
     * 验证模块的梯度
     * <p>
     * 对模块的所有参数进行梯度验证
     *
     * @param module    待验证的模块
     * @param input     输入数据
     * @param lossFunc  损失函数（输入模块输出，返回标量）
     * @param epsilon   扰动大小
     * @param tolerance 梯度误差容差
     * @return 所有参数的梯度检验结果
     */
    public static ModuleGradientCheckResult checkModuleGradient(
            Module module,
            Variable input,
            Function<Variable, Variable> lossFunc,
            double epsilon,
            double tolerance) {

        ModuleGradientCheckResult result = new ModuleGradientCheckResult();

        Map<String, Parameter> params = module.namedParameters();

        for (Map.Entry<String, Parameter> entry : params.entrySet()) {
            String paramName = entry.getKey();
            Parameter param = entry.getValue();

            if (param == null || param.data() == null) {
                continue;
            }

            // 清除梯度
            module.clearGrads();

            // 1. 计算解析梯度
            Variable output = module.forward(input);
            Variable loss = lossFunc.apply(output);
            loss.backward();

            NdArray analyticGrad = param.grad();
            if (analyticGrad == null) {
                result.addFailure(paramName, "No gradient computed");
                continue;
            }

            // 2. 计算数值梯度
            NdArray numericalGrad = computeParameterNumericalGradient(
                    module, input, param, lossFunc, epsilon);

            // 3. 比较梯度
            GradientCheckResult gradResult = compareGradients(analyticGrad, numericalGrad, tolerance);
            result.addResult(paramName, gradResult);
        }

        return result;
    }

    /**
     * 验证模块的梯度（使用默认参数）
     *
     * @param module   待验证的模块
     * @param input    输入数据
     * @param lossFunc 损失函数
     * @return 梯度检验结果
     */
    public static ModuleGradientCheckResult checkModuleGradient(
            Module module,
            Variable input,
            Function<Variable, Variable> lossFunc) {
        return checkModuleGradient(module, input, lossFunc, DEFAULT_EPSILON, DEFAULT_TOLERANCE);
    }

    /**
     * 计算数值梯度
     *
     * @param function 待求导函数
     * @param input    输入变量
     * @param epsilon  扰动大小
     * @return 数值梯度
     */
    private static NdArray computeNumericalGradient(
            Function<Variable, Variable> function,
            Variable input,
            double epsilon) {

        NdArray inputData = input.getValue();
        double[] data = inputData.getArray();
        double[] numericalGrad = new double[data.length];

        // 对每个元素计算数值梯度
        for (int i = 0; i < data.length; i++) {
            double originalValue = data[i];

            // f(x + ε)
            data[i] = originalValue + epsilon;
            Variable inputPlus = new Variable(NdArray.of(data.clone(), inputData.getShape()));
            double valuePlus = function.apply(inputPlus).getValue().getArray()[0];

            // f(x - ε)
            data[i] = originalValue - epsilon;
            Variable inputMinus = new Variable(NdArray.of(data.clone(), inputData.getShape()));
            double valueMinus = function.apply(inputMinus).getValue().getArray()[0];

            // 计算数值梯度
            numericalGrad[i] = (valuePlus - valueMinus) / (2.0 * epsilon);

            // 恢复原始值
            data[i] = originalValue;
        }

        return NdArray.of(numericalGrad, inputData.getShape());
    }

    /**
     * 计算模块参数的数值梯度
     *
     * @param module   模块
     * @param input    输入数据
     * @param param    待验证的参数
     * @param lossFunc 损失函数
     * @param epsilon  扰动大小
     * @return 数值梯度
     */
    private static NdArray computeParameterNumericalGradient(
            Module module,
            Variable input,
            Parameter param,
            Function<Variable, Variable> lossFunc,
            double epsilon) {

        NdArray paramData = param.data();
        double[] data = paramData.getArray();
        double[] numericalGrad = new double[data.length];

        // 对每个参数元素计算数值梯度
        for (int i = 0; i < data.length; i++) {
            double originalValue = data[i];

            // f(θ + ε)
            data[i] = originalValue + epsilon;
            Variable output = module.forward(input);
            double lossPlus = lossFunc.apply(output).getValue().getArray()[0];

            // f(θ - ε)
            data[i] = originalValue - epsilon;
            output = module.forward(input);
            double lossMinus = lossFunc.apply(output).getValue().getArray()[0];

            // 计算数值梯度
            numericalGrad[i] = (lossPlus - lossMinus) / (2.0 * epsilon);

            // 恢复原始值
            data[i] = originalValue;
        }

        return NdArray.of(numericalGrad, paramData.getShape());
    }

    /**
     * 比较解析梯度和数值梯度
     *
     * @param analyticGrad  解析梯度
     * @param numericalGrad 数值梯度
     * @param tolerance     容差
     * @return 梯度检验结果
     */
    private static GradientCheckResult compareGradients(
            NdArray analyticGrad,
            NdArray numericalGrad,
            double tolerance) {

        double[] analytic = analyticGrad.getArray();
        double[] numerical = numericalGrad.getArray();

        double maxError = 0.0;
        double avgError = 0.0;
        int errorCount = 0;

        for (int i = 0; i < analytic.length; i++) {
            double error = Math.abs(analytic[i] - numerical[i]);
            maxError = Math.max(maxError, error);
            avgError += error;

            if (error > tolerance) {
                errorCount++;
            }
        }

        avgError /= analytic.length;
        boolean passed = maxError <= tolerance;

        return new GradientCheckResult(passed, maxError, avgError, errorCount, tolerance);
    }

    /**
     * 梯度检验结果
     */
    public static class GradientCheckResult {
        private final boolean passed;
        private final double maxError;
        private final double avgError;
        private final int errorCount;
        private final double tolerance;

        public GradientCheckResult(boolean passed, double maxError, double avgError,
                                   int errorCount, double tolerance) {
            this.passed = passed;
            this.maxError = maxError;
            this.avgError = avgError;
            this.errorCount = errorCount;
            this.tolerance = tolerance;
        }

        public boolean isPassed() {
            return passed;
        }

        public double getMaxError() {
            return maxError;
        }

        public double getAvgError() {
            return avgError;
        }

        public int getErrorCount() {
            return errorCount;
        }

        public double getTolerance() {
            return tolerance;
        }

        @Override
        public String toString() {
            return String.format("GradientCheck{passed=%s, maxError=%.6e, avgError=%.6e, errorCount=%d, tolerance=%.6e}",
                    passed, maxError, avgError, errorCount, tolerance);
        }
    }

    /**
     * 模块梯度检验结果
     */
    public static class ModuleGradientCheckResult {
        private final Map<String, GradientCheckResult> results = new java.util.LinkedHashMap<>();
        private final Map<String, String> failures = new java.util.LinkedHashMap<>();

        public void addResult(String paramName, GradientCheckResult result) {
            results.put(paramName, result);
        }

        public void addFailure(String paramName, String reason) {
            failures.put(paramName, reason);
        }

        public boolean allPassed() {
            return failures.isEmpty() && results.values().stream().allMatch(GradientCheckResult::isPassed);
        }

        public Map<String, GradientCheckResult> getResults() {
            return results;
        }

        public Map<String, String> getFailures() {
            return failures;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("ModuleGradientCheck{\n");
            sb.append("  passed=").append(allPassed()).append("\n");

            if (!failures.isEmpty()) {
                sb.append("  Failures:\n");
                failures.forEach((name, reason) ->
                        sb.append("    ").append(name).append(": ").append(reason).append("\n"));
            }

            sb.append("  Results:\n");
            results.forEach((name, result) ->
                    sb.append("    ").append(name).append(": ").append(result).append("\n"));

            sb.append("}");
            return sb.toString();
        }
    }
}
