package io.leavesfly.tinyai.func;


import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.util.Config;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 抽象的数学函数基类
 * <p>
 * 在TinyDL深度学习框架中，Function类是所有数学函数操作的基类。
 * 它定义了前向传播和反向传播的接口，并负责构建计算图。
 * 每个函数实例都维护输入变量和输出变量之间的关系。
 */
public abstract class Function implements Serializable {

    /**
     * 函数的输入变量数组
     * 存储传递给该函数的所有输入变量
     */
    protected Variable[] inputs;

    /**
     * 函数的输出变量
     * 存储该函数计算结果的输出变量
     */
    protected Variable output;

    /**
     * 多输出时的输出数组，单输出场景下与output保持一致
     */
    protected Variable[] outputs;

    /**
     * 函数的执行函数，执行函数的前向传播计算并构建计算图
     * <p>
     * 该方法执行以下操作：
     * 1. 验证输入变量数量是否符合要求
     * 2. 从输入变量中提取NdArray值
     * 3. 调用forward方法执行前向传播计算
     * 4. 创建输出变量
     * 5. 在训练模式下构建计算图
     *
     * @param _inputs 输入变量数组
     * @return 计算结果的输出变量
     * @throws RuntimeException 当输入变量数量不符合要求时抛出异常
     */
    public Variable call(Variable... _inputs) {
        // 输入验证：数量匹配且不允许null
        if (requireInputNum() >= 0 && _inputs.length != requireInputNum()) {
            throw new RuntimeException("Function call inputs Variable requireInputNum error! Expected: "
                    + requireInputNum() + ", Actual: " + _inputs.length);
        }
        if (Arrays.stream(_inputs).anyMatch(Objects::isNull)) {
            throw new RuntimeException("Function call inputs Variable cannot be null");
        }

        // 提取NdArray值
        NdArray[] ndArrayInputs = Arrays.stream(_inputs)
                .map(Variable::getValue)
                .toArray(NdArray[]::new);

        // 执行前向传播
        NdArray ndArrayOutput = forward(ndArrayInputs);

        // 创建输出变量
        Variable _output = new Variable(ndArrayOutput);

        // 只在需要构建计算图时挂接
        if (shouldBuildGraph(_inputs)) {
            this.inputs = _inputs;
            this.output = _output;
            this.outputs = new Variable[]{_output};
            _output.setCreator(this);
        }

        return _output;
    }

    /**
     * 多输出函数的执行入口
     *
     * @param _inputs 输入变量数组
     * @return 输出变量数组
     */
    public Variable[] callMulti(Variable... _inputs) {
        if (requireInputNum() >= 0 && _inputs.length != requireInputNum()) {
            throw new RuntimeException("Function call inputs Variable requireInputNum error! Expected: "
                    + requireInputNum() + ", Actual: " + _inputs.length);
        }
        if (Arrays.stream(_inputs).anyMatch(Objects::isNull)) {
            throw new RuntimeException("Function call inputs Variable cannot be null");
        }

        NdArray[] ndArrayInputs = Arrays.stream(_inputs)
                .map(Variable::getValue)
                .toArray(NdArray[]::new);

        NdArray[] ndArrayOutputs = forwardMulti(ndArrayInputs);
        Variable[] _outputs = Arrays.stream(ndArrayOutputs).map(Variable::new).toArray(Variable[]::new);

        if (shouldBuildGraph(_inputs)) {
            this.inputs = _inputs;
            this.outputs = _outputs;
            this.output = _outputs.length > 0 ? _outputs[0] : null;
            for (Variable out : _outputs) {
                out.setCreator(this);
            }
        }

        return _outputs;
    }

    /**
     * 函数的前向传播计算
     * <p>
     * 子类必须实现此方法来定义具体的前向传播计算逻辑。
     * 该方法接收NdArray数组作为输入，返回计算结果的NdArray。
     *
     * @param inputs 输入的NdArray数组
     * @return 前向传播计算结果的NdArray
     */
    public abstract NdArray forward(NdArray... inputs);

    /**
     * 多输出函数的前向传播（默认不支持，子类按需重写）
     */
    public NdArray[] forwardMulti(NdArray... inputs) {
        throw new UnsupportedOperationException("This function does not support multiple outputs");
    }

    /**
     * 函数的反向传播计算（求导）
     * <p>
     * 子类必须实现此方法来定义具体的反向传播计算逻辑。
     * 该方法接收输出变量的梯度，计算并返回输入变量的梯度。
     *
     * @param yGrad 输出变量的梯度
     * @return 输入变量的梯度列表
     */
    public abstract List<NdArray> backward(NdArray yGrad);

    /**
     * 多输出函数的反向传播（默认不支持，子类按需重写）
     *
     * @param yGrads 与输出一一对应的梯度列表
     * @return 输入变量的梯度列表
     */
    public List<NdArray> backwardMulti(List<NdArray> yGrads) {
        throw new UnsupportedOperationException("This function does not support multiple outputs");
    }

    /**
     * 获取函数的输入变量数组
     *
     * @return 输入变量数组
     */
    public Variable[] getInputs() {
        return inputs;
    }

    /**
     * 设置函数的输入变量数组
     *
     * @param inputs 输入变量数组
     */
    public void setInputs(Variable[] inputs) {
        this.inputs = inputs;
    }

    /**
     * 获取函数的输出变量
     *
     * @return 输出变量
     */
    public Variable getOutput() {
        return output;
    }

    public Variable[] getOutputs() {
        return outputs;
    }

    /**
     * 设置函数的输出变量
     *
     * @param output 输出变量
     */
    public void setOutput(Variable output) {
        this.output = output;
    }

    public void setOutputs(Variable[] outputs) {
        this.outputs = outputs;
    }

    /**
     * 获取函数所需的输入参数个数
     * <p>
     * 子类实现此方法来指定函数所需的输入变量数量。
     * 返回-1表示函数可以接受任意数量的输入参数。
     *
     * @return 函数所需的输入参数个数
     */
    public abstract int requireInputNum();

    /**
     * 清理函数资源，断开计算图连接
     * <p>
     * 用于RNN中切断计算图，防止梯度回传过长导致的梯度消失或爆炸问题。
     */
    public void unChain() {
        this.inputs = null;
        this.output = null;
        this.outputs = null;
    }

    /**
     * 当前函数是否返回多输出
     */
    public boolean isMultiOutput() {
        return outputs != null && outputs.length > 1;
    }

    /**
     * 是否需要构建计算图
     */
    protected boolean shouldBuildGraph(Variable[] vars) {
        if (!Config.train) {
            return false;
        }
        return Arrays.stream(vars).anyMatch(v -> v != null && v.isRequireGrad());
    }
}