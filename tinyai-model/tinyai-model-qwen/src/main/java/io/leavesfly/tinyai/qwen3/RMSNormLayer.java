package io.leavesfly.tinyai.qwen3;

import io.leavesfly.tinyai.func.Variable;
import io.leavesfly.tinyai.ndarr.NdArray;
import io.leavesfly.tinyai.nnet.v2.core.Module;
import io.leavesfly.tinyai.nnet.v2.core.Parameter;

/**
 * RMSNorm归一化层
 * 
 * RMSNorm(Root Mean Square Layer Normalization)是LayerNorm的简化版本，
 * 去掉了均值中心化步骤，仅保留缩放操作，计算公式为：
 * 
 * y = (x / RMS(x)) * weight
 * 
 * 其中 RMS(x) = sqrt(mean(x^2) + eps)
 * 
 * 相比LayerNorm的优势：
 * 1. 计算更简单高效
 * 2. 更好的数值稳定性
 * 3. 在大规模语言模型中表现优异
 * 
 * @author leavesfly
 * @version 1.0
 */
public class RMSNormLayer extends Module {
    
    private final int normalizedShape;
    private final double eps;
    private Parameter weight;
    
    /**
     * 构造函数
     * 
     * @param name 层名称
     * @param normalizedShape 归一化的维度大小（通常是hiddenSize）
     * @param eps epsilon值，防止除零，默认1e-6
     */
    public RMSNormLayer(String name, int normalizedShape, double eps) {
        super(name);
        this.normalizedShape = normalizedShape;
        this.eps = eps;
        initializeParameters();
    }
    
    /**
     * 初始化参数
     */
    private void initializeParameters() {
        // 创建可学习的缩放参数weight，初始化为1
        float[] weightData = new float[normalizedShape];
        for (int i = 0; i < normalizedShape; i++) {
            weightData[i] = 1.0f;
        }
        
        NdArray weightArray = NdArray.of(weightData);
        this.weight = new Parameter(weightArray);
        
        // 注册参数
        registerParameter("weight", weight);
    }
    
    /**
     * 前向传播
     * 
     * @param inputs inputs[0]为输入张量 [..., normalizedShape]
     * @return 归一化后的输出 [..., normalizedShape]
     */
    @Override
    public Variable forward(Variable... inputs) {
        if (inputs == null || inputs.length == 0) {
            throw new IllegalArgumentException("RMSNorm输入不能为空");
        }
        
        Variable input = inputs[0];
        validateInput(input);
        
        // 计算 RMS(x) = sqrt(mean(x^2) + eps)
        // 1. x^2
        Variable xSquared = input.mul(input);
        
        // 2. mean(x^2) - 在最后一维求平均
        Variable mean = computeMean(xSquared);
        
        // 3. mean(x^2) + eps
        Variable meanPlusEps = mean.add(new Variable((float) eps));
        
        // 4. sqrt(mean(x^2) + eps)
        Variable rms = meanPlusEps.sqrt();
        
        // 5. x / RMS(x)
        Variable normalized = input.div(rms);
        
        // 6. 乘以可学习的weight参数
        Variable weightVar = new Variable(weight.data());
        Variable output = normalized.mul(weightVar);
        
        return output;
    }
    
    /**
     * 计算最后一维的均值
     * 
     * @param x 输入变量
     * @return 均值变量
     */
    private Variable computeMean(Variable x) {
        NdArray data = x.getValue();
        int[] shape = new int[data.getShape().getDimNum()];
        for (int i = 0; i < shape.length; i++) {
            shape[i] = data.getShape().getDimension(i);
        }
        int lastDim = shape[shape.length - 1];
        
        // 创建输出形状（保持最后一维为1）
        int[] outShape = new int[shape.length];
        System.arraycopy(shape, 0, outShape, 0, shape.length);
        outShape[outShape.length - 1] = 1;  // 保持维度以便广播
        
        // 计算总元素数
        int totalSize = 1;
        for (int dim : shape) {
            totalSize *= dim;
        }
        int batchSize = totalSize / lastDim;
        
        // 计算均值
        float[] inputData = flattenArray(data);
        float[] meanData = new float[batchSize];
        
        for (int i = 0; i < batchSize; i++) {
            float sum = 0.0f;
            int offset = i * lastDim;
            for (int j = 0; j < lastDim; j++) {
                sum += inputData[offset + j];
            }
            meanData[i] = sum / lastDim;
        }
        
        // 重塑为原始形状（最后一维为1）
        NdArray meanArray = NdArray.of(meanData).reshape(io.leavesfly.tinyai.ndarr.Shape.of(outShape));
        return new Variable(meanArray);
    }
    
    /**
     * 将NdArray扁平化为一维数组
     */
    private float[] flattenArray(NdArray array) {
        int totalSize = array.getShape().size();
        float[] result = new float[totalSize];
        int idx = 0;
        
        // 递归获取所有元素
        int[] indices = new int[array.getShape().getDimNum()];
        copyElements(array, indices, 0, result, new int[]{0});
        return result;
    }
    
    /**
     * 递归复制元素
     */
    private void copyElements(NdArray array, int[] indices, int dim, float[] result, int[] counter) {
        if (dim == array.getShape().getDimNum()) {
            result[counter[0]++] = array.get(indices);
            return;
        }
        
        for (int i = 0; i < array.getShape().getDimension(dim); i++) {
            indices[dim] = i;
            copyElements(array, indices, dim + 1, result, counter);
        }
    }
    
    /**
     * 验证输入
     */
    private void validateInput(Variable input) {
        NdArray data = input.getValue();
        int dimNum = data.getShape().getDimNum();
        
        if (dimNum < 1) {
            throw new IllegalArgumentException("输入至少需要1维");
        }
        
        int lastDim = data.getShape().getDimension(dimNum - 1);
        if (lastDim != normalizedShape) {
            throw new IllegalArgumentException(
                String.format("输入最后一维(%d)必须等于normalizedShape(%d)", 
                    lastDim, normalizedShape)
            );
        }
    }
    
    /**
     * 获取归一化维度
     */
    public int getNormalizedShape() {
        return normalizedShape;
    }
    
    /**
     * 获取epsilon值
     */
    public double getEps() {
        return eps;
    }
    
    /**
     * 获取weight参数
     */
    public Parameter getWeight() {
        return weight;
    }
    
    @Override
    public String toString() {
        return String.format("RMSNormLayer{name='%s', normalizedShape=%d, eps=%.1e}", 
            name, normalizedShape, eps);
    }
}
