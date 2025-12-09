package io.leavesfly.tinyai.ndarr;


/**
 * NdArray工具类，提供数组操作的辅助方法
 * <p>
 * 本工具类只依赖NdArray接口定义的方法，不依赖具体实现，确保代码的可移植性。
 * 所有操作都经过性能优化，使用底层数组操作以提高效率。
 */
public class NdArrayUtil {

    /**
     * 按照指定轴对多个NdArray进行合并
     *
     * @param axis     合并的轴向，0表示按第一个维度合并，1表示按第二个维度合并，以此类推
     * @param ndArrays 需要合并的NdArray数组
     * @return 合并后的NdArray
     * @throws IllegalArgumentException 当输入参数不合法时抛出
     */
    public static NdArray merge(int axis, NdArray... ndArrays) {
        // 验证输入参数
        if (ndArrays == null || ndArrays.length == 0) {
            throw new IllegalArgumentException("至少需要一个NdArray进行合并");
        }

        // 如果只有一个数组，直接返回副本（使用高效复制）
        if (ndArrays.length == 1) {
            return copyNdArray(ndArrays[0]);
        }

        // 获取第一个数组作为参考
        NdArray first = ndArrays[0];
        Shape firstShape = first.getShape();

        // 验证轴参数的有效性
        if (axis < 0 || axis >= firstShape.getDimNum()) {
            throw new RuntimeException("axis参数超出数组维度范围: " + axis);
        }

        // 验证所有数组的形状兼容性（除了指定的合并轴）
        validateMergeCompatibility(ndArrays, axis);

        // 计算合并后的新形状
        Shape mergedShape = calculateMergedShape(ndArrays, axis);

        // 创建结果数组
        NdArray result = NdArray.of(mergedShape);

        // 执行合并操作（统一使用底层数组操作，不区分矩阵和多维数组）
        mergeArrays(result, ndArrays, axis);

        return result;
    }

    /**
     * 高效复制NdArray（不依赖具体实现）
     *
     * @param source 源数组
     * @return 复制后的新数组
     */
    private static NdArray copyNdArray(NdArray source) {
        float[] sourceData = source.getArray();
        float[] newData = new float[sourceData.length];
        System.arraycopy(sourceData, 0, newData, 0, sourceData.length);
        return NdArray.of(newData, source.getShape());
    }

    /**
     * 验证数组合并的兼容性
     *
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     * @throws IllegalArgumentException 当数组不兼容时抛出
     */
    private static void validateMergeCompatibility(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();
        int dimNum = firstShape.getDimNum();

        for (int i = 1; i < ndArrays.length; i++) {
            Shape currentShape = ndArrays[i].getShape();

            // 检查维度数是否相同
            if (currentShape.getDimNum() != dimNum) {
                throw new IllegalArgumentException(
                        String.format("数组%d的维度数(%d)与第一个数组的维度数(%d)不匹配",
                                i, currentShape.getDimNum(), dimNum));
            }

            // 检查除合并轴外的其他维度是否相同
            for (int dim = 0; dim < dimNum; dim++) {
                if (dim != axis) {
                    int firstDimSize = firstShape.getDimension(dim);
                    int currentDimSize = currentShape.getDimension(dim);
                    if (currentDimSize != firstDimSize) {
                        throw new IllegalArgumentException(
                                String.format("数组%d在维度%d上的大小(%d)与第一个数组(%d)不匹配",
                                        i, dim, currentDimSize, firstDimSize));
                    }
                }
            }
        }
    }

    /**
     * 计算合并后的形状
     *
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     * @return 合并后的形状
     */
    private static Shape calculateMergedShape(NdArray[] ndArrays, int axis) {
        Shape firstShape = ndArrays[0].getShape();
        int dimNum = firstShape.getDimNum();
        int[] newDimensions = new int[dimNum];

        // 复制所有维度
        for (int i = 0; i < dimNum; i++) {
            newDimensions[i] = firstShape.getDimension(i);
        }

        // 计算合并轴上的总大小
        int totalSize = 0;
        for (NdArray array : ndArrays) {
            totalSize += array.getShape().getDimension(axis);
        }
        newDimensions[axis] = totalSize;

        return Shape.of(newDimensions);
    }

    /**
     * 合并多个数组到结果数组中（统一实现，不区分矩阵和多维数组）
     * <p>
     * 使用底层数组操作，通过计算块大小和偏移量来实现高效合并。
     * <p>
     * 合并策略：
     * - 对于axis=0（按第一个维度合并）：数据是连续存储的，可以直接复制
     * - 对于axis>0（按其他维度合并）：需要按块复制，每个块对应合并轴之前的一个"切片"
     *
     * @param result   结果数组
     * @param ndArrays 待合并的数组
     * @param axis     合并轴
     */
    private static void mergeArrays(NdArray result, NdArray[] ndArrays, int axis) {
        Shape resultShape = result.getShape();
        float[] resultData = result.getArray();
        int dimNum = resultShape.getDimNum();

        // 计算合并轴之前和之后的元素数量（用于确定块大小）
        // elementsBeforeAxis: 合并轴之前所有维度的乘积（需要复制的块数）
        // elementsAfterAxis: 合并轴之后所有维度的乘积（每个块中合并轴维度的步长）
        int elementsBeforeAxis = 1;
        for (int i = 0; i < axis; i++) {
            elementsBeforeAxis *= resultShape.getDimension(i);
        }

        int elementsAfterAxis = 1;
        for (int i = axis + 1; i < dimNum; i++) {
            elementsAfterAxis *= resultShape.getDimension(i);
        }

        // 合并轴上的步长（每个块中，合并轴维度上每个单位对应的元素数）
        int axisStride = elementsAfterAxis;

        // 在结果数组中，合并轴上的当前偏移量
        int resultAxisOffset = 0;

        // 遍历每个待合并的数组
        for (NdArray sourceArray : ndArrays) {
            float[] sourceData = sourceArray.getArray();
            Shape sourceShape = sourceArray.getShape();
            int sourceAxisSize = sourceShape.getDimension(axis);

            // 源数组中，合并轴维度上的块大小（每个块包含的元素数）
            int sourceBlockSize = sourceAxisSize * axisStride;

            // 结果数组中，合并轴维度上的总大小
            int resultAxisSize = resultShape.getDimension(axis);

            // 对于每个"切片"（合并轴之前的所有维度组合），复制对应的数据块
            for (int slice = 0; slice < elementsBeforeAxis; slice++) {
                // 源数组中的起始位置：slice * sourceBlockSize
                int sourceStart = slice * sourceBlockSize;

                // 结果数组中的起始位置：
                // slice * (resultAxisSize * axisStride) + resultAxisOffset * axisStride
                int resultStart = slice * (resultAxisSize * axisStride) + resultAxisOffset * axisStride;

                // 复制数据块
                System.arraycopy(sourceData, sourceStart, resultData, resultStart, sourceBlockSize);
            }

            // 更新结果数组中合并轴上的偏移量
            resultAxisOffset += sourceAxisSize;
        }
    }

    /**
     * 将浮点数组转换为整型数组
     *
     * @param src 浮点数组
     * @return 整型数组
     */
    public static int[] toInt(float[] src) {
        if (src == null) {
            return null;
        }
        int[] res = new int[src.length];
        for (int i = 0; i < src.length; i++) {
            res[i] = (int) src[i];
        }
        return res;
    }

    /**
     * 生成从0开始的连续整数序列
     *
     * @param size 序列长度
     * @return 连续整数数组
     * @throws IllegalArgumentException 当size小于0时抛出
     */
    public static int[] getSeq(int size) {
        if (size < 0) {
            throw new IllegalArgumentException("序列长度不能为负数: " + size);
        }
        int[] seq = new int[size];
        for (int i = 0; i < size; i++) {
            seq[i] = i;
        }
        return seq;
    }
}
