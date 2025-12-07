package io.leavesfly.tinyai.ml.exception;

import io.leavesfly.tinyai.ndarr.Shape;

/**
 * 参数不匹配异常
 * 
 * @author TinyAI
 * @version 1.0
 */
public class ParameterMismatchException extends ModelException {
    
    private static final long serialVersionUID = 1L;
    
    private final String paramName;
    private final Shape expectedShape;
    private final Shape actualShape;
    
    public ParameterMismatchException(String paramName, Shape expectedShape, Shape actualShape) {
        super("PARAMETER_MISMATCH", 
              String.format("参数 %s 形状不匹配: 期望=%s, 实际=%s", 
                           paramName, expectedShape, actualShape));
        this.paramName = paramName;
        this.expectedShape = expectedShape;
        this.actualShape = actualShape;
    }
    
    public ParameterMismatchException(String message, Shape expectedShape, Shape actualShape, boolean isContext) {
        super("PARAMETER_MISMATCH", message);
        this.paramName = isContext ? null : message;
        this.expectedShape = expectedShape;
        this.actualShape = actualShape;
    }
    
    public String getParamName() {
        return paramName;
    }
    
    public Shape getExpectedShape() {
        return expectedShape;
    }
    
    public Shape getActualShape() {
        return actualShape;
    }
}

