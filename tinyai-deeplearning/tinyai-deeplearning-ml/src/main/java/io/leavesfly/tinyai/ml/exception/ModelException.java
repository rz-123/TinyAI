package io.leavesfly.tinyai.ml.exception;

/**
 * 模型相关异常基类
 * 
 * @author TinyAI
 * @version 1.0
 */
public class ModelException extends RuntimeException {
    
    private static final long serialVersionUID = 1L;
    
    private final String errorCode;
    
    public ModelException(String message) {
        super(message);
        this.errorCode = "MODEL_ERROR";
    }
    
    public ModelException(String errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }
    
    public ModelException(String message, Throwable cause) {
        super(message, cause);
        this.errorCode = "MODEL_ERROR";
    }
    
    public ModelException(String errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }
    
    public String getErrorCode() {
        return errorCode;
    }
}

