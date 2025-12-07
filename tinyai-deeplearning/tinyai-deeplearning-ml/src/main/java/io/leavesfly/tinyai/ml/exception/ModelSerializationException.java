package io.leavesfly.tinyai.ml.exception;

/**
 * 模型序列化异常
 * 
 * @author TinyAI
 * @version 1.0
 */
public class ModelSerializationException extends ModelException {
    
    private static final long serialVersionUID = 1L;
    
    public ModelSerializationException(String message) {
        super("SERIALIZATION_ERROR", message);
    }
    
    public ModelSerializationException(String message, Throwable cause) {
        super("SERIALIZATION_ERROR", message, cause);
    }
}

