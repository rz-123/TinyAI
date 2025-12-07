package io.leavesfly.tinyai.ml.exception;

/**
 * 训练相关异常
 * 
 * @author TinyAI
 * @version 1.0
 */
public class TrainingException extends ModelException {
    
    private static final long serialVersionUID = 1L;
    
    public TrainingException(String message) {
        super("TRAINING_ERROR", message);
    }
    
    public TrainingException(String message, Throwable cause) {
        super("TRAINING_ERROR", message, cause);
    }
}

