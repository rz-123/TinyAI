package io.leavesfly.tinyai.agent.pattern.recap;

/**
 * 任务类型枚举
 * 区分原子任务和可分解的复合任务
 * 
 * @author 山泽
 */
public enum TaskType {
    /** 原子任务 - 不可再分解，直接执行 */
    ATOMIC("atomic"),
    
    /** 复合任务 - 可以递归分解为子任务 */
    COMPOSITE("composite");
    
    private final String value;
    
    TaskType(String value) {
        this.value = value;
    }
    
    public String getValue() {
        return value;
    }
    
    @Override
    public String toString() {
        return value;
    }
}
