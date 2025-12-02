package io.leavesfly.tinyai.agent.pattern.recap;

/**
 * 任务状态枚举
 * 跟踪子任务的执行状态
 * 
 * @author 山泽
 */
public enum TaskStatus {
    /** 待执行 */
    PENDING("pending"),
    
    /** 执行中 */
    RUNNING("running"),
    
    /** 已完成 */
    COMPLETED("completed"),
    
    /** 已精炼 - 任务被修改或重新规划 */
    REFINED("refined"),
    
    /** 已跳过 - 任务不再需要执行 */
    SKIPPED("skipped"),
    
    /** 失败 */
    FAILED("failed");
    
    private final String value;
    
    TaskStatus(String value) {
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
