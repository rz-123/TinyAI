package io.leavesfly.tinyai.agent.pattern.recap;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * 执行结果类
 * 记录任务执行的结果、观察和关键发现
 * 
 * @author 山泽
 */
public class ExecutionResult {
    /** 关联的任务ID */
    private final String taskId;
    
    /** 执行是否成功 */
    private final boolean success;
    
    /** 执行输出 */
    private final String output;
    
    /** 错误信息 (如果失败) */
    private final String error;
    
    /** 关键发现/洞察 */
    private final List<String> insights;
    
    /** 执行时间戳 */
    private final LocalDateTime timestamp;
    
    /** 使用的工具 */
    private String toolUsed;
    
    /** 执行耗时 (毫秒) */
    private long duration;
    
    /**
     * 成功结果构造
     */
    public static ExecutionResult success(String taskId, String output) {
        return new ExecutionResult(taskId, true, output, null);
    }
    
    /**
     * 失败结果构造
     */
    public static ExecutionResult failure(String taskId, String error) {
        return new ExecutionResult(taskId, false, null, error);
    }
    
    private ExecutionResult(String taskId, boolean success, String output, String error) {
        this.taskId = taskId;
        this.success = success;
        this.output = output;
        this.error = error;
        this.insights = new ArrayList<>();
        this.timestamp = LocalDateTime.now();
    }
    
    /**
     * 添加关键发现
     */
    public void addInsight(String insight) {
        if (insight != null && !insight.trim().isEmpty()) {
            insights.add(insight);
        }
    }
    
    /**
     * 获取结果摘要 (用于上下文注入)
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        if (success) {
            sb.append("执行成功: ").append(truncate(output, 100));
            if (!insights.isEmpty()) {
                sb.append(" | 发现: ").append(String.join("; ", insights));
            }
        } else {
            sb.append("执行失败: ").append(error);
        }
        return sb.toString();
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ========== Getters ==========
    
    public String getTaskId() {
        return taskId;
    }
    
    public boolean isSuccess() {
        return success;
    }
    
    public String getOutput() {
        return output;
    }
    
    public String getError() {
        return error;
    }
    
    public List<String> getInsights() {
        return new ArrayList<>(insights);
    }
    
    public LocalDateTime getTimestamp() {
        return timestamp;
    }
    
    public String getToolUsed() {
        return toolUsed;
    }
    
    public void setToolUsed(String toolUsed) {
        this.toolUsed = toolUsed;
    }
    
    public long getDuration() {
        return duration;
    }
    
    public void setDuration(long duration) {
        this.duration = duration;
    }
    
    @Override
    public String toString() {
        return String.format("Result[%s]: %s", taskId, success ? "SUCCESS" : "FAILED");
    }
}
