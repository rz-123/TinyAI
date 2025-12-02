package io.leavesfly.tinyai.agent.pattern.recap;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * 子任务类
 * ReCAP中的基本任务单元，支持原子执行和递归分解
 * 
 * @author 山泽
 */
public class SubTask {
    /** 任务唯一ID */
    private final String id;
    
    /** 任务描述 */
    private String description;
    
    /** 任务类型 */
    private TaskType type;
    
    /** 任务状态 */
    private TaskStatus status;
    
    /** 优先级 (数字越小优先级越高) */
    private int priority;
    
    /** 任务上下文/参数 */
    private final Map<String, Object> context;
    
    /** 预估复杂度 (1-5) */
    private int complexity;
    
    /** 所需工具 */
    private String requiredTool;
    
    /**
     * 构造函数
     */
    public SubTask(String description) {
        this(description, TaskType.ATOMIC);
    }
    
    public SubTask(String description, TaskType type) {
        this.id = UUID.randomUUID().toString().substring(0, 8);
        this.description = description;
        this.type = type;
        this.status = TaskStatus.PENDING;
        this.priority = 0;
        this.context = new HashMap<>();
        this.complexity = 1;
    }
    
    /**
     * 判断任务是否需要递归分解
     */
    public boolean needsDecomposition() {
        return type == TaskType.COMPOSITE || complexity > 3;
    }
    
    /**
     * 深拷贝
     */
    public SubTask copy() {
        SubTask copy = new SubTask(this.description, this.type);
        copy.status = this.status;
        copy.priority = this.priority;
        copy.context.putAll(this.context);
        copy.complexity = this.complexity;
        copy.requiredTool = this.requiredTool;
        return copy;
    }
    
    // ========== Getters & Setters ==========
    
    public String getId() {
        return id;
    }
    
    public String getDescription() {
        return description;
    }
    
    public void setDescription(String description) {
        this.description = description;
    }
    
    public TaskType getType() {
        return type;
    }
    
    public void setType(TaskType type) {
        this.type = type;
    }
    
    public TaskStatus getStatus() {
        return status;
    }
    
    public void setStatus(TaskStatus status) {
        this.status = status;
    }
    
    public int getPriority() {
        return priority;
    }
    
    public void setPriority(int priority) {
        this.priority = priority;
    }
    
    public Map<String, Object> getContext() {
        return new HashMap<>(context);
    }
    
    public void addContext(String key, Object value) {
        this.context.put(key, value);
    }
    
    public Object getContextValue(String key) {
        return context.get(key);
    }
    
    public int getComplexity() {
        return complexity;
    }
    
    public void setComplexity(int complexity) {
        this.complexity = Math.max(1, Math.min(5, complexity));
    }
    
    public String getRequiredTool() {
        return requiredTool;
    }
    
    public void setRequiredTool(String requiredTool) {
        this.requiredTool = requiredTool;
    }
    
    @Override
    public String toString() {
        return String.format("[%s] %s (%s, %s)", id, description, type, status);
    }
    
    /**
     * 格式化输出用于提示
     */
    public String formatForPrompt() {
        return String.format("- %s [%s]", description, type == TaskType.COMPOSITE ? "复合" : "原子");
    }
}
