package io.leavesfly.tinyai.agent.pattern.recap;

/**
 * 父级上下文
 * 用于结构化注入恢复 - 从子目标返回时恢复父级计划
 * 是递归栈的元素
 * 
 * @author 山泽
 */
public class ParentContext {
    /** 剩余子任务列表 */
    private final SubTaskList remainingPlan;
    
    /** 最新思考 */
    private final String latestThought;
    
    /** 递归深度 */
    private final int depth;
    
    /** 进入子目标时的任务描述 */
    private final String subGoalDescription;
    
    /** 保存时间戳 */
    private final long timestamp;
    
    /**
     * 构造函数
     */
    public ParentContext(SubTaskList remainingPlan, String latestThought, 
                         int depth, String subGoalDescription) {
        this.remainingPlan = remainingPlan.copy(); // 深拷贝保存
        this.latestThought = latestThought;
        this.depth = depth;
        this.subGoalDescription = subGoalDescription;
        this.timestamp = System.currentTimeMillis();
    }
    
    /**
     * 获取剩余计划 (深拷贝)
     */
    public SubTaskList getRemainingPlan() {
        return remainingPlan.copy();
    }
    
    /**
     * 获取最新思考
     */
    public String getLatestThought() {
        return latestThought;
    }
    
    /**
     * 获取递归深度
     */
    public int getDepth() {
        return depth;
    }
    
    /**
     * 获取子目标描述
     */
    public String getSubGoalDescription() {
        return subGoalDescription;
    }
    
    /**
     * 获取保存时间戳
     */
    public long getTimestamp() {
        return timestamp;
    }
    
    /**
     * 格式化输出用于结构化注入
     */
    public String formatForInjection() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== 父级上下文恢复 (深度:").append(depth).append(") ===\n");
        sb.append("子目标: ").append(subGoalDescription).append("\n");
        sb.append("之前的思考: ").append(latestThought).append("\n");
        sb.append("剩余计划:\n").append(remainingPlan.format());
        return sb.toString();
    }
    
    @Override
    public String toString() {
        return String.format("ParentContext[depth=%d, remaining=%d, subGoal=%s]", 
                depth, remainingPlan.size(), truncate(subGoalDescription, 30));
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
}
