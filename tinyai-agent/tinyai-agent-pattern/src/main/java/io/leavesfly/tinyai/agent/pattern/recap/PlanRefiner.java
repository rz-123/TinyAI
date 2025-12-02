package io.leavesfly.tinyai.agent.pattern.recap;

import java.util.ArrayList;
import java.util.List;

/**
 * 计划精炼器
 * 根据执行结果动态调整剩余计划
 * 支持任务合并、拆分、重排序和跳过
 * 
 * @author 山泽
 */
public class PlanRefiner {
    
    /** 精炼操作类型 */
    public enum RefineAction {
        KEEP,       // 保持不变
        MODIFY,     // 修改任务
        SPLIT,      // 拆分任务
        MERGE,      // 合并任务
        SKIP,       // 跳过任务
        ADD,        // 添加新任务
        REORDER     // 重新排序
    }
    
    /**
     * 精炼剩余计划
     */
    public SubTaskList refine(SubTaskList currentPlan, 
                              ExecutionResult result,
                              String highLevelIntent) {
        if (currentPlan.isEmpty()) {
            return currentPlan;
        }
        
        // 分析执行结果
        RefineDecision decision = analyzeAndDecide(currentPlan, result, highLevelIntent);
        
        // 应用精炼决策
        return applyRefinement(currentPlan, decision);
    }
    
    /**
     * 基于子目标返回结果精炼父级计划
     */
    public SubTaskList refineAfterSubGoal(SubTaskList parentPlan,
                                          String childSummary,
                                          String highLevelIntent) {
        if (parentPlan.isEmpty()) {
            return parentPlan;
        }
        
        // 分析子目标完成情况对父级计划的影响
        List<SubTask> refinedTasks = new ArrayList<>();
        
        for (SubTask task : parentPlan.getRemainingTasks()) {
            // 检查任务是否因子目标完成而可以跳过
            if (canSkipAfterChildCompletion(task, childSummary)) {
                task.setStatus(TaskStatus.SKIPPED);
                continue;
            }
            
            // 检查任务是否需要基于子目标结果修改
            SubTask refinedTask = modifyBasedOnChildResult(task, childSummary);
            refinedTasks.add(refinedTask);
        }
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(refinedTasks);
        return refined;
    }
    
    /**
     * 分析并决定精炼操作
     */
    private RefineDecision analyzeAndDecide(SubTaskList plan, 
                                            ExecutionResult result,
                                            String intent) {
        RefineDecision decision = new RefineDecision();
        
        // 1. 检查执行是否失败
        if (!result.isSuccess()) {
            // 失败时可能需要添加重试任务或调整后续任务
            decision.setAction(RefineAction.MODIFY);
            decision.setReason("执行失败，需要调整后续计划");
            return decision;
        }
        
        // 2. 检查是否发现新的关键信息
        if (!result.getInsights().isEmpty()) {
            // 有新发现时可能需要调整计划
            decision.setAction(RefineAction.MODIFY);
            decision.setReason("发现新信息: " + result.getInsights().get(0));
            return decision;
        }
        
        // 3. 检查输出是否表明某些后续任务可以跳过
        if (result.getOutput() != null && 
            (result.getOutput().contains("已完成") || result.getOutput().contains("无需"))) {
            decision.setAction(RefineAction.SKIP);
            decision.setReason("前序任务输出表明可跳过部分后续任务");
            return decision;
        }
        
        // 默认保持不变
        decision.setAction(RefineAction.KEEP);
        decision.setReason("执行正常，计划保持不变");
        return decision;
    }
    
    /**
     * 应用精炼决策
     */
    private SubTaskList applyRefinement(SubTaskList plan, RefineDecision decision) {
        switch (decision.getAction()) {
            case KEEP:
                return plan;
                
            case MODIFY:
                return modifyPlan(plan, decision);
                
            case SKIP:
                return skipTasks(plan, decision);
                
            case SPLIT:
                return splitTask(plan, decision);
                
            case MERGE:
                return mergeTasks(plan, decision);
                
            case ADD:
                return addTask(plan, decision);
                
            case REORDER:
                return reorderTasks(plan, decision);
                
            default:
                return plan;
        }
    }
    
    /**
     * 修改计划中的任务
     */
    private SubTaskList modifyPlan(SubTaskList plan, RefineDecision decision) {
        List<SubTask> tasks = plan.getRemainingTasks();
        
        // 根据决策修改任务描述或属性
        if (!tasks.isEmpty()) {
            SubTask firstTask = tasks.get(0);
            String newDesc = firstTask.getDescription() + " (基于: " + 
                            truncate(decision.getReason(), 30) + ")";
            firstTask.setDescription(newDesc);
            firstTask.setStatus(TaskStatus.REFINED);
        }
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(tasks);
        return refined;
    }
    
    /**
     * 跳过某些任务
     */
    private SubTaskList skipTasks(SubTaskList plan, RefineDecision decision) {
        List<SubTask> remaining = new ArrayList<>();
        List<SubTask> tasks = plan.getRemainingTasks();
        
        for (int i = 0; i < tasks.size(); i++) {
            SubTask task = tasks.get(i);
            // 跳过第一个可跳过的任务
            if (i == 0 && canSkip(task)) {
                task.setStatus(TaskStatus.SKIPPED);
                continue;
            }
            remaining.add(task);
        }
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(remaining);
        return refined;
    }
    
    /**
     * 拆分任务
     */
    private SubTaskList splitTask(SubTaskList plan, RefineDecision decision) {
        List<SubTask> tasks = plan.getRemainingTasks();
        List<SubTask> result = new ArrayList<>();
        
        for (SubTask task : tasks) {
            if (task.getComplexity() > 3 && task.getType() == TaskType.COMPOSITE) {
                // 拆分复杂任务
                result.add(new SubTask(task.getDescription() + " - 第一阶段"));
                result.add(new SubTask(task.getDescription() + " - 第二阶段"));
            } else {
                result.add(task);
            }
        }
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(result);
        return refined;
    }
    
    /**
     * 合并任务
     */
    private SubTaskList mergeTasks(SubTaskList plan, RefineDecision decision) {
        List<SubTask> tasks = plan.getRemainingTasks();
        if (tasks.size() < 2) {
            return plan;
        }
        
        List<SubTask> result = new ArrayList<>();
        
        // 尝试合并相邻的简单任务
        for (int i = 0; i < tasks.size(); i++) {
            if (i + 1 < tasks.size() && 
                tasks.get(i).getComplexity() <= 2 && 
                tasks.get(i + 1).getComplexity() <= 2) {
                // 合并两个简单任务
                String mergedDesc = tasks.get(i).getDescription() + " 和 " + 
                                   tasks.get(i + 1).getDescription();
                result.add(new SubTask(mergedDesc));
                i++; // 跳过下一个任务
            } else {
                result.add(tasks.get(i));
            }
        }
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(result);
        return refined;
    }
    
    /**
     * 添加新任务
     */
    private SubTaskList addTask(SubTaskList plan, RefineDecision decision) {
        List<SubTask> tasks = plan.getRemainingTasks();
        
        // 在开头添加新任务
        SubTask newTask = new SubTask(decision.getNewTaskDescription());
        tasks.add(0, newTask);
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(tasks);
        return refined;
    }
    
    /**
     * 重新排序任务
     */
    private SubTaskList reorderTasks(SubTaskList plan, RefineDecision decision) {
        List<SubTask> tasks = plan.getRemainingTasks();
        
        // 按优先级排序
        tasks.sort((a, b) -> a.getPriority() - b.getPriority());
        
        SubTaskList refined = new SubTaskList();
        refined.addAll(tasks);
        return refined;
    }
    
    /**
     * 判断任务是否可以跳过
     */
    private boolean canSkip(SubTask task) {
        return task.getComplexity() <= 1 || 
               task.getDescription().contains("可选") ||
               task.getDescription().contains("如果需要");
    }
    
    /**
     * 判断任务是否因子目标完成而可以跳过
     */
    private boolean canSkipAfterChildCompletion(SubTask task, String childSummary) {
        // 如果子目标已经完成了相似的工作
        String taskDesc = task.getDescription().toLowerCase();
        String summary = childSummary.toLowerCase();
        
        return summary.contains("已完成") && summary.contains(extractKeyword(taskDesc));
    }
    
    /**
     * 基于子目标结果修改任务
     */
    private SubTask modifyBasedOnChildResult(SubTask task, String childSummary) {
        SubTask modified = task.copy();
        
        // 如果子目标发现了问题，可能需要调整任务描述
        if (childSummary.contains("问题") || childSummary.contains("错误")) {
            modified.setDescription(task.getDescription() + " (注意子目标发现的问题)");
            modified.setComplexity(Math.min(5, task.getComplexity() + 1));
        }
        
        return modified;
    }
    
    /**
     * 提取关键词
     */
    private String extractKeyword(String text) {
        // 简单实现：取前10个字符
        return text.length() > 10 ? text.substring(0, 10) : text;
    }
    
    /**
     * 截断文本
     */
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    /**
     * 精炼决策内部类
     */
    private static class RefineDecision {
        private RefineAction action = RefineAction.KEEP;
        private String reason = "";
        private String newTaskDescription = "";
        private int targetIndex = -1;
        
        public RefineAction getAction() { return action; }
        public void setAction(RefineAction action) { this.action = action; }
        
        public String getReason() { return reason; }
        public void setReason(String reason) { this.reason = reason; }
        
        public String getNewTaskDescription() { return newTaskDescription; }
        public void setNewTaskDescription(String desc) { this.newTaskDescription = desc; }
        
        public int getTargetIndex() { return targetIndex; }
        public void setTargetIndex(int index) { this.targetIndex = index; }
    }
}
