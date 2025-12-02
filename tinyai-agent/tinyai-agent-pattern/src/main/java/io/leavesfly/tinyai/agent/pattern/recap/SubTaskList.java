package io.leavesfly.tinyai.agent.pattern.recap;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.stream.Collectors;

/**
 * 子任务列表
 * 支持头部弹出执行、精炼操作和深拷贝
 * ReCAP的核心数据结构之一
 * 
 * @author 山泽
 */
public class SubTaskList {
    /** 有序任务列表 */
    private final LinkedList<SubTask> tasks;
    
    /** 已完成任务历史 */
    private final List<SubTask> completedTasks;
    
    public SubTaskList() {
        this.tasks = new LinkedList<>();
        this.completedTasks = new ArrayList<>();
    }
    
    /**
     * 从任务列表构造
     */
    public SubTaskList(List<SubTask> taskList) {
        this.tasks = new LinkedList<>(taskList);
        this.completedTasks = new ArrayList<>();
    }
    
    /**
     * 添加任务到尾部
     */
    public void add(SubTask task) {
        tasks.addLast(task);
    }
    
    /**
     * 添加任务到头部 (用于插入紧急任务)
     */
    public void addFirst(SubTask task) {
        tasks.addFirst(task);
    }
    
    /**
     * 添加多个任务
     */
    public void addAll(List<SubTask> newTasks) {
        tasks.addAll(newTasks);
    }
    
    /**
     * 弹出头部任务执行
     */
    public SubTask popHead() {
        if (tasks.isEmpty()) {
            return null;
        }
        SubTask head = tasks.removeFirst();
        head.setStatus(TaskStatus.RUNNING);
        return head;
    }
    
    /**
     * 查看头部任务 (不移除)
     */
    public SubTask peekHead() {
        return tasks.isEmpty() ? null : tasks.getFirst();
    }
    
    /**
     * 标记任务完成并记录
     */
    public void markCompleted(SubTask task) {
        task.setStatus(TaskStatus.COMPLETED);
        completedTasks.add(task);
    }
    
    /**
     * 判断是否为空
     */
    public boolean isEmpty() {
        return tasks.isEmpty();
    }
    
    /**
     * 获取剩余任务数
     */
    public int size() {
        return tasks.size();
    }
    
    /**
     * 获取所有剩余任务 (只读)
     */
    public List<SubTask> getRemainingTasks() {
        return new ArrayList<>(tasks);
    }
    
    /**
     * 获取已完成任务 (只读)
     */
    public List<SubTask> getCompletedTasks() {
        return new ArrayList<>(completedTasks);
    }
    
    /**
     * 深拷贝 (保存到父级栈时使用)
     */
    public SubTaskList copy() {
        SubTaskList copy = new SubTaskList();
        for (SubTask task : tasks) {
            copy.add(task.copy());
        }
        for (SubTask task : completedTasks) {
            copy.completedTasks.add(task.copy());
        }
        return copy;
    }
    
    /**
     * 替换剩余任务列表 (精炼后使用)
     */
    public void replaceRemaining(List<SubTask> newTasks) {
        tasks.clear();
        tasks.addAll(newTasks);
    }
    
    /**
     * 清空所有任务
     */
    public void clear() {
        tasks.clear();
        completedTasks.clear();
    }
    
    /**
     * 格式化输出用于提示
     */
    public String format() {
        if (tasks.isEmpty()) {
            return "[计划为空]";
        }
        
        StringBuilder sb = new StringBuilder();
        int index = 1;
        for (SubTask task : tasks) {
            sb.append(index++).append(". ").append(task.formatForPrompt()).append("\n");
        }
        return sb.toString().trim();
    }
    
    /**
     * 格式化输出完整状态 (包含已完成任务)
     */
    public String formatFull() {
        StringBuilder sb = new StringBuilder();
        
        if (!completedTasks.isEmpty()) {
            sb.append("已完成任务:\n");
            for (SubTask task : completedTasks) {
                sb.append("  ✓ ").append(task.getDescription()).append("\n");
            }
        }
        
        if (!tasks.isEmpty()) {
            sb.append("剩余任务:\n");
            int index = 1;
            for (SubTask task : tasks) {
                sb.append("  ").append(index++).append(". ").append(task.formatForPrompt()).append("\n");
            }
        }
        
        return sb.toString().trim();
    }
    
    /**
     * 获取任务描述列表 (用于精炼比较)
     */
    public List<String> getDescriptions() {
        return tasks.stream()
                .map(SubTask::getDescription)
                .collect(Collectors.toList());
    }
    
    @Override
    public String toString() {
        return String.format("SubTaskList[remaining=%d, completed=%d]", tasks.size(), completedTasks.size());
    }
}
