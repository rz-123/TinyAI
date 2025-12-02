package io.leavesfly.tinyai.agent.pattern.recap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

/**
 * 计划分解器
 * 将任务一次性分解为完整的有序子任务列表
 * 支持识别复合任务用于后续递归分解
 * 
 * @author 山泽
 */
public class PlanDecomposer {
    
    /** 复合任务关键词 */
    private static final List<String> COMPOSITE_KEYWORDS = Arrays.asList(
        "设计", "开发", "实现", "构建", "创建", "分析", "研究",
        "优化", "重构", "测试", "部署", "集成", "迁移"
    );
    
    /** 原子任务关键词 */
    private static final List<String> ATOMIC_KEYWORDS = Arrays.asList(
        "查询", "搜索", "计算", "读取", "写入", "获取", "设置",
        "验证", "检查", "格式化", "转换", "复制", "删除"
    );
    
    /** 任务模板库 */
    private final Map<String, List<String>> taskTemplates;
    
    public PlanDecomposer() {
        this.taskTemplates = initializeTemplates();
    }
    
    /**
     * 将查询分解为完整的子任务列表
     */
    public SubTaskList decompose(String query) {
        SubTaskList taskList = new SubTaskList();
        
        // 1. 分析查询类型
        QueryType queryType = analyzeQueryType(query);
        
        // 2. 根据类型生成任务列表
        List<SubTask> tasks = generateTasks(query, queryType);
        
        // 3. 设置任务属性
        for (int i = 0; i < tasks.size(); i++) {
            SubTask task = tasks.get(i);
            task.setPriority(i);
            task.setComplexity(estimateComplexity(task.getDescription()));
            task.setType(determineTaskType(task.getDescription()));
            taskList.add(task);
        }
        
        return taskList;
    }
    
    /**
     * 分析查询类型
     */
    private QueryType analyzeQueryType(String query) {
        String lowerQuery = query.toLowerCase();
        
        if (containsAny(lowerQuery, Arrays.asList("分析", "研究", "调研"))) {
            return QueryType.ANALYSIS;
        } else if (containsAny(lowerQuery, Arrays.asList("设计", "架构", "规划"))) {
            return QueryType.DESIGN;
        } else if (containsAny(lowerQuery, Arrays.asList("实现", "开发", "编写", "创建"))) {
            return QueryType.IMPLEMENTATION;
        } else if (containsAny(lowerQuery, Arrays.asList("优化", "改进", "提升"))) {
            return QueryType.OPTIMIZATION;
        } else if (containsAny(lowerQuery, Arrays.asList("测试", "验证", "检查"))) {
            return QueryType.TESTING;
        } else if (containsAny(lowerQuery, Arrays.asList("计算", "求解", "算"))) {
            return QueryType.CALCULATION;
        } else {
            return QueryType.GENERAL;
        }
    }
    
    /**
     * 根据查询类型生成任务列表
     */
    private List<SubTask> generateTasks(String query, QueryType type) {
        List<SubTask> tasks = new ArrayList<>();
        
        switch (type) {
            case ANALYSIS:
                tasks.add(new SubTask("收集相关信息: " + extractTopic(query)));
                tasks.add(new SubTask("分析关键因素", TaskType.COMPOSITE));
                tasks.add(new SubTask("识别模式和趋势"));
                tasks.add(new SubTask("形成分析结论"));
                break;
                
            case DESIGN:
                tasks.add(new SubTask("需求分析: " + extractTopic(query)));
                tasks.add(new SubTask("架构设计", TaskType.COMPOSITE));
                tasks.add(new SubTask("详细设计", TaskType.COMPOSITE));
                tasks.add(new SubTask("设计评审"));
                break;
                
            case IMPLEMENTATION:
                tasks.add(new SubTask("理解需求: " + extractTopic(query)));
                tasks.add(new SubTask("技术方案设计"));
                tasks.add(new SubTask("核心功能实现", TaskType.COMPOSITE));
                tasks.add(new SubTask("测试验证"));
                break;
                
            case OPTIMIZATION:
                tasks.add(new SubTask("性能现状分析: " + extractTopic(query)));
                tasks.add(new SubTask("瓶颈识别", TaskType.COMPOSITE));
                tasks.add(new SubTask("制定优化方案"));
                tasks.add(new SubTask("实施优化"));
                tasks.add(new SubTask("效果验证"));
                break;
                
            case TESTING:
                tasks.add(new SubTask("测试范围分析: " + extractTopic(query)));
                tasks.add(new SubTask("测试用例设计"));
                tasks.add(new SubTask("执行测试"));
                tasks.add(new SubTask("结果分析"));
                break;
                
            case CALCULATION:
                tasks.add(new SubTask("理解计算需求: " + extractTopic(query)));
                tasks.add(new SubTask("执行计算"));
                tasks.add(new SubTask("验证结果"));
                break;
                
            case GENERAL:
            default:
                tasks.add(new SubTask("理解问题: " + extractTopic(query)));
                tasks.add(new SubTask("收集信息"));
                tasks.add(new SubTask("分析处理"));
                tasks.add(new SubTask("生成结果"));
                break;
        }
        
        return tasks;
    }
    
    /**
     * 确定任务类型
     */
    private TaskType determineTaskType(String description) {
        String lowerDesc = description.toLowerCase();
        
        // 检查是否包含复合任务关键词
        if (containsAny(lowerDesc, COMPOSITE_KEYWORDS)) {
            return TaskType.COMPOSITE;
        }
        
        // 检查是否包含原子任务关键词
        if (containsAny(lowerDesc, ATOMIC_KEYWORDS)) {
            return TaskType.ATOMIC;
        }
        
        // 默认为原子任务
        return TaskType.ATOMIC;
    }
    
    /**
     * 估算任务复杂度 (1-5)
     */
    private int estimateComplexity(String description) {
        int complexity = 2; // 默认复杂度
        
        // 根据描述长度增加复杂度
        if (description.length() > 50) complexity++;
        if (description.length() > 100) complexity++;
        
        // 根据关键词调整复杂度
        String lowerDesc = description.toLowerCase();
        if (containsAny(lowerDesc, Arrays.asList("设计", "架构", "优化"))) {
            complexity++;
        }
        if (containsAny(lowerDesc, Arrays.asList("复杂", "困难", "挑战"))) {
            complexity++;
        }
        if (containsAny(lowerDesc, Arrays.asList("简单", "基础", "快速"))) {
            complexity--;
        }
        
        return Math.max(1, Math.min(5, complexity));
    }
    
    /**
     * 从查询中提取主题
     */
    private String extractTopic(String query) {
        // 简单实现：截取前30个字符作为主题
        if (query.length() <= 30) {
            return query;
        }
        return query.substring(0, 30) + "...";
    }
    
    /**
     * 检查是否包含任意关键词
     */
    private boolean containsAny(String text, List<String> keywords) {
        for (String keyword : keywords) {
            if (text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }
    
    /**
     * 初始化任务模板
     */
    private Map<String, List<String>> initializeTemplates() {
        Map<String, List<String>> templates = new HashMap<>();
        
        templates.put("软件开发", Arrays.asList(
            "需求分析", "技术选型", "架构设计", "编码实现", "测试验证", "部署上线"
        ));
        
        templates.put("数据分析", Arrays.asList(
            "数据收集", "数据清洗", "探索性分析", "建模分析", "结果可视化", "报告生成"
        ));
        
        templates.put("问题解决", Arrays.asList(
            "问题定义", "原因分析", "方案设计", "方案实施", "效果评估"
        ));
        
        return templates;
    }
    
    /**
     * 查询类型枚举
     */
    private enum QueryType {
        ANALYSIS,       // 分析类
        DESIGN,         // 设计类
        IMPLEMENTATION, // 实现类
        OPTIMIZATION,   // 优化类
        TESTING,        // 测试类
        CALCULATION,    // 计算类
        GENERAL         // 通用类
    }
}
