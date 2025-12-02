package io.leavesfly.tinyai.agent.pattern.recap;

import java.util.List;

/**
 * 活动提示构建器
 * 保持提示有界，通过结构化注入重新引入关键规划信息
 * 确保截断不会导致高层意图丢失
 * 
 * @author 山泽
 */
public class ActivePromptBuilder {
    
    /** 最大Token数限制 (保持有界) */
    private static final int MAX_TOKENS = 4000;
    
    /** 估算每字符的Token数 */
    private static final double TOKENS_PER_CHAR = 0.5;
    
    /** 共享的Few-shot示例 (避免重复) */
    private static final String SHARED_FEW_SHOT = 
        "示例任务分解:\n" +
        "问题: \"分析并优化系统性能\"\n" +
        "计划:\n" +
        "1. 收集性能指标 [原子]\n" +
        "2. 分析瓶颈原因 [复合]\n" +
        "3. 制定优化方案 [原子]\n" +
        "4. 验证优化效果 [原子]\n\n" +
        "执行头部任务 -> 观察结果 -> 精炼剩余计划 -> 继续执行";
    
    /**
     * 构建完整的活动提示
     */
    public String build(String highLevelIntent, 
                       SubTaskList currentPlan,
                       String latestThought,
                       List<String> keyInsights,
                       int currentDepth) {
        StringBuilder prompt = new StringBuilder();
        
        // 1. 共享Few-shot (不重复，只在顶层包含)
        if (currentDepth == 0) {
            prompt.append("## 任务分解模式\n");
            prompt.append(SHARED_FEW_SHOT);
            prompt.append("\n---\n\n");
        }
        
        // 2. 高层意图 (始终保持 - 最重要)
        prompt.append("## 高层目标\n");
        prompt.append(highLevelIntent);
        prompt.append("\n\n");
        
        // 3. 当前递归深度
        if (currentDepth > 0) {
            prompt.append("## 当前层级\n");
            prompt.append("递归深度: ").append(currentDepth).append("\n\n");
        }
        
        // 4. 关键发现 (压缩后)
        if (keyInsights != null && !keyInsights.isEmpty()) {
            prompt.append("## 关键发现\n");
            String compressedInsights = compressInsights(keyInsights);
            prompt.append(compressedInsights);
            prompt.append("\n\n");
        }
        
        // 5. 当前计划状态
        prompt.append("## 当前计划\n");
        prompt.append(currentPlan.format());
        prompt.append("\n\n");
        
        // 6. 最新思考
        if (latestThought != null && !latestThought.isEmpty()) {
            prompt.append("## 最新思考\n");
            prompt.append(latestThought);
            prompt.append("\n\n");
        }
        
        // 确保有界
        return truncateToLimit(prompt.toString());
    }
    
    /**
     * 构建结构化注入提示 (从子目标返回时)
     */
    public String buildWithInjection(String highLevelIntent,
                                     ParentContext parentContext,
                                     String childSummary,
                                     List<String> keyInsights) {
        StringBuilder prompt = new StringBuilder();
        
        // 1. 高层意图 (始终保持)
        prompt.append("## 高层目标\n");
        prompt.append(highLevelIntent);
        prompt.append("\n\n");
        
        // 2. 父级上下文恢复 (结构化注入)
        prompt.append("## 上下文恢复\n");
        prompt.append(parentContext.formatForInjection());
        prompt.append("\n\n");
        
        // 3. 子目标执行摘要
        prompt.append("## 子目标执行结果\n");
        prompt.append("子目标: ").append(parentContext.getSubGoalDescription()).append("\n");
        prompt.append("结果摘要: ").append(childSummary);
        prompt.append("\n\n");
        
        // 4. 关键发现
        if (keyInsights != null && !keyInsights.isEmpty()) {
            prompt.append("## 关键发现\n");
            prompt.append(compressInsights(keyInsights));
            prompt.append("\n\n");
        }
        
        // 5. 剩余计划
        prompt.append("## 剩余计划\n");
        prompt.append(parentContext.getRemainingPlan().format());
        prompt.append("\n\n");
        
        prompt.append("请基于子目标执行结果，决定是否需要精炼剩余计划，然后继续执行。\n");
        
        return truncateToLimit(prompt.toString());
    }
    
    /**
     * 压缩关键发现 (保持最重要的信息)
     */
    private String compressInsights(List<String> insights) {
        if (insights.isEmpty()) {
            return "";
        }
        
        StringBuilder sb = new StringBuilder();
        // 只保留最近的5条关键发现
        int start = Math.max(0, insights.size() - 5);
        for (int i = start; i < insights.size(); i++) {
            sb.append("- ").append(insights.get(i)).append("\n");
        }
        return sb.toString().trim();
    }
    
    /**
     * 截断到Token限制
     */
    private String truncateToLimit(String text) {
        int estimatedTokens = estimateTokens(text);
        if (estimatedTokens <= MAX_TOKENS) {
            return text;
        }
        
        // 需要截断时，保留开头(高层意图)和结尾(当前计划)
        int targetChars = (int) (MAX_TOKENS / TOKENS_PER_CHAR);
        
        // 保留前1/3和后2/3
        int headLen = targetChars / 3;
        int tailLen = targetChars * 2 / 3;
        
        if (text.length() > headLen + tailLen) {
            return text.substring(0, headLen) + 
                   "\n...[已截断中间内容以保持提示有界]...\n" + 
                   text.substring(text.length() - tailLen);
        }
        
        return text.substring(0, Math.min(text.length(), targetChars));
    }
    
    /**
     * 估算Token数
     */
    private int estimateTokens(String text) {
        return (int) (text.length() * TOKENS_PER_CHAR);
    }
    
    /**
     * 获取共享的Few-shot示例
     */
    public static String getSharedFewShot() {
        return SHARED_FEW_SHOT;
    }
}
