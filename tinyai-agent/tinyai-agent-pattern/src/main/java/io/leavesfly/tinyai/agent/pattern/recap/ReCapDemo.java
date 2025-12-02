package io.leavesfly.tinyai.agent.pattern.recap;

import io.leavesfly.tinyai.agent.pattern.ReActAgent;

/**
 * ReCAP vs ReAct 对比演示
 * 
 * 展示两种模式在不同任务场景下的差异:
 * - ReAct: 扁平循环 (Think→Act→Observe)
 * - ReCAP: 递归层级 (Decompose→Execute→Refine)
 * 
 * @author 山泽
 */
public class ReCapDemo {
    
    public static void main(String[] args) {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║          ReCAP vs ReAct 模式对比演示                              ║");
        System.out.println("║   Recursive Context-Aware Reasoning and Planning                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");
        
        // 演示1: 简单任务对比
        demoSimpleTask();
        
        // 演示2: 复杂任务对比
        demoComplexTask();
        
        // 演示3: 长任务链对比
        demoLongHorizonTask();
        
        // 总结
        printSummary();
    }
    
    /**
     * 演示1: 简单任务
     */
    private static void demoSimpleTask() {
        printDemoHeader("演示1: 简单任务 - 数学计算");
        
        String query = "计算 25 * 4 + 10";
        
        // ReAct模式
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.println("│  ReAct 模式执行                          │");
        System.out.println("└─────────────────────────────────────────┘");
        
        ReActAgent reactAgent = new ReActAgent("ReAct-Simple");
        String reactResult = reactAgent.process(query);
        
        System.out.println("查询: " + query);
        System.out.println("结果: " + reactResult);
        System.out.println("步骤数: " + reactAgent.getSteps().size());
        System.out.println("执行轨迹:");
        System.out.println(reactAgent.getStepsSummary());
        
        System.out.println();
        
        // ReCAP模式
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.println("│  ReCAP 模式执行                          │");
        System.out.println("└─────────────────────────────────────────┘");
        
        ReCapAgent recapAgent = new ReCapAgent("ReCAP-Simple");
        String recapResult = recapAgent.process(query);
        
        System.out.println("查询: " + query);
        System.out.println("结果: " + recapResult);
        System.out.println("步骤数: " + recapAgent.getSteps().size());
        System.out.println("最大递归深度: " + recapAgent.getCurrentDepth());
        
        printSeparator();
    }
    
    /**
     * 演示2: 复杂任务 - 需要任务分解
     */
    private static void demoComplexTask() {
        printDemoHeader("演示2: 复杂任务 - 系统设计");
        
        String query = "设计并实现一个用户认证系统";
        
        // ReAct模式
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.println("│  ReAct 模式执行                          │");
        System.out.println("└─────────────────────────────────────────┘");
        
        ReActAgent reactAgent = new ReActAgent("ReAct-Complex", 8);
        String reactResult = reactAgent.process(query);
        
        System.out.println("查询: " + query);
        System.out.println("步骤数: " + reactAgent.getSteps().size());
        System.out.println("执行轨迹 (前5步):");
        printFirstNSteps(reactAgent.getStepsSummary(), 5);
        System.out.println("...\n");
        System.out.println("【ReAct特点】扁平推理，无层级分解，每步独立决策");
        
        System.out.println();
        
        // ReCAP模式
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.println("│  ReCAP 模式执行                          │");
        System.out.println("└─────────────────────────────────────────┘");
        
        ReCapAgent recapAgent = new ReCapAgent("ReCAP-Complex");
        String recapResult = recapAgent.process(query);
        
        System.out.println("查询: " + query);
        System.out.println("步骤数: " + recapAgent.getSteps().size());
        System.out.println("关键发现数: " + recapAgent.getKeyInsights().size());
        System.out.println("\n执行轨迹 (前8步):");
        printFirstNSteps(recapAgent.getStepsSummary(), 8);
        System.out.println("...\n");
        System.out.println("【ReCAP特点】层级分解，递归执行，动态精炼计划，保持高层意图");
        
        printSeparator();
    }
    
    /**
     * 演示3: 长任务链
     */
    private static void demoLongHorizonTask() {
        printDemoHeader("演示3: 长任务链 - 分析并优化系统性能");
        
        String query = "分析并优化系统性能，包括识别瓶颈、设计方案和验证效果";
        
        // ReCAP模式展示完整能力
        System.out.println("┌─────────────────────────────────────────┐");
        System.out.println("│  ReCAP 模式 - 长任务链处理                │");
        System.out.println("└─────────────────────────────────────────┘");
        
        ReCapAgent recapAgent = new ReCapAgent("ReCAP-LongHorizon");
        String result = recapAgent.process(query);
        
        System.out.println("查询: " + query);
        System.out.println("\n" + result);
        
        // 显示执行详情
        System.out.println("\n执行详情:");
        System.out.println("- 总步骤数: " + recapAgent.getSteps().size());
        System.out.println("- 执行任务数: " + recapAgent.getResults().size());
        System.out.println("- 关键发现: " + recapAgent.getKeyInsights().size() + " 条");
        
        if (!recapAgent.getKeyInsights().isEmpty()) {
            System.out.println("\n关键发现列表:");
            for (String insight : recapAgent.getKeyInsights()) {
                System.out.println("  • " + insight);
            }
        }
        
        printSeparator();
    }
    
    /**
     * 打印对比总结
     */
    private static void printSummary() {
        System.out.println("╔══════════════════════════════════════════════════════════════════╗");
        System.out.println("║                        对比总结                                   ║");
        System.out.println("╚══════════════════════════════════════════════════════════════════╝\n");
        
        System.out.println("┌────────────────┬─────────────────────┬─────────────────────────┐");
        System.out.println("│     特性       │       ReAct         │         ReCAP           │");
        System.out.println("├────────────────┼─────────────────────┼─────────────────────────┤");
        System.out.println("│ 推理方式       │ 扁平循环            │ 递归层级                 │");
        System.out.println("│ 任务分解       │ 无，单步推进        │ 完整子任务列表           │");
        System.out.println("│ 计划调整       │ 每步即时决策        │ 执行后精炼剩余计划       │");
        System.out.println("│ 上下文管理     │ 完整上下文累积      │ 滑动窗口+关键上下文注入  │");
        System.out.println("│ 高层意图       │ 可能丢失            │ 始终保持                 │");
        System.out.println("│ 长任务处理     │ 上下文膨胀          │ 保持有界                 │");
        System.out.println("│ 复杂度         │ O(n)                │ O(n) 但支持更深层级      │");
        System.out.println("└────────────────┴─────────────────────┴─────────────────────────┘\n");
        
        System.out.println("ReCAP的三大核心机制:");
        System.out.println("  1. Plan-ahead decomposition: 一次性生成完整子任务列表");
        System.out.println("  2. Structured context re-injection: 共享上下文+结构化注入");
        System.out.println("  3. Memory-efficient scalability: 活动提示有界+外部状态线性增长");
        System.out.println();
        
        System.out.println("适用场景推荐:");
        System.out.println("  • ReAct: 简单任务、工具调用、快速响应场景");
        System.out.println("  • ReCAP: 复杂任务、长任务链、需要保持全局一致性的场景");
    }
    
    // ========== 辅助方法 ==========
    
    private static void printDemoHeader(String title) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println(title);
        System.out.println("=".repeat(70) + "\n");
    }
    
    private static void printSeparator() {
        System.out.println("\n" + "-".repeat(70) + "\n");
    }
    
    private static void printFirstNSteps(String stepsSummary, int n) {
        String[] lines = stepsSummary.split("\n");
        for (int i = 0; i < Math.min(n, lines.length); i++) {
            System.out.println("  " + lines[i]);
        }
    }
}
