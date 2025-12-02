package io.leavesfly.tinyai.agent.pattern.recap;

import io.leavesfly.tinyai.agent.pattern.Action;
import io.leavesfly.tinyai.agent.pattern.AgentState;
import io.leavesfly.tinyai.agent.pattern.BaseAgent;

import java.util.*;

/**
 * ReCAP模式Agent: Recursive Context-Aware Reasoning and Planning
 * 
 * 核心机制:
 * 1. Plan-ahead decomposition: 一次性生成完整子任务列表，执行头部，精炼剩余
 * 2. Structured context re-injection: 共享LLM上下文，子目标返回时恢复父级计划
 * 3. Memory-efficient scalability: 活动提示有界 + 外部状态线性增长
 * 
 * 相比ReAct的优势:
 * - 支持长任务链而不丢失高层意图
 * - 递归分解复杂任务
 * - 动态精炼计划
 * - 上下文保持有界
 * 
 * @author 山泽
 */
public class ReCapAgent extends BaseAgent {
    
    // ========== 共享LLM上下文 (有界) ==========
    /** 高层意图 - 始终保持 */
    private String highLevelIntent;
    
    /** 当前层级的计划 */
    private SubTaskList currentPlan;
    
    /** 最新思考 */
    private String latestThought;
    
    // ========== 外部状态 (随递归深度线性增长) ==========
    /** 父级计划栈 - 用于结构化注入恢复 */
    private Deque<ParentContext> parentStack;
    
    /** 累积的关键发现 */
    private List<String> keyInsights;
    
    /** 执行结果记录 */
    private List<ExecutionResult> results;
    
    /** 当前递归深度 */
    private int currentDepth;
    
    /** 最大递归深度 */
    private static final int MAX_RECURSION_DEPTH = 5;
    
    // ========== 组件 ==========
    /** 计划分解器 */
    private final PlanDecomposer decomposer;
    
    /** 计划精炼器 */
    private final PlanRefiner refiner;
    
    /** 活动提示构建器 */
    private final ActivePromptBuilder promptBuilder;
    
    /** 随机数生成器 */
    private final Random random;
    
    /**
     * 构造函数
     */
    public ReCapAgent() {
        this("ReCAP Agent", 20);
    }
    
    public ReCapAgent(String name) {
        this(name, 20);
    }
    
    public ReCapAgent(String name, int maxSteps) {
        super(name, maxSteps);
        this.decomposer = new PlanDecomposer();
        this.refiner = new PlanRefiner();
        this.promptBuilder = new ActivePromptBuilder();
        this.random = new Random();
        this.parentStack = new ArrayDeque<>();
        this.keyInsights = new ArrayList<>();
        this.results = new ArrayList<>();
        registerDefaultTools();
    }
    
    /**
     * 注册默认工具
     */
    private void registerDefaultTools() {
        addTool("research", this::researchTool, "研究工具 - 收集信息");
        addTool("analyze", this::analyzeTool, "分析工具 - 深度分析");
        addTool("calculate", this::calculateTool, "计算工具 - 数学计算");
        addTool("validate", this::validateTool, "验证工具 - 结果验证");
        addTool("synthesize", this::synthesizeTool, "综合工具 - 信息整合");
    }
    
    @Override
    public String process(String query) {
        // 初始化
        initializeExecution(query);
        
        setState(AgentState.PLANNING);
        addStep("init", "开始处理: " + query);
        
        // 1. 一次性生成完整子任务列表
        this.currentPlan = decomposer.decompose(query);
        addStep("plan", "完整计划:\n" + currentPlan.format());
        
        // 2. 执行循环
        String result = executeLoop();
        
        setState(AgentState.DONE);
        return result;
    }
    
    /**
     * 初始化执行状态
     */
    private void initializeExecution(String query) {
        clearSteps();
        this.highLevelIntent = query;
        this.parentStack = new ArrayDeque<>();
        this.keyInsights = new ArrayList<>();
        this.results = new ArrayList<>();
        this.currentDepth = 0;
        this.latestThought = "";
        this.currentPlan = new SubTaskList();
    }
    
    /**
     * 核心执行循环
     */
    private String executeLoop() {
        int stepCount = 0;
        
        while (stepCount < maxSteps) {
            stepCount++;
            
            // 计划为空时检查是否需要返回父级
            if (currentPlan.isEmpty()) {
                if (parentStack.isEmpty()) {
                    // 所有层级完成
                    return synthesizeFinalAnswer();
                } else {
                    // 结构化注入: 恢复父级上下文
                    restoreParentContext();
                    continue;
                }
            }
            
            // 弹出头部任务
            SubTask headTask = currentPlan.popHead();
            addStep("execute", String.format("[深度:%d] 执行任务: %s", currentDepth, headTask.getDescription()));
            
            setState(AgentState.THINKING);
            
            // 判断是否需要递归分解
            if (needsRecursiveDecomposition(headTask) && currentDepth < MAX_RECURSION_DEPTH) {
                // 保存当前上下文到栈
                pushCurrentContext(headTask);
                
                // 对子目标进行完整分解
                setState(AgentState.PLANNING);
                currentPlan = decomposer.decompose(headTask.getDescription());
                currentDepth++;
                
                addStep("recurse", String.format("递归分解 [深度:%d]:\n%s", currentDepth, currentPlan.format()));
            } else {
                // 原子执行
                setState(AgentState.ACTING);
                ExecutionResult result = executeAtomicTask(headTask);
                results.add(result);
                
                // 标记完成
                currentPlan.markCompleted(headTask);
                
                setState(AgentState.OBSERVING);
                addStep("observation", result.getSummary());
                
                // 更新最新思考
                latestThought = generateThought(headTask, result);
                
                // 提取关键发现
                extractKeyInsight(result);
                
                // 精炼剩余计划
                setState(AgentState.REFLECTING);
                currentPlan = refiner.refine(currentPlan, result, highLevelIntent);
            }
        }
        
        return "达到最大步骤数限制，任务部分完成。\n当前进度:\n" + currentPlan.formatFull();
    }
    
    /**
     * 判断是否需要递归分解
     */
    private boolean needsRecursiveDecomposition(SubTask task) {
        return task.needsDecomposition();
    }
    
    /**
     * 执行原子任务
     */
    private ExecutionResult executeAtomicTask(SubTask task) {
        long startTime = System.currentTimeMillis();
        
        String desc = task.getDescription();
        String toolName = selectTool(desc);
        
        Map<String, Object> args = new HashMap<>();
        args.put("query", desc);
        args.put("context", latestThought);
        
        Action action = new Action(toolName, args);
        Object result = callTool(action);
        
        long duration = System.currentTimeMillis() - startTime;
        
        ExecutionResult execResult;
        if (action.hasError()) {
            execResult = ExecutionResult.failure(task.getId(), action.getError());
        } else {
            execResult = ExecutionResult.success(task.getId(), String.valueOf(result));
        }
        
        execResult.setToolUsed(toolName);
        execResult.setDuration(duration);
        
        return execResult;
    }
    
    /**
     * 选择合适的工具
     */
    private String selectTool(String description) {
        String desc = description.toLowerCase();
        
        if (containsAny(desc, "收集", "研究", "调研", "信息")) {
            return "research";
        } else if (containsAny(desc, "分析", "识别", "评估")) {
            return "analyze";
        } else if (containsAny(desc, "计算", "求解", "算")) {
            return "calculate";
        } else if (containsAny(desc, "验证", "检查", "测试")) {
            return "validate";
        } else if (containsAny(desc, "综合", "整合", "总结", "生成")) {
            return "synthesize";
        }
        
        return "analyze"; // 默认工具
    }
    
    /**
     * 生成思考
     */
    private String generateThought(SubTask task, ExecutionResult result) {
        if (result.isSuccess()) {
            return String.format("任务'%s'已完成。%s", 
                    truncate(task.getDescription(), 30),
                    result.getInsights().isEmpty() ? "" : "关键发现: " + result.getInsights().get(0));
        } else {
            return String.format("任务'%s'执行失败: %s。需要调整策略。",
                    truncate(task.getDescription(), 30),
                    result.getError());
        }
    }
    
    /**
     * 提取关键发现
     */
    private void extractKeyInsight(ExecutionResult result) {
        if (result.isSuccess() && result.getOutput() != null) {
            // 提取有价值的信息作为关键发现
            String output = result.getOutput();
            if (output.contains("关键") || output.contains("重要") || output.contains("发现")) {
                keyInsights.add(truncate(output, 50));
            }
            
            // 添加结果中的洞察
            keyInsights.addAll(result.getInsights());
        }
    }
    
    /**
     * 保存当前上下文到栈 (进入子目标前)
     */
    private void pushCurrentContext(SubTask subGoalTask) {
        ParentContext ctx = new ParentContext(
                currentPlan,
                latestThought,
                currentDepth,
                subGoalTask.getDescription()
        );
        parentStack.push(ctx);
        addStep("push", "保存父级上下文 [深度:" + currentDepth + "]");
    }
    
    /**
     * 结构化注入: 从子目标返回时恢复父级计划
     */
    private void restoreParentContext() {
        ParentContext parent = parentStack.pop();
        currentDepth = parent.getDepth();
        
        // 生成子目标执行摘要
        String childSummary = summarizeChildExecution();
        
        addStep("restore", "恢复父级上下文 [深度:" + currentDepth + "]\n子目标摘要: " + childSummary);
        
        // 恢复父级计划
        this.currentPlan = parent.getRemainingPlan();
        
        // 合并思考
        this.latestThought = mergeThoughts(parent.getLatestThought(), childSummary);
        
        // 基于子目标结果精炼父级剩余计划
        this.currentPlan = refiner.refineAfterSubGoal(currentPlan, childSummary, highLevelIntent);
    }
    
    /**
     * 生成子目标执行摘要
     */
    private String summarizeChildExecution() {
        if (results.isEmpty()) {
            return "子目标未产生结果";
        }
        
        // 获取最近的结果
        int count = Math.min(3, results.size());
        StringBuilder sb = new StringBuilder();
        for (int i = results.size() - count; i < results.size(); i++) {
            ExecutionResult r = results.get(i);
            sb.append(r.isSuccess() ? "✓ " : "✗ ");
            sb.append(truncate(r.getOutput() != null ? r.getOutput() : r.getError(), 40));
            sb.append("; ");
        }
        return sb.toString();
    }
    
    /**
     * 合并思考
     */
    private String mergeThoughts(String parentThought, String childSummary) {
        return parentThought + " | 子目标结果: " + truncate(childSummary, 50);
    }
    
    /**
     * 综合最终答案
     */
    private String synthesizeFinalAnswer() {
        StringBuilder answer = new StringBuilder();
        answer.append("=== ReCAP 执行完成 ===\n\n");
        
        answer.append("【高层目标】\n").append(highLevelIntent).append("\n\n");
        
        answer.append("【执行统计】\n");
        answer.append("- 最大递归深度: ").append(getMaxDepthReached()).append("\n");
        answer.append("- 执行任务数: ").append(results.size()).append("\n");
        answer.append("- 成功任务: ").append(countSuccessful()).append("\n\n");
        
        if (!keyInsights.isEmpty()) {
            answer.append("【关键发现】\n");
            for (String insight : keyInsights) {
                answer.append("- ").append(insight).append("\n");
            }
            answer.append("\n");
        }
        
        answer.append("【最终结论】\n");
        answer.append(generateConclusion());
        
        return answer.toString();
    }
    
    /**
     * 生成最终结论
     */
    private String generateConclusion() {
        if (results.isEmpty()) {
            return "任务执行未产生结果";
        }
        
        // 收集所有成功的输出
        StringBuilder conclusion = new StringBuilder();
        for (ExecutionResult r : results) {
            if (r.isSuccess() && r.getOutput() != null) {
                conclusion.append(r.getOutput()).append(" ");
            }
        }
        
        if (conclusion.length() == 0) {
            return "任务执行完成，但未产生有效输出";
        }
        
        return conclusion.toString().trim();
    }
    
    /**
     * 获取达到的最大深度
     */
    private int getMaxDepthReached() {
        int maxDepth = 0;
        for (int i = 0; i < getSteps().size(); i++) {
            String content = getSteps().get(i).getContent();
            if (content.contains("深度:")) {
                try {
                    int idx = content.indexOf("深度:") + 3;
                    int endIdx = content.indexOf("]", idx);
                    if (endIdx == -1) endIdx = content.indexOf(" ", idx);
                    if (endIdx == -1) endIdx = idx + 1;
                    int depth = Integer.parseInt(content.substring(idx, endIdx).trim());
                    maxDepth = Math.max(maxDepth, depth);
                } catch (Exception e) {
                    // 忽略解析错误
                }
            }
        }
        return maxDepth;
    }
    
    /**
     * 计算成功任务数
     */
    private long countSuccessful() {
        return results.stream().filter(ExecutionResult::isSuccess).count();
    }
    
    // ========== 工具实现 ==========
    
    private Object researchTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        // 模拟研究结果
        String[] researchResults = {
            "研究发现: 该领域有多个关键因素需要考虑",
            "信息收集完成: 识别了主要的技术方案",
            "调研结果: 发现了3个可行的解决方向"
        };
        String result = researchResults[random.nextInt(researchResults.length)];
        return result + " (关于: " + truncate(query, 30) + ")";
    }
    
    private Object analyzeTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        String[] analyzeResults = {
            "分析完成: 关键瓶颈已识别",
            "深度分析: 发现核心问题在于架构设计",
            "评估结果: 当前方案可行性较高"
        };
        return analyzeResults[random.nextInt(analyzeResults.length)] + " (分析: " + truncate(query, 30) + ")";
    }
    
    private Object calculateTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        return "计算完成: 结果已验证 (计算: " + truncate(query, 30) + ")";
    }
    
    private Object validateTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        int score = 7 + random.nextInt(4); // 7-10分
        return "验证通过: 质量评分 " + score + "/10 (验证: " + truncate(query, 30) + ")";
    }
    
    private Object synthesizeTool(Map<String, Object> args) {
        String query = (String) args.get("query");
        return "综合完成: 已整合所有信息形成完整方案 (综合: " + truncate(query, 30) + ")";
    }
    
    // ========== 辅助方法 ==========
    
    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }
    
    private String truncate(String text, int maxLen) {
        if (text == null) return "";
        return text.length() > maxLen ? text.substring(0, maxLen) + "..." : text;
    }
    
    // ========== Getters ==========
    
    public String getHighLevelIntent() {
        return highLevelIntent;
    }
    
    public int getCurrentDepth() {
        return currentDepth;
    }
    
    public List<String> getKeyInsights() {
        return new ArrayList<>(keyInsights);
    }
    
    public List<ExecutionResult> getResults() {
        return new ArrayList<>(results);
    }
    
    public SubTaskList getCurrentPlan() {
        return currentPlan != null ? currentPlan.copy() : new SubTaskList();
    }
    
    @Override
    public void reset() {
        super.reset();
        this.highLevelIntent = null;
        this.currentPlan = null;
        this.latestThought = "";
        this.parentStack.clear();
        this.keyInsights.clear();
        this.results.clear();
        this.currentDepth = 0;
    }
}
