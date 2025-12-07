package io.leavesfly.tinyai.minimind.cli;

import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.Command;
import io.leavesfly.tinyai.minimind.cli.MiniMindCLI.ArgParser;

/**
 * SFT微调命令
 * 
 * @author leavesfly
 * @since 2024
 */
public class TrainSFTCommand implements Command {
    
    @Override
    public void execute(String[] args) throws Exception {
        ArgParser parser = new ArgParser(args);
        
        if (parser.has("help") || parser.has("h")) {
            printHelp();
            return;
        }
        
        String modelPath = parser.get("model", "model/pretrain.pt");
        String trainFile = parser.get("train-file", "data/sft_train.jsonl");
        String outputDir = parser.get("output-dir", "output/sft");
        int epochs = parser.getInt("epochs", 3);
        float learningRate = parser.getFloat("learning-rate", 1e-4f);
        
        System.out.println("=".repeat(60));
        System.out.println("MiniMind SFT微调");
        System.out.println("=".repeat(60));
        System.out.println("基础模型: " + modelPath);
        System.out.println("训练文件: " + trainFile);
        System.out.println("输出目录: " + outputDir);
        System.out.println("训练轮数: " + epochs);
        System.out.println("学习率: " + learningRate);
        System.out.println("=".repeat(60));
        
        System.out.println("\n提示: SFT微调功能开发中,请参考 Example03-SFT微调示例.java");
    }
    
    @Override
    public String getDescription() {
        return "监督微调(SFT)";
    }
    
    @Override
    public void printHelp() {
        System.out.println("用法: minimind train-sft [options]");
        System.out.println();
        System.out.println("监督微调(SFT)");
        System.out.println();
        System.out.println("选项:");
        System.out.println("  --model FILE           基础模型路径");
        System.out.println("  --train-file FILE      训练数据文件(JSONL格式)");
        System.out.println("  --output-dir DIR       输出目录");
        System.out.println("  --epochs INT           训练轮数 (default: 3)");
        System.out.println("  --learning-rate FLOAT  学习率 (default: 1e-4)");
    }
}
