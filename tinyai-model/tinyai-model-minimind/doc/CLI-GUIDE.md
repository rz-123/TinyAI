# MiniMind CLI 使用指南

## 概述

MiniMind CLI 是一个命令行工具,提供模型训练、文本生成、对话等功能。

## 安装

### 1. 构建项目

```bash
cd tinyai-model/tinyai-model-minimind
mvn clean package
```

### 2. 设置环境

Linux/Mac:
```bash
chmod +x bin/minimind.sh
export PATH=$PATH:$(pwd)/bin
```

Windows:
```cmd
set PATH=%PATH%;%CD%\bin
```

## 命令列表

### 1. train-pretrain - 预训练

```bash
minimind train-pretrain \
  --train-file data/train.txt \
  --vocab-size 6400 \
  --epochs 10 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --output-dir output/pretrain
```

### 2. train-sft - 监督微调

```bash
minimind train-sft \
  --model model/pretrain.pt \
  --train-file data/sft_train.jsonl \
  --epochs 3 \
  --output-dir output/sft
```

### 3. train-lora - LoRA微调

```bash
minimind train-lora \
  --model model/base.pt \
  --train-file data/lora_train.jsonl \
  --lora-rank 8 \
  --lora-alpha 16.0
```

### 4. generate - 文本生成

```bash
minimind generate \
  --model model/trained.pt \
  --prompt "今天天气" \
  --max-length 100 \
  --temperature 0.8 \
  --top-k 50
```

### 5. chat - 交互式对话

```bash
minimind chat --model model/chat.pt
```

### 6. evaluate - 模型评估

```bash
minimind evaluate \
  --model model/trained.pt \
  --test-file data/test.txt
```

### 7. help - 帮助信息

```bash
minimind help
minimind <command> --help
```

### 8. version - 版本信息

```bash
minimind version
```

## 快速开始示例

### 示例1: 预训练

```bash
# 1. 准备训练数据
echo "训练数据示例" > data/train.txt

# 2. 开始预训练
minimind train-pretrain \
  --train-file data/train.txt \
  --epochs 5 \
  --output-dir output/my_model
```

### 示例2: 文本生成

```bash
minimind generate \
  --model output/my_model/final.pt \
  --prompt "人工智能的未来" \
  --max-length 200
```

### 示例3: 交互式对话

```bash
minimind chat --model output/my_model/final.pt

# 进入对话界面后:
# 用户: 你好
# 助手: [模型回复]
# 
# 输入 exit 退出
```

## 常见参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| --train-file | 训练数据文件 | data/train.txt |
| --model | 模型路径 | - |
| --output-dir | 输出目录 | output/ |
| --epochs | 训练轮数 | 10 |
| --batch-size | 批次大小 | 32 |
| --learning-rate | 学习率 | 0.001 |
| --vocab-size | 词表大小 | 6400 |
| --max-length | 最大生成长度 | 100 |
| --temperature | 温度参数 | 1.0 |
| --top-k | Top-K采样 | 50 |
| --lora-rank | LoRA秩 | 8 |
| --lora-alpha | LoRA Alpha | 16.0 |

## 注意事项

1. **内存要求**: 建议至少2GB JVM堆内存
2. **数据格式**: 
   - 预训练: 纯文本文件
   - SFT/LoRA: JSONL格式
3. **模型保存**: 模型会自动保存到output目录
4. **中断训练**: Ctrl+C可安全中断训练

## 故障排除

### 问题1: 内存不足

```bash
# 增加JVM堆内存
export JVM_OPTS="-Xmx4g -Xms1g"
minimind <command> ...
```

### 问题2: 找不到模型文件

确保模型路径正确:
```bash
ls -l model/trained.pt
```

### 问题3: 命令不可用

检查PATH设置:
```bash
which minimind
echo $PATH
```

## 参考资料

- [Example05-预训练流程.java](../examples/Example05-预训练流程.java)
- [Example03-SFT微调示例.java](../examples/Example03-SFT微调示例.java)
- [Example04-LoRA微调.java](../examples/Example04-LoRA微调.java)
- [Example06-文本生成策略.java](../examples/Example06-文本生成策略.java)
- [Example07-模型评估.java](../examples/Example07-模型评估.java)
