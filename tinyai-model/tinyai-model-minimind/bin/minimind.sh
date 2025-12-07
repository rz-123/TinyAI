#!/bin/bash
# MiniMind CLI启动脚本

# 设置Java路径(可根据需要修改)
JAVA_CMD="java"

# 设置JAR包路径
JAR_FILE="tinyai-model-minimind.jar"

# 设置JVM参数
JVM_OPTS="-Xmx2g -Xms512m"

# 检查Java是否可用
if ! command -v $JAVA_CMD &> /dev/null; then
    echo "错误: 未找到Java命令,请安装Java 17或更高版本"
    exit 1
fi

# 检查JAR包是否存在
if [ ! -f "$JAR_FILE" ]; then
    echo "错误: 未找到JAR包 $JAR_FILE"
    echo "请先构建项目: mvn clean package"
    exit 1
fi

# 执行命令
$JAVA_CMD $JVM_OPTS -cp $JAR_FILE io.leavesfly.tinyai.minimind.cli.MiniMindCLI "$@"
