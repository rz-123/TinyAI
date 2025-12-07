#!/bin/bash
# MiniMind API Server 启动脚本 (Linux/Mac)

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 设置Java选项
JAVA_OPTS="-Xmx2g -Xms512m"

# 默认端口
PORT=${1:-8080}

echo "========================================"
echo "Starting MiniMind API Server"
echo "========================================"
echo "Port: $PORT"
echo "Java Options: $JAVA_OPTS"
echo ""

# 编译项目(如果需要)
if [ ! -d "target/classes" ]; then
    echo "Building project..."
    mvn clean compile
fi

# 启动服务器
java $JAVA_OPTS -cp "target/classes:$(mvn dependency:build-classpath -DincludeScope=compile -Dmdep.outputFile=/dev/stdout -q)" \
    io.leavesfly.tinyai.minimind.api.MiniMindAPIServer $PORT

echo ""
echo "Server stopped."
