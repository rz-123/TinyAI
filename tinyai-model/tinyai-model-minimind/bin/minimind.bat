@echo off
REM MiniMind CLI启动脚本(Windows)

REM 设置Java路径
set JAVA_CMD=java

REM 设置JAR包路径
set JAR_FILE=tinyai-model-minimind.jar

REM 设置JVM参数
set JVM_OPTS=-Xmx2g -Xms512m

REM 检查JAR包是否存在
if not exist "%JAR_FILE%" (
    echo 错误: 未找到JAR包 %JAR_FILE%
    echo 请先构建项目: mvn clean package
    exit /b 1
)

REM 执行命令
%JAVA_CMD% %JVM_OPTS% -cp %JAR_FILE% io.leavesfly.tinyai.minimind.cli.MiniMindCLI %*
