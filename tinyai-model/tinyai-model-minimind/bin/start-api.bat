@echo off
REM MiniMind API Server 启动脚本 (Windows)

REM 切换到项目根目录
cd /d "%~dp0\.."

REM 设置Java选项
set JAVA_OPTS=-Xmx2g -Xms512m

REM 默认端口
set PORT=%1
if "%PORT%"=="" set PORT=8080

echo ========================================
echo Starting MiniMind API Server
echo ========================================
echo Port: %PORT%
echo Java Options: %JAVA_OPTS%
echo.

REM 编译项目(如果需要)
if not exist "target\classes\" (
    echo Building project...
    mvn clean compile
)

REM 启动服务器
for /f "delims=" %%i in ('mvn dependency:build-classpath -DincludeScope=compile -Dmdep.outputFile=/dev/stdout -q') do set CLASSPATH=%%i
java %JAVA_OPTS% -cp "target\classes;%CLASSPATH%" io.leavesfly.tinyai.minimind.api.MiniMindAPIServer %PORT%

echo.
echo Server stopped.
pause
