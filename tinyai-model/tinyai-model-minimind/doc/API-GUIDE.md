# MiniMind API 服务使用指南

## 📚 概述

MiniMind API提供OpenAI兼容的REST API接口,支持文本补全和对话功能。基于Java标准库`HttpServer`实现,**零第三方依赖**。

### ✨ 特性

- ✅ **OpenAI兼容**: 兼容OpenAI API格式
- ✅ **零依赖**: 仅使用Java标准库
- ✅ **轻量简洁**: 核心代码<1000行
- ✅ **易于部署**: 单JAR包即可运行
- ✅ **CORS支持**: 支持跨域请求

---

## 🚀 快速开始

### 1. 启动服务器

**Linux/Mac:**
```bash
chmod +x bin/start-api.sh
./bin/start-api.sh 8080
```

**Windows:**
```cmd
bin\start-api.bat 8080
```

**使用Java直接启动:**
```bash
java -cp target/classes io.leavesfly.tinyai.minimind.api.MiniMindAPIServer 8080
```

### 2. 验证服务

```bash
curl http://localhost:8080/health
```

**预期响应:**
```json
{
  "status": "healthy",
  "timestamp": 1702834567890
}
```

---

## 📖 API端点

### 1. 文本补全 `/v1/completions`

**请求示例:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "prompt": "Hello, world!",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 否 | minimind | 模型名称 |
| `prompt` | string/array | 是 | - | 提示文本 |
| `max_tokens` | integer | 否 | 100 | 最大生成长度 |
| `temperature` | float | 否 | 0.7 | 采样温度(0-2) |
| `top_p` | float | 否 | 0.9 | 核采样概率 |
| `stream` | boolean | 否 | false | 流式响应(暂不支持) |

**响应示例:**
```json
{
  "id": "cmpl-7a8b9c0d1e2f",
  "object": "text_completion",
  "created": 1702834567,
  "model": "minimind",
  "choices": [
    {
      "text": "[Generated text...]",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 50,
    "total_tokens": 53
  }
}
```

---

### 2. 对话补全 `/v1/chat/completions`

**请求示例:**
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "minimind",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**请求参数:**

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | string | 否 | minimind | 模型名称 |
| `messages` | array | 是 | - | 消息列表 |
| `max_tokens` | integer | 否 | 100 | 最大生成长度 |
| `temperature` | float | 否 | 0.7 | 采样温度 |
| `top_p` | float | 否 | 0.9 | 核采样概率 |

**消息格式:**
```json
{
  "role": "system|user|assistant",
  "content": "消息内容"
}
```

**响应示例:**
```json
{
  "id": "chatcmpl-7a8b9c0d1e2f",
  "object": "chat.completion",
  "created": 1702834567,
  "model": "minimind",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "[Generated reply...]"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 30,
    "total_tokens": 50
  }
}
```

---

### 3. 模型列表 `/v1/models`

**请求示例:**
```bash
curl http://localhost:8080/v1/models
```

**响应示例:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "minimind",
      "object": "model",
      "created": 1702834567,
      "owned_by": "tinyai"
    }
  ]
}
```

---

### 4. 健康检查 `/health`

**请求示例:**
```bash
curl http://localhost:8080/health
```

**响应示例:**
```json
{
  "status": "healthy",
  "timestamp": 1702834567890
}
```

---

## 💻 代码集成示例

### Python (requests)

```python
import requests

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "minimind",
    "messages": [
        {"role": "user", "content": "你好!"}
    ],
    "max_tokens": 100
}

response = requests.post(url, json=data, headers=headers)
print(response.json())
```

### JavaScript (fetch)

```javascript
fetch('http://localhost:8080/v1/chat/completions', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'minimind',
    messages: [
      {role: 'user', content: 'Hello!'}
    ],
    max_tokens: 100
  })
})
.then(res => res.json())
.then(data => console.log(data));
```

### Java (HttpClient)

```java
HttpClient client = HttpClient.newHttpClient();
HttpRequest request = HttpRequest.newBuilder()
    .uri(URI.create("http://localhost:8080/v1/chat/completions"))
    .header("Content-Type", "application/json")
    .POST(HttpRequest.BodyPublishers.ofString(
        "{\"model\":\"minimind\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}"
    ))
    .build();

HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
System.out.println(response.body());
```

---

## ⚙️ 配置说明

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `PORT` | 服务端口 | 8080 |

### JVM参数

推荐配置:
```bash
-Xmx2g -Xms512m
```

---

## 🔧 开发集成

### 集成实际模型

当前API使用占位实现,需要集成实际的MiniMind模型:

**修改 `CompletionHandler.java`:**
```java
private String generateText(String prompt, int maxTokens, double temperature, double topP) {
    // TODO: 替换为实际模型推理
    MiniMindModel model = loadModel();
    return model.generate(prompt, maxTokens, temperature, topP);
}
```

**修改 `ChatCompletionHandler.java`:**
```java
private String generateChatReply(List<ChatMessage> messages, ...) {
    // TODO: 替换为实际模型推理
    MiniMindModel model = loadModel();
    String context = formatMessages(messages);
    return model.generate(context, maxTokens, temperature, topP);
}
```

---

## 🐛 故障排除

### 1. 端口已占用

**错误信息:**
```
java.net.BindException: Address already in use
```

**解决方法:**
- 更改端口: `./bin/start-api.sh 8081`
- 或终止占用进程: `lsof -ti:8080 | xargs kill`

### 2. 编译错误

**解决方法:**
```bash
mvn clean compile
```

### 3. 内存不足

**解决方法:**
增加JVM堆内存:
```bash
export JAVA_OPTS="-Xmx4g"
./bin/start-api.sh
```

---

## 📝 注意事项

1. **占位实现**: 当前版本为API框架,需要集成实际模型
2. **流式响应**: 暂不支持,将在后续版本实现
3. **认证鉴权**: 当前无认证,生产环境需添加
4. **并发限制**: 默认线程池10个,可根据需要调整

---

## 🔗 相关资源

- [OpenAI API文档](https://platform.openai.com/docs/api-reference)
- [MiniMind CLI指南](./CLI-GUIDE.md)
- [TODO任务清单](./TODO.md)

---

**版本**: v1.0.0  
**作者**: TinyAI Team  
**更新时间**: 2025-12-07
