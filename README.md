# On-Call 助手

这是一个基于 FastAPI 实现的 On-Call 助手示例项目，围绕 `data/` 目录下的 SOP HTML 文档提供三阶段能力：

- `/v1`：关键词搜索
- `/v2`：语义搜索
- `/v3`：On-Call 助手 Agent

项目对应题目中的三个阶段，并分别提供 HTTP API 和简单前端页面。

## 功能说明

### Phase 1：关键词搜索

- `POST /v1/documents`：导入或更新文档
- `GET /v1/search?q=...`：基于关键词检索文档
- `GET /v1`：关键词搜索页面

实现特点：

- 只索引 HTML 的**可见文本**
- 会忽略 `script` 等不可见区域中的内容
- 只返回**正文或标题中真实出现了查询词/短语**的文档
- 使用 BM25 风格评分，并加少量标题/小节标题加权

### Phase 2：语义搜索

- `GET /v2/search?q=...`：语义检索
- `GET /v2`：语义搜索页面

实现特点：

- 配置了 DashScope 时，可使用向量语义检索
- 未配置外部能力时，提供本地 fallback 检索逻辑
- 支持“查询词不要求在文档中原样出现”的相关性搜索

### Phase 3：On-Call 助手 Agent

- `GET /v3`：对话页面
- `POST /v3/chat`：非流式对话接口
- `POST /v3/chat/stream`：SSE 流式对话接口

实现特点：

- Agent 只有一个工具：`readFile(fname)`
- 只能按**精确文件名**读取 `data/` 下的文件
- 不允许列目录，不允许使用通配符
- 前端会展示工具调用过程
- 没有外部大模型能力时，仍可走本地 SOP fallback 回答流程

## 项目结构

```text
.
├── data/              # SOP HTML 文档
├── main.py            # FastAPI 应用、检索逻辑、语义搜索、Agent、前端页面
├── requirements.txt
├── .env.example
└── README.md
```

## 运行环境

- Python 3.10 及以上

安装依赖：

```bash
pip install -r requirements.txt
```

## 环境变量配置

如果希望启用 DashScope 的向量检索和聊天能力，可以将 `.env.example` 复制为 `.env` 后填写：

```env
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
DASHSCOPE_EMBEDDING_DIMENSIONS=1024
DASHSCOPE_CHAT_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
DASHSCOPE_CHAT_MODEL=qwen3.6-plus
```

如果不配置 `DASHSCOPE_API_KEY`：

- `/v1` 可正常工作
- `/v2` 会使用本地 fallback 语义检索
- `/v3` 会使用本地 SOP fallback 回答

## 启动方式

直接运行：

```bash
python main.py
```

启动后可访问：

- `http://127.0.0.1:8000/v1`
- `http://127.0.0.1:8000/v2`
- `http://127.0.0.1:8000/v3`

## 接口说明

### 1. `POST /v1/documents`

请求体：

```json
{
  "id": "sop-001",
  "html": "<html>...</html>"
}
```

响应：

```json
{
  "id": "sop-001",
  "title": "后端服务 On-Call SOP"
}
```

说明：

- 返回状态码为 `201`
- 文档会写入 `data/{id}.html`
- 同时会更新内存中的索引

### 2. `GET /v1/search?q=OOM`

响应结构：

```json
{
  "query": "OOM",
  "results": [
    {
      "id": "sop-001",
      "title": "后端服务 On-Call SOP",
      "snippet": "...",
      "score": 1.23
    }
  ]
}
```

### 3. `GET /v2/search?q=服务器挂了`

响应结构：

```json
{
  "query": "服务器挂了",
  "results": [
    {
      "id": "sop-004",
      "title": "SRE基础设施 On-Call SOP",
      "snippet": "...",
      "score": 0.89
    }
  ]
}
```

### 4. `POST /v3/chat`

请求体：

```json
{
  "message": "数据库主从延迟超过30秒怎么处理？",
  "history": []
}
```

响应结构：

```json
{
  "reply": "## 处理建议 ...",
  "tool_calls": [
    {
      "type": "tool_call",
      "name": "readFile",
      "fname": "sop-002.html",
      "status": "completed",
      "message": "已读取 sop-002.html"
    }
  ]
}
```

### 5. `POST /v3/chat/stream`

返回 `text/event-stream`，会按事件持续输出：

- 工具调用开始
- 工具调用完成
- 回答增量
- 最终完成事件

## 实现说明

### 文档处理

- 启动时会自动加载 `data/` 中已有的种子文档
- 导入文档时会抽取可见文本、标题和结构化片段
- 用于 Phase 1 的关键词索引和 Phase 2 / Phase 3 的后续检索

### `/v1` 的设计原则

- 严格按关键词搜索
- 不引入语义扩展
- 不因为“领域相近”就返回无命中文档

也就是说：

- `OOM` 只会匹配真正包含 `OOM` 的文档
- `replication` 如果只出现在脚本中，则不会命中
- “服务器挂了”这种更偏语义的问题，应该由 `/v2` 处理

### `/v2` 的设计原则

- 负责处理语义相近、词不完全一致的问题
- 优先使用外部 embedding 检索
- 外部能力不可用时，退化到本地混合排序

### `/v3` 的设计原则

- 先缩小候选 SOP
- 再通过唯一工具 `readFile` 读取精确文件
- 再基于已读文档生成回答
- 页面展示真实工具调用过程，而不是直接伪造“已读”

## 快速验证

可以用下面几条快速做 smoke test：

### Phase 1

- `GET /v1/search?q=OOM`
- `GET /v1/search?q=故障`
- `GET /v1/search?q=replication`
- `GET /v1/search?q=CDN`
- `GET /v1/search?q=&`

### Phase 2

- `GET /v2/search?q=服务器挂了`
- `GET /v2/search?q=黑客攻击`
- `GET /v2/search?q=机器学习模型出问题`

### Phase 3

- `数据库主从延迟超过30秒怎么处理？`
- `服务 OOM 了怎么办？`
- `P0 故障的响应流程是什么？`
- `怀疑有人入侵了系统`
- `推荐结果质量下降了`

## 备注

- `__pycache__/` 是 Python 自动生成的字节码缓存目录，可以忽略
- `data/` 目录中目前存放的是题目示例 SOP 文档
