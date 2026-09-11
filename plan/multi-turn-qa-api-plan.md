# RAG 多轮问答接口实施计划

> 版本：v1.0　日期：2026-09-10　状态：待评审
>
> 前提：文档已通过 `/api/knowledge/import` 切片导入 Milvus（collection: `knowledge_base`，bge-m3，1024 维，COSINE）。

---

## 1. 背景与现状

| 现状 | 说明 |
| --- | --- |
| 技术栈 | Spring Boot 4.1.1 + Java 21 + Spring AI 2.0.0 + Milvus（docker-compose）+ LM Studio（OpenAI 兼容协议） |
| 已有能力 | `POST /api/chat` 单轮聊天（无上下文、无知识库）；`POST /api/knowledge/import(-directory)` 文档导入 |
| 缺口 | 无"基于知识库的多轮问答"接口；无全局异常处理；测试几乎为空（仅 contextLoads） |

## 2. 需求目标

1. 新增**知识库问答接口**：先从 Milvus 检索相关文本块，再让大模型基于检索结果回答。
2. 支持**多轮对话**：同一会话内，后一问可以指代前一问（如"它的作者是谁？"），模型能结合历史作答。
3. 具备**规范化错误处理**与完整的**单元测试 + 集成测试**，整体行覆盖率 ≥ 30%。

## 3. 总体设计

### 3.1 RAG 问答流程

```
用户提问(message, conversationId?)
        │
        ▼
① 参数校验（非空、长度 ≤ 2000）
        │
        ▼
② ConversationMemoryService 取会话历史（无 id 则新建会话）
        │
        ▼
③ Milvus 相似度检索（topK=4, threshold=0.5, 取 fileName/score）
        │
        ▼
④ 组装 Prompt：
   System（角色设定：仅依据知识库回答，无依据时如实说明）
   + 检索到的知识片段
   + 最近 N 轮历史（UserMessage/AssistantMessage）
   + 当前用户问题
        │
        ▼
⑤ ChatClient 调用 LM Studio 生成回答
        │
        ▼
⑥ 追加本轮问答到会话历史，返回 {conversationId, answer, sources}
```

> v1 不做"问题改写"（condense question）：直接把历史消息拼入 Prompt，少一次 LLM 调用；作为后续优化项记录。

### 3.2 多轮会话方案

- **会话标识**：服务端生成 UUID 作为 `conversationId`，首次请求可不传，响应中返回；客户端后续请求携带。
- **历史存储**：内存实现（`ConcurrentHashMap<String, Deque<ChatMessage>>`），每会话最多保留**最近 10 轮**（20 条消息），超出淘汰最早轮次。应用重启即清空（可接受，避免引入数据库；后续可换 Redis/DB）。
- **未知 conversationId**：视为新会话，返回新生成的 id（客户端无感，避免 404 分支）。

### 3.3 配置项（新增 application.properties）

```properties
# ============ RAG 问答 ============
rag.qa.top-k=4                      # 检索返回片段数
rag.qa.similarity-threshold=0.5     # 相似度阈值
rag.qa.history-max-turns=10         # 会话保留的最大历史轮数
rag.qa.max-message-chars=2000       # 单条提问最大长度
```

## 4. 接口设计

### 4.1 知识库多轮问答

| 项目 | 内容 |
| --- | --- |
| 路径 | `POST /api/qa` |
| 请求方式 | HTTP POST |
| Content-Type | `application/json` |
| 鉴权 | 无（与现有接口保持一致） |
| 幂等性 | 否 |

**入参**（`QaRequest` record）

| 字段 | 类型 | 必填 | 约束 | 说明 |
| --- | --- | --- | --- | --- |
| message | String | 是 | 1 ~ 2000 字符，去除首尾空白后非空 | 用户当前提问 |
| conversationId | String | 否 | UUID 格式（不校验格式，仅作键） | 会话 id；不传则开启新会话 |

**出参 200**（`QaResponse` record）

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| conversationId | String | 会话 id，多轮请求需回传 |
| answer | String | 模型基于知识库生成的回答 |
| sources | Array | 引用的知识片段来源，检索为空时为 `[]` |
| sources[].fileName | String | 片段来源文档名 |
| sources[].score | Double | 相似度得分（0~1，越高越相关） |

**错误出参**（`ErrorResponse` record，由新增的全局异常处理器 `GlobalExceptionHandler` 返回）

| HTTP | code | 触发场景 |
| --- | --- | --- |
| 400 | `INVALID_ARGUMENT` | 缺少 body / message 为空或纯空白 / message 超长 |
| 400 | `MALFORMED_JSON` | JSON 格式错误（HttpMessageNotReadableException） |
| 500 | `RETRIEVAL_ERROR` | Milvus 检索异常 |
| 502 | `MODEL_ERROR` | LLM 调用失败/超时 |
| 500 | `INTERNAL_ERROR` | 其他未预期异常 |

### 4.2 请求示例

首轮：

```bash
curl -X POST http://localhost:8080/api/qa \
  -H "Content-Type: application/json" \
  -d '{"message": "Spring AI 是什么？"}'
```

```json
{
  "conversationId": "3f2a9c1e-8b7d-4c2a-9f31-0d5e6a7b8c9d",
  "answer": "Spring AI 是……（基于知识库内容）",
  "sources": [
    {"fileName": "spring-ai.pdf", "score": 0.82},
    {"fileName": "spring-ai.pdf", "score": 0.76}
  ]
}
```

追问（携带 conversationId）：

```bash
curl -X POST http://localhost:8080/api/qa \
  -H "Content-Type: application/json" \
  -d '{"message": "它的核心概念有哪些？", "conversationId": "3f2a9c1e-..."}'
```

## 5. 代码改动清单

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `dto/QaRequest.java` | 新增 | 入参 record + Swagger 注解 |
| `dto/QaResponse.java` | 新增 | 出参 record |
| `dto/SourceRef.java` | 新增 | 来源片段（fileName, score） |
| `dto/ErrorResponse.java` | 新增 | 错误响应 record（code, message） |
| `exception/ErrorCode.java` | 新增 | 错误码枚举 |
| `exception/RagException.java` | 新增 | 业务异常（携带 ErrorCode） |
| `exception/GlobalExceptionHandler.java` | 新增 | `@RestControllerAdvice`，统一映射 HTTP 状态码 |
| `service/ConversationMemoryService.java` | 新增 | 内存会话历史管理 |
| `service/QaService.java` | 新增 | 检索 + Prompt 组装 + 调用 LLM + 写历史 |
| `controller/QaController.java` | 新增 | `POST /api/qa` + Swagger 注解 |
| `application.properties` | 修改 | 增加 rag.qa.* 配置 |
| `test/.../QaServiceTest.java` | 新增 | 单元测试 |
| `test/.../ConversationMemoryServiceTest.java` | 新增 | 单元测试 |
| `test/.../QaControllerIT.java` | 新增 | MockMvc 集成测试（mock 外部依赖） |
| `test/.../RagApplicationTests.java` | 修改 | 用 `@MockitoBean` mock 外部依赖，使套件可离线运行 |
| `pom.xml` | 修改 | 增加 jacoco-maven-plugin（verify 阶段强制覆盖率 ≥ 30%） |

现有 `/api/chat`、知识导入接口保持不变。

## 6. 测试计划

### 6.1 测试策略

- **单元测试**：JUnit 5 + Mockito，不起 Spring 容器，mock `VectorStore` / `ChatClient`（含 `ChatClient.ChatClientRequestSpec` 链式调用）/ `ConversationMemoryService`，覆盖 QaService、ConversationMemoryService 全部分支。
- **集成测试**：`@SpringBootTest(webEnvironment = MOCK)` + `MockMvc`，用 `@MockitoBean` mock 掉 `VectorStore`、`EmbeddingModel`、`ChatModel`（LM Studio、Milvus 不需要启动），验证 HTTP → Controller → Service → 异常映射全链路，并用 `ArgumentCaptor` 验证多轮时 Prompt 中确实包含历史消息。
- **端到端冒烟（可选，默认跳过）**：`@EnabledIfEnvironmentVariable(named = "RAG_E2E", matches = "true")`，真实调用 Milvus + LM Studio 验证多轮效果。

### 6.2 单元测试用例

| 编号 | 场景 | 前置 | 输入 | 预期 |
| --- | --- | --- | --- | --- |
| UT-01 | 成功：检索命中 | mock 检索返回 2 个 Document | message="Spring AI 是什么？" | 返回答案非空；sources 含 fileName/score；prompt 含知识片段 |
| UT-02 | 成功：检索为空 | mock 检索返回空列表 | 任意合法 message | 正常返回，sources=[]，prompt 含"知识库未找到相关内容"提示 |
| UT-03 | 成功：多轮携带历史 | 会话中已有 1 轮历史 | message + 同一 conversationId | 发给模型的 prompt 包含上一轮 User/Assistant 消息 |
| UT-04 | 历史截断 | 预置 11 轮历史 | 再次提问 | 仅保留最近 10 轮 |
| UT-05 | 未知 conversationId | 空存储 | 传入不存在的 id | 作为新会话处理，返回新 UUID |
| UT-06 | 失败：message 为空 | — | `""` / `"   "` / `null` | 抛 `RagException(INVALID_ARGUMENT)`，历史不被写入 |
| UT-07 | 失败：message 超长 | — | 2001 字符 | 抛 `RagException(INVALID_ARGUMENT)` |
| UT-08 | 失败：检索异常 | mock 抛 DataAccessException | 合法 message | 抛 `RagException(RETRIEVAL_ERROR)`，原始异常为 cause |
| UT-09 | 失败：模型调用异常 | mock ChatClient 链路抛异常 | 合法 message | 抛 `RagException(MODEL_ERROR)` |
| UT-10 | 历史写入 | UT-01 成功后 | 查询存储 | 本轮 User/Assistant 消息已追加 |

### 6.3 集成测试用例（MockMvc）

| 编号 | 场景 | 请求 | 预期 |
| --- | --- | --- | --- |
| IT-01 | 成功问答 | `POST /api/qa`，`{"message":"什么是RAG"}`，mock 检索命中 | 200；body 含 conversationId（UUID）、answer、sources[].fileName/score |
| IT-02 | 多轮上下文 | IT-01 后携 conversationId 再问 | 200；`ArgumentCaptor` 证实第二次调用模型的 Prompt 含第一轮问答内容；conversationId 不变 |
| IT-03 | 新会话（不传 id） | message 合法、无 conversationId | 200；返回新 conversationId |
| IT-04 | 未知 conversationId | 传随机 UUID | 200；返回**新的** conversationId（视为新会话） |
| IT-05 | 失败：message 空白 | `{"message":"  "}` | 400；`code=INVALID_ARGUMENT` |
| IT-06 | 失败：message 超长 | 2001 字符 | 400；`code=INVALID_ARGUMENT` |
| IT-07 | 失败：缺少 body | POST 空 body | 400；`code=MALFORMED_JSON`（HttpMessageNotReadableException） |
| IT-08 | 失败：JSON 格式错 | `{"message": }` | 400；`code=MALFORMED_JSON` |
| IT-09 | 失败：检索异常 | mock VectorStore 抛异常 | 500；`code=RETRIEVAL_ERROR` |
| IT-10 | 失败：模型异常 | mock ChatClient 抛异常 | 502；`code=MODEL_ERROR` |
| IT-11 | 文档可访问（可选） | `GET /v3/api-docs` | 200，含 `/api/qa` |

### 6.4 覆盖率要求

- 引入 **jacoco-maven-plugin**，绑定 `check` 到 `verify` 阶段，规则：**整体 LINE 覆盖率 ≥ 0.30**（BUNDLE 维度），未达标构建失败。
- 新增核心代码（QaService / ConversationMemoryService / QaController）内部目标 ≥ 70%，以留出余量确保整体达标。
- 执行与查看：

```bash
mvn clean verify                 # 测试 + 覆盖率强制校验
# 报告：rag/target/site/jacoco/index.html
```

## 7. 验收标准

1. **功能**：`POST /api/qa` 按 4.1 契约工作；首轮提问返回答案与来源，携带 conversationId 追问时模型能结合上一轮上下文作答（人工冒烟用例：先问"Spring AI 是什么？"，再问"它的 ChatClient 怎么用？"）。
2. **文档**：Knife4j（/swagger-ui.html）可见新接口及参数说明。
3. **测试**：`mvn clean test` 全部通过，且不依赖 Milvus / LM Studio 启动（外部依赖全部 mock）。
4. **覆盖率**：`mvn clean verify` 通过 JaCoCo 强制校验，整体行覆盖率 ≥ 30%，报告可查。
5. **错误处理**：6.3 中 IT-05 ~ IT-10 的错误场景均返回约定的 HTTP 状态码与 code，接口不抛裸 500 堆栈。
6. **回归**：现有 `/api/chat`、`/api/knowledge/import` 行为不受影响。
7. **真实环境冒烟**（可选 RAG_E2E）：docker compose up 的 Milvus + LM Studio 下，导入样例文档 → 两轮问答 → 第二轮能正确指代第一轮实体。

## 8. 实施步骤

1. DTO + 异常体系（ErrorCode / RagException / GlobalExceptionHandler）
2. `ConversationMemoryService`（内存历史）+ 配置项
3. `QaService`（检索 → Prompt → LLM → 历史）
4. `QaController` + Swagger 注解
5. JaCoCo 接入 pom.xml
6. 编写单元测试（UT-01 ~ UT-10）
7. 编写集成测试（IT-01 ~ IT-11），修正 contextLoads 为可离线运行
8. `mvn clean verify` 核对覆盖率与验收标准，真实环境冒烟

## 9. 风险与说明

| 风险 | 对策 |
| --- | --- |
| 内存历史重启丢失、多实例不共享 | 当前单机场景可接受；预留接口便于后续换 Redis/DB |
| 追问指代不清导致检索偏差 | v1 用历史拼 Prompt 缓解；后续可加"问题改写"（额外一次 LLM 调用） |
| LM Studio 无返回/超时 | ChatClient 异常统一映射 `MODEL_ERROR`(502) |
| Spring Boot 4 测试注解变化 | 使用 `@MockitoBean`（非已废弃的 `@MockBean`） |
