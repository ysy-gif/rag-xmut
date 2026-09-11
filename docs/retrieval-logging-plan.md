# RAG 召回文档详细日志与检索参数调整 — 实施计划

- 日期：2026-09-10
- 状态：待批准（计划已讨论确认两个决策点，未动代码）
- 涉及模块：`rag`（Spring Boot + Spring AI 2.0 + Milvus + LM Studio）

## 1. 背景与目标

当前 [ChatService.java](../rag/src/main/java/com/wuyunbin/rag/service/ChatService.java) 的检索链路只在请求结束时打印一条 INFO 摘要（片段数 + 分数列表），无法看到被召回片段的具体内容与来源，不便于阈值校准与检索质量排查。同时阈值当前为 0.4，需要上调到 0.45。

目标：

1. 召回文档低于相似度阈值 0.45 的过滤掉（阈值写入 `application.properties`）；
2. 召回文档按相似度降序排列（显式排序，不依赖向量库隐式行为）；
3. 以 DEBUG 级别打印每条召回片段的详细日志（含全文内容）。

## 2. 现状核对与澄清

| 事项 | 现状 | 结论 |
| --- | --- | --- |
| topK 是否写死在 `ChatConfig.java` | **不在**。`ChatConfig` 仅含记忆相关 3 个 Bean；检索参数在 `ChatService` 中通过 `@Value` 注入，`application.properties` 已有 `rag.retrieve.candidate-top-k=8`、`rag.retrieve.top-k=4` | **无需改动**（用户此前记忆有误） |
| 阈值配置是否已外置 | 已有 `rag.retrieve.similarity-threshold=0.4` | 仅需改值 0.4 → 0.45 |
| 召回结果顺序 | Milvus COSINE 检索默认降序返回，`filter` 不改变顺序，事实有序 | 补显式排序，让契约确定 |
| `KnowledgeConfig` 中的 `SIMILARITY_THRESHOLD = 0.50` | 这是**导入时语义切分器**的相邻句断开阈值，与检索召回无关 | **不要动** |

## 3. 改动清单

### 改动 1：阈值 0.4 → 0.45（application.properties）

[application.properties](../rag/src/main/resources/application.properties) 第 32 行：

```properties
# 修改前
rag.retrieve.similarity-threshold=0.4
# 修改后
rag.retrieve.similarity-threshold=0.45
```

说明：bge-m3 + COSINE 下弱相关片段通常落在 0.4~0.5 区间，0.45 偏保守，可能滤掉少量弱相关但有用的片段；先按 0.45 跑，结合本次新增的详细日志观察分数分布后再微调（改配置零成本）。

### 改动 2：显式降序排序（ChatService.retrieve()）

[ChatService.java](../rag/src/main/java/com/wuyunbin/rag/service/ChatService.java) 中 `retrieve()` 的流式管道改为：

```java
return candidates.stream()
        .filter(d -> d.getScore() != null)
        .sorted(Comparator.comparingDouble(Document::getScore).reversed())
        .filter(d -> d.getScore() >= similarityThreshold)
        .limit(topK)
        .toList();
```

要点：

- 先过滤 `score == null`，保证 `comparingDouble` 不会 NPE；
- 排序在阈值过滤之前，结果集不变，语义更直观；
- 8 个元素排序成本可忽略。

### 改动 3：召回片段详细日志（ChatService 新增私有方法）

新增 `logRetrievedDocs(String convId, String searchQuery, List<Document> docs)`，在 `doChat` 中检索完成后调用：

- **DEBUG 级别**，且用 `log.isDebugEnabled()` 守卫——避免生产环境关闭 DEBUG 时仍构造大字符串（4 片段 × 最多 1500 字符）的开销，符合"日志非阻塞、分环境"的约束；
- 每个片段打印：序号、score、`file_name`、`chunk_index`、文本长度、**全文内容**；
- 头部打印 convId 与 searchQuery，便于多轮场景下把召回结果与改写后的检索词对应起来；
- 现有的 INFO 摘要日志（片段数 + 分数列表）**保留不动**，生产环境仍有基本可观测性。

日志示意（多行结构化）：

```
召回明细 convId=xxx searchQuery=xxx 命中=2
[1] score=0.62 file_name=手册.pdf chunk_index=3 len=520
    <片段全文>
[2] score=0.47 file_name=手册.pdf chunk_index=7 len=830
    <片段全文>
```

### 改动 4：dev profile 打开服务层 DEBUG（application-dev.properties）

[application-dev.properties](../rag/src/main/resources/application-dev.properties) 追加：

```properties
logging.level.com.wuyunbin.rag=DEBUG
```

与已有的 `logging.level.org.springframework.ai=DEBUG` 并列，仅 dev 调试使用，上线不启用该 profile。

## 4. 明确不改动项

- `ChatConfig.java`：topK 本就外置在 `application.properties`，无需迁移；
- `KnowledgeConfig.java`：`SIMILARITY_THRESHOLD = 0.50` 是语义切分器阈值，与检索无关；
- `retrieve()` 的两级检索结构（宽召回 candidateTopK → 后过滤取 topK）不变；
- `buildContext()`、prompt 模板、会话记忆与锁逻辑均不变。

## 5. 已确认的决策记录

| 决策点 | 结论 | 备注 |
| --- | --- | --- |
| 召回全文日志级别 | **DEBUG** | 生产默认关闭，dev profile 打开 |
| 是否打印候选阶段（宽召回 8 条）分数 | **不打印** | 仅记录最终保留片段；阈值校准依赖最终分数与业务判断 |
| topK 迁移到配置文件 | 无需改动 | 已外置，属记忆偏差 |

## 6. 验证步骤

1. `mvn compile` 通过后以 dev profile 启动（Milvus、LM Studio 就绪）；
2. 提问一个知识库中**已知答案**的问题：
   - DEBUG 日志可见召回明细，片段按 score **降序**；
   - 所有片段 score **≥ 0.45**；
   - 片段全文、file_name、chunk_index 正确可见；
3. 提问一个知识库**无关**的问题：
   - 命中 0 条，`buildContext` 走 `NO_CONTEXT_HINT` 路径，回答明确说明"知识库中未找到相关内容"；
4. 观察 INFO 摘要与 DEBUG 明细的分数一致性；
5. 若发现大量弱相关片段被 0.45 滤掉导致漏召回，回到 `application.properties` 微调阈值（预期 0.42~0.48 区间）。

## 7. 风险与回滚

- 风险低：均为日志与配置/排序调整，不触及记忆、锁、prompt 组装等核心路径；
- 回滚方式：阈值改回 0.4、删除 `logRetrievedDocs` 调用与 dev 日志配置即可，无数据结构变更。
