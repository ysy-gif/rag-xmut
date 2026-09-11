package com.wuyunbin.rag.service;

import com.wuyunbin.rag.dto.ChatResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.concurrent.locks.ReentrantLock;

/**
 * RAG 多轮问答服务。
 *
 * <p>流程：会话记忆读取 → 多轮指代消解（LLM 问题改写，首轮跳过）→ Milvus 检索
 * → 组装 RAG prompt → 主模型回答 → 会话记忆写回。</p>
 *
 * <p>并发：同一 conversationId 的请求通过 per-session 锁串行化（防双击/重试丢历史）；
 * 锁表与记忆仓库同为 LRU 有界（上限 {@code rag.chat.max-sessions}），锁不随 DELETE 摘除，
 * 避免新旧锁错配竞态。</p>
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class ChatService {

    /**
     * RAG 路径 system prompt（仅在本轮有检索时使用）。
     * 所有"如实说明/区分来源"的规则都放在 system 层级，权重高于 user 层指令。
     */
    private static final String RAG_SYSTEM_PROMPT = """
            你是一个严格基于企业知识库回答问题的助手，请遵守以下规则：
            1. 优先依据【参考资料】回答问题，做到准确、简洁、切题；
            2. 如果资料不足以回答问题，或资料与问题无关，必须明确告知用户"知识库中未找到相关内容"；
               此时可以补充常识性内容，但必须明确说明这部分不是来自知识库；
            3. 不编造知识库中不存在的事实、数据或引用；
            4. 使用简体中文回答。
            """;

    /**
     * 降级路径 system prompt（仅当问题改写失败、本轮不检索时使用）。
     * 与 RAG system 分开：该路径的 user 消息里没有参考资料，
     * 若沿用"优先依据参考资料"的措辞会误导模型。
     */
    private static final String PLAIN_CHAT_SYSTEM_PROMPT = """
            你是一个友好的中文对话助手。请基于对话上下文自然地回答用户的最新问题；
            如果不确定或不知道答案，请如实说明，不要编造。
            """;

    /**
     * 问题改写模板：history 严格只包含"当前轮之前"的内容（调用时当前消息尚未写入记忆），
     * question 是独立槽位放当前消息，两者不存在重复。
     */
    private static final String REWRITE_TEMPLATE = """
            以下是不包含当前问题的历史对话：
            {history}
            ----
            请根据历史对话，把用户的最新问题改写成一个独立、完整、脱离上下文也能被准确理解的检索问题：
            补全其中的指代词（如"它""那里""上面的"）和省略成分，保留原意，不添加新信息。
            最新问题：{question}
            要求：只输出改写后的问题本身，不要输出任何解释；如果问题本身已经完整，则原样输出。
            """;

    /**
     * RAG 主 prompt 模板（仅 ragEnabled 路径使用）。
     * 注意：question 槽位固定用用户原话，不要改成改写后的搜索词——
     * 模型回答的应是用户实际问的问题，改写词只用于检索。
     */
    private static final String RAG_CONTEXT_TEMPLATE = """
            参考资料：
            {context}
            ----
            问题：{question}
            """;

    /**
     * 检索无命中时的中性标记：只陈述事实，不含行为指令（行为规则统一由 system 承担）。
     */
    private static final String NO_CONTEXT_HINT = "（知识库中未检索到与该问题相关的内容）";

    /** 单个知识片段送入 prompt 的最大字符数，防上下文超限。 */
    private static final int MAX_FRAGMENT_CHARS = 1500;

    /** 参与改写的历史消息条数上限，控制改写 prompt 的体积。 */
    private static final int REWRITE_HISTORY_LIMIT = 6;

    private final ChatClient chatClient;
    private final ChatMemory chatMemory;
    private final VectorStore vectorStore;

    @Value("${rag.retrieve.candidate-top-k:8}")
    private int candidateTopK;

    @Value("${rag.retrieve.top-k:4}")
    private int topK;

    @Value("${rag.retrieve.similarity-threshold:0.4}")
    private double similarityThreshold;

    @Value("${rag.chat.max-sessions:100}")
    private int maxSessions;

    /**
     * per-session 锁表：与记忆仓库同上限的 LRU。
     * 注意 synchronizedMap 的所有操作（含 computeIfAbsent）都持同一 mutex，
     * removeEldestEntry 内不做日志 IO（锁淘汰无可见性价值，避免持锁 IO 反模式）。
     */
    private final Map<String, ReentrantLock> locks = Collections.synchronizedMap(
            new LinkedHashMap<>(16, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, ReentrantLock> eldest) {
                    return size() > maxSessions;
                }
            });

    /**
     * RAG 多轮问答。
     *
     * @param conversationId 会话 ID；null 或空白则新建会话
     * @param message        用户消息
     * @return 会话 ID 与模型回复
     */
    public ChatResponse chat(String conversationId, String message) {
        String convId = StringUtils.hasText(conversationId) ? conversationId : UUID.randomUUID().toString();
        ReentrantLock lock = locks.computeIfAbsent(convId, k -> new ReentrantLock());
        lock.lock();
        try {
            return doChat(convId, message);
        } finally {
            lock.unlock();
        }
    }

    /**
     * 清除指定会话的记忆。
     *
     * <p>用 {@code locks.get()} 而非 computeIfAbsent：清除不存在的会话零副作用（不占锁表 LRU 槽位）；
     * 有锁则取同一把锁串行化，避免清除与在跑请求交错导致记忆"复活"。</p>
     *
     * @param conversationId 会话 ID
     */
    public void clear(String conversationId) {
        ReentrantLock lock = locks.get(conversationId);
        if (lock == null) {
            chatMemory.clear(conversationId);
            return;
        }
        lock.lock();
        try {
            chatMemory.clear(conversationId);
        } finally {
            lock.unlock();
        }
    }

    private ChatResponse doChat(String convId, String message) {
        List<Message> history = chatMemory.get(convId);

        // searchQuery 语义 = "本轮可用的检索词"；null 表示本轮不检索（改写失败降级）
        String searchQuery;
        if (history.isEmpty()) {
            searchQuery = message;
        } else {
            searchQuery = rewriteQuestion(message, history).orElse(null);
        }
        boolean ragEnabled = searchQuery != null;

        List<Document> docs = List.of();
        boolean retrievalFailed = false;
        if (ragEnabled) {
            try {
                docs = retrieve(searchQuery);
                logRetrievedDocs(convId, searchQuery, docs);
            } catch (Exception e) {
                // Milvus 不可用等：本轮退化为普通对话，不向上抛
                retrievalFailed = true;
                log.error("检索失败，本轮降级为普通对话 convId={}", convId, e);
            }
        }

        String reply;
        if (ragEnabled && !retrievalFailed) {
            String context = buildContext(docs);
            // question 槽位固定用用户原话 message（勿用 searchQuery，否则答非所问）
            String userPrompt = new PromptTemplate(RAG_CONTEXT_TEMPLATE)
                    .render(Map.of("context", context, "question", message));
            reply = doCall(RAG_SYSTEM_PROMPT, history, userPrompt);
        } else {
            if (!ragEnabled) {
                log.warn("问题改写失败，本轮跳过检索 convId={} question={}", convId, message);
            }
            reply = doCall(PLAIN_CHAT_SYSTEM_PROMPT, history, message);
        }

        chatMemory.add(convId, List.of(new UserMessage(message), new AssistantMessage(reply)));

        log.info("chat 完成 convId={} 原始问题={} 搜索词={} ragEnabled={} 命中片段={} 分数={}",
                convId, message, searchQuery, ragEnabled && !retrievalFailed,
                docs.size(), docs.stream().map(Document::getScore).toList());
        return new ChatResponse(convId, reply);
    }

    /**
     * LLM 问题改写（多轮指代消解）。仅在存在历史时调用。
     *
     * @return 改写后的检索问题；LLM 调用失败或结果为空白时返回 {@link Optional#empty()}
     */
    private Optional<String> rewriteQuestion(String question, List<Message> history) {
        List<Message> recent = history.subList(Math.max(0, history.size() - REWRITE_HISTORY_LIMIT), history.size());
        String historyText = historyToText(recent);
        String prompt = new PromptTemplate(REWRITE_TEMPLATE)
                .render(Map.of("history", historyText, "question", question));
        try {
            String rewritten = chatClient.prompt()
                    // 独立参数：低温、限长，避免改写调用啰嗦拉高延迟。
                    // 2.0 的 options() 接收 Builder，框架内部与默认 options combineWith 合并，model 等保留
                    .options(OpenAiChatOptions.builder().temperature(0.2).maxTokens(200))
                    .user(prompt)
                    .call()
                    .content();
            if (StringUtils.hasText(rewritten)) {
                return Optional.of(rewritten.trim());
            }
            log.warn("改写结果为空白，视为失败 question={}", question);
            return Optional.empty();
        } catch (Exception e) {
            log.warn("问题改写调用失败 question={}", question, e);
            return Optional.empty();
        }
    }

    /**
     * 两级检索：先宽召回（topK 直接作为 Milvus 搜索 limit；不设 similarityThreshold，
     * 默认 0.0 = 搜索后 Java 侧过滤全放行），再按相似度降序显式排序、阈值后过滤取前 topK。
     * 显式排序不依赖向量库的隐式返回顺序，保证"越相关越靠前"的契约确定。
     */
    private List<Document> retrieve(String query) {
        List<Document> candidates = vectorStore.similaritySearch(SearchRequest.builder()
                .query(query)
                .topK(candidateTopK)
                .build());
        if (candidates == null) {
            return List.of();
        }
        return candidates.stream()
                .filter(d -> d.getScore() != null)
                .sorted(Comparator.comparingDouble(Document::getScore).reversed())
                .filter(d -> d.getScore() >= similarityThreshold)
                .limit(topK)
                .toList();
    }

    /**
     * 召回片段详细日志（INFO 级别）：会话头 + 分隔线 + 逐条"元数据行 / 全文行" + END 分隔线，
     * 用于检索质量排查与相似度阈值校准。空命中也记录，
     * 便于区分"检索执行了但 0 命中"与"本轮未检索"。
     */
    private void logRetrievedDocs(String convId, String searchQuery, List<Document> docs) {
        String sep = System.lineSeparator();
        StringBuilder detail = new StringBuilder(sep)
                .append("conversationId: ").append(convId).append(sep)
                .append("query        : ").append(searchQuery).append(sep)
                .append("= = = = = = 检索命中 ").append(docs.size()).append(" 条 (score desc) = = = = = =");
        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i);
            String text = doc.getText();
            String tag = "[正文 " + (i + 1) + "]";
            detail.append(sep)
                    .append(tag).append(" id=").append(doc.getId())
                    .append(", score=").append(doc.getScore())
                    .append(", metadata=").append(doc.getMetadata())
                    .append(sep)
                    .append(tag).append(' ').append(text == null ? "" : text);
        }
        detail.append(sep)
                .append("= = = = = = END (共 ").append(docs.size()).append(" 条) = = = = = =");
        log.info("召回明细 {}", detail);
    }

    /**
     * 将检索结果组装为带编号与来源的参考资料文本；每片段超长截断。
     */
    private String buildContext(List<Document> docs) {
        if (docs.isEmpty()) {
            return NO_CONTEXT_HINT;
        }
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < docs.size(); i++) {
            Document doc = docs.get(i);
            String text = doc.getText();
            if (text != null && text.length() > MAX_FRAGMENT_CHARS) {
                text = text.substring(0, MAX_FRAGMENT_CHARS) + "…";
            }
            Object source = doc.getMetadata().get("file_name");
            sb.append('[').append(i + 1).append("] ");
            if (source != null) {
                sb.append("(来源: ").append(source).append(") ");
            }
            sb.append(text).append(System.lineSeparator());
        }
        return sb.toString();
    }

    /**
     * 主模型调用：最终消息顺序为 [System, ...history, UserMessage]，历史由 messages 追加传入。
     */
    private String doCall(String systemPrompt, List<Message> history, String userPrompt) {
        return chatClient.prompt()
                .system(systemPrompt)
                .messages(history)
                .user(userPrompt)
                .call()
                .content();
    }

    private String historyToText(List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            String role = msg instanceof UserMessage ? "用户" : "助手";
            sb.append(role).append(": ").append(msg.getText()).append(System.lineSeparator());
        }
        return sb.toString();
    }
}
