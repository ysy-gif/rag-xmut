package com.wuyunbin.rag.config;

import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.memory.ChatMemoryRepository;
import org.springframework.ai.chat.messages.Message;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 有界的内存会话记忆仓库（LRU 淘汰），替代无上限的 {@code InMemoryChatMemoryRepository}
 * （后者为 final 类无法继承）。
 *
 * <p>内部为 access-order LinkedHashMap，超过 maxSessions 时淘汰最久未访问的会话，
 * 防止大量随机 conversationId 长期运行导致的内存泄漏。
 * 重启即丢失（内存存储），仅适用于演示/内网场景。</p>
 */
@Slf4j
public class BoundedChatMemoryRepository implements ChatMemoryRepository {

    private final Map<String, List<Message>> store;

    public BoundedChatMemoryRepository(int maxSessions) {
        this.store = Collections.synchronizedMap(new LinkedHashMap<>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, List<Message>> eldest) {
                boolean evict = size() > maxSessions;
                if (evict) {
                    // 淘汰打 INFO：会话数据消失需可见。频率至多每次请求一次，
                    // 在 synchronizedMap 的 mutex 内做少量 IO，演示规模可接受
                    log.info("会话记忆 LRU 淘汰: {}", eldest.getKey());
                }
                return evict;
            }
        });
    }

    @Override
    public List<String> findConversationIds() {
        synchronized (store) {
            return List.copyOf(store.keySet());
        }
    }

    @Override
    public List<Message> findByConversationId(String conversationId) {
        List<Message> messages = store.get(conversationId);
        return messages == null ? List.of() : List.copyOf(messages);
    }

    @Override
    public void saveAll(String conversationId, List<Message> messages) {
        if (messages == null || messages.isEmpty()) {
            deleteByConversationId(conversationId);
            return;
        }
        store.put(conversationId, List.copyOf(messages));
    }

    @Override
    public void deleteByConversationId(String conversationId) {
        store.remove(conversationId);
    }
}
