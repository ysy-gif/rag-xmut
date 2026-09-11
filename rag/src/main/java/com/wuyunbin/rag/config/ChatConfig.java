package com.wuyunbin.rag.config;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Spring AI ChatClient 与会话记忆配置。
 */
@Configuration
public class ChatConfig {

    @Bean
    public BoundedChatMemoryRepository boundedChatMemoryRepository(
            @Value("${rag.chat.max-sessions:100}") int maxSessions) {
        return new BoundedChatMemoryRepository(maxSessions);
    }

    @Bean
    public ChatClient chatClient(ChatClient.Builder builder) {
        return builder.build();
    }

    /**
     * 有界窗口会话记忆：每会话保留最近 windowSize 条消息（约 windowSize/2 轮对话），
     * 仓库为 LRU 有界的内存实现，防止会话数无限增长。
     */
    @Bean
    public ChatMemory chatMemory(BoundedChatMemoryRepository repository,
                                 @Value("${rag.chat.memory-window-size:12}") int windowSize) {
        return MessageWindowChatMemory.builder()
                .chatMemoryRepository(repository)
                .maxMessages(windowSize)
                .build();
    }
}
