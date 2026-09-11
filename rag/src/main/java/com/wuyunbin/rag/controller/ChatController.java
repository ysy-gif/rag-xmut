package com.wuyunbin.rag.controller;

import com.wuyunbin.rag.dto.ChatRequest;
import com.wuyunbin.rag.dto.ChatResponse;
import com.wuyunbin.rag.service.ChatService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * RAG 聊天接口。
 */
@RestController
@RequestMapping("/api/chat")
@RequiredArgsConstructor
@Tag(name = "聊天接口", description = "基于知识库的多轮问答接口")
public class ChatController {

    private final ChatService chatService;

    @PostMapping
    @Operation(summary = "RAG 多轮问答", description = "基于知识库回答问题；携带 conversationId 即可多轮对话，"
            + "不传或为空则新建会话。同一会话的请求须串行发送")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request) {
        return chatService.chat(request.conversationId(), request.message());
    }

    @DeleteMapping("/{conversationId}")
    @Operation(summary = "清除会话记忆", description = "清除指定会话的对话历史；幂等，会话不存在同样返回 204")
    public ResponseEntity<Void> clear(@PathVariable String conversationId) {
        chatService.clear(conversationId);
        return ResponseEntity.noContent().build();
    }
}
