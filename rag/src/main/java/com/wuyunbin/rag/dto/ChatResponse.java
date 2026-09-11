package com.wuyunbin.rag.dto;

import io.swagger.v3.oas.annotations.media.Schema;

/**
 * RAG 聊天响应体。
 */
@Schema(description = "RAG 聊天响应")
public record ChatResponse(

        @Schema(description = "会话 ID，多轮对话时需回传")
        String conversationId,

        @Schema(description = "模型回复内容")
        String reply
) {
}
