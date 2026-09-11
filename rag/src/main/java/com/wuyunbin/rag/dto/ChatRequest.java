package com.wuyunbin.rag.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;

/**
 * RAG 聊天请求体。
 */
@Schema(description = "RAG 聊天请求")
public record ChatRequest(

        @NotBlank(message = "message 不能为空")
        @Schema(description = "用户输入的消息内容", example = "学校的图书馆开放时间是什么？", requiredMode = Schema.RequiredMode.REQUIRED)
        String message,

        @Schema(description = "会话 ID；不传或为空则新建会话，响应中返回", example = "3f2a7b6c-1d4e-4f5a-9b8c-2e3d4f5a6b7c")
        String conversationId
) {
}
