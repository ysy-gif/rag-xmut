package com.wuyunbin.rag.dto;

/**
 * 文档导入结果。
 *
 * @param fileCount  导入的文件数量
 * @param chunkCount 生成的文本块数量（写入 Milvus 的向量条数）
 */
public record ImportResult(int fileCount, int chunkCount) {
}
