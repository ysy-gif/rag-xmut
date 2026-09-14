package com.wuyunbin.rag.config;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

import com.wuyunbin.rag.service.MarkdownChunkSplitter;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.transformer.splitter.TextSplitter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * 知识库相关配置：文档语义切分器。
 */
@Configuration
public class KnowledgeConfig {

    /** 相邻句子余弦相似度低于该值视为语义断点（bge-m3 向量，实测同主题 0.5~0.7、跨主题 <0.5） */
    private static final double SIMILARITY_THRESHOLD = 0.50;

    /** 文本块最小长度（字符），过小的块并入相邻块 */
    private static final int MIN_CHUNK_CHARS = 150;

    /** 文本块最大长度（字符），超过则按句硬切 */
    private static final int MAX_CHUNK_CHARS = 800;

    @Bean
    public SemanticTextSplitter semanticTextSplitter(EmbeddingModel embeddingModel) {
        return new SemanticTextSplitter(embeddingModel, SIMILARITY_THRESHOLD, MIN_CHUNK_CHARS, MAX_CHUNK_CHARS);
    }

    /**
     * Markdown 状态机切片器：无状态、线程安全，供清洗切片路径（rag.chunk.filter-enabled=true）使用。
     */
    @Bean
    public MarkdownChunkSplitter markdownChunkSplitter() {
        return new MarkdownChunkSplitter();
    }

    /**
     * 语义切分器：先把文本按句切分，再用嵌入模型计算相邻句子的余弦相似度，
     * 在相似度低于阈值的相邻位置断开，把语义连贯的句子聚合成文本块。
     */
    @Slf4j
    public static class SemanticTextSplitter extends TextSplitter {

        /** 中文句末标点后断开；换行后断开（保留标点在前一句） */
        private static final Pattern SENTENCE_BOUNDARY = Pattern.compile("(?<=[。！？!?；;])|(?<=\\n)");

        private final EmbeddingModel embeddingModel;

        private final double similarityThreshold;

        private final int minChunkChars;

        private final int maxChunkChars;

        public SemanticTextSplitter(EmbeddingModel embeddingModel, double similarityThreshold, int minChunkChars,
                int maxChunkChars) {
            this.embeddingModel = embeddingModel;
            this.similarityThreshold = similarityThreshold;
            this.minChunkChars = minChunkChars;
            this.maxChunkChars = maxChunkChars;
        }

        @Override
        protected List<String> splitText(String text) {
            List<String> sentences = splitSentences(text);
            if (sentences.size() <= 1) {
                return sentences;
            }
            // 批量向量化所有句子
            List<float[]> vectors = embeddingModel.embed(sentences);

            // 1. 相邻句子相似度低于阈值、或遇到 Markdown 标题时断开，聚合成块
            List<String> chunks = new ArrayList<>();
            StringBuilder current = new StringBuilder(sentences.get(0));
            for (int i = 1; i < sentences.size(); i++) {
                double similarity = cosine(vectors.get(i - 1), vectors.get(i));
                log.debug("语义切分 相邻句相似度: {} | 句{} -> 句{}", similarity, i - 1, i);
                if (similarity < similarityThreshold || sentences.get(i).startsWith("#")) {
                    chunks.add(current.toString());
                    current = new StringBuilder();
                }
                current.append(sentences.get(i));
            }
            chunks.add(current.toString());
            log.debug("语义切分: {} 句 -> {} 个语义块", sentences.size(), chunks.size());

            // 2. 过小的块并入相邻块
            for (int i = 0; i < chunks.size(); i++) {
                String chunk = chunks.get(i);
                if (chunk.length() < minChunkChars) {
                    if (i > 0) {
                        chunks.set(i - 1, chunks.get(i - 1) + chunk);
                    }
                    else if (i + 1 < chunks.size()) {
                        chunks.set(i + 1, chunk + chunks.get(i + 1));
                    }
                }
            }
            chunks.removeIf(chunk -> chunk.length() < minChunkChars);

            // 3. 超过最大长度的块按句硬切
            List<String> result = new ArrayList<>();
            for (String chunk : chunks) {
                if (chunk.length() <= maxChunkChars) {
                    result.add(chunk);
                }
                else {
                    result.addAll(hardSplit(chunk));
                }
            }
            return result;
        }

        private List<String> splitSentences(String text) {
            List<String> sentences = new ArrayList<>();
            for (String part : SENTENCE_BOUNDARY.split(text)) {
                String trimmed = part.trim();
                if (!trimmed.isEmpty()) {
                    sentences.add(trimmed);
                }
            }
            return sentences;
        }

        /** 把超长块按句重新切到 maxChunkChars 以内 */
        private List<String> hardSplit(String chunk) {
            List<String> parts = new ArrayList<>();
            StringBuilder current = new StringBuilder();
            for (String sentence : splitSentences(chunk)) {
                if (!current.isEmpty() && current.length() + sentence.length() > maxChunkChars) {
                    parts.add(current.toString());
                    current = new StringBuilder();
                }
                current.append(sentence);
            }
            if (!current.isEmpty()) {
                parts.add(current.toString());
            }
            return parts;
        }

        private double cosine(float[] a, float[] b) {
            double dot = 0;
            double normA = 0;
            double normB = 0;
            for (int i = 0; i < a.length; i++) {
                dot += a[i] * b[i];
                normA += a[i] * a[i];
                normB += b[i] * b[i];
            }
            return normA == 0 || normB == 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
        }
    }
}
