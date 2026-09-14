package com.wuyunbin.rag.service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import com.wuyunbin.rag.config.KnowledgeConfig.SemanticTextSplitter;
import com.wuyunbin.rag.dto.ImportResult;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.document.Document;
import org.springframework.ai.reader.tika.TikaDocumentReader;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.FileSystemResource;
import org.springframework.core.io.Resource;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/**
 * 知识库服务：文档解析 -> 清洗切片 -> 向量化 -> 写入 Milvus。
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class KnowledgeService {

    /** 支持的文档扩展名 */
    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
            "txt", "md", "markdown", "pdf", "doc", "docx", "html", "htm", "csv");

    private final VectorStore vectorStore;

    private final SemanticTextSplitter semanticTextSplitter;

    private final MarkdownChunkSplitter markdownChunkSplitter;

    /** 清洗切片开关：true 走 Markdown 状态机切片（过滤曲调/目录、表格整块保留），false 回退语义切分 */
    @Value("${rag.chunk.filter-enabled:true}")
    private boolean chunkFilterEnabled;

    /**
     * 导入上传的单个文档。
     *
     * @param file 上传的文件
     * @return 导入结果
     */
    public ImportResult importUpload(MultipartFile file) {
        if (file.isEmpty()) {
            throw new IllegalArgumentException("上传文件为空");
        }
        int chunkCount = importResource(file.getResource(), file.getOriginalFilename(), new AtomicInteger());
        return new ImportResult(1, chunkCount);
    }

    /**
     * 导入服务器目录下的全部文档（非递归，按文件名排序）。
     *
     * @param directory 服务器本地目录路径
     * @return 导入结果
     */
    public ImportResult importDirectory(String directory) {
        Path dir = Path.of(directory);
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException("目录不存在或不是文件夹: " + directory);
        }
        try (Stream<Path> stream = Files.list(dir)) {
            List<Path> files = stream.filter(Files::isRegularFile)
                    .filter(this::isSupported)
                    .sorted()
                    .toList();
            AtomicInteger chunkCursor = new AtomicInteger();
            int totalChunks = 0;
            for (Path file : files) {
                totalChunks += importResource(new FileSystemResource(file), file.getFileName().toString(), chunkCursor);
            }
            log.info("目录导入完成: {} 个文件, 共 {} 个文本块", files.size(), totalChunks);
            return new ImportResult(files.size(), totalChunks);
        }
        catch (IOException e) {
            throw new IllegalStateException("读取目录失败: " + directory, e);
        }
    }

    /**
     * 解析 -> 清洗切片 -> 向量化 -> 写入 Milvus，返回生成的文本块数量。
     * chunkCursor 用于 chunk_index 跨文件全局自增。
     */
    private int importResource(Resource resource, String fileName, AtomicInteger chunkCursor) {
        List<Document> documents = new TikaDocumentReader(resource).get();
        if (documents.isEmpty() || documents.stream().allMatch(d -> d.getText().isBlank())) {
            log.warn("文档无可用内容，已跳过: {}", fileName);
            return 0;
        }
        return chunkFilterEnabled
                ? importStatefulChunks(documents, fileName, chunkCursor)
                : importSemanticChunks(documents, fileName, chunkCursor);
    }

    /**
     * 清洗切片路径：状态机过滤校歌曲调行/目录、表格整块保留，直接落库（不经语义二次切分）。
     */
    private int importStatefulChunks(List<Document> documents, String fileName, AtomicInteger chunkCursor) {
        String text = documents.stream()
                .map(Document::getText)
                .filter(t -> t != null && !t.isBlank())
                .collect(Collectors.joining("\n"));
        List<MarkdownChunkSplitter.Chunk> chunks =
                markdownChunkSplitter.split(fileName, chunkCursor.get(), text.lines().toList());
        if (chunks.isEmpty()) {
            log.warn("文档清洗切片后为空，已跳过: {}", fileName);
            return 0;
        }
        List<Document> docs = new ArrayList<>(chunks.size());
        for (MarkdownChunkSplitter.Chunk chunk : chunks) {
            docs.add(new Document(chunk.text(), Map.of(
                    "file_name", fileName,
                    "chunk_index", chunk.chunkIndex(),
                    "section_title", chunk.sectionTitle() == null ? "" : chunk.sectionTitle())));
        }
        vectorStore.add(docs);
        chunkCursor.addAndGet(chunks.size());
        log.info("文档已导入(状态机切片): {} -> {} 个文本块", fileName, chunks.size());
        return chunks.size();
    }

    /**
     * 旧语义切分路径（rag.chunk.filter-enabled=false 时回滚使用）。
     */
    private int importSemanticChunks(List<Document> documents, String fileName, AtomicInteger chunkCursor) {
        List<Document> chunks = semanticTextSplitter.split(documents);
        for (Document chunk : chunks) {
            chunk.getMetadata().put("file_name", fileName);
            chunk.getMetadata().put("chunk_index", chunkCursor.getAndIncrement());
        }
        vectorStore.add(chunks);
        log.info("文档已导入: {} -> {} 个文本块", fileName, chunks.size());
        return chunks.size();
    }

    private boolean isSupported(Path path) {
        String name = path.getFileName().toString();
        int dot = name.lastIndexOf('.');
        return dot > 0 && SUPPORTED_EXTENSIONS.contains(name.substring(dot + 1).toLowerCase());
    }
}
