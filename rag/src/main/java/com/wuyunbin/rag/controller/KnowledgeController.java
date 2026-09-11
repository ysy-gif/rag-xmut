package com.wuyunbin.rag.controller;

import com.wuyunbin.rag.dto.ImportResult;
import com.wuyunbin.rag.service.KnowledgeService;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestPart;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

/**
 * 知识库接口。
 */
@RestController
@RequestMapping("/api/knowledge")
@RequiredArgsConstructor
@Tag(name = "知识库接口", description = "文档导入相关接口")
public class KnowledgeController {

    private final KnowledgeService knowledgeService;

    @PostMapping(value = "/import", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(summary = "导入单个文档", description = "上传 txt/md/pdf/docx 等文档，切分向量化后写入 Milvus")
    public ImportResult importDocument(@RequestPart("file") MultipartFile file) {
        return knowledgeService.importUpload(file);
    }

    @PostMapping("/import-directory")
    @Operation(summary = "导入目录下全部文档", description = "传入服务器本地目录路径，导入其中所有受支持的文档")
    public ImportResult importDirectory(@RequestParam("directory") String directory) {
        return knowledgeService.importDirectory(directory);
    }
}
