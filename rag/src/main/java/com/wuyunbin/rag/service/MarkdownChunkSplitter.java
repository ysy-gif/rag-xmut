package com.wuyunbin.rag.service;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Markdown 手册切片器：行驱动有限状态机，无 Spring 依赖、无状态、线程安全。
 *
 * <p>逐行判定，按以下优先级处理（顺序不可调整，否则会误判）：
 * <ol>
 *   <li>标题行（^ {0,3}#{1,6}\s+.*$）：flush 当前块 -> 复位 inToc/inTableArea -> 作为新块首行；目录标题 flush 后不入块并置 inToc=true</li>
 *   <li>目录状态（inToc）：后续行全部丢弃，直到下一个标题行复位</li>
 *   <li>表格分隔行：置 inTableArea=true，该行入缓冲</li>
 *   <li>空行：无条件丢弃（含表格区域内空行，保证块内无空行噪声）</li>
 *   <li>表格区域（inTableArea）：所有行无条件入缓冲，直到下一个标题行复位；必须早于目录兜底与曲调行过滤，防止 | ---- | 等行被误杀</li>
 *   <li>目录兜底：点线页码行 / 独立"目录|CONTENTS"行 -> 置 inToc=true</li>
 *   <li>校歌曲调行：仅由 数字 . | - 空格 组成且不含中文/英文字母 -> 丢弃</li>
 *   <li>正常累积：其余行入当前语义块</li>
 * </ol>
 */
public class MarkdownChunkSplitter {

    /** 标题行：行首最多 3 个空格 + 1~6 个 # + 至少一个空白 + 内容 */
    private static final Pattern HEADING_LINE = Pattern.compile("^ {0,3}#{1,6}\\s+.*$");

    /** 标题前缀：用于剥离 # 得到标题文本 */
    private static final Pattern HEADING_PREFIX = Pattern.compile("^ {0,3}#{1,6}\\s+");

    /** 目录标题：标题文本恰为"目录 / 目 录 / CONTENTS" */
    private static final Pattern TOC_TITLE = Pattern.compile("^(目\\s*录|CONTENTS)$", Pattern.CASE_INSENSITIVE);

    /** 独立目录行（非标题形式）：单独成行的"目录 / 目 录 / CONTENTS" */
    private static final Pattern TOC_STANDALONE_LINE = Pattern.compile("^\\s*(目\\s*录|CONTENTS)\\s*$",
            Pattern.CASE_INSENSITIVE);

    /** 目录条目兜底：形如"第一章 新生报到 ........ 3"（点线/省略号/连续空白 + 1~3 位页码结尾） */
    private static final Pattern TOC_ENTRY_FALLBACK = Pattern.compile("^.{2,}?[.…\\s]{3,}\\d{1,3}\\s*$");

    /** 表格分隔行：如 | ---- | ---- | */
    private static final Pattern TABLE_SEPARATOR_ROW = Pattern.compile("^\\s*\\|(\\s*:?-{3,}:?\\s*\\|)+\\s*$");

    /** 含中文或英文字母（命中则不是曲调行） */
    private static final Pattern HAS_TEXT_CHAR = Pattern.compile(".*[\\p{IsHan}A-Za-z].*");

    /** 校歌曲调行：仅由 数字 . | - 空格 组成 */
    private static final Pattern NOTATION_LINE = Pattern.compile("^[0-9.\\|\\-\\s]+$");

    /**
     * 切片结果：text 为块文本（行以 \n 连接），sectionTitle 为所属标题文本（文档开头无标题的块为 null）。
     *
     * @param text        块文本
     * @param fileName    来源文件名
     * @param chunkIndex  块序号（支持跨文件全局自增）
     * @param sectionTitle 所属标题文本，可为 null
     */
    public record Chunk(String text, String fileName, int chunkIndex, String sectionTitle) {
    }

    /**
     * 切片（块序号从 0 开始）。
     *
     * @param fileName 来源文件名
     * @param lines    逐行拆分后的 Markdown 文本
     * @return 切片结果
     */
    public List<Chunk> split(String fileName, List<String> lines) {
        return split(fileName, 0, lines);
    }

    /**
     * 切片，startIndex 作为起始块序号（供跨文件全局自增 chunk_index）。
     *
     * @param fileName   来源文件名
     * @param startIndex 起始块序号
     * @param lines      逐行拆分后的 Markdown 文本
     * @return 切片结果
     */
    public List<Chunk> split(String fileName, int startIndex, List<String> lines) {
        List<Chunk> chunks = new ArrayList<>();
        List<String> buffer = new ArrayList<>();
        String sectionTitle = null;
        boolean inToc = false;
        boolean inTableArea = false;
        int chunkIndex = startIndex;

        for (String line : lines) {
            // 1. 标题行：flush -> 复位状态 -> 新块首行（目录标题除外）
            if (HEADING_LINE.matcher(line).matches()) {
                chunkIndex = flush(buffer, chunks, fileName, chunkIndex, sectionTitle);
                inToc = false;
                inTableArea = false;
                if (isTocHeading(line)) {
                    inToc = true;
                }
                else {
                    sectionTitle = HEADING_PREFIX.matcher(line).replaceFirst("").trim();
                    buffer.add(line);
                }
                continue;
            }
            // 2. 目录状态：全部丢弃
            if (inToc) {
                continue;
            }
            // 3. 表格分隔行：进入表格区域
            if (TABLE_SEPARATOR_ROW.matcher(line).matches()) {
                inTableArea = true;
                buffer.add(line);
                continue;
            }
            // 4. 空行：无条件丢弃
            if (line.isBlank()) {
                continue;
            }
            // 5. 表格区域：无条件保留（必须早于目录兜底/曲调行过滤，防止 | ---- | 等行被误杀）
            if (inTableArea) {
                buffer.add(line);
                continue;
            }
            // 6. 目录兜底：点线页码行 / 独立目录行
            if (TOC_ENTRY_FALLBACK.matcher(line).matches() || TOC_STANDALONE_LINE.matcher(line).matches()) {
                inToc = true;
                continue;
            }
            // 7. 校歌曲调行
            if (isNotationLine(line)) {
                continue;
            }
            // 8. 正常累积
            buffer.add(line);
        }
        flush(buffer, chunks, fileName, chunkIndex, sectionTitle);
        return chunks;
    }

    /** 缓冲非空时落为一个 chunk，返回下一个块序号 */
    private static int flush(List<String> buffer, List<Chunk> chunks, String fileName, int chunkIndex,
            String sectionTitle) {
        if (buffer.isEmpty()) {
            return chunkIndex;
        }
        chunks.add(new Chunk(String.join("\n", buffer), fileName, chunkIndex, sectionTitle));
        buffer.clear();
        return chunkIndex + 1;
    }

    /** 标题文本恰为"目录 / 目 录 / CONTENTS"时判定为目录标题 */
    private static boolean isTocHeading(String line) {
        String title = HEADING_PREFIX.matcher(line).replaceFirst("").trim();
        return TOC_TITLE.matcher(title).matches();
    }

    /** 判定顺序：先看是否含中文/英文（含则不是曲调行），再看是否仅由曲调字符组成 */
    private static boolean isNotationLine(String line) {
        if (HAS_TEXT_CHAR.matcher(line).matches()) {
            return false;
        }
        return NOTATION_LINE.matcher(line).matches();
    }
}
