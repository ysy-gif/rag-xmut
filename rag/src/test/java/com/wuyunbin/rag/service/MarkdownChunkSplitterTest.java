package com.wuyunbin.rag.service;

import java.util.List;

import com.wuyunbin.rag.service.MarkdownChunkSplitter.Chunk;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * MarkdownChunkSplitter 单元测试：覆盖曲调过滤、目录过滤、表格整块保留、优先级与全局序号。
 */
class MarkdownChunkSplitterTest {

    private final MarkdownChunkSplitter splitter = new MarkdownChunkSplitter();

    @Test
    @DisplayName("手册样例：曲调行/目录丢弃，调号拍号与歌词保留，表格整块一个 chunk，空行不入块")
    void splitHandbookSample() {
        List<Chunk> chunks = splitter.split("新生手册.txt", List.of(
                "# 集美大学2025级新生入学须知完整文本",
                "2025",
                "新生手册",
                "",
                "## 集美学校校歌",
                "1=F(或G)",
                "2/4(庄严)",
                "5. 5  5. 5  5.  | 3  | 1  3  | 2  -  |",
                "闽海之滨有我集美乡，山明兮水秀，",
                "6. 6  2. 2  | 5  -  | 5. 6  7. 1  | 2 3  4 5  | 6 6  6 6  | 5 4  3 2",
                "胜地冠南疆。天然位置，惟序与黉，英才乐育，蔚为国光。",
                "",
                "## 目录",
                "本科新生入学须知 01",
                "安全防范知识 01",
                "",
                "# 一、本科新生入学须知",
                "登录方式1:登录网址http://yingxin.jmu.edu.cn/web，填写预报到信息。",
                "",
                "# 收费标准一览表（本科）",
                "| 层次 | 专业 | 学费 (元/年) |",
                "| ---- | ---- | ---- |",
                "| 本科 | 软件工程 | 5460 |",
                "| 本科 | 海上专业 | 5460 |",
                "",
                "# 2025级新生报到点安排表",
                "航海学院联系人：林一雄 13859902516"));

        assertEquals(5, chunks.size());

        // chunk0：封面块，纯数字行 2025 被曲调规则丢弃
        assertEquals("# 集美大学2025级新生入学须知完整文本\n新生手册", chunks.get(0).text());
        assertEquals("集美大学2025级新生入学须知完整文本", chunks.get(0).sectionTitle());

        // chunk1：校歌块，两条纯简谱行被丢弃，调号/拍号元数据行与歌词保留
        assertEquals("""
                ## 集美学校校歌
                1=F(或G)
                2/4(庄严)
                闽海之滨有我集美乡，山明兮水秀，
                胜地冠南疆。天然位置，惟序与黉，英才乐育，蔚为国光。""", chunks.get(1).text());
        assertEquals("集美学校校歌", chunks.get(1).sectionTitle());

        // chunk2：目录标题与条目整体消失，标题行复位后正文正常累积
        assertEquals("""
                # 一、本科新生入学须知
                登录方式1:登录网址http://yingxin.jmu.edu.cn/web，填写预报到信息。""", chunks.get(2).text());
        assertFalse(chunks.stream().anyMatch(c -> c.text().contains("本科新生入学须知 01")));

        // chunk3：表格（标题+表头+分隔行+数据行）完整落在一个 chunk，| ---- | 未被曲调规则误杀
        assertEquals("""
                # 收费标准一览表（本科）
                | 层次 | 专业 | 学费 (元/年) |
                | ---- | ---- | ---- |
                | 本科 | 软件工程 | 5460 |
                | 本科 | 海上专业 | 5460 |""", chunks.get(3).text());
        assertEquals("收费标准一览表（本科）", chunks.get(3).sectionTitle());

        // chunk4：最后一个标题块正常收尾
        assertEquals("""
                # 2025级新生报到点安排表
                航海学院联系人：林一雄 13859902516""", chunks.get(4).text());

        // 全局断言：序号从 0 连续自增、文件名透传、块内无空行
        for (int i = 0; i < chunks.size(); i++) {
            assertEquals(i, chunks.get(i).chunkIndex());
            assertEquals("新生手册.txt", chunks.get(i).fileName());
            assertFalse(chunks.get(i).text().contains("\n\n"), "块内不应有空行: chunk" + i);
        }
    }

    @Test
    @DisplayName("目录兜底：标题外出现点线页码行即进入目录状态，后续行丢弃直到下一个标题")
    void tocEntryFallback() {
        List<Chunk> chunks = splitter.split("t.txt", List.of(
                "# 附录",
                "校园地图 ........ 12",
                "欢迎来到集美大学。",
                "# 联系方式",
                "招生办0592-6181301"));

        assertEquals(2, chunks.size());
        assertEquals("# 附录", chunks.get(0).text());
        assertTrue(chunks.get(1).text().contains("招生办0592-6181301"), "含中文的电话行不应被丢弃");
        assertFalse(chunks.stream().anyMatch(c -> c.text().contains("校园地图")));
        assertFalse(chunks.stream().anyMatch(c -> c.text().contains("欢迎来到集美大学")));
    }

    @Test
    @DisplayName("独立目录行：非标题形式的「目 录」单独成行同样触发目录状态")
    void standaloneTocLine() {
        List<Chunk> chunks = splitter.split("t.txt", List.of(
                "前言内容。",
                "目 录",
                "第一章 报到 01",
                "# 正文开始",
                "正文第一段。"));

        assertEquals(2, chunks.size());
        assertEquals("前言内容。", chunks.get(0).text());
        assertNull(chunks.get(0).sectionTitle());
        assertFalse(chunks.stream().anyMatch(c -> c.text().contains("第一章 报到")));
        assertTrue(chunks.get(1).text().contains("正文第一段。"));
    }

    @Test
    @DisplayName("chunk_index 支持跨文件全局自增：startIndex 作为起始序号")
    void globalChunkIndex() {
        List<Chunk> chunks = splitter.split("a.txt", 100, List.of("# 标题", "内容一段。"));
        assertEquals(1, chunks.size());
        assertEquals(100, chunks.get(0).chunkIndex());
        assertEquals("标题", chunks.get(0).sectionTitle());
    }
}
