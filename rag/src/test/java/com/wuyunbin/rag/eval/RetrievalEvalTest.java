package com.wuyunbin.rag.eval;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.wuyunbin.rag.dto.ChatResponse;
import com.wuyunbin.rag.service.ChatService;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.io.ClassPathResource;
import org.springframework.test.util.ReflectionTestUtils;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * RAG 检索参数寻优评测（threshold × topK），三阶段：
 *
 * <p>阶段 A：对两套问题集逐题采集 Milvus 宽召回候选池（POOL_SIZE 条，含分数），
 * 附 embedding 健全性探针与查询一致性校验；<br>
 * 阶段 B：在候选池上离线重放"排序 → 阈值过滤 → 截断"，网格搜索 42 组合，
 * 训练集（freshman2025）选参、验证集（admission2024）同表呈现防过拟合；<br>
 * 阶段 C：用选定参数端到端跑 ChatService 全链路，按关键词覆盖率 / 拒答标记评分。</p>
 *
 * <p>报告覆盖写至 target/rag-eval-report.txt。运行前需 LM Studio 与 Milvus 在线。
 * 运行：mvn test -Dtest=RetrievalEvalTest（不要与其他评测任务并行）。</p>
 */
@SpringBootTest
class RetrievalEvalTest {

    private static final Logger log = LoggerFactory.getLogger(RetrievalEvalTest.class);
    private static final String LS = System.lineSeparator();

    /** 宽召回条数 = rag.retrieve.candidate-top-k（固定，不参与寻优） */
    private static final int POOL_SIZE = 12;

    private static final double[] THRESHOLDS = {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70};
    private static final int[] TOP_KS = {2, 3, 4, 5, 6, 8};

    /** 综合分权重：precise 命中优先于 absent 拒答（校准基调） */
    private static final double W_PRECISE = 1.0;
    private static final double W_SUMMARY = 0.7;
    private static final double W_ABSENT = 0.5;

    /** 端到端答案判定：in_document 题关键词覆盖率阈值 */
    private static final double ANSWER_COVERAGE = 0.6;

    /** 端到端 absent 题拒答标记（答案命中任一即记拒答成功） */
    private static final List<String> REFUSAL_MARKERS = List.of(
            "未找到", "没有找到", "找不到", "未提及", "未在", "知识库中未",
            "没有相关", "未收录", "无法提供", "未检索到", "没有检索到", "知识库中没有", "不包含");

    /** 训练集资源路径与名称 */
    private static final String TRAIN_DS = "freshman2025";
    private static final String TRAIN_RESOURCE = "jmu-freshman-2025-questions.json";
    /** 验证集资源路径与名称 */
    private static final String VALID_DS = "admission2024";
    private static final String VALID_RESOURCE = "jmu-admission-2024-questions.json";

    /** 数据集 → 题号 → 检索判定关键词（top-K 片段任含 1 个即命中；absent 题为空表） */
    private static final Map<String, Map<Integer, List<String>>> RETRIEVAL_KEYWORDS = Map.of(
            TRAIN_DS, Map.ofEntries(
                    Map.entry(1, List.of("9月10日", "7:30")),
                    Map.entry(2, List.of("高崎机场", "厦门北站", "7:00")),
                    Map.entry(3, List.of("12月31日", "落户")),
                    Map.entry(4, List.of("977385535", "1009197588", "470467808")),
                    Map.entry(5, List.of("四位一体", "绿色通道", "奖助贷勤偿补保")),
                    Map.entry(6, List.of("1号窗口", "注册单", "绿色通道")),
                    Map.entry(7, List.of("刷单", "改签", "公检法", "客服")),
                    Map.entry(8, List.of()), Map.entry(9, List.of()), Map.entry(10, List.of())),
            VALID_DS, Map.ofEntries(
                    Map.entry(1, List.of("10390")),
                    Map.entry(2, List.of("160cm")),
                    Map.entry(3, List.of("育德", "6.4万吨")),
                    Map.entry(4, List.of("6181580")),
                    Map.entry(5, List.of("分数优先", "级差", "调剂")),
                    Map.entry(6, List.of("橡胶大王", "华侨旗帜")),
                    Map.entry(7, List.of("130多所", "156个", "500多万")),
                    Map.entry(8, List.of()), Map.entry(9, List.of()), Map.entry(10, List.of())));

    /** 数据集 → 题号 → 答案判定关键词（覆盖率 ≥ 60% 记命中；absent 题为空表走拒答标记） */
    private static final Map<String, Map<Integer, List<String>>> ANSWER_KEYWORDS = Map.of(
            TRAIN_DS, Map.ofEntries(
                    Map.entry(1, List.of("9月10日", "7:30")),
                    Map.entry(2, List.of("高崎机场", "厦门北站")),
                    Map.entry(3, List.of("12月31日")),
                    Map.entry(4, List.of("977385535", "1009197588", "470467808")),
                    Map.entry(5, List.of("四位一体", "绿色通道", "奖助贷勤偿补保")),
                    Map.entry(6, List.of("1号窗口", "注册单", "缴费")),
                    Map.entry(7, List.of("刷单", "改签", "公检法", "客服")),
                    Map.entry(8, List.of()), Map.entry(9, List.of()), Map.entry(10, List.of())),
            VALID_DS, Map.ofEntries(
                    Map.entry(1, List.of("10390")),
                    Map.entry(2, List.of("160cm")),
                    Map.entry(3, List.of("育德", "6.4万吨")),
                    Map.entry(4, List.of("6181580")),
                    Map.entry(5, List.of("分数优先", "级差", "调剂")),
                    Map.entry(6, List.of("橡胶大王", "华侨旗帜", "厦门大学")),
                    Map.entry(7, List.of("130多所", "156个", "500多万")),
                    Map.entry(8, List.of()), Map.entry(9, List.of()), Map.entry(10, List.of())));

    @Autowired
    private VectorStore vectorStore;
    @Autowired
    private ChatService chatService;
    @Autowired
    private EmbeddingModel embeddingModel;

    private final StringBuilder report = new StringBuilder();

    private record Q(int id, String type, String question, boolean inDoc) {}
    private record Cand(double score, String text) {}
    private record Combo(double threshold, int topK) {}
    private record Metrics(double p, double s, double a, double c) {}
    private record Row(double t, int k, Metrics train, Metrics valid) {}
    private record E2e(String dataset, int id, String type, boolean ok, String detail, String answer) {}

    @Test
    void eval() throws Exception {
        report.append("==== RAG 检索参数寻优评测报告 ====").append(LS)
                .append("生成时间: ").append(LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")))
                .append("  候选池 POOL_SIZE=").append(POOL_SIZE).append(LS)
                .append("网格: threshold").append(java.util.Arrays.toString(THRESHOLDS))
                .append(" × topK").append(java.util.Arrays.toString(TOP_KS)).append(LS)
                .append("综合分 = ").append(W_PRECISE).append("*preciseHit + ").append(W_SUMMARY)
                .append("*summaryHit + ").append(W_ABSENT).append("*absentRejection").append(LS).append(LS);
        long t0 = System.currentTimeMillis();
        try {
            preflight();
            Map<String, List<Q>> questions = new LinkedHashMap<>();
            questions.put(TRAIN_DS, loadQuestions(TRAIN_RESOURCE));
            questions.put(VALID_DS, loadQuestions(VALID_RESOURCE));

            // 钉定参数模式：设置 RAG_EVAL_THRESHOLD / RAG_EVAL_TOPK 环境变量后跳过 A/B 阶段，
            // 直接用指定参数跑端到端（用于网格 Top 组合的对照验证）
            String pinT = System.getenv("RAG_EVAL_THRESHOLD");
            String pinK = System.getenv("RAG_EVAL_TOPK");
            Combo best;
            if (pinT != null && pinK != null) {
                best = new Combo(Double.parseDouble(pinT), Integer.parseInt(pinK));
                report.append("[钉定参数模式] threshold=").append(best.threshold())
                        .append(", topK=").append(best.topK())
                        .append("（跳过阶段 A/B，直接端到端）").append(LS).append(LS);
            } else {
                best = phaseB(questions, phaseA(questions));
            }
            phaseC(questions, best);
        } finally {
            flushReport();
        }
        Path reportPath = Paths.get("target", "rag-eval-report.txt").toAbsolutePath();
        log.info("评测完成，耗时 {} s，报告: {}", (System.currentTimeMillis() - t0) / 1000, reportPath);
    }

    // ==================== Preflight ====================

    /**
     * embedding 健全性探针：两段无关文本余弦应显著低于同义文本，
     * 若 > 0.85 说明 LM Studio 静默回退到了错误嵌入模型（历史踩坑），立即失败。
     */
    private void preflight() {
        float[] a = embeddingModel.embed("集美大学的校训是诚毅");
        float[] b = embeddingModel.embed("今天天气晴朗，适合去海边散步");
        double cos = cosine(a, b);
        report.append("[Preflight] 无关文本 embedding 余弦 = ").append(String.format("%.3f", cos))
                .append(cos <= 0.85 ? "（正常）" : "（异常）").append(LS).append(LS);
        assertTrue(cos <= 0.85,
                "无关文本余弦 %.3f > 0.85，嵌入模型疑似回退错误，请核对 LM Studio 加载的 text-embedding-bge-m3".formatted(cos));
    }

    // ==================== 阶段 A：候选池采集 ====================

    private Map<String, Map<Integer, List<Cand>>> phaseA(Map<String, List<Q>> questions) {
        report.append("======== 阶段 A：候选池采集（每题宽召回 ").append(POOL_SIZE).append(" 条）========").append(LS);
        Map<String, Map<Integer, List<Cand>>> pools = new LinkedHashMap<>();
        for (var e : questions.entrySet()) {
            String ds = e.getKey();
            Map<Integer, List<Cand>> m = new LinkedHashMap<>();
            for (Q q : e.getValue()) {
                m.put(q.id(), searchPool(q.question()));
            }
            // 一致性校验：首问重查一次，Top1 应稳定
            Q first = e.getValue().get(0);
            List<Cand> again = searchPool(first.question());
            List<Cand> orig = m.get(first.id());
            boolean consistent = !again.isEmpty() && again.size() == orig.size()
                    && Math.abs(again.get(0).score() - orig.get(0).score()) < 1e-6
                    && again.get(0).text().equals(orig.get(0).text());
            report.append("一致性校验 ").append(ds).append(" 首问重查 Top1: ")
                    .append(consistent ? "一致" : "不一致（仅告警，Milvus HNSW 近似检索允许轻微抖动）").append(LS);

            for (Q q : e.getValue()) {
                String scores = m.get(q.id()).stream()
                        .map(c -> String.format("%.3f", c.score()))
                        .collect(Collectors.joining(" "));
                report.append(pad(ds, 15)).append('#').append(pad(String.valueOf(q.id()), 3))
                        .append('[').append(pad(q.type(), 8)).append("] scores: ").append(scores).append(LS);
            }
            report.append(LS);
            pools.put(ds, m);
        }
        flushReport();
        return pools;
    }

    /** 与 ChatService.retrieve 宽召回语义一致：topK 直接作为 Milvus limit，不设阈值，Java 侧显式降序 */
    private List<Cand> searchPool(String query) {
        List<Document> docs = vectorStore.similaritySearch(
                SearchRequest.builder().query(query).topK(POOL_SIZE).build());
        if (docs == null) {
            return List.of();
        }
        return docs.stream()
                .filter(d -> d.getScore() != null)
                .map(d -> new Cand(d.getScore(), d.getText() == null ? "" : d.getText()))
                .sorted(Comparator.comparingDouble(Cand::score).reversed())
                .toList();
    }

    // ==================== 阶段 B：离线网格搜索 ====================

    private Combo phaseB(Map<String, List<Q>> questions, Map<String, Map<Integer, List<Cand>>> pools) {
        report.append("======== 阶段 B：离线网格搜索（").append(THRESHOLDS.length * TOP_KS.length).append(" 组合）========").append(LS);
        report.append(pad("thr", 7)).append(pad("topK", 6))
                .append('|').append(pad("train(freshman2025) P/S/A/C", 34))
                .append('|').append(pad("valid(admission2024) P/S/A/C", 34)).append(LS);
        List<Row> rows = new ArrayList<>();
        for (double t : THRESHOLDS) {
            for (int k : TOP_KS) {
                Metrics mt = gridMetrics(TRAIN_DS, questions.get(TRAIN_DS), pools.get(TRAIN_DS), t, k);
                Metrics mv = gridMetrics(VALID_DS, questions.get(VALID_DS), pools.get(VALID_DS), t, k);
                rows.add(new Row(t, k, mt, mv));
                report.append(pad(String.format("%.2f", t), 7)).append(pad(String.valueOf(k), 6))
                        .append('|').append(pad(fmtMetrics(mt), 34))
                        .append('|').append(pad(fmtMetrics(mv), 34)).append(LS);
            }
        }
        // 选参：训练集综合分 ↓ → absent 拒答 ↓ → topK ↑小 → threshold ↑严
        rows.sort((x, y) -> {
            int byC = Double.compare(y.train().c(), x.train().c());
            if (byC != 0) return byC;
            int byA = Double.compare(y.train().a(), x.train().a());
            if (byA != 0) return byA;
            int byK = Integer.compare(x.k(), y.k());
            if (byK != 0) return byK;
            return Double.compare(y.t(), x.t());
        });
        report.append(LS).append("训练集综合分 Top5：").append(LS);
        for (int i = 0; i < Math.min(5, rows.size()); i++) {
            Row r = rows.get(i);
            report.append(' ').append(i + 1).append(". thr=").append(r.t()).append(" topK=").append(r.k())
                    .append("  C=").append(String.format("%.3f", r.train().c()))
                    .append(" (train P=").append(String.format("%.2f", r.train().p()))
                    .append(" S=").append(String.format("%.2f", r.train().s()))
                    .append(" A=").append(String.format("%.2f", r.train().a()))
                    .append(" | valid C=").append(String.format("%.3f", r.valid().c()))
                    .append(')').append(LS);
        }
        Row bestRow = rows.get(0);
        Combo best = new Combo(bestRow.t(), bestRow.k());
        report.append(LS).append("选定参数: similarity-threshold=").append(best.threshold())
                .append(", top-k=").append(best.topK())
                .append("（仅依据训练集；验证集同组合 C=").append(String.format("%.3f", bestRow.valid().c()))
                .append(" 用于确认未过拟合）").append(LS).append(LS);
        flushReport();
        return best;
    }

    /** 在候选池上离线重放"排序 → 阈值过滤 → 截断 topK"并计算指标 */
    private Metrics gridMetrics(String ds, List<Q> qs, Map<Integer, List<Cand>> pool, double t, int k) {
        double pHit = 0, pTot = 0, sHit = 0, sTot = 0, aRej = 0, aTot = 0;
        for (Q q : qs) {
            // pool 已按分数降序，filter 保序 → limit(k) 即 top-K
            List<Cand> filtered = pool.get(q.id()).stream()
                    .filter(c -> c.score() >= t)
                    .limit(k)
                    .toList();
            if (q.inDoc()) {
                List<String> kws = RETRIEVAL_KEYWORDS.get(ds).get(q.id());
                boolean hit = filtered.stream().anyMatch(c -> kws.stream().anyMatch(c.text()::contains));
                if ("precise".equals(q.type())) {
                    pTot++;
                    pHit += hit ? 1 : 0;
                } else {
                    sTot++;
                    sHit += hit ? 1 : 0;
                }
            } else {
                aTot++;
                aRej += filtered.isEmpty() ? 1 : 0;
            }
        }
        double p = pTot == 0 ? 0 : pHit / pTot;
        double s = sTot == 0 ? 0 : sHit / sTot;
        double a = aTot == 0 ? 0 : aRej / aTot;
        return new Metrics(p, s, a, W_PRECISE * p + W_SUMMARY * s + W_ABSENT * a);
    }

    private String fmtMetrics(Metrics m) {
        return "P=%.2f S=%.2f A=%.2f C=%.3f".formatted(m.p(), m.s(), m.a(), m.c());
    }

    // ==================== 阶段 C：端到端验证 ====================

    private void phaseC(Map<String, List<Q>> questions, Combo best) throws InterruptedException {
        report.append("======== 阶段 C：端到端验证（threshold=").append(best.threshold())
                .append(", topK=").append(best.topK()).append("）========").append(LS);
        Object origThreshold = ReflectionTestUtils.getField(chatService, "similarityThreshold");
        Object origTopK = ReflectionTestUtils.getField(chatService, "topK");
        try {
            ReflectionTestUtils.setField(chatService, "similarityThreshold", best.threshold());
            ReflectionTestUtils.setField(chatService, "topK", best.topK());

            for (var e : questions.entrySet()) {
                String ds = e.getKey();
                List<E2e> results = new ArrayList<>();
                for (Q q : e.getValue()) {
                    // 全新会话：首轮直用原问题检索，无改写调用；LM Studio 偶发掉线时重试
                    ChatResponse resp = chatWithRetry(q.question());
                    String answer = stripThink(resp.reply());
                    boolean ok;
                    String detail;
                    if (q.inDoc()) {
                        List<String> kws = ANSWER_KEYWORDS.get(ds).get(q.id());
                        long cov = kws.stream().filter(answer::contains).count();
                        int need = (int) Math.ceil(ANSWER_COVERAGE * kws.size());
                        ok = cov >= need;
                        detail = "覆盖 " + cov + "/" + kws.size() + "（需 ≥" + need + "）";
                    } else {
                        String marker = REFUSAL_MARKERS.stream().filter(answer::contains).findFirst().orElse(null);
                        ok = marker != null;
                        detail = marker == null ? "未含拒答标记" : "拒答标记: " + marker;
                    }
                    results.add(new E2e(ds, q.id(), q.type(), ok, detail, answer));
                    log.info("[E2E] {} #{} {} -> {} ({})", ds, q.id(), q.type(), ok ? "PASS" : "FAIL", detail);
                    // 逐题落盘：中断不丢已完成题目的数据
                    report.append(' ').append(pad("#" + q.id(), 4)).append(pad(q.type(), 9))
                            .append(ok ? "PASS" : "FAIL").append("  ").append(detail).append(LS)
                            .append("   答案: ").append(answer.replace(LS, " ")).append(LS);
                    flushReport();
                }

                // 数据集汇总（逐题明细已在上方按题落盘）
                long inDocTotal = results.stream().filter(r -> !"absent".equals(r.type())).count();
                long inDocHit = results.stream().filter(r -> !"absent".equals(r.type()) && r.ok()).count();
                long absentTotal = results.size() - inDocTotal;
                long absentRej = results.stream().filter(r -> "absent".equals(r.type()) && r.ok()).count();
                report.append(LS).append("【").append(ds).append("】文档内命中 ").append(inDocHit).append('/').append(inDocTotal)
                        .append("，absent 拒答 ").append(absentRej).append('/').append(absentTotal).append(LS);
                flushReport();
            }
            report.append(LS).append("======== 最终建议 ========").append(LS)
                    .append("rag.retrieve.similarity-threshold=").append(best.threshold()).append(LS)
                    .append("rag.retrieve.top-k=").append(best.topK()).append(LS);
        } finally {
            // 恢复生产字段，避免污染同上下文的其他测试
            ReflectionTestUtils.setField(chatService, "similarityThreshold", origThreshold);
            ReflectionTestUtils.setField(chatService, "topK", origTopK);
        }
        flushReport();
    }

    /**
     * 对话调用重试：LM Studio 偶发掉线（连接拒绝，历史踩坑）。
     * 每次重试用全新会话 ID，避免半途会话记忆污染；3 次仍失败则抛出。
     */
    private ChatResponse chatWithRetry(String message) throws InterruptedException {
        RuntimeException last = null;
        for (int attempt = 1; attempt <= 3; attempt++) {
            try {
                return chatService.chat(UUID.randomUUID().toString(), message);
            } catch (RuntimeException e) {
                last = e;
                log.warn("对话调用第 {} 次失败，15 s 后重试: {}", attempt, e.getMessage());
                Thread.sleep(15_000);
            }
        }
        throw last;
    }

    // ==================== 工具方法 ====================

    private List<Q> loadQuestions(String classpathLocation) throws IOException {
        try (InputStream in = new ClassPathResource(classpathLocation).getInputStream()) {
            JsonNode root = new ObjectMapper().readTree(in);
            List<Q> list = new ArrayList<>();
            for (JsonNode n : root.path("questions")) {
                list.add(new Q(n.path("id").asInt(), n.path("type").asText(),
                        n.path("question").asText(), n.path("in_document").asBoolean()));
            }
            return list;
        }
    }

    /** 思考型模型可能把推理输出为 <think> 块，评分前剥离 */
    private String stripThink(String s) {
        if (s == null) {
            return "";
        }
        String out = s.replaceAll("(?s)<think>.*?</think>", "").trim();
        int open = out.indexOf("<think>");
        if (open >= 0) {
            out = out.substring(0, open).trim();
        }
        return out;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    /** CJK 按 2 列宽补齐，改善报告表格对齐 */
    private static String pad(String s, int width) {
        int w = s.codePoints().map(c -> c >= 0x2E80 && c <= 0x9FFF || c >= 0xF900 && c <= 0xFAFF ? 2 : 1).sum();
        StringBuilder sb = new StringBuilder(s);
        sb.append(" ".repeat(Math.max(0, width - w)));
        return sb.toString();
    }

    /** 报告分阶段落盘（覆盖写），中断也不丢已完成阶段的数据 */
    private void flushReport() {
        try {
            Path p = Paths.get("target", "rag-eval-report.txt").toAbsolutePath();
            Files.createDirectories(p.getParent());
            Files.writeString(p, report.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            log.warn("评测报告写入失败", e);
        }
    }
}
