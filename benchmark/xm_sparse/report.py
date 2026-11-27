"""Render audited XM/Qwen ablations, keeping observed and missing evidence apart.

This command consumes analysis JSONs; it never runs a model or invents a scan
point. Performance, capacity, independent profiles, and isolated kernel probes
retain their own scopes. The original report can be appended without changing
its pages. All input/output hashes and the preservation check are written to a
sidecar receipt. ReportLab, pypdf, and matplotlib are report-only dependencies.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

ARMS = (
    "baseline",
    "fused",
    "bounds-on-demand",
    "bounds-prefetch",
    "combined",
    "combined-protected",
)
ARM_LABELS = {
    "baseline": "HiSparse 基线",
    "fused": "融合 TMA",
    "bounds-on-demand": "摘要按需",
    "bounds-prefetch": "摘要预取",
    "combined": "融合＋预取",
    "combined-protected": "融合＋保护预取",
}
MIB = 2**20
GIB = 2**30
DEFAULT_FONT = Path("C:/Windows/Fonts/simhei.ttf")
CATEGORY_LABELS = {
    "hisparse_lru_bytes": "HiSparse LRU 槽位／初始化",
    "hisparse_host_mapping_bytes": "请求→DDR KV 映射",
    "request_token_mapping_bytes": "请求 token 映射",
    "hisparse_logical_to_physical_bytes": "逻辑→物理地址映射",
    "hisparse_hot_request_slots_bytes": "请求→hot 槽位表",
    "hisparse_hot_token_ids_bytes": "hot token ID",
    "hisparse_hot_token_locs_bytes": "hot token location",
    "hisparse_selection_output_bytes": "选择结果 device locs",
    "quest_selection_state_bytes": "Quest 选择状态／常量",
    "quest_running_metadata_bytes": "running 页号／token count",
    "quest_prefetch_control_bytes": "预取 epoch／gate／ready／计数",
    "hisparse_batch_state_bytes": "实际 batch 状态",
    "hisparse_diagnostic_counter_bytes": "cache 诊断计数器",
    "compute_kv_descriptors_bytes": "计算端 KV 地址 descriptors",
    "backing_kv_descriptors_bytes": "DDR KV 地址 descriptors",
    "allocator_physical_worklists_bytes": "物理 allocator worklists",
    "allocator_logical_worklists_bytes": "逻辑 allocator worklists",
    "allocator_outer_worklists_bytes": "外层 allocator worklists",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def number(value, *, positive=False):
    require(
        type(value) in (int, float) and math.isfinite(value),
        "Expected a finite numeric measurement",
    )
    require(value > 0 if positive else value >= 0, "Negative/zero measurement")
    return value


def fmt(value, digits=2):
    return "未观测" if value is None else f"{value:,.{digits}f}"


def ranges(values, divisor=1, digits=2):
    values = [value / divisor for value in values if value is not None]
    if not values:
        return "未观测"
    low, high = min(values), max(values)
    return (
        fmt(low, digits) if low == high else f"{fmt(low, digits)}–{fmt(high, digits)}"
    )


def validate_analyses(paths, *, internal_fixture=False):
    result, fingerprints, workloads, commits = [], set(), set(), set()
    for path in paths:
        data = read(path)
        require(data.get("schema_version") == 1, "Unsupported analysis schema")
        require(
            data.get("phase") in ("performance", "capacity"),
            "Expected performance/capacity analysis",
        )
        require(
            data.get("status") in ("analyzed", "no_eligible_samples"),
            "Analysis did not finish",
        )
        require(
            not data.get("internal_layout_fixture") or internal_fixture,
            "Layout fixture cannot be used in a real report",
        )
        source = data.get("source_fingerprint", {}).get("sha256", "")
        workload = data.get("workload_identity", {}).get("sha256", "")
        require(
            re.fullmatch("[0-9a-f]{64}", source) is not None,
            "Missing source fingerprint",
        )
        require(
            re.fullmatch("[0-9a-f]{64}", workload) is not None,
            "Missing frozen workload fingerprint",
        )
        fingerprints.add(source)
        workloads.add(workload)
        for commit in data.get("native_source_commits", []):
            require(
                re.fullmatch("[0-9a-f]{40}", commit) is not None,
                "Missing native source revision",
            )
            commits.add(commit)
        seen = set()
        for row in data.get("rows", []):
            require(row.get("arm") in ARMS, "Unknown experimental arm")
            key = tuple(
                row.get(k)
                for k in (
                    "arm",
                    "input_len",
                    "client_concurrency",
                    "actual_batch",
                    "repeat",
                )
            )
            require(key not in seen, "Duplicate flat sample")
            seen.add(key)
            require(type(row.get("eligible")) is bool, "Eligibility is missing")
            if row["eligible"]:
                number(row.get("mean_tbt_ms"), positive=True)
                number(row.get("stable_tokens_per_s"), positive=True)
                require(
                    type(row.get("actual_batch")) is int and row["actual_batch"] > 0,
                    "Missing actual batch",
                )
            if "client_all_frames_single_token" in row:
                strict = row.get("strict_client_mean_tbt_ms")
                if row["client_all_frames_single_token"] is not True:
                    require(
                        strict is None and row.get("strict_client_p99_tbt_ms") is None,
                        "Multi-token client frames cannot supply strict TBT",
                    )
                elif strict is not None:
                    number(strict, positive=True)
                    number(row.get("strict_client_p99_tbt_ms"), positive=True)
        for sample in data.get("samples", []):
            require(sample.get("arm") in ARMS, "Unknown sample arm")
            client = sample.get("client_timing", {})
            strict = client.get("single_token_arrival_gap_ms")
            if client.get("all_frames_single_token") is not True:
                require(
                    strict is None,
                    "Multi-token frames contain fabricated strict arrival gaps",
                )
            elif strict is not None:
                number(strict.get("mean"), positive=True)
                number(strict.get("p99"), positive=True)
            for boundary in ("memory_before", "memory_after"):
                memory = sample.get(boundary, {})
                number(memory.get("cache_named_gpu_bytes"))
                components = memory.get("cache_components", {})
                backend = memory.get("backend_components", {})
                require(
                    sum(components.values()) == memory["cache_named_gpu_bytes"],
                    "Cache component total differs",
                )
                require(
                    "fused_plan_bytes" not in backend,
                    "Backend plan duplicates the cache ledger plan",
                )
                for value in list(components.values()) + list(backend.values()):
                    number(value)
                require(
                    memory.get("torch_named_total_including_observed_backend_bytes")
                    == memory["cache_named_gpu_bytes"] + sum(backend.values()),
                    "Combined named total differs",
                )
                if (
                    memory.get("allocation_accounting_schema")
                    == "xm_qwen_named_allocations_v2"
                ):
                    for owner in ("cache", "backend"):
                        grouped = defaultdict(int)
                        ids = set()
                        for record in memory.get(owner + "_allocation_groups", []):
                            require(
                                type(record.get("allocation_group")) is int
                                and record["allocation_group"] not in ids,
                                "Repeated or missing allocation group",
                            )
                            require(
                                record.get("alias_names"),
                                "Allocation owner names missing",
                            )
                            ids.add(record["allocation_group"])
                            number(record.get("storage_bytes"))
                            grouped[record["category"]] += record["storage_bytes"]
                        require(
                            dict(grouped) == memory[owner + "_components"],
                            "Named owner records differ from component totals",
                        )
        for row in data.get("same_actual_batch_comparisons", []):
            require(
                row.get("baseline") in ARMS and row.get("changed") in ARMS,
                "Unknown comparison arm",
            )
            require(
                type(row.get("paired_repeats")) is int and row["paired_repeats"] > 0,
                "Comparison repeats are missing",
            )
            number(row.get("baseline_gap_ms", {}).get("mean"), positive=True)
            number(row.get("changed_gap_ms", {}).get("mean"), positive=True)
            effect = row.get("launch_gap_reduction_percent", {})
            require(
                type(effect.get("mean")) in (int, float)
                and math.isfinite(effect["mean"]),
                "Comparison effect is not finite",
            )
            interval = effect.get("confidence_interval_95")
            if interval is not None:
                require(
                    row["paired_repeats"] >= 2
                    and len(interval) == 2
                    and all(
                        type(v) in (int, float) and math.isfinite(v) for v in interval
                    )
                    and interval[0] <= effect["mean"] <= interval[1],
                    "Invalid repeat confidence interval",
                )
        result.append((Path(path).resolve(), data))
    require(
        len(fingerprints) == 1 and len(workloads) == 1,
        "Report cannot combine different implementation/workload fingerprints",
    )
    require(len(commits) <= 1, "Report cannot combine native source revisions")
    return result


def validate_profile(spec):
    arm, separator, declared = spec.partition("=")
    require(separator and arm in ARMS, "Profile must use ARM=PATH")
    path = Path(declared).resolve(strict=True)
    data = read(path)
    if data.get("status") == "analyzed":
        require(data.get("schema_version") == 1, "Unsupported profile schema")
        require(
            re.fullmatch("[0-9a-f]{64}", data.get("sqlite", {}).get("sha256", "")),
            "Independent profile has no SQLite fingerprint",
        )
        number(data.get("actual_batch_size"), positive=True)
        number(data.get("observed_decode_steps"), positive=True)
        for step in data.get("per_step", []):
            number(step.get("total_path_span_us"))
            for stage in step.get("stages", {}).values():
                for key in (
                    "span_us",
                    "kernel_active_union_us",
                    "internal_idle_gap_us",
                ):
                    number(stage.get(key))
    return arm, path, data


def draw_plots(analyses, profiles, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    images = {}
    colors = dict(
        zip(ARMS, ("#59636e", "#2875a2", "#c19b51", "#47867c", "#855f9b", "#b05c58"))
    )
    for index, (_, analysis) in enumerate(analyses):
        comparisons = analysis.get("same_actual_batch_comparisons", [])
        if comparisons:
            height = max(3.0, min(10.0, len(comparisons) * 0.4))
            figure, axis = plt.subplots(figsize=(9, height))
            for position, row in enumerate(comparisons):
                stats = row["launch_gap_reduction_percent"]
                value, interval = stats["mean"], stats.get("confidence_interval_95")
                axis.barh(
                    position,
                    value,
                    color="#2875a2" if value >= 0 else "#b05c58",
                    alpha=0.8,
                )
                if interval:
                    axis.errorbar(
                        value,
                        position,
                        xerr=[[value - interval[0]], [interval[1] - value]],
                        color="#222222",
                        capsize=3,
                        fmt="none",
                    )
            labels = [
                f"{r['baseline']} -> {r['changed']} | L={r['input_len']}, B={r['actual_batch_size']}, n={r['paired_repeats']}"
                for r in comparisons
            ]
            axis.set_yticks(range(len(labels)), labels, fontsize=8)
            axis.axvline(0, color="#444444", linewidth=0.8)
            axis.set_xlabel("Matched scheduler decode-interval reduction (%)")
            axis.set_title(
                "Same requests and decode ordinals; paired-repeat 95% t intervals"
            )
            axis.grid(axis="x", alpha=0.18)
            figure.tight_layout()
            base = directory / f"same-batch-{index + 1:02d}"
            for extension in ("png", "svg"):
                figure.savefig(base.with_suffix("." + extension), dpi=180)
            plt.close(figure)
            images[(index, "same_batch")] = base.with_suffix(".png")
        if analysis["phase"] == "capacity":
            points = analysis.get("observed_stable_capacity_points", [])
            if points:
                figure, axis = plt.subplots(figsize=(8, 4.4))
                grouped = defaultdict(list)
                for point in points:
                    grouped[(point["arm"], point["input_len"])].append(point)
                for (arm, length), rows in sorted(grouped.items()):
                    rows.sort(
                        key=lambda row: (
                            row["actual_batch_size"],
                            row["client_concurrency"],
                        )
                    )
                    axis.plot(
                        [row["actual_batch_size"] for row in rows],
                        [row["stable_tokens_per_second"]["mean"] for row in rows],
                        "o-",
                        color=colors[arm],
                        label=f"{arm}, L={length}",
                    )
                axis.set_xlabel("Observed stable actual decode batch (requests)")
                axis.set_ylabel("Stable scheduler-span output rate (tokens/s)")
                axis.set_title(
                    f"Configured cache budget: {fmt(analysis.get('cache_hbm_budget_bytes', 0) / GIB)} GiB"
                )
                axis.legend(fontsize=7.5)
                axis.grid(alpha=0.2)
                figure.tight_layout()
                base = directory / f"capacity-{index + 1:02d}"
                for extension in ("png", "svg"):
                    figure.savefig(base.with_suffix("." + extension), dpi=180)
                plt.close(figure)
                images[(index, "capacity")] = base.with_suffix(".png")
    passed = [
        (arm, data) for arm, _, data in profiles if data.get("status") == "analyzed"
    ]
    if passed:
        figure, axis = plt.subplots(figsize=(8, 4.3))
        labels = (
            "selection",
            "cache_plan_or_swap",
            "packing",
            "attention_including_merge",
        )
        left = [0.0] * len(passed)
        for label, color in zip(labels, ("#c19b51", "#59636e", "#47867c", "#2875a2")):
            values = [
                data["stage_span_us"].get(label, {}).get("mean", 0)
                for _, data in passed
            ]
            axis.barh(range(len(passed)), values, left=left, label=label, color=color)
            left = [a + b for a, b in zip(left, values)]
        axis.set_yticks(
            range(len(passed)),
            [
                f"{arm}, B={data['actual_batch_size']}, layer={data['layer_index']}"
                for arm, data in passed
            ],
            fontsize=8,
        )
        axis.set_xlabel("One-layer contiguous path span (microseconds)")
        axis.set_title("Independent decode profile: spans include internal idle gaps")
        axis.legend(fontsize=7.5)
        figure.tight_layout()
        base = directory / "attention-path-breakdown"
        for extension in ("png", "svg"):
            figure.savefig(base.with_suffix("." + extension), dpi=180)
        plt.close(figure)
        images["profile"] = base.with_suffix(".png")
    return images


def render_report(
    analyses,
    profiles,
    evidence,
    output,
    *,
    font_path,
    internal_fixture=False,
    images=None,
):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        Image,
        LongTable,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        TableStyle,
    )

    pdfmetrics.registerFont(TTFont("ReportChinese", str(font_path)))
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="ReportChinese",
            fontSize=21,
            leading=29,
            textColor=colors.HexColor("#16374c"),
            spaceAfter=13,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="ReportChinese",
            fontSize=14,
            leading=20,
            textColor=colors.HexColor("#16374c"),
            spaceAfter=10,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="ReportChinese",
            fontSize=9.4,
            leading=15,
            spaceAfter=7,
            wordWrap="CJK",
        ),
        "small": ParagraphStyle(
            "small",
            fontName="ReportChinese",
            fontSize=7.5,
            leading=10.8,
            spaceAfter=5,
            wordWrap="CJK",
        ),
        "cell": ParagraphStyle(
            "cell",
            fontName="ReportChinese",
            fontSize=7.5,
            leading=10.5,
            wordWrap="CJK",
            alignment=TA_LEFT,
        ),
    }
    width = A4[0] - 88
    story, images = [], images or {}

    def paragraph(text, style="body"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def add(text, style="body"):
        story.append(paragraph(text, style))

    def section(title):
        story.append(PageBreak())
        add(title, "heading")

    def table(headers, rows, weights=None):
        weights = weights or [1] * len(headers)
        widths = [width * weight / sum(weights) for weight in weights]
        values = [
            [paragraph(value, "cell") for value in row] for row in [headers] + rows
        ]
        item = LongTable(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef2")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f7f8")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#a4b3bc")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(item)
        story.append(Spacer(1, 9))

    def figure(path, caption):
        item = Image(str(path))
        scale = min(width / item.imageWidth, 305 / item.imageHeight)
        item.drawWidth, item.drawHeight = (
            item.imageWidth * scale,
            item.imageHeight * scale,
        )
        story.append(item)
        add(caption, "small")

    add("Qwen＋Quest：XM-Sparse 软件机制适配实验", "title")
    add("单 H800 PCIe · CPU DDR 全量 KV backing · NoPD", "heading")
    add(
        "本报告将融合检索与 Attention、Quest 历史 min/max 摘要卸载与 GPU 发起预取分开归因。实验沿用 1＋0 的 Qwen＋Quest＋HiSparse；第二项是经授权的 Qwen 摘要适配，不是 DeepSeek/GLM 的训练型 K-index 复现。"
    )
    if internal_fixture:
        add(
            "内部排版验证：下列数值为布局 fixture，不是实验结果，禁止用于汇报。",
            "heading",
        )
    add("结果判断", "heading")
    comparisons = [
        row
        for _, data in analyses
        if data["phase"] == "performance"
        for row in data.get("same_actual_batch_comparisons", [])
        if row["baseline"] == "baseline" and row["changed"] == "fused"
    ]
    if comparisons:
        for row in comparisons[:8]:
            effect = row["launch_gap_reduction_percent"]
            word = "下降" if effect["mean"] >= 0 else "增加"
            add(
                f"同请求、同 decode 序号、实际 batch={row['actual_batch_size']}、输入 {row['input_len']:,} tokens："
                f"融合后稳定 decode 调度间隔 {word} {abs(effect['mean']):.2f}%"
                f"（{row['baseline_gap_ms']['mean']:.2f} → {row['changed_gap_ms']['mean']:.2f} ms；"
                f"配对重复 n={row['paired_repeats']}）。该间隔不是 HTTP token 到达 TBT。"
            )
    else:
        add(
            "尚无满足稳定窗口与请求匹配要求的 baseline→fused 数据；不能形成融合的端到端收益结论。"
        )
    capacity = [
        (data, row)
        for _, data in analyses
        if data["phase"] == "capacity"
        for row in data.get("observed_stable_capacity_points", [])
    ]
    if capacity:
        add(
            "容量扫描已经取得稳定实际 batch 的观测点。表 2 将 fused→combined-protected 作为主对照，"
            "两者使用同一融合 Attention workspace；观测上界不等于硬件最大容量或最优性价比。"
        )
    else:
        add("尚未取得同 cache HBM 预算的容量扫描；摘要卸载的并发与吞吐收益保留待验证。")
    add("证据边界", "heading")
    add(
        "融合内核使用真实 TMA global-to-shared 指令；当前 Attention 为标量 FP32 online softmax＋Split-K，"
        "尚非论文的 Tensor Core 优化内核。预取通过 GPU 发起 device-graph memcpy；正确性与物理 CE 利用率、"
        "带宽、重叠程度是不同证据。缺失 child memcpy 活动、受限 GPU counters 都不能被写成 CE 性能结论。"
    )

    section("1　实验设计、六条路径与指标")
    table(
        ["实验臂", "历史 Quest 摘要", "decode 检索／Attention"],
        [
            ["baseline", "GPU 常驻 min/max", "HiSparse swap-in → packing → FlashInfer"],
            [
                "fused",
                "GPU 常驻 min/max",
                "plan-only → TMA 检索＋Attention；保留 miss refill",
            ],
            [
                "bounds-on-demand",
                "CPU DDR；GPU 分层 stage",
                "当前层按需 GPU 发起 copy；独立 Attention",
            ],
            [
                "bounds-prefetch",
                "CPU DDR；GPU 分层 stage",
                "预取后续层；独立 Attention",
            ],
            ["combined", "CPU DDR；GPU 分层 stage", "TMA 融合＋摘要预取"],
            [
                "combined-protected",
                "CPU DDR；GPU 分层 stage",
                "融合＋保护型预取；容量比较的主要改动臂",
            ],
        ],
        [1.1, 1.25, 2.65],
    )
    add(
        "摘要机制的实际范围：全历史 min/max 卸载到 pinned CPU DDR，复用固定两个 HBM layer stages。"
        "当前 child graph copy 覆盖全部预分配的 R/P 行页范围，不是仅复制活动请求/页。"
        "实现了 on-demand、层间 lookahead 与 Attention gate；保护检查点是公开 CUDA 适配，"
        "已经在途的数据传输可继续完成。"
    )
    add(
        "尚未实现论文按预算逐页 offload、逐步恢复、测量 transfer-budget 的准入，"
        "也未实现跨 decode step 在末尾提前预取下一步 early-layer。单卡没有跨 rank MoE collective 流量；"
        "因此保护结果只覆盖当前单卡 Attention 路径。表 1、表 2 与 A.2 借鉴论文的问题分解方式，"
        "不代表完整作者软件复现。"
    )
    table(
        ["指标", "本报告的定义与限制"],
        [
            ["客户端并发 C", "HTTP 客户端同时在途请求数；不等于 GPU 实际 batch。"],
            [
                "实际 batch B",
                "从 scheduler trace 的恒定请求集合窗口取得真实 decode 请求数。",
            ],
            [
                "稳定 decode 间隔",
                "mean_tbt_ms 字段是相邻 scheduler decode launch 的时间间隔，包含 scheduler 间隙；不是严格客户端逐 token 到达 TBT。",
            ],
            [
                "严格 HTTP TBT",
                "仅全部 token-positive HTTP 帧均为单 token 时观测相邻 token 到达间隔。覆盖完整请求的 decode（排除首 token），包含非稳定 batch 初尾；不从多 token 帧的摊销间隔恢复内部时间戳。",
            ],
            [
                "稳定输出率",
                "B×1000／稳定间隔 ms；覆盖恒定请求集合窗口，不是完整请求吞吐。",
            ],
            [
                "TTFT／请求 TPOT",
                "原始 HTTP 结果的 mean TTFT、每请求平均 TPOT；保留 prefill、准入、排队影响。",
            ],
            [
                "误差／重复",
                "同 batch 对照按相同冻结输入行集合及每请求 decode 序号匹配；95% t 区间基于配对重复均值。单次重复不给区间。",
            ],
        ],
        [1, 3.7],
    )
    add(
        "扣除 prefill 时间不能消除准入与排队差异。NoPD 用于排除分离 prefill/decode 通信路径；"
        "模型、固定 token workload、sampling、Graph 与 hot/backing 配置应随原始 manifest 一起审查。"
    )
    source = analyses[0][1]
    add("实现指纹 SHA256：" + source["source_fingerprint"]["sha256"], "small")
    add("固定 workload SHA256：" + source["workload_identity"]["sha256"], "small")
    add(
        "真实模型运行 git commit："
        + (", ".join(source["native_source_commits"]) or "未记录；仅存在未完成样本"),
        "small",
    )

    section("2　表 1：同实际 batch 的软件机制效果（Table 3 式归因）")
    add(
        "下降率=(原路径间隔−新路径间隔)／原路径间隔；负值表示回退。"
        "只纳入 analysis 中满足稳定步数、恒定请求集合与 decode 序号配对的样本。"
    )
    all_pairs = []
    for index, (_, data) in enumerate(analyses):
        for row in data.get("same_actual_batch_comparisons", []):
            effect = row["launch_gap_reduction_percent"]
            ci = effect.get("confidence_interval_95")
            all_pairs.append(
                [
                    f"{row['baseline']}\n→ {row['changed']}\n{data['phase']}",
                    f"{row['input_len']:,}\nC={row['client_concurrency']}, B={row['actual_batch_size']}",
                    str(row["paired_repeats"]),
                    fmt(row["baseline_gap_ms"]["mean"]),
                    fmt(row["changed_gap_ms"]["mean"]),
                    f"{effect['mean']:.2f}%",
                    "未估计" if ci is None else f"[{ci[0]:.2f}, {ci[1]:.2f}]%",
                ]
            )
    if all_pairs:
        table(
            [
                "对照",
                "输入／C／B",
                "重复",
                "原路径\nms",
                "新路径\nms",
                "间隔下降",
                "95%区间",
            ],
            all_pairs,
            [1.55, 1.45, 0.5, 0.7, 0.7, 0.8, 1],
        )
    else:
        add("未取得合格配对样本。")
    for index, (_, _) in enumerate(analyses):
        if (index, "same_batch") in images:
            figure(
                images[(index, "same_batch")],
                "图 1　同请求匹配的 decode 间隔变化；误差条为配对重复的 95% t 区间。",
            )

    section("3　表 2：同 cache HBM 预算的容量与吞吐")
    add(
        "主比较为 fused→combined-protected：以此分离历史摘要卸载的容量效果与移除 FlashInfer workspace 的效果。"
        "配置的 cache HBM budget 不是模型、activation、Graph、驱动的总显存预算。"
    )
    if capacity:
        for _, data in analyses:
            if data["phase"] != "capacity":
                continue
            budget = data.get("cache_hbm_budget_bytes")
            add(
                "此轮配置 cache HBM budget："
                + (
                    fmt(budget / GIB) + " GiB"
                    if budget
                    else "未记录，不能宣称同预算收益"
                )
            )
            rows = data.get("observed_stable_capacity_points", [])
            table(
                [
                    "臂",
                    "输入 tokens",
                    "客户端 C",
                    "实际 B",
                    "重复",
                    "稳定间隔 ms",
                    "稳定 tokens/s",
                ],
                [
                    [
                        r["arm"],
                        f"{r['input_len']:,}",
                        r["client_concurrency"],
                        r["actual_batch_size"],
                        r["observed_repeats"],
                        fmt(r["mean_launch_gap_ms"]["mean"]),
                        fmt(r["stable_tokens_per_second"]["mean"], 1),
                    ]
                    for r in rows
                ],
                [1.65, 1, 0.7, 0.65, 0.55, 1, 1.05],
            )
            maxima = {}
            for row in rows:
                key = row["arm"], row["input_len"]
                if (
                    key not in maxima
                    or row["actual_batch_size"] > maxima[key]["actual_batch_size"]
                ):
                    maxima[key] = row
            for length in sorted({r["input_len"] for r in rows}):
                left = maxima.get(("fused", length))
                right = maxima.get(("combined-protected", length))
                if left and right and budget:
                    add(
                        f"输入 {length:,} tokens：当前扫描的稳定 batch 观测上界为 "
                        f"fused {left['actual_batch_size']} → combined-protected {right['actual_batch_size']}；"
                        f"对应稳定输出率 {left['stable_tokens_per_second']['mean']:.1f} → "
                        f"{right['stable_tokens_per_second']['mean']:.1f} tokens/s。"
                        "此句比较的是各自已观测容量点，不是同 batch 单步加速。"
                    )
        for index, (_, _) in enumerate(analyses):
            if (index, "capacity") in images:
                figure(
                    images[(index, "capacity")],
                    "图 2　实际稳定 batch 与输出率；连接线仅连接实测扫描点，不代表未测容量。",
                )
    else:
        add(
            "尚无容量扫描结果。保留待验证，不从单 batch 的内存节省外推出最大并发或吞吐。"
        )
    add("失败或不合格样本", "heading")
    excluded = []
    for _, data in analyses:
        excluded.extend(
            [e.get("case", "未知"), e.get("reason", "未说明")]
            for e in data.get("excluded_cases", [])
        )
        excluded.extend(
            [
                f"{r['arm']} / L={r['input_len']}, C={r['client_concurrency']}, B={r['actual_batch']}, r={r['repeat']}",
                f"稳定间隔数 {r['measured_intervals']}；不足 eligibility 门槛",
            ]
            for r in data.get("rows", [])
            if not r["eligible"]
        )
    if excluded:
        table(["样本", "排除原因"], excluded, [2, 3])
    else:
        add("分析文件中未记录排除样本。未扫描的配置不等于已经通过。")

    section("4　完整六臂结果与重复不确定性")
    for _, data in analyses:
        grouped = defaultdict(list)
        for row in data.get("rows", []):
            if row["eligible"]:
                grouped[
                    (
                        row["arm"],
                        row["input_len"],
                        row["client_concurrency"],
                        row["actual_batch"],
                    )
                ].append(row)
        add(
            "阶段："
            + ("同 batch 性能" if data["phase"] == "performance" else "同预算容量"),
            "heading",
        )
        if grouped:
            rows = []
            for (arm, length, concurrency, batch), samples in sorted(grouped.items()):
                gaps = [s["mean_tbt_ms"] for s in samples]
                rates = [s["stable_tokens_per_s"] for s in samples]
                rows.append(
                    [
                        arm,
                        f"{length:,}\nC={concurrency}, B={batch}",
                        len(samples),
                        fmt(statistics.mean(gaps)),
                        ranges(gaps),
                        fmt(statistics.mean(rates), 1),
                        fmt(statistics.mean(s["mean_tpot_ms"] for s in samples)),
                        fmt(statistics.mean(s["ttft_ms"] for s in samples)),
                    ]
                )
            table(
                [
                    "臂",
                    "输入／C／B",
                    "重复",
                    "间隔均值\nms",
                    "重复间范围\nms",
                    "稳定\ntokens/s",
                    "请求\nTPOT ms",
                    "TTFT\nms",
                ],
                rows,
                [1.5, 1.25, 0.5, 0.8, 1.1, 0.8, 0.75, 0.8],
            )
        else:
            add("无合格稳定样本。")
        add("完整请求 HTTP 到达时延（独立口径）", "heading")
        client_groups = defaultdict(list)
        for sample in data.get("samples", []):
            client_groups[
                (sample["arm"], sample["input_len"], sample["client_concurrency"])
            ].append(sample)
        client_rows = []
        for (arm, length, concurrency), samples in sorted(client_groups.items()):
            require(
                len({sample["repeat"] for sample in samples}) == len(samples),
                "A client request sample would be counted more than once",
            )
            stricts, frame_means = [], []
            for sample in samples:
                client = sample.get("client_timing", {})
                strict = (
                    client.get("single_token_arrival_gap_ms")
                    if client.get("all_frames_single_token") is True
                    else None
                )
                if strict is not None:
                    stricts.append(strict)
                frame = client.get("raw_token_positive_frame_gap_ms", {}).get("mean")
                if frame is not None:
                    frame_means.append(frame)
            unavailable = (
                "不可严格定义"
                if any(
                    sample.get("client_timing", {}).get("all_frames_single_token")
                    is False
                    for sample in samples
                )
                else "未观测"
            )
            client_rows.append(
                [
                    arm,
                    f"{length:,}\nC={concurrency}",
                    f"{len(stricts)}/{len(samples)}",
                    sum(stat["samples"] for stat in stricts) if stricts else "未观测",
                    (
                        fmt(statistics.mean(stat["mean"] for stat in stricts))
                        if stricts
                        else unavailable
                    ),
                    ranges([stat["p99"] for stat in stricts]),
                    fmt(statistics.mean(frame_means)) if frame_means else "未观测",
                ]
            )
        if client_rows:
            table(
                [
                    "臂",
                    "输入／C",
                    "strict\n重复数",
                    "单 token\n间隔数",
                    "HTTP TBT\n重复均值 ms",
                    "每次 P99\n范围 ms",
                    "HTTP frame\n间隔均值 ms",
                ],
                client_rows,
                [1.45, 1.15, 0.7, 0.8, 1.1, 1.1, 1.1],
            )
        else:
            add("未提供 HTTP 帧证据。")
        add(
            "HTTP 每个原始样本按 arm／输入／客户端 C／repeat 只计一次，不按多个实际 batch 窗口重复计数。"
            "严格 TBT 列只汇总 strict 重复；P99 显示各重复实测 P99 的范围，不是合并分布的 P99。"
            "HTTP frame gap 为多 token 帧也可观测的原始帧间隔，不能替代单 token TBT。",
            "small",
        )
    add(
        "重复间范围展示实测分散程度，不是置信区间。请求 TPOT 与 TTFT 来自完整请求，"
        "可受预填充和排队影响；不能用这些字段替代同请求匹配的稳定 decode 间隔。"
    )

    section("5　具名 HBM／DDR 分解与 384 MiB workspace 归因")
    add(
        "所有 MiB 按 2^20 bytes 换算。每个 allocation_group 的别名只计一次；"
        "plan 同时存在于 cache ledger 和 backend inventory 时只计 cache ledger。"
        "backend workspace 只显示独立、已经观测的分配；未观测不写作零。"
    )
    all_samples = [
        dict(
            sample,
            _report_phase=data["phase"],
            _report_budget=data.get("cache_hbm_budget_bytes"),
        )
        for _, data in analyses
        for sample in data.get("samples", [])
    ]
    grouped_memory = defaultdict(list)
    for sample in all_samples:
        grouped_memory[
            (
                sample["_report_phase"],
                sample["_report_budget"],
                sample["input_len"],
                sample["client_concurrency"],
            )
        ].append(sample)
    metrics = [
        ("GPU 历史 min/max 摘要", "cache", "quest_resident_summary_bytes"),
        ("GPU 分层摘要 stage", "cache", "quest_layer_staging_bytes"),
        ("GPU running bounds", "cache", "quest_running_bounds_bytes"),
        ("GPU 物理 hot KV", "cache", "physical_compute_kv_bytes"),
        ("GPU plan 数组", "cache", "fused_plan_bytes"),
        ("GPU 其他 cache 管理（合并）", "cache", "other_named_gpu_bytes"),
        ("Attention split partial", "backend", "fused_split_workspace_bytes"),
        ("Attention output", "backend", "fused_output_bytes"),
        ("FlashInfer workspace", "backend", "separate_attention_workspace_bytes"),
        (
            "具名 cache＋backend 合计",
            "direct",
            "torch_named_total_including_observed_backend_bytes",
        ),
        ("CPU DDR 全量 KV", "direct", "cpu_full_kv_backing_bytes"),
        ("CPU DDR 历史摘要", "direct", "cpu_quest_summary_backing_bytes"),
    ]
    for (phase, budget, length, concurrency), samples in sorted(
        grouped_memory.items(), key=lambda item: str(item[0])
    ):
        add(
            f"阶段 {phase}，输入 {length:,} tokens，客户端 C={concurrency}；"
            f"cache budget {fmt(budget / GIB) + ' GiB' if budget else '未记录'}；"
            "每臂为实测 idle 快照范围（MiB）",
            "small",
        )
        arms = [arm for arm in ARMS if any(s["arm"] == arm for s in samples)]
        rows = []
        known = {key for _, owner, key in metrics if owner == "cache"}
        extra = sorted(
            {
                key
                for sample in samples
                for key in sample["memory_after"]["cache_components"]
            }
            - known
        )
        local_metrics = metrics + [
            ("GPU " + CATEGORY_LABELS.get(key, key), "cache", key) for key in extra
        ]
        for label, owner, key in local_metrics:
            row = [label]
            for arm in arms:
                values = []
                for sample in samples:
                    if sample["arm"] != arm:
                        continue
                    memory = sample["memory_after"]
                    if owner == "direct":
                        value = memory.get(key)
                    elif owner == "cache":
                        value = memory["cache_components"].get(key, 0)
                    elif memory["backend_inventory_observed"]:
                        value = memory["backend_components"].get(key, 0)
                    else:
                        value = None
                    values.append(value)
                row.append(ranges(values, MIB))
            rows.append(row)
        table(["分配归属"] + arms, rows, [1.75] + [1] * len(arms))
    if not grouped_memory:
        add("尚无真实模型具名分配快照。")
    add(
        "融合可能移除原路径实际保留的 384 MiB FlashInfer workspace，但这是 Attention 后端的工作区，"
        "不是全量 KV，也不是 Quest 历史摘要。只有快照确实观察到该分配时，才把这个大小作为实测。"
        "摘要卸载独立减少历史 min/max 的 GPU 常驻空间，并引入 GPU stage 与 CPU backing。"
    )
    allocator_groups = defaultdict(list)
    for sample in all_samples:
        before, after = sample["memory_before"].get("allocator"), sample[
            "memory_after"
        ].get("allocator")
        if before is not None or after is not None:
            allocator_groups[
                (
                    sample["_report_phase"],
                    sample["arm"],
                    sample["input_len"],
                    sample["client_concurrency"],
                )
            ].append(sample)
    allocator_rows = []
    for (phase, arm, length, concurrency), samples in sorted(allocator_groups.items()):
        fields = []
        for boundary, key in (
            ("memory_before", "allocated_bytes"),
            ("memory_after", "allocated_bytes"),
            ("memory_before", "reserved_bytes"),
            ("memory_after", "reserved_bytes"),
            ("memory_after", "peak_allocated_bytes"),
            ("memory_after", "peak_reserved_bytes"),
        ):
            fields.append(
                ranges(
                    [
                        (sample[boundary].get("allocator") or {}).get(key)
                        for sample in samples
                    ],
                    MIB,
                    1,
                )
            )
        allocator_rows.append(
            [f"{arm}\n{phase}", f"L={length:,}\nC={concurrency}"] + fields
        )
    if allocator_rows:
        add("Allocator 观测范围（MiB）", "heading")
        table(
            [
                "臂／阶段",
                "样本",
                "allocated\n前 idle",
                "allocated\n后 idle",
                "reserved\n前 idle",
                "reserved\n后 idle",
                "累计 peak\nallocated",
                "累计 peak\nreserved",
            ],
            allocator_rows,
            [1.55, 1.15, 1, 1, 1, 1, 1, 1],
        )
        add(
            "peak 是该 allocator 观测时的累计峰值，未按 prefill／decode 或 Graph 单独 reset；"
            "不将它归因为特定阶段。范围来自重复实测快照，原始 bytes 与 scope 保存在 analysis.json。",
            "small",
        )
    else:
        add("allocated/reserved 未提供可解析原始观测，保留未观测。")
    add(
        "此轮 inventory 不含模型权重、activation、驱动/私有 Graph pool、完整 NVML 采样，"
        "也未由 idle 前后快照证明 prefill／decode 临时峰值。因此不得把具名合计写成进程总 HBM 或峰值。"
    )
    add(
        "v2 inventory 的逐 storage allocation_group、alias_names 与 category 保存在 analysis 与报告随附"
        " named-allocations.csv；以此审查 LRU、mapping、allocator worklists 和 plan 的独立归属。",
        "small",
    )

    section("6　Appendix A.2 式独立 decode 诊断")
    add(
        "按一个选定层的首个 Quest score 到末个 Attention／merge 的连续时间段分解。"
        "分段 span 含内部 idle gap；kernel 活跃区间取并集，不能把跨流 kernel 耗时简单相加当作关键路径。"
        "首个 score 前的预取依赖等待位于本分解范围之外。"
    )
    passed = [
        (arm, data) for arm, _, data in profiles if data.get("status") == "analyzed"
    ]
    if passed:
        stages = (
            "selection",
            "cache_plan_or_swap",
            "packing",
            "attention_including_merge",
        )
        for arm, profile in passed:
            add(
                f"{arm}：实际 B={profile['actual_batch_size']}，层 {profile['layer_index']}，"
                f"独立 decode steps={profile['observed_decode_steps']}",
                "heading",
            )
            rows = []
            for stage in stages:
                stats = profile.get("stage_span_us", {}).get(stage)
                if stats is None:
                    rows.append([stage, "未出现此分段", "—", "—"])
                    continue
                active = [
                    s["stages"].get(stage, {}).get("kernel_active_union_us", 0)
                    for s in profile["per_step"]
                ]
                idle = [
                    s["stages"].get(stage, {}).get("internal_idle_gap_us", 0)
                    for s in profile["per_step"]
                ]
                rows.append(
                    [
                        stage,
                        fmt(stats["mean"]),
                        fmt(statistics.mean(active)),
                        fmt(statistics.mean(idle)),
                    ]
                )
            rows.append(
                [
                    "总路径",
                    fmt(profile["total_path_span_us"]["mean"]),
                    "见分段",
                    "见分段",
                ]
            )
            table(
                ["分段", "span μs", "kernel 活跃并集 μs", "内部 idle gap μs"],
                rows,
                [2, 1, 1.25, 1.25],
            )
            kernels = sorted(
                profile.get("kernel_mean_duration_us", {}).items(),
                key=lambda item: -item[1]["mean"],
            )
            table(
                ["Kernel", "次数", "均值 μs", "范围 μs"],
                [
                    [
                        name,
                        stats["samples"],
                        fmt(stats["mean"]),
                        f"{fmt(stats['min'])}–{fmt(stats['max'])}",
                    ]
                    for name, stats in kernels
                ],
                [3.8, 0.5, 0.7, 1],
            )
            add("SQLite SHA256：" + profile["sqlite"]["sha256"], "small")
        if "profile" in images:
            figure(
                images["profile"],
                "图 3　独立诊断的一层路径分解。各臂不同 batch／层时仅描述各自观测，不构成同条件因果对照。",
            )
    else:
        add(
            "没有通过 decode-scope 审核的独立 profile；该部分保留待测，禁止根据模型总耗时填充 kernel 分段。"
        )
    for arm, _, profile in profiles:
        if profile.get("status") != "analyzed":
            add(
                f"{arm} profile：{profile.get('status', 'unknown')}；{profile.get('reason', '未取得可用独立证据')}。"
            )
    add(
        "device child memcpy 活动缺失时，不把 memcpy cost 填为零。软件 copy counters 的 idle 前后差值"
        "可能包含 bench_serving warmup；它不是硬件 PCIe／CE 传输量，不能用于有效带宽计算。"
    )

    section("7　正确性、机制证据与可复核边界")
    for path, record in evidence:
        arm = (
            record.get("stage", {}).get("xm_adaptation", {}).get("arm")
            if isinstance(record.get("stage"), dict)
            else None
        )
        add(path.name + (" / " + arm if arm else ""), "heading")
        if record.get("scope"):
            add("原始 scope：" + str(record["scope"]), "small")
        gpu_tests = record.get("gpu_tests")
        if isinstance(gpu_tests, dict):
            table(
                ["观测", "值"],
                [
                    [
                        "独立 GPU kernel tests",
                        str(gpu_tests.get("pytest_summary", "未提供")),
                    ],
                    [
                        "FP32 reference max abs error",
                        str(gpu_tests.get("fp32_reference_max_abs_error", "未提供")),
                    ],
                    [
                        "refill tolerance",
                        json.dumps(gpu_tests.get("exact_refill_tolerance", {})),
                    ],
                    ["原始 workload", str(record.get("workload", "未说明"))],
                ],
                [1.3, 3.5],
            )
        if record.get("kind") == "correctness":
            executed = record.get("settings", {}).get("execute") is True
            add(
                "真实模型执行："
                + str(executed)
                + "；manifest 状态："
                + record.get("status", "未记录"),
                "small",
            )
            groups = record.get("groups", [])
            if executed and groups:
                rows = []
                for group in groups:
                    comparison = group.get("comparison_to_cpu_hisparse", {})
                    reuse = group.get("slot_reuse_comparison", {})
                    lifecycle = group.get("lifecycle_results", [])
                    rows.append(
                        [
                            group.get("variant", "未知"),
                            "eager" if group.get("eager") else "Graph",
                            group.get("status", "未记录"),
                            str(comparison.get("exact_greedy_token_match", "未记录")),
                            fmt(comparison.get("max_abs_logprob_error"), 6),
                            fmt(comparison.get("logprob_atol"), 6),
                            str(reuse.get("exact_greedy_token_match", "未记录")),
                            "; ".join(
                                f"{item.get('case', item.get('kind', 'case'))}: {item.get('status', 'unknown')}"
                                for item in lifecycle
                            )
                            or "未提供",
                        ]
                    )
                table(
                    [
                        "KV backing",
                        "执行",
                        "状态",
                        "token exact",
                        "logprob 最大\n绝对误差",
                        "容差",
                        "slot reuse\nexact",
                        "lifecycle",
                    ],
                    rows,
                    [0.8, 0.7, 0.75, 0.8, 1, 0.8, 0.8, 1.75],
                )
                add(
                    "比较方向是当前 CPU DDR HiSparse 臂与独立 HBM Quest，首个 CPU 参考自比较不构成独立证明；"
                    "HBM 组和 eager/Graph、slot reuse/lifecycle 共同限定已验证的路径。",
                    "small",
                )
            else:
                add("未执行的计划不计作模型正确性。", "small")
        if record.get("phase") == "correctness" and isinstance(
            record.get("cases"), list
        ):
            table(
                ["模型实验臂", "case 状态", "退出码"],
                [
                    [
                        case.get("arm", "未知"),
                        case.get("status", "未记录"),
                        case.get("returncode", "未记录"),
                    ]
                    for case in record["cases"]
                ],
                [2, 1.5, 1],
            )
        if record.get("native_mechanism"):
            mechanism = record["native_mechanism"]
            table(
                ["机制", "观测"],
                [[key, str(value)] for key, value in mechanism.items()],
                [1.8, 3],
            )
        if record.get("gpu"):
            add(
                "设备原始观测：" + json.dumps(record["gpu"], ensure_ascii=False),
                "small",
            )
        if record.get("nsight"):
            nsight = record["nsight"]
            add(
                "设备 child graph 已观测："
                + str(nsight.get("graph_child_observed", "未记录"))
                + "；CE profile 已证明："
                + str(nsight.get("ce_dma_verified_by_profile", "未记录")),
                "small",
            )
            add(
                str(nsight.get("gpu_metrics_limitation", "未提供 GPU counter 证据")),
                "small",
            )
        if record.get("frozen_linux_gpu_suite"):
            suite = record["frozen_linux_gpu_suite"]
            table(
                ["冻结 Linux 源码验收", "记录"],
                [
                    [
                        "tests / passed",
                        f"{suite.get('tests', '未记录')} / {suite.get('passed', '未记录')}",
                    ],
                    [
                        "failures / errors / skipped",
                        f"{suite.get('failures', '未记录')} / {suite.get('errors', '未记录')} / {suite.get('skipped', '未记录')}",
                    ],
                    ["XML SHA256", suite.get("xml", {}).get("sha256", "未记录")],
                ],
                [1.5, 3.5],
            )
            normalized = record.get("isolated_overlay_sources", [])
            add(
                f"关联 receipt 覆盖 {len(normalized)} 个早期 overlay 文件；"
                f"其中行尾归一后同源 {sum(item.get('same_after_line_ending_normalization') is True for item in normalized)} 个。"
                "这是源码关联与冻结源测试证据，不是模型输出或性能证据；002 二进制 receipt 不表示所有最终 LF 构建的 raw hash 相同。",
                "small",
            )
        for key in (
            "status",
            "source_state",
            "source_commit",
            "baseline_commit",
            "git_commit",
            "result",
            "adaptation",
        ):
            if key in record:
                add(f"{key}：{record[key]}", "small")
        if record.get("not_established_by_these_tests"):
            add(
                "这些测试没有建立："
                + "；".join(record["not_established_by_these_tests"]),
                "small",
            )
        add("证据 JSON SHA256：" + identity(path)["sha256"], "small")
    if not evidence:
        add(
            "本次生成未提供额外正确性／native mechanism JSON；只有数值 analysis，不能扩展为完整模型正确性结论。"
        )
    add(
        "源码关联范围：早期 isolated probe/overlay 与最终 LF 部署源码可能有不同 raw 文件 hash，"
        "包括 Windows CRLF 行尾差异。模型性能统一以上 analysis 的实现指纹与 native git commit；"
        "早期 probe 的通过不表示其所有 raw source bytes 或二进制与最终模型部署完全相同。"
        "最终冻结源验收应作为独立 receipt 保留。",
        "small",
    )
    add("本次适配能回答什么", "heading")
    add(
        "同请求、同实际 batch 的稳定 decode 成本；同配置 cache HBM budget 下已经观测的容量与输出率；"
        "具名 KV／摘要／管理／workspace 的分配差异。结果无论改善或回退均保留。"
    )
    add("仍不能外推什么", "heading")
    add(
        "论文 B300 scale-up fabric 的性能；训练型 K-index 的预取收益；4＋0/4＋4 的 Attention-DP＋EP 模型；"
        "真实 MPU 的 1＋1 跨卡通路；物理 CE 利用率或重叠；未给定 SLO 和小时成本时的性价比甜点区。"
    )

    section("8　数据来源与复核")
    rows = []
    for path, data in analyses:
        rows.append([path.name, data["phase"], identity(path)["sha256"]])
    for arm, path, _ in profiles:
        rows.append(
            [path.name, f"independent profile / {arm}", identity(path)["sha256"]]
        )
    for path, _ in evidence:
        rows.append([path.name, "separate scoped evidence", identity(path)["sha256"]])
    table(["文件", "角色", "SHA256"], rows, [1.6, 1.4, 3.4])
    add(
        "analysis 的原始 artifact 列表包含 serving result、scheduler trace、manifest 与重放 timing/steady 文件的内容 hash。"
        "本 PDF 的 sidecar receipt 另记录实际输入与输出 hash。派生 hash 证明本次使用的内容身份；"
        "它不替代更早的证据传输与验收记录。"
    )

    output.parent.mkdir(parents=True, exist_ok=True)

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#c9d2d8"))
        canvas.line(44, 42, A4[0] - 44, 42)
        canvas.setFont("ReportChinese", 7.5)
        canvas.setFillColor(colors.HexColor("#516571"))
        text = (
            "内部布局 fixture，非实验数据"
            if internal_fixture
            else "Qwen/Quest on one H800 PCIe + CPU DDR | XM-Sparse adaptation"
        )
        canvas.drawString(44, 29, text)
        canvas.drawRightString(A4[0] - 44, 29, str(document.page))
        canvas.restoreState()

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=44,
        rightMargin=44,
        topMargin=42,
        bottomMargin=56,
        title="Qwen Quest XM-Sparse 软件机制适配实验",
        author="MPU inference experiments",
    )
    document.build(story, onFirstPage=footer, onLaterPages=footer)


def append_pdf(base, addition, output):
    from pypdf import PdfReader, PdfWriter

    require(
        output.resolve() not in (base.resolve(), addition.resolve()),
        "Merged PDF must use a new filename",
    )
    original_hash = identity(base)["sha256"]
    original = PdfReader(base)
    texts = [page.extract_text() for page in original.pages]
    writer = PdfWriter()
    writer.clone_document_from_reader(original)
    writer.append(str(addition))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)
    merged = PdfReader(output)
    require(
        [page.extract_text() for page in merged.pages[: len(texts)]] == texts,
        "Original PDF page text changed",
    )
    require(identity(base)["sha256"] == original_hash, "Original PDF changed")
    return {
        "original_pages": len(texts),
        "appended_pages": len(merged.pages) - len(texts),
        "original_page_texts_exactly_preserved": True,
        "original_file_unchanged": True,
    }


def write_named_allocations(analyses, output):
    rows = []
    for _, analysis in analyses:
        for sample in analysis.get("samples", []):
            for boundary in ("memory_before", "memory_after"):
                memory = sample[boundary]
                for owner in ("cache", "backend"):
                    for record in memory.get(owner + "_allocation_groups", []):
                        rows.append(
                            {
                                "phase": analysis["phase"],
                                "arm": sample["arm"],
                                "input_len": sample["input_len"],
                                "client_concurrency": sample["client_concurrency"],
                                "repeat": sample["repeat"],
                                "boundary": boundary,
                                "owner": owner,
                                "allocation_group": record["allocation_group"],
                                "category": record["category"],
                                "storage_bytes": record["storage_bytes"],
                                "alias_names": json.dumps(
                                    record["alias_names"], ensure_ascii=False
                                ),
                            }
                        )
    if not rows:
        return None
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--profile", action="append", default=[], metavar="ARM=PATH")
    parser.add_argument("--evidence", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-pdf", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--font", type=Path, default=DEFAULT_FONT)
    parser.add_argument("--plot-packages", type=Path)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--internal-layout-fixture", action="store_true")
    args = parser.parse_args(argv)
    require(
        bool(args.base_pdf) == bool(args.merged_output),
        "--base-pdf and --merged-output must be supplied together",
    )
    require(
        not args.internal_layout_fixture or args.base_pdf is None,
        "Internal fixture cannot be appended to a real report",
    )
    require(args.font.is_file(), "Chinese TrueType font not found; pass --font")
    require(not args.output.exists(), "Refusing to overwrite existing PDF")
    if args.merged_output:
        require(not args.merged_output.exists(), "Refusing to overwrite merged PDF")
    if args.plot_packages:
        sys.path.insert(0, str(args.plot_packages.resolve(strict=True)))
    analyses = validate_analyses(
        args.analysis, internal_fixture=args.internal_layout_fixture
    )
    profiles = [validate_profile(spec) for spec in args.profile]
    evidence = [(path.resolve(strict=True), read(path)) for path in args.evidence]
    source = analyses[0][1]
    for path, record in evidence:
        if record.get("phase") == "correctness":
            require(
                record.get("source_fingerprint") == source["source_fingerprint"],
                f"Correctness source differs: {path}",
            )
            require(
                record.get("workload_identity", {}).get("sha256")
                == source["workload_identity"]["sha256"],
                f"Correctness workload differs: {path}",
            )
        if (
            record.get("kind") == "correctness"
            and record.get("settings", {}).get("execute") is True
        ):
            require(
                record.get("runtime", {}).get("git_commit")
                in source["native_source_commits"],
                f"Native correctness source differs: {path}",
            )
            workload = record.get("settings", {}).get("workload_sha256")
            require(
                workload is None or workload == source["workload_identity"]["sha256"],
                f"Native correctness workload differs: {path}",
            )
    images = (
        {}
        if args.skip_plots
        else draw_plots(
            analyses, profiles, args.output.parent / (args.output.stem + "-figures")
        )
    )
    render_report(
        analyses,
        profiles,
        evidence,
        args.output,
        font_path=args.font,
        internal_fixture=args.internal_layout_fixture,
        images=images,
    )
    named_csv = write_named_allocations(
        analyses, args.output.with_name(args.output.stem + "-named-allocations.csv")
    )
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": (
            "internal layout fixture"
            if args.internal_layout_fixture
            else "rendered observed analysis, scoped probes and independent profiles"
        ),
        "inputs": [identity(path) for path in args.analysis + args.evidence]
        + [identity(path) for _, path, _ in profiles],
        "standalone_pdf": identity(args.output),
        "figures": [identity(path) for path in sorted(set(images.values()))],
        "visual_review": "pending_external_render_and_inspection",
    }
    if named_csv:
        receipt["named_allocations_csv"] = identity(named_csv)
    if args.base_pdf:
        receipt["base_pdf"] = identity(args.base_pdf)
        receipt["preservation"] = append_pdf(
            args.base_pdf, args.output, args.merged_output
        )
        receipt["merged_pdf"] = identity(args.merged_output)
    receipt_path = args.output.with_suffix(".receipt.json")
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pdf": str(args.output),
                "receipt": str(receipt_path),
                "merged_pdf": str(args.merged_output) if args.merged_output else None,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
