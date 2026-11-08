"""Create the local ABC experiment report from audited measurements."""

import argparse
import html
from pathlib import Path

from evaluate_abc_v7 import ACTIVE, DEFAULT_OUT, SOURCES, read_json, read_rows, write_csv, write_json

NAMES = dict(walking="行走", sitting_down="坐下", standing_up="起身", bending="弯腰", falling="跌倒过程")


def esc(value):
    return html.escape(str(value))


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def clock_name(entry):
    status = entry["clock"]["status"]
    names = {
        "source_frame_index_divided_by_30": "按原始帧号÷30换算；采样不均匀",
        "nominal_video_30fps": "现有视频名义30 FPS；非官方逐帧时间戳",
    }
    if entry["clip_id"] == "urfd-fall-01-original":
        return "官方源 CSV 时间戳"
    if entry["clip_id"].startswith("multicam"):
        return "仅源帧序；物理时钟未确认"
    return names.get(status, status)


def pending_text(entry):
    cid = entry["clip_id"]
    if cid.startswith("utk"):
        return "官方动作粗区间与坐姿／前倾／离座／站稳边界；人物可见性；源帧号换算时钟假设。"
    if cid.startswith("urfd-fall-01"):
        return "83–112官方过渡代理；113–137翻滚段动作未标；接地101–107、稳定倒地132–138候选；可见性与同事件版本映射。"
    if cid == "urfd-fall-02":
        return "40–69官方过渡代理；70–99翻滚／倒地边界未标；早期静坐与可见性；名义视频时间。"
    if cid.startswith("multicam"):
        return "物理时钟；入画／出画与空房边界；末尾多人目标身份；接地／稳定倒地；源视频损坏范围。"
    return "行走、弯腰或坐下的起止边界与静态内段；人物可见性；名义30 FPS时间假设。"


def table(headers, records):
    return '<div class="scroll"><table><thead><tr>' + "".join(
        f"<th>{esc(x)}</th>" for x in headers
    ) + "</tr></thead><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in record) + "</tr>" for record in records
    ) + "</tbody></table></div>"


def media_block(item):
    path = item.get("file", item.get("path"))
    return f'''<article class="card">
<h3>{esc(item.get("title", item.get("clip_id", item.get("clip", ""))))}</h3>
<p class="muted">{esc(item.get("description", item.get("detail", "")))}</p>
<video controls preload="metadata" playsinline src="{esc(path)}"></video>
<div class="buttons"><button onclick="speed(this,1)">1×</button><button onclick="speed(this,.5)">0.5×</button>
<button onclick="speed(this,.25)">0.25×</button><a href="{esc(path)}">打开视频</a></div>
</article>'''


def timing_rows(out, annotations):
    """Prepare source-clock detection records, not a claim about delivered alerts."""
    cid = "urfd-fall-01-original"
    e = next(e for e in annotations["entries"] if e["clip_id"] == cid)
    landmarks = e["event_landmarks"]
    onset = landmarks.get("onset_proxy", {})
    rows = []
    for mode, folder in SOURCES.items():
        raw = read_rows(folder / cid / "frames.jsonl")
        preds = [r["prediction"] if mode == "A" else r["after"]["prediction"] for r in raw]
        hits = [r for r, p in zip(raw, preds) if p["label"] == "falling"]
        hit = hits[0] if hits else None
        rows.append(dict(
            mode=mode, clip_id=cid, onset_proxy_frame=onset.get("frame"),
            onset_proxy_s=onset.get("time_s"), first_falling_frame=hit["original_frame_id"] if hit else None,
            first_falling_s=hit["source_time_s"] if hit else None,
            detection_delay_from_onset_proxy_s=hit["source_time_s"] - onset["time_s"]
            if hit and onset.get("time_s") is not None else None,
            physical_alert_delivery_s=None, end_to_end_latency_s=None,
            contact=landmarks.get("contact"), stable_lying=landmarks.get("stable_lying"),
            human_reviewed=False,
            meaning="first raw V7 falling label, no alert debounce/delivery; source-clock audit only",
        ))
    write_json(out / "timing-preparation.json", dict(clip_id=cid, landmarks=landmarks, detections=rows))
    write_csv(out / "timing-preparation.csv", rows)
    return landmarks, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output.resolve()
    result = read_json(out / "metrics.json")
    annotations = read_json(out / "annotations/annotations.json")
    audit = read_json(out / "audit/inventory.json")
    aggregate = result["aggregate"]
    metrics = {(m["mode"], m["clip_id"]): m for m in result["clips"]}
    media = read_json(out / "media/manifest.json")
    full_videos = media.get("full_videos", media.get("full", []))
    cases = media.get("cases", [])
    landmarks, timing = timing_rows(out, annotations)
    model_hash = result["protocol"]["model_sha256"]
    agg_rows = []
    for mode, m in aggregate.items():
        falls = m["events_by_action"]["falling"]
        rises = m["events_by_action"]["standing_up"]
        agg_rows.append([mode, f'{falls["detected"]}/{falls["events"]}',
                        f'{rises["detected"]}/{rises["events"]}',
                        f'{m["action_event_hits"]}/{m["action_event_total"]}',
                        number(m["unknown_s"]), number(m["target_unavailable_s"]),
                        str(m["false_action_episodes"]["falling"]), number(m["false_action_s"]["falling"]),
                        str(m["seated_false_rise_samples"]),
                        f'{m["seated_forward_lean_samples"]}/{m["seated_reference_samples"]}'])
    headers = ["组", "跌倒事件命中", "起身事件命中", "五动作事件命中", "unknown 估计秒",
               "目标无观测估计秒", "跌倒误报段", "跌倒误报估计秒", "静坐报起身（动作）采样", "静坐报前倾（phase）/支持"]
    per_clip_rows = []
    for e in annotations["entries"]:
        cid = e["clip_id"]
        ms = [metrics[mode, cid] for mode in SOURCES]
        per_clip_rows.append([
            esc(cid) + ("<br><span class=tag>同事件辅助版本</span>" if not ms[0]["primary"] else ""),
            str(ms[0]["samples"]),
            " / ".join(str(m["target_unavailable_samples"]) for m in ms),
            " / ".join(number(m["unknown_s"]) for m in ms),
            " / ".join(f'{m["action_event_hits"]}/{m["action_event_total"]}' if not m["frame_only"]
                       else "禁用" for m in ms),
        ])
    class_rows = []
    for label in ACTIVE:
        class_rows.append([NAMES[label]] + [
            f'{percent(aggregate[mode]["per_class"][label]["precision"])} / '
            f'{percent(aggregate[mode]["per_class"][label]["recall"])} / '
            f'{percent(aggregate[mode]["per_class"][label]["f1"])}' for mode in SOURCES
        ] + [number(aggregate["A"]["per_class"][label]["support_s"])])
    mcid = "multicam-chute01-cam8"
    mc_rows = [[mode, str(metrics[mode, mcid]["visible_target_unavailable_samples"]),
                str(metrics[mode, mcid]["visible_reference_samples"]),
                str(metrics[mode, mcid]["empty_false_target_samples"]),
                str(metrics[mode, mcid]["empty_reference_samples"])] for mode in SOURCES]
    timing_table = [[r["mode"], str(r["first_falling_frame"]), number(r["first_falling_s"]),
                     number(r["detection_delay_from_onset_proxy_s"])] for r in timing]
    annotation_rows = []
    for e in annotations["entries"]:
        annotation_rows.append([esc(e["clip_id"]), esc(e["event_group_id"]),
                                esc(clock_name(e)), esc(pending_text(e)),
                                "<strong>未完成人工复核</strong>"])
    errors = [
        dict(id="E01", category="目标误准入", clip_id=mcid,
             finding="MultiCam 空房里 A 可能跟踪椅子；B/C 的交叉确认抑制假目标。",
             evidence="比较空房人工待复核区间的 false_target 样本，并播放 multicam-empty。",
             remaining="仅是已看过的场景回归；没有完整人体框、身份真值，不称检测准确率。",
             next_step="独立复核空房与出画边界，再扩展新场景。"),
        dict(id="E02", category="人体缺失与动作缺失分离", clip_id="urfd-fall-01-original",
             finding="C 可保留第134–135帧局部人体，但骨架不足，动作继续 unknown。",
             evidence="同帧 A/B/C 选中目标、confidence_supported_joints 与 presence_only。",
             remaining="补回目标不能直接等同姿态正确或动作正确。",
             next_step="人工核查局部关节与遮挡；保持质量拒识。"),
        dict(id="E03", category="标签边界冲突", clip_id="urfd-fall-01",
             finding="第117帧 C 输出 falling，但官方过渡代理区间在第112帧结束。",
             evidence="旧五动作标注与官方 state 的边界不同，113之后翻滚段在新标签中单列待审。",
             remaining="该帧只证明观测恢复和输出改变，不能直接计作正确动作改善。",
             next_step="独立复核接地、翻滚、稳定倒地区间；不按模型输出改标签。"),
        dict(id="E04", category="动作阶段判别", clip_id="utk-s10_e01-transition",
             finding="静坐内段仍有 forward_lean 输出；主动作 background 与阶段误判是两件事。",
             evidence="state_segments 静坐内段与 A/B/C phase 对照。",
             remaining="前端一致不代表动作阶段正确；官方 standUp 宽区间不能替代离座边界。",
             next_step="先人工复核静坐/前倾/离座边界，再决定是否另开模型优化。"),
        dict(id="E05", category="新增拒识代价", clip_id="UTK / ADL",
             finding="B/C 的新人确认可能增加开场 unknown；事件命中不变也不能忽略覆盖损失。",
             evidence="逐片 unknown 时长、目标缺失和 A→B 改变帧清单。",
             remaining="没有为了减少 unknown 而放宽准入；需同时看假目标与拒识。",
             next_step="在完整片段上保留开场确认，不删去这些采样。"),
        dict(id="E06", category="遮挡与身份连续性", clip_id=mcid,
             finding="门口遮挡、出画交界及末尾多人区间仍需人工核查。",
             evidence="门口短片及 annotations visibility=uncertain 区间。",
             remaining="epoch 数不等于真实 ID 切换；空房不能计成漏人。",
             next_step="补目标身份与可见性真值后再计算跟踪错误率。"),
        dict(id="E07", category="时钟与实时性", clip_id="all",
             finding="MultiCam 只有帧序；C 日志仅记录补检路径耗时，未测端到端交付延迟。",
             evidence="audit / protocol / timing-preparation。",
             remaining="本轮无实时预警提前量，也无实际提醒送达时间。",
             next_step="源时钟核实、采集与推理排队测量独立进行。"),
    ]
    write_json(out / "error-catalog.json", errors)
    write_csv(out / "error-catalog.csv", errors)
    body = [f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定 V7 · A/B/C 感知对照实验</title>
<style>
:root{{--ink:#152434;--muted:#536575;--line:#d6e0e7;--blue:#125e99;--light:#edf4f8}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f7f9;color:var(--ink);font:16px/1.75 system-ui,-apple-system,sans-serif}}
main{{max-width:1280px;margin:auto;padding:30px 24px 80px}}h1{{font-size:34px;line-height:1.3;margin:12px 0}}h2{{font-size:25px;margin-top:45px}}h3{{font-size:19px}}
a{{color:var(--blue)}}.eyebrow{{color:var(--blue);font-weight:700;letter-spacing:.06em}}.muted{{color:var(--muted);font-size:14px}}
.card,.hero{{background:white;border:1px solid var(--line);border-radius:14px;padding:24px;margin:18px 0}}
.hero{{border-top:5px solid var(--blue)}}.note{{padding:17px 20px;background:#fff6df;border-left:4px solid #c99416;margin:18px 0}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}}.stat{{background:var(--light);padding:14px;border-radius:8px}}
.stat b{{font-size:27px;display:block}}nav{{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}}
.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;background:white}}th,td{{padding:11px;border:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#e6eff5;white-space:nowrap}}
video{{display:block;width:100%;background:#111;border-radius:6px}}.buttons{{display:flex;gap:9px;align-items:center;margin-top:10px}}button{{border:1px solid #aabccc;padding:7px 15px;border-radius:6px;background:white;color:var(--ink);cursor:pointer}}
code{{background:var(--light);padding:2px 5px;overflow-wrap:anywhere}}.tag{{display:inline-block;background:#fff0c7;border-radius:4px;padding:0 5px;font-size:12px}}
details{{margin:16px 0}}summary{{cursor:pointer;font-weight:600}}img{{max-width:100%;height:auto}}li{{margin:8px 0}}
@media(max-width:720px){{main{{padding:20px 12px}}h1{{font-size:27px}}.stats{{grid-template-columns:1fr 1fr}}.card,.hero{{padding:16px}}}}
</style></head><body><main>
<div class="eyebrow">FALLPREAGENT / 2026-10-20 / 开发回归</div>
<section class="hero"><h1>固定 V7，拆开人体前端带来的收益与代价</h1>
<p>A：原跟踪，不补检。B：目标确认与跟踪，不补检。C：目标确认与跟踪，加当前图像补检。所有组使用同一个 V7；本轮没有训练。</p>
<div class="stats"><div class="stat"><b>8 / 7</b>输入路数 / 原始事件组</div>
<div class="stat"><b>2483</b>每组原始观测</div><div class="stat"><b>7449</b>缓存重放精确一致</div>
<div class="stat"><b>0</b>新增视觉模型推理</div></div>
<p class="muted">V7 SHA256：<code>{model_hash}</code>。三组日志齐全，模型、输入、原生检测及各组源帧与记录时间对齐已核验。B 历史跟踪文件散列不同，但当前实现逐项重放等价。</p></section>
<div class="note"><strong>指标是对暂定标签的开发回归。</strong>项目标签尚未经过本轮独立人工复核。官方动作区间、官方状态代理和本轮 AI 视觉内段分开保存；不把官方非倒地状态直接解释成 background，也不把躺地当成正在跌倒。原始/低清 fall01 只计一次主事件。</div>
<nav><a href="#metrics">指标对照</a><a href="#cases">重点视频</a><a href="#errors">错误归因</a><a href="#timing">实验三时间准备</a><a href="#labels">待人工复核</a><a href="#full">8路完整视频</a><a href="#files">证据与复现</a></nav>
<h2 id="metrics">主指标：6 个有时间轴、不重复的原始录像组</h2>
<p>主汇总采用 fall01 原始版，低清版只作画质对照；MultiCam 单列帧级结果。事件命中规则为区间内至少一次输出对应动作，不代表持续识别正确，也不是已实现的报警策略。</p>''']
    body.append(table(headers, agg_rows))
    body.append(
        f'<p><strong>本轮结论：</strong>前端主要改善了假目标准入与躺地观测连续性；'
        f'三组动作事件命中均为 {aggregate["C"]["action_event_hits"]}/{aggregate["C"]["action_event_total"]}。'
        f'C 的 unknown 为 {number(aggregate["C"]["unknown_s"])} 秒，低于 B 的 '
        f'{number(aggregate["B"]["unknown_s"])} 秒，但高于 A 的 {number(aggregate["A"]["unknown_s"])} 秒。'
        '补检收益与新人确认代价同时存在，不能据此声称动作整体改善。</p>'
        f'<p class="muted">主汇总源时间跨度约 {number(aggregate["A"]["observed_span_s"])} 秒，'
        f'其中动作参考覆盖约 {number(aggregate["A"]["annotated_action_s"])} 秒；'
        f'另有 {number(aggregate["A"]["unscored_action_s"])} 秒未评分。未评分区间内 falling 输出采样数 A/B/C：'
        f'{aggregate["A"]["unscored_falling_samples"]} / {aggregate["B"]["unscored_falling_samples"]} / '
        f'{aggregate["C"]["unscored_falling_samples"]}，均待边界复核。表中的零跌倒误报仅指已标注区间，'
        '不能解释为全片零误报或长期误报率为零。</p>'
    )
    body.append('''<p class="muted">秒数按相邻源时间差对当前采样输出加权，单次最多保持0.35秒；末帧时长为0，长缺口单列。它是输出占用时长近似，不是逐帧人工真值。unknown保留在动作召回的分母中；未标注区间不计正确或误报。</p>
<details><summary>五动作：源时间加权 precision / recall / F1</summary>
<p>这些数值衡量与冻结参考的对应关系。样本很少、时间窗相关，且部分官方区间语义较粗；不作显著性或跨人物泛化结论。</p>''')
    body.append(table(["动作", "A：P / R / F1", "B：P / R / F1", "C：P / R / F1", "参考支持估计秒"], class_rows))
    body.append("</details><h3>每路输入（顺序均为 A / B / C）</h3>")
    body.append(table(["输入", "观测数", "全片目标无观测数", "unknown估计秒", "动作事件命中"], per_clip_rows))
    body.append('''<p class="muted">全片目标无观测数包含空房、出画、确认等待，不能直接称漏检率。尤其 MultiCam 中 A 的假椅子目标会使“无观测”看起来更少。</p>
<h3>MultiCam：只在指定可见／空房区间检查观测</h3>''')
    body.append(table(["组", "可见区间无目标", "可见参考采样", "空房错误建目标", "空房参考采样"], mc_rows))
    body.append('<p class="muted">可见性区间为已有证据加本轮 AI 目视暂定，尚待人工复核；不包含末尾身份不清的多人区间，不计算秒级动作或提前量。“有目标观测”可能仍选错人或椅子，不能替代目标正确率。</p><h2 id="cases">同帧三列可播放对照</h2><p>左 A、中 B、右 C。播放器变速只用于检查画面，不改变模型输入时间。</p>')
    body.extend(media_block(x) for x in cases)
    body.append('<h2 id="errors">错误归因与下一步</h2>')
    for error in errors:
        body.append(f'<section class="card"><h3>{esc(error["id"])} · {esc(error["category"])}</h3>'
                    f'<p>{esc(error["finding"])}</p><p class="muted">{esc(error["evidence"])}</p>'
                    f'<p><strong>剩余问题：</strong>{esc(error["remaining"])}</p>'
                    f'<p><strong>下一步：</strong>{esc(error["next_step"])}</p></section>')
    body.append('<p><a href="attribution.csv">下载全部改变帧与归因 CSV</a> · <a href="error-catalog.csv">下载错误清单 CSV</a></p>'
                '<h2 id="timing">实验三：时间标注准备</h2>'
                '<p>先以 URFD fall01 原始 RGB 的官方源 CSV 对齐；记录的是首次原始 falling 输出，尚未加入报警持时规则、通知或实际送达测量。</p>')
    body.append(table(["组", "首次falling源帧", "首次falling源秒", "相对官方起跌代理的延迟秒"], timing_table))
    body.append('<p>官方 state=0 起点只作为数据集代理；首次接地与稳定倒地保留候选区间及人工待审状态，不用 state=1 自动替代接地。</p>')
    body.append(table(["标记", "已确定帧 / 秒", "候选区间", "状态"], [
        [esc(name), f'{esc(v.get("frame"))} / {esc(v.get("time_s"))}',
         esc(str(v.get("frame_interval"))) + "<br>" + esc(str(v.get("time_interval_s"))),
         esc(v.get("status"))] for name, v in landmarks.items() if isinstance(v, dict)
    ]))
    body.append('''<div class="note">没有完成端到端实时测量。历史 <code>recovery.wall_s</code> 只包含补检路径，不含完整视觉推理、排队及提醒输出；本轮不把离线源时间差报告为实际可用预警提前量。</div>
<p><a href="timing-preparation.json">时间标注与检测记录 JSON</a> · <a href="timing-preparation.csv">时间记录 CSV</a></p>
<h2 id="labels">未完成人工复核的标注</h2>
<p>以下所有项目级标注均为“待独立人工复核”。官方标签来源保留；human_reviewed=false 指本轮未获独立人工确认，不否认发布方的原始标注工作。图像接触表均来自未叠加模型输出的原 RGB。</p>''')
    body.append(table(["输入", "事件组", "时钟", "待复核项", "人工状态"], annotation_rows))
    body.append('<p><a href="annotations/annotations.json">完整标签 JSON</a> · <a href="annotations/review-notes.md">标签取舍与复核说明</a></p>')
    body.append("<details><summary>原 RGB 接触表（辅助复核，不等于人工真值）</summary>")
    for image in sorted((out / "annotations").glob("*.jpg")):
        body.append(f'<h3>{esc(image.stem)}</h3><img loading="lazy" src="annotations/{esc(image.name)}">')
    body.append('</details><h2 id="full">全部 8 路完整 A/B/C 视频</h2>')
    body.extend(media_block(x) for x in full_videos)
    body.append('''<h2 id="files">证据、验证与复现</h2>
<ul>
<li><a href="metrics.csv">逐组逐片指标 CSV</a> · <a href="metrics.json">完整指标与混淆表 JSON</a></li>
<li><a href="action-metrics.csv">各动作精确率、召回、F1与误报时长 CSV</a></li>
<li><a href="events.csv">每个动作事件命中及首次命中时间</a> · <a href="episodes.json">unknown与误报连续段</a></li>
<li><a href="evaluation-protocol.json">固定模型、日志哈希与统计口径</a></li>
<li><a href="audit/inventory.json">输入/模型/缓存一致性审计</a> · <a href="media/validation.json">视频完整解码校验</a></li>
<li><a href="validation.json">本轮脚本检查与产物校验</a></li>
<li><a href="browser-playback-check.json">浏览器实际播放核验</a> · <a href="browser-playback.png">播放页面截图</a></li>
</ul><p>历史日志和模型没有覆盖。复算入口（在仓库根目录执行）：</p>
<pre><code>.venv/bin/python scripts/evaluate_abc_v7.py
.venv/bin/python scripts/render_abc_v7_media.py
.venv/bin/python scripts/report_abc_v7.py</code></pre>
<p class="muted">本轮定位：已知数据上的开发回归与错误归因。没有训练，没有视觉补跑，没有独立泛化或预防跌倒效果声明。</p>
</main><script>function speed(b,r){const v=b.closest('article').querySelector('video');v.playbackRate=r;v.play();}</script></body></html>''')
    (out / "index.html").write_text("\n".join(body))
    markdown = [
        "# 固定 V7 的 A/B/C 感知对照实验",
        "",
        "本轮复用8路输入、7个原始事件组。模型不训练；全部7449次缓存跟踪与动作重放精确一致。",
        "A=legacy/off；B=verified/off；C=verified/evidence。所有项目标签仍待独立人工复核。",
        "",
        "|组|跌倒事件|起身事件|五动作事件|unknown估计秒|目标无观测估计秒|跌倒误报段|跌倒误报估计秒|静坐报起身采样|静坐报前倾/静坐采样|",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    markdown += ["|" + "|".join(row) + "|" for row in agg_rows]
    markdown += [
        "",
        "主汇总为6个有时间轴、不重复的原始录像组；fall01原始版为主，低清版只作辅助。MultiCam仅帧级。",
        "事件命中采用区间内至少一次动作输出；源时间加权指标计入unknown，标签空白不计正确/错误。",
        "秒数是采样输出左保持至下一源时间的估计，最长0.35秒，末帧0；不是逐帧人工动作真值。",
        f'动作未标注区间仍有 falling 输出采样 A/B/C='
        f'{aggregate["A"]["unscored_falling_samples"]}/'
        f'{aggregate["B"]["unscored_falling_samples"]}/'
        f'{aggregate["C"]["unscored_falling_samples"]}，均待边界复核。零跌倒误报仅指已标注区间，不能解释为全片零误报。',
        "",
        "## 错误归因",
    ]
    for error in errors:
        markdown += [f'- **{error["id"]} {error["category"]}**：{error["finding"]} {error["remaining"]}']
    markdown += [
        "",
        "## 时间标注准备",
        "",
        "URFD fall01原始版：官方state=0作为起跌代理，接地及稳定倒地为AI候选区间，待人工复核。",
        "首次falling输出不等于报警送达；未测端到端延迟，不报告实际预警提前量。",
        "",
        "## 交付",
        "",
        f"- [可播放报告]({out / 'index.html'})",
        f"- [指标 CSV]({out / 'metrics.csv'})",
        f"- [事件结果]({out / 'events.csv'})",
        f"- [错误清单]({out / 'error-catalog.csv'})",
        f"- [待复核标签]({out / 'annotations/annotations.json'})",
        f"- [标签说明]({out / 'annotations/review-notes.md'})",
        f"- [时间标注准备]({out / 'timing-preparation.json'})",
        "",
        f"固定 V7 SHA256：`{model_hash}`",
    ]
    (out / "report.md").write_text("\n".join(markdown) + "\n")
    assert audit["totals"]["missing_groups"] == []
    print(out / "index.html")


if __name__ == "__main__":
    main()
