"""Render the fixed-V7 timing audit and explicit human-review outcomes."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'runs/abc-v7-20261020'
OUT = BASE / 'experiment3-20261021'


def read(name):
    return json.loads((OUT / name).read_text())


def esc(x):
    return html.escape(str(x))


def num(x):
    return '未知' if x is None else f'{x:.3f}'


def interval(xs):
    return '未知' if not xs else '–'.join(num(x) for x in xs)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join(f'<td>{c}</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def main():
    timing = read('timing.json')
    review = read('review-summary.json')
    protocol = read('protocol.json')
    pending = read('pending-annotations.json')
    session = json.loads((BASE/'human-review/review-session.json').read_text())
    rows=[]
    for item in session['queue']:
        rows.append([f'<a href="../human-review/#{item["id"]}">{esc(item["guidance"]["short"])}</a>',esc(item['decision']['display_summary']), '仅代理，物理边界待定' if item['status']=='reviewed_proxy_only' else '区间确认，精确帧未知'])
    body=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>实验三 · V7 时间审计与人工复核</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#1e3340;background:#eef3f4;line-height:1.65}*{box-sizing:border-box}body{margin:0}main{max-width:1140px;margin:auto;padding:28px 28px 65px}h1{font-size:32px;margin:10px 0}h2{font-size:23px;margin-top:36px}h3{font-size:18px}a{color:#176a7e}nav{display:flex;flex-wrap:wrap;gap:18px;padding:12px 0}.card,.note{background:white;padding:20px;border-radius:12px;margin:16px 0;border:1px solid #d7e2e5}.note{border-left:5px solid #bf8b2d;background:#fff7e7}.lead{font-size:19px}.tag{color:#21677a;font-weight:650}.muted{color:#5b6d75;font-size:14px}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:12px 14px;border-bottom:1px solid #dce5e8;text-align:left;vertical-align:top}th{background:#e0ebed;white-space:nowrap}.scroll{overflow-x:auto}video,img.plot{width:100%;border-radius:8px;background:#11242f}button,select{font:inherit;padding:7px 12px;border:1px solid #8cabb5;border-radius:6px;background:#fff;color:#164a5e;margin:5px 5px 5px 0;cursor:pointer}code{font-size:.9em;overflow-wrap:anywhere}li{margin:7px 0}.metrics{display:flex;gap:15px;flex-wrap:wrap}.metric{background:#163e4b;color:#fff;padding:16px 22px;border-radius:10px;flex:1;min-width:180px}.metric b{display:block;font-size:26px}details{margin:14px 0}footer{margin-top:35px;font-size:13px;color:#526a73}@media(max-width:600px){main{padding:18px 12px}h1{font-size:26px}th,td{padding:9px}.card{padding:14px}}
</style><main><a href="../index.html">← 实验一冻结报告</a><h1>实验三：检测发生在什么时候？</h1><p class="muted">2026-10-21 · V7 固定 · 不训练 · 8 路输入 / 7 个原始事件组</p>
<div class="metrics"><div class="metric"><b>4 / 4</b>本轮人工复核决策已保存</div><div class="metric"><b>2 个区间 + 2 个代理</b>整体标注仍为部分复核</div><div class="metric"><b>24 份既有日志</b>A/B/C 主要结果复用原始记录</div></div>
<nav><a href="#review">四项复核</a><a href="#timing">源时间对照</a><a href="#latency">完整处理计时</a><a href="#video">可播放证据</a><a href="#pending">仍待定</a><a href="#errors">错误归因</a><a href="#files">证据文件</a></nav>
<p class="lead">源时间上，fall01 在接地前出现 falling；本次 CPU 无丢帧 FIFO 补测中，A/B/C 的实际处理完成时刻均已晚于接地。处理积压是当前必须先解决的问题。</p>
<section id="review"><h2>四项复核结果</h2>''',table(['项目','已保存的人工决定','确认范围'],rows),'''<p class="muted">s01/s10 的秒数沿用“源帧差 ÷ 30”的相对时钟。代理帧不等于物理离座或真实上升起点。稳定倒地采用躯干/骨盆停止明显整体运动并持续至少 0.5 秒的判据，允许手脚小幅调整。</p></section>
<section id="timing"><h2>固定规则下的源时间对照</h2><p>从视频第一帧开始顺序读取完整记录；首次 <code>label == falling</code> 即为诊断触发，不用真值裁切输入，不加事后平滑或防抖。UTK 单独检查首次 <code>standing_up</code> 和阶段输出；它们不作为跌倒警报。</p>''']
    fallrows=[]
    for r in timing['primary_fall']:
        hit=r['first_falling'] or {}
        fallrows.append([esc(r['mode']),str(hit.get('frame','—')),num(hit.get('time_s')),num(r.get('delay_from_onset_proxy_s')),interval(r.get('contact_margin_s')),interval(r.get('stable_onset_margin_s'))])
    body.append(table(['组','首次 falling 源帧','源时间 s','晚于官方起跌代理 s','接地前源时间余量 s','稳定起点前源时间余量 s'],fallrows))
    body.append('''<div class="note">接地余量 = 人工接地区间 − 首次输出对应的源时间。正数只表示画面顺序上先于接地，不等于真实提醒送达的提前量。83 帧仍是未人工确认的官方起跌代理，因此也不能据此声称完成了真正的起跌前预测。</div>''')
    if (OUT/'source-timeline.png').exists():
        body.append('<a href="source-timeline.png"><img class="plot" src="source-timeline.png" alt="fall01 原始视频源时间轴，区分模型输出、接地及稳定区间"></a>')
    utkrows=[]
    for r in timing['utk']:
        hit=r.get('first_standing_up') or {}
        proxy=r.get('visible_rise_proxy') or {}
        lean=r.get('first_forward_lean') or {}
        rising=r.get('first_rising') or {}
        utkrows.append([esc(r['clip_id'].split('-')[1]),esc(r['mode']),str(proxy.get('frame','—')),str(hit.get('frame','—')),num(r.get('delay_to_visible_proxy_s')),str(rising.get('frame','未出现')),str(lean.get('frame','未出现')),'未知'])
    body.append('<h3>UTK：只比较可见上升代理</h3>'+table(['片段','组','人工代理帧','首次 standing_up 帧','相对代理延迟 s','首次 rising 帧','首次 forward_lean 帧','物理离座'],utkrows))
    body.append('<p>s01 虽然在代理帧输出 standing_up，但没有出现 rising 阶段输出；动作命中不能代替阶段覆盖。以下误报和 unknown 仍按实验一的暂定区间计算，四项人工决定并未确认整段动作真值。</p>')
    body.append(table(['组','暂定已标区间 falling 误报段','未标区间 falling 采样','unknown 估计秒','暂定静坐中 forward_lean / 静坐采样'], [[mode,str(m['false_action_episodes']['falling']),str(m['unscored_falling_samples']),num(m['unknown_s']),f"{m['seated_forward_lean_samples']} / {m['seated_reference_samples']}"] for mode,m in timing['aggregates'].items()]))
    body.append('''<p>s10 的首次 forward_lean 早至 664 帧，处于暂定静坐内段；不能把这段提前输出作为有效起身预测。s10 的 790 代理与原有 798–818 暂定 rising 区间分开保存，未擅自改写阶段真值。</p><p class="muted">稳定起点 4.371–4.571 秒加上 0.5 秒持续条件，得到理论确认阈值 4.871–5.071 秒；按已有观测首次到达阈值约为 147–153 帧（4.871–5.072 秒）。这是派生的最早可核查范围，精确确认时刻仍未知，V7 未输出稳定倒地判定。</p></section>
<section id="latency"><h2>完整本机处理链与排队</h2><p>只为原始 fall01 补测缺失计时：A/B/C 各从第 1 帧处理到第 160 帧，CPU、模型预加载、顺序 FIFO、不丢帧，按发布源时钟安排帧到达。计时包含本地读取/解码、姿态、目标检测（B/C）、补检（C）、跟踪及 V7 输出；没有实际摄像头采集、网络或提醒设备送达。</p>''')
    benchmark_path=OUT/'benchmark/summary.json'
    if benchmark_path.exists():
        benchmark=json.loads(benchmark_path.read_text())
        benchrows=[]
        for g in benchmark['groups']:
            hit=g['raw_falling_trigger']
            benchrows.append([g['group'],num(hit['action_output_wall_offset_s']),num(hit['queue_wait_s']),num(hit['service_wall_s']),interval(hit['controlled_replay_margin_to_contact_s']),num(g['service']['median_s']),num(g['arrival_to_output']['p95_s']),str(g['max_pending_arrived_frames'])])
        body.append(table(['组','首次输出墙钟 s','触发帧排队 s','触发帧处理 s','对接地余量 s（负数=已晚）','逐帧处理中位 s','到达至输出 P95 s','最大等待帧数'],benchrows))
        body.append('<p><strong>本次本机 FIFO 补测三组首次 falling 输出均晚于接地区间。</strong>三个 160 帧运行的动作标签与阶段标签均与缓存逐帧一致；源时间余量与墙钟结果的差异来自计算和排队。A/B/C 顺序各测一次，数值受本机负载影响，不能用单次结果推断组间稳定的速度优劣。</p>')
        body.append('<p class="muted">模型加载、姿态/检测首帧预热在计时前；C 恢复模型已加载但没有推理预热。日志写入开销可进入下一帧排队。一次比较器错误中断的早期尝试单独保留在 benchmark/interrupted-attempt-1，未混入最终统计。接地余量也不含通知送达，实际提醒时间仍未知。全片 P95 包含事件后的积压，不等于首次触发延迟。</p>')
        body.append('<p>C 的恢复推理发生在 113–118、134–136 帧，共 9 帧，合计约 12.328 秒，主要影响后段长尾。87 帧首次触发之前没有执行恢复推理，不能把 C 较 B 首次输出更慢归因于这些恢复调用。</p>')
        if (OUT/'queue-latency.png').exists():
            body.append('<a href="queue-latency.png"><img class="plot" src="queue-latency.png" alt="A/B/C 本机 FIFO 到达至动作输出延迟曲线"></a>')
        body.append('<details><summary>查看完整计时摘要</summary><pre style="white-space:pre-wrap;overflow-wrap:anywhere">'+esc(json.dumps(benchmark,ensure_ascii=False,indent=2))+'</pre></details>')
    else:
        body.append('<p>计时补测进行中；完成后将更新本表。</p>')
    body.append('''<div class="note">单次本机回放用于发现速度瓶颈。样本很少且包含开发/训练重叠，不能推广为泛化准确率；这次也不训练未来 0.5/1 秒失稳预测模型。</div></section>
<section id="video"><h2>复用视频，直接核对时间点</h2><p>三列视频左 A、中 B、右 C，内容仍来自实验一冻结缓存。区间按钮用于跳转检查，播放器按 30 Hz 显示会有取整误差；精确数值以 JSON 源时间为准。</p>''')
    vids=[('fall01','URFD fall01 原始 A/B/C','urfd-fall-01-original',[(2.869,'首次 falling：87 帧'),(3.336,'接地下界：101 帧'),(3.537,'接地上界：107 帧'),(4.371,'稳定下界：132 帧'),(4.571,'稳定上界：138 帧')]),('s01','UTK s01 A/B/C','utk-s01_e01-transition',[(5.933333,'上升代理：722 帧')]),('s10','UTK s10 A/B/C','utk-s10_e01-transition',[(4.4,'过早前倾：664 帧'),(8.6,'上升代理：790 帧'),(8.866667,'起身输出：798 帧')])]
    for vid,title,cid,buttons in vids:
        body.append(f'<div class="card"><h3>{title}</h3><video id="{vid}" controls playsinline preload="metadata" src="../media/full/{cid}.mp4"></video><div>')
        body.extend(f'<button data-video="{vid}" data-seek="{t}">{label}</button>' for t,label in buttons)
        body.append(f'<button data-video="{vid}" data-play="1">播放 / 暂停</button><button data-video="{vid}" data-rate="0.25">0.25×</button><button data-video="{vid}" data-rate="1">正常速度</button></div></div>')
    body.append('''<p><a href="../human-review/#contact">接地逐帧与原始慢放</a> · <a href="../human-review/#stable">稳定倒地逐帧</a> · <a href="../human-review/#utk-s01-seat-off">s01 原始片段</a> · <a href="../human-review/#utk-s10-seat-off">s10 原始片段</a></p></section>
<section id="pending"><h2>仍待定的标注</h2><p>4/4 表示这轮问题已逐项回答，不表示八路视频已全部形成真值。以下均继续保留未知或暂定状态。</p>''')
    pending_zh=[['fall01 原始','接地/稳定起点的精确帧与稳定条件的精确确认时刻；可见起跌时刻；113–137 翻滚段、138–160 background/lying、步行边界和可见性。'],['fall01 低清','与原始视频的逐帧对应关系；接地与稳定标注仍未传播人工确认。同一原始事件只计一次。'],['UTK s01 / s10','物理离座、真实运动起点；前倾/上升/稳定站立的起止；宽动作区间、可见性与相对时钟假设。'],['fall02','起跌、接地、稳定、70–99 翻滚空白区及原始采集时钟；可见性和坐姿负例。'],['ADL04 / ADL07','继承动作边界、可见性及弯腰/坐姿内段；没有独立人体关键点真值。'],['MultiCam','真实时钟、人员进出/空场、多人身份、接地/稳定及解码损坏影响；只保留帧级，不计算秒级提前量。']]
    body.append(table(['输入','未决内容'],pending_zh))
    body.append('''<p><a href="pending-annotations.json">逐项机器可读清单</a> · <a href="pending-annotations.csv">清单 CSV</a></p></section>
<section id="errors"><h2>错误与限制归因</h2>''')
    errors=[{'id':'E1','finding':'源时序余量不能直接转为可用提醒提前量','cause':'模型输出所在视频时刻与实际完成处理时刻不同；FIFO 会积压。','evidence':'timing.json；benchmark/summary.json','next':'先以完整计时定位瓶颈，再评估可部署采样/计算方案。'}, {'id':'E2','finding':'s10 静坐阶段出现 forward_lean','cause':'阶段输出对坐姿变化敏感；首次前倾并不等于有效起身预测。','evidence':'源帧664；timing.json UTK diagnostics','next':'补复核静坐/前倾起止；本轮不调阈值。'}, {'id':'E3','finding':'s10 起身输出晚于可见上升代理8帧','cause':'可见证据与模型阶段切换不同步；不能据此归因为物理离座漏检。','evidence':'人工790；模型798','next':'保留代理延迟约0.267秒，物理离座仍未知。'}, {'id':'E4','finding':'fall01 倒地后局部观测及翻滚段仍影响输出','cause':'画面截断、低质量关键点、恢复开销、部分时段未标。','evidence':'实验一错误清单、timelines.json、C recovery计时','next':'分别记录观测恢复、动作可用性和时延；不以目标存在代替动作正确。'}, {'id':'E5','finding':'非跌倒片段与未标区间必须分开计数','cause':'已标注区间内无 falling 误报，不代表全视频无误报；C 的未标段 falling 更多。','evidence':'timing.json aggregates；实验一A/B/C主汇总未标段 falling采样6/6/11','next':'复核翻滚等空白区后再评估误报。'}, {'id':'E6','finding':'时间与标签覆盖不足以证明泛化预警','cause':'只有1个接地时间已复核的原始跌倒事件，2个UTK代理；开发样本存在训练重叠，MC真实时钟未知。','evidence':'protocol.json；pending-annotations.json','next':'未来预测实验继续延后；后续需独立事件与可靠时钟。'}]
    errors.append({'id':'E7','finding':'s01 起身动作命中，但 rising 阶段未出现','cause':'动作分类输出与阶段输出并不等价；命中 722 帧不能说明阶段识别完整。','evidence':'timing.json：s01 ABC first_standing_up=722，first_rising=null','next':'保留阶段覆盖缺口，待补充阶段边界复核后单独评估。'})
    (OUT/'error-attribution.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2)+'\n')
    body.append(table(['问题','归因','证据 / 后续'],[[esc(e['finding']),esc(e['cause']),esc(e['evidence'])+'<br>'+esc(e['next'])] for e in errors]))
    body.append('''</section><section id="files"><h2>可追溯结果文件</h2><p><a href="timing.csv">时间对照 CSV</a> · <a href="timing.json">完整时间结果 JSON</a> · <a href="review-summary.json">四项复核记录</a> · <a href="annotations.reviewed.snapshot.json">本次标注快照</a> · <a href="timelines.json">24 路逐帧时间线</a> · <a href="protocol.json">固定规则与输入哈希</a> · <a href="error-attribution.json">错误归因 JSON</a></p><p><a href="benchmark/metrics.csv">本机计时 CSV</a> · <a href="benchmark/summary.json">本机计时摘要</a> · <a href="benchmark/protocol.json">计时配置与边界</a> · <a href="validation.json">完整性验证</a></p></section>
<footer>原实验一标注和指标保持冻结；本页使用本轮人工复核后的独立快照。所有推理使用同一个 V7 权重，无训练、无阈值优化、无对外通知。</footer></main>
<script>document.querySelectorAll('button[data-video]').forEach(b=>b.onclick=()=>{const v=document.getElementById(b.dataset.video);if(b.dataset.seek){v.pause();v.currentTime=Number(b.dataset.seek)}if(b.dataset.rate)v.playbackRate=Number(b.dataset.rate);if(b.dataset.play){if(v.paused)v.play();else v.pause()}});</script></html>''')
    (OUT/'index.html').write_text(''.join(body))
    print(OUT/'index.html')


if __name__=='__main__':
    main()
