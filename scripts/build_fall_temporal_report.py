"""Render the completed fall-evidence experiment without modifying its results."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/stgcn-fall-20261107'


def main():
    summary = json.loads((OUT / 'B/summary.json').read_text())
    rows = []
    for folder, title in [('head-c-st', 'C：ST 特征'),
                          ('head-c-common', 'C：共同样本对照'),
                          ('head-d-geometry', 'D：ST＋整体运动与姿态')]:
        path = OUT / 'official-dev' / folder / 'report.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        metrics = report['pooled_oof_metrics']
        recall = [m['per_class']['fall_process']['recall'] for m in metrics]
        false_positive = [m['confusion_matrix'][0][1] for m in metrics]
        negatives = report['class_counts']['non_fall_activity']
        rows.append(f'<tr><td>{title}</td><td>{report["samples"]}</td>'
                    f'<td>{min(recall):.1%}–{max(recall):.1%}</td>'
                    f'<td>{min(false_positive)}–{max(false_positive)} / {negatives}</td></tr>')
    html = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:,">
<title>跌倒事件时序证据审计</title>
<style>
:root{color-scheme:dark}*{box-sizing:border-box}body{max-width:1180px;margin:24px auto;padding:20px;background:#112027;color:#e0ede7;font:16px/1.6 system-ui}a{color:#9ed6bf}h1{line-height:1.3}h2{font-size:19px}video{width:100%;max-height:65vh;background:#080e12;border-radius:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 ui-monospace,monospace;margin:0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{padding:18px;background:#21343b;border-radius:8px;margin:16px 0}select{font:inherit;padding:10px;margin:8px 0;width:100%}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:10px;border-bottom:1px solid #41606a}.muted{color:#bed0cb}.tag{font-size:13px;color:#dfc98f}details>summary{cursor:pointer}#b{max-height:700px;overflow:auto}@media(max-width:700px){.grid{grid-template-columns:1fr}body{padding:12px}table{font-size:13px}th,td{padding:6px}}
</style>
<p class="tag">2026-11-07 · 离线开发实验 / 实时旁路 · 尚未接入主报警</p>
<h1>让时序证据服务于跌倒事件识别</h1>
<p>目标是减少漏检、误报与报警延迟。基础动作作为诊断；观测不足、模型判断和报警事件分别记录。</p>
<p><a href="B/metrics.json">逐片指标与审计记录</a> · <a href="http://127.0.0.1:8810/">实时证据旁路</a> · <a href="http://127.0.0.1:8780/">原实时基线</a></p>
<div class="card"><h2>本轮结论</h2>
<p>10 段旧开发视频、4598 次观测，原有 14 次倒地演示事件结果保持不变。B 产生 6 次支持观测、83 次冲突观测。3 个事件的支持证据比原报警源帧早约 0.77–1.05 秒出现；这不是已验证的报警提速。</p>
<p>原判断观测覆盖率 PRIMARY_COVERAGE；时序证据可用覆盖率 TEMPORAL_COVERAGE。正式正常监测小时为 0，真实报警输出时间未记录，因此正式每小时误报率与输出延迟留空。</p>
<p class="muted">两条分支共享 YOLO 骨架；一致不能视为两份独立保证。高分无法越过目标身份、当前观测、下降与倒卧证据的约束。</p></div>
<div class="card"><h2>小型跌倒过程判别头：实际训练结果</h2>
<table><thead><tr><th>方案</th><th>有效窗口</th><th>跌倒过程窗口召回</th><th>负窗口误判</th></tr></thead><tbody>HEAD_ROWS</tbody></table>
<p>按原视频分组留一交叉验证，固定 3 个随机种子。使用已有 URFD 官方深度帧状态向 RGB 帧序号转移的弱标签；未独立确认 RGB 对齐。人物与机位未独立隔离，不能视为迁移验证。</p>
<p>30 个模拟跌倒序列仅 13 个有可用的跌倒过程窗口；其余 17 个仍在覆盖分母中。D 在共同样本上没有显示收益：误判更多，召回相当或下降，因此本轮不采用该判别头接入主报警。C/D 都只保存离线权重。</p></div>
<h2>A 原报警与 B 共享骨架时序证据回放</h2>
<label for="clip">选择开发片段</label><select id="clip"></select>
<video id="video" controls></video>
<div class="grid"><div class="card"><h2>A：原报警与事件管理</h2><pre id="a">选择时点查看</pre></div>
<div class="card"><h2>B：时序证据（共享骨架）</h2><pre id="b">选择时点查看</pre></div></div>
<p class="muted">视频标注框是 A 的结果，B 不生成报警。下一步需要人工复核跌倒过程与主动躺下边界、连续正常监测时间，以及新人物、新机位的独立视频。</p>
<script>
const $=id=>document.getElementById(id);let rows=[],generation=0;
async function load(){const n=++generation,c=$('clip').value;rows=[];$('a').textContent=$('b').textContent='正在加载';const response=await fetch('B/'+c+'.jsonl');if(!response.ok)throw Error('回放数据不可用');const text=await response.text();if(n!==generation)return;rows=text.trim().split('\n').map(JSON.parse);$('video').src='media/'+c+'.mp4';paint()}
function paint(){let r=null,t=$('video').currentTime;for(const x of rows){if(x.time_s>t)break;r=x}if(!r||t-r.time_s>.45){$('a').textContent=$('b').textContent='当前无返回观测';return}$('a').textContent=JSON.stringify({time_s:r.time_s,fall:r.fall,observable:r.primary_observable},null,2);$('b').textContent=JSON.stringify(r.temporal_evidence,null,2)}
fetch('clips.json').then(x=>x.json()).then(clips=>{for(const c of clips){const o=document.createElement('option');o.value=c.id;o.textContent=c.title;$('clip').append(o)}return load()}).catch(e=>{$('a').textContent=e.message});$('clip').onchange=()=>load().catch(e=>{$('a').textContent=e.message});$('video').ontimeupdate=paint;
</script></html>'''
    html = html.replace('PRIMARY_COVERAGE', f'{summary["primary_observable_seconds"] / summary["recording_seconds"]:.1%}')
    html = html.replace('TEMPORAL_COVERAGE', f'{summary["temporal_observable_seconds"] / summary["recording_seconds"]:.1%}')
    html = html.replace('HEAD_ROWS', ''.join(rows))
    # The JavaScript parser needs a literal backslash in the newline delimiter.
    html = html.replace("split('\n')", "split('\\n')")
    (OUT / 'index.html').write_text(html)


if __name__ == '__main__':
    main()
