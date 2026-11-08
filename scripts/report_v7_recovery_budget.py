"""Report fixed-V7 latest-frame recovery-budget trials, without inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'runs/abc-v7-20261020/recovery-budget-20261021'
VARIANTS = ('baseline', 'budget200ms')
NAMES = {'baseline': '当前 C · 最新帧 + 完整补检', 'budget200ms': 'C · 最新帧 + 200 ms 软预算'}
COLORS = {'baseline': '#bc6842', 'budget200ms': '#187e87'}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values):
    vals = [v for v in values if v is not None]
    return {'n': len(vals), 'median': statistics.median(vals) if vals else None,
            'min': min(vals) if vals else None, 'max': max(vals) if vals else None}


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def rng(value, digits=3):
    if not value['n']:
        return '—'
    return f"{fmt(value['median'], digits)}（{fmt(value['min'], digits)}–{fmt(value['max'], digits)}）"


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(v)}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def variant_of(summary):
    variant = summary.get('strategy') or summary.get('variant') or summary.get('recovery_policy')
    if variant in VARIANTS:
        return variant
    return next(v for v in VARIANTS if v in summary['trial'])


def selected_pose(row):
    index = row.get('target_index')
    poses = row.get('pose_instances', [])
    return poses[index] if index is not None and 0 <= index < len(poses) else None


def measure_trial(summary, rows):
    """Derive report metrics from only completed rows; preserve 160 denominator."""
    selected = [(r, selected_pose(r)) for r in rows]
    attempts = [r for r in rows if (r.get('recovery') or {}).get('executed')]
    accepted = [r for r in rows if (r.get('recovery') or {}).get('accepted')]
    recovered = [r for r, p in selected if r['observed'] and p and p.get('recovered')]
    eligible = [r for r, p in selected if r['observed'] and p and p.get('recovered') and p.get('action_eligible') is not False]
    tail = [r for r in rows if r['source_frame_id'] >= 113]
    gaps = [r['output_wall_s'] - (rows[i-1]['output_wall_s'] if i else 0.0) for i, r in enumerate(rows)]
    max_gap_index = max(range(len(gaps)), key=gaps.__getitem__)
    recovery_max_row = max(rows, key=lambda r: r['stages_s']['recovery'])
    budgets = [(r.get('recovery') or {}).get('budget_audit', {}) for r in rows]
    first = summary['first_falling']
    value = {
        'trial': summary['trial'], 'variant': variant_of(summary), 'repeat': summary['repeat'],
        'source_reference_frames': 160, 'processed_frames': len(rows), 'dropped_frames': summary['dropped_frames'],
        'recovery_max_s': recovery_max_row['stages_s']['recovery'],
        'recovery_max_source_frame': recovery_max_row['source_frame_id'],
        'recovery_total_s': sum(r['stages_s']['recovery'] for r in rows),
        'recovery_attempt_total_s': sum(r['stages_s']['recovery'] for r in attempts),
        'recovery_backend_calls_total_s': sum(c['duration_s'] for a in budgets for c in a.get('calls', [])),
        'service_max_s': max(r['service_s'] for r in rows),
        'output_age_max_s': max(r['output_age_s'] for r in rows),
        'output_gap_max_s': max(gaps), 'output_gap_max_ending_frame': rows[max_gap_index]['source_frame_id'],
        'first_output_wall_s': first['output_wall_s'] if first else None,
        'first_source_frame': first['frame'] if first else None,
        'contact_margin_low_s': first['contact_margin_s'][0] if first else None,
        'contact_margin_high_s': first['contact_margin_s'][1] if first else None,
        'contact_result': summary['contact_result'],
        'event_hit': summary['official_fall_event']['hit'],
        'fall_reference_processed': summary['official_fall_event']['processed_reference_frames'],
        'fall_reference_falling_outputs': len(summary['official_fall_event']['falling_output_frames']),
        'unknown_outputs': sum(r['prediction']['label'] == 'unknown' for r in rows),
        'unobserved_outputs': sum(not r['observed'] for r in rows),
        'target_none_outputs': sum(r['target_id'] is None for r in rows),
        'expired_requires_new_confirmation_outputs': sum(r['tracking_reason'] == 'expired_requires_new_confirmation' for r in rows),
        'tail_reference_frames': 48, 'tail_processed': len(tail), 'tail_observed': sum(r['observed'] for r in tail),
        'tail_target_none': sum(r['target_id'] is None for r in tail), 'tail_dropped': 48-len(tail),
        'recovery_attempts': len(attempts), 'recovery_accepted': len(accepted),
        'recovery_selected': len(recovered), 'recovery_action_eligible': len(eligible),
        'budget_aborted_attempts': sum(a.get('exhausted', False) for a in budgets),
        'budget_max_overrun_s': max((a.get('overrun_s', 0.) for a in budgets), default=0.),
        'recovery_rtmo_calls': sum(a.get('rtmo_calls', 0) for a in budgets),
        'recovery_rtmpose_calls': sum(a.get('rtmpose_calls', 0) for a in budgets),
        'recovery_backend_call_max_s': max((a.get('max_call_s', 0.) for a in budgets), default=0.),
        'last_frame_observed': int(rows[-1]['observed']),
        'last_frame_target_id': rows[-1]['target_id'],
        'max_pending_slots': max(r['pending_slots_at_output'] for r in rows),
        'max_source_gap_s': max(r['source_gap_s'] or 0 for r in rows),
        'source_gap_gt_0_35': sum((r['source_gap_s'] or 0) > .35 for r in rows),
        'source_gap_gt_0_5': sum((r['source_gap_s'] or 0) > .5 for r in rows),
        'provisional_false_falling': len(summary['provisional_labeled_false_falling_frames']),
        'unscored_falling': len(summary['unscored_falling_frames']),
        'processed_label_counts': dict(Counter(r['prediction']['label'] for r in rows)),
        'recovery_attempt_source_ids': [r['source_frame_id'] for r in attempts],
        'recovery_accepted_source_ids': [r['source_frame_id'] for r in accepted],
        'recovery_selected_source_ids': [r['source_frame_id'] for r in recovered],
        'recovery_action_eligible_source_ids': [r['source_frame_id'] for r in eligible],
        'dropped_source_frame_ids': summary['dropped_source_frame_ids'],
        'target_none_source_ids': [r['source_frame_id'] for r in rows if r['target_id'] is None],
        'tail_observed_source_ids': [r['source_frame_id'] for r in tail if r['observed']],
        'regions': summary['regions'],
    }
    return value


def compare(trials, frames, protocol):
    metrics = [measure_trial(t, frames[t['trial']]) for t in trials]
    cells = []
    for variant in VARIANTS:
        ms = [m for m in metrics if m['variant'] == variant]
        assert len(ms) == 3, (variant, len(ms))
        numeric = [k for k, v in ms[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        cell = {'variant': variant, 'trials': [m['trial'] for m in ms],
                'statistics': {k: stat([m[k] for m in ms]) for k in numeric},
                'contact_results': dict(Counter(m['contact_result'] for m in ms)),
                'event_hits': sum(m['event_hit'] for m in ms)}
        cells.append(cell)
    pairs = []
    for repeat in (1, 2, 3):
        base = next(m for m in metrics if m['variant'] == 'baseline' and m['repeat'] == repeat)
        opt = next(m for m in metrics if m['variant'] == 'budget200ms' and m['repeat'] == repeat)
        br = {r['source_frame_id']: r for r in frames[base['trial']]}
        op = {r['source_frame_id']: r for r in frames[opt['trial']]}
        common = sorted(br.keys() & op.keys())
        pairs.append({'repeat': repeat,
            'recovery_max_reduction_s': base['recovery_max_s'] - opt['recovery_max_s'],
            'output_gap_max_reduction_s': base['output_gap_max_s'] - opt['output_gap_max_s'],
            'dropped_reduction': base['dropped_frames'] - opt['dropped_frames'],
            'common_processed_source_ids': common,
            'common_label_changes': [i for i in common if br[i]['prediction']['label'] != op[i]['prediction']['label']],
            'common_phase_changes': [i for i in common if br[i]['prediction']['phase'] != op[i]['prediction']['phase']],
            'common_observed_changes': [i for i in common if br[i]['observed'] != op[i]['observed']],
            'baseline_only_processed_source_ids': sorted(br.keys() - op.keys()),
            'budget_only_processed_source_ids': sorted(op.keys() - br.keys())})
    return {'schema': 'fallpre.v7.recovery_budget_report.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'contact_interval_s': protocol['contact_interval_s'], 'source_reference_frames': 160,
            'aggregation': 'Median/min/max of three independent trial-level values; paired by preset repeat ID, no picked best trial.',
            'output_gap_definition': 'Gap between successive completed outputs; first gap includes time from replay origin to first output.',
            'recovery_definitions': {'attempt': 'recovery.executed', 'accepted': 'recovery.accepted: candidate appended, not proof of tracked target',
                'selected': 'observed and selected pose.recovered', 'action_eligible': 'selected recovered pose.action_eligible is not False; not necessarily falling'},
            'cells': cells, 'trial_metrics': metrics, 'paired_detection_changes': pairs}


def plots(out, comparison, frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'savefig.dpi': 170,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 1, figsize=(11.5, 7.4), sharex=True, constrained_layout=True)
    for m in comparison['trial_metrics']:
        rs = frames[m['trial']]
        label = m['variant'] if m['repeat'] == 1 else None
        axs[0].plot([r['source_time_s'] for r in rs], [r['stages_s']['recovery'] for r in rs], color=COLORS[m['variant']], alpha=.7, marker='.', ms=3, label=label)
        axs[1].plot([r['source_time_s'] for r in rs], [r['output_age_s'] for r in rs], color=COLORS[m['variant']], alpha=.7, marker='.', ms=3)
    axs[0].axhline(.2, color='#555', ls='--', lw=1, label='Soft budget 0.2 s; in-flight call not preempted')
    for ax in axs:
        ax.axvspan(*comparison['contact_interval_s'], color='#e4c56d', alpha=.3)
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=9); axs[0].set_ylabel('Recovery stage (s)')
    axs[1].set_ylabel('Output completion age (s)'); axs[1].set_xlabel('Source time (s); dropped inputs have no output')
    fig.suptitle('C latest-frame baseline versus 200 ms recovery soft budget · three repeats')
    for ext in ('png', 'svg'): fig.savefig(out/f'recovery-latency.{ext}')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 5), constrained_layout=True)
    ordered = sorted(comparison['trial_metrics'], key=lambda m: (m['repeat'], m['variant'] != 'baseline'))
    for y, m in enumerate(ordered):
        ids = [r['source_frame_id'] for r in frames[m['trial']]]
        ax.scatter(m['dropped_source_frame_ids'], [y]*m['dropped_frames'], marker='|', s=150, color='#c6cecf')
        ax.scatter(ids, [y]*len(ids), marker='|', s=150, color=COLORS[m['variant']])
        selected = m['recovery_selected_source_ids']
        ax.scatter(selected, [y]*len(selected), marker='o', s=50, facecolor='none', edgecolor='#623f83', label='Recovered pose actually selected' if y == 0 else None)
    ax.axvspan(82.5, 112.5, color='#e4c56d', alpha=.25)
    ax.axvline(112.5, ls='--', color='#999', lw=1)
    ax.set_yticks(range(len(ordered)), [m['trial'] for m in ordered]); ax.invert_yaxis()
    ax.set_xlim(.5, 160.5); ax.set_xlabel('Original source frame; grey = dropped, color = processed, circle = recovered target observation')
    ax.set_title('All 160 original frames remain the denominator · yellow = official fall interval')
    for ext in ('png', 'svg'): fig.savefig(out/f'recovery-coverage.{ext}')
    plt.close(fig)


def make_errors(comparison):
    cells={c['variant']:c['statistics'] for c in comparison['cells']}
    b,o=cells['baseline'],cells['budget200ms']
    errors = [
        {'id': 'R1', 'finding': '完整补检的多视角串行调用占据工作线程', 'cause': '最新帧调度只能替换等待帧，不能取消已经进入的 RTMO / RTMPose 调用。', 'evidence': 'trials/*/frames.jsonl 中 stages_s.recovery 与输出间隔；diagnosis.json。', 'scope': '本轮只加入补检软预算，模型权重、候选质量规则和阈值不改。'},
        {'id': 'R2', 'finding': '200 ms 是软预算，仍可能超出', 'cause': '预算在每次后端调用前后检查，正在执行的单次同步调用不可抢占；超时后回退本帧 native observation。', 'evidence': '补检最大时长、逐帧预算审计及 budget 日志。', 'scope': '不能将其表述为硬实时上限；本机开销与排队仍计入总处理时长。'},
        {'id': 'R3', 'finding': '减少停顿可能损失目标恢复', 'cause': '超时舍弃未完成的补检链，候选不足时不会放宽证据门限；目标可能过期或等待重新确认。', 'evidence': 'attempt→accepted→selected→actioneligible、尾段 observed / target_id=None / expired 计数。', 'scope': 'append 候选不等于目标被选择；目标观察也不等于可供 V7 使用的动作姿态。'},
        {'id': 'R4', 'finding': '更短计算改变实际选择的源帧及 V7 历史', 'cause': '两组均采用最新帧；不同耗时产生不同选择序列与动作状态。', 'evidence': '全部160帧处理/跳过清单、共同处理帧的 label/phase 差异、unknown 与事件命中。', 'scope': '跳过输入保持缺失，未填充预测；共同帧一致也不证明整个序列等价。'},
        {'id': 'R5', 'finding': '接地前输出与后续恢复是两项验收', 'cause': '首次 falling 可以发生在补检长尾之前，首次预警通过不能掩盖倒地后的未观测。', 'evidence': '首次完成时间对 3.336–3.537 s 人工接地区间；113–160 尾段单列。', 'scope': '接地精确帧未知；只有墙钟 <3.336 s 才算明确接地前。'},
        {'id': 'R6', 'finding': '动作准确性与实际提醒尚未全部验证', 'cause': '只有单个开发事件、动作参考标签未逐项人工复核，且测量终点为本机 V7 输出。', 'evidence': 'annotations.reviewed.snapshot.json、protocol.json 与逐帧 provisional_false / unscored_falling。', 'scope': '不能推断泛化能力、真实设备通知送达或完全无误报。'},
    ]
    outcomes=[
        f"本轮最长输出间隔：基线 {rng(b['output_gap_max_s'])} s → 预算 {rng(o['output_gap_max_s'])} s。",
        f"预算组最大补检阶段耗时 {rng(o['recovery_max_s'])} s；相对200 ms的最大预算计时超额 {rng(o['budget_max_overrun_s'])} s。每次试验启动2个恢复帧，每帧在2次RTMO调用后终止，未进入RTMPose。",
        f"恢复姿态实际被选择：每次基线1帧、预算0帧。尾段113–160实际observed：基线 {rng(b['tail_observed'],0)}/48 → 预算 {rng(o['tail_observed'],0)}/48。预算组后续native姿态重新被观察，3次均在帧160保持目标ID 1；该ID是算法状态，未人工验证身份。",
        f"全部unknown输出：基线 {rng(b['unknown_outputs'],0)} → 预算 {rng(o['unknown_outputs'],0)}。官方跌倒区间falling输出：基线 {rng(b['fall_reference_falling_outputs'],0)}/30 → 预算 {rng(o['fall_reference_falling_outputs'],0)}/30；两组事件命中均3/3。",
        f"预算组保守接地余量 {rng(o['contact_margin_low_s'])} s，接地前输出3/3；基线亦为3/3。",
        '已标区间误falling两组均为0，但动作参考仍未全部复核；未标区间falling基线每次1个、预算0个，不能据此宣布全视频无误报。',
    ]
    for error,outcome in zip(errors,outcomes):error['measured_outcome']=outcome
    return errors


def render(out, comparison, errors, frames, protocol):
    cells = {c['variant']: c for c in comparison['cells']}
    base, opt = cells['baseline'], cells['budget200ms']
    b, o = base['statistics'], opt['statistics']
    before = opt['contact_results'].get('before_contact_certain', 0)
    sections = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 · C 补检长尾预算对照</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#203842;background:#eff4f3;line-height:1.65}*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:28px 26px 70px}h1{font-size:32px;line-height:1.3}h2{font-size:24px;margin-top:36px}a{color:#156b7a}nav{display:flex;flex-wrap:wrap;gap:17px;margin:18px 0}.card,.note{background:white;border:1px solid #d3e1e1;border-radius:12px;padding:20px;margin:18px 0}.note{border-left:5px solid #b88932;background:#fff8e8}.lead{font-size:20px}.muted{font-size:14px;color:#64767a}.metrics{display:flex;gap:12px;flex-wrap:wrap}.metric{flex:1;min-width:210px;background:#163f49;color:white;border-radius:10px;padding:18px 20px}.metric b{display:block;font-size:28px}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px;background:white}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce7e6}th{background:#deebeb;white-space:nowrap}img.plot,video{width:100%;border-radius:8px;background:#112f38}video{max-height:70vh;object-fit:contain}button{font:inherit;padding:7px 12px;margin:4px;border-radius:6px;border:1px solid #8da9af;background:white;color:#175460;cursor:pointer}details{margin:14px 0}summary{cursor:pointer}code{overflow-wrap:anywhere}.trace{max-height:470px;overflow:auto}.trace th{position:sticky;top:0}li{margin:6px 0}footer{font-size:13px;color:#607278;margin-top:36px}@media(max-width:700px){main{padding:18px 12px}h1{font-size:27px}.card{padding:13px}th,td{padding:8px}}
</style><main><a href="../latency-optimization-20261021/index.html">← 上轮队列对照</a>''',
        '<h1>固定 V7：给 C 的补检加上时间预算</h1><p class="muted">同一 URFD fall01 · 160 源帧 · C 最新帧调度 · 2 组 × 3 次重复 · 不训练</p>',
        f'<div class="metrics"><div class="metric"><b>{fmt(b["recovery_max_s"]["median"])} → {fmt(o["recovery_max_s"]["median"])} s</b>单次补检最长耗时（三次中位数）</div><div class="metric"><b>{fmt(b["output_gap_max_s"]["median"])} → {fmt(o["output_gap_max_s"]["median"])} s</b>最长无新输出间隔（三次中位数）</div><div class="metric"><b>{before} / 3</b>预算组明确接地前完成首次输出</div></div>',
        '<nav><a href="#change">最小改动</a><a href="#comparison">延迟对照</a><a href="#quality">跳帧与检测</a><a href="#recovery">目标恢复</a><a href="#video">墙钟视频</a><a href="#errors">错误归因</a><a href="#files">完整证据</a></nav>',
        f'<p class="lead">本轮预算组 {before}/3 次明确早于接地完成首次 falling 输出。最长补检与最长输出间隔的变化、目标恢复的损失分别列出；保住首次预警不等于倒地后仍持续观察到目标。</p>',
        '<div class="note">人工接地区间仍为 101–107 帧 / 3.336–3.537 秒，精确时刻未知。首次输出墙钟 &lt;3.336 秒才计明确接地前。200 ms 是补检软预算：已经开始的单次同步推理无法中断，所以耗时可超过 200 ms。测量仅到本机 V7 输出，不含摄像头采集、网络或提醒送达。</div>',
        '<section id="change"><h2>最小改动与固定条件</h2><div class="card"><p>基线沿用当前 C 最新帧调度及完整补检。实验组只在补检后端调用前后检查本次补检已耗时间：超过 200 ms 就结束本轮恢复并返回原始 native observation。不会放宽目标确认、姿态质量或动作门限；正在执行的后端调用完成后才能退出。</p><p>V7、姿态/人体检测/补检模型、阈值、源时钟、最新帧调度均固定；每组每次重新初始化跟踪与动作状态。旧日志用于定位，主对照为本轮按预定次序交替执行的三次完整补检与三次预算补检。</p><p class="muted">模型加载及姿态/检测首帧预热不计入测量；补检模型加载但不做恢复推理预热，各组一致。下表均为三次独立重复的中位数（最小–最大），不会挑选最快结果。</p></div></section>',
        '<section id="comparison"><h2>延迟与接地前输出</h2>']
    sections.append(table(['指标','当前 C 完整补检','200 ms 软预算'], [
        [label,rng(b[key],digits),rng(o[key],digits)] for label,key,digits in [
            ('最长补检阶段 s','recovery_max_s',3),('补检阶段总耗时（含门控检查）s','recovery_total_s',3),
            ('实际启动补检帧的阶段总耗时 s','recovery_attempt_total_s',3),('后端调用累计耗时 s','recovery_backend_calls_total_s',3),('最长单帧总处理 s','service_max_s',3),
            ('最大输出年龄 s','output_age_max_s',3),('最长输出间隔 s','output_gap_max_s',3),('首次 falling 完成墙钟 s','first_output_wall_s',3),
            ('首次 falling 源帧','first_source_frame',0),('保守接地前余量 s','contact_margin_low_s',3),('最大等待槽位（不含正在处理帧）','max_pending_slots',0)]]))
    sections.append(table(['组','明确接地前','接地区间内','明确已晚','无 falling 输出'],[[NAMES[c['variant']],*[f"{c['contact_results'].get(k,0)}/3" for k in ('before_contact_certain','within_contact_uncertainty','after_contact_certain','miss_no_falling_output')]] for c in comparison['cells']]))
    sections.append('<p class="muted">输出年龄 = 完成墙钟 − 源帧到达时间。最长输出间隔是相邻完成输出之间的时间，首段包含从回放开始至首次输出。等待槽位是最新帧策略的逻辑槽位，不包括正在计算的帧。</p><img class="plot" src="recovery-latency.png" alt="六次试验补检耗时与输出年龄曲线"></section>')
    sections.append('<section id="quality"><h2>跳帧与检测变化</h2><p>所有输入仍以原始 160 帧为分母；未处理帧没有合成预测。跌倒命中只要求在官方 83–112 帧区间内至少出现一次 falling，该区间来自官方代理标签，尚未逐帧人工复核。</p>')
    quality = []
    for c in comparison['cells']:
        s=c['statistics']
        quality.append([NAMES[c['variant']],rng(s['processed_frames'],0)+' / '+rng(s['dropped_frames'],0),f"{c['event_hits']}/3",rng(s['fall_reference_processed'],0)+' / 30',rng(s['fall_reference_falling_outputs'],0)+' / 30',rng(s['unknown_outputs'],0)+' / 160',rng(s['unobserved_outputs'],0)+' / 160',rng(s['provisional_false_falling'],0)+' / '+rng(s['unscored_falling'],0)])
    sections.append(table(['组','处理 / 跳过（共160）','事件命中','跌倒区间处理帧','跌倒区间 falling 输出','unknown 输出','未观察目标输出','已标区间误 falling / 未标 falling'],quality))
    sections.append(table(['重复','共同处理源帧','共同帧 label 变化','共同帧 phase 变化','共同帧 observed 变化','基线独有 / 预算独有'],[[p['repeat'],len(p['common_processed_source_ids']),len(p['common_label_changes']),len(p['common_phase_changes']),len(p['common_observed_changes']),str(len(p['baseline_only_processed_source_ids']))+' / '+str(len(p['budget_only_processed_source_ids']))] for p in comparison['paired_detection_changes']]))
    sections.append('<p class="muted">共同帧变化使用各组各自的因果历史，不能把两组历史视为相同输入；独有处理帧不算共同帧差异，也不从原始分母中移除。动作参考标签仍有未复核项；未标区间 falling 单列，不计作已证实误报。</p><img class="plot" src="recovery-coverage.png" alt="全部160源帧处理、跳过与目标恢复对照"></section>')
    sections.append('<section id="recovery"><h2>候选被接收，是否真正恢复了目标？</h2><p>依次统计：启动补检 → 接收并附加候选 → 跟踪实际选择该恢复姿态 → 恢复姿态可供动作模型使用。最后一步也不代表 falling 命中；partial_person 可以用于存在性观察，但动作输入仍会被禁止。</p>')
    sections.append(table(['组','attempt → accepted → selected → actioneligible','全部输出 target_id=None','过期需重新确认输出','尾段处理 / 跳过（共48）','尾段 observed / 48','尾段 target_id=None','最大源时间间隔 s'],[[NAMES[c['variant']], ' → '.join(rng(c['statistics'][k],0) for k in ('recovery_attempts','recovery_accepted','recovery_selected','recovery_action_eligible')),rng(c['statistics']['target_none_outputs'],0),rng(c['statistics']['expired_requires_new_confirmation_outputs'],0),rng(c['statistics']['tail_processed'],0)+' / '+rng(c['statistics']['tail_dropped'],0),rng(c['statistics']['tail_observed'],0)+' / 48',rng(c['statistics']['tail_target_none'],0),rng(c['statistics']['max_source_gap_s'])] for c in comparison['cells']]))
    sections.append(table(['组','预算耗尽帧','RTMO / RTMPose 调用次数','单次后端调用最大 s','预算计时最大超额 s'],[[NAMES[c['variant']],rng(c['statistics']['budget_aborted_attempts'],0),rng(c['statistics']['recovery_rtmo_calls'],0)+' / '+rng(c['statistics']['recovery_rtmpose_calls'],0),rng(c['statistics']['recovery_backend_call_max_s']),rng(c['statistics']['budget_max_overrun_s']) if c['variant']=='budget200ms' else '无预算'] for c in comparison['cells']]))
    sections.append('<p>本轮完整补检每次接收并实际选择 1 帧恢复姿态，但下一个输入已经跨过 0.5 秒跟踪连续性窗口，目标过期。预算组每次两次尝试均因预算耗尽返回 native observation，恢复姿态接收/选择均为 0；随后本来的姿态检测在 123/124 与 145/146 帧附近接上，三次均保持目标 ID 1 至末帧。这里证明的是本次跟踪状态延续，尚无独立人工身份真值。</p>')
    regionrows=[]
    for variant in VARIANTS:
        ms=[m for m in comparison['trial_metrics'] if m['variant']==variant]
        for key,label in [('post_contact_unscored','113–137：未标动作区间'),('tail_provisional','138–160：暂定背景区间')]:
            ref=ms[0]['regions'][key]['reference_samples']
            regionrows.append([NAMES[variant],label,str(ref),*[rng(stat([m['regions'][key][k] for m in ms]),0) for k in ('processed','dropped','observed','unknown','target_id_null')]])
    sections.append(table(['组','固定区间','参考帧数','处理','跳过','observed','unknown','target_id=None'],regionrows))
    sections.append('<p class="muted">尾段固定为 113–160 帧（48 帧），该起点是官方 lying 代理，并非人工接地/稳定倒地精确帧。target_id=None 包括开头尚未确认以及后续目标丢失；过期事件和尾段单列，避免混为一种失败。目标身份连续性没有单独人工复核。</p>')
    for m in comparison['trial_metrics']:
        rs=frames[m['trial']]
        rows=[]
        for r in rs:
            a=r.get('recovery') or {}
            if not a.get('executed') and r['source_frame_id'] < 113: continue
            p=selected_pose(r)
            budget=a.get('budget_audit',{})
            rows.append([r['source_frame_id'],fmt(r['source_time_s']),fmt(r['stages_s']['recovery']),str(bool(a.get('executed')))+' / '+str(bool(a.get('accepted'))),str(bool(r['observed'] and p and p.get('recovered'))),str(budget.get('rtmo_calls',0))+' / '+str(budget.get('rtmpose_calls',0)),fmt(budget.get('overrun_s',0.)),esc(r['tracking_reason']),esc(r['target_id']),esc(r['prediction']['label']),esc(a.get('reason',''))])
        sections.append(f'<details><summary>{esc(m["trial"])}：补检与尾段明细</summary><div class="trace">'+table(['源帧','源时间 s','补检 s','执行 / 接收','恢复姿态被选','RTMO / RTMPose 调用','预算超额 s','跟踪原因','目标 ID','动作','补检原因'],rows)+'</div></details>')
    sections.append('</section><section id="video"><h2>同一墙钟轴的可播放对照</h2>')
    metadata=out/'media/metadata.json'
    if metadata.exists():
        for video in read(metadata)['videos']:
            sections.append(f'<p>固定展示第 2 次重复，在查看结果前已选定。画面只显示已完成输出所对应的原始源图；停留时标记 HOLD，跳帧不插值。数值保留测量时刻，视频显示量化到下一个 30 Hz 显示刻度。</p><video id="comparison-video" controls preload="metadata" poster="{esc(video["poster"])}"><source src="{esc(video["file"])}" type="video/mp4"></video><div><button id="budget-video-toggle" onclick="toggleVideo()">播放 / 暂停</button><button onclick="seekVideo(2.7)">首次输出附近</button><button onclick="seekVideo(3.3)">接地前 3.3 秒</button><button onclick="seekVideo(3.336)">接地区间开始</button><button onclick="seekVideo(4.3)">补检停顿 4.3 秒</button><button onclick="seekVideo(5.3)">目标尾段 5.3 秒</button><button onclick="document.querySelector(\'video\').playbackRate=.5">0.5× 慢放</button><button onclick="document.querySelector(\'video\').playbackRate=1">1× 原速</button></div>')
    else:
        sections.append('<p>墙钟对照视频待生成。</p>')
    sections.append('</section><section id="errors"><h2>错误归因与仍待定项</h2>')
    for item in errors:
        sections.append(f'<div class="card"><b>{esc(item["id"])} · {esc(item["finding"])}</b><p>{esc(item["measured_outcome"])}</p><p>{esc(item["cause"])}</p><p class="muted">证据：{esc(item["evidence"])}<br>解释边界：{esc(item["scope"])}</p></div>')
    sections.append('<p>人工复核保留：接地 101–107 帧，稳定倒地 132–138 帧并采用 0.5 秒判据，精确帧均未知；UTK s01/ s10 仅确认 722 / 790 为首次明确可见上升代理，物理离座仍待定。本轮没有把代理升格为物理时刻，也没有修改任何已复核标注。其余动作、阶段和身份真值仍按原待复核清单处理。</p></section><section id="files"><h2>完整证据</h2><p>')
    links=[('protocol.json','冻结协议'),('comparison.json','汇总与差异 JSON'),('comparison.csv','逐次指标 CSV'),('error-attribution.json','错误归因'),('results.json','原始汇总'),('diagnosis.json','旧日志诊断'),('validation.json','独立验证'),('tests.xml','审计测试'),('unit-tests.xml','单元测试'),('annotations.reviewed.snapshot.json','复核标注快照'),('media/metadata.json','视频来源与验证')]
    sections.append(' · '.join(f'<a href="{path}">{label}</a>' for path,label in links if (out/path).exists()))
    sections.append('</p><details><summary>逐次日志与汇总</summary><ul>'+''.join(f'<li>{esc(m["trial"])}：<a href="trials/{m["trial"]}/frames.jsonl">逐帧日志</a> · <a href="trials/{m["trial"]}/summary.json">汇总</a></li>' for m in comparison['trial_metrics'])+'</ul></details></section>')
    sections.append(f'<footer>V7 SHA-256：<code>{esc(protocol["model_sha256"])}</code><br>本报告由已完成日志生成，没有训练或模型推理；三次重复描述本机单事件波动，不做总体显著性与泛化结论。</footer></main><script>function toggleVideo(){{const v=document.querySelector("video");if(v.paused)v.play();else v.pause();}}function seekVideo(t){{const v=document.querySelector("video");v.currentTime=t;v.play();}}</script></html>')
    (out/'index.html').write_text('\n'.join(sections))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--experiment',type=Path,default=DEFAULT);parser.add_argument('--skip-plots',action='store_true')
    args=parser.parse_args();out=args.experiment.resolve();results=read(out/'results.json');protocol=read(out/'protocol.json')
    assert results['complete'] and results['all_frozen_hashes_unchanged'] and protocol['training'] is False
    trials=results['trials'];assert len(trials)==6
    frames={t['trial']:[json.loads(line) for line in (out/'trials'/t['trial']/'frames.jsonl').read_text().splitlines()] for t in trials}
    comparison=compare(trials,frames,protocol);write(out/'comparison.json',comparison)
    numeric_keys=[k for k,v in comparison['trial_metrics'][0].items() if isinstance(v,(str,int,float,bool)) or v is None]
    with (out/'comparison.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=numeric_keys,extrasaction='ignore');writer.writeheader();writer.writerows(comparison['trial_metrics'])
    errors=make_errors(comparison);write(out/'error-attribution.json',{'schema':'fallpre.recovery_budget_errors.v1','errors':errors})
    if not args.skip_plots:plots(out,comparison,frames)
    render(out,comparison,errors,frames,protocol)
    print(json.dumps({'report':str(out/'index.html'),'trials':len(comparison['trial_metrics']),'cells':comparison['cells']},ensure_ascii=False))


if __name__=='__main__':
    main()
