"""Render matched fixed-V7 FIFO/latest timing experiments without model imports."""
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
BASE = ROOT / 'runs/abc-v7-20261020'
OUT = BASE / 'latency-optimization-20261021'


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stat(values):
    vs = [v for v in values if v is not None]
    return {'n': len(vs), 'median': statistics.median(vs) if vs else None,
            'min': min(vs) if vs else None, 'max': max(vs) if vs else None}


def number(v, digits=3):
    return '—' if v is None else f'{v:.{digits}f}'


def rng(s, digits=3):
    if not s or not s['n']:
        return '无输出'
    if s['min'] == s['max']:
        return number(s['median'], digits)
    return f"{number(s['median'], digits)}（{number(s['min'], digits)}–{number(s['max'], digits)}）"


def span(s, digits=0):
    if s['min'] == s['max']:
        return number(s['min'], digits)
    return f"{number(s['min'], digits)}–{number(s['max'], digits)}"


def esc(x):
    return html.escape(str(x))


def table(head, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in head) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</tbody></table></div>'


def aggregate(trials, frames, bounds):
    cells = []
    for mode in 'ABC':
        for policy in ('fifo', 'latest'):
            ts = sorted((t for t in trials if t['mode'] == mode and t['policy'] == policy), key=lambda t: t['repeat'])
            assert len(ts) == 3, (mode, policy, len(ts))
            firsts = [t['first_falling'] for t in ts]
            cell = {'mode': mode, 'policy': policy, 'trials': [t['trial'] for t in ts],
                    'repeats': len(ts), 'contact_results': dict(Counter(t['contact_result'] for t in ts)),
                    'first_output_wall_s': stat([f['output_wall_s'] if f else None for f in firsts]),
                    'first_source_frame': stat([f['frame'] if f else None for f in firsts]),
                    'first_source_time_s': stat([f['source_time_s'] if f else None for f in firsts]),
                    'first_queue_wait_s': stat([f['queue_wait_s'] if f else None for f in firsts]),
                    'first_service_s': stat([f['service_s'] if f else None for f in firsts]),
                    'first_output_age_s': stat([f['output_age_s'] if f else None for f in firsts]),
                    'contact_margin_low_s': stat([f['contact_margin_s'][0] if f else None for f in firsts]),
                    'contact_margin_high_s': stat([f['contact_margin_s'][1] if f else None for f in firsts]),
                    'event_hit_repeats': sum(t['official_fall_event']['hit'] for t in ts),
                    'full_prediction_equality_repeats': sum(t['full_prediction_equality_count'] == 160 and t['processed_frames'] == 160 for t in ts) if policy == 'fifo' else None,
                    'quality_denominator': 160}
            simple = ['processed_frames', 'dropped_frames', 'max_pending_slots', 'max_source_gap_s',
                      'source_gaps_gt_0_35_s', 'source_gaps_gt_0_5_s', 'outputs_age_gt_0_35_s',
                      'outputs_age_gt_0_5_s', 'unknown_processed_frames', 'unobserved_processed_frames',
                      'epoch_transitions_among_observed', 'last_output_wall_s']
            for key in simple:
                cell[key] = stat([t[key] for t in ts])
            for metric in ('service', 'queue_wait', 'output_age'):
                cell[metric] = {k: stat([t[metric][k] for t in ts]) for k in ('median_s', 'p95_s', 'max_s')}
            cell['coverage'] = {k: stat([t['coverage'][k] for t in ts]) for k in ts[0]['coverage'] if k != 'note'}
            cell['processed_fall_reference_frames'] = stat([t['official_fall_event']['processed_reference_frames'] for t in ts])
            cell['falling_output_reference_frames'] = stat([len(t['official_fall_event']['falling_output_frames']) for t in ts])
            for key in ('provisional_labeled_false_falling_frames', 'unscored_falling_frames', 'same_source_fifo_label_changes', 'same_source_fifo_phase_changes', 'recovery_executed_frames'):
                cell[key + '_count'] = stat([len(t[key]) for t in ts])
            cell['recovery_total_s'] = stat([sum(r['stages_s']['recovery'] for r in frames[t['trial']]) for t in ts])
            cell['stages_before_first_falling'] = {}
            for stage in ('decode', 'pose', 'detector', 'recovery', 'tracking', 'action'):
                cell['stages_before_first_falling'][stage] = stat([
                    sum(r['stages_s'][stage] for r in frames[t['trial']] if r['source_frame_id'] <= t['first_falling']['frame'])
                    for t in ts if t['first_falling']])
            cells.append(cell)
    contrasts = []
    for mode in 'ABC':
        pairs = []
        for repeat in (1, 2, 3):
            fifo = next(t for t in trials if (t['mode'], t['policy'], t['repeat']) == (mode, 'fifo', repeat))
            latest = next(t for t in trials if (t['mode'], t['policy'], t['repeat']) == (mode, 'latest', repeat))
            pairs.append({'repeat': repeat,
                          'first_output_reduction_s': fifo['first_falling']['output_wall_s'] - latest['first_falling']['output_wall_s'] if fifo['first_falling'] and latest['first_falling'] else None,
                          'max_pending_reduction': fifo['max_pending_slots'] - latest['max_pending_slots'],
                          'output_age_p95_reduction_s': fifo['output_age']['p95_s'] - latest['output_age']['p95_s']})
        contrasts.append({'mode': mode, 'paired_by_repeat': pairs,
                          'first_output_reduction_s': stat([p['first_output_reduction_s'] for p in pairs])})
    return {'schema': 'fallpre.v7.queue_policy_comparison.v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'contact_interval_s': bounds, 'aggregation': 'Each cell reports median/min/max of three trial-level quantities. P95 is calculated within each trial, never from pooled unequal sparse frames.',
            'cells': cells, 'contrasts': contrasts}


def plots(out, trials, frames, bounds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'figure.dpi': 120, 'savefig.dpi': 170,
                         'axes.spines.top': False, 'axes.spines.right': False})
    colors = {'fifo': '#bd603c', 'latest': '#157e89'}
    fig, axs = plt.subplots(3, 1, figsize=(11.5, 9), sharex=True, constrained_layout=True)
    for ax, mode in zip(axs, 'ABC'):
        for t in trials:
            if t['mode'] != mode:
                continue
            rs = frames[t['trial']]
            ax.plot([r['source_time_s'] for r in rs], [r['output_age_s'] for r in rs],
                    color=colors[t['policy']], alpha=.65, lw=1.5, marker='.', ms=2,
                    label=t['policy'].upper() if t['repeat'] == 1 else None)
        ax.axvspan(*bounds, color='#eacb7c', alpha=.35, label='Reviewed contact interval' if mode == 'A' else None)
        ax.set_ylabel(f'{mode} output age (s)')
        ax.grid(alpha=.18)
    axs[0].legend(loc='upper left', ncol=3)
    axs[-1].set_xlabel('Source time (s); only actually processed inputs are plotted')
    fig.suptitle('Output age = completion wall time - source arrival time\nThree repeats per policy; curves do not imply results on dropped frames')
    fig.savefig(out/'queue-output-age.png'); fig.savefig(out/'queue-output-age.svg'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 5), constrained_layout=True)
    for mi, mode in enumerate('ABC'):
        for pi, policy in enumerate(('fifo', 'latest')):
            ys = mi*2+pi
            ts = [t for t in trials if t['mode'] == mode and t['policy'] == policy]
            xs = [t['first_falling']['output_wall_s'] for t in ts if t['first_falling']]
            for i, x in enumerate(xs):
                ax.scatter(x, ys+(i-1)*.12, s=50, color=colors[policy], zorder=3)
            if xs:
                ax.plot([min(xs), max(xs)], [ys, ys], color=colors[policy], lw=2)
            if len(xs) < 3:
                ax.text(0, ys, f'{3-len(xs)} missed triggers', va='center')
    ax.axvspan(*bounds, color='#eacb7c', alpha=.55, label='Contact 3.336–3.537 s')
    ax.axvline(bounds[0], color='#9a7008', ls='--', lw=1)
    ax.set_yticks(range(6), [f'{m} {p.upper()}' for m in 'ABC' for p in ('fifo', 'latest')])
    ax.invert_yaxis(); ax.set_xlabel('First falling output completion, wall offset (s)')
    ax.grid(axis='x', alpha=.2); ax.legend(loc='lower right')
    ax.set_title('First output versus reviewed contact boundary · all 18 trials')
    fig.savefig(out/'first-output-deadline.png'); fig.savefig(out/'first-output-deadline.svg'); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11.5, 7), constrained_layout=True)
    ordered = sorted(trials, key=lambda t: (t['mode'], t['policy'] == 'latest', t['repeat']))
    for y, t in enumerate(ordered):
        rs = frames[t['trial']]
        dropped = t['dropped_source_frame_ids']
        if dropped:
            ax.scatter(dropped, [y]*len(dropped), marker='|', color='#c7cdd0', s=85)
        ax.scatter([r['source_frame_id'] for r in rs], [y]*len(rs), marker='|', color=colors[t['policy']], s=85)
    ax.axvspan(83-.5, 112+.5, color='#eacb7c', alpha=.25)
    ax.set_yticks(range(18), [t['trial'] for t in ordered]); ax.invert_yaxis()
    ax.set_xlim(.5, 160.5); ax.set_xlabel('Original source frame (1–160); grey = dropped, color = processed')
    ax.set_title('Full source coverage retained as denominator · yellow = official falling 83–112')
    fig.savefig(out/'source-coverage.png'); fig.savefig(out/'source-coverage.svg'); plt.close(fig)


def make_errors(comparison):
    return [
        {'id': 'Q1', 'finding': 'FIFO 等待导致首次输出错过接地', 'cause': '完整处理速度低于约 30 Hz 到达速度，已经过时的等待帧继续占用计算。', 'evidence': '旧计时 diagnosis.json；本次三次重复的 FIFO 队列等待与输出时刻。', 'next': '采用显式最新到达帧调度，保持源帧号与源时间；不要把跳过的输入填成已处理。'},
        {'id': 'Q2', 'finding': '只保留一个等待帧不保证每次输出都新鲜', 'cause': '正在处理的推理无法被队列替换；C 的长恢复调用仍可能产生过时输出及大源时间间隔。', 'evidence': 'C latest service/output_age 最大值、recovery 逐帧日志及 reset gap 计数。', 'next': '下一轮单独评估恢复调用预算或独立执行；本轮未改恢复门限或算法。'},
        {'id': 'Q3', 'finding': '跳帧改变 V7 与跟踪的时间历史', 'cause': '输入序列变稀，阶段/动作、目标身份和 unknown 都可能改变；不能复用原序列的最终预测作为优化结果。', 'evidence': '最新策略重新执行模型；逐帧 source_gap、unknown、label/phase changes 和原始 160 帧覆盖表。', 'next': '以原始分母记录缺失和未观测，未来增加独立事件，特别是短跌倒和遮挡。'},
        {'id': 'Q4', 'finding': '一次事件命中不等于逐帧检测保持一致', 'cause': '命中只要求在官方 83–112 帧区间内至少输出一次 falling；丢弃帧没有模型输出。', 'evidence': 'official_fall_event 与 coverage 字段；各重复的命中帧明细。', 'next': '保留事件命中、覆盖、unknown、未标区间输出四个独立指标。'},
        {'id': 'Q5', 'finding': '对接地的墙钟余量有明确测量边界', 'cause': '本机源时钟调度从已发布视频帧到达开始；没有真实摄像头、网络、通知设备延迟。接地精确帧仍未知。', 'evidence': 'protocol.json；人工接地区间 101–107 帧 / 3.336–3.537 秒。', 'next': '两端均为正才记为本机回放明确接地前；后续需实际设备链路验证。'},
        {'id': 'Q6', 'finding': '历史计时不能单独充当优化对照', 'cause': '环境负载与重复间波动会变化；本次重新测同配置 FIFO 作为主要对照。', 'evidence': '18 次交替运行；每组策略 3 次；旧日志只用于定位瓶颈。', 'next': '保留全部重复及失败，不挑选最快一次；单事件开发/训练重叠，不推断泛化。'},
    ]


def render(out, comparison, trials, frames, protocol, diagnosis):
    cells = comparison['cells']
    latest = [c for c in cells if c['policy'] == 'latest']
    before = sum(c['contact_results'].get('before_contact_certain', 0) for c in latest)
    fifo_before = sum(c['contact_results'].get('before_contact_certain', 0) for c in cells if c['policy'] == 'fifo')
    misses = sum(c['contact_results'].get('miss_no_falling_output', 0) for c in cells)
    colors = {'fifo': 'FIFO 完整顺序', 'latest': '最新到达帧'}
    mainrows = []
    latencyrows = []
    qualityrows = []
    gaprows = []
    for c in cells:
        name = c['mode'] + ' · ' + colors[c['policy']]
        result = c['contact_results']
        mainrows.append([name, rng(c['first_output_wall_s']), span(c['first_source_frame']),
                        f"{result.get('before_contact_certain', 0)}/3", f"{result.get('within_contact_uncertainty', 0)}/3", f"{result.get('after_contact_certain', 0)}/3", f"{result.get('miss_no_falling_output', 0)}/3",
                        span(c['max_pending_slots']), span(c['processed_frames'])+' / '+span(c['dropped_frames'])])
        latencyrows.append([name, rng(c['queue_wait']['median_s'])+' / '+rng(c['queue_wait']['p95_s']),
                            rng(c['service']['median_s'])+' / '+rng(c['service']['p95_s']),
                            rng(c['output_age']['median_s'])+' / '+rng(c['output_age']['p95_s']), span(c['output_age']['max_s'], 3),
                            rng(c['first_queue_wait_s'])+' / '+rng(c['first_service_s'])])
        qualityrows.append([name, f"{c['event_hit_repeats']}/3", span(c['processed_fall_reference_frames'])+' / 30',
                            span(c['falling_output_reference_frames'])+' / 30', span(c['unknown_processed_frames'])+' / 160',
                            span(c['unobserved_processed_frames'])+' / 160', span(c['coverage']['matched_processed_labeled_samples'])+' / '+span(c['coverage']['all_labeled_source_samples']),
                            span(c['provisional_labeled_false_falling_frames_count'])+' / '+span(c['unscored_falling_frames_count'])])
        gaprows.append([name, span(c['max_source_gap_s'], 3), span(c['source_gaps_gt_0_35_s'])+' / '+span(c['source_gaps_gt_0_5_s']),
                        span(c['epoch_transitions_among_observed']), span(c['recovery_executed_frames_count']), span(c['recovery_total_s'],3),
                        span(c['same_source_fifo_label_changes_count'])+' / '+span(c['same_source_fifo_phase_changes_count'])])
    title = '固定 V7：减少等待帧，能否接地前输出？'
    body = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>V7 · 耗时与队列优化对照</title><style>
:root{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;color:#203741;background:#eff4f4;line-height:1.65}*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:28px 26px 70px}h1{font-size:32px;line-height:1.3;margin:16px 0}h2{font-size:24px;margin-top:34px}h3{font-size:18px}p{margin:12px 0}a{color:#176b7c}nav{display:flex;gap:18px;flex-wrap:wrap;padding:12px 0}.card,.note{background:#fff;border:1px solid #d6e2e5;border-radius:12px;padding:20px;margin:16px 0}.note{border-left:5px solid #bc8a29;background:#fff8e7}.lead{font-size:20px}.muted{font-size:14px;color:#61737b}.metrics{display:flex;gap:14px;flex-wrap:wrap}.metric{background:#163f4a;color:white;border-radius:10px;padding:18px 21px;flex:1;min-width:200px}.metric b{display:block;font-size:29px}table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th,td{text-align:left;vertical-align:top;padding:11px 13px;border-bottom:1px solid #dce6e8}th{background:#e0ebed;white-space:nowrap}.scroll{overflow-x:auto}img.plot{width:100%;background:white;border-radius:8px}video{width:100%;border-radius:8px;background:#132e38}button,select{font:inherit;padding:7px 11px;border:1px solid #8daab4;border-radius:6px;background:white;color:#175164;margin:5px 5px 5px 0;cursor:pointer}code{overflow-wrap:anywhere;font-size:.9em}details{margin:14px 0}summary{cursor:pointer}footer{margin-top:35px;font-size:13px;color:#536b74}.bad{color:#a13f2e}.good{color:#156c55}.trace{max-height:510px;overflow:auto}.trace th{position:sticky;top:0}.small{font-size:12px}li{margin:7px 0}@media(max-width:700px){main{padding:18px 12px}h1{font-size:27px}.card{padding:14px}th,td{padding:8px}.metric{min-width:140px}}
</style><main><a href="../experiment3-20261021/index.html">← 实验三时间审计</a>''',
            f'<h1>{title}</h1><p class="muted">2026-10-21 · 同一原始 fall01，160 源帧 · A/B/C × 2 种调度 × 3 次重复 · 固定 V7，CPU / 8 线程，不训练</p>',
            f'<div class="metrics"><div class="metric"><b>{fifo_before} / 9</b>FIFO 明确接地前输出</div><div class="metric"><b>{before} / 9</b>最新策略明确接地前输出</div><div class="metric"><b>≤ 1 等待帧</b>最新策略，另有 1 帧正在处理</div></div>',
            '<nav><a href="#change">最小改动与瓶颈</a><a href="#comparison">前后对照</a><a href="#quality">检测与丢帧</a><a href="#trace">新结果逐帧核查</a><a href="#video">参考视频</a><a href="#errors">错误归因</a><a href="#files">完整证据</a></nav>']
    if before == 9:
        body.append('<p class="lead">本次 9 次最新帧回放都在人工接地区间开始前完成首次 falling 输出。队列积压明显减少，但这不保证所有输出都及时；恢复推理的长尾与丢帧后的检测覆盖仍需分别检查。</p>')
    else:
        body.append(f'<p class="lead">最新帧策略在 9 次回放中有 {before} 次明确早于接地。未成功的重复也完整保留；不能把队列长度受限视为实时完成的保证。</p>')
    body.append('''<div class="note">接地已人工确认保留 101–107 帧，即 3.336–3.537 秒，精确时刻未知。首次输出墙钟 &lt; 3.336 秒才计“明确接地前”；落入区间为“不确定”；&gt; 3.537 秒为“明确已晚”。这里测量的是本机按源时钟安排到达至 V7 输出完成，没有摄像头采集、网络或通知送达。单个开发事件且存在训练重叠，不能证明泛化预警能力。</div>
<section id="change"><h2>先复用旧计时定位，再只改调度</h2><p>历史日志显示，首次 falling 之前，姿态推理和 B/C 的人体检测占主要计算时间；约 30 Hz 的输入到达速度快于处理速度，FIFO 等待不断增加。C 的耗时恢复发生在首次 falling 之后，不能用它解释最初错过接地。</p>''')
    diagrows = []
    for mode in 'ABC':
        g = diagnosis['groups'][mode] if isinstance(diagnosis.get('groups'), dict) else None
        if g:
            s = g['slices']['through_first_falling_source_frame_87']
            diagrows.append([mode, number(s['service']['mean_s']), number(s['stages']['pose']['fraction_of_service']*100,1)+'%', number(s['stages']['detector']['fraction_of_service']*100,1)+'%', number(g['first_falling']['queue_wait_s']), number(g['first_falling']['service_wall_s'])])
    if diagrows:
        body.append(table(['历史组','87 帧之前平均处理 s','姿态占比','检测占比','触发帧等待 s','触发帧处理 s'],diagrows))
    body.append('''<p>唯一采用的改动：工作线程空闲时，只取已经到达的最新源帧，较旧等待帧被替换；正在处理的帧不被中断。原始帧号与时间保留，姿态、检测、补检、跟踪和 V7 均对实际选中帧重新执行，未选帧不插值、不合成、不使用未来帧。</p><p>这使文件回放能够显式对照已有实时采集的单等待帧语义；代码检查发现实时摄像头采集本来就有该替换机制，所以本轮不是修复已证实的摄像头 FIFO 缺陷。中性首帧测试中，2 线程比 8 线程慢，因此没有采用线程数改动。</p><p class="muted">主对照是本轮同配置重新测量的 FIFO。旧日志仅作为瓶颈依据，不拿跨时段单次测量代替优化对照。各策略三次按预设次序交替执行；模型加载和姿态/检测首帧预热排除，C 恢复模型加载但不做推理预热，所有策略遵循相同规则。</p></section>
<section id="comparison"><h2>优化前后：同输入、同权重、三次重复</h2><p>首次输出为从第一帧开始因果计算，首次 <code>label == falling</code>；无真值截断、无防抖、无阈值调整。秒数显示中位数（最小–最大），计数显示三次范围。等待帧不包含正在处理的那一帧。</p>''')
    body.append('<p class="muted">一次早期尝试因缓存字典索引层级错误在首帧日志写入前中止，没有可用测量；代码修正后按原策略与顺序重跑全部 18 次。原协议、代码及空日志保存在 <a href="interrupted-attempt-1/interruption.json">中止尝试记录</a>，未混入统计。</p>')
    body.append(table(['组 / 调度','首次输出墙钟 s','首次源帧','明确接地前','区间内','明确已晚','未触发','最大等待帧','处理 / 丢弃（共 160）'],mainrows))
    marginrows=[]
    for c in cells:
        marginrows.append([c['mode']+' · '+colors[c['policy']],rng(c['first_source_time_s']),rng(c['contact_margin_low_s']),rng(c['contact_margin_high_s'])])
    body.append(table(['组 / 调度','触发对应源时间 s','对接地下界余量 s','对接地上界余量 s'],marginrows))
    body.append('<p class="muted">余量 = 接地边界 − 实际输出完成墙钟；正数表示更早。源时间只标识画面发生时刻，不能代替输出完成时刻。两个余量列分别聚合三次结果，逐次配对值见下方完整运行明细。</p><a href="first-output-deadline.png"><img class="plot" src="first-output-deadline.png" alt="全部 18 次首次 falling 输出完成墙钟与接地区间"></a><h3>等待、处理和结果年龄必须分开看</h3>')
    body.append(table(['组 / 调度','等待中位 / P95 s','处理中位 / P95 s','输出年龄中位 / P95 s','输出年龄最大值范围 s','首次触发等待 / 处理 s'],latencyrows))
    body.append('<p class="muted">每个“中位 / P95”先在一次运行内计算，再显示三次的中位数（范围）；没有把不同保留帧数的运行混池。输出年龄 = 输出完成墙钟 − 输入源时间。最大等待帧为 latest 保留槽数；日志另记未处理到达帧数，其中包含将被替换的帧，两者不能混用。</p><a href="queue-output-age.png"><img class="plot" src="queue-output-age.png" alt="A/B/C 三次重复的输出年龄曲线，包含 C 恢复长尾"></a></section>')
    body.append('<section id="quality"><h2>检测保留了什么，又丢掉了什么？</h2><p>官方 falling 区间仍为 83–112 共 30 帧，未改变真值。一次事件命中只要求该区间内至少有一帧输出 falling。动作参考仍大多未完成人工复核；接地标注的确认不等于确认整段动作真值。</p>')
    body.append(table(['组 / 调度','事件命中','处理官方跌倒帧','区间内 falling 输出','unknown / 原始帧','目标不可用 / 原始帧','暂定已标动作正确 / 全部已标','已标 falling 误报 / 未标 falling'],qualityrows))
    body.append('<p>所有输入仍以原始 160 帧为分母，丢帧是没有输出，不能计正确；表中 unknown 和目标不可用只计实际处理后出现的情况，不能据其减少声称完整覆盖改善。“暂定已标动作正确”以全部有动作参考的原始帧为分母，丢弃的已标帧同样不计命中。已标区间误报和未标区间 falling 分开报告。</p><a href="source-coverage.png"><img class="plot" src="source-coverage.png" alt="全部 18 次运行的原始 160 帧处理和丢弃覆盖"></a><h3>稀疏历史与恢复长尾</h3>')
    body.append(table(['组 / 调度','最大源时间间隔 s','间隔 >0.35 / >0.5 s','可见目标 epoch 变化','执行恢复帧数','恢复总耗时 s','同源标签变化 / 阶段变化'],gaprows))
    fifo_eq=sum(c['full_prediction_equality_repeats'] for c in cells if c['policy']=='fifo')
    body.append(f'<p>本轮 FIFO 完整输入有 {fifo_eq}/9 次逐帧完整预测字典与历史缓存完全一致。最新策略的输入历史改变，“同源标签/阶段变化”只描述实际处理帧与旧完整 FIFO 的差异，不是准确率或回归失败判据。间隔 &gt;0.35 / &gt;0.5 秒是超过动作/跟踪重置阈值的输入间隔数量，不自动等同于确认发生的重置次数；实际 target epoch 另列。</p><div class="note">等待槽受限不能中断正在进行的恢复推理。C 若在一帧耗时很久，仍会输出陈旧结果；此时下一帧虽然取最新输入，也可能已跨越短动作并重置历史。下一步宜单独验证恢复耗时预算或独立执行，本轮没有修改恢复算法、阈值或权重。</div></section>')
    trialrows=[]
    for t in sorted(trials,key=lambda t:(t['repeat'],t['mode'],t['policy'])):
        f=t['first_falling']
        resultzh={'before_contact_certain':'接地前','within_contact_uncertainty':'区间内 / 不确定','after_contact_certain':'已晚','miss_no_falling_output':'未触发'}[t['contact_result']]
        trialrows.append([f'<a href="trials/{t["trial"]}/summary.json">{esc(t["trial"])}</a>',str(f['frame']) if f else '—',number(f['output_wall_s']) if f else '—',(' / '.join(number(v) for v in f['contact_margin_s'])) if f else '—',resultzh,str(t['max_pending_slots']),str(t['processed_frames'])+' / '+str(t['dropped_frames']),str(t['official_fall_event']['hit']),f'<a href="trials/{t["trial"]}/frames.jsonl">逐帧日志</a>'])
    body.append('<details><summary>展开全部 18 次运行，含所有未达标结果</summary>'+table(['运行','首次源帧','输出墙钟 s','余量下 / 上界 s','判定','最大等待','处理 / 丢弃','事件命中','明细'],trialrows)+'</details>')
    body.append('''<section id="trace"><h2>优化后逐帧日志：直接核对实际输出</h2><p>选择任意一次新运行，表格展示该次真正处理的输入与重新计算的结果。丢弃源帧列明确列出没有输出的输入；不以旧视频覆盖层代替新策略结果。</p><label for="trace-select">运行：</label><select id="trace-select">''')
    ordered=sorted(trials,key=lambda t:(t['mode'], t['policy']=='latest',t['repeat']))
    for t in ordered:
        body.append(f'<option value="{t["trial"]}">{t["trial"]}</option>')
    body.append('</select><p id="trace-summary" class="muted"></p><div class="scroll trace"><table><thead><tr><th>源帧 / 源秒</th><th>输出墙钟 s</th><th>等待 / 处理 / 年龄 s</th><th>动作 / 阶段</th><th>目标</th><th>此前丢弃帧</th><th>恢复耗时 s</th></tr></thead><tbody id="trace-body"></tbody></table></div></section>')
    body.append('<section id="video"><h2>视频与逐帧输出相互核对</h2>')
    if (out/'media/C-r2-fifo-vs-latest.mp4').exists():
        body.append('''<div class="card"><h3>本轮 C · 第 2 次重复：按实际输出墙钟播放</h3><p>左为本轮 FIFO，右为本轮最新帧策略。画面沿本机回放墙钟推进，只在该次处理实际完成后显示对应结果；等待期间保持最后完成输出（HOLD），不能解读为对中间丢帧的预测。第 2 次重复按预定编号选取，完整统计仍包含全部 3 次。</p><video id="optimized-video" controls playsinline preload="metadata" poster="media/C-r2-fifo-vs-latest-contact.jpg" src="media/C-r2-fifo-vs-latest.mp4"></video><div><button id="optimized-play-toggle">墙钟对照：播放 / 暂停</button><button id="optimized-rate-slow" data-optimized-rate="0.25">墙钟对照：0.25×</button><button id="optimized-rate-normal" data-optimized-rate="1">墙钟对照：正常速度</button><button id="optimized-seek-contact" data-optimized-seek="3.3">墙钟 3.3 秒：接地前对照</button><button id="optimized-seek-fifo" data-optimized-seek="6.9">墙钟 6.9 秒：FIFO 首检后</button></div><p class="muted">该视频由新逐帧日志与原始图像生成，无新增模型推理。播放轴 30 Hz，结果只在完成后下一个显示刻度出现（最多约 33 毫秒显示取整）；画面数值保留日志实测时刻。完成时年龄与当前保持显示年龄分别标注，无预测或姿态插值。这不是实际摄像头或通知录屏。<a href="media/metadata.json">完整显示语义与校验</a></p></div>''')
    body.append('''<div class="card"><h3>源时间参考：历史 A/B/C 完整 FIFO 覆盖</h3><p><strong>下方 A/B/C 覆盖视频来自实验一冻结的完整 FIFO 缓存。</strong>它用来查看同一个人、动作与源帧位置，不能展示本轮最新帧策略的标签或输出墙钟。优化后的真实结果请看上方逐帧日志。</p><video id="reference-video" controls playsinline preload="metadata" src="../media/full/urfd-fall-01-original.mp4"></video><div><button data-seek="2.869">历史首次 falling：87 帧</button><button data-seek="3.336">接地下界：101 帧</button><button data-seek="3.537">接地上界：107 帧</button><button data-rate="0.25">0.25×</button><button data-rate="1">正常速度</button><button id="play-toggle">播放 / 暂停</button></div></div><p><a href="../human-review/#contact">打开原始接地候选帧与慢放</a> · <a href="../experiment3-20261021/index.html#review">四项人工复核及待定标注</a></p><p class="muted">播放器按编码帧率取整，精确源时间以 JSON 为准。接地精确帧、稳定精确起点、UTK 物理离座等未决项保持原状；本次只评估 fall01 的处理调度。</p></section><section id="errors"><h2>错误归因与下一步</h2>''')
    body.append(table(['观察','归因 / 证据','下一步'],[[esc(e['finding']),esc(e['cause'])+'<br><span class="muted">'+esc(e['evidence'])+'</span>',esc(e['next'])] for e in make_errors(comparison)]))
    body.append('''</section><section id="files"><h2>可追溯文件</h2><p><a href="comparison.csv">对照 CSV</a> · <a href="comparison.json">聚合与配对结果 JSON</a> · <a href="results.json">全部运行摘要</a> · <a href="protocol.json">预设规则与固定哈希</a> · <a href="diagnosis.json">旧计时瓶颈分析</a> · <a href="thread-probe.json">未采用的线程探测</a> · <a href="error-attribution.json">错误归因</a> · <a href="annotations.reviewed.snapshot.json">人工复核标注快照</a> · <a href="report-source.py">报告生成代码快照</a> · <a href="validation.json">完整性验证</a> · <a href="scheduler-tests.xml">调度测试结果</a></p><p><a href="first-output-deadline.svg">首次输出图 SVG</a> · <a href="queue-output-age.svg">输出年龄图 SVG</a> · <a href="source-coverage.svg">丢帧覆盖图 SVG</a></p></section><footer>固定 V7 SHA256 <code>'''+esc(protocol['model_sha256'])+'''</code>。本轮仅新增调度对照，不训练，不改变模型、检测参数或人工标注，不发送实际通知。所有重复与逐帧丢弃记录均保留。</footer></main>''')
    data={}
    for t in trials:
        data[t['trial']]={'summary':f"处理 {t['processed_frames']}/160 帧；丢弃 {t['dropped_frames']}/160 帧；官方跌倒区间命中：{t['official_fall_event']['hit']}；最大源时间间隔 {t['max_source_gap_s']:.3f} 秒。",
                         'rows':[[r['source_frame_id'],r['source_time_s'],r['output_wall_s'],r['queue_wait_s'],r['service_s'],r['output_age_s'],r['prediction']['label'],r['prediction']['phase'],r['observed'],r['dropped_before_this_frame'],r['stages_s']['recovery']] for r in frames[t['trial']]]}
    body.append('<script>const traces='+json.dumps(data,ensure_ascii=False).replace('</','<\\/')+''';const select=document.getElementById('trace-select');function renderTrace(){const d=traces[select.value];document.getElementById('trace-summary').textContent=d.summary;const tb=document.getElementById('trace-body');tb.replaceChildren();for(const r of d.rows){const tr=document.createElement('tr');const n=v=>v.toFixed(3);for(const v of [r[0]+' / '+n(r[1]),n(r[2]),n(r[3])+' / '+n(r[4])+' / '+n(r[5]),r[6]+' / '+r[7],r[8]?'可观测':'不可用',r[9].length?r[9].join(', '):'—',n(r[10])]){const td=document.createElement('td');td.textContent=v;tr.appendChild(td)}tb.appendChild(tr)}}select.value='C-latest-r1';select.onchange=renderTrace;renderTrace();const v=document.getElementById('reference-video');document.querySelectorAll('button[data-seek]').forEach(b=>b.onclick=()=>{v.pause();v.currentTime=Number(b.dataset.seek)});document.querySelectorAll('button[data-rate]').forEach(b=>b.onclick=()=>v.playbackRate=Number(b.dataset.rate));document.getElementById('play-toggle').onclick=()=>v.paused?v.play():v.pause();const ov=document.getElementById('optimized-video');if(ov){document.querySelectorAll('button[data-optimized-seek]').forEach(b=>b.onclick=()=>{ov.pause();ov.currentTime=Number(b.dataset.optimizedSeek)});document.querySelectorAll('button[data-optimized-rate]').forEach(b=>b.onclick=()=>ov.playbackRate=Number(b.dataset.optimizedRate));document.getElementById('optimized-play-toggle').onclick=()=>ov.paused?ov.play():ov.pause();}</script></html>''')
    (out/'index.html').write_text(''.join(body))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();out=args.output
    results=read(out/'results.json');protocol=read(out/'protocol.json');diagnosis=read(out/'diagnosis.json')
    assert results['complete'] and len(results['trials'])==18
    trials=results['trials'];frames={t['trial']:[json.loads(s) for s in (out/'trials'/t['trial']/'frames.jsonl').read_text().splitlines()] for t in trials}
    for t in trials:
        assert len(frames[t['trial']])==t['processed_frames']
        assert t['processed_frames']+t['dropped_frames']==160
    comparison=aggregate(trials,frames,protocol['contact_interval_s'])
    comparison['provenance']=[{'file':str(p),'sha256':sha(p)} for p in [out/'results.json',out/'protocol.json',out/'diagnosis.json',out/'thread-probe.json',Path(__file__)]]
    write(out/'comparison.json',comparison)
    fields=['mode','policy','repeats','first_output_median_s','first_output_min_s','first_output_max_s','first_source_frame_min','first_source_frame_max','before_contact','within_contact_interval','after_contact','no_trigger','max_pending_min','max_pending_max','processed_min','processed_max','dropped_min','dropped_max','event_hit_repeats','queue_wait_median_of_medians_s','queue_wait_median_p95_s','service_median_of_medians_s','service_median_p95_s','output_age_median_of_medians_s','output_age_median_p95_s','output_age_max_s','unknown_min','unknown_max','unobserved_min','unobserved_max','source_gap_max_s','reference_samples']
    with (out/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in comparison['cells']:
            row={'mode':c['mode'],'policy':c['policy'],'repeats':c['repeats'],'event_hit_repeats':c['event_hit_repeats'],'reference_samples':160,'source_gap_max_s':c['max_source_gap_s']['max']}
            for dst,key in [('first_output','first_output_wall_s'),('first_source_frame','first_source_frame'),('max_pending','max_pending_slots'),('processed','processed_frames'),('dropped','dropped_frames'),('unknown','unknown_processed_frames'),('unobserved','unobserved_processed_frames')]:
                for s in ('min','max'):
                    row[dst+'_'+s+('_s' if dst=='first_output' else '')]=c[key][s]
            row['first_output_median_s']=c['first_output_wall_s']['median']
            for dst,key in [('before_contact','before_contact_certain'),('within_contact_interval','within_contact_uncertainty'),('after_contact','after_contact_certain'),('no_trigger','miss_no_falling_output')]:row[dst]=c['contact_results'].get(key,0)
            for metric in ('queue_wait','service','output_age'):
                row[metric+'_median_of_medians_s']=c[metric]['median_s']['median'];row[metric+'_median_p95_s']=c[metric]['p95_s']['median']
            row['output_age_max_s']=c['output_age']['max_s']['max'];writer.writerow(row)
    write(out/'error-attribution.json',make_errors(comparison))
    if not args.no_plots:plots(out,trials,frames,protocol['contact_interval_s'])
    render(out,comparison,trials,frames,protocol,diagnosis)
    (out/'report-source.py').write_bytes(Path(__file__).read_bytes())
    print(out/'index.html')


if __name__=='__main__':
    main()
