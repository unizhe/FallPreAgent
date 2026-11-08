"""Build a fixed-V7 unknown-cause report from completed trials only; no inference."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

import report_v7_recovery_budget as previous
from report_v7_recovery_budget import read, write, stat, fmt, rng, esc, table, selected_pose

ROOT=Path(__file__).resolve().parents[1]
DEFAULT=ROOT/'runs/abc-v7-20261020/unknown-reduction-20261021'
VARIANTS=('budget200ms','costaware200ms')
NAMES={'budget200ms':'当前 C：200 ms 按已耗时间检查','costaware200ms':'改动 C：200 ms + 下次调用耗时准入'}
COLORS={'budget200ms':'#bc6842','costaware200ms':'#187e87'}


def audit_trials(audit):
    """Normalize the independently generated audit without inferring reset events."""
    trials=audit['trials']
    return trials if isinstance(trials,dict) else {t['trial']:t for t in trials}


def audit_rows(trial):
    rows=trial.get('frames',trial.get('rows',[]))
    return {r['source_frame_id']:r for r in rows}


def measure(summary,rows,audit):
    previous.VARIANTS=VARIANTS
    m=previous.measure_trial(summary,rows)
    m['unknown_fraction_source']=m['unknown_outputs']/160
    m['unknown_fraction_processed']=m['unknown_outputs']/m['processed_frames']
    m['native_observed']=sum(r['observed'] and not r['recovery_selected'] for r in rows)
    m['target_expiry_events']=len(summary['tracking_statistics']['target_expired_frames'])
    m['action_audit']=audit
    m['max_valid_action_gap_s']=max(r['gap_since_last_valid_s'] for r in audit['frames'] if r['action_observed'] and r['previous_valid'])
    # These are actual state-transition measurements from the independent audit.
    m.update(action_metrics(audit))
    budgets=[(r.get('recovery') or {}).get('budget_audit',{}) for r in rows]
    m['predicted_call_rejections']=sum(a.get('exhaustion_stage')=='predicted_call_overrun' for a in budgets)
    m['elapsed_budget_aborts']=sum(a.get('exhausted',False) and a.get('exhaustion_stage')!='predicted_call_overrun' for a in budgets)
    m['recovery_selected_source_ids']=[r['source_frame_id'] for r in rows if r['recovery_selected']]
    m['unknown_source_ids']=[r['source_frame_id'] for r in rows if r['prediction']['label']=='unknown']
    m['target_expiry_source_ids']=summary['tracking_statistics']['target_expired_frames']
    return m


def action_metrics(audit):
    """Use explicit audit schema and fail on mismatch rather than defaulting to zero."""
    causes=audit['unknown_categories']
    resets=audit['reset_counts']
    assert sum(causes.values())==audit['unknown_count']
    missing_keys={'initial_target_unconfirmed','recovery_abandoned_missing_observation','target_unavailable_after_expiry_or_loss','other_missing_observation','partial_pose_withheld'}
    known=missing_keys|{'initial_warmup','post_gap_reset_warmup','classifier_uncertainty'}
    out={'unknown_cause_counts':dict(causes),
         'unknown_missing':sum(v for k,v in causes.items() if k in missing_keys),
         'unknown_recovery_abandoned':causes.get('recovery_abandoned_missing_observation',0),
         'unknown_initial_target':causes.get('initial_target_unconfirmed',0),
         'unknown_target_unavailable':causes.get('target_unavailable_after_expiry_or_loss',0),
         'unknown_startup':causes.get('initial_warmup',0),
         'unknown_postgap':causes.get('post_gap_reset_warmup',0),
         'unknown_low_confidence':causes.get('classifier_uncertainty',0),
         'unknown_other':sum(v for k,v in causes.items() if k not in known),
         'actual_v3_resets':resets['v3']['all_clear_calls'],
         'actual_v4_resets':resets['v4']['all_clear_calls'],
         'v3_valid_history_resets':resets['v3']['valid_history_clear_calls'],
         'v4_valid_history_resets':resets['v4']['valid_history_clear_calls'],
         'v3_gap_resets':resets['v3'].get('reason:last_valid_observation_gap_gt_0_35',0),
         'v4_gap_resets':resets['v4'].get('reason:last_valid_observation_gap_gt_0_35',0)}
    assert sum(out[k] for k in ['unknown_missing','unknown_startup','unknown_postgap','unknown_low_confidence','unknown_other'])==audit['unknown_count']
    return out


def compare(trials,frames,protocol,audit):
    ats=audit_trials(audit)
    metrics=[measure(s,frames[s['trial']],ats[s['trial']]) for s in trials]
    cells=[]
    for variant in VARIANTS:
        ms=[m for m in metrics if m['variant']==variant]
        assert len(ms)==3
        numeric=[k for k,v in ms[0].items() if isinstance(v,(int,float)) and not isinstance(v,bool)]
        cells.append({'variant':variant,'trials':[m['trial'] for m in ms],
           'statistics':{k:stat([m[k] for m in ms]) for k in numeric},
           'event_hits':sum(m['event_hit'] for m in ms),
           'contact_results':dict(Counter(m['contact_result'] for m in ms))})
    pairs=[]
    for repeat in (1,2,3):
        a,b=[next(m for m in metrics if m['variant']==v and m['repeat']==repeat) for v in VARIANTS]
        aa,bb=({r['source_frame_id']:r for r in frames[m['trial']]} for m in (a,b))
        common=sorted(aa.keys()&bb.keys())
        pairs.append({'repeat':repeat,'unknown_reduction':a['unknown_outputs']-b['unknown_outputs'],
           'unknown_source_fraction_reduction':a['unknown_fraction_source']-b['unknown_fraction_source'],
           'unknown_processed_fraction_reduction':a['unknown_fraction_processed']-b['unknown_fraction_processed'],
           'processed_increase':b['processed_frames']-a['processed_frames'],
           'common_source_ids':common,'common_label_change_ids':[i for i in common if aa[i]['prediction']['label']!=bb[i]['prediction']['label']],
           'common_phase_change_ids':[i for i in common if aa[i]['prediction']['phase']!=bb[i]['prediction']['phase']],
           'common_observed_change_ids':[i for i in common if aa[i]['observed']!=bb[i]['observed']],
           'baseline_only_source_ids':sorted(aa.keys()-bb.keys()),'costaware_only_source_ids':sorted(bb.keys()-aa.keys())})
    return {'schema':'fallpre.v7.unknown_reduction_report.v1','created_utc':datetime.now(timezone.utc).isoformat(),
       'source_reference_frames':160,'contact_interval_s':protocol['contact_interval_s'],
       'aggregation':'Median/min/max of all three preset repeats; no best-run selection. Unknown denominators source=160 and processed are both retained.',
       'causal_limit':'Scheduling differs with wall time. Cause classifications describe executed control flow; cross-variant count changes are not isolated additive causal effects.',
       'trial_metrics':metrics,'cells':cells,'pairs':pairs}


def full_frame_inventory(out,comparison,frames,protocol):
    ann=read(out/'annotations.reviewed.snapshot.json')
    entry=next(e for e in ann['entries'] if e['clip_id']==protocol['clip_id'])
    times=entry['clock']['source_times_s']
    inventory=[]
    for m in comparison['trial_metrics']:
        rs={r['source_frame_id']:r for r in frames[m['trial']]}
        ars=audit_rows(m['action_audit'])
        for fid in range(1,161):
            r=rs.get(fid);a=ars.get(fid,{})
            item={'trial':m['trial'],'strategy':m['variant'],'repeat':m['repeat'],
                  'source_frame_id':fid,'source_time_s':times[fid-1],
                  'processed':r is not None,'dropped':r is None,'prediction':None,'phase':None,
                  'unknown_cause':None,'observed':None,'target_id':None,'tracking_reason':None,
                  'recovery_attempt':None,'recovery_selected':None,'recovery_exhaustion_stage':None,
                  'recovery_s':None,'output_wall_s':None,'output_age_s':None,
                  'action_audit':a or None}
            if r:
                b=r['recovery']['budget_audit']
                item.update(prediction=r['prediction']['label'],phase=r['prediction']['phase'],
                    unknown_cause=a.get('unknown_category'),
                    observed=r['observed'],target_id=r['target_id'],tracking_reason=r['tracking_reason'],
                    recovery_attempt=r['recovery']['executed'],recovery_selected=r['recovery_selected'],
                    recovery_exhaustion_stage=b.get('exhaustion_stage'),recovery_s=r['stages_s']['recovery'],
                    output_wall_s=r['output_wall_s'],output_age_s=r['output_age_s'])
            inventory.append(item)
    assert len(inventory)==960
    write(out/'all-source-frames.json',{'schema':'fallpre.v7.all_source_frames.v1','rows':inventory,
        'note':'All 160 source IDs per trial. Dropped inputs have null prediction and null observed, never false/correct/unknown.'})
    flat=[{**r,'action_audit':json.dumps(r['action_audit'],ensure_ascii=False,separators=(',',':')) if r['action_audit'] else ''} for r in inventory]
    with (out/'all-source-frames.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    return inventory


def plots(out,comparison,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':120,'savefig.dpi':170,'axes.spines.top':False,'axes.spines.right':False})
    ordered=sorted(comparison['trial_metrics'],key=lambda m:(m['repeat'],m['variant']!='budget200ms'))
    fig,axs=plt.subplots(2,1,figsize=(11.7,7.7),constrained_layout=True)
    labels=[m['trial'] for m in ordered]
    # Control-flow classes are mutually exclusive; unclassified remains explicit.
    bottom=[0]*len(ordered)
    for key,name,color in [('unknown_missing','Missing/ineligible input','#b55d47'),('unknown_startup','Initial history warmup','#a39373'),('unknown_postgap','History warmup after gap reset','#5e7fa8'),('unknown_low_confidence','Low-confidence output','#9c6ba1')]:
        vals=[m[key] for m in ordered]
        axs[0].bar(range(len(ordered)),vals,bottom=bottom,label=name,color=color)
        bottom=[a+b for a,b in zip(bottom,vals)]
    other=[m['unknown_outputs']-b for m,b in zip(ordered,bottom)]
    assert min(other)>=0
    if any(other):axs[0].bar(range(len(ordered)),other,bottom=bottom,label='Other classified unknown',color='#535b60')
    axs[0].set_xticks(range(len(ordered)),labels,rotation=10);axs[0].set_ylabel('Unknown outputs / 160 source inputs');axs[0].legend(ncol=2,fontsize=9)
    axs[0].set_title('Unknown control-flow causes; dropped inputs stay separate')
    for y,m in enumerate(ordered):
        rs=frames[m['trial']];known=[r['source_frame_id'] for r in rs if r['prediction']['label']!='unknown'];unknown=m['unknown_source_ids']
        axs[1].scatter(m['dropped_source_frame_ids'],[y]*m['dropped_frames'],marker='|',s=140,color='#cbd2d1')
        axs[1].scatter(known,[y]*len(known),marker='|',s=140,color=COLORS[m['variant']])
        axs[1].scatter(unknown,[y]*len(unknown),marker='x',s=25,color='#803c39')
    axs[1].axvspan(82.5,112.5,color='#e4c56d',alpha=.25)
    axs[1].set_yticks(range(len(ordered)),labels);axs[1].invert_yaxis();axs[1].set_xlim(.5,160.5)
    axs[1].set_xlabel('Source frame: grey = dropped; color = non-unknown; x = unknown; yellow = official fall interval')
    for ext in ('png','svg'):fig.savefig(out/f'unknown-causes.{ext}')
    plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11.7,7.3),sharex=True,constrained_layout=True)
    for m in comparison['trial_metrics']:
        rs=frames[m['trial']];color=COLORS[m['variant']]
        axs[0].plot([r['source_time_s'] for r in rs],[r['stages_s']['recovery'] for r in rs],marker='.',ms=3,alpha=.65,color=color,label=m['variant'] if m['repeat']==1 else None)
        axs[1].plot([r['source_time_s'] for r in rs],[r['output_age_s'] for r in rs],marker='.',ms=3,alpha=.65,color=color)
    axs[0].axhline(.2,ls='--',lw=1,color='#555',label='Unchanged 0.2 s soft recovery budget')
    for ax in axs:ax.axvspan(*comparison['contact_interval_s'],color='#e4c56d',alpha=.25);ax.grid(alpha=.2)
    axs[0].set_ylabel('Recovery stage (s)');axs[0].legend(fontsize=9)
    axs[1].set_ylabel('Output completion age (s)');axs[1].set_xlabel('Source time (s); dropped frames have no output')
    fig.suptitle('Same latest-frame scheduler and V7; 200 ms budget with/without call-cost admission')
    for ext in ('png','svg'):fig.savefig(out/f'unknown-latency.{ext}')
    plt.close(fig)


def errors(comparison):
    cells={c['variant']:c['statistics'] for c in comparison['cells']};b,o=[cells[v] for v in VARIANTS]
    return [
      {'id':'U1','finding':'补检放弃使当前帧仍然缺少动作姿态','evidence':'recovery.exhausted、selected_instance、action-audit unknown原因','explanation':'预算结束返回原始观测；只有缺失或不适用姿态才直接记为unknown，不能把每个预算终止都当unknown。',
       'outcome':f"本轮 missing 类 unknown：{rng(b['unknown_missing'],0)} → {rng(o['unknown_missing'],0)}；恢复实际选中：{rng(b['recovery_selected'],0)} → {rng(o['recovery_selected'],0)}。"},
      {'id':'U2','finding':'最新帧跳过输入不会自行产生unknown输出','evidence':'all-source-frames.json/.csv完整960行，跳过帧prediction=null','explanation':'跳帧使有效姿态在源时间上更疏；只有后续被处理帧才能输出unknown。各组采样序列变化，不把跳帧数差与unknown数差作一一因果对应。',
       'outcome':f"跳帧/160：{rng(b['dropped_frames'],0)} → {rng(o['dropped_frames'],0)}；unknown/160：{rng(b['unknown_outputs'],0)} → {rng(o['unknown_outputs'],0)}。"},
      {'id':'U3','finding':'真实动作历史重置使用上次有效姿态时间','evidence':'action-audit.json重放与真实V3/V4 reset事件；不使用相邻处理帧gap标志替代','explanation':'补检帧虽被处理但没有有效姿态，不会刷新有效姿态时钟。下次有效姿态间隔超过原0.35秒阈值会清空历史，随后出现warmup unknown。重置门限未修改。',
       'outcome':f"post-gap warmup unknown：{rng(b['unknown_postgap'],0)} → {rng(o['unknown_postgap'],0)}；V3有效历史重置 {rng(b['v3_valid_history_resets'],0)} → {rng(o['v3_valid_history_resets'],0)}，V4有效历史重置 {rng(b['v4_valid_history_resets'],0)} → {rng(o['v4_valid_history_resets'],0)}。"},
      {'id':'U4','finding':'首次补检调用仍可能超预算','evidence':'budget_audit calls、exhaustion_stage及estimated_next_call_s','explanation':'新准入只用本次同后端已完成调用的最长耗时；首调用无样本，后续调用也可能比估计更慢。200ms仍是软预算。',
       'outcome':f"最大补检耗时 s：{rng(b['recovery_max_s'])} → {rng(o['recovery_max_s'])}；最长输出间隔 s：{rng(b['output_gap_max_s'])} → {rng(o['output_gap_max_s'])}。"},
      {'id':'U5','finding':'unknown减少不自动等于动作识别正确','evidence':'官方83–112帧覆盖、共同帧预测变化、未标falling单列','explanation':'动作标签、目标身份和全部骨架并无完整人工真值。113–137帧未评分，后段background仍为暂定参考。',
       'outcome':f"官方跌倒区间falling输出/30：{rng(b['fall_reference_falling_outputs'],0)} → {rng(o['fall_reference_falling_outputs'],0)}；未标区间falling：{rng(b['unscored_falling'],0)} → {rng(o['unscored_falling'],0)}。"},
      {'id':'U6','finding':'接地前输出、目标连续和unknown需要分别验收','evidence':'实际完成墙钟对人工接地区间；target_expired及native observed','explanation':'本机输出不含采集、网络及通知送达。目标ID是算法状态而非人工身份真值。三对重复只检验本机同一开发事件。',
       'outcome':f"改动组最保守接地余量 {fmt(o['contact_margin_low_s']['min'])} s；目标过期事件 {rng(b['target_expiry_events'],0)} → {rng(o['target_expiry_events'],0)}。"}]


def render(out,c,errs,frames,protocol,diagnosis,pending):
    cells={x['variant']:x for x in c['cells']};b,o=[cells[v]['statistics'] for v in VARIANTS]
    # Reuse only CSS from the established local report; previous artifact is untouched.
    previous_html=(out.parent/'recovery-budget-20261021/index.html').read_text()
    css=previous_html.split('<style>',1)[1].split('</style>',1)[0]
    sections=[f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>固定 V7 · unknown 归因与最小改动对照</title><style>{css}</style><main>',
      '<a href="../recovery-budget-20261021/index.html">← 上轮补检预算对照</a><h1>固定 V7：查清 unknown，再缩短有效姿态间隔</h1>',
      '<p class="muted">同一 URFD fall01 · 160 源帧 · C 最新帧调度 · 固定 200 ms 软预算 · 2 组 × 3 次 · 不训练</p>',
      f'<div class="metrics"><div class="metric"><b>{rng(b["unknown_outputs"],0)} → {rng(o["unknown_outputs"],0)}</b>unknown / 160（中位数与范围）</div><div class="metric"><b>{cells["costaware200ms"]["contact_results"].get("before_contact_certain",0)} / 3</b>改动组明确接地前完成首次 falling</div><div class="metric"><b>{rng(o["target_expiry_events"],0)}</b>改动组目标过期事件</div></div>',
      '<nav><a href="#diagnosis">旧日志原因</a><a href="#change">最小改动</a><a href="#comparison">新对照</a><a href="#history">历史重置</a><a href="#quality">跳帧与检测</a><a href="#video">可播放视频</a><a href="#errors">错误归因</a><a href="#pending">待复核</a><a href="#files">完整证据</a></nav>',
      '<div class="note">接地仍为 101–107 帧 / 3.336–3.537 秒，精确时刻未知。完成墙钟早于 3.336 秒才计明确接地前。200 ms 是补检软预算，不能中断已开始的同步调用。以下是本机受控文件回放，未测量物理采集、网络和提醒送达。</div>',
      '<section id="diagnosis"><h2>先复用旧日志：unknown 从哪里来</h2><div class="card"><p>将旧完整补检与旧 200 ms 日志中的已选姿态，按原顺序重放到冻结 V7，并核对原预测。补检放弃可使当前帧缺姿态；其后的最新帧跳过与总处理耗时拉长“上次有效姿态”到“本次有效姿态”的时间；超过原 0.35 秒动作历史阈值后，后续帧需要重新积累历史。</p><p>相邻处理帧间隔小于 0.35 秒，并不保证有效姿态间隔小于阈值。直接缺失、启动预热、重置后预热是输出的互斥原因；补检与跳帧是上游因素，不能与这些原因简单相加。</p><p>下表保留六条旧日志各自执行路径。旧两组处理帧不同，计数差用于定位机制，不是严格控制采样后的独立因果贡献。</p></div>']
    dtrials=audit_trials(diagnosis)
    drows=[]
    for name,t in dtrials.items():
        a=action_metrics(t);s=t.get('summary',t)
        total=s.get('unknown_count',s.get('unknown_outputs',sum(a['unknown_cause_counts'].values())))
        drows.append([esc(name),str(total),*[str(a[k]) for k in ['unknown_missing','unknown_startup','unknown_postgap','unknown_low_confidence','actual_v3_resets','actual_v4_resets']]])
    sections.append(table(['旧日志','unknown','缺姿态','初始预热','gap后预热','低置信度','V3 reset（状态核实）','V4 reset（状态核实）'],drows))
    sections.append('<p class="muted">所有日志级原因和每次reset的前后历史均可查 <a href="diagnosis.json">旧日志诊断</a>。V7小模型重放不涉及视觉模型推理，不能用重放时间替代端到端耗时。</p></section>')
    sections.append('<section id="change"><h2>唯一改动：预计下一次调用装不下，就提前结束补检</h2><div class="card"><p>两组均保留最新帧调度和 200 ms 预算。新组在每个后端调用前，用“本次补检内、同一种后端已完成调用的最长耗时”估算下一次调用成本；若已耗时间加该成本达到预算，就不启动该调用，立即返回本帧原始姿态。首次调用没有样本时照常尝试，估计在每次补检重新开始。</p><p>没有关闭动作历史重置、延长 0.35/0.5 秒阈值、复制旧姿态或填补跳帧。V7与视觉模型、候选质量规则均固定。此机制不读取帧号、事件时刻或将来的耗时。</p><p class="muted">主比较为本轮六次按预定顺序交替执行的完整流水线。模型加载和姿态/检测首帧预热排除，各组相同；补检模型加载但不做恢复推理预热。三次中位数（最小–最大）。</p></div></section><section id="comparison"><h2>三对新试验：unknown 与实时性</h2>')
    keys=[('处理 / 160','processed_frames',0),('unknown / 160','unknown_outputs',0),('unknown 占全部源帧比例','unknown_fraction_source',4),('unknown 占实际处理帧比例','unknown_fraction_processed',4),('跳帧 / 160','dropped_frames',0),('最大有效动作姿态间隔 s','max_valid_action_gap_s',3),('最长补检 s','recovery_max_s',3),('补检阶段总耗时 s','recovery_total_s',3),('最长输出间隔 s','output_gap_max_s',3),('最大输出年龄 s','output_age_max_s',3),('首次 falling 完成墙钟 s','first_output_wall_s',3),('首次 falling 源帧','first_source_frame',0),('保守接地余量 s','contact_margin_low_s',3),('最大等待槽位','max_pending_slots',0)]
    sections.append(table(['指标',NAMES[VARIANTS[0]],NAMES[VARIANTS[1]]],[[label,rng(b[k],digits),rng(o[k],digits)] for label,k,digits in keys]))
    sections.append(table(['重复','原组 unknown / processed（/160）','新组 unknown / processed（/160）','unknown 少几帧','保守接地余量 原→新 s'],[[p['repeat'],*[f"{m['unknown_outputs']} / {m['processed_frames']}（{m['unknown_outputs']}/160）" for m in [next(x for x in c['trial_metrics'] if x['repeat']==p['repeat'] and x['variant']==v) for v in VARIANTS]],str(p['unknown_reduction']),' → '.join(fmt(next(x for x in c['trial_metrics'] if x['repeat']==p['repeat'] and x['variant']==v)['contact_margin_low_s']) for v in VARIANTS)] for p in c['pairs']]))
    sections.append('<p class="muted">输出年龄=完成墙钟−源时间；最长输出间隔包括从起点到首个完成输出。处理/跳过按160源帧完整分区；unknown仅指实际输出，跳过帧不改记为unknown，也不记为正确。</p><img class="plot" src="unknown-latency.png" alt="六次试验补检耗时及输出年龄"></section>')
    sections.append('<section id="history"><h2>区分缺姿态与历史重新预热</h2>')
    sections.append(table(['新日志','unknown','缺姿态','初始预热','gap后预热','低置信度','V3 reset 总 / 有效历史','V4 reset 总 / 有效历史','预算结束：估算拒绝 / 已耗超时'],[[esc(m['trial']),m['unknown_outputs'],m['unknown_missing'],m['unknown_startup'],m['unknown_postgap'],m['unknown_low_confidence'],str(m['actual_v3_resets'])+' / '+str(m['v3_valid_history_resets']),str(m['actual_v4_resets'])+' / '+str(m['v4_valid_history_resets']),str(m['predicted_call_rejections'])+' / '+str(m['elapsed_budget_aborts'])] for m in c['trial_metrics']]))
    sections.append('<p>新日志的真实reset由独立动作重放包裹并记录原模型reset方法和有效姿态时钟；旧日志用状态前后核实。表中总调用包含初次初始化和目标确认，有效历史重置仅计清空已有有效姿态历史，未用相邻处理帧gap标志推算。warmup是缺少足够历史，不能解释成置信度下降。unknown减少是否伴随识别正确，需要另看参考标签和人工复核。</p><img class="plot" src="unknown-causes.png" alt="unknown原因和全部源帧覆盖"></section>')
    grouped={v:[m for m in c['trial_metrics'] if m['variant']==v] for v in VARIANTS}
    totals={v:{k:sum(m[k] for m in ms) for k in ['unknown_outputs','unknown_initial_target','unknown_startup','unknown_recovery_abandoned','unknown_target_unavailable','unknown_postgap','unknown_low_confidence']} for v,ms in grouped.items()}
    sections.append('<div class="card"><p>三次合计的互斥分解（用于解释执行路径，不能代替逐次比例）：</p>'+table(['组','总unknown','初始未确认','启动预热','补检放弃且缺姿态','目标已丢失','gap后预热','低置信度'],[[NAMES[v],*[totals[v][k] for k in ['unknown_outputs','unknown_initial_target','unknown_startup','unknown_recovery_abandoned','unknown_target_unavailable','unknown_postgap','unknown_low_confidence']]] for v in VARIANTS])+'<p>补检放弃的直接缺姿态仍在，差别主要是随后有效姿态更早返回，没有越过原动作历史阈值。启动预热的差别来自起始采样序列，不归功于接地后的补检改动。</p></div>')
    resetrows=[]
    for m in c['trial_metrics']:
        for a in m['action_audit']['frames']:
            h=a['histories']['v3']
            if not h['cleared_valid_history']:continue
            resetrows.append([esc(m['trial']),a['source_frame_id'],esc(a['previous_valid']['source_frame_id'] if a['previous_valid'] else None),fmt(a['gap_since_last_valid_s']),fmt(a['source_gap_s']),esc(', '.join(h['reasons'])),esc(a['unknown_category'])])
    sections.append('<details><summary>清空有效 V3 历史的实际事件</summary>'+table(['试验','当前源帧','上次有效源帧','有效姿态间隔 s','相邻处理帧间隔 s','reset原因','当前unknown原因'],resetrows)+'</details>')
    sections.append('<section id="quality"><h2>检测覆盖、恢复与目标连续性</h2>')
    sections.append(table(['组','跌倒事件命中','83–112处理/30','83–112 falling/30','native observed/160','恢复实际选中','目标过期事件','最后目标ID/observed','已标误falling / 未标falling'],[[NAMES[x['variant']],f"{x['event_hits']}/3",rng(x['statistics']['fall_reference_processed'],0),rng(x['statistics']['fall_reference_falling_outputs'],0),rng(x['statistics']['native_observed'],0),rng(x['statistics']['recovery_selected'],0),rng(x['statistics']['target_expiry_events'],0),'; '.join(str(m['last_frame_target_id'])+'/'+str(bool(m['last_frame_observed'])) for m in c['trial_metrics'] if m['variant']==x['variant']),rng(x['statistics']['provisional_false_falling'],0)+' / '+rng(x['statistics']['unscored_falling'],0)] for x in c['cells']]))
    sections.append(table(['重复','共同处理源帧','共同帧label变化','共同帧phase变化','共同帧observed变化','原组独有 / 新组独有'],[[p['repeat'],len(p['common_source_ids']),len(p['common_label_change_ids']),len(p['common_phase_change_ids']),len(p['common_observed_change_ids']),str(len(p['baseline_only_source_ids']))+' / '+str(len(p['costaware_only_source_ids']))] for p in c['pairs']]))
    sections.append('<p class="muted">两组实际源帧及因果历史不同；共同帧预测差异也包含历史差异。目标ID和target_expiry是算法状态，不等于独立身份真值。官方83–112跌倒参考、其他动作区间仍未完整人工复核；113–137未评分，不能用已标区间零误报宣布全视频零误报。</p>')
    for m in c['trial_metrics']:
        rows=[]
        ars=audit_rows(m['action_audit'])
        for r in frames[m['trial']]:
            if r['prediction']['label']!='unknown' and not r['recovery']['executed']:continue
            a=ars.get(r['source_frame_id'],{})
            rows.append([r['source_frame_id'],esc(r['prediction']['label']),esc(a.get('unknown_category') or '—'),str(r['observed']),esc(r['target_id']),fmt(r['stages_s']['recovery']),esc(r['recovery']['budget_audit'].get('exhaustion_stage')),esc(r['tracking_reason'])])
        sections.append(f'<details><summary>{esc(m["trial"])}：unknown 与补检帧</summary>'+table(['源帧','输出','独立归因','observed','目标ID','补检 s','预算退出阶段','跟踪原因'],rows)+'</details>')
    sections.append('</section><section id="video"><h2>第2次重复：同墙钟可播放对照</h2><p>代表重复在查看结果前已固定。源图只在其模型输出完成后出现，等待期间明确标记HOLD；视频停留不代表新预测。播放显示量化至下一个30 Hz刻度，数值保留实际测量。</p>')
    if (out/'media/metadata.json').exists():
        v=read(out/'media/metadata.json')['videos'][0]
        sections.append(f'<video id="comparison-video" controls preload="metadata" poster="{esc(v["poster"])}"><source src="{esc(v["file"])}" type="video/mp4"></video><div><button id="unknown-video-toggle" onclick="toggleVideo()">播放 / 暂停</button><button onclick="seekVideo(2.7)">首次 falling 附近</button><button onclick="seekVideo(3.3)">接地前 3.3 秒</button><button onclick="seekVideo(3.336)">接地区间开始</button><button onclick="seekVideo(4.1)">补检后的动作历史</button><button onclick="seekVideo(4.7)">第二次补检附近</button><button onclick="document.querySelector(\'video\').playbackRate=.5">0.5× 慢放</button><button onclick="document.querySelector(\'video\').playbackRate=1">1× 原速</button></div>')
    else:sections.append('<p>视频尚未生成。</p>')
    sections.append('</section><section id="errors"><h2>错误归因与解释边界</h2>')
    for e in errs:sections.append(f'<div class="card"><b>{esc(e["id"])} · {esc(e["finding"])}</b><p>{esc(e["outcome"])}</p><p>{esc(e["explanation"])}</p><p class="muted">证据：{esc(e["evidence"])}</p></div>')
    sections.append('</section><section id="pending"><h2>人工标注仍待定，未自动升级</h2><p>保留四项已确认决定：接地101–107、稳定倒地132–138且采用0.5秒判据，精确帧未知；UTK s01的722与s10的790仅是首次明确可见上升代理，物理离座仍待定。本轮没有修改标注。</p>')
    sections.append(table(['素材','项目','状态','剩余事项'],[[esc(i[k]) for k in ['clip_id','item','status','detail']] for i in pending['summary_zh']]))
    sections.append('</section><section id="files"><h2>完整证据</h2><p>')
    links=[('comparison.csv','6次指标CSV'),('comparison.json','汇总与共同帧差异'),('all-source-frames.csv','960行全源帧CSV'),('all-source-frames.json','全源帧JSON'),('diagnosis.json','旧日志归因'),('action-audit.json','新日志动作审计'),('error-attribution.json','错误归因'),('pending-annotations.json','待复核114项'),('protocol.json','冻结协议'),('results.json','原始汇总'),('annotations.reviewed.snapshot.json','已复核标注快照'),('validation.json','独立验证'),('tests.xml','审计测试'),('unit-tests.xml','单元测试'),('media/metadata.json','视频来源与解码验证')]
    sections.append(' · '.join(f'<a href="{p}">{label}</a>' for p,label in links if (out/p).exists()))
    sections.append('</p><details><summary>6次逐帧原始日志</summary><ul>'+''.join(f'<li>{esc(m["trial"])}：<a href="trials/{m["trial"]}/frames.jsonl">frames.jsonl</a> · <a href="trials/{m["trial"]}/summary.json">summary.json</a></li>' for m in c['trial_metrics'])+'</ul></details></section>')
    sections.append(f'<footer>V7 SHA-256：<code>{esc(protocol["model_sha256"])}</code><br>报告与视频仅由已完成日志生成，没有训练或视觉模型推理。三对重复描述本机同一开发事件，不能证明泛化或真实通知延迟。</footer></main><script>function toggleVideo(){{const v=document.querySelector("video");if(v.paused)v.play();else v.pause();}}function seekVideo(t){{const v=document.querySelector("video");v.currentTime=t;v.play();}}</script></html>')
    (out/'index.html').write_text('\n'.join(sections))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,default=DEFAULT);p.add_argument('--skip-plots',action='store_true')
    args=p.parse_args();out=args.experiment.resolve();results=read(out/'results.json');protocol=read(out/'protocol.json')
    assert results['complete'] and results['all_frozen_hashes_unchanged'] and protocol['training'] is False
    audit=read(out/'action-audit.json');diagnosis=read(out/'diagnosis.json')
    frames={t['trial']:[json.loads(l) for l in (out/'trials'/t['trial']/'frames.jsonl').read_text().splitlines()] for t in results['trials']}
    c=compare(results['trials'],frames,protocol,audit);write(out/'comparison.json',c)
    keys=[k for k,v in c['trial_metrics'][0].items() if isinstance(v,(str,int,float,bool)) or v is None]
    with (out/'comparison.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(c['trial_metrics'])
    full_frame_inventory(out,c,frames,protocol)
    errs=errors(c);write(out/'error-attribution.json',{'schema':'fallpre.v7.unknown_errors.v1','errors':errs})
    pending_path=out.parent/'experiment3-20261021/pending-annotations.json'
    pending=read(pending_path);(out/'pending-annotations.json').write_bytes(pending_path.read_bytes())
    if not args.skip_plots:plots(out,c,frames)
    render(out,c,errs,frames,protocol,diagnosis,pending)
    print(json.dumps({'report':str(out/'index.html'),'trials':len(c['trial_metrics']),'inventory_rows':960,'cells':c['cells']},ensure_ascii=False))


if __name__=='__main__':main()
