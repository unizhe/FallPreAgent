"""Standalone research figures for the reviewed V7 timing audit."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/fallpre-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/abc-v7-20261020/experiment3-20261021'
s=json.loads((OUT/'timing.json').read_text())
b=json.loads((OUT/'benchmark/summary.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,ax=plt.subplots(figsize=(11,4.4),layout='constrained')
ax.axvspan(3.336,3.537,color='#dfb060',alpha=.45,label='Human contact interval')
ax.axvspan(4.371,4.571,color='#81b7aa',alpha=.5,label='Human stable-onset interval')
ax.axvline(2.736,color='#6d7780',ls='--',label='Official onset proxy (unreviewed)')
ax.axvline(2.869,color='#17627b',lw=2,label='First V7 falling: A / B / C')
ax.text(2.869,.86,'f87\n2.869 s',ha='left',color='#17627b')
ax.text(3.4365,.57,'f101–107\n3.336–3.537 s',ha='center')
ax.text(4.471,.76,'f132–138\n4.371–4.571 s',ha='center')
ax.annotate('',(3.336,.3),(2.869,.3),arrowprops={'arrowstyle':'<->','color':'#17627b'})
ax.text(3.1025,.35,'0.467 s minimum',ha='center',fontsize=10)
ax.set(xlim=(2.5,5.35),ylim=(0,1),yticks=[],xlabel='Published source time (s)',title='URFD fall01: source-time alignment; no processing latency included')
ax.legend(loc='lower right',fontsize=9)
fig.savefig(OUT/'source-timeline.png',dpi=180);fig.savefig(OUT/'source-timeline.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
colors={'A':'#16627a','B':'#b7702d','C':'#8b537e'}
for g in b['groups']:
 mode=g['group'];rows=[json.loads(l) for l in (OUT/f'benchmark/{mode}.frames.jsonl').read_text().splitlines()]
 axes[0].plot([r['source_time_s'] for r in rows],[r['local_arrival_to_output_s'] for r in rows],color=colors[mode],label=mode)
 hit=g['raw_falling_trigger'];axes[0].scatter([hit['source_time_s']],[hit['local_arrival_to_output_s']],color=colors[mode],s=35)
 y='ABC'.index(mode)
 axes[1].scatter([hit['action_output_wall_offset_s']],[y],color=colors[mode],s=65)
 axes[1].text(hit['action_output_wall_offset_s']+.12,y+.08,f"{mode}: {hit['action_output_wall_offset_s']:.3f} s",color=colors[mode])
axes[0].set(xlabel='Input frame source time (s)',ylabel='Arrival to action output (s)',title='FIFO accumulation across the full clip')
axes[0].legend(title='Mode');axes[0].grid(alpha=.15)
axes[1].axvspan(3.336,3.537,color='#dfb060',alpha=.5)
axes[1].axvline(2.869,color='#6d7780',ls='--')
axes[1].text(3.55,-.62,'Contact window: 3.336–3.537 s',fontsize=9)
axes[1].text(2.75,2.55,'f87 arrives\n2.869 s',ha='right',fontsize=9)
axes[1].set(xlim=(2.5,8.35),ylim=(-.85,2.9),yticks=[0,1,2],yticklabels=['A','B','C'],xlabel='Wall time from first scheduled arrival (s)',title='First falling output after local processing')
fig.suptitle('One warm-start CPU run per mode · FIFO, no frame drops · no notification delivery',fontsize=12)
fig.savefig(OUT/'queue-latency.png',dpi=180);fig.savefig(OUT/'queue-latency.svg');plt.close(fig)
print('Saved source and queue timing figures')
