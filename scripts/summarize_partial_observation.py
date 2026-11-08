"""Report the separation between partial target presence and usable action pose."""
import html
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
from fallpre.weekly.human import open_video_writer,sha256

OUT=ROOT/'runs/partial-observation-20261020'
BASE=ROOT/'runs/pose-refinement-20261020'
FRESH=ROOT/'runs/partial-observation-fresh-20261020'


def rows(p):return [json.loads(l) for l in p.read_text().splitlines()]


def main():
    summary=json.loads((OUT/'summary.json').read_text());checks={};streams={}
    for s in summary['clips']:
        cid=s['clip'];stream=rows(OUT/cid/'frames.jsonl');streams[cid]=stream;old=rows(BASE/cid/'frames.jsonl')
        changed=[];native_changed=[];lost=[]
        for r,b in zip(stream,old):
            n=len(r['native_pose_instances'])
            native=lambda side:{a['index'] for a in r[side]['admission'] if a['accepted'] and a['index']<n}
            if native('before')!=native('after'):native_changed.append(r['original_frame_id'])
            before=b['pose_instances'][r['before']['target_index']] if r['before']['observed'] else None
            after=r['pose_instances'][r['after']['target_index']] if r['after']['observed'] else None
            if before!=after:changed.append(r['original_frame_id'])
            if before and not after:lost.append(r['original_frame_id'])
        checks[cid]=dict(selected_pose_changed=changed,native_admission_changed=native_changed,lost_observations=lost,
            missing_after=[r['original_frame_id'] for r in stream if not r['after']['observed']],
            action_phase_identical=all((r['before']['prediction']['label'],r['before']['prediction']['phase'])==
                                      (r['after']['prediction']['label'],r['after']['prediction']['phase']) for r in stream))
        assert not native_changed and not lost
    details=[]
    for r in streams['urfd-fall-01-original']:
        if r['after']['presence_only']:
            p=r['pose_instances'][r['after']['target_index']]
            assert r['after']['prediction']['label']=='unknown' and r['after']['prediction']['pose_input_withheld']
            details.append(dict(frame=r['original_frame_id'],joints=[i for i,v in enumerate(p['joint_supported']) if v],
                evidence=r['recovery']['refinement']['partial_body_evidence'],action_eligible=p['action_eligible']))
    checks['partial_evidence']=details
    mc=streams['multicam-chute01-cam8'];ly=[r for r in mc if 1109<=r['original_frame_id']<=1285]
    empty=[r for r in mc if 112<=r['original_frame_id']<=843]
    checks['mc_lying']=dict(frames=len(ly),observed=sum(r['after']['observed'] for r in ly))
    checks['mc_empty']=dict(frames=len(empty),observed=sum(r['after']['observed'] for r in empty))
    fresh=[]
    for cid in ['urfd-fall-01-original','utk-s01_e01-transition']:
        a=rows(FRESH/cid/'frames.jsonl');b=streams[cid]
        c=dict(clip=cid,frames=len(a),same_native=all(x['native_pose_instances']==y['native_pose_instances'] for x,y in zip(a,b)),
            same_detect=all(x['detector_instances']==y['detector_instances'] for x,y in zip(a,b)),
            same_augmented=all(x['instances']==y['pose_instances'] for x,y in zip(a,b)),
            same_target=all(x['target_index']==y['after']['target_index'] for x,y in zip(a,b)),
            same_prediction=all(x['prediction']==y['after']['prediction'] for x,y in zip(a,b)))
        assert len(a)==len(b) and all(v for k,v in c.items() if k.startswith('same_'));fresh.append(c)
    (OUT/'fresh-integration-check.json').write_text(json.dumps(fresh,indent=2))
    (OUT/'regression-checks.json').write_text(json.dumps(checks,indent=2))
    cases=[
        ('urfd-partial','urfd-fall-01-original',4.1,4.9,.25,'URFD 134–135：局部人体可见，动作仍未知',
         '两帧均从无目标变成局部目标观测，每帧保留 6 个当前图像支持的关节。共同证据是肘—腕 7–9 与膝—踝 13–15；不要求它们组成连续三关节链。右侧 PRESENCE ONLY / action withheld 表示目标可见，但没有送入动作模型。默认 0.25× 慢放仅改变播放器速率。'),
        ('urfd-incomplete','urfd-fall-01-original',3.4,4.1,.25,'仍未解决：骨架不完整、关节位置未获真值验证',
         '第 113–118 帧继续沿用上轮部分骨架，头面、手腕或腿部仍可能缺失或偏位。本轮没有新增这些关节，也没有从相邻帧复制坐标。134–135 恢复目标不代表能判断当前动作、跌倒持续状态或安全。'),
        ('urfd-lowres','urfd-fall-01',3.4,4.2,.25,'仍未解决：同事件低清第 117 帧仍漏检',
         '低清 RGB 中第 117 帧没有足够的当前人体候选，局部准入条件不能凭空创建人体，结果仍未知。它与原始 RGB 属于同一事件。'),
        ('multicam-target','multicam-chute01-cam8',1049/30,1510/30,1,'MultiCam：躺地目标、椅子与旁人进入回归',
         '全片选中人体框和骨架与上轮逐帧一致；躺地区间 177/177 帧有观测，未重新转选椅子。帧 1413 的既有短缺口仍存在。30 FPS 是展示速度，物理 FPS 未确认，继续禁用动作推理。'),
        ('multicam-empty','multicam-chute01-cam8',449/30,550/30,1,'MultiCam：空房椅子回归',
         '空房复核区间 112–843 共 732 帧仍无目标观测。左缘椅子原始假框仍被拒绝，原生 Pose 的准入集合没有变化。'),
        ('utk-regression','utk-s10_e01-transition',3.9,9.9,1,'UTK：动作输出保持，坐姿误判仍在',
         '两段 UTK 共 164 个观察，选中姿态与动作/阶段输出保持；坐姿内段 654–754 中仍有 23/25 个 forward_lean。这是旧动作模型失败，本轮没有训练或修改它。'),
    ]
    folder=OUT/'cases';folder.mkdir(exist_ok=True);records=[]
    for name,cid,start,end,rate,title,detail in cases:
        cap=cv2.VideoCapture(str(OUT/cid/'comparison.mp4'));writer,codec=open_video_writer(folder/(name+'.mp4'),30,(1280,720));i=n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if round(start*30)<=i<round(end*30):writer.write(frame);n+=1
            i+=1
        cap.release();writer.release()
        records.append(dict(id=name,clip=cid,start=start,end=end,frames=n,playback_rate=rate,title=title,detail=detail))
    (OUT/'cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    validation=[]
    for path in sorted([*OUT.glob('*/comparison.mp4'),*folder.glob('*.mp4'),*FRESH.glob('*/overlay.mp4')]):
        cap=cv2.VideoCapture(str(path));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(720,1280,3);n+=1
        cap.release();assert n==expected and n>0
        validation.append(dict(file=str(path.relative_to(ROOT)),frames=n,complete_decode=True))
    (OUT/'video-validation.json').write_text(json.dumps(validation,indent=2))
    files=['src/fallpre/weekly/pose_refinement.py','src/fallpre/weekly/pose_recovery.py','src/fallpre/weekly/verified_tracking.py',
           'scripts/replay_public_samples.py','scripts/compare_partial_observation.py','scripts/summarize_partial_observation.py','tests/test_partial_observation.py']
    snap=OUT/'final-code';snap.mkdir(exist_ok=True)
    for file in files:shutil.copyfile(ROOT/file,snap/Path(file).name)
    (OUT/'final-code-hashes.json').write_text(json.dumps({f:sha256(ROOT/f) for f in files},indent=2))
    esc=html.escape
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>URFD 局部人体准入修复 · 2026-10-20</title>
<style>body{max-width:1220px;margin:auto;padding:26px;background:#f3f5f7;color:#17212b;font:16px/1.75 system-ui,sans-serif}h1{font-size:30px}h2{margin-top:36px}a{color:#145d91}.card{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:22px;margin:22px 0}video{width:100%;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #cad4db;text-align:left}th{background:#e5edf2}.note{padding:16px;background:#fff8e7;border-left:4px solid #ba8218}code,pre{background:#e7edf2}pre{padding:16px;white-space:pre-wrap}button{padding:8px 15px;margin:5px;cursor:pointer}</style>
<h1>URFD 134–135：局部人体不再被完整骨架条件整体拒绝</h1>
<p>2026-10-20 · 无训练、无新权重 · 当前图像证据 · 不按帧号推理、不复制历史骨架</p>
<div class="note"><strong>已修复目标层漏检：</strong>原始 URFD 第 134、135 帧均恢复局部人体观测，各保留 6 个当前图像支持的关节。躺地区间目标缺失 <strong>2→0</strong>；全片仅首帧确认等待仍无目标（3→1）。<br><strong>动作保持未知：</strong>这些局部点不足以送入动作模型，显示 PRESENCE ONLY / action withheld。完整骨架、躺地状态判断与报警保持仍未解决。MultiCam 目标选择与椅子准入、UTK 回归均保持。</div>
<p><a href="#cases">6 段重点视频</a> · <a href="#metrics">回归统计</a> · <a href="#limits">剩余失败</a> · <a href="#full">8 段完整对照</a> · <a href="regression-checks.json">逐项证据</a></p>
<h2>原因与修复边界</h2><p>两帧原先均有 6 个图内、置信度和多视图一致性合格的关节，其中 4 个身体关节获当前 RTMO 与 RTMPose 共同支持。旧规则额外要求肩—肘—腕或髋—膝—踝等连续三关节链，因此缺肩、缺髋时把整个目标拒绝了。</p>
<p>新规则独立判断“局部人体仍可见”：至少 6 个已通过原检查的关节、4 个跨模型对应的身体关节、两条不共用关节的标准肢体线段，且每段长度为当前人体框对角线的 0.02–0.60。134–135 的证据是肘—腕（7–9）和膝—踝（13–15）。阈值与规则不接收帧号、机位、椅子坐标或动作参考标签。</p>
<p>此分支只接纳已确认目标的当前图像候选，沿用身份、时效、几何关联与关节置信度要求；不初始化新人、不填补过期身份。它输出 <code>observation_quality=partial_person</code>、<code>action_eligible=false</code>。跟踪器保留实际当前观测，动作模型收到缺失姿态输入，输出 unknown；不使用人体框推造肩髋，不用过去骨架补点。</p>
<p>接入范围仍为公开样本入口。默认 <code>--recovery refined</code> 开启局部观测；<code>--recovery refined-strict</code> 关闭本次新分支，复现上轮三关节链要求。原始 V7 权重、生产 <code>fallpre run</code> 与历史 weekly 入口未更改。</p>
<h2 id="cases">重点可播放结果</h2><p>左侧上轮，右侧本轮。绿色为选中目标，红色为拒绝，橙色为其他准入人体。URFD 短片默认 0.25× 慢放，只改播放器速度，不改源时钟、推理帧或日志。</p>''']
    for r in records:
        parts.append(f'''<section class="card"><h3>{esc(r['title'])}</h3><p>{esc(r['detail'])}</p><video controls preload="metadata" playsinline onloadedmetadata="this.playbackRate={r['playback_rate']}" src="cases/{r['id']}.mp4"></video><button onclick="this.parentElement.querySelector('video').playbackRate=.25">0.25× 慢放</button><button onclick="this.parentElement.querySelector('video').playbackRate=1">1× 播放</button></section>''')
    parts.append('<h2 id="metrics">回归统计</h2><p>8 路输入、2483 个原始观察。目标不可用含初始确认与空房，不是漏检率。所有之前可用的目标仍可用，原生 Pose 的准入集合保持。</p><table><tr><th>输入</th><th>观察数</th><th>目标不可用 前→后</th><th>局部观测专用帧</th><th>动作/阶段变化帧</th></tr>')
    for s in summary['clips']:
        parts.append(f'<tr><td>{s["clip"]}</td><td>{s["frames"]}</td><td>{s["missing_before"]} → {s["missing_after"]}</td><td>{s["presence_only_frames"]}</td><td>{s["changed_actions"]} / {s["changed_phases"]}</td></tr>')
    parts.append('''</table><ul><li>MultiCam 全片选中人体框、骨架与基线逐帧一致。躺地区间 1109–1285 为 177/177 观测；空房区间 112–843 为 0/732 观测。没有新增椅子准入或转选。</li><li>UTK 两段共 164 个观察未触发局部分支，动作与阶段保持；4 段非跌倒输入仍无 falling。URFD 两种分辨率属于同一事件，不能当成两个独立成功事件。</li></ul>
<h3>两帧的实际局部证据</h3><table><tr><th>原帧</th><th>当前支持的关节索引</th><th>跨模型不相交线段</th><th>动作可用</th></tr>''')
    for d in details:
        parts.append(f'<tr><td>{d["frame"]}</td><td>{d["joints"]}</td><td>{d["evidence"]["supported_segments"]}</td><td>否，unknown</td></tr>')
    parts.append('''</table><p>COCO17 的 7/9 为同侧肘/腕，13/15 为同侧膝/踝。<a href="urfd-fall-01-original/review/000134.jpg">134 帧对照图</a> · <a href="urfd-fall-01-original/review/000135.jpg">135 帧对照图</a>。本轮代理目视复核人体位置，无独立人工关节真值；一致性不保证解剖定位正确。</p>
<h2 id="limits">剩余失败与限制</h2><ul>
<li><strong>原始 134–135 的完整骨架与动作仍未知。</strong>没有肩、髋等足够证据，不能从两条可见肢体线段推断躺地姿态、速度或预警。目标被检出不意味着报警决策已经正确。</li>
<li><strong>低清 URFD 第 117 帧仍漏检。</strong>没有足够当前候选，局部规则也不创建人体。第 113–118 的部分骨架偏位、缺点和第 136 的局部下半身均沿用上轮结果。</li>
<li><strong>MultiCam 第 1413 帧仍有既有短暂缺口。</strong>本轮没有改善该候选不足问题。椅子原始假检测仍存在，只是在准入/目标层被拒绝；多人贴近交叉与模型共同认错尚无充分验证。</li>
<li><strong>UTK 坐姿阶段误判仍在。</strong>s10 坐姿内段 23/25 个 forward_lean；未训练动作模型。本轮仅解耦目标可见与动作可分析，未增加躺地状态或报警保持逻辑。</li>
<li>当前 CPU 复杂补检仍约 1 秒以上，离线视频可播放不代表实时 30 FPS。局部规则不增加模型推理次数；计算量主要来自既有 RTMO 和 8 次当前图像 RTMPose。</li>
<li>这是既有开发样本回归，不是独立盲测；没有官方逐帧人框与关节真值，不能宣称检测召回率或姿态准确率达到 100%。MultiCam 物理 FPS 未确认，继续仅帧级判断；其源 AVI 宏块警告仍保留。</li></ul>
<h2>验证与复现</h2><p>108 项相关测试通过，覆盖不相交线段、塌缩/超长线段拒绝、平移尺度不变性、当前证据、动作输入隔离及 V2–V7 回归。另从 RGB 重新运行 UTK s01 和 URFD 原始 fall01 共 233 帧，检测、当前补检姿态、目标选择与完整预测逐项一致。<a href="tests.xml">测试记录</a> · <a href="fresh-integration-check.json">新入口一致性</a>。</p>
<p>基线 <code>runs/pose-refinement-20261020</code> 未覆盖。修改前代码在 <code>baseline-code/</code>，最终代码在 <code>final-code/</code>。对照复用相同原生 YOLO Pose/Detect 缓存，当前 RGB 的 RTMO/RTMPose 重新推理；输入、基线日志、模型权重均校验散列。没有下载新权重、训练或安装依赖。模型获取与配置沿用上轮记录。</p>
<pre># 新版，必须选择新目录
.venv/bin/python scripts/replay_public_samples.py --output runs/新的局部观测实验
# 上轮三关节链准入基线
.venv/bin/python scripts/replay_public_samples.py --recovery refined-strict --output runs/新的严格基线
# 本轮前后对照
.venv/bin/python scripts/compare_partial_observation.py --output runs/新的局部观测对照</pre>
<p>目录名为复现模板；本轮已运行完整对照与两段 RGB 全链路回归。<a href="protocol.json">冻结配置与输入模式</a> · <a href="final-code-hashes.json">代码散列</a> · <a href="video-validation.json">视频解码验证</a>。</p>
<h2 id="full">8 段完整前后对照</h2>''')
    for s in summary['clips']:
        cid=s['clip'];parts.append(f'<section class="card"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/comparison.mp4"></video><p><a href="{cid}/frames.jsonl">逐帧目标、局部证据和动作输入</a></p></section>')
    parts.append(f'<p>{len(validation)} 个 H.264 视频完整解码。所有报告中的帧号与区间仅用于审计，不进入推理规则。</p></html>')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(videos=len(validation),partial_frames=details,selected_changes={c:v['selected_pose_changed'] for c,v in checks.items() if 'selected_pose_changed' in v}),ensure_ascii=False))


if __name__=='__main__':main()
