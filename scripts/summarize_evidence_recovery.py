"""Audit current-image candidate verification and preserve playable comparisons."""
import html
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from fallpre.weekly.human import open_video_writer,sha256
from fallpre.weekly.pose_refinement import visible_joints

OUT=ROOT/'runs/evidence-recovery-20261020'
BASE=ROOT/'runs/partial-observation-20261020'
FRESH=ROOT/'runs/evidence-recovery-fresh-20261020'


def rows(path):return [json.loads(l) for l in path.read_text().splitlines()]


def selected(row):
    idx=row['after']['target_index']
    return None if idx is None else row['pose_instances'][idx]


def geometry(person):return None if person is None else {k:person[k] for k in ['box','kpts']}


def main():
    summary=json.loads((OUT/'summary.json').read_text());streams={};checks={}
    for s in summary['clips']:
        cid=s['clip'];current=rows(OUT/cid/'frames.jsonl');old=rows(BASE/cid/'frames.jsonl');streams[cid]=current
        assert len(current)==len(old)==s['frames']
        changed=[];native_changed=[];lost=[];prediction_changes=[];epochs=[]
        for r,b in zip(current,old):
            n=len(r['native_pose_instances'])
            admit=lambda side:{a['index'] for a in r[side]['admission'] if a['accepted'] and a['index']<n}
            if admit('before')!=admit('after'):native_changed.append(r['original_frame_id'])
            before,after=selected(b),selected(r)
            if geometry(before)!=geometry(after):changed.append(r['original_frame_id'])
            if before and not after:lost.append(r['original_frame_id'])
            if r['before']['prediction']!=r['after']['prediction']:prediction_changes.append(r['original_frame_id'])
            if r['before']['target_epoch']!=r['after']['target_epoch']:epochs.append(r['original_frame_id'])
        assert not native_changed and not lost and not epochs
        expected=[117] if cid=='urfd-fall-01' else [1413] if cid=='multicam-chute01-cam8' else []
        assert changed==expected
        if cid!='urfd-fall-01':assert not prediction_changes
        checks[cid]=dict(selected_geometry_changed=changed,native_admission_changed=native_changed,lost_observations=lost,
            epoch_changed=epochs,full_prediction_changed=prediction_changes,
            missing_after=[r['original_frame_id'] for r in current if not r['after']['observed']])
    low=streams['urfd-fall-01'][116];mc=streams['multicam-chute01-cam8'];m=mc[1412]
    assert low['after']['observed'] and m['after']['observed']
    assert low['recovery']['refinement']['detector_supported_proposal']
    assert m['recovery']['trigger']=='native_pose_ambiguous'
    checks['targeted_frames']={cid:dict(frame=r['original_frame_id'],before_reason=r['before']['reason'],
        after_reason=r['after']['reason'],trigger=r['recovery']['trigger'],selected_box=selected(r)['box'],
        visible_joints=int(visible_joints(selected(r)['kpts'],shape).sum()),
        before_action=r['before']['prediction']['label'],after_action=r['after']['prediction']['label'])
        for cid,r,shape in [('urfd-fall-01',low,(240,320,3)),('multicam-chute01-cam8',m,(480,720,3))]}
    for name,start,end in [('lying',1109,1285),('empty',112,843)]:
        rs=[r for r in mc if start<=r['original_frame_id']<=end]
        checks['mc_'+name]=dict(frames=len(rs),observed=sum(r['after']['observed'] for r in rs))
    assert checks['mc_empty']['observed']==0 and checks['mc_lying']['observed']==177
    fresh=[]
    for cid in ['urfd-fall-01','multicam-chute01-cam8','urfd-fall-01-original','utk-s01_e01-transition']:
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
        ('urfd-117','urfd-fall-01',3.5,4.2,.25,'低清 URFD 117：当前候选通过复核，缺口补回',
         '3.867 秒，第 117 帧。旧版只有椅子假骨架获得 YOLO Pose 输出；真实人体 RTMO 候选未进入复核。新版保留 11 个通过一致性检查的当前关节，目标仍为 1，动作 unknown → falling。骨架仍缺点且未有人工关节真值；标签改变不证明新增了躺地状态识别。'),
        ('multicam-1413','multicam-chute01-cam8',1395/30,1430/30,.25,'MultiCam 1413：局部候选歧义经当前 RGB 补检消除',
         '旧版两个局部候选的关联分数为 0.399、0.532，差值 0.133 小于 0.15，故不强选；已有候选又阻止 RTMO 执行。新版 RTMO 当前全图给出 0.848 置信度人体框、关联分数 0.915，沿用原排序规则选中目标 2。远处两人保持为其他人，左侧椅子仍拒绝；不进行动作推理。'),
        ('multicam-fall','multicam-chute01-cam8',1070/30,1310/30,1,'MultiCam 跌倒及躺地：椅子与目标选择回归',
         '暂定躺地区间 1109–1285 的 177 帧仍全部有当前观测，框与骨架逐帧不变。视频展示 30 FPS，不用于物理速度或提前量计算。'),
        ('multicam-empty','multicam-chute01-cam8',449/30,550/30,1,'MultiCam 空房：椅子假检测仍被拒绝',
         '空房复核区间 112–843，共 732 帧，目标观测仍为 0。原生 YOLO 的椅子假框并未消失，本轮没有新增准入；这是该样本回归，不代表所有椅子误检已解决。'),
        ('original-partial','urfd-fall-01-original',3.4,4.9,.25,'剩余失败：原始 URFD 骨架不完整、局部动作未知',
         '原始 RGB 的全部选中框、骨架和完整预测与基线一致。134–135 帧仍各仅有 6 个支持关节，动作输入被隔离，输出 unknown；113–118 周边仍有缺点、位置不稳。原始和低清属于同一事件，不能计作两个独立跌倒实验。'),
        ('utk-regression','utk-s10_e01-transition',3.9,9.9,1,'剩余失败：UTK 坐姿阶段仍被判为前倾',
         '两段 UTK 共 164 个观察的目标、骨架及完整预测均不变。s10 的坐姿内段 654–754 中，23/25 个观察仍报 forward_lean。本轮未训练或修改动作模型。'),
        ('multicam-door','multicam-chute01-cam8',78/30,112/30,.25,'剩余失败：MultiCam 门口遮挡与出画交界跟踪不连续',
         '89–111 附近存在既有目标空观测，98 仅恢复一次局部观测。门口小人体、遮挡与出画混合，不能把整个区间或随后空房都计作漏检。短时恢复窗口过期后仍需重新确认身份，本轮没有利用未来帧或历史骨架延长它。'),
    ]
    folder=OUT/'cases';folder.mkdir(exist_ok=True);records=[]
    for name,cid,start,end,rate,title,detail in cases:
        cap=cv2.VideoCapture(str(OUT/cid/'comparison.mp4'));writer,codec=open_video_writer(folder/(name+'.mp4'),30,(1280,720));i=n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if round(start*30)<=i<round(end*30):writer.write(frame);n+=1
            i+=1
        cap.release();writer.release();assert n>0
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
    suites=ET.parse(OUT/'tests.xml').getroot();tests=sum(int(s.attrib.get('tests',0)) for s in suites.iter('testsuite'))
    assert all(int(s.attrib.get('failures',0))+int(s.attrib.get('errors',0))==0 for s in suites.iter('testsuite'))
    files=['src/fallpre/weekly/pose_refinement.py','src/fallpre/weekly/pose_recovery.py','src/fallpre/weekly/verified_tracking.py',
           'scripts/replay_public_samples.py','scripts/compare_evidence_recovery.py','scripts/summarize_evidence_recovery.py','tests/test_evidence_recovery.py']
    snap=OUT/'final-code';snap.mkdir(exist_ok=True)
    for file in files:shutil.copyfile(ROOT/file,snap/Path(file).name)
    (OUT/'final-code-hashes.json').write_text(json.dumps({f:sha256(ROOT/f) for f in files},indent=2))
    esc=html.escape
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>URFD 117 与 MultiCam 1413 · 当前图像证据修复</title>
<style>body{max-width:1220px;margin:auto;padding:26px;background:#f3f5f7;color:#17212b;font:16px/1.75 system-ui,sans-serif}h1{font-size:30px}h2{margin-top:36px}a{color:#145d91}.card{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:22px;margin:22px 0}video{width:100%;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #cad4db;text-align:left}th{background:#e5edf2}.note{padding:16px;background:#fff8e7;border-left:4px solid #ba8218}code,pre{background:#e7edf2}pre{padding:16px;white-space:pre-wrap}button{padding:8px 15px;margin:5px;cursor:pointer}</style>
<h1>两个目标缺口已补回：先找候选，再核对门控</h1>
<p>2026-10-20 · 冻结模型 · 当前 RGB 推理 · 无帧号规则、无历史骨架补齐</p>
<div class="note"><strong>低清 URFD 第 117 帧与 MultiCam 第 1413 帧恢复当前目标观测。</strong>前者是已有弱候选没有进入姿态复核，后者是局部候选关联有歧义且补检被跳过。全回归只有这两帧的选中框/骨架改变；原生候选准入集合不变，未新增椅子准入或丢失已有目标。<br>目标补回不等同骨架准确、躺地识别或报警闭环完成。</div>
<p><a href="#cases">重点可播放对照</a> · <a href="#metrics">回归统计</a> · <a href="#limits">剩余失败</a> · <a href="#full">全部视频</a> · <a href="regression-checks.json">逐项核对</a></p>
<h2>根因核查：修正上轮“候选不足”的笼统表述</h2>
<table><tr><th>缺口</th><th>当前候选证据</th><th>阻断位置</th><th>通用修复</th></tr>
<tr><td>低清 URFD 117</td><td>RTMO 旋转全图人体框置信度 0.479；YOLO Detect 同位置 person 0.193；RTMO 有 5 个合格身体关节。椅子假候选在另一位置。</td><td>RTMO 原质量门槛需至少 8 点，未通过；候选又低于单提案 0.60 且没有第二 RTMO 视图框吻合，故 RTMPose 未执行。不是完全没有人体候选。</td><td>仅把与当前独立 Detect 弱人体框高度重叠的 RTMO 候选送入复核；需 RTMPose 当前视图一致性及至少 4 个跨模型身体关节支持才可采用。本帧保留 11 点，4 点跨模型吻合。</td></tr>
<tr><td>MultiCam 1413</td><td>两个原生局部候选均在旧目标附近，关联分数 0.399、0.532；二者框 IoU 小于 0.5。</td><td>差值 0.133 小于歧义阈值 0.15，跟踪器拒绝强选；补检入口却仅检查“是否有候选”，因而跳过。</td><td>补检入口与跟踪器共享相同歧义判定。仅在歧义时允许当前图像补检；RTMO 全图产生 0.848 人体框、关联 0.915，随后仍用原排序与歧义阈值选择，不指定候选编号。</td></tr></table>
<p><a href="urfd-fall-01-diagnosis.json">URFD 诊断日志</a> · <a href="multicam-chute01-cam8-diagnosis.json">MultiCam 诊断日志</a>。诊断中为解释入口阻断而单独运行当前图像模型；最终对照及独立重跑均通过正式入口，诊断帧号不进入算法。</p>
<h2>实际修改与边界</h2>
<p>新模式 <code>--recovery evidence</code>：弱 Detect 框仅作复核线索（person ≥0.15、与当前 RTMO 框 IoU ≥0.70），不能直接准入新人；当前 RTMO 至少 4 个可见身体关节，RTMPose 沿用 8 视图、一致性距离与置信度检查，新增线索至少需 4 个跨模型身体关节吻合。原生新目标确认阈值仍为 0.25，连续两次确认要求不变。</p>
<p>身份、短缺口期限、几何关联、候选排序与歧义阈值保持；清晰原生候选不触发补检。历史人体框只用于短时关联/搜索，输出坐标全部来自当前图像；不输入未来帧，不复制旧骨架，不训练。真实多人仍有歧义时允许保持未知。MultiCam 继续只做帧级目标关联，不执行动作模型，不用未知物理 FPS 计算速度。</p>
<p>这些规则由已知失败样本开发，未声称是独立测试。新增阈值是可审计的工程选择，尚需跨人物、跨场景验证；不应继续针对这些片段反复调阈值后宣称泛化提升。</p>
<h2 id="cases">7 段重点结果与剩余失败</h2><p>左侧上轮，右侧本轮。绿色选中目标、红色拒绝、橙色其他人体。短片提供 0.25× 慢放；播放器变速不改变推理时钟或帧数。</p>''']
    for r in records:
        parts.append(f'''<section class="card"><h3>{esc(r['title'])}</h3><p>{esc(r['detail'])}</p><video controls preload="metadata" playsinline onloadedmetadata="this.playbackRate={r['playback_rate']}" src="cases/{r['id']}.mp4"></video><button onclick="this.parentElement.querySelector('video').playbackRate=.25">0.25× 慢放</button><button onclick="this.parentElement.querySelector('video').playbackRate=1">1× 播放</button></section>''')
    parts.append('<h2 id="metrics">回归统计</h2><p>8 路输入、2483 个原始观察。目标不可用包含确认等待、空房、出画及漏检，不直接当作漏检率。两种 URFD 分辨率是同一事件。</p><table><tr><th>输入</th><th>观察数</th><th>目标不可用 前→后</th><th>动作/阶段标签变化帧</th></tr>')
    for s in summary['clips']:
        parts.append(f'<tr><td>{s["clip"]}</td><td>{s["frames"]}</td><td>{s["missing_before"]} → {s["missing_after"]}</td><td>{s["changed_actions"]} / {s["changed_phases"]}</td></tr>')
    parts.append('''</table><ul><li>低清 URFD 全片只剩第 1 帧确认等待；117 帧动作 unknown → falling，阶段 unknown → other。新增观测影响后续时序支持和分数：117–160 完整预测记录有所不同，但其余帧动作/阶段标签不变。</li><li>原始 URFD、两段 UTK、URFD ADL04/07 与 fall02 的选中框、骨架及完整预测逐帧一致。没有新的坐下/弯腰 falling 标签。</li><li>MultiCam 仅 1413 的选中框/骨架改变，逻辑身份不换；躺地区间 177/177 观察、空房 0/732 观察，原生椅子准入集合不变。新增 RTMO 不会直接建立新身份。</li></ul>
<h2 id="limits">剩余失败</h2><ul>
<li><strong>骨架缺失与偏位：</strong>低清 117 虽补回 11 点，仍缺部分肩、肘、腕等；无人工逐关节真值，多模型一致也可能共同定位错误。原始 134–135 仍为 6 点局部人体，动作 unknown。</li>
<li><strong>动作与报警未闭环：</strong>低清 117 的 falling 是冻结时序模型输出，不能据此宣称新增了可靠躺地检测、预跌倒提醒或报警保持。完整骨架与持续躺地状态仍需解决。</li>
<li><strong>UTK 坐姿阶段误报：</strong>s10 坐姿内段仍 23/25 个 forward_lean，未训练动作模型。</li>
<li><strong>MultiCam 门口：</strong>遮挡/出画交界仍有既有短缺口，场景内全部人的身份没有持续验证；两个真实人交叉时仍可能未知。椅子原始假检测仍存在，本轮只是没有新增准入或转选。</li>
<li><strong>性能：</strong>当前复杂 CPU 补检仍可超过 1 秒，报告视频播放流畅不代表实时推理。无新模型下载或训练，未切换生产入口。</li>
<li><strong>数据边界：</strong>这是同一批开发样本回归，没有独立标注的召回率/姿态精度。MultiCam 源 AVI 的宏块解码警告仍存在；其“1285”是宏块编号，不能当帧号。预览仅采用示意 30 FPS。</li></ul>''')
    parts.append(f'''<h2>验证与复现</h2><p>{tests} 项测试通过；{len(validation)} 个 H.264 视频完整解码。对低清 URFD、原始 URFD、MultiCam 全片与 UTK s01 共 {sum(c['frames'] for c in fresh)} 帧，从 RGB 重新运行全部视觉模型，检测、补检、选中目标和完整预测与缓存对照一致。</p>
<p><a href="tests.xml">测试记录</a> · <a href="fresh-integration-check.json">RGB 全链路一致性</a> · <a href="video-validation.json">视频解码</a> · <a href="protocol.json">配置与冻结权重散列</a> · <a href="final-code-hashes.json">最终代码散列</a></p>
<p>上轮 <code>runs/partial-observation-20261020</code> 保留；修改前后代码分别在 baseline-code 与 final-code。全部输入与模型散列在推理前后核对。完整对照复用相同原生 YOLO 缓存，RTMO/RTMPose 对当前 RGB 重新推理；另有以上四段独立全链路重跑。</p>
<pre># 新版默认 evidence，使用新的输出目录
.venv/bin/python scripts/replay_public_samples.py --output runs/新的证据复核实验
# 上轮局部人体基线
.venv/bin/python scripts/replay_public_samples.py --recovery refined --output runs/新的局部观测基线
# 全部前后对照
.venv/bin/python scripts/compare_evidence_recovery.py --output runs/新的证据对照</pre>
<p>目录名为复现模板，本轮完整对照及四段 RGB 重跑均已执行。</p><h2 id="full">8 段完整前后对照</h2>''')
    for s in summary['clips']:
        cid=s['clip'];parts.append(f'<section class="card"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/comparison.mp4"></video><p><a href="{cid}/frames.jsonl">逐帧候选、门控、目标及预测</a></p></section>')
    parts.append('</html>');(OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(tests=tests,videos=len(validation),fresh_frames=sum(c['frames'] for c in fresh),targeted=checks['targeted_frames']),ensure_ascii=False))


if __name__=='__main__':main()
