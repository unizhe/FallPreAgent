"""Render the frozen before/after audit and export playable case excerpts."""
import html
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
from fallpre.weekly.human import open_video_writer,sha256

OUT=ROOT/'runs/verified-tracking-20261020-v2'


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    cases=[
        ('multicam-fall','multicam-chute01-cam8',1049/30,1300/30,
         'MultiCam：目标选错已修复；没有人体观测时仍然未知',
         '原帧 1135、1252：左边旧版选中左缘椅子；右边新版继续关联床垫上的演员，逻辑目标保持 2。红框是被拒绝的原始 Pose 假检测。帧 1285 没有演员的 Pose，新版输出未知；没有把躺地人补画出来。'),
        ('multicam-empty','multicam-chute01-cam8',449/30,550/30,
         'MultiCam：空房中的椅子假骨架不再建立目标',
         '代表帧 501：旧版把左缘椅子当人持续跟踪；新版没有得到人类别确认，因此拒绝准入。算法没有使用左侧位置、房间遮罩、椅子坐标或这个片段的帧号。'),
        ('urfd-lowres','urfd-fall-01',3.3,160/30,
         'URFD 低清回归：目标不可用帧 15 → 7，跌倒输出逐帧保留',
         '原版帧 113–127 目标不可用，后来 ID 1 换成 3。新版只延续当前实际观测，短缺口内保留最近一次交叉确认的位置依据，避免部分身体框覆盖完整人体后无法接回。全片只建立一个逻辑目标。7 帧包含新目标初始确认等待 1 帧。'),
        ('urfd-missing','urfd-fall-01-original',3.5,4.7,
         '仍未解决：URFD 原始 RGB 的躺地漏检测',
         '原帧 113–118 和 134–136 共 9 帧没有 YOLO-Pose 人体观测；新版同样不能输出有效骨架。全片目标不可用 11 → 10 帧，新版其中 1 帧是初始确认等待。第 119、137 帧有 Pose 时可短缺口接回。'),
        ('utk-phase','utk-s10_e01-transition',3.9,9.9,
         '仍未解决：UTK 坐姿阶段被判成前倾',
         '原帧 654–754 的 25 个坐姿观察中，23 帧 phase 仍为 forward_lean，2 帧 seated；主动作均为 background。此次修复没有改动作模型，坐姿阶段误判保留。真实起身内段 798–818 的 6 个观察仍检出 standing_up。'),
    ]
    folder=OUT/'cases';folder.mkdir(exist_ok=True);records=[]
    for name,cid,start,end,title,detail in cases:
        cap=cv2.VideoCapture(str(OUT/cid/'comparison.mp4'))
        writer,codec=open_video_writer(folder/(name+'.mp4'),30,(1280,720));i=n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if round(start*30)<=i<round(end*30):writer.write(frame);n+=1
            i+=1
        cap.release();writer.release()
        records.append(dict(id=name,clip=cid,start_preview_s=start,end_preview_s=end,title=title,detail=detail,frames=n,codec=codec))
    (OUT/'cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    validation=[]
    paths=list(OUT.glob('*/comparison.mp4'))+list(folder.glob('*.mp4'))+list((OUT/'fresh-integration').glob('*/overlay.mp4'))
    for path in sorted(paths):
        cap=cv2.VideoCapture(str(path));declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(720,1280,3);n+=1
        cap.release();assert n==declared and n>0
        validation.append(dict(file=str(path.relative_to(OUT)),frames=n,fps=fps,complete_decode=True,bytes=path.stat().st_size))
    (OUT/'video-validation.json').write_text(json.dumps(validation,indent=2))
    (OUT/'final-code-hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha256(p) for p in [
        ROOT/'src/fallpre/weekly/verified_tracking.py',ROOT/'scripts/replay_public_samples.py',
        ROOT/'scripts/compare_verified_tracking.py',ROOT/'tests/test_verified_tracking.py']},indent=2))
    esc=html.escape
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MultiCam 修复与 URFD / UTK 回归 · 2026-10-20</title>
<style>body{max-width:1220px;margin:auto;padding:26px;background:#f3f5f7;color:#17212b;font:16px/1.75 system-ui,sans-serif}h1{font-size:30px}h2{margin-top:35px}a{color:#145d91}.card{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:22px;margin:22px 0}video{width:100%;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #cad4db;text-align:left}th{background:#e5edf2}.note{padding:16px;background:#fff8e7;border-left:4px solid #ba8218}.small{font-size:14px;color:#485762}code,pre{background:#e7edf2}pre{padding:16px;white-space:pre-wrap}</style>
<h1>MultiCam 椅子误检与目标选择修复</h1>
<p>2026-10-20 · 未训练、未修改模型权重或动作阈值 · 保留原版与首次修复回归 · MultiCam 仅按帧分析。</p>
<div class="note"><strong>已改善：</strong>帧 1135、1252 不再转选椅子，继续选择床垫演员；帧 1285 拒绝椅子并输出未知。URFD 低清 fall01 的目标不可用帧 15 → 7；所有 URFD/UTK 的 falling、standing_up 输出掩码均逐帧一致。<br><strong>仍有缺口：</strong>Pose 本身仍会漏掉躺地人、给椅子产生假骨架；此次修复在准入和关联层阻止它被当成目标。UTK 坐姿阶段误判、多人交叉身份及实时性能尚未解决。</div>
<p><a href="#cases">5 段重点对照</a> · <a href="#regression">回归统计</a> · <a href="#full">全部 8 段</a> · <a href="protocol.json">冻结配置与权重</a> · <a href="regression-checks.json">逐帧回归检查</a> · <a href="video-validation.json">视频解码验证</a></p>
<h2>改了什么</h2>
<ol><li><strong>新目标需要交叉确认：</strong>使用本地冻结的 YOLO11n Detect 为 Pose 框提供 person 类别证据（分数 ≥0.25，IoU ≥0.30），并要求同一 ByteTrack 连续两次得到确认。未确认 Pose 留在日志和红框中，不成为跟踪目标。</li>
<li><strong>已有目标允许短缺口恢复：</strong>不把 Detect 识别成 chair/suitcase 当成绝对否决，避免错杀坐着／躺着的人。用最近观测及短期交叉确认框，检查相对中心距离、框重叠、面积变化；只输出当前帧真实的 Pose。短缺口内逻辑目标可保持，ByteTrack 原始 ID 单独展示。</li>
<li><strong>歧义与过期显式处理：</strong>相邻候选难以区分时输出未知。过期后必须重新确认并创建新 epoch，不把新目标当成先前绑定的老人。当前实验入口自动选新 epoch；它不构成真实照护对象的身份重绑定。</li>
<li><strong>位置与场景无关：</strong>修复组件只接收人体框、骨架、检测类别和当前／历史观测，不接收数据集名、参考动作、椅子坐标或事件帧号。空间约束按人体框尺度归一化。合成测试覆盖平移与缩放后的结果一致性。</li></ol>
<p>接入范围：<code>scripts/replay_public_samples.py</code> 默认使用 verified；<code>--tracking legacy</code> 保留旧逻辑。原 V7 模型和历史 weekly 演示、生产 <code>fallpre run</code> 未自动切换此策略。本轮验证的是公开 RGB 样本推理入口及可复用跟踪组件。</p>
<h2>对照方法与时间边界</h2>
<ul><li>8 路输入共 2483 个原始观察。对照使用前轮完全相同的 Pose 日志，额外的 Detect 在第一修订中逐帧运行；第二修订复用经校验的同一 Detect 观测。所有输入、Pose 日志、权重和算法文件都有散列记录。</li>
<li>另以新入口对 UTK s01 和低清 URFD fall01 共 233 帧重新运行两个视觉模型；Pose、Detect、目标选择和动作输出与对照结果逐项一致。</li>
<li>MultiCam 不运行动作模型，不算真实速度、延迟或预警提前量。恢复窗口明确使用 15 个观察帧；ByteTrack 内部合成时钟仅作逐帧 bookkeeping。播放器 30 FPS 不是物理采集速度。URFD/UTK 继续使用已有源时钟。</li>
<li>对照左侧旧版，右侧新版；绿色是选中目标，橙色是其他准入观测，红色是被拒绝的 Pose。左侧 Target 是旧 ByteTrack ID；右侧 Target 是新逻辑 epoch，不能直接比较两个数字是否相等；框上的 BT 才是原始 ByteTrack ID。</li></ul>
<h2 id="cases">5 段重点对照：修复与残余失败</h2>''']
    for r in records:
        parts.append(f'<section class="card"><h3>{esc(r["title"])}</h3><p>{esc(r["detail"])}</p><video controls preload="metadata" playsinline src="cases/{r["id"]}.mp4"></video></section>')
    parts.append('''<h2 id="regression">全部输入回归</h2><p>“目标不可用”计入开场确认等待；不等于人体漏检率。尤其 MultiCam 旧版很多“可用目标”其实是椅子，不能把减少未知当成唯一目标。</p>
<table><tr><th>输入</th><th>观察帧</th><th>目标不可用：前 → 后</th><th>主动作标签变动帧</th><th>新版建立的逻辑 epoch</th></tr>''')
    for s in summary['clips']:
        parts.append(f'<tr><td><a href="#{s["clip"]}">{s["clip"]}</a></td><td>{s["frames"]}</td><td>{s["no_target_before"]} → {s["no_target_after"]}</td><td>{s["changed_action_frames"]}</td><td>{esc(str(s["observed_epochs"]))}</td></tr>')
    parts.append('''</table><ul>
<li>URFD fall01 的原始和低清输入属同一事件，不能重复计入检出率。两次独立 URFD 跌倒均继续检出；全部 falling 的逐帧布尔掩码保持一致。</li>
<li>UTK 两段 standing_up 的逐帧掩码保持一致，分别为 6 和 7 个观察。复核的 82 个坐姿观察仍无 standing_up；四段非跌倒输入 494 帧仍无 falling。坐姿阶段错误并未修复。</li>
<li>UTK s01/s10 与 URFD ADL04 各新增开头 1 帧目标确认等待，ADL07 增加 3 帧；相应部分早期主动作输出变为 unknown，不是所有输出完全不变。</li>
<li>MultiCam 暂定 Falling 区间 1080–1108 中 27/29 帧有目标观测；Lying 区间 1109–1285 中 127/177 帧有观测，仍有 50 帧未知。这个数是观测可用性，不是经过人体框真值认证的召回率。关键帧 1135、1252、1285 已目视检查。</li>
<li>MultiCam 的 1392 个未准入 Pose 候选不能全部计为“修正的假人”；未做人框逐帧真值标注。新全片不可用 862 帧中有较长空房阶段，旧版此时常选椅子。第 501 帧给出直接对照。</li></ul>
<h2>仍未解决与修改代价</h2><ul>
<li>没有检测出人体时仍不能提取骨架，MultiCam 第 1285 帧及 URFD 原始 fall01 的 9 帧漏检明确保留。高置信假骨架也仍存在于 Pose 原始输出。</li>
<li>第二检测器并非独立真值：若两个模型共同认错，或者真实人在首帧就躺地而两个模型都不能确认，仍可能误准入或长期未知。尚无足够开机即躺地、轮椅、遮挡老人样本。</li>
<li>空间连续性不是外观重识别。多人近距离交叉、椅子与身体高度重叠、持续部分身体观测可能导致歧义或漂移；本轮仅用合成歧义测试，不声称多人场景已验证。</li>
<li>添加 Detect 增加计算量，首个目标增加确认等待；尚未做部署机实时延迟测试。未改报警保持、JEV 决策、米制距离或生产身份绑定。</li>
<li>第一次修订使低清 fall01 不可用帧 15 → 28，原因是局部身体框覆盖了完整身体的关联依据。第二修订保留短期已确认框后降到 7。首次输出和代码保存在相邻目录 <code>runs/verified-tracking-20261020</code>，未覆盖或隐藏该退化。两轮均为开发回归，非独立盲测。</li>
<li>URFD/UTK 为历史 train/dev 数据；MultiCam 用于本轮开发。旧参考动作和坐姿内段为官方标签加代理暂定复核，并无新的独立人工仲裁。MultiCam 源 AVI 的宏块解码警告仍存在，编码后的可播放不证明源帧无损坏。</li></ul>
<h2 id="full">全部 8 段完整对照</h2>''')
    for s in summary['clips']:
        cid=s['clip'];parts.append(f'<section class="card" id="{cid}"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/comparison.mp4"></video><p><a href="{cid}/frames.jsonl">逐帧前后目标、检测证据和动作</a> · <a href="{cid}/summary.json">统计</a></p></section>')
    parts.append('''<h2>验证与复现</h2>
<p>85 项相关测试通过（新跟踪行为、旧 V2–V7）；8 个完整对照、5 个短片及 2 个新推理视频共 15 个 H.264 文件均完整解码。<a href="tests.xml">测试记录</a> · <a href="fresh-integration-check.json">新入口一致性</a> · <a href="final-code-hashes.json">代码快照校验</a>。</p>
<pre># 全部样本重新推理，必须选择新的输出目录
.venv/bin/python scripts/replay_public_samples.py --tracking verified --output runs/新目录
# 旧逻辑对照（原输出已保留，无需覆盖）
.venv/bin/python scripts/replay_public_samples.py --tracking legacy --output runs/另一个新目录
# 复用原 Pose 并重新运行 Detect 的完整对照
.venv/bin/python scripts/compare_verified_tracking.py --output runs/新的对照目录</pre>
<p>源组件：<code>src/fallpre/weekly/verified_tracking.py</code>；行为测试：<code>tests/test_verified_tracking.py</code>。本页所有计数都可由 JSONL 复算。没有用真实动作标签选择目标或修正模型输出。</p></html>''')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(videos=len(validation),cases=len(records)),ensure_ascii=False))


if __name__=='__main__':main()
