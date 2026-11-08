"""Export current-image refinement evidence; all frame ranges are evaluation only."""
import html
import json
import shutil
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from fallpre.weekly.human import open_video_writer,sha256
from fallpre.weekly.pose_refinement import visible_joints

OUT=ROOT/'runs/pose-refinement-20261020'
BASE=ROOT/'runs/pose-recovery-20261020'
FRESH=ROOT/'runs/pose-refinement-fresh-20261020'


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    streams={s['clip']:rows(OUT/s['clip']/'frames.jsonl') for s in summary['clips']}
    audit={}
    for cid,stream in streams.items():
        old=rows(BASE/cid/'frames.jsonl')
        changes=[];lost=[];native_admission=[]
        for r,b in zip(stream,old):
            native=len(r['native_pose_instances'])
            before={a['index'] for a in r['before']['admission'] if a['accepted'] and a['index']<native}
            after={a['index'] for a in r['after']['admission'] if a['accepted'] and a['index']<native}
            if before!=after:native_admission.append(r['original_frame_id'])
            if r['before']['observed'] and not r['after']['observed']:lost.append(r['original_frame_id'])
            bp=b['pose_instances'][r['before']['target_index']] if r['before']['observed'] else None
            ap=r['pose_instances'][r['after']['target_index']] if r['after']['observed'] else None
            if bp!=ap:changes.append(r['original_frame_id'])
        audit[cid]=dict(native_admission_changes=native_admission,previously_observed_now_missing=lost,
                       selected_pose_changed_frames=changes,
                       missing_after=[r['original_frame_id'] for r in stream if not r['after']['observed']],
                       prediction_identical=all(r['before']['prediction']==r['after']['prediction'] for r in stream))
        assert not native_admission and not lost
    mc=streams['multicam-chute01-cam8'];lying=[r for r in mc if 1109<=r['original_frame_id']<=1285]
    empty=[r for r in mc if 112<=r['original_frame_id']<=843]
    audit['mc_lying']=dict(frames=len(lying),before=sum(r['before']['observed'] for r in lying),after=sum(r['after']['observed'] for r in lying))
    audit['mc_empty']=dict(frames=len(empty),before=sum(r['before']['observed'] for r in empty),after=sum(r['after']['observed'] for r in empty))
    nonfall=[r for cid in ['utk-s01_e01-transition','utk-s10_e01-transition','urfd-adl-04','urfd-adl-07'] for r in streams[cid]]
    audit['nonfall']=dict(frames=len(nonfall),falling_after=sum(r['after']['prediction']['label']=='falling' for r in nonfall))
    joint_review=[]
    cid='urfd-fall-01-original';old=rows(BASE/cid/'frames.jsonl')
    for fid in [113,114,115,116,117,118,134,135,136]:
        r=streams[cid][fid-1];b=old[fid-1]
        bp=b['pose_instances'][r['before']['target_index']] if r['before']['observed'] else None
        ap=r['pose_instances'][r['after']['target_index']] if r['after']['observed'] else None
        joint_review.append(dict(frame=fid,before=int(visible_joints(bp['kpts'],(480,640,3)).sum()) if bp else 0,
            after=int(visible_joints(ap['kpts'],(480,640,3)).sum()) if ap else 0,
            after_unknown=[i for i,v in enumerate(visible_joints(ap['kpts'],(480,640,3))) if not v] if ap else list(range(17))))
    audit['urfd_joint_review']=joint_review
    (OUT/'regression-checks.json').write_text(json.dumps(audit,indent=2))
    # All changed/refined poses, and unresolved URFD candidates, are reviewable.
    sheet=OUT/'audit-sheets';sheet.mkdir(exist_ok=True)
    for cid,stream in streams.items():
        selected=[r for r in stream if r['after']['refined'] or ('urfd' in cid and r['recovery']['executed'])]
        if not selected:continue
        canvas=np.full((((len(selected)+2)//3)*240,960,3),245,np.uint8)
        for i,r in enumerate(selected):
            im=cv2.imread(str(OUT/cid/'review'/f"{r['original_frame_id']:06d}.jpg"))
            y,x=divmod(i,3);im=cv2.resize(im[50:540,640:],(320,215))
            canvas[y*240:y*240+215,x*320:(x+1)*320]=im
            cv2.putText(canvas,f"{r['original_frame_id']} {'REFINED' if r['after']['refined'] else 'MISSING'}",(x*320+5,y*240+235),0,.5,(0,0,0),1)
        cv2.imwrite(str(sheet/f'{cid}.jpg'),canvas)

    cases=[
        ('urfd-progress','urfd-fall-01-original',3.4,4.1,'URFD：补回 115–118，改善已补回的骨架',
         '左侧是上轮 RTMO 版本，右侧为本轮当前图像复核。第 113、114 帧在图内且通过置信度阈值的关节由 9→12、8→11；第 115–118 帧从无目标变为局部人体观测。未通过检查的关节仍为未知，不能称完整骨架。'),
        ('urfd-residual','urfd-fall-01-original',4.1,4.9,'URFD：134–135 仍漏检；136 恢复部分下半身',
         '第 134–135 帧未通过可连接肢体链检查，仍不输出骨架。第 136 帧恢复 7 个有一致性支持的关节，肩、头等仍缺失。此时 V7 可能输出 background；它不代表“人已安全”，本轮没有实现躺地状态或报警保持。'),
        ('urfd-lowres','urfd-fall-01',3.4,4.2,'低清 URFD 回归：仍在第 117 帧漏检',
         '同一事件的低清版，躺地区间缺失 3→1 帧；第 118–119 帧补回，第 117 帧仍缺少足够当前人体候选。与原始 RGB 不可重复计为两个独立事件。'),
        ('multicam-lying','multicam-chute01-cam8',1049/30,1510/30,'MultiCam：躺地目标与旁人进入回归',
         '标注躺地区间 1109–1285 的 177 帧前后都有目标观测，选中框和骨架保持相同；1135、1252、1285 的左缘椅子继续被拒绝。后续旁人进入时仍选择床垫演员。仅帧级分析，30 FPS 为展示速度。'),
        ('multicam-empty','multicam-chute01-cam8',449/30,550/30,'MultiCam：椅子假框没有重新准入',
         '空房复核区间 112–843 共 732 帧，目标观测前后均为 0。全片只有第 98 帧新增一个目标观测，位置在门口的真实人物，逐帧截图已复核；原始椅子假检测仍保留为红框。'),
        ('utk-regression','utk-s10_e01-transition',3.9,9.9,'UTK：输出保持，坐姿阶段误判仍在',
         '两段 UTK 共 164 个原始观察，未触发新分支，完整预测逐帧一致。s10 坐姿 654–754 的 25 个观察中仍有 23 个 forward_lean；这项动作模型错误本轮未修复。'),
    ]
    folder=OUT/'cases';folder.mkdir(exist_ok=True);records=[]
    for name,cid,start,end,title,detail in cases:
        cap=cv2.VideoCapture(str(OUT/cid/'comparison.mp4'));writer,codec=open_video_writer(folder/(name+'.mp4'),30,(1280,720));i=n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if round(start*30)<=i<round(end*30):writer.write(frame);n+=1
            i+=1
        cap.release();writer.release()
        records.append(dict(id=name,clip=cid,start=start,end=end,frames=n,title=title,detail=detail,codec=codec))
    (OUT/'cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    validation=[]
    for path in sorted([*OUT.glob('*/comparison.mp4'),*folder.glob('*.mp4'),*FRESH.glob('*/overlay.mp4')]):
        cap=cv2.VideoCapture(str(path));expected=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(720,1280,3);n+=1
        cap.release();assert n==expected and n>0
        validation.append(dict(file=str(path.relative_to(ROOT)),frames=n,complete_decode=True,bytes=path.stat().st_size))
    (OUT/'video-validation.json').write_text(json.dumps(validation,indent=2))
    files=['src/fallpre/weekly/pose_refinement.py','src/fallpre/weekly/pose_recovery.py','src/fallpre/weekly/verified_tracking.py',
           'scripts/replay_public_samples.py','scripts/compare_pose_refinement.py','scripts/summarize_pose_refinement.py','tests/test_pose_refinement.py']
    snapshot=OUT/'final-code';snapshot.mkdir(exist_ok=True)
    for file in files:shutil.copyfile(ROOT/file,snapshot/Path(file).name)
    (OUT/'final-code-hashes.json').write_text(json.dumps({f:sha256(ROOT/f) for f in files},indent=2))
    esc=html.escape
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>URFD 当前图像姿态复核 · 2026-10-20</title>
<style>body{max-width:1220px;margin:auto;padding:26px;background:#f3f5f7;color:#17212b;font:16px/1.75 system-ui,sans-serif}h1{font-size:30px}h2{margin-top:36px}a{color:#145d91}.card{background:#fff;border:1px solid #dce4e9;border-radius:12px;padding:22px;margin:22px 0}video,img{max-width:100%;width:100%;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #cad4db;text-align:left}th{background:#e5edf2}.note{padding:16px;background:#fff8e7;border-left:4px solid #ba8218}code,pre{background:#e7edf2}pre{padding:16px;white-space:pre-wrap}</style>
<h1>URFD 躺地漏检与局部骨架复核</h1><p>2026-10-20 · 所有权重冻结，无训练 · 当前图像推理，不复制历史骨架 · MultiCam 仅帧级分析</p>
<div class="note"><strong>已改善：</strong>指定的 7 个 URFD 原始 RGB 漏检帧，补回 115、116、117、118、136，共 5 帧；134–135 仍未知。躺地缺失 <strong>7→2</strong>，全片目标不可用 <strong>8→3</strong>（含首帧确认）。第 113–114 帧关节覆盖有所增加，但仍非完整骨架。<br><strong>回归：</strong>MultiCam 躺地区间保持 177/177 观测，椅子未重新准入；UTK 两段完整预测逐帧一致。<strong>残余：</strong>遮挡关节、截断身体与坐姿阶段错误仍存在，姿态一致性不是关节真值。</div>
<p><a href="#cases">6 段重点对照</a> · <a href="#metrics">统计与关节证据</a> · <a href="#limits">残余失败</a> · <a href="#full">全部 8 段</a> · <a href="regression-checks.json">逐项回归</a> · <a href="protocol.json">冻结配置与散列</a></p>
<h2>修改方式</h2><p>上一轮要求肩与髋都可见，会整体拒绝一部分被画面边缘截断的人体；另一些候选只有 3–7 个可靠关节。本轮增加冻结 RTMPose-m，对当前 RTMO 候选框中的 RGB 做单人姿态复核。</p>
<ol><li>沿用原新目标确认、短缺口和原生 Pose 优先规则。没有确认目标、目标已过期或当前有原生可关联姿态时，不用新模型建立或替换身份。</li>
<li>当前 RTMO 候选需要通过既有空间连续性，至少 4 个图内身体关节，并满足置信度 ≥0.60 或两个当前视图框重叠 ≥0.50。之前已准入但不完整的 RTMO 观测也可进入复核。</li>
<li>裁剪来自当前 RTMO 检测框；对当前 RGB 作 4 个旋转 × 2 个尺度（1.0、1.15），共 8 次 RTMPose 推理。使用 SDK 指定的 RGB 归一化、1.25 框扩展和 SimCC 解码；变换后坐标还原回当前原图。</li>
<li>关节需在原图内、置信度 ≥0.30，有至少两个预测位置在框对角线 0.08 范围内一致。仅同方向的两个尺度不足以添加关节，需另一旋转支持，或当前 RTMO 对该关节也有近位置证据。竞争位置支持接近则保留未知。</li>
<li>最终需至少 6 个有支持的关节、至少 3 个身体关节与当前 RTMO 对应，并形成可连接的三关节肢体链。替换已有补检姿态还需增加至少 2 个有效关节。未支持关节置信度为 0，明确输出 partial；不沿用过去坐标。</li></ol>
<p>这些条件均不接收片段名、事件帧号、椅子坐标或参考动作。帧号只用于本页审计与导出短片。新增组件 <code>src/fallpre/weekly/pose_refinement.py</code>；公开样本 CLI 默认 <code>--recovery refined</code>。<code>--recovery rtmo</code> 保留上轮版本，<code>--recovery off</code> 保留更早的准入修复。生产 <code>fallpre run</code> 与历史 weekly 入口未切换。</p>
<h2 id="cases">重点可播放对照</h2><p>左侧：上轮 RTMO 补检；右侧：本轮。绿框选中目标，红框拒绝，橙框其他准入人体。RTMPose 复核帧标为 PARTIAL。未补出人体时不画历史骨架。</p>''']
    for r in records:
        parts.append(f'<section class="card"><h3>{esc(r["title"])}</h3><p>{esc(r["detail"])}</p><video controls preload="metadata" playsinline src="cases/{r["id"]}.mp4"></video></section>')
    parts.append('<h2 id="metrics">全部回归</h2><p>目标不可用包含空房与初始确认等待，不是漏检率。所有此前有目标的帧都保留目标；原生 Pose 准入集合逐帧保持。</p><table><tr><th>输入</th><th>观察数</th><th>目标不可用 前→后</th><th>RTMPose 采用帧</th><th>动作/阶段变化帧</th></tr>')
    for s in summary['clips']:
        parts.append(f'<tr><td>{s["clip"]}</td><td>{s["frames"]}</td><td>{s["missing_before"]} → {s["missing_after"]}</td><td>{s["refined_frames"]}</td><td>{s["changed_actions"]} / {s["changed_phases"]}</td></tr>')
    parts.append('''</table><p>4 段非跌倒视频共 494 个观察，仍无 falling；UTK standing_up 仍为 6、7 帧。URFD fall01 原始 falling 28→31，低清仍为 30；两个独立 URFD 跌倒仍有 falling 输出。这不说明躺地状态正确，静止人体可能被旧 V7 判为 background。</p>
<h3>原始 URFD 关键帧的关节覆盖</h3><p>下表统一统计图内且置信度 ≥0.30 的 COCO17 关节。新结果还通过上述一致性约束；两个模型置信度未校准，关节数量不是姿态准确率。0 表示没有准入目标。所有新增姿态仍是 partial。</p>
<table><tr><th>原帧号</th><th>前：图内过阈值关节</th><th>后：有一致性支持的关节</th><th>仍未知的 COCO17 索引</th></tr>''')
    for r in joint_review:
        parts.append(f'<tr><td>{r["frame"]}</td><td>{r["before"]}/17</td><td>{r["after"]}/17</td><td>{r["after_unknown"]}</td></tr>')
    parts.append('''</table><p>COCO17：0–4 头面部，5–6 肩，7–8 肘，9–10 腕，11–12 髋，13–14 膝，15–16 踝。</p>
<p><a href="audit-sheets/urfd-fall-01-original.jpg">全部原始 URFD 补检/漏检帧联系表</a> · <a href="audit-sheets/urfd-fall-01.jpg">低清联系表</a> · <a href="multicam-chute01-cam8/review/000098.jpg">MultiCam 新增门口观测</a>。本轮代理已逐图目视核查全部 13 个新采用姿态；没有独立人工关节真值。</p>
<h2 id="limits">仍未解决与代价</h2>
<ul><li><strong>URFD 原始第 134–135 帧仍无骨架：</strong>局部关节有证据，但不满足连通肢体链要求，未放宽成“有几个点就算人”。低清第 117 帧缺少足够当前候选，仍漏检。</li>
<li><strong>第 136 帧只恢复局部下半身；第 113–118 帧仍有缺失关节。</strong>遮挡与图外关节不能通过单 RGB 确认。模型也可能在可见人体区域内给出错误左右或关节位置，即使多视图一致也无法保证正确。本轮不能声称完整人体骨架已恢复。</li>
<li>试验第一版允许同一方向的两个尺度为新关节作证，URFD 第 113 帧在躯干附近产生可疑踝点。最终版增加跨方向或当前 RTMO 同关节支持后拒绝该点。首版和第二次小范围对照保留在 <code>runs/pose-refinement-probe-20261020</code> 与 <code>runs/pose-refinement-probe-v2-20261020</code>，未覆盖失败尝试。</li>
<li>UTK s10 坐姿内段仍为 23/25 个 forward_lean；s01 坐姿内段仍有 sitting_down/unknown。此次没有修改或训练动作模型。恢复目标不等于正确识别躺地或维持报警；第 136 帧的 background 是具体未解决例子。</li>
<li>MultiCam 本批没有椅子误准入或目标选择退化，新增第 98 帧为门口真实人物；长时间遮挡、两个老人近距离交叉、开机即躺地仍未验证。新分支不能建立新身份或恢复过期身份。</li>
<li>当前 CPU 上 URFD 补检分支单帧中位数约 1.35 秒；MultiCam 约 0.18 秒、最大约 1.44 秒，额外 RTMPose 最多 8 次。仅计补检分支，排除 YOLO 和编码；离线视频 30 FPS 不代表实时能力。</li></ul>
<h2>验证范围与来源</h2><ul>
<li>基线 <code>runs/pose-recovery-20261020</code> 未修改，修改前代码存于本轮 <code>baseline-code/</code>。8 路输入共 2483 个观察，输入及缓存散列均校验；复用相同原生 YOLO Pose/Detect，当前 RTMO 与 RTMPose 重新推理。没有拿旧补检骨架作为新输入。</li>
<li>另对 UTK s01 与 URFD 原始 fall01 共 233 帧从 RGB 重新运行全部视觉模型：原生检测、增强姿态、目标与完整预测逐项一致。101 项相关测试通过。<a href="fresh-integration-check.json">新入口一致性</a> · <a href="tests.xml">测试记录</a>。</li>
<li>本轮是开发样本回归，URFD 与 UTK 为历史 train/dev 输入，MultiCam 已用于开发；无独立盲测。低清与原始 URFD 为同一事件。关节一致性与目视复核不是 mAP/姿态真值评价。</li>
<li>UTK 稀疏观察按既有帧号/30 时钟展示，显示重复不增加推理帧；URFD 原始使用发布 CSV 时钟。MultiCam 真实 FPS 未确认，不运行动作预测，不算速度或预警提前量。源 AVI 宏块警告保留，输出能解码不证明源数据无损坏。</li>
<li>新增冻结模型来自 <a href="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip">OpenMMLab 官方 RTMPose-m ONNX SDK</a>，本轮成功下载并检查。ONNX 大小 54,330,655 字节；SHA256 <code>5c0a4bf67953e6d2ac43ce15e77dc9d5d354ae18430a47d2c5963a7bc5683e3c</code>。<a href="models/acquisition.json">获取记录</a> · <a href="models/pipeline.json">原始预处理配置</a> · <a href="https://github.com/open-mmlab/mmpose/blob/main/projects/rtmpose/README.md">官方项目</a>。采用原 SDK 的 RGB 转换，解码采用两轴最大值的较小值作保守分数。</li>
<li>仍使用本地隔离 ONNX Runtime 1.20.1，未安装新依赖；原 YOLO/RTMO/V7 权重和新增 RTMPose 权重全程散列保持。模型只做推理，未训练。实现当前依赖本工作区 <code>runs/pose-refinement-20261020/models/</code> 与既有 <code>tmp/group-meeting/rtmo-deps</code>，干净 checkout 需先准备这些依赖。</li></ul>
<h2 id="full">全部 8 段完整对照</h2>''')
    for s in summary['clips']:
        cid=s['clip'];parts.append(f'<section class="card"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/comparison.mp4"></video><p><a href="{cid}/frames.jsonl">逐帧当前候选、关节支持与动作</a> · <a href="{cid}/summary.json">统计</a></p></section>')
    parts.append(f'''<h2>结果与复现</h2><p>{len(validation)} 个 H.264 视频均完整解码。<a href="video-validation.json">逐视频验证</a> · <a href="final-code-hashes.json">代码散列</a> · <a href="final-code/">最终代码快照</a>。</p>
<pre># 当前版本，默认 verified + refined；必须新建输出目录
.venv/bin/python scripts/replay_public_samples.py --output runs/新实验
# 上轮 RTMO 版本
.venv/bin/python scripts/replay_public_samples.py --recovery rtmo --output runs/新基线
# 完整前后对照，复用原生 YOLO 观测，新运行 RTMO 和 RTMPose
.venv/bin/python scripts/compare_pose_refinement.py --output runs/新对照
# 本轮实际重新推理的两个样本；这里给出新目录模板
.venv/bin/python scripts/replay_public_samples.py --output runs/新回归 --clip utk-s01_e01-transition --clip urfd-fall-01-original</pre>
<p>本轮实际执行了完整对照与上述两段完整重新推理；模板目录名仅用于后续复现。不要覆盖已保留的基线与结果。</p></html>''')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(videos=len(validation),audit=audit,joints=joint_review),ensure_ascii=False))


if __name__=='__main__':main()
