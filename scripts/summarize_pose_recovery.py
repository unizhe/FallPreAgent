"""Build an auditable HTML report and playable excerpts; annotations are evaluation only."""
import html
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
from fallpre.weekly.human import open_video_writer, sha256

OUT = ROOT / 'runs/pose-recovery-20261020'
FRESH = ROOT / 'runs/pose-recovery-fresh-check-20261020'


def load_rows(cid):
    return [json.loads(line) for line in (OUT / cid / 'frames.jsonl').read_text().splitlines()]


def main():
    summary = json.loads((OUT / 'summary.json').read_text())
    all_rows = {s['clip']: load_rows(s['clip']) for s in summary['clips']}
    checks = {}
    for cid, rows in all_rows.items():
        checks[cid] = dict(
            native_admission_changed=sum(
                {a['index'] for a in r['before']['admission'] if a['accepted']} !=
                {a['index'] for a in r['after']['admission'] if a['accepted'] and a['index'] < len(r['native_pose_instances'])}
                for r in rows),
            previously_observed_now_missing=[r['original_frame_id'] for r in rows if r['before']['observed'] and not r['after']['observed']],
            action_phase_identical=all((r['before']['prediction']['label'], r['before']['prediction']['phase']) ==
                                      (r['after']['prediction']['label'], r['after']['prediction']['phase']) for r in rows),
            missing_after=[r['original_frame_id'] for r in rows if not r['after']['observed']])
        assert checks[cid]['native_admission_changed'] == 0
        assert not checks[cid]['previously_observed_now_missing']
    mc = all_rows['multicam-chute01-cam8']
    lying = [r for r in mc if 1109 <= r['original_frame_id'] <= 1285]
    empty = [r for r in mc if 112 <= r['original_frame_id'] <= 843]
    checks['multicam_lying'] = dict(frames=len(lying), before_missing=sum(not r['before']['observed'] for r in lying),
                                  after_missing=sum(not r['after']['observed'] for r in lying))
    checks['multicam_empty'] = dict(frames=len(empty), before_observed=sum(r['before']['observed'] for r in empty),
                                  after_observed=sum(r['after']['observed'] for r in empty))
    nonfall = [r for cid in ['utk-s01_e01-transition', 'utk-s10_e01-transition', 'urfd-adl-04', 'urfd-adl-07'] for r in all_rows[cid]]
    checks['nonfall'] = dict(frames=len(nonfall), falling_before=sum(r['before']['prediction']['label']=='falling' for r in nonfall),
                            falling_after=sum(r['after']['prediction']['label']=='falling' for r in nonfall))
    seated = [r for cid, lo, hi in [('utk-s01_e01-transition',680,704),('utk-s10_e01-transition',654,754),('urfd-adl-07',133,180)]
              for r in all_rows[cid] if lo <= r['original_frame_id'] <= hi]
    checks['seated_review_intervals'] = dict(frames=len(seated), standing_up_after=sum(r['after']['prediction']['label']=='standing_up' for r in seated),
                                            annotation='previous proxy visual review, not independently adjudicated')
    (OUT / 'regression-checks.json').write_text(json.dumps(checks, indent=2))

    cases = [
        ('multicam-lying','multicam-chute01-cam8',1049/30,1300/30,
         'MultiCam：躺地区间目标缺失 50 → 0 帧',
         '原帧 1109–1285 共 177 帧；第 1285 帧右侧通过当前图像 RTMO 补回床垫上的演员，左缘椅子继续为红框。绿色表示选中目标，红色是假候选被拒绝，橙色是其他准入人体。此处仅评估目标观测可用性，未计算骨架准确率。30 FPS 为展示速度。'),
        ('multicam-empty','multicam-chute01-cam8',449/30,550/30,
         '椅子回归：空房中不创建补检目标',
         '第 501 帧仍能看到 YOLO 的椅子假框，但前后都不将其准入。整个空房复核区间 112–843 共 732 帧，前后目标观测均为 0。补检不能创建身份；它只处理已经确认、尚未过期的目标。'),
        ('multicam-return','multicam-chute01-cam8',1350/30,1510/30,
         'MultiCam：旁人进入时继续保留床垫演员',
         '全片共补回 107 帧，其中 57 帧在暂定躺地标签区间外，包括倒下前后的姿势和演员从床垫起身。逐张检查全部补检帧的联系表，未见补检改选椅子；后部旁人进入时仍跟随床垫演员。这不是近距离多人交叉的身份验证。'),
        ('urfd-original','urfd-fall-01-original',3.4,4.8,
         'URFD 原始 RGB：部分改善，遮挡与边缘漏检仍在',
         '第 113、114 帧补回当前人体观测；115–118、134–136 仍缺失，共 7 帧。全片另有首帧确认等待，目标不可用总数 10 → 8。第 113 帧的身体被画面下沿截断，补回的骨架只有部分关节可信，不应视为完整姿态恢复。'),
        ('urfd-lowres','urfd-fall-01',3.4,4.8,
         'URFD 低清版：补回 3 帧，仍有 3 帧躺地漏检',
         '第 113、114、115 帧补回；第 117–119 帧仍未知。全片目标不可用 7 → 4（含首帧确认等待）。它与原始 RGB 为同一次跌倒，不能作为两个独立成功事件。旋转视图的一致性不能保证关节位置正确。'),
        ('utk-seated','utk-s10_e01-transition',3.9,9.9,
         'UTK 回归：输出未退化，坐姿阶段错误仍未解决',
         '两段 UTK 共 164 个原始观察，补检触发次数为 0，动作与阶段输出逐帧一致。s10 的坐姿内段 654–754 中，25 个观察仍有 23 个被判为 forward_lean，主动作均为 background；真实起身输出保持。此处是现有动作模型失败，不是本轮人体漏检修复的成果。'),
    ]
    folder = OUT / 'cases'; folder.mkdir(exist_ok=True)
    records = []
    for name, cid, start, end, title, detail in cases:
        cap = cv2.VideoCapture(str(OUT / cid / 'comparison.mp4'))
        writer, codec = open_video_writer(folder / (name+'.mp4'), 30, (1280,720))
        i = n = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            if round(start*30) <= i < round(end*30):
                writer.write(frame); n += 1
            i += 1
        cap.release(); writer.release()
        records.append(dict(id=name, clip=cid, start_preview_s=start, end_preview_s=end,
                            title=title, detail=detail, frames=n, codec=codec))
    (OUT/'cases.json').write_text(json.dumps(records, ensure_ascii=False, indent=2))
    validation = []
    for path in sorted([*OUT.glob('*/comparison.mp4'), *folder.glob('*.mp4'), *FRESH.glob('*/overlay.mp4')]):
        cap = cv2.VideoCapture(str(path)); declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); n = 0
        while True:
            ok, frame = cap.read()
            if not ok: break
            assert frame.shape == (720,1280,3); n += 1
        cap.release(); assert n == declared and n > 0
        validation.append(dict(file=str(path.relative_to(ROOT)), frames=n, complete_decode=True, bytes=path.stat().st_size))
    (OUT/'video-validation.json').write_text(json.dumps(validation, indent=2))
    files = ['src/fallpre/weekly/pose_recovery.py','src/fallpre/weekly/verified_tracking.py','scripts/replay_public_samples.py',
             'scripts/compare_pose_recovery.py','scripts/summarize_pose_recovery.py','tests/test_pose_recovery.py']
    snapshot = OUT/'final-code'; snapshot.mkdir(exist_ok=True)
    for file in files: shutil.copyfile(ROOT/file, snapshot/Path(file).name)
    (OUT/'final-code-hashes.json').write_text(json.dumps({file:sha256(ROOT/file) for file in files},indent=2))
    esc = html.escape
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>躺地人体补检 · MultiCam / URFD / UTK · 2026-10-20</title>
<style>body{max-width:1220px;margin:auto;padding:26px;background:#f3f5f7;color:#17212b;font:16px/1.75 system-ui,sans-serif}h1{font-size:30px}h2{margin-top:36px}a{color:#145d91}.card{background:white;border:1px solid #dce4e9;border-radius:12px;padding:22px;margin:22px 0}video{width:100%;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border:1px solid #cad4db;text-align:left}th{background:#e5edf2}.note{padding:16px;background:#fff8e7;border-left:4px solid #ba8218}code,pre{background:#e7edf2}pre{padding:16px;white-space:pre-wrap}.small{font-size:14px;color:#485762}</style>
<h1>躺地人体补检：MultiCam 改善明显，URFD 仍有缺口</h1>
<p>核查与运行日期：2026-10-20 · 未训练，所有模型权重保持 · 前后采用同一输入与时钟 · MultiCam 仅帧级分析。</p>
<div class="note"><strong>本轮结果：</strong>MultiCam 暂定躺地区间 177 帧，目标缺失 <strong>50 → 0</strong>；URFD 原始 fall01 躺地区间缺失 <strong>9 → 7</strong>，低清版 <strong>6 → 3</strong>。URFD 两种输入属于同一事件。没有重新准入之前拒绝的原生 Pose 候选；UTK 两段动作和阶段输出逐帧一致。<br><strong>仍未解决：</strong>URFD 边缘截断、遮挡造成的漏检和不完整骨架；UTK 坐姿被判前倾；补检 CPU 延迟较高。目标有框不等于骨架准确，也不等于跌倒报警已经可靠。</div>
<p><a href="#cases">6 段重点视频</a> · <a href="#metrics">全部回归</a> · <a href="#limits">残余失败与代价</a> · <a href="#full">8 段完整对照</a> · <a href="protocol.json">权重/配置/原日志散列</a> · <a href="regression-checks.json">回归明细</a></p>
<h2>本次修复的依据与约束</h2>
<p>之前的缺口是 YOLO Pose 没有输出演员骨架，或只输出椅子假候选，跟踪器无法凭空恢复。本轮保留上一轮新目标准入规则，新增本地已有的冻结 RTMO-s 作为当前画面补检。没有按房间、椅子位置、数据集名字、动作标签或帧号写推理规则。</p>
<ol><li>只服务于已确认且未过期的目标：URFD/UTK 最后观测距当前不超过 0.5 秒，MultiCam 不超过 15 个观察帧。当前原生 Pose 已有可关联观测时，不替换、不追加。</li>
<li>先在当前全图运行 RTMO。原向候选置信度 ≥0.50、至少 8 个在图内且分数 ≥0.30 的关节，并有肩部与髋部证据，且通过既有位置连续性检查，才可返回。</li>
<li>无可靠原向候选时，对当前全图及由最近人体框自动产生的 2 倍尺度正方形裁剪，分别做四向旋转推理。候选框 IoU ≥0.50、躯干关节归一化差异 ≤0.20 才可提供跨视图一致性；竞争候选难区分则放弃。最多 8 次 RTMO 推理。</li>
<li>补检仅追加当前推理得到的 COCO17 坐标，沿用同一准入与关联逻辑作最终选择。不复制历史骨架，不插值补人，不使用未来帧，也不允许补检初始化新身份。旋转还原保留左右关节语义。</li></ol>
<p>组件接入公开样本 CLI，默认 verified + rtmo。<code>--recovery off</code> 复现上轮准入修复，<code>--tracking legacy</code> 默认关闭补检。生产 <code>fallpre run</code> 与历史 weekly 入口未切换；当前尚是可复用组件与样本实验入口。</p>
<h2>对照与验证边界</h2>
<ul><li>基线是 <code>runs/verified-tracking-20261020-v2</code>（上一轮椅子/关联修复），不是最初椅子误检版。基线输出未覆盖，修改前代码保存在本目录 <code>baseline-code/</code>；本轮代码在 <code>final-code/</code>。</li>
<li>8 路输入、2483 个原始观察。复用逐项散列校验的原 YOLO Pose/Detect 缓存，RTMO 在解码的当前 RGB 上重新运行。另对 UTK s01 和 URFD 原始 fall01 共 233 帧重新运行所有视觉模型：原生检测、补检坐标、选中目标及完整动作预测与对照逐项一致。</li>
<li>前后左右均为逻辑目标 epoch；框上 BT 是 ByteTrack 原始 ID。绿框是选中人体，橙框其他准入人体，红框被拒绝。显示 RTMO 的帧为实际补检；未知帧没有画历史人。</li>
<li>UTK 为稀疏保存 RGB 观察，播放器按原帧号/30 的既有时间轴重复上个观察用于展示；重复画面不算额外推理。URFD 原始使用发布 CSV 时钟。MultiCam 原物理 FPS 未确认，只按帧关联，播放器 30 FPS 纯展示，不算真实速度、预警提前量，也不运行动作模型。</li>
<li>计数为目标观测可用性，不是经过人框/关节真值评估的召回率、mAP 或 IDF1。人工式图像检查由本轮代理执行：全部 112 个补回帧已用联系表复核，未见补回椅子；无独立人工仲裁。MultiCam 参考区间来自之前映射的发布标签，1-based 边界仍沿用暂定解释。</li></ul>
<h2 id="cases">6 段重点对照</h2>''']
    for r in records:
        parts.append(f'<section class="card"><h3>{esc(r["title"])}</h3><p>{esc(r["detail"])}</p><video controls preload="metadata" playsinline src="cases/{r["id"]}.mp4"></video></section>')
    parts.append('''<h2 id="metrics">全部回归统计</h2><p>“目标不可用”含空房、首帧确认等待；不可当作全片漏检率。所有此前有目标的帧都仍有目标。下表只报主动作/阶段标签变化，不把它解读为准确率改善。</p><table><tr><th>输入</th><th>观察数</th><th>目标不可用 前→后</th><th>补检成功帧</th><th>动作/阶段标签变化帧</th></tr>''')
    for s in summary['clips']:
        parts.append(f'<tr><td>{s["clip"]}</td><td>{s["frames"]}</td><td>{s["missing_before"]} → {s["missing_after"]}</td><td>{s["recovery_selected_frames"]}</td><td>{s["changed_actions"]} / {s["changed_phases"]}</td></tr>')
    parts.append('''</table><ul><li>MultiCam 全片补回 107 帧；标注躺地区间补回 50 帧，其余 57 帧包括标签区间外仍在床垫上的姿态与起身。所有原生 Pose 的准入集合与基线一致，没有把之前被拒绝的椅子候选重新准入。该结论只覆盖这些样本。</li>
<li>UTK 两段共 164 帧未触发补检，主动作和阶段逐帧一致，standing_up 仍分别是 6、7 帧。4 段非跌倒样本共 494 个观察，前后均无 falling。复核坐姿内段共 82 个观察仍无 standing_up；静态坐姿被判 sitting_down/forward_lean 的旧错误保留。</li>
<li>URFD 两次独立跌倒均仍有 falling 输出。fall01 原始版 falling 26 → 28 帧、低清版 26 → 30 帧；低清额外有 1 帧历史窗造成的标签改变，不能把每个动作变化都归为该帧补检。没有新增事件级或预警性能结论。</li></ul>
<h2 id="limits">仍未解决的失败与修改代价</h2>
<ol><li><strong>URFD 原始 fall01 第 115–118、134–136 帧仍无目标观测。</strong>第 135 帧人体躺地、贴近下边缘，旋转候选未同时通过关节质量与一致性；保留未知。低清版第 117–119 帧同样失败。原始第 113–114 帧虽有框，但遮挡和边缘截断使下肢骨架不可靠，不能用于准确的足点或米制距离。</li>
<li><strong>UTK s10 坐姿阶段仍误判。</strong>654–754 内的 25 个观察中，23 个 forward_lean、2 个 seated，主动作都是 background。s01 坐姿 680–704 的 9 个观察阶段均 unknown，3 个主动作仍为 sitting_down。没有改动 V7 模型。</li>
<li><strong>原始椅子假检测仍在。</strong>它们在本次准入/选择层未成为目标，但两个模型共同认错、椅子与人体高度重叠、多人贴近交叉仍可能失败。没有真实开机即躺地的确认能力提升：补检不能建立新目标。丢失超过窗口必须重新准入，不能保证原老人身份。</li>
<li><strong>CPU 成本明显增加。</strong>MultiCam 触发补检的 128 帧，单帧耗时中位数约 0.143 秒，最大约 1.21 秒；URFD 原始 9 帧、低清 6 帧分别中位数约 1.11、1.13 秒。它们只计补检分支，不包括 YOLO、跟踪、动作或编码；首次触发含会话初始化。离线对照可播放不代表实时 30 FPS。</li>
<li><strong>泛化证据仍有限。</strong>本轮是历史 train/dev 样本上的开发回归，MultiCam 也是当前开发样本；URFD 两个分辨率属同一事件。未训练也不能使其成为盲测。未验证养老院老人、轮椅、长遮挡或密集多人；未修改报警保持、JEV 或米制环境模块。</li>
<li>源 MultiCam AVI 仍输出一个宏块解码警告；“Error at MB: 1285”是宏块索引，不是第 1285 帧的诊断。输出视频完整解码不证明输入源无损坏。</li></ol>
<p>补检帧视觉审计：<a href="audit-sheets/">9 张联系表</a>（7 张 MultiCam，2 张 URFD）。第 1285 帧见 <a href="multicam-chute01-cam8/review/001285.jpg">MultiCam 直接证据</a>；第 135 帧见 <a href="urfd-fall-01-original/review/000135.jpg">URFD 残余漏检</a>。</p>
<h2 id="full">8 段完整前后对照</h2>''')
    for s in summary['clips']:
        cid = s['clip']
        parts.append(f'<section class="card"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/comparison.mp4"></video><p><a href="{cid}/frames.jsonl">逐帧检测、补检候选与选择证据</a> · <a href="{cid}/summary.json">统计</a></p></section>')
    parts.append(f'''<h2>复现、冻结权重和验证记录</h2>
<p><strong>93 项相关测试通过；{len(validation)} 个 H.264 视频完整解码。</strong> <a href="tests.xml">测试报告</a> · <a href="video-validation.json">逐视频解码</a> · <a href="fresh-integration-check.json">233 帧全链路重新推理一致性</a> · <a href="final-code-hashes.json">最终代码散列</a>。</p>
<p>RTMO 权重是本地既有 <code>runs/meeting-20261014/models/rtmo-s.onnx</code>，39,617,685 字节，SHA256 <code>d0703d40d19f3921da51ae725402d5fdae4d2478c7442072d3101bd396f370d8</code>。官方归档来源：<a href="https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.zip">OpenMMLab RTMO-s ONNX SDK</a>（本轮复用本地已有文件，未重新下载）。获取记录在 <code>runs/meeting-20261014/models/rtmo-acquisition.json</code>。预处理与本地 RTMLib RTMO 实现核对：BGR、640 像素左上对齐填充、COCO17、无归一化。运行时复用 <code>tmp/group-meeting/rtmo-deps</code> 的 ONNX Runtime 1.20.1；主环境 OpenCV 4.14.0 / NumPy 2.4.6。这是当前工作区已有的隔离依赖路径，干净 checkout 需先按既有归档配置依赖。</p>
<pre># 新版全部样本，从 RGB 重新运行；新建输出目录
.venv/bin/python scripts/replay_public_samples.py --output runs/新补检目录
# 上轮准入/关联基线
.venv/bin/python scripts/replay_public_samples.py --recovery off --output runs/新基线目录
# 缓存 YOLO + 当前图像新 RTMO 的前后对照
.venv/bin/python scripts/compare_pose_recovery.py --output runs/新对照目录
# 已实际运行的全链路回归选择
.venv/bin/python scripts/replay_public_samples.py --output runs/新回归目录 --clip utk-s01_e01-transition --clip urfd-fall-01-original</pre>
<p>上述命令使用新的目录名作为复现模板；本轮实际运行了缓存对照及两段全链路回归，记录见协议与一致性检查。不要覆盖本次结果目录。参考标签、展示短片帧范围与审计区间只用于离线评估和网页导出，不进入 <code>PoseRecovery</code> 的推理输入。</p></html>''')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(videos=len(validation),cases=len(records),checks=checks['multicam_lying'])))


if __name__ == '__main__':
    main()
