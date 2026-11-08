"""Export review intervals, case clips and a local playable audit report."""
import csv
import html
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import cv2
from fallpre.weekly.human import open_video_writer

OUT = ROOT / 'runs/public-samples-20261020'


def main():
    (OUT/'dataset-survey-source.md').write_text((ROOT/'docs/research/公开视频数据集调研与首轮实验建议-2026-10-19.md').read_text())
    summary = json.loads((OUT / 'summary.json').read_text())
    spans = {
        'utk-s01_e01-transition': [(680,704,'seated'),(716,736,'rising'),(752,782,'standing')],
        'utk-s10_e01-transition': [(654,754,'seated'),(798,818,'rising'),(832,890,'standing')],
        'urfd-adl-04': [(65,137,'bending')], 'urfd-adl-07': [(133,180,'seated')],
    }
    reviewed = []
    for cid, intervals in spans.items():
        rows = [json.loads(l) for l in (OUT/cid/'frames.jsonl').read_text().splitlines()]
        for a,b,state in intervals:
            rr = [r for r in rows if a <= r['original_frame_id'] <= b]
            reviewed.append(dict(clip=cid, first_original_frame=a, last_original_frame=b,
                tentative_state=state, observed_frames=len(rr),
                actions=dict(Counter(r['prediction']['label'] for r in rr)),
                phases=dict(Counter(r['prediction']['phase'] for r in rr)),
                annotation='Post-hoc agent visual review of raw contact sheets; conservative interior intervals; no independent human adjudication.'))
    (OUT/'review-intervals.json').write_text(json.dumps(reviewed,ensure_ascii=False,indent=2))
    with (OUT/'review-intervals.csv').open('w') as f:
        keys = list(reviewed[0]); writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in reviewed: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,dict) else v for k,v in row.items()})
    cases = [
        ('utk-s10-seated-phase','utk-s10_e01-transition',3.9,9.9,
         '坐着时，阶段输出长期为“前倾”',
         '原帧 654–754 的 25 个已保存观察中，主动作均为 background；阶段仅 2 帧 seated、23 帧 forward_lean。不是“起身”误报，但若 JEV 把前倾直接当作离座风险证据会产生错误。代表帧 690（5.267 秒）；实际起身段 798–818 的 6 个观察均检出 standing_up。'),
        ('utk-s01-unknown-rise','utk-s01_e01-transition',4.4,7.6,
         '起身动作检出，但四阶段输出不完整',
         '原帧 680–704 的 9 个坐姿观察全部 phase=unknown，其中 3 帧主动作仍 sitting_down。上升内段 716–736 的 6 个观察，2 帧 background、4 帧 standing_up，但阶段全部 unknown；主动作开始报起身为帧 722（5.933 秒）。这是阶段缺失和起始段漏检，不能把 unknown 当成 seated。'),
        ('urfd-original-loss','urfd-fall-01-original',2.5,5.333,
         '原始 RGB：跌倒后漏检测／失关联',
         '帧 87–112 报 falling；帧 113–119、134–137 共 11 帧目标不可用，其中 9 帧无检测，另 2 帧有检测但未关联。原始 RGB 重获后仍为 ID 1；躺地阶段多为 background。动作分类中的 falling 不是持久报警状态，本轮未验证报警保持逻辑。'),
        ('urfd-lowres-id','urfd-fall-01',2.5,5.333,
         '同一跌倒的低清版本：目标 ID 改变',
         '320×240 旧 RGB 裁剪版的帧 113–127 共 15 帧无目标关联，但各帧仍存在候选检测；帧 128 后目标由 ID 1 换成 ID 3。与原始 640×480 属同一事件，只能用于输入质量对照，不能算第二次独立跌倒成功。'),
        ('multicam-wrong-target','multicam-chute01-cam8',1079/30,1300/30,
         'MultiCam：躺倒后目标转移到左缘椅子误检',
         '只看帧号：帧 1080–1106 仍关联演员 ID 15；帧 1132 起选中 ID 27，代表帧 1135、1252 的绿色目标框落在左缘椅子／衣物，床垫上的真实人物是未关联的橙色框。帧 1285 只检出左缘误检，漏检躺地演员。30 FPS 仅供观看，不是实际速度；未运行任何动作或预警模型。'),
    ]
    case_dir=OUT/'cases';case_dir.mkdir(exist_ok=True)
    records=[]
    for name,cid,start,end,title,detail in cases:
        cap=cv2.VideoCapture(str(OUT/cid/'overlay.mp4'))
        writer,codec=open_video_writer(case_dir/(name+'.mp4'),30,(1280,720))
        i=0;n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            if round(start*30)<=i<round(end*30):
                writer.write(frame)
                if n==0:cv2.imwrite(str(case_dir/(name+'.jpg')),frame)
                n+=1
            i+=1
        cap.release();writer.release()
        records.append(dict(id=name,clip=cid,start_preview_s=start,end_preview_s=end,
            title=title,detail=detail,frames=n,codec=codec))
    (OUT/'failure-cases.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
    validation=[]
    for path in sorted(OUT.glob('*/overlay.mp4'))+sorted(case_dir.glob('*.mp4')):
        cap=cv2.VideoCapture(str(path));n=0;fps=cap.get(cv2.CAP_PROP_FPS)
        declared=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));codec=int(cap.get(cv2.CAP_PROP_FOURCC))
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(720,1280,3)
            n+=1
        cap.release();assert n==declared and n>0
        validation.append(dict(path=str(path.relative_to(OUT)),frames=n,fps=fps,
            duration_s=n/fps,codec=''.join(chr((codec>>8*i)&255) for i in range(4)),bytes=path.stat().st_size))
    (OUT/'video-validation.json').write_text(json.dumps(validation,indent=2))
    def esc(s):return html.escape(str(s))
    parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>公开 RGB 首轮实验 · 2026-10-20</title><style>
body{font:16px/1.7 system-ui,sans-serif;background:#f3f5f7;color:#17212b;max-width:1180px;margin:auto;padding:28px}
h1{font-size:30px}h2{margin-top:38px}h3{margin-bottom:5px}.card{background:white;padding:24px;margin:22px 0;border-radius:12px;border:1px solid #dde3e8}
video{width:100%;max-height:660px;background:#181818}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #ccd5dd;padding:8px;text-align:left}th{background:#e8edf2}.note{border-left:4px solid #c68b1c;padding:10px 18px;background:#fff9e9}a{color:#145f9c}.small{font-size:14px;color:#4c5966}code{background:#eaf0f4;padding:2px 5px}.controls button{margin:5px;padding:7px 12px}</style>
<h1>公开 RGB 首轮实验：可播放结果与失败案例</h1>
<p>执行日期：2026-10-20。冻结 YOLO11n-Pose + ByteTrack + 当前 V7；逐帧重新推理，无训练、无调参、无真实通知。8 路输入、2483 个原观察；URFD fall-01 的两个版本来自同一事件。下方 13 个 H.264 视频全部完整解码通过。</p>
<p><a href="#cases">先看 5 个问题片段</a> · <a href="#all">全部 8 段</a> · <a href="manifest.json">输入与模型 SHA-256</a> · <a href="summary.json">统计</a> · <a href="review-intervals.csv">暂定复核标签 CSV</a> · <a href="video-validation.json">视频验证</a></p>
<div class="note"><strong>结论：</strong>本组弯腰、坐下与坐姿中未观察到跌倒误报，已复核坐姿内段未出现 standing_up。但坐姿阶段识别不稳定，URFD 躺地后检测／跟踪中断，MultiCam 出现椅子误检及选错目标。检测框或高置信关键点不保证是真人，更不保证身份正确。</div>
<h2>结果范围与读法</h2>
<ul><li>动作标签 background 包含多个静态状态；V7 phase 是坐到站的专用阶段头，不能当作通用坐／站／躺分类器。主动作与 phase 可能不一致，视频同时显示。</li>
<li>UTK 使用原 JPEG 帧号差÷30 的既有源时钟；保留稀疏观察并在播放中重复上一帧。原始 URFD 使用 CSV 毫秒时间。图中 Source clock 比播放器秒数更适合查证。</li>
<li>MultiCam 原文件头写 120 FPS，物理帧率未确认。本轮关闭 V3/V7；按解码顺序关联，跟踪器内部以每帧 1/30 的合成索引运行（约 15 帧保留窗），不是物理秒级推断。输出以 30 FPS 展示，每帧均一对一保留，不计算速度、延迟或提前量。</li>
<li>绿色是自动选定目标，橙色是其他检测，ID ? 表示没有活动轨迹关联。图中 ankles reliable 仅表示关键点分数≥0.2，并非位置正确的人工验证。</li>
<li>参考动作从既有标签读取，未进入模型。UTK 官方动作区间较宽，sitDown/standUp 包含静止准备时间；另用原画面保守内段检查坐姿与上升。新复核是代理事后目视暂定标签，无独立人工仲裁。</li></ul>
<h2>小样本检查</h2><table><tr><th>项目</th><th>本次观察</th><th>限制</th></tr>
<tr><td>坐姿→误报起身</td><td>UTK s01 / s10 + URFD ADL07：暂定坐姿内段 82 个观察，standing_up 为 0；其中 3 帧仍报 sitting_down</td><td>相邻帧高度相关；不是 82 次独立试验，坐姿阶段也未全部识别</td></tr>
<tr><td>弯腰／坐下→误报跌倒</td><td>四段非跌倒录像共 494 个观察，falling 为 0；ADL04 弯腰内段 73/73 报 bending</td><td>非跌倒片段合计仅约 31 秒；不能估算每小时部署误报率</td></tr>
<tr><td>实际跌倒</td><td>URFD fall01 原始 RGB 帧 87–112、fall02 帧 49–75 报 falling；2 个独立跌倒均有命中</td><td>旧训练／开发回归事件，不是独立测试或临床检出率</td></tr>
<tr><td>起跌与提前量</td><td>fall01 官方姿态标注首个 falling 为帧 83（2.736 秒），首个模型 falling 为帧 87（2.869 秒），相差 +0.133 秒</td><td>是相对姿态标签的事后检出延迟，不是预警提前量；官方标签与 RGB 同步／临床起跌边界仍有误差</td></tr>
</table><h2 id="cases">5 个具体问题片段</h2>''']
    for r in records:
        parts.append(f'<section class="card"><h3>{esc(r["title"])}</h3><p>{esc(r["detail"])}</p><video controls preload="metadata" playsinline src="cases/{r["id"]}.mp4"></video><p class="small">此片段从全片提取；画面保留原始帧号与源时钟。</p></section>')
    parts.append('<h2 id="all">全部 8 段：完整叠加结果</h2><table><tr><th>输入</th><th>观察帧</th><th>无任何检测</th><th>选定目标不可用</th><th>目标 ID（不等于真实人物数）</th></tr>')
    for s in summary['clips']:
        parts.append(f'<tr><td><a href="#{s["clip"]}">{s["clip"]}</a></td><td>{s["frames"]}</td><td>{s["no_detection"]}</td><td>{s["no_target"]}</td><td>{esc(s["observed_target_ids"])}</td></tr>')
    parts.append('</table><p class="small">MultiCam 中“目标可用”可能实际关联到了椅子，因此不能将 1−无目标比例当成人体召回率。未提供人体框真值，不报告 mAP、PCK 或 IDF1。</p>')
    for s in summary['clips']:
        cid=s['clip']
        parts.append(f'<section class="card" id="{cid}"><h3>{cid}</h3><video controls preload="metadata" playsinline src="{cid}/overlay.mp4"></video><p>动作计数：{esc(s["counts"])}</p><p><a href="{cid}/frames.jsonl">逐帧检测／COCO17／轨迹／动作</a> · <a href="{cid}/summary.json">本段摘要</a></p></section>')
    parts.append('''<h2>证据、限制与下一步</h2><ul>
<li>YOLO 输入大小 640、检测阈值 0.1、IoU 0.45、CPU；骨架绘制阈值 0.2；沿用现有 ByteTrack 与 V7 默认配置。冻结模型散列运行前后相同。</li>
<li>UTK s01 属 train，s10 在当前 V7 已属 dev；URFD 对照来自历史 train。原始 fall01 虽像素质量不同，仍是同一训练事件。本次是故障定位与复现，不能宣称新人物泛化。</li>
<li>MultiCam 全部 1559 帧均返回，但源 MPEG4 解码记录一次 Error at MB: 1285。1285 是宏块编号，不代表坏帧就是第 1285 帧；损坏位置尚未定位。编码后的完整可解码不等于源画面无损坏。</li>
<li>URFD fall01 新原始输入 640×480；旧 fall01/fall02/ADL04/ADL07 是 320×240 的 RGB 拼图右半裁剪。质量对照揭示风险，但仅一对录像，不能概括分辨率与失跟踪的因果关系。</li>
<li>MultiCam 官方 code6 Moving up 在本样本是处理衣物，未映射为椅子起身。场景1参考视角按勘误 cam8 处理；帧标签按 1-based 暂用，正式指标前还需校正边界。</li>
<li>本轮没有运行报警闭环、未来预测或二维米制环境模块。缺标定时不输出 2 米距离；暂无多人交叉身份真值，不能据此验证养老院大厅的多人跟踪能力。</li>
<li>建议先复核 MultiCam 左缘假人、床垫演员漏检及目标重新选择策略，再补坐姿／前倾／离座／站稳的独立人工标签；之后另开受控修改与留出评估。本轮不据这些案例修改模型或阈值。</li></ul>
<p>复现：在仓库根目录运行 <code>.venv/bin/python scripts/replay_public_samples.py --output runs/另一个新目录</code>。脚本必须使用新目录；结果整理脚本 <code>scripts/summarize_public_samples.py</code> 固定指向本轮目录。模型与输入完整哈希见 manifest.json。</p>
<p>数据背景与原始来源：<a href="dataset-survey-source.md">前轮调研报告</a>。结果仅作内部研究复核，原数据许可继续适用。</p></html>''')
    (OUT/'index.html').write_text('\n'.join(parts))
    print(json.dumps(dict(videos_validated=len(validation),review_intervals=len(reviewed),failure_cases=len(records)),ensure_ascii=False))


if __name__=='__main__':main()
