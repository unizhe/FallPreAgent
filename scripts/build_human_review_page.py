"""Build a read-only evidence viewer; human decisions are made in conversation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/abc-v7-20261020/human-review"
ANNOTATIONS = OUT.parent / "annotations/annotations.json"

GUIDANCE = {
    "contact": {
        "short": "① 首次接地",
        "title": "URFD fall01 · 首次非足部接地",
        "definition": "寻找手、前臂、膝或躯干等非足部首次触及地面的时刻。脚原本就在地面，不计入这项标记。",
        "look_for": [
            "先看正常速度，再看慢放；用逐帧图对比手掌与地面的相对位置。",
            "手伸向地面、手影接近地面或身体已经很低，都不足以单独确认接触。",
            "若能看见接触后手部支撑、停止下移或身体绕接触点转动，可作为辅助证据。",
            "若运动模糊或画面下缘遮住首次接触，保留候选区间与不确定性，不必强行选一帧。",
        ],
        "candidate_note": "既有待审候选为源帧 101–107（3.336–3.537 秒）；这只是待核查范围，允许向前或向后修订。",
        "reply": "接地：确认区间 101–107，无法精确到单帧；或 接地：确认第 ___ 帧，依据是 ___；或 接地：目前无法判断。",
    },
    "stable": {
        "short": "② 稳定倒地",
        "title": "URFD fall01 · 稳定倒地起点",
        "definition": "建议判据：躯干和骨盆停止明显的整体下落、侧翻或平移，并持续至少 0.5 秒；允许手脚小幅调整。这个判据也需要你确认。",
        "look_for": [
            "观察躯干与骨盆的整体运动，而不只看某一只手或脚是否停止。",
            "必须检查候选帧之后至少 0.5 秒，排除短暂停顿后继续翻滚。",
            "标记的是稳定开始时刻；满足 0.5 秒持续条件的确认时刻应另外记录。",
            "例如选择 138 帧（4.571 秒）为起点，需检查到 153 帧（5.072 秒），才能确认约 0.5 秒的持续条件。",
            "源帧 160 是片尾。候选上界 138 后仍有约 0.734 秒可检查。",
            "138–153 帧的肩背和上臂仍有调整，请判断属于允许的小动作还是明显躯干运动；146 帧及以后起始则缺少足够片尾验证半秒。",
            "人物部分位于画外；不能确认整体稳定时，保留区间或未知。",
        ],
        "candidate_note": "既有待审候选为源帧 132–138（4.371–4.571 秒）；官方 lying 从 113 帧开始，与稳定倒地不是同一个标记。",
        "reply": "稳定倒地：采用／修改 0.5 秒判据；确认第 ___ 帧或区间 ___–___；理由 ___。",
    },
    "utk-s01-seat-off": {
        "short": "③ UTK s01 离座",
        "initial_frame": 716,
        "title": "UTK s01 · 臀部离开椅面",
        "definition": "寻找臀部首次不再由椅面承托的时刻。能看见臀部与椅面分离、随后持续上升，是主要视觉依据；仍坐着的前倾不算离座。",
        "look_for": [
            "对比最后一张确定仍坐着的观测与第一张确定已离座的观测。",
            "只有头部抬高或上身前倾，不能判定已经离座。",
            "重点比较 716、720、722：前两张接触处被遮挡；722 已出现明显上升，但这不等于看到了精确离座瞬间。",
            "椅面与臀部接触处被遮住时，允许只确认一个区间。",
            "若只能确认身体开始上升，应记录为可见上升代理，不能当作精确物理离座。",
            "现有源帧不连续，局部最大间隔 8 帧；逐帧按钮只切换真实存在的观测。",
        ],
        "candidate_note": "检查窗口 704–740，重点核查 716–722（既有相对时间约 5.733–5.933 秒）。可确认离座区间，也可只确认 722 为首次明确可见上升的代理帧；两者分开记录。",
        "reply": "s01：确认离座区间 ___–___，精确帧未知；或 只确认 722 为可见上升代理，物理离座时刻待定；或 暂无法判断。",
    },
    "utk-s10-seat-off": {
        "short": "④ UTK s10 离座",
        "initial_frame": 786,
        "title": "UTK s10 · 臀部离开椅面",
        "definition": "沿用 s01 的离座判据：臀部首次脱离椅面支撑，并由后续持续上升确认；前倾与离座分别记录。",
        "look_for": [
            "先找最后确定坐着与首次确定离座的两张观测。",
            "官方 standUp 是宽动作区间，其起点不等于可见离座时刻。",
            "现有输入通常每 4 个源帧一个观测，不能确认缺失帧内的精确瞬间。",
            "若接触处被衣物或身体遮挡，应保留不确定区间。",
            "重点对比 786、790、794：790 有向上变化，794 上升更清楚；请判断哪张符合首次明确可见上升的标准。",
        ],
        "candidate_note": "检查窗口 754–810，重点比较 786–794。接触处仍有遮挡，可只确认可见上升代理；该检查范围不自动成为物理离座边界。",
        "reply": "s10：只确认 ___ 帧为首次明确可见上升代理，物理离座待定；或 确认离座区间 ___–___，依据 ___；或 暂无法判断。",
    },
}

PAGE = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>人工复核 · 接地、稳定倒地与离座</title>
<style>
:root{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;color:#1c2b35;background:#f1f4f6;line-height:1.65}
*{box-sizing:border-box}body{margin:0}main{max-width:1100px;margin:auto;padding:25px 18px 50px}
h1{font-size:28px;line-height:1.3;margin:8px 0 12px}h2{font-size:23px}h3{font-size:18px;margin:15px 0 8px}
p{margin:10px 0}a{color:#145d80}button,select,input{font:inherit}button,select{border:1px solid #c4d1d9;background:white;padding:9px 13px;border-radius:8px;cursor:pointer}
button:hover{background:#e5eef3}button:disabled{opacity:.4;cursor:default}button.active{background:#174f64;color:white;border-color:#174f64}
.muted{color:#596974;font-size:14px}.banner{background:#fff2d6;border-left:4px solid #c89434;padding:13px 17px;border-radius:6px}.badge{color:#8c5b0d;font-weight:650}
nav{display:flex;flex-wrap:wrap;gap:8px;margin:22px 0}.card{border:1px solid #dce4e9;background:white;border-radius:14px;padding:20px;margin:18px 0}
.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.row label{margin-left:auto}.frame{width:100%;height:auto;display:block;border-radius:8px;background:#15202a}
.controls{margin:14px 0}.controls input{width:100%;margin:14px 0;accent-color:#17566d}video{display:block;background:#101820;width:100%;border-radius:9px}
ul{padding-left:23px}li{margin:8px 0}.answer{border:1px dashed #9fb3bf;border-radius:8px;padding:14px;background:#f6fafb}
.sheet{width:100%;height:auto;display:block}.small{font-size:13px}.warning{color:#855300}
@media(max-width:600px){main{padding:15px 12px}h1{font-size:23px}.card{padding:14px}.row label{margin-left:0}}
</style><main>
<a href="../index.html">← 返回 A/B/C 报告</a>
<h1>逐项人工复核</h1>
<p>① 首次非足部接地 → ② 稳定倒地 → ③ UTK s01 离座 → ④ UTK s10 离座</p>
<div class="banner"><span id="reviewStatus" class="badge">等待你的确认</span><br>
这里仅展示原始 RGB 与固定区域放大。请在聊天中逐项确认；浏览、播放或切换帧不会被当成确认，也不会改写任何标签。</div>
<nav id="nav" aria-label="复核项目"></nav>
<section class="card">
<h2 id="title"></h2><p id="definition"></p><p id="candidate" class="warning"></p>
<h3>判定时看什么</h3><ul id="criteria"></ul><p id="clock" class="muted"></p>
</section>
<section class="card">
<h3>原始观测逐帧查看</h3><p class="muted">左：完整原画面。右：同一帧的固定区域放大。文字标在画面之外；没有姿态骨架或预测标签。</p>
<img id="frame" class="frame" alt="当前原始候选帧与固定区域放大">
<div class="controls"><div class="row">
<button id="prev">上一可用帧</button><button id="next">下一可用帧</button>
<label for="frameSelect">源帧 <select id="frameSelect"></select></label></div>
<input id="slider" type="range" min="0" step="1" aria-label="切换可用源帧">
<div id="frameInfo"></div><p id="gap" class="muted"></p>
<a id="fullImage" target="_blank">打开当前大图</a></div>
</section>
<section class="card">
<h3>带前后文的片段</h3>
<div class="row"><button id="normal">正常速度</button><button id="slow">0.25× 慢放</button><button id="replay">从头播放</button></div>
<p id="videoNote" class="muted"></p><video id="video" controls playsinline preload="metadata"></video>
<p><a id="videoLink" target="_blank">打开当前视频</a></p>
<p class="muted">慢放只重复原观测，不生成中间姿态。视频播放器的进度是播放时间；请用画面上方的源帧号／源时间做标注。</p>
</section>
<section class="card"><h3>候选帧并排对照</h3><p><a id="sheetLink" target="_blank">打开原尺寸接触表</a></p><img id="sheet" class="sheet" alt="候选源帧接触表"></section>
<section class="card"><h3>在聊天中确认这一项</h3><p id="reply" class="answer"></p>
<p class="muted">不能确定时可以保留区间或未知。确认区间只表示你认可这个不确定范围，不会被写成精确帧。收到明确回复后，将仅更新对应标记并保留旧值与确认依据。</p></section>
<footer class="muted"><a href="media/manifest.json">素材来源与播放映射</a> · <a href="review-session.json">本轮复核队列与原始标签哈希</a>
<p><a id="annotationLink">当前人工复核标注</a> · <a href="../annotations/annotations.json">实验一冻结基线</a>。人工确认保存为独立当前版本，实验一的指标和审计仍对应冻结基线。</p>
<p><a href="../experiment3-20261021/index.html">查看四项复核汇总与实验三结果 →</a>。精确接地时刻未知时，时间结果也保留区间；实际提醒送达时间需另外测量。</p></footer>
</main><script>
const cases=__CASES__, guidance=__GUIDANCE__, review=__REVIEW__;
const $=id=>document.getElementById(id); let current,position=0,slow=true;
function frameView(i){
 position=Math.max(0,Math.min(current.frames.length-1,Number(i)));const f=current.frames[position];
 $("frame").src=f.file;$("fullImage").href=f.file;$("slider").value=position;$("frameSelect").value=position;
 $("frameInfo").textContent=`源帧 ${f.frame_id} · 源时间 ${Number(f.source_time_s).toFixed(3)} 秒 · 可用观测 ${position+1}/${current.frames.length}`;
 const before=current.frames[position-1],after=current.frames[position+1];
 $("gap").textContent=(before?`与上一可用观测相隔 ${f.frame_id-before.frame_id} 源帧 / ${(f.source_time_s-before.source_time_s).toFixed(3)} 秒。`:"已到上下文起点。")+
 (after?` 下一可用源帧为 ${after.frame_id}。`:" 已到上下文终点。");
 $("prev").disabled=position===0;$("next").disabled=position===current.frames.length-1;
}
function videoView(useSlow){
 slow=useSlow;const file=slow?current.video_slow:current.video_normal;
 $("video").pause();$("video").src=file;$("video").load();$("videoLink").href=file;
 $("slow").classList.toggle("active",slow);$("normal").classList.toggle("active",!slow);
 $("videoNote").textContent=slow?"已选择 0.25× 烘焙慢放：播放器按 1× 播放，即为源动作的四分之一速度。":"已选择正常速度：依据既有源时间显示，采样缺口保持上一观测。";
}
function choose(id){
 current=cases.find(c=>c.id===id)||cases[0];const g=guidance[current.id];
 $("title").textContent=g.title;$("definition").textContent=g.definition;$("candidate").textContent=g.candidate_note;
 $("criteria").replaceChildren(...g.look_for.map(t=>{const e=document.createElement("li");e.textContent=t;return e}));
 $("clock").textContent=current.clock_note||"";$("reply").textContent=g.reply;
 const decision=review.queue.find(x=>x.id===current.id);
 if(decision?.human_reviewed){$("candidate").textContent=decision.decision.display_summary;if(decision.decision.confirmed_definition)$("definition").textContent=decision.decision.confirmed_definition;$("reply").textContent=decision.status==="reviewed_proxy_only"?"已保存可见上升代理。物理离座仍待定，没有将代理帧当成物理离座时刻。":"本项已按你的明确回复保存。确认的是区间；精确帧与精确时刻仍为未知。其余项目需分别确认。";}
 $("sheet").src=current.contact_sheet;$("sheetLink").href=current.contact_sheet;
 $("frameSelect").replaceChildren(...current.frames.map((f,i)=>{const e=document.createElement("option");e.value=i;e.textContent=f.frame_id;return e}));
 $("slider").max=current.frames.length-1;
 const first=g.initial_frame??current.candidate_frame_range[0];frameView(Math.max(0,current.frames.findIndex(f=>f.frame_id>=first)));
 videoView(true);document.querySelectorAll("nav button").forEach(b=>b.classList.toggle("active",b.dataset.id===current.id));
 history.replaceState(null,"","#"+current.id);
}
cases.forEach(c=>{const b=document.createElement("button"),r=review.queue.find(x=>x.id===c.id);b.textContent=guidance[c.id].short+(r?.status==="reviewed_proxy_only"?"（仅代理）":r?.human_reviewed?" ✓":"");b.dataset.id=c.id;b.onclick=()=>choose(c.id);$("nav").append(b)});
$("prev").onclick=()=>frameView(position-1);$("next").onclick=()=>frameView(position+1);
$("slider").oninput=e=>frameView(e.target.value);$("frameSelect").onchange=e=>frameView(e.target.value);
$("normal").onclick=()=>videoView(false);$("slow").onclick=()=>videoView(true);
$("replay").onclick=()=>{$("video").currentTime=0;$("video").play()};
const completed=review.queue.filter(x=>x.human_reviewed).length;
const proxyCount=review.queue.filter(x=>x.status==="reviewed_proxy_only").length;
$("reviewStatus").textContent=completed?`已复核 ${completed}/4 项`+(proxyCount?` · 其中 ${proxyCount} 项仅确认上升代理，物理离座待定`:"")+(completed===4?" · 本轮队列已完成，整体标注仍为部分复核":` · ${4-completed} 项待复核`):"等待你的确认 · 当前未更新标注";
$("annotationLink").href=review.annotation_link;
window.addEventListener("hashchange",()=>choose(location.hash.slice(1)));
choose(location.hash.slice(1)||review.current_case_id);
</script></html>"""


def main():
    manifest = json.loads((OUT / "media/manifest.json").read_text())
    cases = manifest["cases"]
    assert {c["id"] for c in cases} == set(GUIDANCE)
    for case in cases:
        assert case["frames"]
        for file in [case["contact_sheet"], case["video_normal"], case["video_slow"]]:
            assert (OUT / file).is_file(), file
        for frame in case["frames"]:
            assert (OUT / frame["file"]).is_file(), frame["file"]
    view_keys = ("id", "clip_id", "frames", "contact_sheet", "video_normal", "video_slow",
                 "candidate_frame_range", "context_frame_range", "clock_note")
    view_cases = [{k: c[k] for k in view_keys} for c in cases]
    for case in view_cases:
        case["clock_note"] = (
            "时间来自 URFD 原始 RGB 对应的官方 CSV；本段源帧连续。"
            if case["clip_id"].startswith("urfd") else
            "时间按既有源帧号差÷30换算；现有观测不连续，慢放保持上一观测，不插入新画面。"
        )
    data = json.dumps(view_cases, ensure_ascii=False).replace("</", r"<\/")
    notes = json.dumps(GUIDANCE, ensure_ascii=False).replace("</", r"<\/")
    session = OUT / "review-session.json"
    if not session.exists():
        session.write_text(json.dumps({
            "schema": "fallpre.human_review.session.v1",
            "annotation_file": str(ANNOTATIONS.relative_to(ROOT)),
            "annotation_sha256_before": hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
            "status": "awaiting_explicit_user_decisions",
            "policy": "Viewing or selecting frames does not confirm a label. Only explicit chat decisions update the relevant item. Preserve interval uncertainty.",
            "queue": [
                {"id": c["id"], "clip_id": c["clip_id"], "status": "pending",
                 "human_reviewed": False, "decision": None, "guidance": GUIDANCE[c["id"]]}
                for c in cases
            ],
        }, ensure_ascii=False, indent=2) + "\n")
    state = json.loads(session.read_text())
    state["annotation_link"] = "../" + str((ROOT / state["annotation_file"]).relative_to(OUT.parent))
    review = json.dumps(state, ensure_ascii=False).replace("</", r"<\/")
    (OUT / "index.html").write_text(
        PAGE.replace("__CASES__", data).replace("__GUIDANCE__", notes).replace("__REVIEW__", review)
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
