'use strict';

// Production UI functions with mocked media/network/DOM boundaries, no real camera.
// Run: node --test tests/webcam_recording_ui.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const crypto = require('node:crypto');
const Recording = require('../assets/webcam/recording.js');

const html = fs.readFileSync(path.resolve(__dirname, '../assets/webcam/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const production = script.split("$('source-type').addEventListener")[0];
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
};
const flush = () => new Promise(resolve => setImmediate(resolve));
const result = () => ({
  width: 960, height: 540, persons: [], inference_ms: 20, action_ms: 10,
  model: {id: 'yolo26n-pose', label: 'YOLO26n-Pose', sha256: 'pose-hash'},
  action: {label: 'unknown', raw_label: 'unavailable', phase: 'unknown', target_index: null,
    reason: '关节不可见', diagnostics: {sample: {label: 'unknown', reason_code: 'missing_joints',
      reason: '关节不可见', stage: 'input', quality: {complete_side: false}}}},
  fall: {state: 'unobservable', observed: false}, fall_mode: 'pose',
  quality: {camera_stable: true}, fall_observation: {source: 'none', person: null, selection: {}},
});

function harness({record = true, fastTimeout = false, deferCapture = false, source = production} = {}) {
  const nodes = new Map(), fetches = [], downloads = [], captures = [], canvasCalls = [];
  let transform = {sx: 1, sy: 1, tx: 0, ty: 0};
  const transformStack = [];
  const canvasCall = (method, args) => canvasCalls.push({method, args, transform: {...transform},
    strokeStyle: canvasContext.strokeStyle, fillStyle: canvasContext.fillStyle});
  const canvasContext = {
    save() { transformStack.push({...transform}); },
    restore() { transform = transformStack.pop(); },
    translate(x, y) { transform.tx += transform.sx*x; transform.ty += transform.sy*y; },
    scale(x, y) { transform.sx *= x; transform.sy *= y; },
    drawImage(...args) { canvasCall('drawImage', args); }, setLineDash() {},
    beginPath() {}, moveTo(...args) { canvasCall('moveTo', args); },
    lineTo(...args) { canvasCall('lineTo', args); }, stroke() {},
    arc(...args) { canvasCall('arc', args); }, fill() {}, clearRect() {},
    strokeRect(...args) { canvasCall('strokeRect', args); }, fillRect() {},
    fillText(...args) { canvasCall('fillText', args); },
    measureText: text => ({width: String(text).length * 8}),
  };
  const element = () => ({
    children: [], style: {}, dataset: {}, checked: false, disabled: false, hidden: false,
    width: 960, height: 540, videoWidth: 960, videoHeight: 540, currentTime: 10,
    textContent: '', value: '', files: [], focusCalls: 0,
    get options() { return this.children.filter(item => item.isOption); },
    get selectedOptions() { return this.options.length ? this.options.filter(item => item.value === this.value) : [{textContent: 'Test camera'}]; },
    classList: {toggle() {}},
    getContext: () => canvasContext,
    toBlob(callback) {
      if (deferCapture) captures.push(callback);
      else queueMicrotask(() => callback(new Blob(['jpeg'], {type: 'image/jpeg'})));
    },
    append(...items) { this.children.push(...items); },
    prepend(...items) { this.children.unshift(...items); },
    replaceChildren(...items) {
      this.children = items;
      if (items.some(item => item.isOption)) this.value = (items.find(item => item.selected) || items[0]).value;
    },
    focus() { this.focusCalls++; },
    pause() {}, load() {}, removeAttribute() {}, remove() {}, click() {},
    addEventListener() {}, removeEventListener() {},
  });
  const document = {
    body: element(),
    getElementById(id) { if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id); },
    createElement: element,
  };
  const track = {stopCalls: 0, onended: null, stop() { this.stopCalls++; }, getSettings: () => ({deviceId: 'test-camera'})};
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  const media = {devices: [], enumerateCalls: 0, permissionCalls: 0};
  const recordingStop = deferred();
  const metadata = {
    id: 'recording-001', basename: 'sample-001', extension: 'webm', mime_type: 'video/webm',
    elapsed_ms: 1500, partial: false, byte_length: 9, errors: [],
    anchors: {requested: {performance_ms: 900, source_video_time_s: 10}},
  };
  const fakeRecording = {
    id: metadata.id, stopCalls: 0, getMetadata: () => metadata,
    stop(reason) { this.stopCalls++; this.reason = reason; return recordingStop.promise; },
  };
  let clock = 1000;
  const context = vm.createContext({
    document, console, Blob, URL, crypto: crypto.webcrypto, AbortController, DOMException,
    performance: {now: () => clock},
    setTimeout: (callback, ms) => setTimeout(callback, fastTimeout && ms === 5000 ? 15 : ms),
    clearTimeout, setInterval, clearInterval, queueMicrotask,
    MediaRecorder: function MediaRecorder() {}, FallpreRecording: Recording,
    Option: function Option(text, value, defaultSelected = false, selected = false) {
      return {isOption: true, textContent: text, value, defaultSelected, selected};
    },
    navigator: {mediaDevices: {
      async enumerateDevices() { media.enumerateCalls++; return media.devices; },
      async getUserMedia() { media.permissionCalls++; return stream; },
    }},
    fakeStream: stream, fakeRecording, recordRequested: record,
    fetch(url, options) {
      if (url === '/stop') { fetches.push({url, options}); return Promise.resolve({ok: true}); }
      const pending = deferred(); fetches.push({url, options, pending}); return pending.promise;
    },
  });
  vm.runInContext(source, context);
  const run = code => vm.runInContext(code, context);
  run(`
    $('source-type').value='camera';$('pose-model').value='yolo26n-pose';$('fall-mode').value='pose';
    modelOptions=[{id:'yolo26n-pose',label:'YOLO26n-Pose',available:true}];ready=true;
    session={streamId:'stream-00000000001',recordRequested,sourceType:'camera',sourceName:'Test camera',modelId:'yolo26n-pose',fallMode:'pose'};
    run=1;active=true;starting=false;stopping=false;sessionController=new AbortController();stream=fakeStream;
    resetResults();recording=recordRequested?fakeRecording:null;
  `);
  context.downloads = downloads;
  run('downloadBlob=(blob,name)=>downloads.push({blob,name});');
  return {
    run, context, nodes, track, fetches, metadata, fakeRecording, recordingStop, downloads, canvasCalls, media,
    setClock(value) { clock = value; },
    finishCapture() { assert(captures.length); captures.shift()(new Blob(['jpeg'], {type: 'image/jpeg'})); },
    submit(t = 11.5) { return run(`inferFrame(1,${t})`); },
    respond(payload = result(), ok = true) {
      const request = fetches.find(item => item.url === '/infer' && !item.responded);
      assert(request, 'an inference request is pending'); request.responded = true;
      request.pending.resolve({ok, json: async () => payload});
    },
    finishVideo() { recordingStop.resolve({blob: new Blob(['raw-video'], {type: 'video/webm'}), metadata}); },
  };
}

test('full production script parses and its static DOM references exist', () => {
  new vm.Script(script);
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length);
  const references = [...script.matchAll(/\$\('([^']+)'\)/g)].map(match => match[1]);
  assert(references.every(id => ids.includes(id)));
  assert.match(html, /<script src="\/recording\.js"><\/script>/);
});

test('scrubbing reuses saved frames and window probabilities while inference time keeps advancing', () => {
  const h = harness({record: false});
  h.context.payload = result();
  h.run(`
    session.sourceType='video';session.agentEnabled=true;
    videoReview.duration=30;fileURL='blob:demo';
    reviewSource.readyState=2;
    activeAgentRun={run_id:'test-video',provider_status:{provider:'aliyun'}};
    agentHistoryCache=[
      {cursor:1,source_time:2.6,result:{status:'ready',fall_probability:0.7,
       source_window:{start_s:0.5,end_s:2.5},provider_metadata:{answers:{
        evidence_sufficiency:{probabilities:{sufficient:0.9}},
        next_step:{probabilities:{emit_result:0.8}}}}}},
      {cursor:2,source_time:2.5,result:{status:'ready',fall_probability:0.2,
       source_window:{start_s:0.5,end_s:2.5}}},
      {cursor:3,source_time:3.5,result:{status:'ready',fall_probability:1,
       source_window:{start_s:2.6,end_s:3.4}}},
    ];
    lastSource=2;renderResult(payload,2,30);
    lastSource=3;renderResult(payload,3,30);
    reviewVideoAt(2);seekVideoReviewSource();paintVideoReviewFrame();
  `);
  assert.equal(h.nodes.get('video-review-image').hidden, false);
  assert.match(h.nodes.get('source-time').textContent, /回看时间 00:02\.0/);
  assert.match(h.nodes.get('agent-result-scope').textContent, /回看/);
  assert.equal(h.nodes.get('agent-probability').textContent, '70.0%');
  assert.equal(h.nodes.get('agent-sufficient').textContent, '90.0%');
  assert.equal(h.nodes.get('agent-emit_result').textContent, '80.0%');
  assert.equal(h.run('video.currentTime'), 10);
  assert.equal(h.run('lastSource'), 3);
  assert.equal(h.fetches.length, 0);

  h.run(`
    lastSource=4;renderResult(payload,4,30);
    renderAgentProjection({projection:{current:{status:'ready',current_applicable:true,
      fall_probability:0.4,source_window:{start_s:2,end_s:4}}}});
  `);
  assert.equal(h.nodes.get('agent-probability').textContent, '70.0%');
  assert.equal(h.run('videoReview.time'), 2);
  assert.equal(Number(h.nodes.get('video-progress').value), 2);

  h.run('reviewVideoAt(25);seekVideoReviewSource()');
  assert.equal(h.run('videoReview.time'), 4);
  assert.equal(h.nodes.get('agent-status').textContent, '此处暂无决策');
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  h.run('exitVideoReview()');
  assert.equal(h.nodes.get('video-review-image').hidden, true);
  assert.match(h.nodes.get('source-time').textContent, /源时间 00:04\.0/);
  assert.equal(h.nodes.get('agent-probability').textContent, '40.0%');
  assert.equal(h.fetches.length, 0);
});

test('recordings retain more than 600 real observation calls; ordinary sessions retain the last 600', () => {
  for (const record of [true, false]) {
    const h = harness({record}); h.context.payload = result();
    h.run('for(let i=0;i<1001;i++)updateDiagnostics(payload,i/20);');
    const report = h.run('buildDiagnosticReport()');
    assert.equal(report.sample_retention.retained, record ? 1001 : 600);
    assert.equal(report.sample_retention.limit, record ? null : 600);
    assert.equal(report.sample_retention.omitted, record ? 0 : 401);
    assert.equal(report.sample_retention.truncated, !record);
    assert.equal(report.samples[0].t, record ? 0 : 401/20);
    assert.equal(report.samples.at(-1).t, 50);
    assert.equal(h.run('diagnosticState.received'), 1001);
    assert.equal(report.samples[0].reason_code, 'missing_joints');
  }
});

test('report snapshots isolate nested samples, requests and metadata from later mutation', () => {
  const h = harness(); h.context.payload = result();
  h.run(`updateDiagnostics(payload,12,{recording_id:'recording-001',recording_time_s:2});
    diagnosticState.requests.push({source_time_s:12,status:'returned'});
    globalThis.snapshot=buildDiagnosticReport(diagnosticState,recording.getMetadata());
    diagnosticState.samples[0].label='standing';diagnosticState.requests[0].status='changed';
    diagnosticState.samples[0].client_timing.recording_time_s=999;
    recording.getMetadata().anchors.requested.source_video_time_s=999;
  `);
  assert.equal(h.run('snapshot.samples[0].label'), 'unknown');
  assert.equal(h.run('snapshot.samples[0].client_timing.recording_time_s'), 2);
  assert.equal(h.run('snapshot.requests[0].status'), 'returned');
  assert.equal(h.run('snapshot.recording.anchors.requested.source_video_time_s'), 10);
});

test('stop drains the final inference before releasing the camera and snapshots its paired timing', async () => {
  const h = harness();
  const inference = h.submit(11.5); await flush();
  assert.equal(h.fetches[0].options.headers['X-Source-Time'], '11.5');
  assert.equal(h.fetches[0].options.headers['X-Source-Type'], 'camera');
  assert.equal(h.run('diagnosticState.requests[0].status'), 'pending');
  const finishing = h.run("stop('Finished','Stopped','user')");
  assert.equal(h.run('stopping'), true);
  assert.equal(h.run('active'), false);
  assert.equal(h.fakeRecording.stopCalls, 1);
  assert.equal(h.track.stopCalls, 0);
  assert.equal(h.run('sessionController.signal.aborted'), false);
  assert.equal(h.run('stop()'), finishing, 'duplicate stop shares the same finalization');
  h.finishVideo(); await flush();
  assert.equal(h.track.stopCalls, 0, 'video finalization alone must not drop the in-flight diagnostic');
  assert.equal(h.run('savedRecordings.length'), 0);
  h.setClock(1200); h.respond();
  assert.equal(await inference, true);
  await finishing;
  assert.equal(h.track.stopCalls, 1);
  assert.equal(h.run('stopping'), false);
  assert.equal(h.run('recording'), null);
  const stored = h.run('savedRecordings[0]');
  const report = JSON.parse(await stored.json.text());
  assert.equal(await stored.blob.text(), 'raw-video');
  assert.equal(report.recording.diagnostic_drain.status, 'settled');
  assert.equal(report.samples.length, 1);
  assert.equal(report.requests.length, 1);
  assert.equal(report.requests[0].status, 'returned');
  assert.equal(report.samples[0].t, 11.5, 'source timestamp is preserved for the action state machine');
  assert.equal(report.samples[0].client_timing.recording_time_s, 1.5);
  assert.equal(report.samples[0].client_timing.recording_elapsed_s, 0.1);
  assert.equal(report.samples[0].client_timing.recording_id, report.recording.id);
  assert.equal(report.samples[0].client_timing.source_time_s, report.requests[0].source_time_s);
  assert.equal(report.samples[0].client_timing.captured_performance_ms, report.requests[0].captured_performance_ms);
  assert.equal(report.samples[0].client_timing.latency_ms, 200);
  assert.equal(report.samples[0].client_timing.input_width, 960);
  assert.equal(report.recording.video_filename, 'sample-001.webm');
  assert.equal(report.recording.diagnostics_filename, 'sample-001.json');
  assert.equal(report.recording.mirror, false);
  const serialized = await stored.json.text();
  h.run('session.recordRequested=false;resetResults();updateDiagnostics({},50);');
  assert.equal(await stored.json.text(), serialized, 'starting another session does not rewrite the saved pair');
});

test('finalization waits for the encoded tail even when inference has already returned', async () => {
  const h = harness();
  const inference = h.submit(); await flush();
  const finishing = h.run('stop()');
  h.respond(); await inference; await flush();
  assert.equal(h.track.stopCalls, 0);
  assert.equal(h.run('savedRecordings.length'), 0);
  h.finishVideo(); await finishing;
  assert.equal(h.track.stopCalls, 1);
  assert.equal(h.run('savedRecordings.length'), 1);
});

test('stopping during JPEG preparation still drains that captured frame', async () => {
  const h = harness({deferCapture: true});
  const inference = h.submit();
  assert.equal(h.run('diagnosticState.requests[0].status'), 'capturing');
  const finishing = h.run('stop()'); h.finishVideo(); await flush();
  assert.equal(h.fetches.length, 0);
  assert.equal(h.track.stopCalls, 0);
  h.finishCapture(); await flush(); h.respond();
  assert.equal(await inference, true); await finishing;
  const report = JSON.parse(await h.run('savedRecordings[0].json').text());
  assert.equal(report.samples.length, 1);
  assert.equal(report.requests[0].status, 'returned');
});

test('late JPEG callbacks do not overwrite the timeout cause in retained diagnostics', async () => {
  const h = harness({fastTimeout: true, deferCapture: true});
  const inference = h.submit();
  const finishing = h.run('stop()'); h.finishVideo(); await finishing;
  const stored = h.run('savedRecordings[0]');
  assert.equal(JSON.parse(await stored.json.text()).requests[0].status, 'timeout_on_stop');
  h.finishCapture(); assert.equal(await inference, false);
  assert.equal(h.run('diagnosticState.requests[0].status'), 'timeout_on_stop');
  assert.equal(h.fetches.filter(item => item.url === '/infer').length, 0);
});

test('drain timeout is retained as a failed observation request and rejects late results', async () => {
  const h = harness({fastTimeout: true});
  const inference = h.submit(); await flush();
  const finishing = h.run('stop()'); h.finishVideo();
  await finishing;
  const stored = h.run('savedRecordings[0]');
  const report = JSON.parse(await stored.json.text());
  assert.equal(report.recording.diagnostic_drain.status, 'timeout');
  assert.equal(report.requests.length, 1);
  assert.equal(report.requests[0].status, 'timeout_on_stop');
  assert.match(report.requests[0].error, /仍未返回/);
  assert.equal(report.samples.length, 0, 'an unavailable HTTP response must not count as model unknown');
  assert.equal(h.track.stopCalls, 1);
  assert.equal(h.fetches[0].options.signal.aborted, true);
  h.respond(); assert.equal(await inference, false);
  assert.equal(h.run('diagnosticState.received'), 0);
  assert.equal((JSON.parse(await stored.json.text())).samples.length, 0);
});

test('HTTP failures remain attributable to the captured request rather than fabricated action labels', async () => {
  const h = harness();
  const inference = h.submit(); const rejected = assert.rejects(inference, /推理失败/);
  await flush();
  const finishing = h.run('stop()'); h.finishVideo();
  h.respond({error: '推理失败'}, false);
  await rejected; await finishing;
  const report = JSON.parse(await h.run('savedRecordings[0].json').text());
  assert.equal(report.recording.diagnostic_drain.status, 'failed');
  assert.equal(report.requests[0].status, 'error');
  assert.equal(report.requests[0].error, '推理失败');
  assert.equal(report.requests[0].recording_id, report.recording.id);
  assert.equal(report.samples.length, 0);
});

test('empty recordings still expose the paired diagnostics export', async () => {
  const h = harness();
  h.run('exportDiagnostics()');
  assert.equal(h.downloads.length, 1);
  const report = JSON.parse(await h.downloads[0].blob.text());
  assert.equal(report.recording.id, 'recording-001');
  assert.equal(report.samples.length, 0);
  assert.equal(h.downloads[0].name, 'sample-001.json');
});

test('source cadence uses source timestamps independently of response timing and resets per session', () => {
  const h = harness(); h.context.payload = result();
  for (const t of [0, .133, .266, .399]) {
    h.setClock(1000 + t * 9000);
    h.run(`updateDiagnostics(payload,${t});`);
  }
  assert.match(h.nodes.get('source-cadence').textContent, /7\.5 次 \/ 秒/);
  assert.match(h.nodes.get('source-cadence').textContent, /133 ms/);
  h.run('resetResults();');
  assert.equal(h.nodes.get('source-cadence').textContent, '动作采样：等待连续观测');
  h.run('updateDiagnostics(payload,10);');
  assert.equal(h.nodes.get('source-cadence').textContent, '动作采样：等待连续观测');
});

test('paired export identifies the active action pipeline independently of pose weights', () => {
  const h = harness();
  h.run(`serverStatus={action_pipeline:{id:'webcam-actions',label:'实时起坐',revision:'timing-static-v1',private_value:'omit'}};resetResults();`);
  const report = h.run('buildDiagnosticReport()');
  assert.equal(report.action_pipeline.revision, 'timing-static-v1');
  assert.equal(report.model.id, 'yolo26n-pose');
  assert.equal(report.action_pipeline.private_value, undefined);
});


test('recorded diagnostics retain static-recovery support and transition veto evidence', () => {
  const h = harness(); h.context.payload = result();
  h.context.payload.action.diagnostics.sample.reconciliation_evidence = {
    phase: 'standing', support: 'upright_geometry', duration_s: .65, samples: 6,
    stable: true, transition_blocked: false, relative_core_spread: .008, common_core_travel: .05,
    coordinates: [[1,2,3]],
  };
  h.run('updateDiagnostics(payload,12);');
  const evidence = h.run('buildDiagnosticReport().samples[0].reconciliation_evidence');
  assert.equal(evidence.support, 'upright_geometry');
  assert.equal(evidence.phase, 'standing');
  assert.equal(evidence.transition_blocked, false);
  assert.equal(evidence.common_core_travel, .05);
  assert.equal(evidence.coordinates, undefined);
});

test('paired recording exports rise provenance and shows the same confirmation reason', async () => {
  const h = harness(); h.context.payload = result();
  h.context.payload.action.diagnostics.sample.rise_evidence = {
    code:'rise_confirmed', reason:'已观察到从坐姿持续抬髋伸腿',
    rising:true, veto_seated:true, stop_rise:false, applied:true,
    anchor_available:true, anchor_kind:'observed_descent', anchor_age_s:.4,
    hip_up:.12, shoulder_up:.08, hip_absolute_up:.13, shoulder_absolute_up:.09, horizontal:.02, thigh_extension:.04, knee_opening:.05,
    motion_duration_s:.266, positive_steps:2, additional_up:.012, current_continues:true, current_feet_stable:true,
    foot_motion:.01, reset_reason:null, standing_complete:false,
    keypoints:[[1,2,3]], internal:{image:'private'},
  };
  h.run('updateDiagnostics(payload,12);');
  const text = h.nodes.get('diagnostic-rise').textContent;
  assert.match(text, /已观察到从坐姿持续抬髋伸腿/);
  assert.match(text, /已用于当前判断/);
  assert.match(text, /离座证据不再支持坐姿/);
  assert.match(text, /0\.27 秒/);
  assert.match(text, /2 次向上变化/);
  h.context.finalized = {blob:new Blob(['raw-video'],{type:'video/webm'}),metadata:h.metadata};
  h.run("keepRecording(finalized,{status:'settled'});");
  const report = JSON.parse(await h.run('savedRecordings[0].json.text()'));
  const evidence = report.samples[0].rise_evidence;
  assert.equal(evidence.code,'rise_confirmed');
  assert.equal(evidence.hip_up,.12);
  assert.equal(evidence.hip_absolute_up,.13);
  assert.equal(evidence.shoulder_absolute_up,.09);
  assert.equal(evidence.horizontal,.02);
  assert.equal(evidence.anchor_kind,'observed_descent');
  assert.equal(evidence.standing_complete,false);
  assert.equal(evidence.positive_steps,2);
  assert.equal(evidence.additional_up,.012);
  assert.equal(evidence.current_feet_stable,true);
  assert.equal(evidence.keypoints,undefined);
  assert.equal(evidence.internal,undefined);
  h.run('resetResults();');
  assert.equal(h.nodes.get('diagnostic-rise').textContent,'等待观测');
});

test('rise evidence absence and rejected scalar fields do not claim an accepted rise', () => {
  const h=harness(); h.context.payload=result();
  h.run('updateDiagnostics(payload,12);');
  assert.equal(h.run('buildDiagnosticReport().samples[0].rise_evidence'),null);
  assert.match(h.nodes.get('diagnostic-rise').textContent,/未提供/);
  h.context.payload.action.diagnostics.sample.rise_evidence={
    reason:'等待实际抬髋',applied:false,veto_seated:false,
    positive_steps:[1,2],motion_duration_s:NaN,hip_up:Infinity,
  };
  h.run('updateDiagnostics(payload,12.13);');
  const evidence=h.run('buildDiagnosticReport().samples[1].rise_evidence');
  assert.equal(evidence.positive_steps,null);
  assert.equal(evidence.motion_duration_s,null);
  assert.equal(evidence.hip_up,null);
  assert.equal(h.nodes.get('diagnostic-rise').textContent,'等待实际抬髋');
});

test('recording retains handoff, target gap and background recovery and resets the display', () => {
  const h=harness();h.context.payload=result();
  Object.assign(h.context.payload.action.diagnostics.sample,{
    static_evidence:{code:'static_continuity',stable:true,phase:'seated',points:[[1,2,3]]},
    target_continuity:{history_retained:true,gap_s:.2},
  });
  h.context.payload.quality={camera_stable:true,code:'reference_recovered',reason:'新参考已验证',
    metrics:{matches:36,reference_generation:2,reference_trusted:true,pixels:[[1,2,3]]}};
  h.run('updateDiagnostics(payload,12);');
  const s=h.run('buildDiagnosticReport().samples[0]');
  assert.equal(s.static_evidence.code,'static_continuity');
  assert.equal(s.static_evidence.points,undefined);
  assert.equal(s.target_continuity.history_retained,true);
  assert.equal(s.scene_quality.code,'reference_recovered');
  assert.equal(s.scene_quality.metrics.matches,36);
  assert.equal(s.scene_quality.metrics.pixels,undefined);
  assert.match(h.nodes.get('diagnostic-continuity').textContent,/保留历史，当前显示未知/);
  assert.match(h.nodes.get('diagnostic-background').textContent,/新参考已验证/);
  h.run('resetResults()');
  for(const id of ['diagnostic-static','diagnostic-continuity','diagnostic-background'])
    assert.equal(h.nodes.get(id).textContent,'等待观测');
});

test('capture plans distinguish development and holdout, and freeze selection for each session', async () => {
  const h=harness();
  h.run("$('capture-segment').value='02';session.capturePlan=snapshotCapturePlan();resetResults();$('capture-segment').value='05';renderCapturePlan();");
  assert.match(h.nodes.get('capture-purpose').textContent,/独立验证/);
  assert.equal(h.run('diagnosticState.capture_plan.id'),'02');
  assert.equal(h.run('diagnosticState.capture_plan.split'),'development');
  h.context.finalized={blob:new Blob(['original-pixels'],{type:'video/webm'}),metadata:{...h.metadata,stop_reason:'duration_limit'}};
  h.run("keepRecording(finalized,{status:'settled'});");
  const saved=h.run('savedRecordings[0]');
  const report=JSON.parse(await saved.json.text());
  assert.equal(report.capture_plan.id,'02');
  assert.equal(report.recording.capture_plan.id,'02');
  assert.equal(report.capture_plan.baseline.action_revision,'2026-10-28.19');
  assert.equal(report.capture_plan.label_status,'unannotated');
  const row=h.nodes.get('recordings-list').children[0];
  assert.match(row.children[0].textContent,/第 02 段.*床边/);
  await row.children[2].children[0].onclick();
  const buf=Buffer.from(await h.downloads[0].blob.arrayBuffer());
  const entries={}; let offset=0;
  while(buf.readUInt32LE(offset)===0x04034b50){
    const size=buf.readUInt32LE(offset+18),nameLength=buf.readUInt16LE(offset+26),extraLength=buf.readUInt16LE(offset+28);
    const name=buf.subarray(offset+30,offset+30+nameLength).toString();
    const start=offset+30+nameLength+extraLength;entries[name]=buf.subarray(start,start+size);offset=start+size;
  }
  assert.equal(entries['sample-001.webm'].toString(),'original-pixels');
  assert.equal(JSON.parse(entries['sample-001.json']).capture_plan.id,'02');
  assert.match(entries['录制说明.txt'].toString(),/第 02 段：床边自然起坐/);
  assert.match(entries['录制说明.txt'].toString(),/不是实际动作标签/);
  h.run('session.capturePlan=snapshotCapturePlan();resetResults();');
  assert.equal(h.run('buildDiagnosticReport().capture_plan.split'),'holdout');
  assert.equal(JSON.parse(await saved.json.text()).capture_plan.id,'02');
});

test('recording filenames encode selected segment and split; recording locks its selector', () => {
  const h=harness();
  h.context.capturedOptions=[];
  h.run(`FallpreRecording={...FallpreRecording,RawRecordingSession:class {
    constructor(options){capturedOptions.push(options);}
    start(){} getMetadata(){return {anchors:{requested:{performance_ms:1000}}}}
  }};`);
  for(const id of ['01','02','03','04','05','06']){
    h.run(`$('capture-segment').value='${id}';session.capturePlan=snapshotCapturePlan();startRecording();clearInterval(recordTimer);`);
    const options=h.context.capturedOptions.at(-1);
    assert.match(options.basename,new RegExp(`^fallpre-camera-seg${id}-${Number(id)>4?'holdout':'dev'}-`));
    assert.equal(h.nodes.get('capture-segment').disabled,true);
    assert.match(h.nodes.get('recording-status').textContent,new RegExp(`第 ${id} 段`));
  }
});

test('batch recording rejects a changed baseline or model while ordinary preview stays usable', () => {
  const h=harness();
  h.run("active=false;recording=null;serverStatus={action_pipeline:{revision:'2026-10-28.19'},device:'cpu',imgsz:640};$('fall-mode').value='rgb';controls();");
  assert.equal(h.nodes.get('record').disabled,false);
  h.run("$('pose-model').value='yolo11n-pose';controls();");
  assert.equal(h.nodes.get('record').disabled,true);
  h.run("$('pose-model').value='yolo26n-pose';serverStatus.action_pipeline.revision='future';controls();");
  assert.equal(h.nodes.get('record').disabled,true);
  assert.equal(h.nodes.get('start').disabled,false);
  assert.match(h.nodes.get('capture-config-note').textContent,/版本/);
});

test('independent axes display together and export scalar evidence without replacing actions or falls', () => {
  const h=harness();h.context.payload=result();
  h.context.payload.behavior={schema_version:1,target_index:null,target_epoch:4,
    posture:{label:'upright',text:'直立',reason:'躯干与腿部支持直立'},
    transition:{label:'none',text:'无姿态转换'},
    locomotion:{label:'walking',text:'行走',reason:'连续位移与脚步变化'},
    turning:{label:'turning',text:'转身',reason:'躯干方向连续改变'},
    path_change:{label:'changing',text:'路径改变'},
    metrics:{foot_motion:.19,heading_change:.32,points:[[1,2,3]],invalid:Infinity},
    quality:{camera_stable:true,source:'observed',keypoints:[[1,2,3]]}};
  h.run('renderResult(payload,12,40);');
  assert.equal(h.nodes.get('behavior-posture').textContent,'直立');
  assert.equal(h.nodes.get('behavior-locomotion').textContent,'行走');
  assert.equal(h.nodes.get('behavior-turning').textContent,'转身');
  assert.match(h.nodes.get('behavior-path').textContent,/路径改变/);
  assert.equal(h.nodes.get('action').textContent,'未知');
  assert.match(h.nodes.get('fall-state').textContent,/无法判断/);
  const report=h.run('buildDiagnosticReport()'),b=report.samples[0].behavior;
  assert.equal(report.version,5);
  assert.equal(b.locomotion.label,'walking');assert.equal(b.turning.label,'turning');
  assert.equal(b.metrics.foot_motion,.19);assert.equal(b.metrics.points,undefined);assert.equal(b.metrics.invalid,undefined);
  assert.equal(b.quality.camera_stable,true);assert.equal(b.quality.keypoints,undefined);
  h.run('resetResults();');
  assert.equal(h.nodes.get('behavior-turning').textContent,'不可判断');
  assert.equal(h.run('diagnosticState.samples.length'),0);
});

test('missing behavior never presents stillness or not-turning as an observed result', () => {
  const h=harness();h.context.payload=result();
  h.run('renderResult(payload,12,40);');
  for(const id of ['behavior-posture','behavior-locomotion','behavior-turning'])
    assert.equal(h.nodes.get(id).textContent,'不可判断');
  assert.equal(h.run('buildDiagnosticReport().samples[0].behavior'),null);
});

test('behavior labels stay on their own associated box and do not transfer to the fall target', () => {
  const h=harness();h.context.payload=result();
  const person=(x)=>({box:[x,30,x+100,400],keypoints:[]});
  h.context.payload.persons=[person(20),person(300)];
  h.context.payload.action.target_index=0;h.context.payload.action.text='站立';
  h.context.payload.action.selection={target_epoch:3};
  h.context.payload.behavior={target_index:0,posture:{text:'直立'},locomotion:{text:'行走'},turning:{text:'转身'}};
  h.context.payload.fall={state:'confirmed',label:'FALL',observed:true,target_epoch:9,event_target_epoch:9};
  h.context.payload.fall_observation={source:'raw',person:h.context.payload.persons[1],selection:{raw_index:1,target_epoch:9}};
  const rows=h.run('frameOverlays(payload)');
  const action=rows.find(row=>row.role==='action'),fall=rows.find(row=>row.role==='fall');
  assert.equal(action.color,'#91efbf');assert.match(action.label,/动作：站立/);
  assert.match(action.lines.join(' '),/直立.*行走.*转身/);
  assert.equal(fall.color,'#ff8f91');assert.equal(fall.lines.length,0);assert.equal(fall.bottom,true);
  assert.equal(fall.person.box[0],300);
  h.context.payload.persons=[];h.context.payload.action.target_index=null;h.context.payload.behavior.target_index=null;
  h.context.payload.fall_observation={source:'none',person:null,selection:{}};
  assert.equal(h.run('frameOverlays(payload).length'),0,'missing targets do not fabricate boxes');
});

test('an unobservable fall branch does not gray out an independently associated behavior box', () => {
  const h=harness();h.context.payload=result();
  const person={box:[20,50,200,420],keypoints:[]};
  h.context.payload.persons=[person];h.context.payload.action.target_index=0;h.context.payload.action.text='坐着';
  h.context.payload.behavior={target_index:0,posture:{text:'坐姿'},locomotion:{text:'静止'},turning:{text:'不可判断'}};
  h.context.payload.fall={state:'unobservable',observed:false,target_epoch:1};
  h.context.payload.fall_observation={source:'raw',person,selection:{raw_index:0,target_epoch:1}};
  const rows=h.run('frameOverlays(payload)');
  assert.equal(rows.find(row=>row.role==='action').color,'#91efbf');
  assert.equal(rows.find(row=>row.role==='fall').drawBox,false);
  assert.equal(rows.find(row=>row.role==='fall').lines.length,0);
});

test('demo labels anchor to the displayed box upper-left, including mirrored and frame-edge targets', () => {
 const h=harness();
 const cases=[
  {box:[350,180,550,500],mirror:false,x:350,y:72,placement:'above'},
  {box:[350,10,550,500],mirror:false,x:350,y:13,placement:'inside-top'},
  {box:[740,10,940,500],mirror:true,x:20,y:13,placement:'inside-top'},
  {box:[1,5,959,535],mirror:false,x:1,y:8,placement:'inside-top'},
 ];
 for(const c of cases){h.context.c=c;const a=h.run('overlayPosition(c.box,180,104,960,540,c.mirror)');
  assert.equal(a.x,c.x);assert.equal(a.y,c.y);assert.equal(a.placement,c.placement);
  assert(a.x>=0&&a.x+180<=960&&a.y>=0&&a.y+104<=540);
 }
});

test('presentation merges only the same current subject and prioritizes the red label', () => {
 const h=harness();h.context.payload=result();
 const person={box:[20,50,220,430],keypoints:[]};
 h.context.payload.persons=[person];h.context.payload.action={label:'standing',text:'站立',target_index:0,selection:{target_epoch:1}};
 h.context.payload.behavior={target_index:0,posture:{label:'upright',text:'直立'},locomotion:{label:'walking',text:'行走'},turning:{label:'turning',text:'转身'}};
 let rows=h.run('presentationOverlays(payload)');assert.equal(rows.length,1);assert.equal(rows[0].label,'站立');
 assert.equal(rows[0].lines.length,1);assert.equal(rows[0].lines[0],'行走 · 转身');
 h.context.payload.fall={state:'confirmed',observed:true,target_epoch:9,event_target_epoch:9};
 h.context.payload.fall_observation={source:'raw',person,selection:{raw_index:0,target_epoch:9}};
 rows=h.run('presentationOverlays(payload)');assert.equal(rows.length,1);assert.equal(rows[0].color,'#ff8f91');assert.equal(rows[0].label,'跌倒已确认');
 const other={box:[400,50,600,430],keypoints:[]};h.context.payload.persons.push(other);
 h.context.payload.fall_observation.person=other;h.context.payload.fall_observation.selection.raw_index=1;
 rows=h.run('presentationOverlays(payload)');assert.equal(rows.length,2);
 assert.equal(rows[0].label,'站立');assert.equal(rows[1].label,'跌倒已确认');
});

test('lying posture stays dashed yellow and cannot inherit red event wording', () => {
 const h=harness();h.context.payload=result();h.context.payload.source_time=12;
 h.context.payload.persons=[{box:[20,50,400,200],keypoints:[]}];
 h.context.payload.action.target_index=0;
 h.context.payload.lying_observation={label:'lying',observed:true,target_index:0,source_time:12};
 const rows=h.run('presentationOverlays(payload)');assert.equal(rows.length,1);
 assert.equal(rows[0].label,'倒卧姿态');assert.equal(rows[0].color,'#ffd179');assert.equal(rows[0].dashed,true);
 assert.equal(rows[0].lines[0],'跌倒过程未确认');
});

function measuredPerson() {
  const keypoints=Array.from({length:17},()=>[0,0,0]);
  keypoints[5]=[80,100,.9];keypoints[6]=[120,100,.9];
  keypoints[11]=[85,200,.9];keypoints[12]=[115,200,.9];
  keypoints[13]=[85,280,.1];keypoints[14]=[NaN,280,.9];
  keypoints[15]=[970,390,.9];keypoints[16]=[-10,390,.9];
  return {box:[20,50,220,430],keypoints};
}

test('unconfirmed targets retain the v18 gray skeleton and candidate box without a display toggle', () => {
  const h=harness();h.context.payload=result();
  h.context.payload.persons=[measuredPerson()];
  h.run('draw({width:960,height:540},payload)');
  const joints=h.canvasCalls.filter(call=>call.method==='arc');
  assert.deepEqual(joints.map(call=>call.args.slice(0,2)),[[80,100],[120,100],[85,200],[115,200]]);
  assert(h.canvasCalls.some(call=>call.method==='lineTo'),'measured connected joints have bones');
  const boxes=h.canvasCalls.filter(call=>call.method==='strokeRect');
  assert.equal(boxes.length,1,'current candidates remain visible before confirmation');
  assert.equal(boxes[0].strokeStyle,'#758794');
  assert(joints.every(call=>call.strokeStyle==='#758794'&&call.fillStyle==='#758794'));
  assert.equal(h.canvasCalls.find(call=>call.method==='fillText').args[0],'目标待确认');
  assert.doesNotMatch(html,/id="(?:mirror|show-skeleton|show-candidates)"/);
});

test('merged yellow and red boxes keep the current pose skeleton exactly once', () => {
  for(const state of ['suspected','confirmed','lying']){
    const h=harness();h.context.payload=result();const p=measuredPerson();
    Object.assign(h.context.payload,{source_time:12,persons:[p],action:{label:'standing',text:'站立',target_index:0}});
    if(state==='lying')h.context.payload.lying_observation={label:'lying',observed:true,target_index:0,source_time:12};
    else{
      h.context.payload.fall={state,observed:true,target_epoch:9,event_target_epoch:9};
      h.context.payload.fall_observation={source:'raw',person:p,selection:{raw_index:0,target_epoch:9}};
    }
    h.run('draw({width:960,height:540},payload)');
    assert.equal(h.canvasCalls.filter(call=>call.method==='arc').length,4,state+' retains measured joints');
    const boxes=h.canvasCalls.filter(call=>call.method==='strokeRect');
    assert.equal(boxes.length,1,state+' has one merged box');
    assert.equal(boxes[0].strokeStyle,state==='confirmed'?'#ff8f91':'#ffd179');
  }
});

test('RGB-only detections cannot reuse an earlier pose skeleton', () => {
  const h=harness();h.context.payload=result();h.context.payload.persons=[measuredPerson()];
  h.run('draw({width:960,height:540},payload)');
  h.canvasCalls.length=0;
  h.context.payload.persons=[];
  h.context.payload.fall={state:'confirmed',observed:true,target_epoch:9,event_target_epoch:9};
  h.context.payload.fall_observation={source:'rgb_fall',person:{box:[20,50,400,200],keypoints:[]},selection:{target_epoch:9}};
  h.run('draw({width:960,height:540},payload)');
  assert.equal(h.canvasCalls.filter(call=>call.method==='strokeRect').length,1);
  assert.equal(h.canvasCalls.filter(call=>['arc','moveTo','lineTo'].includes(call.method)).length,0);
});

test('camera, skeleton and boxes share the default mirror transform while labels remain readable', () => {
  const h=harness();h.context.payload=result();h.context.payload.persons=[measuredPerson()];
  h.context.payload.action={label:'standing',text:'站立',target_index:0};
  h.run('draw({width:960,height:540},payload)');
  const mirrored={sx:-1,sy:1,tx:960,ty:0};
  for(const call of h.canvasCalls.filter(call=>['drawImage','arc','moveTo','lineTo','strokeRect'].includes(call.method))){
    assert.deepEqual(call.transform,mirrored,call.method+' matches mirrored video');
  }
  const label=h.canvasCalls.find(call=>call.method==='fillText');
  assert.equal(label.args[0],'站立');
  assert.deepEqual(label.transform,{sx:1,sy:1,tx:0,ty:0});
  assert.equal(label.args[1],960-220+7,'label begins at visible box left, with its text inset');
  assert.equal(h.run('buildDiagnosticReport().client_config.display_mirror'),true);
});

function idlePhoneHarness(devices) {
  const h=harness({record:false});h.media.devices=devices;
  h.run("active=false;stream=null;$('camera').value='mac';controls();");
  return h;
}
const macCamera={kind:'videoinput',deviceId:'mac',label:'MacBook Air 相机'};

test('detected phones appear directly in the camera selector without a shortcut button', async () => {
  const h=idlePhoneHarness([macCamera,{kind:'videoinput',deviceId:'phone',label:'UniZhe 的 iPhone 相机'}]);
  await h.run('refreshCameras(false)');
  assert.deepEqual(h.nodes.get('camera').options.map(option=>option.value),['','mac','phone']);
  assert.match(h.nodes.get('camera').options[2].textContent,/iPhone/);
  assert.equal(h.nodes.get('camera').value,'mac','enumeration does not switch the input');
  assert.equal(h.media.permissionCalls,0,'labeled devices need no permission stream');
  assert.equal(h.run('active'),false);assert.equal(h.run('stream'),null);
  assert.doesNotMatch(html,/id="phone-camera"/);
  assert.doesNotMatch(script,/usePhoneCamera|choosingPhone/);
});

test('refresh preserves the explicitly selected phone among multiple cameras', async () => {
  const h=idlePhoneHarness([macCamera,
    {kind:'videoinput',deviceId:'one',label:'iPhone camera'},
    {kind:'videoinput',deviceId:'two',label:'DroidCam camera'},
  ]);
  h.run("$('camera').value='two'");await h.run('refreshCameras(false)');
  assert.equal(h.nodes.get('camera').value,'two');
  assert.match(h.nodes.get('camera-status').textContent,/DroidCam/);
  h.media.devices=[macCamera];await h.run('refreshCameras(false)');
  assert.equal(h.nodes.get('camera').value,'','disconnected device cannot remain a fake option');
  assert.equal(h.nodes.get('camera').options.length,2);
});

test('camera dropdown and refresh stay locked during preview, recording, startup or shutdown', () => {
  for(const mode of ['preview','recording','starting','stopping']){
    const h=idlePhoneHarness([macCamera,{kind:'videoinput',deviceId:'phone',label:'iPhone'}]);
    h.context.mode=mode;
    h.run("active=['preview','recording'].includes(mode);recording=mode==='recording'?fakeRecording:null;starting=mode==='starting';stopping=mode==='stopping';controls();");
    assert.equal(h.nodes.get('camera').disabled,true,mode);
    assert.equal(h.nodes.get('refresh').disabled,true,mode);
    assert.equal(h.nodes.get('camera').value,'mac');
  }
});

test('measured skeleton colors and joints match the actual v18 renderer, with no new completeness threshold', t => {
  // Historical experiment snapshots are local-only. Missing history skips only
  // this comparison; current renderer, recording, and lifecycle tests still run.
  let v18Html;
  try {
    v18Html=fs.readFileSync(path.resolve(__dirname,'../runs/low-fall-fix-20261027/after-source/assets/webcam/index.html'),'utf8');
  } catch(error) {
    if(error.code!=='ENOENT')throw error;
    t.skip('Optional v18 experiment snapshot is not distributed with the source repository');
    return;
  }
  const v18Production=v18Html.match(/<script>([\s\S]*?)<\/script>/)[1].split("$('source-type').addEventListener")[0];
  const payloads=[];
  for(const target of [null,0,1]){
    const payload=result();payload.persons=[measuredPerson(),{...measuredPerson(),box:[300,50,500,430],
      keypoints:measuredPerson().keypoints.map(k=>[k[0]+300,k[1],k[2]])}];
    payload.action.target_index=target;payload.action.label='standing';payload.action.text='站立';
    payloads.push(payload);
  }
  // A fully measured candidate is still gray until the existing backend selects it.
  const complete=result();complete.persons=[{box:[20,50,220,430],keypoints:Array.from({length:17},(_,i)=>[80+i,100+i,.99])}];
  payloads.push(complete);
  // Preserve the original independent fall-target colors as well as shared targets.
  for(const state of ['unobservable','normal','suspected','confirmed'])for(const shared of [true,false]){
    const payload=result();const p=measuredPerson();payload.persons=[p];payload.action.target_index=shared?0:null;
    payload.fall={state,observed:state!=='unobservable',target_epoch:9,event_target_epoch:9};
    payload.fall_observation={source:'raw',person:p,selection:{raw_index:0,target_epoch:9}};
    payloads.push(payload);
  }
  for(const payload of payloads){
    const old=harness({source:v18Production}),current=harness();
    for(const h of [old,current]){
      h.context.payload=payload;h.run("$('mirror').checked=true;draw({width:960,height:540},payload)");
    }
    const skeleton=h=>h.canvasCalls.filter(call=>['arc','moveTo','lineTo'].includes(call.method));
    assert.deepEqual(skeleton(current),skeleton(old),JSON.stringify({target:payload.action.target_index,fall:payload.fall.state}));
    assert(skeleton(current).length>0);
  }
});

test('compact labels cannot turn unknown actions into a known auxiliary posture or mutate backend results', () => {
  const h=harness();const payload=result();payload.persons=[measuredPerson()];payload.action.target_index=0;
  payload.behavior={target_index:0,posture:{label:'upright',text:'直立'},locomotion:{label:'unknown',text:'未知'}};
  h.context.payload=payload;const before=JSON.stringify(payload);
  h.run('draw({width:960,height:540},payload)');
  assert.equal(h.canvasCalls.find(call=>call.method==='fillText').args[0],'动作待确认');
  assert.equal(JSON.stringify(payload),before);
});

test('ordinary transition points are neutral records and never replace action or fall state', () => {
  const h=harness();h.context.payload=result();
  const point={event_id:'transition-000001',kind:'sit_to_stand_started',text:'已确认坐姿后开始起身（普通动作转换）',
    category:'activity_transition',occurred_at_s:12,detected_at_s:12,target_epoch:7,
    pre_reminder_candidate:true,post_fall_evidence:false,prediction_validated:false,
    evidence:{anchor_posture:'seated',anchor_since_s:11,current_action:'standing_up',interpretation:'普通动作'}};
  h.context.payload.risk_points={schema_version:'risk-transition-points-v1',observer_only:true,reminders_enabled:false,
    source_time_s:12,activity_observed:true,event_count:1,new_events:[point],recent_events:[point],
    unimplemented:['walking_or_turning_instability']};
  h.run('renderResult(payload,12,40);');
  assert.match(h.nodes.get('risk-point-current').textContent,/本帧新增.*普通动作转换/);
  assert.equal(h.nodes.get('risk-point-count').textContent,'1 个观察点');
  assert.equal(h.nodes.get('action').textContent,'未知');
  assert.match(h.nodes.get('fall-state').textContent,/无法判断/);
  assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
  const row=h.nodes.get('risk-point-history').children[0];
  assert.match(row.children[0].textContent,/普通动作转换/);
  assert.match(row.children[1].textContent,/发生 00:12.0.*输出 00:12.0/);
  const report=h.run('buildDiagnosticReport()');
  assert.equal(report.samples[0].risk_points.reminders_enabled,false);
  assert.equal(report.samples[0].risk_points.new_events[0].prediction_validated,false);
  assert.equal(report.samples[0].risk_points.recent_events,undefined,'samples export newly emitted points once');
  h.run('resetResults();');
  assert.equal(h.nodes.get('risk-point-history').children.length,0);
  assert.equal(h.nodes.get('risk-point-count').textContent,'0 个观察点');
});

test('historical transition points remain history and cannot appear as new on a later frame', () => {
  const h=harness();h.context.payload=result();
  const event={event_id:'transition-000002',kind:'fall_confirmed',text:'红色：当前跌倒事件确认',category:'fall_confirmation',
    occurred_at_s:10,detected_at_s:10,target_epoch:'rgb-1',post_fall_evidence:true};
  h.context.payload.risk_points={event_count:1,new_events:[event],recent_events:[event]};
  h.run('renderResult(payload,12,40);');
  assert.equal(h.nodes.get('risk-point-current').textContent,'本帧无新增转换点');
  assert.equal(h.nodes.get('risk-point-history').children.length,1);
  assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
  delete h.context.payload.risk_points;
  h.run('renderResult(payload,12.13,40);');
  assert.match(h.nodes.get('risk-point-current').textContent,/未提供/);
  assert.equal(h.nodes.get('risk-point-history').children.length,0);
});

test('transition and new RGB evidence exports whitelist scalars and omit all coordinate arrays', () => {
  const h=harness();h.context.payload=result();
  const witness={t_s:12,target_epoch:'rgb-2',source:'person_detector',reference_kind:'seated',
    center_drop_norm:.4,top_drop_norm:.5,height_ratio:.7,handoff_at_s:12.2,end_box:[1,2,3,4],keypoints:[[1,2,3]],secret:'omit'};
  const event={event_id:'transition-000003',kind:'body_descent_observed',text:'身体下降',category:'fall_evidence',
    occurred_at_s:12,detected_at_s:12,target_epoch:'rgb-2',evidence:{witness,keypoints:[[1,2,3]],private_value:'omit'}};
  h.context.payload.risk_points={schema_version:'risk-transition-points-v1',observer_only:true,reminders_enabled:false,
    event_count:1,new_events:[event],recent_events:[event],unimplemented:['support_loss',{private:'omit'}],pixels:'omit'};
  h.context.payload.rgb_diagnostics={reference_kind:'seated',reference_ready:true,measured_descent:witness,
    descent_witness:witness,measured_descent_ready:true,current_low_pose:true,delayed_handoff_ready:true,
    person_ambiguity_retained:true,person_candidate_count:1,support_pending_s:.4,
    pose_geometry:{tilt:55,seated:false,torso_confidence:.8,hip:[1,2],shoulder:[3,4],feet:[[5,6]]},
    person_parts_removed:[{index:1,outer_index:0,containment:.99,area_ratio:.3,box:[1,2,3,4]}]};
  h.context.payload.fall.transitions=[{kind:'rearmed',t_s:12,target_epoch:'rgb-3',previous_event_id:'rgb-fall-1',reference_kind:'seated'}];
  h.run('updateDiagnostics(payload,12);');
  const sample=h.run('buildDiagnosticReport().samples[0]');
  assert.equal(sample.rgb_diagnostics.measured_descent.reference_kind,'seated');
  assert.equal(sample.rgb_diagnostics.pose_geometry.tilt,55);
  assert.equal(sample.rgb_diagnostics.person_parts_removed[0].containment,.99);
  assert.equal(sample.fall.transitions[0].previous_event_id,'rgb-fall-1');
  assert.equal(sample.risk_points.new_events[0].evidence.witness.handoff_at_s,12.2);
  assert.equal(sample.risk_points.unimplemented.length,1);
  const exported=JSON.stringify([sample.risk_points,sample.rgb_diagnostics]);
  for(const forbidden of ['end_box','keypoints','secret','private_value','pixels','"hip"','"shoulder"','"feet"','"box"'])
    assert(!exported.includes(forbidden),`coordinate or unapproved field not exported: ${forbidden}`);
});


function rgbLifecyclePayload() {
  const payload=result(),person={box:[100,200,430,500],confidence:.9,keypoints:[]};
  payload.fall_mode='rgb';
  payload.fall_observation={source:'rgb_fall',person,selection:{target_epoch:'rgb-2',state:'locked'}};
  payload.fall={state:'confirmed',text:'低位倒卧事件已确认',observed:true,target_epoch:'rgb-2',event_target_epoch:'rgb-2',
    event_id:'rgb-fall-2',current_event_id:'rgb-fall-2',last_confirmed_event_id:'rgb-fall-2',confirmed_at_s:12,
    new_event:true,lifecycle:'new_confirmed',event_type:'low_support_to_lying'};
  return payload;
}

test('new low-lying confirmation and continued historical event use distinct labels without duplicate counts', () => {
  const h=harness();h.context.payload=rgbLifecyclePayload();
  h.run('renderResult(payload,12,40);');
  assert.equal(h.nodes.get('fall-state').textContent,'低位倒卧事件已确认');
  assert.equal(h.nodes.get('fall-lifecycle').textContent,'事件阶段：本帧新确认');
  assert.equal(h.nodes.get('fall-event-ids').textContent,'当前事件：rgb-fall-2 · 历史最近确认：rgb-fall-2');
  assert.equal(h.nodes.get('events').children.length,1);
  assert.match(h.nodes.get('events').children[0].children[0].textContent,/低位倒卧确认（原因待核对）/);
  assert.match(h.run("frameOverlays(payload).find(x=>x.role==='fall').label"),/低位倒卧事件已确认.*本帧新确认/);
  h.context.payload.fall.new_event=false;h.context.payload.fall.lifecycle='ongoing_confirmed';
  h.run('renderResult(payload,12.15,40);');
  assert.equal(h.nodes.get('fall-lifecycle').textContent,'事件阶段：既有事件持续');
  assert.equal(h.nodes.get('events').children.length,1,'repeated red is the same event');
  const report=h.run('buildDiagnosticReport()');
  assert.equal(report.events.length,1);assert.equal(report.events[0].event_type,'low_support_to_lying');
  assert.equal(report.samples[1].fall.new_event,false);
});

test('low posture reference keeps old event in history without displaying a current red confirmation', () => {
  const h=harness();h.context.payload=rgbLifecyclePayload();
  Object.assign(h.context.payload.fall,{state:'unobservable',text:'暂不可判断',new_event:false,lifecycle:'low_posture_ready',
    current_event_id:null,target_epoch:'rgb-3',event_target_epoch:'rgb-2',event_type:null});
  h.context.payload.fall_observation.selection.target_epoch='rgb-3';
  h.run('renderResult(payload,13,40);');
  assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
  assert.equal(h.nodes.get('fall-state').textContent,'低位参考待观察');
  assert.match(h.nodes.get('fall-lifecycle').textContent,/不代表安全恢复/);
  assert.equal(h.nodes.get('fall-event-ids').textContent,'当前事件：— · 历史最近确认：rgb-fall-2');
  const row=h.run("frameOverlays(payload).find(x=>x.role==='fall')");
  assert.notEqual(row.color,'#ff8f91');assert.match(row.label,/低位参考待观察/);
  h.context.payload.fall_observation={source:'none',person:null,selection:{}};
  h.context.payload.fall.observed=false;
  h.run('renderResult(payload,13.15,40);');
  assert.match(h.nodes.get('fall-state').textContent,/无法判断/);
  assert(!h.nodes.get('fall-lifecycle').textContent.includes('低位参考待观察'));
});

test('stale confirmation cannot become a current red or a new confirmation when its target is unobserved', () => {
  const h=harness();h.context.payload=rgbLifecyclePayload();
  h.context.payload.fall_observation={source:'none',person:null,selection:{}};
  h.run('renderResult(payload,12,40);');
  assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
  assert.match(h.nodes.get('fall-lifecycle').textContent,/本帧无新确认/);
  assert.equal(h.nodes.get('fall-event-ids').textContent,'当前事件：— · 历史最近确认：rgb-fall-2');
  assert.equal(h.run("frameOverlays(payload).filter(x=>x.role==='fall').length"),0);
});

test('v18 event and low-transition diagnostics export scalar lifecycle evidence without coordinates', () => {
  const h=harness();h.context.payload=rgbLifecyclePayload();
  h.context.payload.fall.transitions=[{kind:'rearmed',t_s:12,target_epoch:'rgb-2',recovery_scope:'low_posture_separation',
    previous_event_id:'rgb-fall-1',reference_kind:'low_support',event_type:'low_support_to_lying'}];
  h.context.payload.rgb_diagnostics={reference_kind:'low_support',low_transition:{reference_age_s:.8,shoulder_drop_torso:.9,
    recent_hip_drop_torso:.3,qualifying:true,reference_expires_s:2.5,box:[1,2,3,4]},
    descent_witness:{reference_at_s:11,shoulder_drop_torso:.9,recent_hip_drop_torso:.3,shoulder:[1,2],hip:[3,4]}};
  h.context.payload.risk_points={new_events:[{event_id:'transition-2',kind:'fall_rearmed',occurred_at_s:12,detected_at_s:12,
    evidence:{recovery_scope:'low_posture_separation',reference_kind:'low_support',event_type:'low_support_to_lying',box:[1,2,3,4]}}]};
  h.run('updateDiagnostics(payload,12);');
  const sample=h.run('buildDiagnosticReport().samples[0]');
  assert.equal(sample.fall.current_event_id,'rgb-fall-2');assert.equal(sample.fall.last_confirmed_event_id,'rgb-fall-2');
  assert.equal(sample.fall.new_event,true);assert.equal(sample.fall.lifecycle,'new_confirmed');
  assert.equal(sample.fall.transitions[0].recovery_scope,'low_posture_separation');
  assert.equal(sample.rgb_diagnostics.low_transition.reference_expires_s,2.5);
  assert.equal(sample.rgb_diagnostics.low_transition.qualifying,true);
  assert.equal(sample.rgb_diagnostics.descent_witness.reference_at_s,11);
  assert.equal(sample.risk_points.new_events[0].evidence.reference_kind,'low_support');
  assert.equal(sample.risk_points.new_events[0].evidence.event_type,'low_support_to_lying');
  const exported=JSON.stringify([sample.fall,sample.rgb_diagnostics,sample.risk_points]);
  for(const field of ['"box"','"shoulder"','"hip"'])assert(!exported.includes(field));
});

test('v19 lying overlay is current-only and cannot create a red event', () => {
 const h=harness();h.context.payload=result();
 h.context.payload.source_time=12;
 h.context.payload.persons=[{box:[10,10,200,90],confidence:.95,keypoints:[]}];
 h.context.payload.lying_observation={label:'lying',observed:true,target_index:0,source_time:12,new_event:false,event_id:null};
 assert.equal(h.run('visibleLying(payload)'),true);
 assert.equal(h.run("frameOverlays(payload).filter(r=>r.role==='lying').length"),1);
 assert.equal(h.run('currentFallState(payload)'),'unobservable');
 h.context.payload.lying_observation.source_time=11;
 assert.equal(h.run('visibleLying(payload)'),false);
 assert.equal(h.run("frameOverlays(payload).filter(r=>r.role==='lying').length"),0);
});

test('v19 camera-blocked reason and frontal reference are actionable and exported without joints', () => {
 const h=harness();h.context.payload=result();
 h.context.payload.live_setup={camera_stable:false,camera_reason:'检测到相机移动',person_count:1,complete_sides:2,fall_reason:'相机不稳定',sampling_slow:false,keypoints:[[1,2,3]]};
 h.context.payload.action.view_bend={ready:true,confirmed:false,code:'reference_ready',compression:.9,reference:[[1,2,3]]};
 h.run('renderResult(payload,12,40);');
 assert.match(h.nodes.get('live-fall-check').textContent,/重新建立参考/);
 assert.match(h.nodes.get('live-bend-check').textContent,/请先正面站稳/);
 const sample=h.run('buildDiagnosticReport().samples[0]');
 assert.equal(sample.live_setup.camera_stable,false);
 assert.equal(sample.view_bend.compression,.9);
 assert(!JSON.stringify(sample.live_setup).includes('keypoints'));
 assert(!JSON.stringify(sample.view_bend).includes('[[1,2,3]]'));
});

test('v19 reference restart is disabled during recordings and video replay', async () => {
 const h=harness();h.run('controls();');
 assert.equal(h.nodes.get('reset-view').disabled,true);
 await h.run('resetView();');
 assert.equal(h.fetches.length,0);
 h.run("recording=null;$('source-type').value='video';controls();");
 assert.equal(h.nodes.get('reset-view').disabled,true);
 await h.run('resetView();');
 assert.equal(h.fetches.length,0);
});

function decisionContext(overrides={}) {
 return {schema_version:1,source_time_s:12,pose_observation_status:'observable',
  event_identity_status:'current',fall_evidence_status:'unresolved',text:'跌倒过程待确认',
  reason_code:'classification_geometry_conflict',event_state:'unobservable',changes_alarm:false,
  establishes_safety:false,establishes_recovery:false,...overrides};
}

test('current visible unresolved evidence changes explanation without changing event state or colors', () => {
 const h=harness();h.context.payload=result();h.context.payload.source_time=12;
 h.context.payload.decision_context=decisionContext();
 const before=JSON.stringify(h.context.payload);
 h.run('renderResult(payload,12,40)');
 assert.equal(h.nodes.get('fall-state').textContent,'跌倒过程待确认');
 assert.match(h.nodes.get('decision-context-summary').textContent,/骨架可见.*跌倒过程待确认/);
 assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
 assert.equal(h.run('currentFallState(payload)'),'unobservable');
 assert.equal(h.run('diagnosticState.events.length'),0);
 assert.equal(JSON.stringify(h.context.payload),before);
 h.run('resetResults()');
 assert.equal(h.nodes.get('decision-context-summary').textContent,'观测与跌倒证据：等待观测');
});

test('camera and missing observation remain unavailable while identity ambiguity stays explicit', () => {
 for(const [reason,pose,identity,text] of [
  ['camera_quality_unavailable','unavailable','not_established','画面稳定性不足'],
  ['no_strong_candidate','unavailable','not_established','当前骨架观测不足'],
  ['multiple_person_candidates','observable','ambiguous','目标关联有歧义，跌倒证据不可用']]){
  const h=harness();h.context.payload=result();h.context.payload.source_time=12;
  h.context.payload.decision_context=decisionContext({reason_code:reason,pose_observation_status:pose,
   event_identity_status:identity,fall_evidence_status:'unavailable',text});
  h.run('renderResult(payload,12,40)');
  assert.equal(h.nodes.get('fall-state').textContent,text);
  assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
  assert.equal(h.run('diagnosticState.events.length'),0);
  if(identity==='ambiguous')assert.match(h.nodes.get('decision-context-summary').textContent,/目标关联有歧义/);
 }
});

test('stale future malformed or unclocked semantic context cannot override old display behavior', () => {
 for(const change of [{source_time_s:11},{source_time_s:13},{schema_version:2},
  {fall_evidence_status:'safe'},{pose_observation_status:'safe'},{event_identity_status:'guessed'}]){
  const h=harness();h.context.payload=result();h.context.payload.source_time=12;
  h.context.payload.decision_context=decisionContext(change);
  h.run('renderResult(payload,12,40)');
  assert.equal(h.nodes.get('fall-state').textContent,'无法判断');
  assert.match(h.nodes.get('decision-context-summary').textContent,/未提供分项状态/);
 }
 const h=harness();h.context.payload=result();h.context.payload.decision_context=decisionContext();
 h.run('renderResult(payload,12,40)');assert.equal(h.nodes.get('fall-state').textContent,'无法判断');
});

test('semantic explanations cannot downgrade confirmed events or bypass stale event identity', () => {
 const h=harness();h.context.payload=rgbLifecyclePayload();h.context.payload.source_time=12;
 h.context.payload.decision_context=decisionContext();
 h.run('renderResult(payload,12,40)');
 assert.equal(h.nodes.get('fall-card').dataset.state,'confirmed');
 assert.notEqual(h.nodes.get('fall-state').textContent,'跌倒过程待确认');
 h.context.payload.fall_observation={source:'none',person:null,selection:{}};
 h.context.payload.decision_context=decisionContext({fall_evidence_status:'confirmed',text:'事件已确认'});
 h.run('renderResult(payload,12,40)');
 assert.equal(h.nodes.get('fall-card').dataset.state,'unobservable');
 assert.equal(h.nodes.get('fall-state').textContent,'无法判断');
 assert.match(h.nodes.get('decision-context-summary').textContent,/未提供分项状态/);
});

test('diagnostic export retains split semantic scalars without image coordinates', () => {
 const h=harness();h.context.payload=result();h.context.payload.source_time=12;
 h.context.payload.decision_context=decisionContext({box:[1,2,3,4],keypoints:[[1,2,3]]});
 h.context.payload.tool_event={schema_version:1,tool:'fall_detection',state:'unobservable',
  decision_context:h.context.payload.decision_context};
 h.run('updateDiagnostics(payload,12)');
 const sample=h.run('buildDiagnosticReport().samples[0]');
 assert.equal(sample.decision_context.pose_observation_status,'observable');
 assert.equal(sample.decision_context.fall_evidence_status,'unresolved');
 assert.equal(sample.decision_context.source_time_s,12);
 assert.equal(sample.decision_context.changes_alarm,false);
 assert.equal(sample.decision_context.establishes_safety,false);
 assert.equal(sample.decision_context.establishes_recovery,false);
 assert.equal(sample.tool_event.decision_context.fall_evidence_status,'unresolved');
 assert.equal(sample.decision_context.box,undefined);assert.equal(sample.decision_context.keypoints,undefined);
 assert.equal(sample.fall.state,'unobservable');
});
