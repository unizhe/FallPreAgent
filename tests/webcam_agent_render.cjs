const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'assets/webcam/index.html'), 'utf8');
const inline = html.match(/<script>\s*([\s\S]*?)<\/script>/)?.[1];
assert.ok(inline, 'production webcam script is present');
const statusBegin = inline.indexOf('function agentStatusText');
const begin = inline.indexOf('function probabilityKindLabel');
const end = inline.indexOf('function clearAgentDisplay', begin);
assert.ok(statusBegin >= 0 && begin > statusBegin && end > begin, 'production status and probability render functions are present');
const productionRenderers = inline.slice(statusBegin, begin) + inline.slice(begin, end);

function harness() {
  const nodes = new Map();
  const createNode = tag => ({
    tag, textContent: '', dataset: {}, children: [], hidden: false,
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; },
  });
  const context = vm.createContext({
    $: id => {
      if (!nodes.has(id)) nodes.set(id, createNode(id));
      return nodes.get(id);
    },
    document: {createElement: createNode},
    fmtTime: value => Number.isFinite(value) ? `t${value}` : '—',
  });
  vm.runInContext(`
    let activeAgentRun={run_id:'run-offline',provider:'off',model:'—',stages:{}};
    let lastRenderedResult={fall:{state:'normal'}};
    let agentLatestDecisionId=null;
    let agentFeedbackVersion=0;
    const videoReview={time:null};
    let latestAgentView=null,latestAgentError=null;
    ${productionRenderers}
  `, context);
  return {context, nodes, run: source => vm.runInContext(source, context)};
}

test('production renderers label synthetic and model probabilities with each result origin', () => {
  const h = harness();
  const latestBase = {
    run_id: 'run-render-probe', run_status: 'active',
    provider_status: {provider: 'aliyun', model: 'decision-model-preview', network_enabled: false,
      stages: {fall_head: {selected_variant: 'cd', variants: {}}}},
    projection: {provider_status: 'ready', current: {
      status: 'ready', current_applicable: true, fall_probability: 0.63,
      probability_kind: 'synthetic_fixture', provider: 'mock', model: 'mock-fixture-v1',
      decision_id: 'decision-mock', source_window: {start_s: 1.0, end_s: 3.0},
    }, latest_result: {status: 'ready'}, latest_task_status: 'completed', stale_reason_codes: []},
  };
  h.run(`activeAgentRun={run_id:'run-render-probe',status:'active',provider:'off',model:'—',stages:{}}`);
  h.context.renderAgentProjection(latestBase);
  const mockCurrent = {
    probability: h.nodes.get('agent-probability').textContent,
    kind: h.nodes.get('agent-probability-kind').textContent,
    provider: h.nodes.get('agent-provider').textContent,
  };
  assert.equal(mockCurrent.probability, '63.0%');
  assert.match(mockCurrent.kind, /合成测试概率/);
  assert.match(mockCurrent.provider, /mock \/ mock-fixture-v1/);
  assert.doesNotMatch(mockCurrent.kind, /未校准模型估计/);

  const aliyunLatest = structuredClone(latestBase);
  aliyunLatest.projection.current = {
    ...aliyunLatest.projection.current, fall_probability: 0.04,
    probability_kind: 'uncalibrated_model_estimate', provider: 'aliyun', model: 'decision-model-preview',
    decision_id: 'decision-aliyun',
  };
  h.context.renderAgentProjection(aliyunLatest);
  const aliyunCurrent = {
    probability: h.nodes.get('agent-probability').textContent,
    kind: h.nodes.get('agent-probability-kind').textContent,
    provider: h.nodes.get('agent-provider').textContent,
  };
  assert.equal(aliyunCurrent.probability, '4.0%');
  assert.match(aliyunCurrent.kind, /未校准模型估计/);
  assert.match(aliyunCurrent.provider, /aliyun \/ decision-model-preview/);

  h.context.renderAgentHistory([
    {cursor: 1, source_time: 3.6, attempt_history: [], result: {
      status: 'ready', fall_probability: 0.63, probability_kind: 'synthetic_fixture', provider: 'mock',
      model: 'mock-fixture-v1', source_window: {start_s: 1.0, end_s: 3.0},
      provider_metadata: {request_id: 'offline-mock-history', input_tokens: 0},
    }},
    {cursor: 2, source_time: 4.4, attempt_history: [], result: {
      status: 'ready', fall_probability: 0.04, probability_kind: 'uncalibrated_model_estimate', provider: 'aliyun',
      model: 'decision-model-preview', source_window: {start_s: 1.8, end_s: 4.3},
      provider_metadata: {request_id: 'offline-aliyun-history', input_tokens: 119},
    }},
  ]);
  const history = h.nodes.get('agent-history').children.map(row => ({
    title: row.children[0].textContent, detail: row.children[1].textContent,
  }));
  assert.match(history[0].title, /合成测试概率/);
  assert.match(history[0].detail, /provider mock/);
  assert.match(history[1].title, /未校准模型估计/);
  assert.match(history[1].detail, /provider aliyun/);
  assert.match(history[1].detail, /offline-aliyun-history/);
  assert.match(history[0].detail, /offline-mock-history/);

  if (process.env.FALLPRE_PAGE_PROBE_OUT) {
    fs.mkdirSync(path.dirname(process.env.FALLPRE_PAGE_PROBE_OUT), {recursive: true});
    fs.writeFileSync(process.env.FALLPRE_PAGE_PROBE_OUT, JSON.stringify({
      browser: 'not launched; production DOM render functions executed offline',
      current: {mock: mockCurrent, aliyun: aliyunCurrent},
      history,
      assertions: ['mock synthetic label', 'Aliyun uncalibrated label', 'per-row provider and kind'],
    }, null, 2) + '\n');
  }
});

test('production projection clears current probability for a non-ready result', () => {
  const h = harness();
  const latest = {
    run_status: 'active', provider_status: {provider: 'aliyun', model: 'decision-model-preview', network_enabled: false,
      stages: {}},
    projection: {current: {status: 'unavailable', current_applicable: false,
      fall_probability: 0.91, probability_kind: 'uncalibrated_model_estimate', provider: 'aliyun'},
      latest_result: {status: 'unavailable'}, latest_task_status: 'completed', stale_reason_codes: ['STREAM_STOPPED']},
  };
  h.context.renderAgentProjection(latest);
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  assert.equal(h.nodes.get('agent-probability-kind').textContent, '当前概率不可用');
  assert.equal(h.nodes.get('agent-window').textContent, '来源窗口：—');
});

test('current and historical distributions come from the provider answers and stopping clears the current display', () => {
  const h = harness();
  const result = {
    status: 'ready', current_applicable: true, provider: 'aliyun',
    probability_kind: 'uncalibrated_model_estimate', fall_probability: 0.7,
    source_window: {start_s: 15.1, end_s: 17.6},
    provider_metadata: {answers: {
      evidence_sufficiency: {probabilities: {sufficient: 0.7, insufficient: 0.3, conflicting: 0}},
      next_step: {probabilities: {emit_result: 0.9, observe_more: 0.1, request_review: 0}},
    }},
  };
  const latest = {
    run_status: 'active', provider_status: {provider: 'aliyun'},
    projection: {current_applicable: true, current: result},
  };
  const ids = ['probability', 'sufficient', 'insufficient', 'conflicting', 'emit_result', 'observe_more', 'request_review'];
  const displayed = () => ids.map(id => h.nodes.get('agent-' + id).textContent);
  h.context.renderAgentProjection(latest);
  assert.deepEqual(displayed(), ['70.0%', '70.0%', '30.0%', '0.0%', '90.0%', '10.0%', '0.0%']);

  h.context.renderAgentProjection({...latest, run_status: 'stopped'});
  assert.deepEqual(displayed(), ids.map(() => '—'));
  assert.match(h.nodes.get('agent-status').textContent, /已停止/);
  h.context.renderAgentProjection({...latest, run_status: 'stopped',
    projection: {current_applicable: false, historical_result: result}});
  assert.deepEqual(displayed(), ['70.0%', '70.0%', '30.0%', '0.0%', '90.0%', '10.0%', '0.0%']);
  assert.match(h.nodes.get('agent-result-scope').textContent, /历史/);
  assert.match(h.nodes.get('agent-window').textContent, /历史窗口/);
  h.context.renderAgentProjection({...latest, provider_status: {provider: 'off'}});
  assert.deepEqual(displayed(), ids.map(() => '—'));
  assert.match(h.nodes.get('agent-status').textContent, /云端决策未启用/);

  h.context.renderAgentHistory([{cursor: 1, result}]);
  const [row] = h.nodes.get('agent-history').children;
  assert.match(row.children[2].textContent, /充分 70.0% · 不足 30.0% · 冲突 0.0%/);
  assert.match(row.children[3].textContent, /输出结果 90.0% · 继续观察 10.0% · 请求复核 0.0%/);
});
