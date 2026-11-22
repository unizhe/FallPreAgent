'use strict';

// Exercise the production Agent polling and invalidation functions with a small DOM/network boundary.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.resolve(__dirname, '../assets/webcam/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const requestCode = script.slice(script.indexOf('async function agentRequest'), script.indexOf('\nfunction stopAgentPolling'));
const probabilitiesCode = script.slice(script.indexOf('const AGENT_CHOICE_OPTIONS'), script.indexOf('function renderAgentProjection'));
const start = script.indexOf('function clearAgentDisplay()');
const end = script.indexOf('\nasync function submitAgentFeedback', start);
const production = requestCode + probabilitiesCode + script.slice(start, end);

test('video waits for the submitted task when another task has already completed', async () => {
  const start = script.indexOf('async function waitForVideoDecision');
  const end = script.indexOf('\nfunction stop(', start);
  assert.ok(start >= 0 && end > start);
  const history = [{task_id: 'earlier-task'}];
  const waits = [];
  const context = vm.createContext({
    active: true, run: 7, agentHistoryCache: history, agentVideoCadenceMs: 250,
    activeAgentRun: {provider_status: {provider: 'aliyun', decision_timeout_s: 8}},
    performance: {now: () => 0}, setTimeout: callback => waits.push(callback),
  });
  vm.runInContext(script.slice(start, end), context);
  const completion = context.waitForVideoDecision(7, {
    task_id: 'new-task', trace: {temporal_status: 'ready', head_status: 'ready'},
  });
  assert.equal(waits.length, 1);
  history.push({task_id: 'new-task'});
  waits.shift()();
  await completion;
  assert.equal(waits.length, 0);
});

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return {promise, resolve, reject};
}

function harness() {
  const nodes = new Map();
  const context = vm.createContext({
    Promise, Error, AbortController, setTimeout, clearTimeout,
    fetch: async () => { throw Error('fetch was not stubbed'); },
    location: {origin: 'http://127.0.0.1'}, token: 'test-session-token',
    agentClientSessionId: 'test-client-session',
    $: id => {
      if (!nodes.has(id)) nodes.set(id, {textContent: '', dataset: {}, children: [], replaceChildren(...rows) { this.children = rows; }});
      return nodes.get(id);
    },
    activeAgentRun: {run_id: 'run-current'}, agentPollGeneration: 7, agentPollBusy: false,
    agentAfterId: 0, agentHistoryCache: [], agentLatestDecisionId: 'decision-old', agentFeedbackVersion: 0,
    videoReview: {time: null}, latestAgentView: null, latestAgentError: null,
    renderAgentProjection(latest) {
      nodes.get('agent-probability').textContent = `${latest.projection.fall_probability * 100}%`;
      nodes.get('agent-window').textContent = latest.projection.window;
    },
    renderAgentHistory(items) { context.agentHistoryCache.push(...items); },
    agentRequest: async () => { throw Error('agentRequest was not stubbed'); },
  });
  vm.runInContext(production, context);
  return {context, nodes, run: code => vm.runInContext(code, context)};
}

test('latest failure clears current probability/window and retains separately displayed history', async () => {
  const h = harness();
  h.run(`$('agent-panel').dataset.state='ready';$('agent-probability').textContent='72%';
    $('agent-window').textContent='来源窗口：00:01 — 00:03';$('agent-provider').textContent='provider / model：aliyun / model';
    $('agent-head').textContent='识别层：cd';$('agent-history').children=[{textContent:'已保存历史'}];`);
  h.context.boundedAgentRequest = async path => {
    if (path.endsWith('/latest')) throw Error('offline');
    return {items: [], next_after_id: 0};
  };
  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  assert.equal(h.nodes.get('agent-window').textContent, '来源窗口：—');
  assert.equal(h.nodes.get('agent-provider').textContent, 'provider / model：—');
  assert.equal(h.nodes.get('agent-panel').dataset.state, 'unavailable');
  assert.equal(h.nodes.get('agent-history').children[0].textContent, '已保存历史');
  assert.match(h.nodes.get('agent-status').textContent, /offline/);
});

test('latest and history errors are independent; a fresh projection can render while history stays visible', async () => {
  const h = harness();
  h.run(`$('agent-probability').textContent='72%';$('agent-window').textContent='old window';
    $('agent-history').children=[{textContent:'prior row'}];`);
  h.context.boundedAgentRequest = async path => {
    if (path.endsWith('/latest')) return {projection: {fall_probability: 0.31, window: 'new window'}};
    throw Error('history offline');
  };
  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '31%');
  assert.equal(h.nodes.get('agent-window').textContent, 'new window');
  assert.equal(h.nodes.get('agent-history').children[0].textContent, 'prior row');
  assert.match(h.nodes.get('agent-history-status').textContent, /历史查询暂不可用/);
});

test('ready projection clears on HTTP 500 and recovers only from a fresh latest response', async () => {
  const h = harness();
  h.run(`$('agent-panel').dataset.state='ready';$('agent-probability').textContent='72%';
    $('agent-window').textContent='来源窗口：old';$('agent-history').children=[{textContent:'saved history'}];`);
  let latestCalls = 0;
  h.context.fetch = async path => {
    if (path.endsWith('/latest')) {
      latestCalls++;
      if (latestCalls === 1) return {ok: false, status: 500, json: async () => ({error: 'injected HTTP 500'})};
      return {ok: true, json: async () => ({projection: {fall_probability: 0.22, window: 'fresh window'}})};
    }
    return {ok: true, json: async () => ({items: [], next_after_id: 0})};
  };
  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  assert.equal(h.nodes.get('agent-window').textContent, '来源窗口：—');
  assert.equal(h.nodes.get('agent-history').children[0].textContent, 'saved history');
  assert.match(h.nodes.get('agent-status').textContent, /injected HTTP 500/);

  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '22%');
  assert.equal(h.nodes.get('agent-window').textContent, 'fresh window');
});

test('ready projection clears after a hung latest query and recovers on the next fresh poll', async () => {
  const h = harness();
  h.run(`$('agent-panel').dataset.state='ready';$('agent-probability').textContent='72%';
    $('agent-window').textContent='来源窗口：old';`);
  h.context.setTimeout = callback => { callback(); return 1; };
  h.context.clearTimeout = () => {};
  h.context.agentRequest = path => path.endsWith('/latest')
    ? new Promise(() => {})
    : Promise.resolve({items: [], next_after_id: 0});
  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  assert.equal(h.nodes.get('agent-window').textContent, '来源窗口：—');
  assert.match(h.nodes.get('agent-status').textContent, /超过 2\.2 秒/);

  h.context.agentRequest = path => path.endsWith('/latest')
    ? Promise.resolve({projection: {fall_probability: 0.18, window: 'recovered window'}})
    : Promise.resolve({items: [], next_after_id: 0});
  await h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  assert.equal(h.nodes.get('agent-probability').textContent, '18%');
  assert.equal(h.nodes.get('agent-window').textContent, 'recovered window');
});

test('late responses from a prior run generation cannot restore current values', async () => {
  const h = harness(), latest = deferred(), history = deferred();
  h.run(`$('agent-probability').textContent='—';$('agent-window').textContent='来源窗口：—';`);
  h.context.boundedAgentRequest = path => path.endsWith('/latest') ? latest.promise : history.promise;
  const polling = h.run('pollAgentRun(activeAgentRun,agentPollGeneration)');
  h.run(`agentPollGeneration=8;activeAgentRun={run_id:'run-new'};`);
  latest.resolve({projection: {fall_probability: 0.99, window: 'late window'}});
  history.resolve({items: [{cursor: 1}], next_after_id: 1});
  await polling;
  assert.equal(h.nodes.get('agent-probability').textContent, '—');
  assert.equal(h.nodes.get('agent-window').textContent, '来源窗口：—');
  assert.equal(h.context.agentHistoryCache.length, 0);
});

test('bounded request timeout rejects a hung Agent query', async () => {
  const h = harness();
  h.context.agentRequest = () => new Promise(() => {});
  h.context.setTimeout = callback => { callback(); return 1; };
  h.context.clearTimeout = () => {};
  await assert.rejects(h.run("boundedAgentRequest('/latest')"), /2\.2 秒/);
});
