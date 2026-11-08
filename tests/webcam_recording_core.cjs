'use strict';

// Run: node --test tests/webcam_recording_core.cjs
const test = require('node:test');
const assert = require('node:assert/strict');
const {chooseMimeType, videoExtension, RawRecordingSession, makeStoredZip} = require('../assets/webcam/recording.js');

function fakeRecorder({supported = ['video/webm;codecs=vp8'], final = 'tail', brokenStop = false, startError = false} = {}) {
  return class FakeRecorder extends EventTarget {
    static instances = [];
    static isTypeSupported(type) { return supported.includes(type); }
    constructor(stream, options) {
      super(); this.stream = stream; this.options = options;
      this.mimeType = options && options.mimeType || 'video/webm';
      this.state = 'inactive'; this.stopCalls = 0;
      this.constructor.instances.push(this);
    }
    emit(name, properties = {}) { this.dispatchEvent(Object.assign(new Event(name), properties)); }
    chunk(text, timecode) { this.emit('dataavailable', {data: new Blob([text], {type: this.mimeType}), timecode}); }
    start(timeslice) {
      if (startError) throw new DOMException('no recorder', 'NotSupportedError');
      this.timeslice = timeslice; this.state = 'recording';
      queueMicrotask(() => this.emit('start'));
    }
    stop() {
      this.stopCalls++; this.state = 'inactive';
      if (!brokenStop) setTimeout(() => {
        this.chunk(final, 1743.5);
        setTimeout(() => this.emit('stop'), 0);
      }, 0);
    }
  };
}

function fixture(Recorder = fakeRecorder(), options = {}) {
  const track = {stops: 0, stop() { this.stops++; },
    getSettings() { return {width: 1280, height: 720, frameRate: 30, deviceId: 'private-device'}; }};
  const stream = {getVideoTracks: () => [track], getAudioTracks: () => []};
  const video = {currentTime: 12.25};
  let now = 1000;
  const recording = new RawRecordingSession({stream, video, id: 'sample-01', basename: 'sample-01',
    clock: () => now, MediaRecorderClass: Recorder, ...options});
  return {recording, video, stream, track, Recorder, setNow(value) { now = value; }};
}

test('MIME choice follows tested support, including Safari MP4 and browser default', () => {
  assert.equal(chooseMimeType(fakeRecorder()), 'video/webm;codecs=vp8');
  assert.equal(chooseMimeType(fakeRecorder({supported: ['video/mp4;codecs=avc1.42E01E', 'video/webm']})), 'video/mp4;codecs=avc1.42E01E');
  assert.equal(chooseMimeType(fakeRecorder({supported: ['video/mp4']})), 'video/mp4');
  assert.equal(chooseMimeType(fakeRecorder({supported: []})), '');
  assert.equal(chooseMimeType({isTypeSupported() { throw Error('Unsupported query'); }}), '');
  assert.equal(chooseMimeType({}), '');
  assert.equal(videoExtension('video/mp4;codecs=avc1.42E01E'), 'mp4');
  assert.equal(videoExtension('video/webm;codecs=vp8'), 'webm');
  assert.equal(videoExtension('video/custom'), 'video');
});

test('original stream is recorded directly, with observational timing and no camera shutdown', async () => {
  const f = fixture();
  const initial = f.recording.start();
  assert.equal(initial.anchors.requested.performance_ms, 1000);
  assert.equal(initial.anchors.requested.source_video_time_s, 12.25);
  f.setNow(1020); f.video.currentTime = 12.27;
  await Promise.resolve();
  const recorder = f.Recorder.instances[0];
  assert.equal(recorder.stream, f.stream);
  assert.equal(recorder.timeslice, 1000);
  recorder.chunk('first', 99);
  recorder.chunk('', undefined);
  f.setNow(2080); f.video.currentTime = 13.33;
  const result = await f.recording.stop('user');
  assert.equal(await result.blob.text(), 'firsttail');
  assert.equal(result.metadata.elapsed_ms, 1060);
  assert.equal(result.metadata.chunk_count, 2);
  assert.equal(result.metadata.data_event_count, 3);
  assert.equal(result.metadata.chunks[1].timecode_ms, null);
  assert.equal(result.metadata.received_stop_event, true);
  assert.equal(result.metadata.partial, false);
  assert.equal(result.metadata.audio, false);
  assert.equal(result.metadata.overlay, false);
  assert.equal(result.metadata.video_settings.frame_rate, 30);
  assert.equal(JSON.stringify(result.metadata).includes('private-device'), false);
  assert.equal(f.track.stops, 0);
  assert.match(result.metadata.timing_note, /not encoded frame timestamps/);
});

test('stop is idempotent and waits for the asynchronous final chunk followed by stop', async () => {
  let callbackCount = 0;
  const f = fixture(fakeRecorder(), {onStop() { callbackCount++; }});
  f.recording.start(); await Promise.resolve();
  let settled = false;
  const pending = f.recording.stop('user'); pending.then(() => { settled = true; });
  assert.equal(pending, f.recording.stop('duplicate'));
  await Promise.resolve();
  assert.equal(settled, false);
  const result = await pending;
  assert.equal(await result.blob.text(), 'tail');
  assert.equal(f.Recorder.instances[0].stopCalls, 1);
  assert.equal(callbackCount, 1);
  assert.equal(result.metadata.stop_reason, 'user');
  assert.equal(await f.recording.stop(), result);
  assert.throws(() => f.recording.start(), /已经启动/);
});

test('error keeps previous and final chunks, marks partial, and awaits stop', async () => {
  const f = fixture(); f.recording.start(); await Promise.resolve();
  const recorder = f.Recorder.instances[0];
  recorder.chunk('before-');
  recorder.emit('error', {error: new DOMException('encoder failed', 'EncodingError')});
  const result = await f.recording.stop();
  assert.equal(await result.blob.text(), 'before-tail');
  assert.equal(result.metadata.partial, true);
  assert.equal(result.metadata.status, 'error');
  assert.equal(result.metadata.errors[0].name, 'EncodingError');
  assert.equal(result.metadata.received_stop_event, true);
  assert.equal(result.metadata.stop_reason, 'recorder_error');
});

test('missing stop event times out with a stable partial result instead of hanging', async () => {
  const f = fixture(fakeRecorder({brokenStop: true}), {finalizationTimeoutMs: 10});
  f.recording.start(); await Promise.resolve();
  const recorder = f.Recorder.instances[0]; recorder.chunk('saved');
  const result = await f.recording.stop();
  assert.equal(await result.blob.text(), 'saved');
  assert.equal(result.metadata.partial, true);
  assert.equal(result.metadata.received_stop_event, false);
  assert.equal(result.metadata.errors[0].stage, 'finalization_timeout');
  recorder.chunk('late'); recorder.emit('stop');
  assert.equal(f.recording.getMetadata().byte_length, 5);
  assert.equal(await f.recording.stop(), result);
});

test('zero-byte start failures preserve metadata and return an exportable result', async () => {
  const f = fixture(fakeRecorder({startError: true}));
  assert.throws(() => f.recording.start(), /no recorder/);
  const result = await f.recording.stop();
  assert.equal(result.blob.size, 0);
  assert.equal(result.metadata.partial, true);
  assert.equal(result.metadata.errors[0].stage, 'start');
  assert.equal(f.track.stops, 0);
});

test('empty data events do not produce fake bytes, and absence of supported types uses actual browser MIME', async () => {
  const f = fixture(fakeRecorder({supported: [], final: ''}));
  f.recording.start(); await Promise.resolve();
  assert.equal(f.Recorder.instances[0].options, undefined);
  const result = await f.recording.stop();
  assert.equal(result.blob.size, 0);
  assert.equal(result.metadata.chunk_count, 0);
  assert.equal(result.metadata.data_event_count, 1);
  assert.equal(result.metadata.mime_type, 'video/webm');
});

test('recording requires a video-only stream', () => {
  assert.throws(() => new RawRecordingSession({stream: {getVideoTracks: () => []}}), /视频流/);
  assert.throws(() => new RawRecordingSession({stream: {getVideoTracks: () => [{}], getAudioTracks: () => [{}]}}), /无音轨/);
});

function parseStoredZip(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const eocd = bytes.length - 22;
  assert.equal(view.getUint32(eocd, true), 0x06054b50);
  const count = view.getUint16(eocd + 10, true), directorySize = view.getUint32(eocd + 12, true);
  const directoryStart = view.getUint32(eocd + 16, true);
  assert.equal(directoryStart + directorySize, eocd);
  let cursor = directoryStart;
  const entries = [];
  for (let i = 0; i < count; i++) {
    assert.equal(view.getUint32(cursor, true), 0x02014b50);
    assert.equal(view.getUint16(cursor + 10, true), 0); // stored, no compression
    const nameSize = view.getUint16(cursor + 28, true), local = view.getUint32(cursor + 42, true);
    const name = new TextDecoder().decode(bytes.slice(cursor + 46, cursor + 46 + nameSize));
    const size = view.getUint32(cursor + 24, true), checksum = view.getUint32(cursor + 16, true);
    assert.equal(view.getUint32(local, true), 0x04034b50);
    assert.equal(view.getUint32(local + 14, true), checksum);
    assert.equal(view.getUint32(local + 22, true), size);
    assert.equal(view.getUint16(local + 6, true), 0x0800);
    assert.equal(view.getUint16(local + 26, true), nameSize);
    assert.equal(new TextDecoder().decode(bytes.slice(local + 30, local + 30 + nameSize)), name);
    const data = bytes.slice(local + 30 + nameSize, local + 30 + nameSize + size);
    // Separate bitwise reference implementation verifies content and CRC independently.
    let crc = 0xffffffff;
    for (const byte of data) {
      crc ^= byte;
      for (let bit = 0; bit < 8; bit++) crc = (crc & 1) ? ((crc >>> 1) ^ 0xedb88320) : (crc >>> 1);
    }
    assert.equal(checksum, (crc ^ 0xffffffff) >>> 0);
    entries.push({name, data, checksum}); cursor += 46 + nameSize;
  }
  assert.equal(cursor, eocd);
  return entries;
}

test('ZIP local headers, central directory, UTF-8 names, offsets, CRC and untouched video bytes agree', async () => {
  const zip = await makeStoredZip([
    {name: 'sample.webm', blob: new Blob(['123456789'])},
    {name: '诊断.json', blob: new Blob(['{"unknown":3}'], {type: 'application/json'})},
    {name: 'empty.txt', blob: new Blob([])},
  ]);
  assert.equal(zip.type, 'application/zip');
  const entries = parseStoredZip(new Uint8Array(await zip.arrayBuffer()));
  assert.deepEqual(entries.map(entry => entry.name), ['sample.webm', '诊断.json', 'empty.txt']);
  assert.equal(entries[0].checksum, 0xcbf43926);
  assert.equal(new TextDecoder().decode(entries[0].data), '123456789');
  assert.equal(JSON.parse(new TextDecoder().decode(entries[1].data)).unknown, 3);
  assert.equal(entries[2].data.length, 0);
  assert.equal(parseStoredZip(new Uint8Array(await (await makeStoredZip([])).arrayBuffer())).length, 0);
});

test('ZIP rejects ambiguous paths and duplicate names', async () => {
  for (const name of ['../video.webm', '/tmp/video.webm', 'a/../b', 'a\\b', 'bad\0name']) {
    await assert.rejects(makeStoredZip([{name, blob: new Blob(['x'])}]), /filename/);
  }
  await assert.rejects(makeStoredZip([{name: 'same', blob: new Blob([])}, {name: 'same', blob: new Blob([])}]), /duplicate/);
});
