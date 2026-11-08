/* Raw camera recording and a dependency-free, uncompressed ZIP writer.
 * The camera stream is recorded directly; no canvas or overlay is captured.
 */
(function (root) {
  'use strict';

  const MIME_TYPES = [
    'video/webm;codecs=vp8',
    'video/mp4;codecs=avc1.42E01E',
    'video/webm',
    'video/mp4',
  ];
  const finite = value => Number.isFinite(value) ? value : null;

  function chooseMimeType(Recorder = root.MediaRecorder) {
    if (!Recorder || typeof Recorder.isTypeSupported !== 'function') return '';
    return MIME_TYPES.find(type => {
      try { return Recorder.isTypeSupported(type); } catch (_) { return false; }
    }) || '';
  }

  function videoExtension(mimeType) {
    if (/^video\/mp4(?:;|$)/i.test(mimeType || '')) return 'mp4';
    if (/^video\/webm(?:;|$)/i.test(mimeType || '')) return 'webm';
    if (/^video\/ogg(?:;|$)/i.test(mimeType || '')) return 'ogv';
    return 'video';
  }

  class RawRecordingSession {
    constructor({stream, video, id, basename, clock, onStatus, onStop, MediaRecorderClass, finalizationTimeoutMs = 5000} = {}) {
      if (!stream || !stream.getVideoTracks || !stream.getVideoTracks().length) {
        throw new Error('录像需要有效的摄像头视频流。');
      }
      if (stream.getAudioTracks && stream.getAudioTracks().length) {
        throw new Error('原始录像只接受无音轨的视频流。');
      }
      this.stream = stream;
      this.video = video;
      this.id = String(id || '');
      this.basename = String(basename || this.id || 'camera-recording');
      this.clock = clock || (() => root.performance.now());
      this.Recorder = MediaRecorderClass || root.MediaRecorder;
      this.onStatus = onStatus;
      this.onStop = onStop;
      this.status = 'idle';
      this.anchors = {requested: null, started: null, stop_requested: null, stopped: null};
      this.errors = [];
      this.chunks = [];
      this.chunkEvents = [];
      this.bytes = 0;
      this.recorder = null;
      this.mimeType = '';
      this.stopReason = null;
      this.finalResult = null;
      this.finishTimer = null;
      this.finalizationTimeoutMs = Math.max(1, Math.min(5000, finalizationTimeoutMs));
      this.done = new Promise(resolve => { this.resolveDone = resolve; });
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings ? track.getSettings() : {};
      // Exclude device/group identifiers from the exported metadata.
      this.videoSettings = {
        width: finite(settings.width), height: finite(settings.height),
        frame_rate: finite(settings.frameRate), facing_mode: settings.facingMode || null,
      };
    }

    _anchor() {
      return {
        performance_ms: finite(this.clock()),
        source_video_time_s: finite(this.video && this.video.currentTime),
        wall_time: new Date().toISOString(),
      };
    }

    _notify() {
      if (this.onStatus) {
        try { this.onStatus(this.getMetadata()); } catch (_) { /* UI cannot break finalization. */ }
      }
    }

    _error(error, stage) {
      this.errors.push({stage, name: String(error && error.name || 'Error'),
        message: String(error && error.message || error || 'Unknown recording error'), ...this._anchor()});
      this._notify();
    }

    start() {
      if (this.status !== 'idle') throw new Error('这段录像已经启动过。');
      if (!this.Recorder) throw new Error('当前浏览器不支持 MediaRecorder 录像。');
      this.anchors.requested = this._anchor();
      try {
        const mimeType = chooseMimeType(this.Recorder);
        this.recorder = mimeType ? new this.Recorder(this.stream, {mimeType}) : new this.Recorder(this.stream);
        this.mimeType = this.recorder.mimeType || mimeType;
        this.recorder.addEventListener('start', () => {
          if (this.finalResult) return;
          this.anchors.started = this._anchor();
          if (this.status === 'starting') this.status = 'recording';
          this._notify();
        });
        this.recorder.addEventListener('dataavailable', event => {
          if (this.finalResult) return;
          const data = event.data;
          const size = data && Number.isFinite(data.size) ? data.size : 0;
          this.chunkEvents.push({index: this.chunkEvents.length, byte_length: size,
            timecode_ms: finite(event.timecode), ...this._anchor()});
          if (size > 0) {
            this.chunks.push(data);
            this.bytes += size;
            if (!this.mimeType && data.type) this.mimeType = data.type;
          }
        });
        this.recorder.addEventListener('error', event => {
          if (this.finalResult) return;
          this.status = 'error';
          this.stopReason = this.stopReason || 'recorder_error';
          this._error(event.error || new Error('MediaRecorder error'), 'recording');
          // Browsers emit a final dataavailable and then stop after an error.
          // Never resolve here: doing so would lose that final chunk.
          this.stop('recorder_error');
        });
        this.recorder.addEventListener('stop', () => this._finish(true));
        this.status = 'starting';
        this.recorder.start(1000);
        this._notify();
        return this.getMetadata();
      } catch (error) {
        this.status = 'error';
        this.stopReason = 'start_error';
        this._error(error, 'start');
        this._finish(false);
        throw error;
      }
    }

    stop(reason = 'user') {
      if (this.finalResult) return this.done;
      if (!this.anchors.stop_requested) this.anchors.stop_requested = this._anchor();
      if (!this.stopReason) this.stopReason = reason;
      if (!this.recorder || this.status === 'idle') {
        this._finish(false);
        return this.done;
      }
      if (this.finishTimer === null) {
        this.finishTimer = root.setTimeout(() => {
          this.status = 'error';
          this._error(new Error('浏览器未完成录像封装；已保留收到的部分视频。'), 'finalization_timeout');
          this._finish(false);
        }, this.finalizationTimeoutMs);
      }
      if (this.recorder.state !== 'inactive' && this.status !== 'stopping') {
        this.status = 'stopping';
        this._notify();
        try { this.recorder.stop(); } catch (error) {
          this.status = 'error';
          this._error(error, 'stop');
          // The retained partial file remains available if the browser rejects stop.
          this._finish(false);
        }
      }
      // An inactive recorder can still have final data/stop events queued.
      return this.done;
    }

    _finish(receivedStopEvent) {
      if (this.finalResult) return;
      if (this.finishTimer !== null) { root.clearTimeout(this.finishTimer); this.finishTimer = null; }
      this.anchors.stopped = this._anchor();
      this.receivedStopEvent = receivedStopEvent;
      this.status = this.errors.length ? 'error' : 'stopped';
      const blob = new Blob(this.chunks, {type: this.mimeType || 'application/octet-stream'});
      this.finalResult = {blob, metadata: this.getMetadata()};
      this.resolveDone(this.finalResult);
      this._notify();
      if (this.onStop) {
        try { this.onStop(this.finalResult); } catch (_) { /* The result is already retained. */ }
      }
      // Media tracks belong to the caller; stopping a recording must not stop preview/inference.
    }

    getMetadata() {
      const start = this.anchors.started || this.anchors.requested;
      const end = this.anchors.stop_requested || this.anchors.stopped;
      const elapsed = start ? Math.max(0, (end ? end.performance_ms : this.clock()) - start.performance_ms) : 0;
      return {
        schema_version: 1, id: this.id, basename: this.basename, status: this.status,
        mime_type: this.mimeType, extension: videoExtension(this.mimeType),
        video_settings: {...this.videoSettings}, audio: false, overlay: false,
        elapsed_ms: finite(elapsed), byte_length: this.bytes,
        chunk_count: this.chunks.length, data_event_count: this.chunkEvents.length,
        chunks: this.chunkEvents.map(item => ({...item})),
        anchors: Object.fromEntries(Object.entries(this.anchors).map(([name, value]) => [name, value && {...value}])),
        stop_reason: this.stopReason, received_stop_event: !!this.receivedStopEvent,
        partial: this.errors.length > 0 || (this.status === 'stopped' && !this.receivedStopEvent),
        errors: this.errors.map(error => ({...error})),
        timing_note: 'performance_ms and source_video_time_s are browser observation anchors, not encoded frame timestamps. MediaRecorder start/data events may be delayed; 1000 ms chunk timeslice is not a video clock. Align diagnostics approximately using the anchors, then verify action boundaries against the saved video.',
      };
    }
  }

  const crcTable = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let bit = 0; bit < 8; bit++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    crcTable[i] = c >>> 0;
  }
  function crc32(bytes) {
    let crc = 0xffffffff;
    for (const byte of bytes) crc = crcTable[(crc ^ byte) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
  }

  async function makeStoredZip(entries) {
    if (!Array.isArray(entries) || entries.length > 65535) throw new Error('ZIP entry limit exceeded.');
    const localParts = [], directoryParts = [], names = new Set();
    let offset = 0, directorySize = 0;
    for (const entry of entries) {
      const name = String(entry.name || '');
      if (!name || /^[\/\\]/.test(name) || /[\\\x00-\x1f]/.test(name) || name.split('/').includes('..') || names.has(name)) {
        throw new Error('Invalid or duplicate ZIP filename.');
      }
      names.add(name);
      const filename = new TextEncoder().encode(name);
      if (filename.length > 65535) throw new Error('ZIP filename too long.');
      const blob = entry.blob;
      if (!blob || typeof blob.arrayBuffer !== 'function') throw new Error('ZIP entry must contain a Blob.');
      if (blob.size > 0xffffffff) throw new Error('ZIP64 is not supported.');
      const bytes = new Uint8Array(await blob.arrayBuffer());
      const checksum = crc32(bytes), size = bytes.byteLength;
      const local = new Uint8Array(30 + filename.length), view = new DataView(local.buffer);
      view.setUint32(0, 0x04034b50, true);
      view.setUint16(4, 20, true); view.setUint16(6, 0x0800, true); // UTF-8, stored.
      view.setUint16(12, 0x0021, true); // 1980-01-01; reproducible archive timestamps.
      view.setUint32(14, checksum, true); view.setUint32(18, size, true); view.setUint32(22, size, true);
      view.setUint16(26, filename.length, true); local.set(filename, 30);
      const central = new Uint8Array(46 + filename.length), cv = new DataView(central.buffer);
      cv.setUint32(0, 0x02014b50, true); cv.setUint16(4, 20, true); cv.setUint16(6, 20, true);
      cv.setUint16(8, 0x0800, true); cv.setUint16(14, 0x0021, true);
      cv.setUint32(16, checksum, true); cv.setUint32(20, size, true); cv.setUint32(24, size, true);
      cv.setUint16(28, filename.length, true); cv.setUint32(42, offset, true); central.set(filename, 46);
      localParts.push(local, blob); directoryParts.push(central);
      offset += local.length + size; directorySize += central.length;
      if (offset + directorySize > 0xffffffff) throw new Error('ZIP64 is not supported.');
    }
    const end = new Uint8Array(22), ev = new DataView(end.buffer);
    ev.setUint32(0, 0x06054b50, true);
    ev.setUint16(8, entries.length, true); ev.setUint16(10, entries.length, true);
    ev.setUint32(12, directorySize, true); ev.setUint32(16, offset, true);
    return new Blob([...localParts, ...directoryParts, end], {type: 'application/zip'});
  }

  const api = {chooseMimeType, videoExtension, RawRecordingSession, makeStoredZip};
  root.FallpreRecording = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof globalThis !== 'undefined' ? globalThis : window);
