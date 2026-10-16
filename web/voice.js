/* voice-agent widget. No brain inside: records push-to-talk audio, streams it to the voice service
 * over WebSocket, plays the spoken reply as it arrives. Embed anywhere:
 *   <script src="https://voice.example/voice.js"></script>
 *   const agent = new VoiceAgent({url: "https://voice.example", token: "..."}); agent.connect();
 *   agent.on("transcript", m => ...); agent.startRecording(); agent.stopRecording();
 * Needs a secure context (https or localhost) for the microphone. */
(function (global) {
  class VoiceAgent {
    constructor(opts) {
      this.url = (opts.url || location.origin).replace(/\/$/, "");
      this.token = opts.token || "";
      this.session = opts.session || null;
      this.debug = !!opts.debug;
      this.handlers = {};
      this.ws = null;
      this.rec = null;
      this.stream = null;
      this.queue = [];
      this.playing = false;
      this.ctx = null;
      this.unlocked = false;
      this.current = null;
    }
    on(type, fn) { (this.handlers[type] = this.handlers[type] || []).push(fn); return this; }
    emit(type, msg) { (this.handlers[type] || []).forEach((f) => { try { f(msg); } catch (e) { console.error(e); } }); }
    connect() {
      const u = new URL(this.url + "/ws");
      u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
      if (this.token) u.searchParams.set("token", this.token);
      if (this.session) u.searchParams.set("session", this.session);
      const ws = new WebSocket(u.toString());
      ws.binaryType = "arraybuffer";
      ws.onopen = () => this.emit("open", {});
      ws.onclose = (e) => this.emit("close", { code: e.code });
      ws.onerror = () => this.emit("error", { text: "websocket error" });
      ws.onmessage = (e) => {
        let m; try { m = JSON.parse(e.data); } catch (_) { return; }
        if (this.debug) console.log("voice <", m.type, m);
        if (m.type === "ready") this.session = m.session;
        if (m.type === "sentence") this.lastProvider = m.provider;
        if (m.type === "audio") this.enqueue(m);
        this.emit(m.type, m);
      };
      this.ws = ws;
      return this;
    }
    close() { if (this.ws) this.ws.close(); this.ws = null; }
    send(obj) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }
    sendText(text) { this.unlock(); this.stopPlayback(); this.send({ type: "text", text }); }
    cancel() { this.stopPlayback(); this.send({ type: "cancel" }); }
    reset() { this.stopPlayback(); this.send({ type: "reset" }); }
    async startRecording() {
      if (this.rec) return;
      this.unlock();
      this.stopPlayback();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("microphone needs https or localhost");
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"].find((t) => global.MediaRecorder && MediaRecorder.isTypeSupported(t));
      const rec = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
      const ws = this.ws;
      rec.ondataavailable = (e) => { if (e.data && e.data.size && ws && ws.readyState === 1) e.data.arrayBuffer().then((b) => ws.send(b)); };
      rec.onstop = () => {
        // ondataavailable for the final chunk fires before onstop; give its arrayBuffer() a tick.
        setTimeout(() => this.send({ type: "utterance_end" }), 60);
        if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
        this.rec = null;
      };
      rec.start(250);
      this.rec = rec;
      this.emit("recording", { mime: rec.mimeType });
    }
    stopRecording() { if (this.rec && this.rec.state !== "inactive") this.rec.stop(); }
    /* Playback goes through Web Audio. Safari (and iOS, and Chrome's autoplay policy) refuse sound that
     * starts outside a user gesture; an AudioContext created and resumed inside a tap stays unlocked,
     * so call unlock() from the hold-to-talk pointerdown or the Connect click. */
    unlock() {
      const AC = global.AudioContext || global.webkitAudioContext;
      if (!AC) return;
      if (!this.ctx) this.ctx = new AC();
      if (this.ctx.state === "suspended") this.ctx.resume().catch(() => {});
      if (!this.unlocked) {                      // one silent buffer inside the gesture seals the deal on iOS
        const b = this.ctx.createBuffer(1, 1, 22050); const s = this.ctx.createBufferSource();
        s.buffer = b; s.connect(this.ctx.destination); try { s.start(0); } catch (_) {}
        this.unlocked = true;
      }
    }
    enqueue(m) {
      const bytes = Uint8Array.from(atob(m.b64), (c) => c.charCodeAt(0));
      if (!this.ctx) this.unlock();
      if (!this.ctx) { this.emit("error", { text: "no audio output available in this browser" }); return; }
      const seq = m.seq;
      this.pending = (this.pending || Promise.resolve()).then(() =>
        this.ctx.decodeAudioData(bytes.buffer.slice(0)).then((buf) => {
          this.queue.push(buf);
          if (!this.playing) this.playNext();
        }, (err) => this.emit("error", { text: `could not decode ${m.mime} (${m.provider}, seq ${seq}): ${err && err.message || err}` }))
      );
    }
    playNext() {
      const buf = this.queue.shift();
      if (!buf) { this.playing = false; this.current = null; this.emit("idle", {}); return; }
      this.playing = true;
      if (this.ctx.state === "suspended") {
        this.ctx.resume().catch(() => {});
        this.emit("status", { text: "sound is blocked until you tap the page once" });
      }
      const src = this.ctx.createBufferSource();
      src.buffer = buf; src.connect(this.ctx.destination);
      src.onended = () => { if (this.current === src) this.playNext(); };
      this.current = src;
      try { src.start(0); this.emit("playing", { seconds: buf.duration }); } catch (e) { this.emit("error", { text: "playback failed: " + e.message }); this.playNext(); }
    }
    stopPlayback() { this.queue = []; const c = this.current; this.current = null; if (c) { try { c.stop(0); } catch (_) {} } this.playing = false; }
  }
  global.VoiceAgent = VoiceAgent;
})(window);
