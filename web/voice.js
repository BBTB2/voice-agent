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
      this.closed = false;
      const u = new URL(this.url + "/ws");
      u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
      if (this.token) u.searchParams.set("token", this.token);
      if (this.session) u.searchParams.set("session", this.session);
      const ws = new WebSocket(u.toString());
      ws.binaryType = "arraybuffer";
      ws.onopen = () => { this.retries = 0; this.emit("open", {}); };
      ws.onclose = (e) => { this.emit("close", { code: e.code }); this.maybeReconnect(e.code); };
      ws.onerror = () => this.emit("error", { text: "websocket error" });
      ws.onmessage = (e) => {
        let m; try { m = JSON.parse(e.data); } catch (_) { return; }
        if (m.type === "tick") return;              // server keepalive
        if (this.debug) console.log("voice <", m.type, m);
        if (m.type === "ready") this.session = m.session;
        if (m.type === "sentence") this.lastProvider = m.provider;
        if (m.type === "audio") this.enqueue(m);
        this.emit(m.type, m);
      };
      this.ws = ws;
      return this;
    }
    close() { this.closed = true; if (this.ws) this.ws.close(); this.ws = null; }
    /* A phone drops the socket whenever the browser goes to the background (switching to another app, the
     * screen locking). Reconnect with the same session so the conversation continues; give up after a few
     * tries or on a refusal (bad token 4401, origin 4403). */
    maybeReconnect(code) {
      if (this.closed || code === 4401 || code === 4403 || code === 1000) return;
      this.retries = (this.retries || 0) + 1;
      if (this.retries > 5) return;
      const wait = Math.min(1000 * this.retries, 5000);
      this.emit("status", { text: `connection dropped, reconnecting in ${wait / 1000} s` });
      const go = () => { if (!this.closed) { this.ws = null; this.connect(); } };
      setTimeout(() => {
        if (global.document && document.visibilityState === "hidden") {
          const once = () => { if (document.visibilityState === "visible") { document.removeEventListener("visibilitychange", once); go(); } };
          document.addEventListener("visibilitychange", once);
        } else go();
      }, wait);
    }
    send(obj) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(obj)); }
    sendText(text) { this.unlock(); this.stopPlayback(); this.send({ type: "text", text }); }
    cancel() { this.stopPlayback(); this.send({ type: "cancel" }); }
    reset() { this.stopPlayback(); this.send({ type: "reset" }); }
    /** "I could not hear that": the service sends the last reply to the person's Telegram as a voice note. */
    resend() { this.send({ type: "resend" }); }
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
    stopRecording() { if (this.ctx && this.ctx.state !== "running") this.ctx.resume().catch(() => {}); if (this.rec && this.rec.state !== "inactive") this.rec.stop(); }
    /* Playback goes through Web Audio. Safari (and iOS, and Chrome's autoplay policy) refuse sound that
     * starts outside a user gesture; an AudioContext created and resumed inside a tap stays unlocked,
     * so call unlock() from the hold-to-talk pointerdown or the Connect click. */
    unlock() {
      const AC = global.AudioContext || global.webkitAudioContext;
      if (!AC) return;
      if (!this.ctx) {
        this.ctx = new AC();
        // Phone speakers are quiet with TTS at its native level: a compressor plus a modest boost, no clipping.
        const comp = this.ctx.createDynamicsCompressor(); const gain = this.ctx.createGain();
        gain.gain.value = this.gain || 1.6; comp.connect(gain); gain.connect(this.ctx.destination);
        this.out = comp;
      }
      if (this.ctx.state !== "running") this.ctx.resume().then(() => { if (this.queue.length && !this.playing) this.playNext(); }, () => {});
      if (!this.unlocked) {                      // one silent buffer inside the gesture seals the deal on iOS
        const b = this.ctx.createBuffer(1, 1, 22050); const s = this.ctx.createBufferSource();
        s.buffer = b; s.connect(this.out); try { s.start(0); } catch (_) {}
        this.unlocked = true;
      }
      /* iOS mutes Web Audio while the ring/silent switch is on silent, because the page's audio session
       * is "ambient". A looping silent <audio> element started inside the same tap moves the session to
       * "playback", which the switch does not silence. Harmless elsewhere (it is silence). */
      if (!this.keepalive && global.Audio) {
        try {
          const a = new global.Audio(VoiceAgent.silentWav()); a.loop = true; a.setAttribute("playsinline", "");
          a.play().catch(() => {}); this.keepalive = a;
        } catch (_) {}
      }
    }
    /** 0.1 s of 8 kHz mono silence as a WAV data URI (built once, ~1 KB). */
    static silentWav() {
      if (VoiceAgent._silent) return VoiceAgent._silent;
      const rate = 8000, n = 800, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
      const str = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
      str(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); str(8, "WAVE"); str(12, "fmt "); v.setUint32(16, 16, true);
      v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true);
      v.setUint16(32, 2, true); v.setUint16(34, 16, true); str(36, "data"); v.setUint32(40, n * 2, true);
      let bin = ""; const bytes = new Uint8Array(buf); for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      VoiceAgent._silent = "data:audio/wav;base64," + btoa(bin);
      return VoiceAgent._silent;
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
      if (this.ctx.state !== "running") {
        // iOS parks the context ("interrupted") after the microphone or another app used the audio; resume it
        // and play once it is back. Only if that fails does the page need a tap.
        this.queue.unshift(buf); this.playing = false;
        this.ctx.resume().then(() => this.playNext(), () => {});
        setTimeout(() => { if (this.ctx.state !== "running") this.emit("status", { text: "sound is blocked until you tap the page once" }); }, 1500);
        return;
      }
      const src = this.ctx.createBufferSource();
      src.buffer = buf; src.connect(this.out);
      src.onended = () => { if (this.current === src) this.playNext(); };
      this.current = src;
      try { src.start(0); this.emit("playing", { seconds: buf.duration }); } catch (e) { this.emit("error", { text: "playback failed: " + e.message }); this.playNext(); }
    }
    stopPlayback() { this.queue = []; const c = this.current; this.current = null; if (c) { try { c.stop(0); } catch (_) {} } this.playing = false; }
  }
  global.VoiceAgent = VoiceAgent;
})(window);
