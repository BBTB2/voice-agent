/* voice-agent widget. No brain inside: records push-to-talk audio, streams it to the voice service
 * over WebSocket, plays the spoken reply as it arrives. Embed anywhere:
 *   <script src="https://voice.example/voice.js"></script>
 *   const agent = new VoiceAgent({url: "https://voice.example", token: "..."}); agent.connect();
 *   agent.on("transcript", m => ...); agent.startRecording(); agent.stopRecording();
 * Apps (stage 9): the token picks the app on the service side (brain, origins, caps). When the app's brain is
 * somebody's site that needs the signed-in person's own credential (prof-snarky: the Clerk session token), pass
 * `auth: async () => token`; the widget attaches it to every turn and answers the service's "auth_request" with a
 * fresh one. `persona` (string, optional) and `threadId` (string, optional) ride along with each turn for brains that
 * take them. threadId is prof-snarky's follow-up memory, read afresh on every turn ("" = new question): the service
 * echoes `thread_id` in meta and the host sets `agent.threadId` from it to continue. The service sends a
 * "meta" event after the words (citations, chat id, credits) and "error" with kind "auth" when the brain refused the person.
 * Camera (stage 10, "eyes"): agent.enableCamera({preview: videoEl}) keeps the back camera live in the preview;
 * every hold then also records a clip and takes one full-resolution still at release. The still goes first,
 * then the words, then the clip, so a weak signal degrades to still-only instead of failing. The service
 * sends a "vision" event with the description before the reply. agent.disableCamera() stops the camera.
 * Needs a secure context (https or localhost) for the microphone and the camera. */
(function (global) {
  class VoiceAgent {
    constructor(opts) {
      this.url = (opts.url || location.origin).replace(/\/$/, "");
      this.token = opts.token || "";
      this.session = opts.session || null;
      this.auth = typeof opts.auth === "function" ? opts.auth : null;   // async () => the person's credential for the brain
      this.persona = opts.persona || null;
      this.threadId = opts.threadId || null;   // prof-snarky follow-up memory; "" sent = new question
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
      this.camStream = null;          // live camera (video track only; the microphone is taken per hold as before)
      this.preview = null;
      this.clipChunks = [];
    }
    get cameraOn() { return !!this.camStream; }
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
      ws.onopen = () => { this.retries = 0; if (this.cameraOn) this.send({ type: "camera", on: true }); this.emit("open", {}); };
      ws.onclose = (e) => { this.emit("close", { code: e.code }); this.maybeReconnect(e.code); };
      ws.onerror = () => this.emit("error", { text: "websocket error" });
      ws.onmessage = (e) => {
        let m; try { m = JSON.parse(e.data); } catch (_) { return; }
        if (m.type === "tick") return;              // server keepalive
        if (this.debug) console.log("voice <", m.type, m);
        if (m.type === "ready") { this.session = m.session; this.app = m.app; this.needsAuth = !!m.needs_auth; }
        if (m.type === "auth_request") { this.sendAuth(); return; }
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
    /** Fetch the person's credential (if an auth callback was given) and send it on its own. */
    sendAuth() {
      if (!this.auth) return Promise.resolve(null);
      return Promise.resolve().then(() => this.auth()).then((t) => { this.send({ type: "auth", token: t || "" }); return t; },
        (e) => { this.emit("error", { text: "auth: " + (e && e.message || e), kind: "auth" }); return null; });
    }
    /** The turn-ending message with the credential and persona attached (both optional). */
    sendTurn(msg) {
      if (this.persona) msg.persona = this.persona;
      msg.thread_id = this.threadId || "";
      const go = (t) => { if (t) msg.auth = t; this.send(msg); };
      if (!this.auth) return go(null);
      Promise.resolve().then(() => this.auth()).then(go, () => go(null));
    }
    sendText(text) { this.unlock(); this.stopPlayback(); this.sendTurn({ type: "text", text }); }
    cancel() { this.stopPlayback(); this.send({ type: "cancel" }); }
    reset() { this.stopPlayback(); this.threadId = null; this.send({ type: "reset" }); }
    /** "I could not hear that": the service sends the last reply to the person's Telegram as a voice note. */
    resend() { this.send({ type: "resend" }); }
    /** Binary frame with a tag byte in front (camera mode): 1 audio, 2 clip, 3 still. */
    sendTagged(tag, buf) {
      const ws = this.ws; if (!ws || ws.readyState !== 1) return;
      const out = new Uint8Array(buf.byteLength + 1); out[0] = tag; out.set(new Uint8Array(buf), 1); ws.send(out.buffer);
    }
    async startRecording() {
      if (this.rec) return;
      this.unlock();
      this.stopPlayback();
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("microphone needs https or localhost");
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"].find((t) => global.MediaRecorder && MediaRecorder.isTypeSupported(t));
      const rec = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined);
      const ws = this.ws;
      const camera = this.cameraOn;
      rec.ondataavailable = (e) => {
        if (!(e.data && e.data.size && ws && ws.readyState === 1)) return;
        e.data.arrayBuffer().then((b) => { if (camera) this.sendTagged(1, b); else ws.send(b); });
      };
      rec.onstop = () => {
        // ondataavailable for the final chunk fires before onstop; give its arrayBuffer() a tick. In camera mode
        // the still must be on the wire first, then the words, then the clip (weak links get the still at least).
        const finish = () => {
          this.sendTurn(camera ? { type: "utterance_end", camera: true, clip: this.clipRec ? "pending" : "none" } : { type: "utterance_end" });
          this.utteranceSent = true;
          this.flushClip();
        };
        setTimeout(() => { (this.stillPromise || Promise.resolve()).then(finish, finish); }, 60);
        if (this.stream) { this.stream.getTracks().forEach((t) => t.stop()); this.stream = null; }
        this.rec = null;
      };
      if (camera) this.startClip();
      rec.start(250);
      this.rec = rec;
      this.emit("recording", { mime: rec.mimeType, camera });
    }
    stopRecording() {
      if (this.ctx && this.ctx.state !== "running") this.ctx.resume().catch(() => {});
      if (this.cameraOn && this.rec && this.rec.state !== "inactive") this.stillPromise = this.takeStill();
      if (this.rec && this.rec.state !== "inactive") this.rec.stop();
      if (this.clipRec && this.clipRec.state !== "inactive") this.clipRec.stop();
    }
    /* ---- camera (stage 10) ---- */
    async enableCamera(opts) {
      opts = opts || {};
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) throw new Error("camera needs https or localhost");
      if (this.camStream) return this.camStream;
      const video = { facingMode: opts.facing || "environment", width: { ideal: opts.width || 1920 }, height: { ideal: opts.height || 1080 } };
      this.camStream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      this.preview = opts.preview || this.preview || null;
      if (!this.preview && global.document) { this.preview = document.createElement("video"); this.preview.style.display = "none"; document.body.appendChild(this.preview); }
      this.preview.setAttribute("playsinline", ""); this.preview.muted = true; this.preview.srcObject = this.camStream;
      this.preview.play().catch(() => {});
      this.send({ type: "camera", on: true });
      this.emit("camera", { on: true, settings: this.camStream.getVideoTracks()[0].getSettings() });
      return this.camStream;
    }
    disableCamera() {
      if (this.camStream) { this.camStream.getTracks().forEach((t) => t.stop()); this.camStream = null; }
      if (this.preview) this.preview.srcObject = null;
      this.send({ type: "camera", on: false });
      this.emit("camera", { on: false });
    }
    startClip() {
      if (!this.camStream || !global.MediaRecorder) { this.clipRec = null; return; }
      const mime = ["video/mp4", "video/webm;codecs=vp8", "video/webm;codecs=vp9", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
      let rec;
      try { rec = new MediaRecorder(new MediaStream(this.camStream.getVideoTracks()), Object.assign({ videoBitsPerSecond: 1500000 }, mime ? { mimeType: mime } : {})); }
      catch (e) { this.emit("error", { text: "clip recorder: " + e.message }); this.clipRec = null; return; }
      this.clipChunks = []; this.clipDone = false; this.utteranceSent = false;
      rec.ondataavailable = (e) => { if (e.data && e.data.size) this.clipChunks.push(e.data); };
      rec.onstop = () => { this.clipDone = true; this.flushClip(); };
      this.send({ type: "clip_begin", mime: rec.mimeType || mime || "video/mp4" });
      rec.start(500);
      this.clipRec = rec;
    }
    /** Send the buffered clip once both the words are on the wire and the recorder has stopped. */
    flushClip() {
      if (!this.clipRec || !this.clipDone || !this.utteranceSent) return;
      const chunks = this.clipChunks; this.clipChunks = []; this.clipRec = null;
      const total = chunks.reduce((n, c) => n + c.size, 0);
      this.emit("clip", { bytes: total });
      let p = Promise.resolve();
      chunks.forEach((c) => { p = p.then(() => c.arrayBuffer()).then((b) => this.sendTagged(2, b)); });
      p.then(() => this.send({ type: "clip_end" }));
    }
    /** One full-resolution JPEG from the live camera, sent as a tagged binary frame. */
    takeStill() {
      return new Promise((resolve) => {
        try {
          const v = this.preview;
          if (!v || !v.videoWidth) return resolve(false);
          const c = document.createElement("canvas"); c.width = v.videoWidth; c.height = v.videoHeight;
          c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
          c.toBlob((blob) => {
            if (!blob) return resolve(false);
            blob.arrayBuffer().then((b) => { this.sendTagged(3, b); this.emit("still", { bytes: b.byteLength, width: c.width, height: c.height }); resolve(true); }, () => resolve(false));
          }, "image/jpeg", 0.85);
        } catch (e) { resolve(false); }
      });
    }
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
