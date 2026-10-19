# voice-agent

A standalone voice front end with **no brain inside**. It turns speech into text, sends the text to any
OpenAI-compatible chat endpoint, and speaks the reply, sentence by sentence as it streams. Providers for
each speech leg sit in an ordered **waterfall**: OpenAI, Azure AI Speech, Google Cloud Speech, Google Gemini, then a free local
floor (faster-whisper + Piper) that never goes on quota. One container, one YAML per deployment, one
`<script>` tag per site.

This is its own repository (BBTB2/voice-agent, private, since 17 Oct 2026). First consumer: Mahadeva
(BBTB2/shambho-jarvis), whose brain is the Hermes bridge and which carries this repo as the `voice/` subtree.
Decisions and economics: `/mnt/project-files/voice-agent/plan.md` in the Shambho project, `docs/decisions.md`
D23 and D26 in shambho-jarvis. Nothing here imports from any other project.

## Use it in another project

Two ways, pick one per project:

- **Run the container** beside your app and point it at your brain: `docker build -t voice-agent .` then the
  `docker run` line below with `BRAIN_URL` = any OpenAI-compatible `/v1` base, and drop
  `<script src="https://your-host/voice.js">` + `new VoiceAgent({url, token})` into the page.
- **Vendor the code as a subtree** (what shambho-jarvis does), so the consuming repo needs no second deploy key:
  `git subtree add --prefix voice https://github.com/BBTB2/voice-agent.git main --squash` once, then
  `git subtree pull --prefix voice https://github.com/BBTB2/voice-agent.git main --squash` to update, and
  `git subtree push --prefix voice https://github.com/BBTB2/voice-agent.git main` to send fixes made in place back here.

```
browser / any site ── voice.js (push-to-talk, playback) ──WebSocket──▶ voice-agent
                                                                       ├─ STT waterfall  openai > azure > google > gemini > local
                                                                       ├─ brain          POST /v1/chat/completions (stream), prompt_cache_key = session
                                                                       └─ TTS waterfall  openai > azure > google > gemini > local
```

## Run

```
docker build -t voice-agent voice/
docker run --rm -p 127.0.0.1:18090:8090 -e VOICE_PROFILE=mahadeva -e VOICE_TOKEN=... -e OPENAI_API_KEY=... \
  -e AZURE_SPEECH_KEY=... -e AZURE_SPEECH_REGION=swedencentral -e GEMINI_API_KEY=... \
  -e BRAIN_URL=http://host:8077/v1 -e BRAIN_API_KEY=... -v voice-data:/data voice-agent
```

Open `http://127.0.0.1:18090/` (the microphone needs https or localhost). `VOICE_PROFILE` picks
`config/profiles/<name>.yaml` over `config/default.yaml`; `VOICE_CONFIG=/path.yaml` overlays once more.
Secrets only ever arrive as environment variables.

## Endpoints

| Path | What |
|---|---|
| `GET /` | standalone page (push-to-talk, typed fallback, provider status) |
| `GET /voice.js` | the embeddable widget (`new VoiceAgent({url, token})`) |
| `GET /nhs/`, `/nhs/reach.html` | NHS demo page and network reachability test |
| `GET /docs/nhs-ig-note.md`, `/docs/synthetic-cases.md` | IG note and demo scripts |
| `WS /ws?token=&session=` | the voice loop (below); the server sends `{"type":"tick"}` every 25 s (`service.keepalive_seconds`) so proxies such as Cloudflare never see an idle socket during a long turn, and `voice.js` reconnects with the same session when a phone browser drops the socket in the background |
| `GET /healthz` | no auth: version, profile, active providers |
| `GET /api/status` | waterfall state: per provider ok / cooling / parked, spend this month, budget |
| `POST /api/stt` (multipart `file`) | transcribe through the waterfall |
| `POST /api/tts` `{text, provider?}` | speak through the waterfall (or one named provider) |
| `POST /api/probe?providers=a,b` | each provider speaks a test phrase and must hear it back; pass/fail per provider |
| `POST /api/reset?provider=` | clear a breaker |

Auth: `?token=` or `Authorization: Bearer`. Tokens come from `service.tokens` (`${VOICE_TOKEN}`).
Browser origins: the page's own host is always allowed; sites that embed `voice.js` go in `service.allowed_origins`
(`VOICE_ALLOWED_ORIGINS=https://radread.co.uk,https://sitewright.uk`); any other browser origin gets WebSocket close 4403.

### WebSocket protocol

Client → server: binary frames = audio chunks of one utterance (any container ffmpeg reads: webm/opus,
mp4, ogg, wav); then `{"type":"utterance_end"}`. Or `{"type":"text","text":"..."}` to skip the
microphone. `{"type":"cancel"}`, `{"type":"reset"}` (new brain session), `{"type":"ping"}`.

Server → client: `ready{session, profile, stt[], tts[], vision[], camera, brain}` · `transcript{text, provider, ms}` ·
`token{text}` (brain deltas) · `sentence{text, provider, seq}` then `audio{seq, mime, b64}` ·
`status{text}` (a waterfall fall-through, shown to the user) · `error{text}` · `done{turn_ms, reply}`.

### Camera (stage 10, "eyes")

With the camera switched on in the page (`agent.enableCamera({preview})` in the widget), every hold also records the
back camera. Order on release: the still (tag 3) first, then `utterance_end{camera:true, clip:"pending"|"none"}`,
then the clip chunks and `clip_end`, so a weak link degrades to still-only instead of failing. Once the page sends
`{"type":"camera","on":true}` every binary frame carries a tag byte: 1 audio chunk, 2 clip chunk, 3 the still (JPEG);
`{"type":"clip_begin","mime"}` opens a clip, `{"type":"still","mime","b64"}` is the JSON form of tag 3. A typed turn
can carry `camera:true` too. The service waits `camera.clip_wait_seconds` (12) for the clip, keeps its last
`camera.clip_max_seconds` (8) re-encoded small (`voice_agent/video.py`, ffmpeg, nothing kept on disk), and runs the
**vision leg** of the waterfall: `gemini` reads clip + still natively, `openai` (fallback) gets the still plus three
frames pulled from the clip. The description goes to the brain inside the user turn as
`[Camera, live, described by gemini: ...]` (`voice_agent/vision.py`) and, with `brain.send_still` (default true),
the still rides along as an OpenAI `image_url` data part; the Hermes bridge forwards it as an ACP image block so the
agent can look again itself. The page gets `vision{provider, text, used, note, usd, ms}` before the reply.
The model is told to say what it cannot read or see rather than guess, and the brain is told to pass that on.
Cost: a 5 s clip + still is roughly 3-4k Gemini input tokens, well under a penny a question at Oct 2026 list prices;
`usage.json` counts `looks` per provider under the same monthly cap. `camera.enabled: false` hides it all (nhs-demo).

## Delivery fallback (Telegram voice note)

With `fallback.telegram_bot_token` and `fallback.telegram_chat_id` set (`VOICE_TELEGRAM_BOT_TOKEN`,
`VOICE_TELEGRAM_CHAT_ID`; the stage 6 installer copies them from the gateway's `hermes.env` for the mahadeva
profile), a reply the browser lost mid-turn (phone in the background, signal gone) is finished anyway and sent
to that chat as one OGG/Opus voice note made from the same sentence clips, with the text as caption. The page's
"Send to Telegram" button (`{"type":"resend"}` on the socket) sends the last reply the same way when the person
could not hear it. No extra synthesis: the clips are reused. Without a bot token nothing changes.

## Waterfall (decision 4a)

Per leg, in config order. A provider is skipped when it has no key, is over its monthly budget
(`budget_usd`, default $10; usage estimated from list prices and persisted in `/data/usage.json`), or is
cooling (15 min after a 429, 5xx, timeout over 8 s, or any adapter error). A non-retryable failure (401,
400, missing model) parks the provider until restart or `/api/reset`. The local floor has no budget and
a 60 s timeout. `VOICE_FAIL_PROVIDERS=openai,azure` forces failures for testing.

Estimated cost per conversation minute (30 s speech each way): OpenAI ~$0.01, Azure ~$0.015, Google Cloud
~$0.019 (Neural2 voice; ~$0.014 with a Standard voice), Gemini ~$0.005 (preview pricing), local $0. Brain cost is separate and the same on every path.

## Providers

- **openai**: `gpt-4o-mini-transcribe` + `gpt-4o-mini-tts` (mp3); vision fallback `gpt-4o-mini` (`OPENAI_VISION_MODEL`). Env `OPENAI_API_KEY`.
- **azure**: Azure AI Speech REST, short-audio STT (≤60 s utterances) + neural TTS, works with an
  AIServices multi-service key. Env `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`.
- **gemini**: Gemini API (AI Studio key): audio understanding as STT, Gemini TTS (PCM → WAV). Env
  `GEMINI_API_KEY`; models overridable with `GEMINI_STT_MODEL`, `GEMINI_TTS_MODEL`, `GEMINI_VISION_MODEL` (the camera leg's first choice; it reads the clip as video).
- **google**: Google Cloud Speech-to-Text v1 (sync recognize, `latest_short`, LINEAR16) + Text-to-Speech v1
  (`en-GB-Neural2-B`, mp3). Needs a Cloud project with both APIs enabled and billing on. Credential is
  either `GOOGLE_SPEECH_API_KEY` (API key restricted to the two APIs) or `GOOGLE_SPEECH_CREDENTIALS_B64`
  (service-account JSON, base64; the adapter mints its own OAuth token, RS256 via `cryptography`).
  `GOOGLE_TTS_VOICE` picks the voice and with it the price tier (Standard $4/M chars, Neural2 $16/M,
  Chirp3-HD $30/M); `GOOGLE_STT_MODEL` the STT model. Data logging is off, so STT is $0.024/min after the
  free 60 min a month.
- **local**: faster-whisper `small` int8 + Piper `en_GB-alan-medium`, baked into the image.

Adding a provider = one file in `voice_agent/providers/` with `transcribe()` and `synthesize()`, one
line in `providers/__init__.py`, one block in the YAML.

## Profiles

- `mahadeva`: all five providers, brain = `http://jarvis-hermes-bridge:8077/v1`.
- `nhs-demo`: Azure then local only, EU/UK region, synthetic-cases banner, IG note. Decision 5b.

## Tests

`pytest voice/tests` runs offline with fake providers (waterfall, budgets, breaker, WS loop, sentence
splitting, the camera leg with ffmpeg-made clips). On asgard, `tests/stage6/run.sh` probes the real providers and
the bridge; `tests/stage10/run.sh` sends a synthetic clip and still through the real vision leg.
