# voice-agent

A standalone voice front end with **no brain inside**. It turns speech into text, sends the text to any
OpenAI-compatible chat endpoint, and speaks the reply, sentence by sentence as it streams. Providers for
each speech leg sit in an ordered **waterfall**: OpenAI, Azure AI Speech, Google Cloud Speech, Google Gemini, then a free local
floor (faster-whisper + Piper) that never goes on quota. One container, one YAML per deployment, one
`<script>` tag per site.

First consumer: Mahadeva (shambho-jarvis stage 6), whose brain is the stage 4 Hermes bridge. Decisions and
economics: `/mnt/project-files/voice-agent/plan.md` in the Shambho project, `docs/decisions.md` D23+ here.
This folder is self-contained so it can move to its own repository (`git subtree split -P voice`) without
edits; nothing in it imports from the rest of shambho-jarvis.

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
| `WS /ws?token=&session=` | the voice loop (below) |
| `GET /healthz` | no auth: version, profile, active providers |
| `GET /api/status` | waterfall state: per provider ok / cooling / parked, spend this month, budget |
| `POST /api/stt` (multipart `file`) | transcribe through the waterfall |
| `POST /api/tts` `{text, provider?}` | speak through the waterfall (or one named provider) |
| `POST /api/probe?providers=a,b` | each provider speaks a test phrase and must hear it back; pass/fail per provider |
| `POST /api/reset?provider=` | clear a breaker |

Auth: `?token=` or `Authorization: Bearer`. Tokens come from `service.tokens` (`${VOICE_TOKEN}`).

### WebSocket protocol

Client → server: binary frames = audio chunks of one utterance (any container ffmpeg reads: webm/opus,
mp4, ogg, wav); then `{"type":"utterance_end"}`. Or `{"type":"text","text":"..."}` to skip the
microphone. `{"type":"cancel"}`, `{"type":"reset"}` (new brain session), `{"type":"ping"}`.

Server → client: `ready{session, profile, stt[], tts[], brain}` · `transcript{text, provider, ms}` ·
`token{text}` (brain deltas) · `sentence{text, provider, seq}` then `audio{seq, mime, b64}` ·
`status{text}` (a waterfall fall-through, shown to the user) · `error{text}` · `done{turn_ms, reply}`.

## Waterfall (decision 4a)

Per leg, in config order. A provider is skipped when it has no key, is over its monthly budget
(`budget_usd`, default $10; usage estimated from list prices and persisted in `/data/usage.json`), or is
cooling (15 min after a 429, 5xx, timeout over 8 s, or any adapter error). A non-retryable failure (401,
400, missing model) parks the provider until restart or `/api/reset`. The local floor has no budget and
a 60 s timeout. `VOICE_FAIL_PROVIDERS=openai,azure` forces failures for testing.

Estimated cost per conversation minute (30 s speech each way): OpenAI ~$0.01, Azure ~$0.015, Google Cloud
~$0.019 (Neural2 voice; ~$0.014 with a Standard voice), Gemini ~$0.005 (preview pricing), local $0. Brain cost is separate and the same on every path.

## Providers

- **openai**: `gpt-4o-mini-transcribe` + `gpt-4o-mini-tts` (mp3). Env `OPENAI_API_KEY`.
- **azure**: Azure AI Speech REST, short-audio STT (≤60 s utterances) + neural TTS, works with an
  AIServices multi-service key. Env `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION`.
- **gemini**: Gemini API (AI Studio key): audio understanding as STT, Gemini TTS (PCM → WAV). Env
  `GEMINI_API_KEY`; models overridable with `GEMINI_STT_MODEL`, `GEMINI_TTS_MODEL`.
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
splitting). On asgard, `tests/stage6/run.sh` probes the real providers and the bridge.
