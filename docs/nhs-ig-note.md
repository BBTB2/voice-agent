# Voice assistant in an NHS trust: information governance note (demo grade)

Written 16 Oct 2026 for the stage 6 demo profile. Purpose: show a skeptic the deployment shape that
would be compliant, and be honest about what the demo on asgard is not. **The demo processes
synthetic cases only.** Nothing below is legal advice; a trust's IG lead and Caldicott Guardian own
the decision.

## 1. What the system is

A voice front end with no model of its own. Three legs per turn:

1. **Speech-to-text**: the user's audio goes to a speech provider and comes back as text.
2. **Brain**: the text goes to a chat model endpoint (OpenAI-compatible API) that holds the conversation.
3. **Text-to-speech**: the reply text goes to a speech provider and comes back as audio.

Every leg is a configuration line, so the same container runs with consumer providers on a
personal server (the demo) or with trust-tenant Azure resources only (production).

## 2. Data flows, demo versus trust deployment

| Element | Demo on asgard (today) | Trust deployment (what IG would approve) |
|---|---|---|
| Where the service runs | Netcup server in Nuremberg, Germany, personal | Trust's Azure subscription (UK South) or on-premises VM; same container image |
| Speech provider | Azure AI Speech, Sweden Central, personal subscription | Azure AI Speech in the trust tenant, UK South, no data retention (default for the Speech REST API: audio is not stored) |
| Brain | Mahadeva (Hermes agent + DeepSeek API, personal) | Azure OpenAI in the trust tenant, or a trust-hosted model; the API is identical |
| Transport | HTTPS/WSS over Tailscale or ssh tunnel | HTTPS/WSS behind the trust's reverse proxy, Entra ID sign-in in front |
| Identity | Shared token | Entra ID (trust SSO), per-user tokens, audit of who spoke when |
| Audio retention | Not stored by the service; providers per their terms | Not stored; provider contracts under the trust's Microsoft agreement (DPA, UK residency) |
| Transcripts | In memory for the session; the brain keeps history | Logged to the trust's SIEM or not at all, per DPIA |
| Patient data | **None, synthetic only** | Only after DPIA sign-off, DTAC, clinical safety case (DCB0129/DCB0160), and a named clinical safety officer |

## 3. Why the provider choice does not solve IG

A Microsoft route is **likely reachable** from NHS networks because NHSmail is Microsoft 365 and
`*.microsoft.com` hosts are usually allowed; `*.speech.microsoft.com` and `*.cognitiveservices.azure.com`
are not Microsoft 365 and may still be filtered (test with `/nhs/reach.html`). Reachability is a
network fact. Compliance is a contract and governance fact: a personal Azure subscription is not a
trust processor, whatever network it is reached from. The demo profile proves the technical shape;
the trust supplies the tenant.

## 4. DPIA headings a trust would fill in

1. Nature, scope, context and purpose of processing (voice dictation, lookup, hands-free notes).
2. Lawful basis and the common-law duty of confidentiality (direct care; implied consent for care, explicit for anything else).
3. Data categories: audio (special category if health content), transcripts, user identity, timestamps.
4. Processors and sub-processors: Microsoft (Azure Speech, Azure OpenAI) under the trust's agreement; no other vendors in the Azure-only profile.
5. International transfers: none when the trust pins UK South; Sweden Central is EU and would need the trust's position on EU adequacy.
6. Retention: audio not stored; transcripts per local policy; brain history cleared per session.
7. Security: TLS everywhere, SSO, no API keys in the browser (the widget holds a short-lived token, the server holds the keys), container isolation, patching.
8. Clinical safety: DCB0129 (manufacturer) and DCB0160 (deploying organisation) hazard logs; the assistant does not make clinical decisions, it transcribes and summarises, and every output is reviewed by the clinician before it enters a record.
9. Individual rights, transparency notice, and how a patient would object to voice processing in a consultation.
10. Residual risk and sign-off: IG lead, Caldicott Guardian, Clinical Safety Officer, DPO.

## 5. The honest limits of the demo

- The demo brain is Mahadeva: a personal agent with DeepSeek behind it. It is there so the demo has something to say. Its answers about "this department's protocol" come from Rock's own workspace files, not a trust source.
- Latency over a cascade is 2 to 5 s per turn on a personal server; a trust deployment in-region would be similar. Nobody should promise "real-time".
- No clinical safety case exists. The demo is a technology demonstration, which is why it runs on synthetic cases and says so on screen.
- If a trust wants it, the next step is not more code: it is a sponsor, a DPIA, an Azure subscription in the trust tenant and a pilot scope with no patient identifiers.

## 6. What a skeptic will ask, and the short answers

- *"Does the audio leave the trust?"* In production, no further than the trust's own Azure tenant in UK South. In the demo, yes, to a personal Azure resource in Sweden; that is why the demo is synthetic.
- *"Which AI company hears my patients?"* Microsoft, under the trust's existing agreement, and only in production. OpenAI and Google are not in the Azure-only profile.
- *"What if Azure is down or over quota?"* The service falls back to an on-device model (whisper + Piper) that never leaves the server; quality drops, nothing stops.
- *"Can I turn it off?"* Yes: the service is one container with no stored audio; stopping it ends all processing.
