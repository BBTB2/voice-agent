"""prof-snarky as a brain (stage 9, 20 Oct 2026).

prof-snarky (BBTB2/prof-snarky, www.prof-snarky.com) answers one question at a time through its Next.js route
``POST /api/query`` {query, mode, persona?, force} -> {answer, citations, figures, creditsUsed, chatId, ...}, behind a
Clerk sign-in, charging the signed-in person one credit (or a free search) a question. The voice service never
holds a prof-snarky credential of its own: the page hands over the person's Clerk session token for each turn and
this adapter forwards it as ``Authorization: Bearer``, so credits, the approval gate, the rate limits and the chat
history stay exactly the site's. No history is sent: the site keeps the conversation itself (threads, 21 Oct 2026); the page's thread id rides along as ``threadId`` and comes back in ``meta``.

The answer comes back as markdown with ``[^N]`` footnotes, ``(Fig. F1)`` markers, ``![F1: ...](url)`` figure blocks
and a ``**Sources:**`` footer; ``spoken_answer`` turns it into something a voice should say and ``meta`` carries the
raw answer, citations and ids to the page so the chat can show the usual message.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import AsyncIterator

import httpx

from ..brain import BaseBrain, BrainAuthError, BrainError

LOG = logging.getLogger("voice.brain.prof-snarky")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)

_FIG_BLOCK = re.compile(r"!\[F\d+(?::[^\]]*)?\]\([^)]*\)\s*\n?\*F\d+\.[^*]*\*\s*", re.S)
_FIG_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_FIG_MARK = re.compile(r"\s*\(Fig\.?\s*F?\d+\)")
_FOOTNOTE = re.compile(r"\[\^\d+\]")
_SOURCES = re.compile(r"^\s*\**Sources:?\**.*$", re.M | re.I)
_FOOTNOTE_DEF = re.compile(r"^\[\^\d+\]:.*$", re.M)
_HEADING = re.compile(r"^\s*#{1,6}\s*", re.M)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)


def spoken_answer(markdown: str) -> str:
    """The site's markdown answer as spoken prose: no figures, footnote marks, sources footer, headings or table rows.
    Bullets become sentences. The sentence splitter downstream strips the remaining emphasis marks."""
    t = markdown or ""
    t = _FIG_BLOCK.sub("", t)
    t = _FIG_IMG.sub("", t)
    t = _FIG_MARK.sub("", t)
    t = _FOOTNOTE_DEF.sub("", t)
    t = _FOOTNOTE.sub("", t)
    t = _SOURCES.sub("", t)
    t = _TABLE_ROW.sub("", t)
    t = _HEADING.sub("", t)
    t = _BULLET.sub("", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{2,}", "\n", t)
    lines = []
    for line in t.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line[-1] not in ".!?…:;":
            line += "."
        lines.append(line)
    return " ".join(lines).strip()


class ProfSnarkyBrain(BaseBrain):
    kind = "prof-snarky"
    needs_auth = True

    def __init__(self, cfg: dict):
        self.url = str(cfg.get("url") or "").strip()
        self.persona = str(cfg.get("persona") or "").strip() or None      # empty = the person's default on the site
        self.mode = str(cfg.get("mode") or "quick")
        self.timeout = float(cfg.get("timeout_seconds", 130))
        self.max_chars = int(cfg.get("max_query_chars", 2000))             # the route's zod limit
        self.speak_replay = bool(cfg.get("speak_similar", True))           # a stored similar answer is spoken, free
        self.say_sources = bool(cfg.get("say_sources", True))
        self.label = "prof-snarky" + (f" ({self.persona})" if self.persona else "")

    def _client(self) -> httpx.AsyncClient:
        # One client per turn: a turn is seconds apart from the next, the site sits behind Cloudflare (idle
        # keep-alives get cut anyway), and a pooled connection must not outlive the event loop it was made on.
        return httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=15.0), follow_redirects=False,
                                 headers={"User-Agent": "voice-agent (prof-snarky brain)", "Accept": "application/json"})

    def configured(self) -> bool:
        return bool(self.url)

    async def stream(self, session: str, user_text: str, history: list[dict] | None = None,
                     first_turn: bool = False, images: list[tuple[str, bytes]] | None = None,
                     auth: str | None = None, meta: dict | None = None) -> AsyncIterator[str]:
        meta = meta if meta is not None else {}
        if not auth:
            raise BrainAuthError("You are not signed in to prof-snarky on this page, so I cannot ask the Prof. Sign in and try again.")
        query = (user_text or "").strip()[: self.max_chars]
        body: dict = {"query": query, "mode": self.mode, "force": False}
        if meta.get("persona") or self.persona:
            body["persona"] = meta.get("persona") or self.persona
        thread_reset = False
        tid = meta.get("thread_id")
        if tid:
            if isinstance(tid, str) and _UUID.match(tid):
                body["threadId"] = tid
            else:
                LOG.info("dropping malformed thread_id")
                meta.pop("thread_id", None)
                thread_reset = True                  # the page shows its reset notice
        try:
            async with self._client() as client:
                r = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {auth}"})
        except httpx.HTTPError as e:
            raise BrainError(f"prof-snarky unreachable: {e}") from e
        if r.status_code in (301, 302, 303, 307, 308) or r.status_code == 401:
            raise BrainAuthError("prof-snarky did not accept your sign-in. Reload the page, sign in, and try again.")
        if r.status_code == 400:
            raise BrainAuthError("prof-snarky could not read that question. Tap New question and ask again.")
        if r.status_code == 402:
            raise BrainAuthError("You have no credits left on prof-snarky. Top up on the credits page and ask again.")
        if r.status_code == 429:
            raise BrainAuthError("prof-snarky says slow down: too many questions in a minute. Wait a moment and ask again.")
        if r.status_code >= 400:
            detail = r.text[:200].replace("\n", " ")
            raise BrainError(f"prof-snarky HTTP {r.status_code}: {detail}")
        try:
            data = r.json()
        except ValueError as e:
            raise BrainError("prof-snarky answered with something that is not JSON (a Cloudflare challenge page?)") from e
        if data.get("similar"):
            sim = data["similar"]
            meta.update({"similar": True, "chat_id": sim.get("id"), "answer": sim.get("answer", ""), "citations": sim.get("citations") or [],
                         "credits_used": 0, "persona": sim.get("persona"), "asked": sim.get("query")})
            meta.pop("thread_id", None)    # a replay starts a new question; the page clears its thread
            if not self.speak_replay:
                raise BrainError("prof-snarky offered a stored answer; replay is switched off")
            when = _when(sim.get("createdAt"))
            yield f"You asked something very similar{when}, so here is that answer again, free. "
            yield spoken_answer(sim.get("answer", ""))
            return
        answer = str(data.get("answer") or "")
        citations = data.get("citations") or []
        meta.update({"similar": False, "chat_id": data.get("chatId"), "answer": answer, "citations": citations,
                     "figures": data.get("figures") or [], "credits_used": data.get("creditsUsed"),
                     "free_search_used": data.get("freeSearchUsed"), "persona": data.get("persona"),
                     "archetype": data.get("archetype") or "", "thread_reset": thread_reset or bool(data.get("threadReset")),
                     "free_followups_left": data.get("freeFollowupsLeft")})
        if data.get("threadId"):
            meta["thread_id"] = data["threadId"]
        else:
            meta.pop("thread_id", None)
        spoken = spoken_answer(answer)
        if not spoken:
            raise BrainError("prof-snarky returned an empty answer")
        yield spoken
        if self.say_sources and citations:
            n = len(citations)
            yield f" {'One source is' if n == 1 else f'{n} sources are'} listed in the chat."

    async def close(self) -> None:
        return None


def _when(iso: str | None) -> str:
    if not iso:
        return ""
    try:
        d = datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        return " on " + d.strftime("%d %B").lstrip("0")
    except ValueError:
        return ""
