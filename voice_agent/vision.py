"""The camera leg (stage 10): what the vision model is asked, and the text the brain receives.

The voice service never answers from the picture itself. A vision model turns the clip and the still into a
careful written description aimed at the person's question; that description, and the still, go to the brain
(Mahadeva), which answers with its own knowledge, tools and memory. Accuracy over speed: the model is told to
say what it cannot make out instead of guessing.
"""
from __future__ import annotations

PROMPT = """You are the eyes of a voice assistant. The person is holding a phone camera at something and
asking about it; you see a short clip from the hold and one sharp still taken as they released the button.
The still is the clearest frame: use it for anything small or printed; use the clip for movement, context and
other angles. Write a description for the assistant, which cannot see the picture, so that it can answer the
question below well.

Rules:
- Describe exactly what is visible: objects, text (transcribe every legible word, number and price exactly as
  printed, with currency symbols), colours, markings, size cues, setting, anything moving.
- For a living thing give the identifying features you can actually see (shape, colouring, markings, size
  relative to its surroundings) and your best identification with a confidence (sure / likely / unsure).
- If something the question needs is unreadable, out of frame, blurred or too dark, say so plainly; never
  invent text, prices or species. If you can see almost nothing useful, say that in one sentence.
- Plain prose, no markdown, no headings, under 180 words. Do not answer the question yourself beyond the
  description; the assistant answers the person.

The person's question (spoken, may be rough): {question}"""


def prompt_for(question: str) -> str:
    q = (question or "").strip() or "(no words were heard; describe what is in front of the camera)"
    return PROMPT.format(question=q)


def brain_text(question: str, description: str | None, provider: str | None, note: str | None = None) -> str:
    """The user turn the brain receives: the words first, then what the camera showed."""
    head = (question or "").strip() or "(I pointed the camera at this without saying anything.)"
    if description:
        body = (f"[Camera, live, described by {provider or 'the vision model'}: {description.strip()}]\n"
                "Answer about what the camera shows, in short spoken sentences. If the description says something "
                "could not be seen or read, say so and ask me to hold the camera closer or steadier rather than guess.")
    else:
        body = (f"[Camera: {note or 'nothing usable arrived from the camera this time'}.] "
                "Tell me briefly that you could not see, and ask me to try again holding the phone steady.")
    return f"{head}\n\n{body}"
