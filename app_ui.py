import streamlit as st
import ollama
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


# ---------- Data Models ----------
class ArticleRequest(BaseModel):
    topic: str
    keywords: List[str] = Field(default_factory=list)
    audience: str = "General readers"
    tone: str = "Informative and engaging"
    language: str = "English"
    target_words: int = 1000
    include_outline: bool = True
    include_seo: bool = True
    include_references: bool = False
    reading_level: str = "Easy to read (Grade 8–10)"
    extra_instructions: str = ""


class GenConfig(BaseModel):
    model: str = "phi3:3.8b"   # lightweight, good on CPU
    temperature: float = 0.7
    seed: Optional[int] = None


# ---------- Prompt Builder ----------
def build_prompt(req: ArticleRequest) -> str:
    kw_str = ", ".join(req.keywords) if req.keywords else "N/A"
    outline_block = (
        "Include a clear outline before the main content.\n"
        "- H1 title\n- 5–8 H2 sections with concise H3s where helpful\n"
    ) if req.include_outline else "Skip the outline; start directly with the article.\n"

    seo_block = (
        "After the article, output an SEO block with:\n"
        "- SEO Title (≤ 60 chars)\n- Meta Description (≤ 155 chars)\n"
        "- 8–12 SEO Keywords (comma-separated)\n"
    ) if req.include_seo else ""

    refs_block = "Add a short 'References' section with 3–5 plausible sources (titles only, no links).\n" if req.include_references else ""

    return f"""
You are a senior content strategist and expert writer.

Goal
-----
Write a comprehensive, well-structured article in {req.language} for the topic:
"{req.topic}"

Audience: {req.audience}
Tone/Voice: {req.tone}
Reading Level: {req.reading_level}
Target Length: ~{req.target_words} words
Primary Keywords: {kw_str}

Structure & Style
-----------------
{outline_block}
- Use Markdown formatting.
- Use scannable headings, short paragraphs, and bullet lists where helpful.
- Provide concrete examples and actionable tips.
- Avoid fluff; keep it factual and clear.
- Natural keyword usage; avoid keyword stuffing.

Content Requirements
--------------------
- Strong hook and crisp thesis in intro.
- Each H2 should fully cover one major point.
- Add comparisons, pros/cons, pitfalls, or checklists where useful.
- Conclude with practical summary or next steps.

Output Order
------------
1) Outline (if requested)
2) Full article in Markdown
{('3) SEO Block\n' if req.include_seo else '')}{('4) References\n' if req.include_references else '')}

Extra Instructions
------------------
{req.extra_instructions}
""".strip()


# ---------- Ollama Call ----------
def generate_article(prompt: str, cfg: GenConfig) -> str:
    messages = [
        {"role": "system", "content": "Write clear, accurate, reader-friendly content. Prefer Markdown."},
        {"role": "user", "content": prompt},
    ]

    stream = ollama.chat(
        model=cfg.model,
        messages=messages,
        stream=True,
        options={
            "temperature": cfg.temperature,
            **({"seed": cfg.seed} if cfg.seed is not None else {}),
        },
    )

    chunks = []
    for chunk in stream:
        content = chunk.get("message", {}).get("content", "")
        if content:
            chunks.append(content)
    return "".join(chunks)


# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI Article Generator", page_icon="📝", layout="wide")
st.title("📝 AI Article Generator")
st.caption("Generate SEO-ready, long-form articles locally using Ollama.")

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.text_input("Ollama model", value="phi3:3.8b")
    temperature = st.slider("Creativity (temperature)", 0.0, 1.5, 0.7, 0.1)
    use_seed = st.checkbox("Use fixed seed", value=False)
    seed = st.number_input("Seed", value=42, step=1, disabled=not use_seed)
    target_words = st.slider("Target length (words)", 300, 3000, 1000, 50)

st.subheader("✍️ Article Inputs")
topic = st.text_input("Topic", placeholder="e.g., How to Build an AI Article Generator with Streamlit + Ollama")
keywords_raw = st.text_input("Keywords (comma-separated)", placeholder="ai article generator, streamlit, ollama, seo")
audience = st.text_input("Audience", value="General readers")
tone = st.text_input("Tone", value="Informative and engaging")
reading_level = st.selectbox("Reading level", ["Easy to read (Grade 8–10)", "Intermediate (Grade 11–12)", "Advanced/Technical"])
language = st.text_input("Language", value="English")
include_outline = st.checkbox("Include outline", value=True)
include_seo = st.checkbox("Include SEO block", value=True)
include_refs = st.checkbox("Include references", value=False)
extra = st.text_area("Extra instructions", placeholder="e.g., Add examples, include checklist")

if st.button("🚀 Generate Article"):
    if not topic.strip():
        st.error("Please enter a topic.")
        st.stop()

    req = ArticleRequest(
        topic=topic.strip(),
        keywords=[k.strip() for k in keywords_raw.split(",") if k.strip()],
        audience=audience.strip(),
        tone=tone.strip(),
        language=language.strip(),
        target_words=int(target_words),
        include_outline=include_outline,
        include_seo=include_seo,
        include_references=include_refs,
        reading_level=reading_level,
        extra_instructions=extra.strip(),
    )

    cfg = GenConfig(
        model=model.strip(),
        temperature=float(temperature),
        seed=int(seed) if use_seed else None,
    )

    with st.status("Generating article…", state="running"):
        prompt = build_prompt(req)
        content = generate_article(prompt, cfg)

    st.success("✅ Article generated!")
    st.markdown("---")
    st.subheader("📄 Draft")
    st.markdown(content)

    # Download button
    fname = f"article_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    st.download_button(
        "⬇️ Download Markdown",
        data=content.encode("utf-8"),
        file_name=fname,
        mime="text/markdown",
    )
