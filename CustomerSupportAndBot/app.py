#!/usr/bin/env python3
# app.py — ATAL Cloud Customer Support and BOT
# Tabs: Customer Support (Retrieval) + Employee Support (MCQ Student) + EURI Chat
# - Customer Support  uses FAISS + EURI embeddings (text-embedding-3-small)
# - Employee Support reads a CSV (no EURI needed)
# - Chat (EURI) uses only EURI chat-completions API where gpt-4.1-nano is hosted

from pathlib import Path
from typing import Dict, List, Any, Optional
import os, json, re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import faiss
import requests
from dotenv import load_dotenv
from base64 import b64encode
APP_DIR = Path(__file__).parent.resolve()
# --------------- Page config (set ONCE, at top) ---------------
st.set_page_config(
    page_title="ATAL Cloud Customer Support and BOT",
    page_icon="☁️",
    layout="wide",
)

# --------------- Sticky brand header (logo + red bold title) ---------------
LOGO_PATH = os.getenv("ATAL_LOGO_PATH", "assets/atal_cloud_logo.png")

def _find_logo_bytes() -> bytes | None:
    """
    Try multiple locations in priority order:
    1) ATAL_LOGO_PATH (env or st.secrets)
    2) CustomerSupportAndBot/assets/atal_cloud_logo.png  (next to app.py)
    3) repo-root/assets/atal_cloud_logo.png
    """
    # 1) Secrets/env override
    override = os.getenv("ATAL_LOGO_PATH") or (st.secrets.get("ATAL_LOGO_PATH") if "ATAL_LOGO_PATH" in st.secrets else None)
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))

    # 2) next to app.py
    candidates.append(APP_DIR / "assets" / "atal_cloud_logo.png")
    # 3) repo root assets (in case you moved it)
    candidates.append(Path("assets") / "atal_cloud_logo.png")

    for p in candidates:
        try:
            if p.exists() and p.is_file():
                return (APP_DIR / p if not p.is_absolute() and not p.exists() else p).read_bytes()
        except Exception:
            continue
    return None

def render_brand_header():
    logo_b64 = ""
    b = _find_logo_bytes()
    if b:
        logo_b64 = b64encode(b).decode("utf-8")
    # Build the header HTML (use components.html so Streamlit doesn't sanitize it)
    html = f"""
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <style>
      .ac-sticky {{
        position: sticky; top: 0; z-index: 1000;
        background: #fff; border-bottom: 1px solid #eee;
      }}
      .ac-header {{
        max-width: 1200px; margin: 0 auto; padding: 10px 12px;
        display: flex; align-items: center; gap: 12px; justify-content: center;
      }}
      .ac-logo {{ height: 40px; }}
      .ac-title {{ margin:0; color:#dc2626; font-weight:800; font-size:24px; line-height:1.2; }}
      body {{ margin:0; }}
    </style>
  </head>
  <body>
    <div class="ac-sticky">
      <div class="ac-header">
        {f'<img class="ac-logo" alt="ATAL Cloud" src="data:image/png;base64,{logo_b64}" />' if logo_b64 else ''}
        <h2 class="ac-title">ATAL Cloud Customer Support and BOT</h2>
      </div>
    </div>
  </body>
</html>
"""
    components.html(html, height=66, scrolling=False)
    # Tiny debug hint if the logo wasn't found — remove once you see it working
    if not b:
        st.caption("Logo not found. Set ATAL_LOGO_PATH or keep assets/atal_cloud_logo.png next to app.py.")

# Call once before tabs
render_brand_header()

# Call ONCE before creating tabs
render_brand_header()

# --------------- Env / Secrets ---------------
load_dotenv()
# Promote Streamlit Cloud "Secrets" to env vars if not already set
try:
    for k in [
        "EURI_API_KEY","EURON_URL","EURON_MODEL",
        "EURON_CHAT_URL","EURON_CHAT_MODEL","EURON_CHAT_TEMPERATURE"
    ]:
        if k in st.secrets and not os.getenv(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

EURI_API_KEY = os.getenv("EURI_API_KEY", "")
EURON_URL    = os.getenv("EURON_URL", "https://api.euron.one/api/v1/euri/embeddings")
EURON_MODEL  = os.getenv("EURON_MODEL", "text-embedding-3-small")
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {EURI_API_KEY}"}

# Chat (EURI) — pure chat completions
EURON_CHAT_URL         = os.getenv("EURON_CHAT_URL", "https://api.euron.one/api/v1/euri/chat/completions")
EURON_CHAT_MODEL       = os.getenv("EURON_CHAT_MODEL", "gpt-4.1-nano")
EURON_CHAT_TEMPERATURE = float(os.getenv("EURON_CHAT_TEMPERATURE", "0.2"))

# --------------- Paths ---------------
INDEX_PATH = Path("data/index/docs.faiss.index")
META_PATH  = Path("data/index/docs.faiss.metadata.jsonl")
MCQ_DEFAULT_CSV = Path("data/eval/mcqs.csv")

DEFAULT_QUERIES = [
    "refund policy for damaged items",
    "how long is the return window",
    "international shipping customs fees",
    "warranty coverage duration",
    "lost package what to do",
]

# --------------- Shared helpers ---------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def l2_normalize(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    n = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-12
    return arr / n

def embed_texts(texts: List[str]) -> np.ndarray:
    """EURI embeddings -> (N, D) float32 L2-normalized."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    payload = {"input": texts, "model": EURON_MODEL}
    # Using data=json.dumps(...) to match your working flow
    resp = requests.post(EURON_URL, headers=HEADERS, data=json.dumps(payload), timeout=60)
    if resp.status_code != 200:
        try:
            msg = resp.json()
        except Exception:
            msg = resp.text
        raise RuntimeError(f"EURI embed error {resp.status_code}: {msg}")
    data = resp.json().get("data", [])
    if not data:
        return np.zeros((0, 0), dtype=np.float32)
    if all("index" in row for row in data):
        data = sorted(data, key=lambda r: r["index"])
    embs = np.array([row["embedding"] for row in data], dtype=np.float32)
    return l2_normalize(embs)

@st.cache_resource(show_spinner=False)
def load_index_and_meta():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"FAISS index not found: {INDEX_PATH.resolve()}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {META_PATH.resolve()}")
    return faiss.read_index(str(INDEX_PATH)), read_jsonl(META_PATH)

# --------------- Customer Support (Retrieval) table ---------------
def render_custom_table(rows, max_height_px: int = 480, widths: dict = None):
    """
    Precise widths + bottom horizontal scrollbar + dynamic height.
    rank/score/id: narrow; title: medium; snippet: widest (nowrap).
    """
    import html
    if widths is None:
        widths = {
            "rank": "4ch",
            "score": "6ch",
            "id": "10ch",
            "title": "36ch",
            "snippet": "96ch",
        }
    headers = ["rank", "score", "id", "title", "snippet"]

    def _sum_ch(wmap):
        total = 0.0
        for h in headers:
            m = re.fullmatch(r"(\d+(?:\.\d+)?)ch", str(wmap.get(h, "0ch")).strip())
            if m:
                total += float(m.group(1))
        return total
    total_ch = max(120.0, _sum_ch(widths))
    table_min_width = f"{int(round(total_ch))}ch"

    colgroup = "\n".join(
        f'<col class="col-{h}" style="min-width:{widths[h]}; width:{widths[h]}; max-width:{widths[h]};" />'
        for h in headers
    )
    thead = "<tr>" + "".join(f"<th class='{h}'>{html.escape(h)}</th>" for h in headers) + "</tr>"

    body_rows = []
    for i, r in enumerate(rows):
        bg = "#86efac" if (i % 2 == 0) else "#93c5fd"
        tds = []
        for h in headers:
            val = r.get(h, "")
            if h == "score":
                try:
                    val = f"{float(val):.3f}"
                except Exception:
                    val = str(val)
            tds.append(f"<td class='{h}'>{html.escape(str(val))}</td>")
        body_rows.append(f"<tr style=\"background-color:{bg};\">{''.join(tds)}</tr>")

    table_html = f"""
<table class="ztbl">
  <colgroup>{colgroup}</colgroup>
  <thead>{thead}</thead>
  <tbody>{''.join(body_rows)}</tbody>
</table>
"""

    dynamic = 56 + 44 * max(len(rows), 1)
    calc_height = max(220, min(max_height_px, dynamic)) + 12

    html_doc = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{ --header-bg:#0ea5e9; --border:#e5e7eb; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:0; background:white; }}
  .wrap {{
    max-height:{max_height_px}px;
    min-height:180px;
    overflow-y:auto;
    overflow-x:auto;       /* horizontal scrollbar at bottom */
    border:1px solid var(--border);
    border-radius:8px;
    padding:6px;
    background:white;
  }}
  .ztbl {{
    border-collapse: collapse;
    table-layout: fixed;
    width: max-content;
    min-width: {table_min_width};
  }}
  .ztbl th {{
    position: sticky; top: 0;
    background-color: var(--header-bg);
    color: white; text-align: left;
    padding: 6px 8px; border-bottom: 1px solid var(--border);
    white-space: nowrap; z-index:1;
  }}
  .ztbl td {{
    padding: 8px; vertical-align: top; line-height: 1.2;
    word-break: break-word; overflow-wrap: anywhere;
  }}
  .ztbl th.rank,  .ztbl td.rank  {{ text-align: right; white-space: nowrap; }}
  .ztbl th.score, .ztbl td.score {{ text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }}
  .ztbl th.id,    .ztbl td.id    {{ white-space: nowrap; }}
  .ztbl td.snippet {{ white-space: nowrap; }}
</style>
</head>
<body style="margin:0;padding:0;">
  <div class="wrap">
    {table_html}
  </div>
</body>
</html>
"""
    components.html(html_doc, height=calc_height, scrolling=False)

# --------------- Customer Support (Retrieval) view ---------------
def view_retrieval():
    st.subheader("🔎 Customer Support (EURI + FAISS)")
    q_pre = st.selectbox("Predefined query", DEFAULT_QUERIES, index=0, key="ret_q_pre")
    q_custom = st.text_input("…or type your own", "", key="ret_q_custom")
    query = (q_custom or q_pre).strip()
    k = st.slider("Top-K", 3, 15, 5, 1, key="ret_k")
    run = st.button("Search", key="ret_run")

    if not run:
        st.info("Pick or type a query, then click **Search**.")
        return

    if not EURI_API_KEY:
        st.error("Missing **EURI_API_KEY** in .env for Customer Support (Retrieval) mode.")
        return

    try:
        index, metas = load_index_and_meta()
    except Exception as e:
        st.error(str(e))
        return

    try:
        qv = embed_texts([query])
        if qv.size == 0:
            st.warning("Embedding returned empty vector. Check your EURI credentials / model.")
            return
        D, I = index.search(qv.astype(np.float32), int(k))
        rows = []
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), start=1):
            m = metas[idx] if 0 <= idx < len(metas) else {}
            rows.append({
                "rank": rank,
                "score": float(score),
                "id": m.get("id",""),
                "title": m.get("title","Untitled"),
                "snippet": (m.get("text","") or "").replace("\n"," "),
            })
        st.markdown("### Results")
        render_custom_table(rows, max_height_px=480, widths={
            "rank":"4ch","score":"6ch","id":"10ch","title":"36ch","snippet":"96ch"
        })
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        df = pd.DataFrame(rows, columns=["rank","score","id","title","snippet"])
        st.download_button(
            "Download CSV",
            df.to_csv(index=False).encode("utf-8"),
            file_name="search_results.csv",
            mime="text/csv",
            key="ret_dl",
        )
    except Exception as e:
        st.error(str(e))

# --------------- Employee Support (MCQ Student) ---------------
REQUIRED_MCQ = {"question","A","B","C","D","correct"}

def load_mcq_csv() -> pd.DataFrame:
    file = st.file_uploader("Upload MCQ CSV", type=["csv"], key="mcq_upload")
    if file:
        df = pd.read_csv(file)
        st.success("Loaded uploaded CSV.")
        return df
    if MCQ_DEFAULT_CSV.exists():
        st.info(f"Using default: {MCQ_DEFAULT_CSV}")
        return pd.read_csv(MCQ_DEFAULT_CSV)
    st.warning("No CSV found. Upload one with columns: question,A,B,C,D,correct")
    return pd.DataFrame()

def validate_mcq_df(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return "Empty or missing CSV."
    missing = REQUIRED_MCQ - set(df.columns)
    if missing:
        return f"CSV missing columns: {', '.join(sorted(missing))}"
    if not df["correct"].isin(["A","B","C","D"]).all():
        return "Column 'correct' must contain only A/B/C/D."
    return None

def view_mcq_student():
    st.subheader("🧑‍💼 Employee Support")
    df = load_mcq_csv()
    err = validate_mcq_df(df)
    if err:
        st.error(err)
        return
    st.success(f"Loaded {len(df)} questions.")

    selections: Dict[int, str] = {}
    with st.form("quiz_form", clear_on_submit=False):
        for i, row in df.iterrows():
            st.markdown(f"**Q{i+1}.** {row['question']}")
            prev = st.session_state.get(f"quiz_sel_{i}", "")
            opts = ["", "A", "B", "C", "D"]
            def fmt(opt, r=row):
                if opt == "": return "Select an option…"
                return f"{opt}. {r[opt]}"
            choice = st.selectbox(
                label=f"Question {i+1}",
                options=opts,
                index=opts.index(prev) if prev in opts else 0,
                format_func=lambda opt, r=row: fmt(opt, r),
                key=f"quiz_sel_{i}",
            )
            selections[i] = choice
            st.markdown("---")

        col1, col2 = st.columns([1, 1])
        submitted = col1.form_submit_button("Check answers", type="primary")
        reset     = col2.form_submit_button("Reset")

    if reset:
        for i in range(len(df)):
            st.session_state.pop(f"quiz_sel_{i}", None)
        st.success("Selections cleared.")
        return

    if not submitted:
        return

    missing = [i for i, ch in selections.items() if ch not in {"A","B","C","D"}]
    if missing:
        miss = ", ".join(f"Q{n+1}" for n in missing)
        st.warning(f"Please answer all questions before checking. Missing: {miss}")
        return

    correct_count = sum(1 for i, row in df.iterrows() if selections[i] == row["correct"])
    total = len(df)
    wrong = total - correct_count
    pct = 100.0 * correct_count / total if total else 0.0

    st.markdown(f"### Results: **{correct_count}/{total} correct**, **{wrong} wrong** — **{pct:.0f}%**")
    st.markdown("---")

    for i, row in df.iterrows():
        user = selections[i]
        correct = row["correct"]
        correct_text = row[correct]
        why = row.get("why", "")
        title = row.get("title", "")
        doc_id = row.get("id", "")
        provenance = f"[{title}] (id={doc_id})" if title or doc_id else ""

        st.markdown(f"**Q{i+1}.** {row['question']}")
        if user == correct:
            msg = f"✅ Correct — {correct}. {correct_text}"
            if isinstance(why, str) and why.strip(): msg += f"\n\n**Why:** {why}"
            if provenance: msg += f"\n\n*Source:* {provenance}"
            st.success(msg)
        else:
            chosen_text = row[user]
            msg = f"❌ Your answer: {user}. {chosen_text}\n\n**Correct:** {correct}. {correct_text}"
            if isinstance(why, str) and why.strip(): msg += f"\n\n**Why:** {why}"
            if provenance: msg += f"\n\n*Source:* {provenance}"
            st.error(msg)
        st.markdown("---")

    if st.button("Retake quiz", key="mcq_retake"):
        for i in range(len(df)):
            st.session_state.pop(f"quiz_sel_{i}", None)
        st.rerun()

# --------------- Chat BOT (EURI) ---------------
def euri_chat(messages: List[Dict[str, str]],
              model: Optional[str] = None,
              temperature: Optional[float] = None,
              max_tokens: int = 400) -> str:
    """Pure EURI chat completions."""
    mdl = (model or EURON_CHAT_MODEL or "gpt-4.1-nano").strip()
    temp = EURON_CHAT_TEMPERATURE if temperature is None else float(temperature)
    payload = {
        "model": mdl,
        "messages": messages,
        "temperature": temp,
        "max_tokens": max_tokens,
    }
    # IMPORTANT: use json= so server sees the `model` field
    resp = requests.post(EURON_CHAT_URL, headers=HEADERS, json=payload, timeout=90)
    if resp.status_code >= 400:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"EURI chat error {resp.status_code}: {detail}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def view_chat_euri_only():
    st.subheader("💬 Chat BOT (EURI)")
    if not EURI_API_KEY:
        st.error("Missing **EURI_API_KEY** in .env for Chat mode.")
        return

    st.caption(f"Using model: `{EURON_CHAT_MODEL}` at `{EURON_CHAT_URL}`")

    if "chat_euri_messages" not in st.session_state:
        st.session_state.chat_euri_messages = [
            {"role": "system", "content": "You are ATAL Cloud’s helpful assistant. Be concise and accurate."}
        ]

    colA, colB, colC = st.columns([1,1,1])
    temperature = colA.slider("Temperature", 0.0, 1.0, float(EURON_CHAT_TEMPERATURE), 0.05, key="chat_temp")
    max_tokens  = colB.slider("Max tokens", 64, 2048, 400, 16, key="chat_maxtok")
    if colC.button("Clear chat"):
        st.session_state.chat_euri_messages = [
            {"role": "system", "content": st.session_state.chat_euri_messages[0]["content"]}
        ]
        st.rerun()

    for m in st.session_state.chat_euri_messages:
        if m["role"] == "system":
            st.info(f"**System:** {m['content']}")
        elif m["role"] == "user":
            st.chat_message("user").markdown(m["content"])
        elif m["role"] == "assistant":
            st.chat_message("assistant").markdown(m["content"])

    user_input = st.chat_input("Type your message…")
    if not user_input:
        return

    st.session_state.chat_euri_messages.append({"role": "user", "content": user_input})
    try:
        assistant_text = euri_chat(
            messages=st.session_state.chat_euri_messages,
            model=EURON_CHAT_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        st.error(str(e))
        return

    st.session_state.chat_euri_messages.append({"role": "assistant", "content": assistant_text})
    st.chat_message("user").markdown(user_input)
    st.chat_message("assistant").markdown(assistant_text)

# --------------- Tabs ---------------
tab1, tab2, tab3 = st.tabs(["🔎 Customer Support", "📝 Employee Support", "💬 Chat BOT (EURI)"])
with tab1:
    view_retrieval()
with tab2:
    view_mcq_student()
with tab3:
    view_chat_euri_only()
