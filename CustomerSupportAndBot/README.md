# ATAL Cloud Customer Support and Bot

This Streamlit app demonstrates three practical workflows:

- 🔎 **Customer Support (Retrieval)** — Search Support help articles/policies using **EURI embeddings** (`text-embedding-3-small`) + **FAISS**.
- 🧑‍💼 **Employee Support** — Take a multiple-choice quiz loaded from a CSV (no EURI required).
- 💬 **EURI Chat** — Chat with an OpenAI-compatible model **hosted on EURI** (default: **`gpt-4.1-nano`**).

Works locally with a tiny **public sample dataset**; plug in your own content to turn it into a real support assistant.

## 🚀 Quick Start

> Python **3.10+** recommended. 

1) **Install**
```bash
pip install -r requirements.txt


2) **Environment**
# copy the example, then put your real key in .env 

EURI_API_KEY="euri-REPLACE_ME"

# Embeddings (Customer Support / Retrieval)
EURON_URL="https://api.euron.one/api/v1/euri/embeddings"
EURON_MODEL="text-embedding-3-small"

# Chat (EURI Chat)
EURON_CHAT_URL="https://api.euron.one/api/v1/euri/chat/completions"
EURON_CHAT_MODEL="gpt-4.1-nano"
EURON_CHAT_TEMPERATURE="0.2"

3) Create tiny public dataset (first time only)

python create_public_sample.py


4) Build a local FAISS index (first time only)

python build_sample_index.py


5) Run the app

streamlit run app.py

Open the URL Streamlit prints (usually http://localhost:8501).
.....................................................................
🧩 Tabs & Features
🔎 Customer Support (Retrieval)

Input: Natural-language question on Help/policy
Engine: FAISS similarity search (Inner Product).
Output: Table with rank, score, doc id, title, snippet, plus CSV download.

Data files
Index: data/index/docs.faiss.index
Metadata: data/index/docs.faiss.metadata.jsonl (stores id, title, text)
Rebuild the index whenever you update data/sample/passages.jsonl (or your own dataset).

..........................

Employee Support

Input CSV columns: question,A,B,C,D,correct 
Flow: Select an answer for each question → Check answers → score + per-question feedback.
Use your own CSV: Upload from the tab. Default is data/eval/mcqs.csv.

.........................................

EURI Chat

Input: your message in the chat box.
Engine: EURI chat completions at ${EURON_CHAT_URL} using ${EURON_CHAT_MODEL} (default gpt-4.1-nano).
Controls: temperature, max tokens, clear chat.

This tab does not use FAISS; it’s a pure LLM chat hosted by EURI.

.........................................

Test Ideas

Customer Support:

“refund policy for damaged items”
“how long is the return window”
“international shipping customs fees”
“promo code stacking allowed?”
Employee Support: upload your company MCQs or use default.


EURI Chat: general or support-style questions.