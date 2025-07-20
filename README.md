# 🍕 SliceSense AI / Doughbot

**An AI-powered pizza restaurant review assistant built with Ollama, LangChain, and Chroma.**

> *Copy-paste this entire markdown into `README.md` in your project root.*

---

## 📚 Table of Contents

* [Why SliceSense AI?](#-why-slicesense-ai)
* [Quick Start](#-quick-start)
* [Project Overview](#-project-overview)
* [Features](#-features)
* [Architecture](#-architecture)
* [Prerequisites](#-prerequisites)
* [Installation](#-installation)
* [Example Review Data CSV](#-example-review-data-csv)
* [Configuration](#-configuration)
* [Run the App](#-run-the-app)
* [Usage Flow](#-usage-flow)
* [Code Walkthrough](#-code-walkthrough)

  * [`vector.py`](#vectorpy)
  * [`main.py`](#mainpy)
* [Extending the System](#-extending-the-system)
* [Troubleshooting](#-troubleshooting)
* [Project Structure](#-project-structure)
* [Roadmap Ideas](#-roadmap-ideas)
* [License](#-license)
* [Acknowledgments](#-acknowledgments)

---

## 💡 Why SliceSense AI?

Customers (and staff!) constantly ask things like *“How’s the service?”* or *“Which pizza is most recommended?”* You already have the answers hidden in review data. SliceSense AI turns your historical **customer reviews into a conversational knowledge base** that can answer new, natural-language questions in real time.

---

## ⚡ Quick Start

```bash
# 1. Clone or create a project folder & enter it
mkdir slicesense-ai && cd slicesense-ai

# 2. Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS / Linux
source .venv/bin/activate

# 3. Install Python deps
pip install --upgrade pip
pip install langchain-ollama langchain-chroma langchain-core pandas

# 4. Install Ollama (https://ollama.com) then pull required models
ollama pull mxbai-embed-large
ollama pull llama3.2

# 5. Add your review CSV (see example below)
# 6. Run the app
python main.py
```

---

## 🧭 Project Overview

SliceSense AI is a **retrieval-augmented generation (RAG)** command‑line tool that:

1. Loads pizza restaurant reviews from CSV.
2. Embeds them using **Ollama** (local models; private, fast, offline‑friendly).
3. Stores/reuses embeddings in a **persistent Chroma vector store**.
4. At query time, pulls the **top‑k semantically similar reviews**.
5. Uses a **prompt + LLM (llama3.2)** to craft an answer grounded in those reviews.

---

## ✨ Features

* **Semantic Retrieval** – Finds meaningfully related reviews, not just keyword matches.
* **Grounded Answers** – Model responds based on retrieved customer experiences.
* **Persistent Embeddings** – Only embed once; reuse across runs.
* **Interactive CLI** – Ask any question: service quality, best toppings, wait times, etc.
* **Modular** – Swap models, change vector store path, tune retriever settings.

---

## 🏗 Architecture

```
          ┌────────────────────┐
          │ realistic_restaurant_reviews.csv │
          └────────────┬────────┘
                       │  load (pandas)
                       ▼
                ┌───────────────┐    embed (OllamaEmbeddings / mxbai-embed-large)
                │   Documents   │──────────────────────────────────────────────┐
                └──────┬────────┘                                              │
                       │ add                                                     │
                       ▼                                                         │
                ┌──────────────────────────────┐                                 │
                │  Chroma Vector Store (disk)  │<──────────── persistent ───────┘
                └────────────┬────────────────┘
                             │ retriever(top_k=5)
                             ▼
User Q ───► main.py ──► Prompt Template ──► LLM (llama3.2 via Ollama) ──► Answer
```

---

## ✅ Prerequisites

* **Python**: 3.8+
* **Ollama** installed locally (macOS, Linux, Windows WSL / native) – download from the official site.
* Models downloaded locally:

  * `mxbai-embed-large` (embeddings)
  * `llama3.2` (generation)
* Python packages: `langchain-ollama`, `langchain-chroma`, `langchain-core`, `pandas`.

> *Tip:* You can substitute other embedding or LLM models that Ollama supports by editing constants in the code.

---

## 📥 Installation

1. **Clone / create project folder**.
2. **Create venv & install deps** (see Quick Start above).
3. **Install & run Ollama daemon** (it should run in the background; most installers do this automatically).
4. **Pull models**:

   ```bash
   ollama pull mxbai-embed-large
   ollama pull llama3.2
   ```
5. **Add your CSV** (see below).

---

## 🧾 Example Review Data CSV

Minimum required columns: `Title`, `Review`, `Rating`, `Date`.

> Save as `realistic_restaurant_reviews.csv` in the project root.

```csv
Title,Review,Rating,Date
"Great Pizza, Slow Service","The pizza was delicious, but we waited a long time for our order.",4,2023-01-15
"Best Pepperoni Ever!","Absolutely loved the pepperoni pizza here. Highly recommend.",5,2023-01-20
"Disappointing Experience","The crust was soggy and the toppings were sparse.",2,2023-02-01
```

You can include many more rows; the system scales with your data.

---

## ⚙ Configuration

Basic settings are defined in code (feel free to parameterize later):

| Setting         | Where       | Default                 | Description                               |
| --------------- | ----------- | ----------------------- | ----------------------------------------- |
| Embedding Model | `vector.py` | `mxbai-embed-large`     | Used to turn reviews into vectors.        |
| LLM Model       | `main.py`   | `llama3.2`              | Used to generate answers.                 |
| Vector DB Path  | `vector.py` | `./chrome_langchain_db` | Local persistent Chroma store.            |
| Top K           | `vector.py` | 5                       | Number of reviews retrieved per question. |

---

## ▶ Run the App

```bash
python main.py
```

Sample interaction:

```
-------------------------------
Ask your question (q to quit): How was the service?
> The service experience has been mixed. Several reviewers praised the staff friendliness, but a few mentioned long wait times on busy nights... (example output)
```

Type `q` to exit.

---

## 🔁 Usage Flow

1. First run:

   * CSV is loaded.
   * Reviews embedded & stored in Chroma.
   * DB persists to disk.
2. Later runs:

   * Existing Chroma DB is reused; no re-embedding unless CSV changes.
3. Query loop:

   * User asks a question.
   * Retriever pulls top‑k semantically similar reviews.
   * Prompt + LLM answer with grounded summary.

---

## 🔍 Code Walkthrough

### `vector.py`

Handles **data loading**, **embedding**, **Chroma persistence**, and returns a **retriever**.

> Copy this file into your project as `vector.py`.

```python
# vector.py
"""
Vector store builder & retriever factory for SliceSense AI.

Loads review CSV, builds (or reuses) a persistent Chroma DB backed by
Ollama embeddings, and exposes a LangChain retriever.
"""

import os
import pandas as pd
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# ------------------ Config ------------------ #
CSV_PATH = "realistic_restaurant_reviews.csv"
CHROMA_DIR = "./chrome_langchain_db"  # persistent storage directory
EMBED_MODEL = "mxbai-embed-large"     # pulled via: ollama pull mxbai-embed-large
TOP_K = 5                              # number of docs to retrieve per query


def load_reviews(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load review data from CSV; ensure required columns exist."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = {"Title", "Review", "Rating", "Date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")
    # Drop rows with empty Review text
    df = df.dropna(subset=["Review"]).reset_index(drop=True)
    return df


def build_documents(df: pd.DataFrame):
    """Convert review rows into LangChain Document objects."""
    docs = []
    for _, row in df.iterrows():
        metadata = {
            "title": row.get("Title", ""),
            "rating": int(row.get("Rating", 0)) if not pd.isna(row.get("Rating")) else None,
            "date": str(row.get("Date", "")),
        }
        # Use both title + review in page_content for richer embedding context
        text_parts = [str(row.get("Title", "")), str(row.get("Review", ""))]
        text = ". ".join([p for p in text_parts if p])
        docs.append(Document(page_content=text, metadata=metadata))
    return docs


def get_vector_store(persist_directory: str = CHROMA_DIR):
    """Return (vector_store, retriever) building Chroma & embeddings if needed."""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # If directory exists and contains a Chroma DB, load it; else create new.
    # Chroma auto-detects existing DB in the directory.
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )

    # Detect empty DB (no collections) by counting.
    # Chroma LangChain wrapper does not always expose a simple length; use collection count.
    # We'll insert only if empty.
    if not vector_store._client.get_or_create_collection(vector_store._collection_name).count():  # type: ignore[attr-defined]
        print("[SliceSense] Building Chroma DB from CSV...")
        df = load_reviews()
        docs = build_documents(df)
        vector_store.add_documents(docs)
        vector_store.persist()
        print(f"[SliceSense] Indexed {len(docs)} reviews.")
    else:
        print("[SliceSense] Using existing Chroma DB (no re-embedding).")

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    return vector_store, retriever


# If run directly, do a quick smoke test
if __name__ == "__main__":
    _, r = get_vector_store()
    results = r.get_relevant_documents("How was the service?")
    print(f"Retrieved {len(results)} docs:")
    for d in results:
        print("-", d.metadata.get("title"), "| rating:", d.metadata.get("rating"))
```

---

### `main.py`

Provides the **interactive CLI**, builds the **prompt**, and generates answers from retrieved reviews.

> Copy this file into your project as `main.py`.

```python
# main.py
"""
SliceSense AI interactive Q&A CLI over pizza restaurant reviews.
"""

from typing import List
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from vector import get_vector_store  # local import

# ------------------ Config ------------------ #
LLM_MODEL = "llama3.2"  # pulled via: ollama pull llama3.2


# ------------------ Prompt ------------------ #
SYSTEM_PROMPT = (
    "You are SliceSense AI, an expert assistant that answers questions about a pizza "
    "restaurant using actual customer reviews. Your job: summarize what the reviews say, "
    "identify themes, and be honest about mixed feedback. If a user asks a question that "
    "is not covered by the reviews, say so and suggest asking the restaurant directly. "
    "Always ground your response in the retrieved reviews; do not fabricate details."
)

USER_PROMPT_TEMPLATE = (
    "User question: {question}\n\n"
    "Here are relevant customer reviews (metadata may include title, rating, date):\n\n{context}\n\n"
    "Provide a helpful, concise answer grounded in these reviews. If opinions differ, summarize both sides."
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT_TEMPLATE),
])


# ------------------ Helpers ------------------ #

def format_docs(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        title = meta.get("title", "Untitled")
        rating = meta.get("rating", "?")
        date = meta.get("date", "?")
        lines.append(f"[{i}] {title} (Rating: {rating}, Date: {date})\n{d.page_content}")
    return "\n\n".join(lines)


# ------------------ Main Loop ------------------ #

def main():
    # Build / load vector store + retriever
    _, retriever = get_vector_store()

    # Init LLM
    llm = OllamaLLM(model=LLM_MODEL)

    # Build the chain: prompt -> LLM
    chain = prompt | llm

    print("-" * 31)
    print("Ask your question (q to quit):", end=" ")

    while True:
        user_q = input().strip()
        if user_q.lower() in {"q", "quit", "exit"}:
            print("Goodbye!")
            break
        if not user_q:
            print("Please enter a question (or q to quit):", end=" ")
            continue

        # Retrieve docs
        docs = retriever.get_relevant_documents(user_q)
        context = format_docs(docs)

        # Invoke chain
        response = chain.invoke({"question": user_q, "context": context})

        print("\n--- Answer ---")
        print(response)
        print("\n-------------------------------")
        print("Ask another question (q to quit):", end=" ")


if __name__ == "__main__":
    main()
```

---

## 🧪 Quick Local Test (No CLI)

Want to test retrieval + model programmatically?

```python
from vector import get_vector_store
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

_, retriever = get_vector_store()
llm = OllamaLLM(model="llama3.2")

prompt = ChatPromptTemplate.from_template("Reviews:\n{context}\n\nQ: {q}\nA:")

docs = retriever.get_relevant_documents("Is the crust crispy?")
context = "\n\n".join(d.page_content for d in docs)
ans = llm.invoke(prompt.format(context=context, q="Is the crust crispy?"))
print(ans)
```

---

## 🧯 Troubleshooting

### ❗ Ollama pull fails with network / DNS error

If you see something like:

```
Error: max retries exceeded: Get "https://...cloudflarestorage.com/..."
```

Try:

1. Check internet connectivity.
2. Re-run the pull command: `ollama pull mxbai-embed-large`.
3. If behind a proxy, set `HTTP_PROXY` / `HTTPS_PROXY` env vars.
4. Restart the Ollama service / daemon.
5. Update Ollama to the latest version.

### ❗ Module not found: `langchain_ollama`

Make sure you installed the package in the active environment:

```
pip install langchain-ollama
```

### ❗ No such file: `realistic_restaurant_reviews.csv`

Confirm file path (same dir as scripts) or update `CSV_PATH` in `vector.py`.

### ❗ Empty / wrong embeddings after schema change

If you change the CSV significantly, delete the `chrome_langchain_db/` directory to force a rebuild.

---

## 📁 Project Structure

```
.
├── main.py
├── vector.py
├── realistic_restaurant_reviews.csv
├── chrome_langchain_db/    # auto-created on first run
│   └── ...                 # Chroma DB files
└── README.md               # (this file)
```

Optional extras you might add:

```
.env                     # environment overrides
requirements.txt         # pinned deps
Makefile                 # convenience commands
notebooks/               # experiments
scripts/                 # data prep utilities
```

---

## 🧭 Roadmap Ideas

* ✅ Multi-restaurant or multi-location support (metadata filter on restaurant\_id).
* ✅ Add sentiment summarization (positive/negative themes).
* 🔄 Streaming answers.
* 🔄 Web UI (Gradio / FastAPI + React front-end).
* 🔄 Auto-refresh embeddings when CSV changes.
* 🔄 Reranking w/ cross-encoders for higher answer fidelity.
* 🔄 Evaluate answer grounding vs. held-out reviews.

---

## 📄 License

MIT, Apache-2.0, or your preferred license. Example MIT stub:

```text
MIT License

Copyright (c) 2025 <Your Name>

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

---

## 🙌 Acknowledgments

* [Ollama](https://ollama.com/) for easy local LLM + embedding hosting.
* [LangChain](https://python.langchain.com/) for composable LLM pipelines.
* [Chroma](https://www.trychroma.com/) for fast local vector storage.
* Everyone who leaves pizza reviews so our AI can learn what’s tasty. 🍕

---

**Happy building!** If you want me to generate a sample dataset with synthetic reviews, a `requirements.txt`, or a `Makefile`, just ask.
