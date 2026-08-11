# AskDocs

A retrieval-augmented QA system over arXiv papers: pull papers from the arXiv
API → extract text from the PDFs → chunk → embed → store in Qdrant → retrieve
on query → generate an answer grounded in the retrieved context.

Currently a **single working notebook** (`ragger.ipynb`). Everything below
that pipeline runs end-to-end; the rest is a scoped roadmap.

## What's working right now

**Ingestion**
- Queries the arXiv API (`export.arxiv.org/api/query`) via `urllib` + `feedparser`, parsing Atom entries into `{id: {url, title, author, tags}}`.
- Downloads each PDF into memory (`BytesIO`, no files written to disk) and extracts text with `unstructured.partition_pdf` (`strategy="fast"` — text-layer extraction, no layout model, keeps the dependency footprint down).
- Filters out `Header` / `Footer` / `PageBreak` elements before concatenating each paper into one `Document`, with arXiv metadata (id, title, author, tags) attached.
- Respects arXiv's rate limit (3s minimum between requests) regardless of success or failure on a given paper.

**Chunking**
- `RecursiveCharacterTextSplitter` (tiktoken `cl100k_base` encoding), 500 tokens / 50 overlap.

**Embedding + storage**
- `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings`.
- Qdrant as the vector store (`langchain_qdrant.QdrantVectorStore`), run locally via Docker, REST API on port 6333.

**Retrieval**
- Similarity search via `store.as_retriever(search_type="similarity", search_kwargs={"k": 4})`.

**Generation**
- `HuggingFaceEndpoint` (currently `Qwen/Qwen2.5-7B-Instruct`) wrapped in `ChatHuggingFace`.
- Manual prompt template stuffing retrieved chunks into context, with a system prompt instructing the model to answer from context and admit when it can't.

## Setup

**Requirements**
```bash
pip install -r requirements.txt
```

**Environment**

Create a `.env` with your Hugging Face token:
```
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

**Qdrant** (via Docker)
```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -v ./qdrant:/qdrant/storage qdrant/qdrant
```
Storage persists to a local `./qdrant` volume, so re-running ingestion doesn't require re-embedding from scratch unless you want a fresh collection.

**Run**

Open `ragger.ipynb` and run top to bottom. Qdrant must be running first.

## Known limitations (current state)

- Everything lives in one notebook — no module structure yet, hard to reuse or test pieces independently.
- No interface — querying means running notebook cells by hand.
- No citation grounding in generation — the model answers from context but doesn't reliably cite *which* retrieved chunk/paper it drew from.
- `strategy="fast"` means figures, tables, and captions aren't reliably categorized (arXiv PDFs are text-native, so prose extraction itself is fine — structure around non-text elements is not).
- Single-hop retrieval only, no reranking — fine for a handful of papers, will degrade as the corpus grows.
- No evaluation — answer quality is judged by eye.

## Roadmap

Roughly in priority order:

1. **Citation grounding** — tag each retrieved chunk with its source (paper id/title) in the prompt, and require the model to cite which source(s) it used per claim.
2. **Interface** — wrap the pipeline in a minimal Streamlit app (query box, retrieved-chunks view, generated answer) so it's demoable without opening the notebook.
3. **Reorganize into modules** — split the notebook into `ingest/`, `retrieve/`, `generate/` modules so later upgrades touch one file each instead of the whole notebook.
4. **Hybrid retrieval + reranking** — BM25 + dense with reciprocal rank fusion, cross-encoder reranker.
5. **Agentic loop (LangGraph)** — query decomposition for multi-hop questions, self-correction / re-retrieval when context is judged insufficient.
6. **Evaluation (RAGAS)** — a small hand-built Q&A benchmark scored for faithfulness, context precision/recall, answer relevance.
7. **API wrapper (FastAPI) + Docker Compose** — decouple the pipeline from any one UI, package the whole stack (app + Qdrant) for reproducible deployment.

## Notes on dependencies

`unstructured`'s PDF path pulls in system-level dependencies beyond pip
(notably Poppler, for page-count/PDF-info calls) — not obvious from
`pip install unstructured` alone, and worth flagging for anyone else setting
this up fresh. `strategy="fast"` avoids the heavier layout-model dependencies
(`hi_res` needs those) but still needs Poppler on PATH.