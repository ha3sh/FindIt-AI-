# FindIt AI — Lost & Found matcher (CLIP)

Owner types a text description of a lost item; finders upload photos of found items. Rank found items by similarity. CPU only, no training.



## Stack (do not substitute)

Python 3.11+, PyTorch (CPU), open_clip ViT-B-32-quickgelu pretrained='openai', FastAPI, Streamlit, SQLite (sqlite3) + NumPy.

Dev OS: Windows/PowerShell. Use pathlib; no bash-only commands.



## Core rules

- Embeddings: L2-normalised 512-d float32. Similarity = dot product.

- Image embedding computed once at upload, stored as BLOB.

- Score S = α·cos(text, image) + (1−α)·S_attr. α in app/config.py; α=1 is the CLIP-only baseline.

- S_attr = mean CLIP similarity between the item image and prompts like "a black bag", built from colour/type/brand parsed from the query. Parsing is rule-based (keyword lists). No LLM API.

- Matching only suggests. Admin marks claimed. Never auto-release.

- Load CLIP once (app startup / pytest session fixture), never per request.



## DB table `items`

item_id INTEGER PK, image_path TEXT, embedding BLOB, attributes TEXT(JSON), place TEXT, note TEXT, found_on TIMESTAMP, status TEXT ('unclaimed'|'claimed')



## API

POST /found · POST /search · GET /items · GET /items/{id} · PATCH /items/{id}/status



## UI (Streamlit, calls the API)

Report Found Item · Search Lost Item (cards + score) · Admin View (status toggle)



## Layout

app/{config,embedding,vocab,preprocess,matching,db,main}.py · ui/streamlit_app.py · eval/evaluate.py · tests/ · data/ (images + db, git-ignored)



## Out of scope — do NOT add

User auth, claims workflow, Postgres/SQLAlchemy, React, mobile app, face recognition, model training, LLM/VLM calls.



## Conventions

- Uploads: JPEG/PNG/WEBP only, max 5 MB; invalid → HTTP 400 with a clear message.

- Metrics: Recall@1/3/5, MRR, latency.

- Keep modules small and replaceable. Change only what's asked. Don't reprint unchanged files. Keep replies short; after tests, report only the pass/fail summary.

- REFERENCE_NOTES.md (if present) summarises a third-party repo: read only when working on matching/preprocess. Ideas only, never copy code.


- Never list files recursively (it floods context with .venv). Use: git ls-files. Never read .venv/.

- UI is Streamlit only. Look comes from CSS injected via st.markdown plus .streamlit/config.toml. No React, JS frameworks, npm, Tailwind or CDN assets.
