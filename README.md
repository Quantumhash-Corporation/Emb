# BGE-Large-EN-v1.5 Embedding API
**FastAPI + FlagEmbedding | QuantumHash Corporation**

---

## 🐍 Python Version — Konsa Use Karo?

| Version | Status | Recommendation |
|---------|--------|----------------|
| Python 3.9 | Old | ❌ Use mat karo |
| Python 3.10 | Stable | ⚠️ Theek hai |
| **Python 3.11** | **Best** | ✅ **Ye use karo** |
| Python 3.12 | Latest | ⚠️ Kuch ML packages shaky hain |
| Python 3.13 | Too new | ❌ Torch support nahi |

**👉 Python 3.11 kyun?**
- PyTorch 2.5.x fully tested on 3.11
- FlagEmbedding 1.4.0 — 3.11 pe best compatibility
- 3.12 pe numpy/torch ke beech minor issues aa sakte hain
- Speed: 3.11 me significant performance improvements hain over 3.10

---

## 📁 Project Structure

```
bge-api/
├── app/
│   └── main.py          ← FastAPI server (main code yahan hai)
├── tests/
│   └── test_api.py      ← Test script
├── .env.example         ← Environment variables template
├── requirements.txt     ← Python dependencies
├── start.sh             ← Startup script
├── nginx-bge-api.conf   ← Nginx reverse proxy config
└── README.md
```

---

## 🚀 Setup Guide (Server: 192.168.8.17)

### Step 1 — Conda Environment (Python 3.11)

```bash
conda create -n bge python=3.11 -y
conda activate bge
```

### Step 2 — Dependencies Install

```bash
cd ~/LLM/BGE-API   # ya jahan bhi rakha hai
pip install -r requirements.txt

# PyTorch CUDA version (tumhare dual Quadro ke liye)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Environment Setup

```bash
cp .env.example .env
nano .env
# BGE_API_KEYS=qh-bge-yourkey123,qh-bge-anotherkey456
```

### Step 4 — Server Start

```bash
# Development
bash start.sh

# Production (background me)
nohup uvicorn app.main:app --host 0.0.0.0 --port 8100 --workers 1 > bge-api.log 2>&1 &

# Ya PM2 se (recommended)
pm2 start "uvicorn app.main:app --host 0.0.0.0 --port 8100" --name bge-api
```

### Step 5 — Test

```bash
python tests/test_api.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server + GPU status |
| POST | `/v1/embeddings` | Text → embeddings generate karo |
| POST | `/v1/similarity` | Query vs passages similarity |
| POST | `/v1/rerank` | Documents ko rerank karo |
| GET | `/docs` | Swagger UI |

---

## 🔑 API Key Usage

Har request me `X-API-Key` header lagao:

```bash
curl -X POST https://emb.quantumhash.me/v1/embeddings \
  -H "X-API-Key: qh-bge-yourkey123" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"], "is_query": false}'
```

---

## 🐍 Python Client Example

```python
import httpx

client = httpx.Client()
headers = {"X-API-Key": "qh-bge-yourkey123"}

# Documents embed karo
response = client.post(
    "https://emb.quantumhash.me/v1/embeddings",
    headers=headers,
    json={
        "texts": ["Your document text here"],
        "is_query": False,
        "normalize": True,
    }
)
embeddings = response.json()["embeddings"]
print(f"Embedding shape: {len(embeddings[0])}d")  # 1024

# Queries embed karo (instruction auto add hoti hai)
response = client.post(
    "https://emb.quantumhash.me/v1/embeddings",
    headers=headers,
    json={
        "texts": ["What is machine learning?"],
        "is_query": True,
    }
)
```

---

## 🎮 GPU Tips (Tumhare Dual Quadro RTX 5000)

```bash
# Dono GPUs use karo
export CUDA_VISIBLE_DEVICES=0,1

# Sirf ek GPU use karo
export CUDA_VISIBLE_DEVICES=0

# GPU memory check
nvidia-smi

# FP16 mode (default ON agar GPU available) — ~1.5-2GB VRAM use karta hai
```

---

## ⚠️ Important Notes

1. **Workers = 1** rakhna zaroori hai — multiple workers GPU memory double karenge
2. **Similarity scores 0.6-1.0** range mein aate hain — 0.8+ ko similar maano
3. **is_query=True** sirf search queries ke liye — documents pe False rakho
4. **normalize=True** cosine similarity ke liye — Qdrant ke saath use karna

---

## 🔗 Qdrant Integration Example

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import httpx
import uuid

# BGE API se embedding lo
def get_embedding(text: str, is_query: bool = False) -> list:
    r = httpx.post(
        "https://emb.quantumhash.me/v1/embeddings",
        headers={"X-API-Key": "your-key"},
        json={"texts": [text], "is_query": is_query}
    )
    return r.json()["embeddings"][0]

# Qdrant me store karo
qdrant = QdrantClient("https://qdrant.quantumhash.me", api_key="your-qdrant-key")

embedding = get_embedding("Your document content here")
qdrant.upsert(
    collection_name="my_collection",
    points=[PointStruct(id=str(uuid.uuid4()), vector=embedding, payload={"text": "..."})]
)

# Search karo
query_emb = get_embedding("search query", is_query=True)
results = qdrant.search(collection_name="my_collection", query_vector=query_emb, limit=5)
```
