Here's a concise English explanation and examples for your GitHub post:

---

# Qwen3EmbeddingClient

A production-ready Python client for running **Qwen3-Embedding-0.6B** locally without any external server.

## Features

- 🚀 Direct model loading from Hugging Face or local path
- 🎯 GPU (CUDA), CPU, and Apple Silicon (MPS) support  
- ⚡ Half-precision (FP16) and 4-bit quantization
- 💾 LRU-like caching for repeated texts
- 📦 Batched inference for high throughput
- 🔄 Async interface for FastAPI/asyncio
- 📊 Cosine similarity & ranking utilities

## Quick Start

```python
from qwen3_embedding import Qwen3EmbeddingClient

# Initialize client (auto-detects GPU/MPS)
client = Qwen3EmbeddingClient(
    use_fp16=True,      # half-precision for speed
    batch_size=32,      # batch size for inference
    normalize=True      # L2 normalize embeddings
)

# Single text embedding
result = client.embed_single("Hello world")
print(f"Dimensions: {len(result['embedding'])}")
print(f"Tokens: {result['tokens']}")
print(f"Time: {result['time_ms']}ms")

# Batch processing
texts = ["Machine learning is fascinating", "LLMs are transforming AI", "Embeddings capture semantics"]
batch = client.embed_batch(texts)
print(f"Total tokens: {batch['total_tokens']}")
print(f"Avg time: {batch['avg_time_ms']}ms")

# Find most similar texts
query = "AI models"
query_emb = client.embed_single(query)["embedding"]
candidate_emb = [r["embedding"] for r in batch["results"]]

ranked = client.rank_by_similarity(
    query_embedding=query_emb,
    candidate_embeddings=candidate_emb,
    texts=texts,
    top_k=2
)

for item in ranked:
    print(f"{item['rank']}. {item['text']} (score: {item['score']})")
```

## Async Usage (FastAPI)

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
client = Qwen3EmbeddingClient()

class TextRequest(BaseModel):
    text: str

@app.post("/embed")
async def embed(request: TextRequest):
    result = await client.async_embed_single(request.text)
    return {"embedding": result["embedding"]}
```

## Advanced Configuration

```python
# Memory-efficient (4-bit quantization on GPU)
client = Qwen3EmbeddingClient(
    use_4bit=True,      # requires bitsandbytes
    max_length=8192     # Qwen3 supports long context
)

# CPU-only with caching disabled
client = Qwen3EmbeddingClient(
    device="cpu",
    use_fp16=False,
    cache_embeddings=False
)

# Different pooling strategies
client = Qwen3EmbeddingClient(pooling_mode="mean")     # mean pooling
client = Qwen3EmbeddingClient(pooling_mode="last_token")  # default
```

## Utility Methods

```python
# Compare two embeddings
similarity = client.cosine_similarity(emb1, emb2)

# Full similarity matrix (N×N)
matrix = client.similarity_matrix([emb1, emb2, emb3])

# Model info
info = client.model_info()
print(f"Dimensions: {info['embedding_dim']}")
print(f"Device: {info['device']}")

# Memory management
client.clear_cache()      # clear cached embeddings
client.free_memory()      # release GPU memory
```

## Context Manager

```python
async with Qwen3EmbeddingClient() as client:
    result = await client.async_embed_single("Hello")
    # Auto cleanup on exit
```

## Performance Notes

- **FP16** reduces VRAM by ~50% with minimal quality loss
- **4-bit quantization** uses ~75% less memory (GPU only)  
- **Batch processing** is ~3-5x faster than sequential calls
- **Caching** eliminates recomputation for duplicate texts

## Requirements

```
torch>=2.0.0
transformers>=4.35.0
numpy>=1.24.0
bitsandbytes>=0.41.0   # optional, for 4-bit
```

---

This client is ideal for RAG systems, semantic search, clustering, and any application needing local high-quality embeddings.
