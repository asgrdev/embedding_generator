import asyncio
import gc
import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Optional, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


# ─────────────────────────────────────────────────────────────
# TypedDicts
# ─────────────────────────────────────────────────────────────

class EmbeddingResult(TypedDict):
    text:       str
    embedding:  list[float]
    tokens:     int
    model:      str
    time_ms:    float


class BatchEmbeddingResult(TypedDict):
    results:        list[EmbeddingResult]
    total_tokens:   int
    total_time_ms:  float
    avg_time_ms:    float


class ModelInfo(TypedDict):
    model_name:     str
    embedding_dim:  int
    max_length:     int
    device:         str
    dtype:          str
    quantized:      bool


# ─────────────────────────────────────────────────────────────
# کلاس اصلی
# ─────────────────────────────────────────────────────────────

class Qwen3EmbeddingClient:
    """
    اجرای مستقیم مدل Qwen3-Embedding-0.6B بدون نیاز به سرور

    ویژگی‌ها:
    - بارگذاری مستقیم مدل از Hugging Face یا مسیر لوکال
    - پشتیبانی از GPU/CPU/MPS (Apple Silicon)
    - بهینه‌سازی با half precision و quantization
    - کش embedding برای متن‌های تکراری
    - پردازش batch موازی
    - Pooling هوشمند (last token / mean / cls)
    """

    MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"

    def __init__(
        self,
        model_path:   Optional[str] =  "/models/Qwen3-Embedding-0.6B",
        device:       Optional[str] = None,
        use_fp16:     bool = True,
        use_4bit:     bool = False,
        batch_size:   int  = 32,
        max_length:   int  = 8192,
        pooling_mode: Literal["last_token", "mean", "cls"] = "last_token",
        cache_embeddings: bool = True,
        normalize:    bool = True,
    ):
        """
        پارامترها:
            model_path:       مسیر لوکال مدل یا None برای دانلود خودکار
            device:           'cuda' | 'cpu' | 'mps' | None (خودکار)
            use_fp16:         استفاده از float16 برای کاهش مصرف VRAM
            use_4bit:         کوانتیزیشن 4-bit با bitsandbytes
            batch_size:       اندازه batch برای پردازش موازی
            max_length:       حداکثر طول توکن
            pooling_mode:     روش ادغام token embeddings
            cache_embeddings: کش کردن نتایج برای متن‌های تکراری
            normalize:        نرمال‌سازی L2 خروجی
        """
        self.model_path       = model_path or self.MODEL_ID
        self.batch_size       = batch_size
        self.max_length       = max_length
        self.pooling_mode     = pooling_mode
        self.cache_embeddings = cache_embeddings
        self.normalize        = normalize
        self.use_fp16         = use_fp16
        self.use_4bit         = use_4bit

        self.device = self._detect_device(device)
        self._cache: dict[str, list[float]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

        self.tokenizer, self.model = self._load_model()
        self.embedding_dim = self._detect_embedding_dim()

        logger.info(
            f"✅ مدل آماده | device={self.device} | "
            f"dim={self.embedding_dim} | pooling={pooling_mode}"
        )
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()
    # ──────────────────────────────────────────
    # راه‌اندازی
    # ──────────────────────────────────────────

    def _detect_device(self, device: Optional[str]) -> str:
        """تشخیص خودکار بهترین device موجود"""
        if device:
            return device
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"🔥 GPU یافت شد: {gpu_name}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("🍎 Apple Silicon MPS فعال")
            return "mps"
        logger.info("💻 اجرا روی CPU")
        return "cpu"

    def _load_model(self):
        """بارگذاری tokenizer و مدل با بهینه‌سازی‌های لازم"""
        logger.info(f"📥 بارگذاری مدل از: {self.model_path}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

        # تنظیمات بارگذاری بر اساس سخت‌افزار
        load_kwargs: dict = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if self.use_4bit and self.device == "cuda":
            # کوانتیزیشن 4-bit → نیاز به: pip install bitsandbytes
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                logger.info("⚡ کوانتیزیشن 4-bit فعال")
            except ImportError:
                logger.warning("bitsandbytes نصب نیست، 4-bit غیرفعال")

        elif self.use_fp16 and self.device in ("cuda", "mps"):
            load_kwargs["torch_dtype"] = torch.float16

        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModel.from_pretrained(self.model_path, **load_kwargs)

        # انتقال به device اگر quantization نباشد
        if not self.use_4bit:
            model = model.to(self.device)

        model.eval()
        return tokenizer, model

    def _detect_embedding_dim(self) -> int:
        """تشخیص ابعاد embedding از مدل"""
        try:
            return self.model.config.hidden_size
        except AttributeError:
            # fallback: تست با یک متن کوچک
            with torch.no_grad():
                enc = self.tokenizer(
                    "test",
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)
                out = self.model(**enc)
                return out.last_hidden_state.shape[-1]

    # ──────────────────────────────────────────
    # Pooling
    # ──────────────────────────────────────────

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask:    torch.Tensor,
    ) -> torch.Tensor:
        """اعمال pooling بر خروجی مدل"""

        if self.pooling_mode == "last_token":
            # Qwen3 با last non-padding token بهترین نتیجه را دارد
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size  = last_hidden_state.shape[0]
            return last_hidden_state[
                torch.arange(batch_size, device=last_hidden_state.device),
                seq_lengths,
            ]

        elif self.pooling_mode == "mean":
            # میانگین وزن‌دار با attention mask
            mask_expanded = (
                attention_mask.unsqueeze(-1)
                .float()
                .expand_as(last_hidden_state)
            )
            sum_embeddings  = (last_hidden_state * mask_expanded).sum(dim=1)
            sum_mask        = mask_expanded.sum(dim=1).clamp(min=1e-9)
            return sum_embeddings / sum_mask

        else:  # cls
            return last_hidden_state[:, 0, :]

    # ──────────────────────────────────────────
    # توکن‌سازی
    # ──────────────────────────────────────────

    def _tokenize(self, texts: list[str]) -> dict:
        """توکن‌سازی با padding و truncation"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

    def _count_tokens(self, text: str) -> int:
        """شمارش توکن بدون padding"""
        return len(
            self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
            )["input_ids"]
        )

    # ──────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────

    def _forward(self, texts: list[str]) -> torch.Tensor:
        """اجرای forward pass و برگرداندن embeddings"""
        encoded = self._tokenize(texts)

        with torch.no_grad():
            output = self.model(**encoded)

        embeddings = self._pool(
            output.last_hidden_state,
            encoded["attention_mask"],
        )

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings.float().cpu()

    # ──────────────────────────────────────────
    # رابط عمومی - Sync
    # ──────────────────────────────────────────

    def embed_single(self, text: str) -> EmbeddingResult:
        """
        Embedding یک متن - اجرای همزمان (sync)

        مثال:
            result = client.embed_single("سلام دنیا")
            print(result["embedding"][:5])
        """
        # بررسی کش
        cache_key = self._make_cache_key(text)
        if self.cache_embeddings and cache_key in self._cache:
            return EmbeddingResult(
                text=text,
                embedding=self._cache[cache_key],
                tokens=self._count_tokens(text),
                model=self.model_path,
                time_ms=0.0,
            )

        t0 = time.perf_counter()
        tensor = self._forward([text])
        elapsed = (time.perf_counter() - t0) * 1000

        embedding = tensor[0].tolist()

        if self.cache_embeddings:
            self._cache[cache_key] = embedding

        return EmbeddingResult(
            text=text,
            embedding=embedding,
            tokens=self._count_tokens(text),
            model=self.model_path,
            time_ms=round(elapsed, 2),
        )

    def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """
        Embedding چندین متن - پردازش batch - اجرای همزمان (sync)

        مثال:
            results = client.embed_batch(["متن اول", "متن دوم"])
        """
        if not texts:
            return BatchEmbeddingResult(
                results=[],
                total_tokens=0,
                total_time_ms=0.0,
                avg_time_ms=0.0,
            )

        t_start   = time.perf_counter()
        results   = []
        all_embeddings: list[list[float]] = []

        # جداسازی cache hits و misses
        cached_map:   dict[int, list[float]] = {}
        uncached_idx: list[int]              = []
        uncached_txt: list[str]              = []

        for i, text in enumerate(texts):
            key = self._make_cache_key(text)
            if self.cache_embeddings and key in self._cache:
                cached_map[i] = self._cache[key]
            else:
                uncached_idx.append(i)
                uncached_txt.append(text)

        # پردازش متن‌های بدون کش در batch‌ها
        new_embeddings: list[list[float]] = []
        for i in range(0, len(uncached_txt), self.batch_size):
            chunk  = uncached_txt[i: i + self.batch_size]
            tensor = self._forward(chunk)
            new_embeddings.extend(tensor.tolist())

        # ذخیره در کش
        for idx_in_uncached, original_idx in enumerate(uncached_idx):
            emb = new_embeddings[idx_in_uncached]
            if self.cache_embeddings:
                key = self._make_cache_key(texts[original_idx])
                self._cache[key] = emb

        # ساخت لیست نهایی با ترتیب درست
        new_ptr = 0
        for i, text in enumerate(texts):
            if i in cached_map:
                emb = cached_map[i]
            else:
                emb = new_embeddings[new_ptr]
                new_ptr += 1
            all_embeddings.append(emb)

        total_ms = (time.perf_counter() - t_start) * 1000

        for text, emb in zip(texts, all_embeddings):
            results.append(
                EmbeddingResult(
                    text=text,
                    embedding=emb,
                    tokens=self._count_tokens(text),
                    model=self.model_path,
                    time_ms=round(total_ms / len(texts), 2),
                )
            )

        return BatchEmbeddingResult(
            results=results,
            total_tokens=sum(r["tokens"] for r in results),
            total_time_ms=round(total_ms, 2),
            avg_time_ms=round(total_ms / len(texts), 2),
        )

    # ──────────────────────────────────────────
    # رابط عمومی - Async (thread pool)
    # ──────────────────────────────────────────

    async def async_embed_single(self, text: str) -> EmbeddingResult:
        """نسخه async از embed_single - مناسب برای FastAPI / asyncio"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed_single,
            text,
        )

    async def async_embed_batch(
        self, texts: list[str]
    ) -> BatchEmbeddingResult:
        """نسخه async از embed_batch"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self.embed_batch,
            texts,
        )

    # ──────────────────────────────────────────
    # ابزارهای مقایسه
    # ──────────────────────────────────────────

    def cosine_similarity(
        self,
        a: list[float],
        b: list[float],
    ) -> float:
        """شباهت کسینوسی بین دو embedding"""
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def similarity_matrix(
        self,
        embeddings: list[list[float]],
    ) -> np.ndarray:
        """
        ماتریس شباهت N×N برای N embedding

        مثال:
            matrix = client.similarity_matrix([emb1, emb2, emb3])
            # matrix[i][j] = شباهت بین embedding i و j
        """
        mat = np.array(embeddings, dtype=np.float32)
        # نرمال‌سازی اگر قبلاً نشده
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        mat_normed = mat / norms
        return (mat_normed @ mat_normed.T).astype(float)

    def rank_by_similarity(
        self,
        query_embedding:    list[float],
        candidate_embeddings: list[list[float]],
        texts:              list[str],
        top_k:              int = 5,
    ) -> list[dict]:
        """
        رتبه‌بندی متن‌ها بر اساس شباهت به query

        برگشتی: لیست مرتب شده از {'text', 'score', 'rank'}
        """
        scores = [
            self.cosine_similarity(query_embedding, emb)
            for emb in candidate_embeddings
        ]
        ranked = sorted(
            zip(texts, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        return [
            {"text": txt, "score": round(score, 4), "rank": i + 1}
            for i, (txt, score) in enumerate(ranked)
        ]

    # ──────────────────────────────────────────
    # کش و مدیریت
    # ──────────────────────────────────────────

    def _make_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def clear_cache(self) -> None:
        """پاک‌سازی کش embeddings"""
        self._cache.clear()
        logger.info("🗑️ کش پاک شد")

    def cache_size(self) -> int:
        """تعداد آیتم‌های کش شده"""
        return len(self._cache)

    def model_info(self) -> ModelInfo:
        """اطلاعات مدل بارگذاری شده"""
        dtype = str(next(self.model.parameters()).dtype)
        return ModelInfo(
            model_name=self.model_path,
            embedding_dim=self.embedding_dim,
            max_length=self.max_length,
            device=self.device,
            dtype=dtype,
            quantized=self.use_4bit,
        )

    def free_memory(self) -> None:
        """آزاد کردن حافظه GPU"""
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("🧹 حافظه آزاد شد")

    def __repr__(self) -> str:
        info = self.model_info()
        return (
            f"Qwen3EmbeddingClient("
            f"model={info['model_name']}, "
            f"dim={info['embedding_dim']}, "
            f"device={info['device']}, "
            f"cache={self.cache_size()} items)"
        )
