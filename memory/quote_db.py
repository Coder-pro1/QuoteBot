"""
QuoteDBManager  —  Multi-Signal Composite Scorer
================================================
Scoring formula per candidate quote:
  final = (0.50 × semantic_sim) + (0.35 × emotion_sim) + (0.15 × type_weight)

  semantic_sim  : 1 / (1 + l2_distance)          — existing FAISS hit
  emotion_sim   : cosine_similarity(query_vec, emotion_vec)  — pre-cached at load
  type_weight   : quote=1.0, catchphrase=0.75, signature_word=0.5

Emotion vectors are embedded ONCE at load time from distinct emotion labels.
No extra model loads, no extra FAISS index, zero overhead at query time.
"""
import json
import os
from typing import Optional, Any, List, Dict, cast
import faiss  # type: ignore
import numpy as np  # type: ignore
from core.shared_encoder import get_encoder  # type: ignore


# Signal weights
W_SEMANTIC = 0.70
W_EMOTION  = 0.30


class QuoteDBManager:
    def __init__(
        self,
        json_path:  str = "data/quote_dictionary.json",
        index_path: str = "data/indexes/quote.index",
    ):
        self.json_path: str = json_path
        self.index_path: str = index_path
        self.model = get_encoder()
        self.quotes: List[Dict[str, Any]] = []
        self.index: Any = None

        # Pre-computed emotion vectors  {label: unit_vec}
        self._emotion_vecs: dict = {}

        self._load_or_build()

    # ── Build / Load ──────────────────────────────────────────

    def _load_or_build(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Missing {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            self.quotes = json.load(f)

        # FAISS index on quote USECASE/CONTEXT instead of raw text
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            print("🏗️ Building Cinematic Quote Index (Usecase Architecture)...")
            texts = []
            for q in self.quotes:
                # Merge the quote text and its situational usecase so the vector contains BOTH
                base_usecase = q.get("usecase", "")
                if not base_usecase:
                    emotion = q.get("emotion", "general")
                    base_usecase = f"Situation: User needs a quote about {emotion}."
                
                # The final document embedded in FAISS contains literal quote AND situational usecase
                compound_doc = f"Quote: {q['text']}. {base_usecase}"
                texts.append(compound_doc)
                
            embs = self.model.encode(texts, convert_to_numpy=True)
            # Normalize for Cosine Similarity
            faiss.normalize_L2(embs)
            embs = embs.astype("float32")
            
            d = embs.shape[1]
            index_obj = faiss.IndexFlatIP(d)  # Inner Product (Cosine Sim)
            index_obj.add(embs)
            faiss.write_index(index_obj, self.index_path)
            self.index = index_obj

        # Pre-embed distinct emotion labels (done once, very fast)
        self._build_emotion_cache()

    def _build_emotion_cache(self):
        """Embed each distinct emotion label once and cache as unit vectors."""
        emotions = list({q.get("emotion", "general") for q in self.quotes})
        vecs     = self.model.encode(emotions, convert_to_numpy=True).astype("float32")
        # Normalize to unit vectors for cosine similarity via dot product
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        unit_vecs = vecs / norms
        self._emotion_vecs = dict(zip(emotions, unit_vecs))

    # ── Helpers ───────────────────────────────────────────────

    def _emotion_sim(self, query_unit_vec: np.ndarray, emotion_label: str) -> float:
        """Cosine similarity between query vector and an emotion label vector."""
        emotion_vec = self._emotion_vecs.get(emotion_label)
        if emotion_vec is None:
            return 0.0
        return float(np.dot(query_unit_vec, emotion_vec))

    # ── Public Search ─────────────────────────────────────────

    def search_quote(
        self,
        query:        str,
        top_k:        int  = 5,
        used_indices: list = [],
    ) -> List[Dict[str, Any]]:
        """
        Returns up to top_k candidates sorted by composite score (highest first).
        Each result dict has: index, semantic_sim, emotion_sim, type_weight,
        composite_score, metadata.
        """
        idx_obj = self.index
        if idx_obj is None:
            return []
        
        # Encode query and normalize for Cosine Similarity index
        raw_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(raw_vec)
        raw_vec = raw_vec.astype("float32")

        # Unit vector mapping for emotion metadata comparison
        query_unit = raw_vec

        # FAISS over-fetch so we have room to filter used indices
        fetch_k = min(top_k * 4, int(idx_obj.ntotal))
        distances, indices = idx_obj.search(raw_vec, fetch_k)  # type: ignore

        results: list[dict[str, Any]] = []
        for dist_item, idx_item in zip(distances[0], indices[0]):  # type: ignore
            idx = int(idx_item)
            # distances from IndexFlatIP with normalized vectors are exactly cosine similarities (0.0 to 1.0 typically)
            cosine_sim = float(dist_item)
            
            if idx == -1 or idx in used_indices:
                continue

            quote_data = self.quotes[idx]

            # ── Signal 1: Semantic similarity (Cosine: -1.0 to 1.0, clip to 0) ──
            semantic_sim = max(0.0, cosine_sim)

            # ── Signal 2: Emotion match ──────────────────────────
            emotion_label = quote_data.get("emotion", "general")
            emotion_sim   = max(0.0, self._emotion_sim(query_unit[0], emotion_label))

            # ── Composite ─────────────────────────────────────────
            composite: float = (
                W_SEMANTIC * semantic_sim
                + W_EMOTION  * emotion_sim
            )

            results.append({
                "index":         idx,
                "semantic_sim":  float(f"{semantic_sim:.3f}"),
                "emotion_sim":   float(f"{emotion_sim:.3f}"),
                "type_weight":   1.0,
                "composite":     float(f"{composite:.3f}"),
                "metadata":      quote_data,
            })

        # Sort highest composite first
        results.sort(key=lambda x: x["composite"], reverse=True)
        top_results: list[dict[str, Any]] = []
        for i in range(min(top_k, len(results))):
            top_results.append(results[i])
        return top_results

    def search_catchphrase(
        self,
        query:        str,
        top_k:        int  = 3,
        used_indices: list = [],
    ) -> list:
        """
        Search specifically for catchphrases (type='catchphrase').
        Returns up to top_k catchphrase candidates sorted by composite score.
        """
        # Get broader search results first
        all_results = self.search_quote(query, top_k=top_k * 3, used_indices=used_indices)
        
        # Filter to only catchphrases
        catchphrases = [
            r for r in all_results 
            if r["metadata"].get("type") == "catchphrase"
        ]
        result_len = min(len(catchphrases), top_k)
        return [catchphrases[i] for i in range(result_len)]
