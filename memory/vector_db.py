import os
import json
import time
import faiss
import numpy as np
from core.shared_encoder import get_encoder

class VectorDBManager:
    def __init__(self, data_dir: str = "data", index_dir: str = "data/indexes", model_name: str = "all-MiniLM-L6-v2"):
        """
        Advanced 3-Layer FAISS Database.
        Stores not just embeddings, but structured metadata (timestamp, importance, type).
        Supports dynamic Decay Scoring (Similarity + Recency + Importance).
        """
        self.data_dir = data_dir
        self.index_dir = index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        
        self.model = get_encoder()
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Format: {"domain": {"index": faiss.IndexFlatL2, "data": [{"text": str, "timestamp": float, "importance": float, "type": str}]}}
        self.databases = {}
        self._sync_all_indexes()

    def _sync_all_indexes(self):
        """Builds indices from base JSONs (like slang_dictionary) for static knowledge."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)
            return

        json_files = [f for f in os.listdir(self.data_dir) if f.endswith(".json")]
        for filename in json_files:
            domain_name = filename.replace(".json", "")
            self._load_or_build_index(domain_name, os.path.join(self.data_dir, filename))
            
        # Ensure our personal Long-Term memory domains exist even if empty
        for domain in ["preferences", "learning_progress", "personal_context"]:
            if domain not in self.databases:
                self._init_empty_domain(domain)

    def _init_empty_domain(self, domain_name: str):
        """Creates an empty FAISS index for dynamic memory insertion."""
        index_path = os.path.join(self.index_dir, f"{domain_name}.index")
        meta_path = os.path.join(self.index_dir, f"{domain_name}_meta.json")
        
        if os.path.exists(index_path) and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                cached_meta = json.load(f)
            self.databases[domain_name] = {
                "index": faiss.read_index(index_path),
                "data": cached_meta["data"]
            }
        else:
            self.databases[domain_name] = {
                "index": faiss.IndexFlatL2(self.dimension),
                "data": []
            }
            faiss.write_index(self.databases[domain_name]["index"], index_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({"data": []}, f)

    def _load_or_build_index(self, domain_name: str, source_json_path: str):
        """(Legacy Static Load) For standard dictionaries that don't need timestamps."""
        index_path = os.path.join(self.index_dir, f"{domain_name}.index")
        meta_path = os.path.join(self.index_dir, f"{domain_name}_meta.json")
        current_mtime = os.path.getmtime(source_json_path)
        
        needs_rebuild = True
        if os.path.exists(index_path) and os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                cached_meta = json.load(f)
            if cached_meta.get("source_mtime") == current_mtime:
                needs_rebuild = False
                self.databases[domain_name] = {
                    "index": faiss.read_index(index_path),
                    "data": cached_meta["data"]
                }
        
        if needs_rebuild:
            with open(source_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
                
            new_index = faiss.IndexFlatL2(self.dimension)
            valid_data = []
            corpus = []
            
            for item in raw_data:
                combined_text = " | ".join([f"{k}: {v}" for k, v in item.items() if isinstance(v, str)])
                if combined_text.strip():
                    corpus.append(combined_text)
                    valid_data.append(item) 
            
            if corpus:
                embeddings = self.model.encode(corpus, convert_to_numpy=True)
                new_index.add(embeddings)
                
            self.databases[domain_name] = {
                "index": new_index,
                "data": valid_data
            }
            faiss.write_index(new_index, index_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump({"source_mtime": current_mtime, "data": valid_data}, f)

    def _save_domain(self, domain_name: str):
        """Standard helper to physically dump the FAISS index and JSON Meta array to disk"""
        db = self.databases[domain_name]
        index_path = os.path.join(self.index_dir, f"{domain_name}.index")
        meta_path = os.path.join(self.index_dir, f"{domain_name}_meta.json")
        faiss.write_index(db["index"], index_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({"data": db["data"]}, f)

    def add_memory(self, domain_name: str, text: str, importance: float = 0.5, mem_type: str = "general"):
        """
        Dynamically adds a structured memory.
        If a very similar memory exists, it OVERWRITES it to prevent duplication and conflicting rules.
        """
        if domain_name not in self.databases:
            self._init_empty_domain(domain_name)
            
        db = self.databases[domain_name]
        
        # 1. Compute Vector for the new memory
        embedding = self.model.encode([text], convert_to_numpy=True)
        
        # 2. Deduplication / Overwrite Check
        if db["index"].ntotal > 0:
            distances, indices = db["index"].search(embedding, 1)
            # Threshold Check: Distance < 1.0 means the semantic meaning is basically identical
            if indices[0][0] != -1 and distances[0][0] < 1.0:
                match_idx = indices[0][0]
                old_text = db['data'][match_idx]['text']
                print(f"🔄 [Memory Update] Overwriting old memory: '{old_text[:40]}...' -> '{text[:40]}...'")
                
                # Update the JSON tracking metadata
                db["data"][match_idx] = {
                    "text": text,
                    "timestamp": time.time(),
                    "importance": importance,
                    "type": mem_type
                }
                
                # Rebuild FAISS index from the updated JSON array
                # (Since we can't delete directly from FlatL2 easily, we rebuild. Fast for personal DBs)
                new_index = faiss.IndexFlatL2(self.dimension)
                corpus_embeddings = self.model.encode([m["text"] for m in db["data"]], convert_to_numpy=True)
                new_index.add(corpus_embeddings)
                db["index"] = new_index
                
                self._save_domain(domain_name)
                return
        
        # 3. Add to FAISS and local list normally if no close duplicate exists
        memory_obj = {
            "text": text,
            "timestamp": time.time(),
            "importance": importance,
            "type": mem_type
        }
        
        db["index"].add(embedding)
        db["data"].append(memory_obj)
        
        # 4. Save persistently
        self._save_domain(domain_name)
        print(f"💾 [Memory Saved] -> '{domain_name}': {text[:50]}...")

    def search_with_decay(self, domain_name: str, query: str, top_k: int = 3) -> list:
        """
        Retrieves memories and applies the composite scoring formula:
        Score = Similarity + Importance_Weight + Recency_Weight
        """
        if domain_name not in self.databases:
            return []
            
        db = self.databases[domain_name]
        if db["index"].ntotal == 0:
            return []
            
        # 1. Fetch a wider net (e.g., top 10) so we can sort them manually by decay
        fetch_k = min(10, db["index"].ntotal)
        query_emb = self.model.encode([query], convert_to_numpy=True)
        
        # FAISS returns L2 distances (lower is better, 0 is exact match)
        distances, indices = db["index"].search(query_emb, fetch_k)
        
        scored_results = []
        current_time = time.time()
        
        for i, idx in enumerate(indices[0]):
            if idx == -1: continue
            
            memory_obj = db["data"][idx]
            
            # Convert FAISS L2 distance to a "Similarity Score" (0 to 1, higher is better)
            # L2 dist can technically be anything, but we invert it.
            base_similarity = 1.0 / (1.0 + distances[0][i])
            
            # Enforce a strict minimum semantic relevance threshold so we don't return random memories.
            # 0.40 score ~ 1.5 L2 distance. Irrelevant hits usually sit below 0.35.
            if base_similarity < 0.40:
                continue
            
            # Static items (like slang dictionary) don't have timestamps
            if "timestamp" not in memory_obj:
                scored_results.append((base_similarity, memory_obj))
                continue
                
            # --- DECAY MATH ---
            age_in_seconds = current_time - memory_obj.get("timestamp", current_time)
            age_in_days = age_in_seconds / 86400.0
            
            importance = memory_obj.get("importance", 0.5)
            
            # Decay Formula: similarity * (1 / (1 + age_in_days))
            recency_weight = base_similarity * (1.0 / (1.0 + age_in_days))
            
            # Final Score
            final_score = base_similarity + (importance * 0.5) + recency_weight
            
            scored_results.append((final_score, memory_obj))
            
        # Sort by highest final score
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # Return only the top_k elements, stripping the score away
        return [item[1] for item in scored_results[:top_k]]

    def search(self, domain_name: str, query: str, top_k: int = 3) -> list:
        """Standard search without decay routing to search_with_decay"""
        return self.search_with_decay(domain_name, query, top_k)

    def format_search_results(self, matches: list) -> str:
        if not matches:
            return ""
        context = ""
        for item in matches:
            # If it's a dynamic memory, just grab the 'text', else it's a dictionary record
            if "text" in item:
                context += f" - {item['text']}\n"
            else:
                context += " - " + ", ".join([f"**{k}**: {v}" for k, v in item.items()]) + "\n"
        return context
