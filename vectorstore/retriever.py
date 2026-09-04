

from typing import List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from config.constants import RetrievalMode
from vectorstore.faiss_store import FAISSStore, SearchResult, load_faiss_store
from ingestion.embedder import Embedder, create_embedder
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class RetrievalResult:
    
    content: str
    source: str
    score: float
    metadata: dict

class Retriever:
    
    
    def __init__(
        self,
        store: Optional[FAISSStore] = None,
        embedder: Optional[Embedder] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        mode: RetrievalMode = RetrievalMode.SIMILARITY
    ):
        
        self._store = store
        self._embedder = embedder
        self.top_k = top_k or settings.top_k_results
        self.threshold = threshold or settings.similarity_threshold
        self.mode = mode
    
    @property
    def store(self) -> FAISSStore:
        
        if self._store is None:
            self._store = load_faiss_store()
            if self._store is None:
                raise ValueError("No FAISS index found. Run ingestion first.")
        return self._store
    
    @property
    def embedder(self) -> Embedder:
        
        if self._embedder is None:
            self._embedder = create_embedder()
        return self._embedder
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        k = top_k or self.top_k
        thresh = threshold or self.threshold
        
        try:
            query_embedding = self.embedder.embed_query(query)
            
            search_results = self.store.search(
                query_embedding,
                top_k=k,
                threshold=thresh
            )
            
            results = [
                RetrievalResult(
                    content=r.content,
                    source=r.source,
                    score=r.score,
                    metadata=r.metadata
                )
                for r in search_results
            ]
            
            logger.info(f"Retrieved {len(results)} documents for query")
            for i, res in enumerate(results):
                logger.debug(f"  Result {i+1}: score={res.score:.3f}, source={Path(res.source).name}")
                
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            raise
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        include_sources: bool = True
    ) -> Tuple[str, List[RetrievalResult]]:
        
        results = self.retrieve(query, top_k, threshold)
        
        if not results:
            return "", []
        
        context_parts = []
        for i, result in enumerate(results, 1):
            if include_sources:
                source_info = f"[Source {i}: {result.source}]"
                context_parts.append(f"{source_info}\n{result.content}")
            else:
                context_parts.append(result.content)
        
        context = "\n\n---\n\n".join(context_parts)
        
        return context, results
    
    def retrieve_mmr(
        self,
        query: str,
        top_k: Optional[int] = None,
        diversity: float = 0.5
    ) -> List[RetrievalResult]:
        
        k = top_k or self.top_k
        
        candidates = self.retrieve(query, top_k=k * 3, threshold=0.0)
        
        if len(candidates) <= k:
            return candidates
        
        selected = [candidates[0]]
        candidate_pool = candidates[1:]
        
        while len(selected) < k and candidate_pool:
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(candidate_pool):
                relevance = candidate.score
                
                max_sim = max(
                    self._text_similarity(candidate.content, s.content)
                    for s in selected
                )
                diversity_score = 1 - max_sim
                
                mmr_score = diversity * relevance + (1 - diversity) * diversity_score
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(candidate_pool.pop(best_idx))
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def is_ready(self) -> bool:
        
        try:
            return self.store.num_documents > 0
        except Exception:
            return False

def create_retriever(
    store: Optional[FAISSStore] = None,
    embedder: Optional[Embedder] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None
) -> Retriever:
    
    return Retriever(store, embedder, top_k, threshold)

def get_retriever() -> Retriever:
    
    return create_retriever()