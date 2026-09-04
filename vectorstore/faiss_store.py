

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config.settings import settings
from config.constants import EMBEDDING_DIMENSION, METADATA_KEYS
from ingestion.embedder import EmbeddedChunk, Embedder
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class SearchResult:
    
    content: str
    metadata: Dict[str, Any]
    score: float
    
    @property
    def source(self) -> str:
        
        return self.metadata.get(METADATA_KEYS["source"], "unknown")

class FAISSStore:
    
    
    def __init__(
        self,
        index_path: Optional[str] = None,
        dimension: int = EMBEDDING_DIMENSION
    ):
        
        self.index_path = Path(index_path or settings.faiss_index_path)
        self.dimension = dimension
        
        self._index = None
        self._documents: List[Dict[str, Any]] = []
        self._initialized = False
        
        self.index_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def index(self):
        
        if self._index is None:
            self._create_index()
        return self._index
    
    @property
    def num_documents(self) -> int:
        
        return len(self._documents)
    
    def _create_index(self) -> None:
        
        try:
            import faiss
            
            self._index = faiss.IndexFlatIP(self.dimension)
            self._initialized = True
            logger.info(f"Created new FAISS index with dimension {self.dimension}")
            
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            raise
    
    def add_embeddings(
        self,
        embedded_chunks: List[EmbeddedChunk]
    ) -> None:
        
        if not embedded_chunks:
            logger.warning("No chunks to add")
            return
        
        embeddings = np.array(
            [ec.embedding for ec in embedded_chunks],
            dtype=np.float32
        )
        
        faiss = self._get_faiss()
        faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        
        for ec in embedded_chunks:
            self._documents.append({
                "content": ec.content,
                "metadata": ec.metadata
            })
        
        logger.info(f"Added {len(embedded_chunks)} chunks to index")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: Optional[float] = None
    ) -> List[SearchResult]:
        
        if self.num_documents == 0:
            logger.warning("Index is empty")
            return []
        
        query = np.array([query_embedding], dtype=np.float32)
        
        faiss = self._get_faiss()
        faiss.normalize_L2(query)
        
        k = min(top_k, self.num_documents)
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # Invalid index
                continue
            
            if threshold and score < threshold:
                continue
            
            doc = self._documents[idx]
            results.append(SearchResult(
                content=doc["content"],
                metadata=doc["metadata"],
                score=float(score)
            ))
        
        return results
    
    def save(self, path: Optional[str] = None) -> None:
        
        save_path = Path(path) if path else self.index_path
        save_path.mkdir(parents=True, exist_ok=True)
        
        faiss = self._get_faiss()
        
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))
        
        docs_file = save_path / "documents.pkl"
        with open(docs_file, "wb") as f:
            pickle.dump(self._documents, f)
        
        meta_file = save_path / "metadata.json"
        metadata = {
            "dimension": self.dimension,
            "num_documents": self.num_documents
        }
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved index to {save_path}")
    
    def load(self, path: Optional[str] = None) -> bool:
        
        load_path = Path(path) if path else self.index_path
        
        index_file = load_path / "index.faiss"
        docs_file = load_path / "documents.pkl"
        
        if not index_file.exists() or not docs_file.exists():
            logger.warning(f"Index files not found at {load_path}")
            return False
        
        try:
            faiss = self._get_faiss()
            
            self._index = faiss.read_index(str(index_file))
            
            with open(docs_file, "rb") as f:
                self._documents = pickle.load(f)
            
            self._initialized = True
            logger.info(f"Loaded index from {load_path} ({self.num_documents} documents)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            return False
    
    def clear(self) -> None:
        
        self._index = None
        self._documents = []
        self._initialized = False
        self._create_index()
        logger.info("Index cleared")
    
    def delete_index(self) -> None:
        
        if self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path)
            logger.info(f"Deleted index at {self.index_path}")
    
    def _get_faiss(self):
        
        try:
            import faiss
            return faiss
        except ImportError:
            logger.error("faiss not installed. Install with: pip install faiss-cpu")
            raise

def create_faiss_store(
    index_path: Optional[str] = None,
    dimension: int = EMBEDDING_DIMENSION
) -> FAISSStore:
    
    return FAISSStore(index_path, dimension)

def load_faiss_store(
    index_path: Optional[str] = None,
    dimension: int = EMBEDDING_DIMENSION
) -> Optional[FAISSStore]:
    
    store = FAISSStore(index_path, dimension)
    if store.load():
        return store
    return None