

from typing import List, Optional, Union
import numpy as np
from dataclasses import dataclass

from config.settings import settings
from config.constants import EMBEDDING_DIMENSION
from ingestion.splitter import TextChunk
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class EmbeddedChunk:
    
    chunk: TextChunk
    embedding: np.ndarray
    
    @property
    def content(self) -> str:
        
        return self.chunk.content
    
    @property
    def metadata(self) -> dict:
        
        return self.chunk.metadata

class Embedder:
    
    
    def __init__(self, model_name: Optional[str] = None):
        
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self._dimension = EMBEDDING_DIMENSION
    
    @property
    def model(self):
        
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def dimension(self) -> int:
        
        return self._dimension
    
    def _load_model(self) -> None:
        
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self._dimension}")
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return np.zeros(self._dimension)
        
        try:
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        
        if not texts:
            return np.empty((0, self._dimension))
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def embed_chunk(self, chunk: TextChunk) -> EmbeddedChunk:
        
        embedding = self.embed_text(chunk.content)
        return EmbeddedChunk(chunk=chunk, embedding=embedding)
    
    def embed_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> List[EmbeddedChunk]:
        
        if not chunks:
            return []
        
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_texts(texts, batch_size, show_progress)
        
        embedded_chunks = [
            EmbeddedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        return embedded_chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        
        return self.embed_text(query)

def create_embedder(model_name: Optional[str] = None) -> Embedder:
    
    return Embedder(model_name)

def embed_chunks(
    chunks: List[TextChunk],
    model_name: Optional[str] = None,
    batch_size: int = 32
) -> List[EmbeddedChunk]:
    
    embedder = create_embedder(model_name)
    return embedder.embed_chunks(chunks, batch_size)