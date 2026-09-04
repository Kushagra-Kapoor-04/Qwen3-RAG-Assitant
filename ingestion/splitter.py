

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re

from config.settings import settings
from config.constants import (
    ChunkingStrategy,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    METADATA_KEYS
)
from ingestion.loader import Document
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class TextChunk:
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_index: int = 0
    
    @property
    def source(self) -> str:
        
        return self.metadata.get(METADATA_KEYS["source"], "unknown")
    
    def __len__(self) -> int:
        
        return len(self.content)

class TextSplitter:
    
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    ):
        
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.strategy = strategy
        
        self.chunk_size = max(MIN_CHUNK_SIZE, min(MAX_CHUNK_SIZE, self.chunk_size))
        self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 2)
        
        self._separators = [
            "\n\n\n",  # Multiple newlines (section breaks)
            "\n\n",     # Paragraph breaks
            "\n",       # Line breaks
            ". ",       # Sentence endings
            "! ",       # Exclamation endings
            "? ",       # Question endings
            "; ",       # Semicolons
            ", ",       # Commas
            " ",        # Words
            ""          # Characters
        ]
    
    def split_document(self, document: Document) -> List[TextChunk]:
        
        if not document.content or not document.content.strip():
            logger.warning(f"Empty document: {document.source}")
            return []
        
        text = self._clean_text(document.content)
        
        if self.strategy == ChunkingStrategy.RECURSIVE:
            chunks_text = self._recursive_split(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            chunks_text = self._sentence_split(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPH:
            chunks_text = self._paragraph_split(text)
        else:
            chunks_text = self._character_split(text)
        
        chunks = []
        for idx, chunk_text in enumerate(chunks_text):
            if chunk_text.strip():
                chunk_metadata = document.metadata.copy()
                chunk_metadata[METADATA_KEYS["chunk_id"]] = f"{document.source}_{idx}"
                
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    metadata=chunk_metadata,
                    chunk_index=idx
                ))
        
        logger.info(f"Split document into {len(chunks)} chunks: {document.source}")
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[TextChunk]:
        
        all_chunks = []
        for document in documents:
            chunks = self.split_document(document)
            all_chunks.extend(chunks)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        return all_chunks
    
    def _clean_text(self, text: str) -> str:
        
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    
    def _recursive_split(self, text: str) -> List[str]:
        
        return self._split_with_separators(text, self._separators)
    
    def _split_with_separators(
        self,
        text: str,
        separators: List[str]
    ) -> List[str]:
        
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        if not separators:
            return self._force_split(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    if len(current_chunk) > self.chunk_size:
                        sub_chunks = self._split_with_separators(
                            current_chunk, remaining_separators
                        )
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_chunk)
                
                if len(split) > self.chunk_size:
                    sub_chunks = self._split_with_separators(
                        split, remaining_separators
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split
        
        if current_chunk:
            if len(current_chunk) > self.chunk_size:
                sub_chunks = self._split_with_separators(
                    current_chunk, remaining_separators
                )
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_chunk)
        
        return self._add_overlap(chunks)
    
    def _force_split(self, text: str) -> List[str]:
        
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    def _add_overlap(self, chunks: List[str]) -> List[str]:
        
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks
        
        overlapped_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            overlap_text = prev_chunk[-self.chunk_overlap:]
            
            space_idx = overlap_text.rfind(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            
            overlapped_chunk = overlap_text + " " + curr_chunk
            overlapped_chunks.append(overlapped_chunk.strip())
        
        return overlapped_chunks
    
    def _sentence_split(self, text: str) -> List[str]:
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return self._merge_to_chunks(sentences)
    
    def _paragraph_split(self, text: str) -> List[str]:
        
        paragraphs = text.split('\n\n')
        return self._merge_to_chunks(paragraphs)
    
    def _character_split(self, text: str) -> List[str]:
        
        return self._force_split(text)
    
    def _merge_to_chunks(self, segments: List[str]) -> List[str]:
        
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue
            
            test_chunk = current_chunk + "\n\n" + segment if current_chunk else segment
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if len(segment) > self.chunk_size:
                    sub_chunks = self._force_split(segment)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = segment
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return self._add_overlap(chunks)

def split_documents(
    documents: List[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
) -> List[TextChunk]:
    
    splitter = TextSplitter(chunk_size, chunk_overlap, strategy)
    return splitter.split_documents(documents)