

from enum import Enum
from typing import List, Set

class DocumentType(str, Enum):
    
    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"

class ChunkingStrategy(str, Enum):
    
    RECURSIVE = "recursive"
    CHARACTER = "character"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"

class RetrievalMode(str, Enum):
    
    SIMILARITY = "similarity"
    MMR = "mmr"  # Maximum Marginal Relevance
    HYBRID = "hybrid"

class EvaluationResult(str, Enum):
    
    GROUNDED = "grounded"
    PARTIALLY_GROUNDED = "partially_grounded"
    UNGROUNDED = "ungrounded"
    UNCERTAIN = "uncertain"

class LogLevel(str, Enum):
    
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

SUPPORTED_EXTENSIONS: Set[str] = {".pdf", ".txt", ".docx", ".md"}

MIME_TYPES: dict = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".md": "text/markdown"
}

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
MIN_CHUNK_SIZE: int = 100
MAX_CHUNK_SIZE: int = 2000

DEFAULT_TOP_K: int = 5
MAX_TOP_K: int = 20
MIN_SIMILARITY_SCORE: float = 0.0
MAX_SIMILARITY_SCORE: float = 1.0

EMBEDDING_DIMENSION: int = 384

GROUNDING_CONFIDENCE_THRESHOLD: float = 0.7
REGENERATION_MAX_ATTEMPTS: int = 3

MAX_CONTEXT_TOKENS: int = 4096
MAX_RESPONSE_TOKENS: int = 1024
MAX_INPUT_TOKENS: int = 512

METADATA_KEYS = {
    "source": "source",
    "page": "page",
    "chunk_id": "chunk_id",
    "timestamp": "timestamp",
    "document_type": "document_type",
    "title": "title",
    "author": "author"
}

ERROR_MESSAGES = {
    "file_not_found": "The specified file was not found: {path}",
    "unsupported_format": "Unsupported file format: {format}",
    "embedding_failed": "Failed to generate embeddings: {error}",
    "retrieval_failed": "Failed to retrieve documents: {error}",
    "llm_error": "LLM generation failed: {error}",
    "index_not_found": "FAISS index not found at: {path}",
    "validation_error": "Validation failed: {error}"
}

SUCCESS_MESSAGES = {
    "ingestion_complete": "Successfully ingested {count} documents",
    "index_created": "FAISS index created successfully",
    "index_loaded": "FAISS index loaded successfully",
    "query_processed": "Query processed successfully"
}

STATUS_CODES = {
    "success": 200,
    "created": 201,
    "bad_request": 400,
    "not_found": 404,
    "internal_error": 500
}

def get_supported_extensions() -> List[str]:
    
    return list(SUPPORTED_EXTENSIONS)

def is_supported_extension(extension: str) -> bool:
    
    return extension.lower() in SUPPORTED_EXTENSIONS

def get_error_message(key: str, **kwargs) -> str:
    
    template = ERROR_MESSAGES.get(key, "Unknown error")
    return template.format(**kwargs)

def get_success_message(key: str, **kwargs) -> str:
    
    template = SUCCESS_MESSAGES.get(key, "Operation completed")
    return template.format(**kwargs)