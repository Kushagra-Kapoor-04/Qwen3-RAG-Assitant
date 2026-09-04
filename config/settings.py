

import os
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL"
    )
    ollama_model: str = Field(
        default="qwen3:latest",
        description="Qwen3 model name in Ollama"
    )
    
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    faiss_index_path: str = Field(
        default="./data/processed/faiss_index",
        description="Path to FAISS index directory"
    )
    chunk_size: int = Field(
        default=500,
        description="Size of text chunks for splitting"
    )
    chunk_overlap: int = Field(
        default=50,
        description="Overlap between consecutive chunks"
    )
    
    top_k_results: int = Field(
        default=5,
        description="Number of documents to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.0,
        description="Minimum similarity score for retrieval"
    )
    
    hallucination_threshold: float = Field(
        default=0.5,
        description="Threshold for hallucination detection"
    )
    max_regeneration_attempts: int = Field(
        default=3,
        description="Maximum attempts for answer regeneration"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    log_file: str = Field(
        default="./logs/rag_assistant.log",
        description="Log file path"
    )
    
    raw_data_path: str = Field(
        default="./data/raw",
        description="Path to raw documents"
    )
    processed_data_path: str = Field(
        default="./data/processed",
        description="Path to processed data"
    )
    metadata_path: str = Field(
        default="./data/metadata",
        description="Path to metadata storage"
    )
    
    def get_faiss_index_path(self) -> Path:
        
        return Path(self.faiss_index_path)
    
    def get_raw_data_path(self) -> Path:
        
        return Path(self.raw_data_path)
    
    def get_processed_data_path(self) -> Path:
        
        return Path(self.processed_data_path)
    
    def get_metadata_path(self) -> Path:
        
        return Path(self.metadata_path)
    
    def ensure_directories(self) -> None:
        
        directories = [
            self.get_raw_data_path(),
            self.get_processed_data_path(),
            self.get_metadata_path(),
            Path(self.log_file).parent
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

settings = Settings()