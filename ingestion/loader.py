

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config.constants import (
    DocumentType,
    SUPPORTED_EXTENSIONS,
    METADATA_KEYS,
    get_error_message
)
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class Document:
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def source(self) -> str:
        
        return self.metadata.get(METADATA_KEYS["source"], "unknown")
    
    @property
    def document_type(self) -> str:
        
        return self.metadata.get(METADATA_KEYS["document_type"], "unknown")

class DocumentLoader:
    
    
    def __init__(self):
        
        self._loaders = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            ".md": self._load_txt,  # Markdown uses same loader as txt
        }
    
    def load_file(self, file_path: str) -> Optional[Document]:
        
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        extension = path.suffix.lower()
        
        if extension not in self._loaders:
            logger.error(f"Unsupported file format: {extension}")
            return None
        
        try:
            loader_func = self._loaders[extension]
            content = loader_func(path)
            
            metadata = {
                METADATA_KEYS["source"]: str(path.absolute()),
                METADATA_KEYS["document_type"]: extension[1:],  # Remove dot
                METADATA_KEYS["title"]: path.stem,
                METADATA_KEYS["timestamp"]: datetime.now().isoformat()
            }
            
            logger.info(f"Successfully loaded: {file_path}")
            return Document(content=content, metadata=metadata)
            
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            return None
    
    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True
    ) -> List[Document]:
        
        path = Path(directory_path)
        
        if not path.exists():
            logger.error(f"Directory not found: {directory_path}")
            return []
        
        if not path.is_dir():
            logger.error(f"Path is not a directory: {directory_path}")
            return []
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                doc = self.load_file(str(file_path))
                if doc:
                    documents.append(doc)
        
        logger.info(f"Loaded {len(documents)} documents from {directory_path}")
        return documents
    
    def _load_pdf(self, path: Path) -> str:
        
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(str(path))
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[Page {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
            
        except ImportError:
            logger.error("pypdf not installed. Install with: pip install pypdf")
            raise
    
    def _load_txt(self, path: Path) -> str:
        
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        
        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        with open(path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")

def load_documents(source_path: str, recursive: bool = True) -> List[Document]:
    
    loader = DocumentLoader()
    path = Path(source_path)
    
    if path.is_file():
        doc = loader.load_file(source_path)
        return [doc] if doc else []
    elif path.is_dir():
        return loader.load_directory(source_path, recursive)
    else:
        logger.error(f"Path does not exist: {source_path}")
        return []