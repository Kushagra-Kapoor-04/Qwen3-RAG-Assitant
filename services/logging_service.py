

import sys
from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
from datetime import datetime
import json

from config.settings import settings

@dataclass
class QueryLog:
    
    question: str
    answer: str
    sources: list
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_grounded: bool = True
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class LoggerAdapter:
    
    
    def __init__(self, name: str):
        
        self.name = name
        self._logger = None
        self._use_loguru = False
        self._setup_logger()
    
    def _setup_logger(self) -> None:
        
        try:
            from loguru import logger
            self._logger = logger
            self._use_loguru = True
        except ImportError:
            import logging
            self._logger = logging.getLogger(self.name)
            self._use_loguru = False
            
            if not self._logger.handlers:
                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
    
    def debug(self, message: str, **kwargs) -> None:
        
        if self._use_loguru:
            self._logger.debug(message, **kwargs)
        else:
            self._logger.debug(message)
    
    def info(self, message: str, **kwargs) -> None:
        
        if self._use_loguru:
            self._logger.info(message, **kwargs)
        else:
            self._logger.info(message)
    
    def warning(self, message: str, **kwargs) -> None:
        
        if self._use_loguru:
            self._logger.warning(message, **kwargs)
        else:
            self._logger.warning(message)
    
    def error(self, message: str, **kwargs) -> None:
        
        if self._use_loguru:
            self._logger.error(message, **kwargs)
        else:
            self._logger.error(message)
    
    def critical(self, message: str, **kwargs) -> None:
        
        if self._use_loguru:
            self._logger.critical(message, **kwargs)
        else:
            self._logger.critical(message)

class LoggingService:
    
    
    _instance: Optional['LoggingService'] = None
    _initialized: bool = False
    
    def __new__(cls):
        
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        
        if not self._initialized:
            self._setup_logging()
            self._query_logs: list = []
            LoggingService._initialized = True
    
    def _setup_logging(self) -> None:
        
        try:
            from loguru import logger
            
            logger.remove()
            
            logger.add(
                sys.stdout,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                    "<level>{message}</level>"
                ),
                level=settings.log_level,
                colorize=True
            )
            
            log_file = Path(settings.log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            logger.add(
                str(log_file),
                format=(
                    "{time:YYYY-MM-DD HH:mm:ss} | "
                    "{level: <8} | "
                    "{name}:{function}:{line} | "
                    "{message}"
                ),
                level=settings.log_level,
                rotation="10 MB",
                retention="7 days",
                compression="gz"
            )
            
            self._logger = logger
            
        except ImportError:
            import logging
            
            logging.basicConfig(
                level=getattr(logging, settings.log_level, logging.INFO),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(sys.stdout),
                    logging.FileHandler(settings.log_file)
                ]
            )
            
            self._logger = logging.getLogger("rag_assistant")
    
    def log_query(self, query_log: QueryLog) -> None:
        
        self._query_logs.append(query_log)
        
        log_data = {
            "question": query_log.question[:100],
            "sources": len(query_log.sources),
            "grounded": query_log.is_grounded,
            "time_ms": query_log.processing_time_ms
        }
        
        self.info(f"Query processed: {json.dumps(log_data)}")
    
    def get_query_logs(self) -> list:
        
        return self._query_logs.copy()
    
    def save_query_logs(self, path: Optional[str] = None) -> None:
        
        save_path = Path(path or settings.metadata_path) / "query_logs.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        logs_data = [
            {
                "question": log.question,
                "answer": log.answer,
                "sources": log.sources,
                "timestamp": log.timestamp,
                "is_grounded": log.is_grounded,
                "processing_time_ms": log.processing_time_ms,
                "metadata": log.metadata
            }
            for log in self._query_logs
        ]
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False)
        
        self.info(f"Saved {len(logs_data)} query logs to {save_path}")
    
    def debug(self, message: str, **kwargs) -> None:
        
        try:
            self._logger.debug(message, **kwargs)
        except Exception:
            print(f"DEBUG: {message}")
    
    def info(self, message: str, **kwargs) -> None:
        
        try:
            self._logger.info(message, **kwargs)
        except Exception:
            print(f"INFO: {message}")
    
    def warning(self, message: str, **kwargs) -> None:
        
        try:
            self._logger.warning(message, **kwargs)
        except Exception:
            print(f"WARNING: {message}")
    
    def error(self, message: str, **kwargs) -> None:
        
        try:
            self._logger.error(message, **kwargs)
        except Exception:
            print(f"ERROR: {message}")
    
    def critical(self, message: str, **kwargs) -> None:
        
        try:
            self._logger.critical(message, **kwargs)
        except Exception:
            print(f"CRITICAL: {message}")

_logging_service: Optional[LoggingService] = None

def get_logging_service() -> LoggingService:
    
    global _logging_service
    if _logging_service is None:
        _logging_service = LoggingService()
    return _logging_service

def get_logger(name: str) -> LoggerAdapter:
    
    get_logging_service()
    return LoggerAdapter(name)