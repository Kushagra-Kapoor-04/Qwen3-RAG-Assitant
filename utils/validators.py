

import os
import re
from pathlib import Path
from typing import Optional, List, Tuple, Any
from dataclasses import dataclass

from config.constants import (
    SUPPORTED_EXTENSIONS,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    MAX_TOP_K,
    MIN_SIMILARITY_SCORE,
    MAX_SIMILARITY_SCORE
)
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class ValidationResult:
    
    is_valid: bool
    message: str = ""
    cleaned_value: Any = None

def validate_query(query: str) -> ValidationResult:
    
    if not query:
        return ValidationResult(
            is_valid=False,
            message="Query cannot be empty"
        )
    
    if not isinstance(query, str):
        return ValidationResult(
            is_valid=False,
            message="Query must be a string"
        )
    
    cleaned = query.strip()
    
    if len(cleaned) == 0:
        return ValidationResult(
            is_valid=False,
            message="Query cannot be empty or whitespace only"
        )
    
    if len(cleaned) < 3:
        return ValidationResult(
            is_valid=False,
            message="Query must be at least 3 characters"
        )
    
    if len(cleaned) > 10000:
        return ValidationResult(
            is_valid=False,
            message="Query exceeds maximum length (10000 characters)"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=cleaned
    )

def validate_file_path(path: str) -> ValidationResult:
    
    if not path:
        return ValidationResult(
            is_valid=False,
            message="Path cannot be empty"
        )
    
    file_path = Path(path)
    
    if not file_path.exists():
        return ValidationResult(
            is_valid=False,
            message=f"Path does not exist: {path}"
        )
    
    if file_path.is_file():
        extension = file_path.suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            return ValidationResult(
                is_valid=False,
                message=f"Unsupported file extension: {extension}"
            )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=str(file_path.absolute())
    )

def validate_directory_path(path: str) -> ValidationResult:
    
    if not path:
        return ValidationResult(
            is_valid=False,
            message="Directory path cannot be empty"
        )
    
    dir_path = Path(path)
    
    if not dir_path.exists():
        return ValidationResult(
            is_valid=False,
            message=f"Directory does not exist: {path}"
        )
    
    if not dir_path.is_dir():
        return ValidationResult(
            is_valid=False,
            message=f"Path is not a directory: {path}"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=str(dir_path.absolute())
    )

def validate_chunk_size(size: int) -> ValidationResult:
    
    if not isinstance(size, int):
        return ValidationResult(
            is_valid=False,
            message="Chunk size must be an integer"
        )
    
    if size < MIN_CHUNK_SIZE:
        return ValidationResult(
            is_valid=False,
            message=f"Chunk size must be at least {MIN_CHUNK_SIZE}"
        )
    
    if size > MAX_CHUNK_SIZE:
        return ValidationResult(
            is_valid=False,
            message=f"Chunk size cannot exceed {MAX_CHUNK_SIZE}"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=size
    )

def validate_top_k(k: int) -> ValidationResult:
    
    if not isinstance(k, int):
        return ValidationResult(
            is_valid=False,
            message="top_k must be an integer"
        )
    
    if k < 1:
        return ValidationResult(
            is_valid=False,
            message="top_k must be at least 1"
        )
    
    if k > MAX_TOP_K:
        return ValidationResult(
            is_valid=False,
            message=f"top_k cannot exceed {MAX_TOP_K}"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=k
    )

def validate_threshold(threshold: float) -> ValidationResult:
    
    if not isinstance(threshold, (int, float)):
        return ValidationResult(
            is_valid=False,
            message="Threshold must be a number"
        )
    
    if threshold < MIN_SIMILARITY_SCORE:
        return ValidationResult(
            is_valid=False,
            message=f"Threshold must be at least {MIN_SIMILARITY_SCORE}"
        )
    
    if threshold > MAX_SIMILARITY_SCORE:
        return ValidationResult(
            is_valid=False,
            message=f"Threshold cannot exceed {MAX_SIMILARITY_SCORE}"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=float(threshold)
    )

def validate_context(context: str) -> ValidationResult:
    
    if context is None:
        return ValidationResult(
            is_valid=True,
            cleaned_value=""
        )
    
    if not isinstance(context, str):
        return ValidationResult(
            is_valid=False,
            message="Context must be a string"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=context.strip()
    )

def validate_answer(answer: str) -> ValidationResult:
    
    if answer is None:
        return ValidationResult(
            is_valid=False,
            message="Answer cannot be None"
        )
    
    if not isinstance(answer, str):
        return ValidationResult(
            is_valid=False,
            message="Answer must be a string"
        )
    
    return ValidationResult(
        is_valid=True,
        cleaned_value=answer.strip()
    )

def sanitize_filename(filename: str) -> str:
    
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    sanitized = sanitized.strip('. ')
    
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext
    
    return sanitized or "unnamed"

def is_valid_url(url: str) -> bool:
    
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$',
        re.IGNORECASE
    )
    return bool(url_pattern.match(url))

def validate_batch(
    items: List[Any],
    validator: callable
) -> Tuple[List[Any], List[str]]:
    
    valid_items = []
    errors = []
    
    for i, item in enumerate(items):
        result = validator(item)
        if result.is_valid:
            valid_items.append(result.cleaned_value or item)
        else:
            errors.append(f"Item {i}: {result.message}")
    
    return valid_items, errors