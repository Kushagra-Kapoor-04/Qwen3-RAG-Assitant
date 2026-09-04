

import os
import json
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import time

from services.logging_service import get_logger

logger = get_logger(__name__)

def timer(func):
    
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

def generate_id(content: str) -> str:
    
    return hashlib.md5(content.encode()).hexdigest()[:12]

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_list(nested: List[List[Any]]) -> List[Any]:
    
    return [item for sublist in nested for item in sublist]

def safe_json_loads(text: str, default: Any = None) -> Any:
    
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: str = "{}") -> str:
    
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except (TypeError, ValueError):
        return default

def ensure_directory(path: Union[str, Path]) -> Path:
    
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_file_size(path: Union[str, Path]) -> int:
    
    file_path = Path(path)
    return file_path.stat().st_size if file_path.exists() else 0

def get_file_extension(path: Union[str, Path]) -> str:
    
    return Path(path).suffix.lower()

def format_timestamp(dt: Optional[datetime] = None) -> str:
    
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def format_duration(seconds: float) -> str:
    
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_bytes(bytes_count: int) -> str:
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024
    return f"{bytes_count:.1f} PB"

def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def clean_text(text: str) -> str:
    
    import re
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def merge_dicts(
    base: Dict[str, Any],
    override: Dict[str, Any]
) -> Dict[str, Any]:
    
    result = base.copy()
    
    for key, value in override.items():
        if (
            key in result and
            isinstance(result[key], dict) and
            isinstance(value, dict)
        ):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,)
):
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}), "
                            f"retrying in {current_delay}s: {str(e)}"
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
            
            raise last_exception
        return wrapper
    return decorator

def create_progress_bar(
    total: int,
    prefix: str = "Progress",
    length: int = 50
):
    
    def update(current: int) -> str:
        filled = int(length * current // total)
        bar = '█' * filled + '░' * (length - filled)
        percent = 100 * current / total
        return f"\r{prefix}: |{bar}| {percent:.1f}%"
    
    return update

def get_env_or_default(key: str, default: Any = None) -> Any:
    
    return os.environ.get(key, default)

def is_running_in_notebook() -> bool:
    
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return False