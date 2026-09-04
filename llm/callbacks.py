

from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import time

from services.logging_service import get_logger

logger = get_logger(__name__)

class StreamCallback(ABC):
    
    
    @abstractmethod
    def on_token(self, token: str) -> None:
        
        pass
    
    def on_stream_end(self) -> None:
        
        pass

class PrintStreamCallback(StreamCallback):
    
    
    def __init__(self, end: str = "", flush: bool = True):
        
        self.end = end
        self.flush = flush
    
    def on_token(self, token: str) -> None:
        
        print(token, end=self.end, flush=self.flush)
    
    def on_stream_end(self) -> None:
        
        print()

class CollectorStreamCallback(StreamCallback):
    
    
    def __init__(self):
        
        self._tokens: list = []
    
    def on_token(self, token: str) -> None:
        
        self._tokens.append(token)
    
    @property
    def text(self) -> str:
        
        return "".join(self._tokens)
    
    def clear(self) -> None:
        
        self._tokens.clear()

class FunctionStreamCallback(StreamCallback):
    
    
    def __init__(
        self,
        callback_fn: Callable[[str], None],
        end_callback_fn: Optional[Callable[[], None]] = None
    ):
        
        self.callback_fn = callback_fn
        self.end_callback_fn = end_callback_fn
    
    def on_token(self, token: str) -> None:
        
        self.callback_fn(token)
    
    def on_stream_end(self) -> None:
        
        if self.end_callback_fn:
            self.end_callback_fn()

@dataclass
class LLMStats:
    
    total_calls: int = 0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_time_seconds: float = 0.0
    errors: int = 0
    
    @property
    def avg_tokens_per_call(self) -> float:
        
        return self.total_tokens / max(1, self.total_calls)
    
    @property
    def avg_time_per_call(self) -> float:
        
        return self.total_time_seconds / max(1, self.total_calls)

class CallbackHandler:
    
    
    def __init__(
        self,
        verbose: bool = False,
        track_stats: bool = True
    ):
        
        self.verbose = verbose
        self.track_stats = track_stats
        self.stats = LLMStats()
        
        self._start_time: Optional[float] = None
        self._current_prompt: Optional[str] = None
    
    def on_llm_start(self, prompt: str) -> None:
        
        self._start_time = time.time()
        self._current_prompt = prompt
        
        if self.verbose:
            logger.info(f"LLM started with prompt length: {len(prompt)}")
    
    def on_llm_end(self, response: Any) -> None:
        
        if self._start_time:
            elapsed = time.time() - self._start_time
            
            if self.track_stats:
                self.stats.total_calls += 1
                self.stats.total_time_seconds += elapsed
                
                if hasattr(response, 'prompt_tokens'):
                    self.stats.total_prompt_tokens += response.prompt_tokens
                if hasattr(response, 'completion_tokens'):
                    self.stats.total_completion_tokens += response.completion_tokens
                if hasattr(response, 'total_tokens'):
                    self.stats.total_tokens += response.total_tokens
            
            if self.verbose:
                logger.info(f"LLM completed in {elapsed:.2f}s")
        
        self._start_time = None
        self._current_prompt = None
    
    def on_llm_error(self, error: Exception) -> None:
        
        if self.track_stats:
            self.stats.errors += 1
        
        logger.error(f"LLM error: {str(error)}")
        
        self._start_time = None
        self._current_prompt = None
    
    def get_stats(self) -> LLMStats:
        
        return self.stats
    
    def reset_stats(self) -> None:
        
        self.stats = LLMStats()

class CompositeCallbackHandler(CallbackHandler):
    
    
    def __init__(self, handlers: list):
        
        super().__init__()
        self.handlers = handlers
    
    def on_llm_start(self, prompt: str) -> None:
        
        super().on_llm_start(prompt)
        for handler in self.handlers:
            handler.on_llm_start(prompt)
    
    def on_llm_end(self, response: Any) -> None:
        
        super().on_llm_end(response)
        for handler in self.handlers:
            handler.on_llm_end(response)
    
    def on_llm_error(self, error: Exception) -> None:
        
        super().on_llm_error(error)
        for handler in self.handlers:
            handler.on_llm_error(error)

def create_callback_handler(
    verbose: bool = False,
    track_stats: bool = True
) -> CallbackHandler:
    
    return CallbackHandler(verbose, track_stats)

def create_stream_callback(on_token: Callable[[str], None]) -> StreamCallback:
    
    return FunctionStreamCallback(on_token)