

from typing import Optional, List, Dict, Any, Iterator
from dataclasses import dataclass

from config.settings import settings
from llm.callbacks import CallbackHandler, StreamCallback
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class LLMResponse:
    
    content: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def is_empty(self) -> bool:
        
        return not self.content or not self.content.strip()

class Qwen3LLM:
    
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        callback_handler: Optional[CallbackHandler] = None
    ):
        
        self.model_name = model_name or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.callback_handler = callback_handler
        
        self._client = None
    
    @property
    def client(self):
        
        if self._client is None:
            self._create_client()
        return self._client
    
    def _create_client(self) -> None:
        
        try:
            import ollama
            
            self._client = ollama.Client(host=self.base_url)
            logger.info(f"Created Ollama client for {self.base_url}")
            
        except ImportError:
            logger.error("ollama not installed. Install with: pip install ollama")
            raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.callback_handler:
            self.callback_handler.on_llm_start(prompt)
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": tokens
                }
            )
            
            content = response.get("message", {}).get("content", "")
            
            result = LLMResponse(
                content=content,
                model=self.model_name,
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=(
                    response.get("prompt_eval_count", 0) +
                    response.get("eval_count", 0)
                )
            )
            
            if self.callback_handler:
                self.callback_handler.on_llm_end(result)
            
            return result
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            if self.callback_handler:
                self.callback_handler.on_llm_error(e)
            raise
    
    def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream_callback: Optional[StreamCallback] = None
    ) -> Iterator[str]:
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        if self.callback_handler:
            self.callback_handler.on_llm_start(prompt)
        
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": tokens
                },
                stream=True
            )
            
            full_response = ""
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_response += token
                    if stream_callback:
                        stream_callback.on_token(token)
                    yield token
            
            if self.callback_handler:
                result = LLMResponse(content=full_response, model=self.model_name)
                self.callback_handler.on_llm_end(result)
                
        except Exception as e:
            logger.error(f"LLM streaming failed: {str(e)}")
            if self.callback_handler:
                self.callback_handler.on_llm_error(e)
            raise
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens or self.max_tokens
        
        try:
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": temp,
                    "num_predict": tokens
                }
            )
            
            content = response.get("message", {}).get("content", "")
            
            return LLMResponse(
                content=content,
                model=self.model_name,
                prompt_tokens=response.get("prompt_eval_count", 0),
                completion_tokens=response.get("eval_count", 0),
                total_tokens=(
                    response.get("prompt_eval_count", 0) +
                    response.get("eval_count", 0)
                )
            )
            
        except Exception as e:
            logger.error(f"LLM chat failed: {str(e)}")
            raise
    
    def is_available(self) -> bool:
        
        try:
            models = self.client.list()
            model_names = [m.get("name", "") for m in models.get("models", [])]
            
            for name in model_names:
                if self.model_name.split(":")[0] in name:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check model availability: {str(e)}")
            return False
    
    def pull_model(self) -> bool:
        
        try:
            logger.info(f"Pulling model: {self.model_name}")
            self.client.pull(self.model_name)
            logger.info(f"Model pulled successfully: {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            return False

def create_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: int = 1024
) -> Qwen3LLM:
    
    return Qwen3LLM(model_name, temperature=temperature, max_tokens=max_tokens)

def get_llm() -> Qwen3LLM:
    
    return create_llm()