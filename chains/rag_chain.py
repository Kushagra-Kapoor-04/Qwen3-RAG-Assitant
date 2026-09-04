

from typing import Optional, Tuple, List
from dataclasses import dataclass

from config.settings import settings
from config.prompts import get_rag_prompt, NO_CONTEXT_RESPONSE, DEFAULT_RESPONSES
from vectorstore.retriever import Retriever, RetrievalResult, get_retriever
from llm.qwen3_llm import Qwen3LLM, LLMResponse, get_llm
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class RAGResponse:
    
    answer: str
    context: str
    sources: List[str]
    retrieval_results: List[RetrievalResult]
    llm_response: Optional[LLMResponse] = None
    is_grounded: bool = True
    confidence: float = 1.0
    
    @property
    def has_context(self) -> bool:
        
        return bool(self.context and self.context.strip())
    
    @property
    def source_count(self) -> int:
        
        return len(self.sources)

class RAGChain:
    
    
    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm: Optional[Qwen3LLM] = None,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        
        self._retriever = retriever
        self._llm = llm
        self.top_k = top_k or settings.top_k_results
        self.threshold = threshold or settings.similarity_threshold
    
    @property
    def retriever(self) -> Retriever:
        
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever
    
    @property
    def llm(self) -> Qwen3LLM:
        
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> RAGResponse:
        
        if not question or not question.strip():
            return RAGResponse(
                answer=DEFAULT_RESPONSES["empty_query"],
                context="",
                sources=[],
                retrieval_results=[],
                is_grounded=True
            )
        
        k = top_k or self.top_k
        thresh = threshold or self.threshold
        
        try:
            context, results = self.retriever.retrieve_with_context(
                question,
                top_k=k,
                threshold=thresh
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            return RAGResponse(
                answer=DEFAULT_RESPONSES["retrieval_failed"],
                context="",
                sources=[],
                retrieval_results=[],
                is_grounded=True
            )
        
        if not results:
            return RAGResponse(
                answer=NO_CONTEXT_RESPONSE,
                context="",
                sources=[],
                retrieval_results=[],
                is_grounded=True
            )
        
        sources = list(set(r.source for r in results))
        
        try:
            prompt = get_rag_prompt(context, question)
            llm_response = self.llm.generate(prompt)
            answer = llm_response.content.strip()
            
        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            return RAGResponse(
                answer=DEFAULT_RESPONSES["llm_error"],
                context=context,
                sources=sources,
                retrieval_results=results,
                is_grounded=False
            )
        
        return RAGResponse(
            answer=answer,
            context=context,
            sources=sources,
            retrieval_results=results,
            llm_response=llm_response,
            is_grounded=True
        )
    
    def query_with_streaming(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ):
        
        if not question or not question.strip():
            yield DEFAULT_RESPONSES["empty_query"]
            return
        
        k = top_k or self.top_k
        thresh = threshold or self.threshold
        
        try:
            context, results = self.retriever.retrieve_with_context(
                question,
                top_k=k,
                threshold=thresh
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {str(e)}")
            yield DEFAULT_RESPONSES["retrieval_failed"]
            return
        
        if not results:
            yield NO_CONTEXT_RESPONSE
            return
        
        prompt = get_rag_prompt(context, question)
        
        try:
            for token in self.llm.generate_stream(prompt):
                yield token
        except Exception as e:
            logger.error(f"LLM streaming failed: {str(e)}")
            yield DEFAULT_RESPONSES["llm_error"]
    
    def is_ready(self) -> bool:
        
        try:
            return self.retriever.is_ready()
        except Exception:
            return False

def create_rag_chain(
    retriever: Optional[Retriever] = None,
    llm: Optional[Qwen3LLM] = None,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None
) -> RAGChain:
    
    return RAGChain(retriever, llm, top_k, threshold)

def get_rag_chain() -> RAGChain:
    
    return create_rag_chain()