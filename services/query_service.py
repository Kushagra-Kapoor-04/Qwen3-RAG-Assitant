

from typing import Optional, Tuple, Dict, Any, Iterator
from dataclasses import dataclass, field
from datetime import datetime

from config.settings import settings
from config.prompts import NO_CONTEXT_RESPONSE
from chains.rag_chain import RAGChain, RAGResponse, get_rag_chain
from chains.evaluation_chain import (
    EvaluationChain,
    EvaluationOutput,
    get_evaluation_chain
)
from chains.regeneration_chain import (
    RegenerationChain,
    RegenerationResult,
    get_regeneration_chain
)
from services.logging_service import get_logger, QueryLog

logger = get_logger(__name__)

@dataclass
class QueryResult:
    
    question: str
    answer: str
    context: str
    sources: list
    
    is_grounded: bool = True
    was_regenerated: bool = False
    regeneration_attempts: int = 0
    
    evaluation: Optional[EvaluationOutput] = None
    regeneration_result: Optional[RegenerationResult] = None
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "is_grounded": self.is_grounded,
            "was_regenerated": self.was_regenerated,
            "regeneration_attempts": self.regeneration_attempts,
            "timestamp": self.timestamp,
            "processing_time_ms": self.processing_time_ms
        }

class QueryService:
    
    
    def __init__(
        self,
        rag_chain: Optional[RAGChain] = None,
        evaluation_chain: Optional[EvaluationChain] = None,
        regeneration_chain: Optional[RegenerationChain] = None,
        enable_evaluation: bool = True,
        enable_regeneration: bool = True
    ):
        
        self._rag_chain = rag_chain
        self._evaluation_chain = evaluation_chain
        self._regeneration_chain = regeneration_chain
        self.enable_evaluation = enable_evaluation
        self.enable_regeneration = enable_regeneration
        
        self._query_history: list = []
    
    @property
    def rag_chain(self) -> RAGChain:
        
        if self._rag_chain is None:
            self._rag_chain = get_rag_chain()
        return self._rag_chain
    
    @property
    def evaluation_chain(self) -> EvaluationChain:
        
        if self._evaluation_chain is None:
            self._evaluation_chain = get_evaluation_chain()
        return self._evaluation_chain
    
    @property
    def regeneration_chain(self) -> RegenerationChain:
        
        if self._regeneration_chain is None:
            self._regeneration_chain = get_regeneration_chain()
        return self._regeneration_chain
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> QueryResult:
        
        import time
        start_time = time.time()
        
        logger.info(f"Processing query: {question[:50]}...")
        
        rag_response = self.rag_chain.query(question, top_k, threshold)
        
        if not rag_response.has_context:
            return self._create_result(
                question=question,
                answer=rag_response.answer,
                context="",
                sources=[],
                start_time=start_time
            )
        
        answer = rag_response.answer
        evaluation = None
        regeneration_result = None
        was_regenerated = False
        
        if self.enable_evaluation:
            evaluation = self.evaluation_chain.evaluate(
                context=rag_response.context,
                question=question,
                answer=answer
            )
            
            logger.info(f"Evaluation result: {evaluation.result.value}")
            
            if (
                self.enable_regeneration and
                not evaluation.is_grounded and
                evaluation.needs_regeneration
            ):
                regeneration_result = self.regeneration_chain.regenerate(
                    context=rag_response.context,
                    question=question,
                    original_answer=answer,
                    evaluation=evaluation
                )
                
                answer = regeneration_result.final_answer
                was_regenerated = regeneration_result.was_regenerated
                
                logger.info(
                    f"Regeneration: attempts={regeneration_result.attempts}, "
                    f"grounded={regeneration_result.is_grounded}"
                )
        
        result = self._create_result(
            question=question,
            answer=answer,
            context=rag_response.context,
            sources=rag_response.sources,
            evaluation=evaluation,
            regeneration_result=regeneration_result,
            was_regenerated=was_regenerated,
            start_time=start_time
        )
        
        return result

    def stream_query(
        self,
        question: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> Iterator[Any]:
        """
        Stream query results token by token.
        Yields tokens string by string, then yields a final QueryResult object.
        """
        import time
        start_time = time.time()
        
        logger.info(f"Streaming query: {question[:50]}...")
        
        # We need context for streaming too
        try:
            k = top_k or settings.top_k_results
            thresh = threshold or settings.similarity_threshold
            context, results = self.rag_chain.retriever.retrieve_with_context(
                question, top_k=k, threshold=thresh
            )
        except Exception as e:
            logger.error(f"Retrieval failed for stream: {str(e)}")
            yield NO_CONTEXT_RESPONSE
            return

        if not results:
            yield NO_CONTEXT_RESPONSE
            return

        sources = list(set(r.source for r in results))
        full_answer = ""
        
        # Stream from RAG chain
        try:
            for token in self.rag_chain.query_with_streaming(question, top_k=top_k, threshold=threshold):
                full_answer += token
                yield token
        except Exception as e:
            logger.error(f"Streaming failed: {str(e)}")
            yield "\n[Error during generation]"
            return

        # Perform evaluation after streaming finishes
        evaluation = None
        regeneration_result = None
        was_regenerated = False
        
        try:
            if self.enable_evaluation:
                evaluation = self.evaluation_chain.evaluate(
                    context=context,
                    question=question,
                    answer=full_answer
                )
                
                if (
                    self.enable_regeneration and
                    not evaluation.is_grounded and
                    evaluation.needs_regeneration
                ):
                    regeneration_result = self.regeneration_chain.regenerate(
                        context=context,
                        question=question,
                        original_answer=full_answer,
                        evaluation=evaluation
                    )
                    
                    full_answer = regeneration_result.final_answer
                    was_regenerated = regeneration_result.was_regenerated
        except Exception as e:
            logger.error(f"Evaluation/Regeneration failed: {str(e)}")
            # Continue to yield result even if evaluation failed

        result = self._create_result(
            question=question,
            answer=full_answer,
            context=context,
            sources=sources,
            evaluation=evaluation,
            regeneration_result=regeneration_result,
            was_regenerated=was_regenerated,
            start_time=start_time
        )
        
        self._log_query(result)
        
        # Yield the final result object so the UI can update metadata
        yield result
    
    def query_simple(self, question: str) -> str:
        
        result = self.query(question)
        return result.answer
    
    def query_with_sources(
        self,
        question: str
    ) -> Tuple[str, list]:
        
        result = self.query(question)
        return result.answer, result.sources
    
    def is_ready(self) -> bool:
        
        return self.rag_chain.is_ready()
    
    def get_query_history(self) -> list:
        
        return self._query_history.copy()
    
    def clear_history(self) -> None:
        
        self._query_history.clear()
    
    def _create_result(
        self,
        question: str,
        answer: str,
        context: str,
        sources: list,
        evaluation: Optional[EvaluationOutput] = None,
        regeneration_result: Optional[RegenerationResult] = None,
        was_regenerated: bool = False,
        start_time: float = 0.0
    ) -> QueryResult:
        
        import time
        processing_time = (time.time() - start_time) * 1000 if start_time else 0
        
        is_grounded = True
        if evaluation:
            is_grounded = evaluation.is_grounded
        if regeneration_result:
            is_grounded = regeneration_result.is_grounded
        
        return QueryResult(
            question=question,
            answer=answer,
            context=context,
            sources=sources,
            is_grounded=is_grounded,
            was_regenerated=was_regenerated,
            regeneration_attempts=(
                regeneration_result.attempts if regeneration_result else 0
            ),
            evaluation=evaluation,
            regeneration_result=regeneration_result,
            processing_time_ms=processing_time
        )
    
    def _log_query(self, result: QueryResult) -> None:
        
        self._query_history.append(result.to_dict())
        
        if len(self._query_history) > 100:
            self._query_history = self._query_history[-100:]

def create_query_service(
    enable_evaluation: bool = True,
    enable_regeneration: bool = True
) -> QueryService:
    
    return QueryService(
        enable_evaluation=enable_evaluation,
        enable_regeneration=enable_regeneration
    )

def get_query_service() -> QueryService:
    
    return create_query_service()