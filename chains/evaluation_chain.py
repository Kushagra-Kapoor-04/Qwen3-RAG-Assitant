

import json
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from config.settings import settings
from config.prompts import get_evaluation_prompt
from config.constants import EvaluationResult
from llm.qwen3_llm import Qwen3LLM, get_llm
from chains.rag_chain import RAGResponse
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationOutput:
    
    is_grounded: bool
    confidence: float
    result: EvaluationResult
    ungrounded_claims: List[str] = field(default_factory=list)
    explanation: str = ""
    raw_response: str = ""
    
    @property
    def needs_regeneration(self) -> bool:
        
        return not self.is_grounded and self.confidence > 0.5
    
    @property
    def is_partially_grounded(self) -> bool:
        
        return self.result == EvaluationResult.PARTIALLY_GROUNDED

class EvaluationChain:
    
    
    def __init__(
        self,
        llm: Optional[Qwen3LLM] = None,
        threshold: Optional[float] = None
    ):
        
        self._llm = llm
        self.threshold = threshold or settings.hallucination_threshold
    
    @property
    def llm(self) -> Qwen3LLM:
        
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    def evaluate(
        self,
        context: str,
        question: str,
        answer: str
    ) -> EvaluationOutput:
        
        if not answer or not answer.strip():
            return EvaluationOutput(
                is_grounded=True,
                confidence=1.0,
                result=EvaluationResult.GROUNDED,
                explanation="Empty answer"
            )
        
        if not context or not context.strip():
            return EvaluationOutput(
                is_grounded=False,
                confidence=0.0,
                result=EvaluationResult.UNGROUNDED,
                explanation="No context available for verification"
            )
        
        not_found_phrases = [
            "documents do not contain",
            "not found in the provided",
            "no information available",
            "cannot find",
            "not mentioned in"
        ]
        
        answer_lower = answer.lower()
        for phrase in not_found_phrases:
            if phrase in answer_lower:
                return EvaluationOutput(
                    is_grounded=True,
                    confidence=1.0,
                    result=EvaluationResult.GROUNDED,
                    explanation="Answer appropriately indicates missing information"
                )
        
        try:
            prompt = get_evaluation_prompt(context, question, answer)
            response = self.llm.generate(prompt, temperature=0.0)
            
            return self._parse_evaluation_response(response.content)
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return EvaluationOutput(
                is_grounded=True,  # Fail open
                confidence=0.5,
                result=EvaluationResult.UNCERTAIN,
                explanation=f"Evaluation error: {str(e)}"
            )
    
    def evaluate_rag_response(
        self,
        rag_response: RAGResponse
    ) -> EvaluationOutput:
        
        return self.evaluate(
            context=rag_response.context,
            question="",  # Question not needed if we have context
            answer=rag_response.answer
        )
    
    def _parse_evaluation_response(self, response: str) -> EvaluationOutput:
        
        try:
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            is_grounded = data.get("is_grounded", True)
            confidence = float(data.get("confidence", 0.5))
            ungrounded_claims = data.get("ungrounded_claims", [])
            explanation = data.get("explanation", "")
            
            if is_grounded:
                result = EvaluationResult.GROUNDED
            elif confidence < 0.3:
                result = EvaluationResult.UNCERTAIN
            elif len(ungrounded_claims) <= 1:
                result = EvaluationResult.PARTIALLY_GROUNDED
            else:
                result = EvaluationResult.UNGROUNDED
            
            return EvaluationOutput(
                is_grounded=is_grounded,
                confidence=confidence,
                result=result,
                ungrounded_claims=ungrounded_claims,
                explanation=explanation,
                raw_response=response
            )
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {str(e)}")
            
            response_lower = response.lower()
            is_grounded = "true" in response_lower and "grounded" in response_lower
            
            return EvaluationOutput(
                is_grounded=is_grounded,
                confidence=0.5,
                result=EvaluationResult.UNCERTAIN,
                explanation="Could not parse evaluation response",
                raw_response=response
            )
    
    def quick_check(self, context: str, answer: str) -> bool:
        
        if not answer or not context:
            return True
        
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        common_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
            'through', 'during', 'before', 'after', 'above', 'below',
            'between', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all',
            'each', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
            'too', 'very', 'just', 'and', 'but', 'if', 'or', 'because',
            'until', 'while', 'this', 'that', 'these', 'those', 'it'
        }
        
        answer_content = answer_words - common_words
        context_content = context_words - common_words
        
        if not answer_content:
            return True
        
        overlap = answer_content & context_content
        overlap_ratio = len(overlap) / len(answer_content)
        
        return overlap_ratio >= 0.5

def create_evaluation_chain(
    llm: Optional[Qwen3LLM] = None,
    threshold: Optional[float] = None
) -> EvaluationChain:
    
    return EvaluationChain(llm, threshold)

def get_evaluation_chain() -> EvaluationChain:
    
    return create_evaluation_chain()