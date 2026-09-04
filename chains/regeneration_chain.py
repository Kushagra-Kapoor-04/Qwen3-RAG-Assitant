

from typing import Optional, List
from dataclasses import dataclass, field

from config.settings import settings
from config.prompts import get_regeneration_prompt, NO_CONTEXT_RESPONSE
from llm.qwen3_llm import Qwen3LLM, LLMResponse, get_llm
from chains.rag_chain import RAGResponse, RAGChain
from chains.evaluation_chain import (
    EvaluationChain,
    EvaluationOutput,
    get_evaluation_chain
)
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class RegenerationResult:
    
    final_answer: str
    attempts: int
    was_regenerated: bool
    evaluation_history: List[EvaluationOutput] = field(default_factory=list)
    answer_history: List[str] = field(default_factory=list)
    is_grounded: bool = True
    
    @property
    def converged(self) -> bool:
        
        return self.is_grounded
    
    @property
    def max_attempts_reached(self) -> bool:
        
        return self.attempts >= settings.max_regeneration_attempts

class RegenerationChain:
    
    
    def __init__(
        self,
        llm: Optional[Qwen3LLM] = None,
        evaluation_chain: Optional[EvaluationChain] = None,
        max_attempts: Optional[int] = None
    ):
        
        self._llm = llm
        self._evaluation_chain = evaluation_chain
        self.max_attempts = max_attempts or settings.max_regeneration_attempts
    
    @property
    def llm(self) -> Qwen3LLM:
        
        if self._llm is None:
            self._llm = get_llm()
        return self._llm
    
    @property
    def evaluation_chain(self) -> EvaluationChain:
        
        if self._evaluation_chain is None:
            self._evaluation_chain = get_evaluation_chain()
        return self._evaluation_chain
    
    def regenerate(
        self,
        context: str,
        question: str,
        original_answer: str,
        evaluation: EvaluationOutput
    ) -> RegenerationResult:
        
        if not context or not context.strip():
            return RegenerationResult(
                final_answer=NO_CONTEXT_RESPONSE,
                attempts=0,
                was_regenerated=False,
                is_grounded=True
            )
        
        current_answer = original_answer
        current_evaluation = evaluation
        
        answer_history = [original_answer]
        evaluation_history = [evaluation]
        
        attempt = 0
        
        while (
            not current_evaluation.is_grounded and
            attempt < self.max_attempts
        ):
            attempt += 1
            logger.info(f"Regeneration attempt {attempt}/{self.max_attempts}")
            
            issues = self._format_issues(current_evaluation)
            
            try:
                prompt = get_regeneration_prompt(
                    context=context,
                    question=question,
                    previous_answer=current_answer,
                    issues=issues
                )
                
                response = self.llm.generate(prompt, temperature=0.0)
                new_answer = response.content.strip()
                
            except Exception as e:
                logger.error(f"Regeneration failed: {str(e)}")
                break
            
            if self._is_same_answer(new_answer, current_answer):
                logger.warning("Regeneration produced same answer")
                break
            
            new_evaluation = self.evaluation_chain.evaluate(
                context, question, new_answer
            )
            
            current_answer = new_answer
            current_evaluation = new_evaluation
            
            answer_history.append(new_answer)
            evaluation_history.append(new_evaluation)
            
            if new_evaluation.is_grounded:
                logger.info("Regeneration successful - answer is grounded")
                break
        
        if not current_evaluation.is_grounded:
            logger.warning("Max attempts reached, using conservative response")
            current_answer = self._create_conservative_response(
                context, question, answer_history
            )
        
        return RegenerationResult(
            final_answer=current_answer,
            attempts=attempt,
            was_regenerated=attempt > 0,
            evaluation_history=evaluation_history,
            answer_history=answer_history,
            is_grounded=current_evaluation.is_grounded
        )
    
    def regenerate_rag_response(
        self,
        rag_response: RAGResponse,
        evaluation: EvaluationOutput,
        question: str
    ) -> RegenerationResult:
        
        return self.regenerate(
            context=rag_response.context,
            question=question,
            original_answer=rag_response.answer,
            evaluation=evaluation
        )
    
    def _format_issues(self, evaluation: EvaluationOutput) -> str:
        
        issues = []
        
        if evaluation.ungrounded_claims:
            issues.append("Ungrounded claims:")
            for claim in evaluation.ungrounded_claims:
                issues.append(f"  - {claim}")
        
        if evaluation.explanation:
            issues.append(f"\nExplanation: {evaluation.explanation}")
        
        return "\n".join(issues) if issues else "Answer contains unverified information."
    
    def _is_same_answer(self, new_answer: str, old_answer: str) -> bool:
        
        def normalize(text):
            return ' '.join(text.lower().split())
        
        return normalize(new_answer) == normalize(old_answer)
    
    def _create_conservative_response(
        self,
        context: str,
        question: str,
        answer_history: List[str]
    ) -> str:
        
        history_block = "\n\n".join(
            f"Attempt {i + 1}: {ans}" for i, ans in enumerate(answer_history)
        )

        prompt = (
            "You were unable to produce a fully grounded answer to the "
            "question below after multiple attempts. Instead of guessing, "
            "write a brief, honest response that only states what is "
            "directly supported by the context, and explicitly notes which "
            "parts of the question the context does not answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Previous attempts (all contained unverifiable claims):\n{history_block}\n\n"
            "Conservative, fully-grounded answer:"
        )

        try:
            response = self.llm.generate(prompt, temperature=0.0)
            return response.content.strip()
        except Exception:
            return NO_CONTEXT_RESPONSE

def create_regeneration_chain(
    llm: Optional[Qwen3LLM] = None,
    evaluation_chain: Optional[EvaluationChain] = None,
    max_attempts: Optional[int] = None
) -> RegenerationChain:
    
    return RegenerationChain(llm, evaluation_chain, max_attempts)

def get_regeneration_chain() -> RegenerationChain:
    
    return create_regeneration_chain()