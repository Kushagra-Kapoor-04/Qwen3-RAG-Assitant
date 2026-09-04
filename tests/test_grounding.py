

import pytest
from unittest.mock import Mock, patch, MagicMock

class TestEvaluationChain:
    
    
    def test_evaluate_grounded_answer(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content='{"is_grounded": true, "confidence": 0.9, "ungrounded_claims": [], "explanation": "Answer is well grounded"}'
        )
        
        chain = EvaluationChain(llm=mock_llm)
        
        context = "Python is a programming language created by Guido van Rossum."
        question = "Who created Python?"
        answer = "Python was created by Guido van Rossum."
        
        result = chain.evaluate(context, question, answer)
        
        assert result.is_grounded == True
        assert result.confidence > 0.5
    
    def test_evaluate_ungrounded_answer(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content='{"is_grounded": false, "confidence": 0.8, "ungrounded_claims": ["Claim about being the most popular"], "explanation": "The context does not claim Python is the most popular"}'
        )
        
        chain = EvaluationChain(llm=mock_llm)
        
        context = "Python is a programming language."
        question = "Tell me about Python"
        answer = "Python is the most popular programming language in the world."
        
        result = chain.evaluate(context, question, answer)
        
        assert result.is_grounded == False
        assert len(result.ungrounded_claims) > 0
    
    def test_evaluate_empty_answer(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        chain = EvaluationChain()
        
        result = chain.evaluate(
            context="Some context",
            question="Some question",
            answer=""
        )
        
        assert result.is_grounded == True
    
    def test_evaluate_no_context(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        chain = EvaluationChain()
        
        result = chain.evaluate(
            context="",
            question="Question",
            answer="Some answer"
        )
        
        assert result.is_grounded == False
    
    def test_evaluate_not_found_response(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        chain = EvaluationChain()
        
        result = chain.evaluate(
            context="Some unrelated context",
            question="Question about something else",
            answer="The provided documents do not contain this information."
        )
        
        assert result.is_grounded == True
    
    def test_quick_check_grounded(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        chain = EvaluationChain()
        
        context = "Python is a programming language used for web development."
        answer = "Python is used for web development."
        
        is_grounded = chain.quick_check(context, answer)
        
        assert is_grounded == True
    
    def test_quick_check_ungrounded(self):
        
        from chains.evaluation_chain import EvaluationChain
        
        chain = EvaluationChain()
        
        context = "Python is a programming language."
        answer = "Java is the best language for mobile development and runs on virtual machines."
        
        is_grounded = chain.quick_check(context, answer)
        
        assert is_grounded == False

class TestRegenerationChain:
    
    
    def test_regenerate_improves_answer(self):
        
        from chains.regeneration_chain import RegenerationChain
        from chains.evaluation_chain import EvaluationOutput, EvaluationResult
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="Based on the context, Python is a programming language."
        )
        
        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = EvaluationOutput(
            is_grounded=True,
            confidence=0.9,
            result=EvaluationResult.GROUNDED
        )
        
        chain = RegenerationChain(llm=mock_llm, evaluation_chain=mock_eval)
        
        initial_evaluation = EvaluationOutput(
            is_grounded=False,
            confidence=0.8,
            result=EvaluationResult.UNGROUNDED,
            ungrounded_claims=["Python is the best language"]
        )
        
        result = chain.regenerate(
            context="Python is a programming language.",
            question="What is Python?",
            original_answer="Python is the best programming language.",
            evaluation=initial_evaluation
        )
        
        assert result.was_regenerated == True
        assert result.is_grounded == True
    
    def test_regenerate_no_context(self):
        
        from chains.regeneration_chain import RegenerationChain
        from chains.evaluation_chain import EvaluationOutput, EvaluationResult
        
        chain = RegenerationChain()
        
        evaluation = EvaluationOutput(
            is_grounded=False,
            confidence=0.8,
            result=EvaluationResult.UNGROUNDED
        )
        
        result = chain.regenerate(
            context="",
            question="Question",
            original_answer="Some answer",
            evaluation=evaluation
        )
        
        assert "do not contain" in result.final_answer.lower()
    
    def test_max_attempts_reached(self):
        
        from chains.regeneration_chain import RegenerationChain
        from chains.evaluation_chain import EvaluationOutput, EvaluationResult
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = MagicMock(
            content="Still ungrounded answer"
        )
        
        mock_eval = MagicMock()
        mock_eval.evaluate.return_value = EvaluationOutput(
            is_grounded=False,
            confidence=0.8,
            result=EvaluationResult.UNGROUNDED,
            ungrounded_claims=["claim"]
        )
        
        chain = RegenerationChain(
            llm=mock_llm,
            evaluation_chain=mock_eval,
            max_attempts=2
        )
        
        evaluation = EvaluationOutput(
            is_grounded=False,
            confidence=0.8,
            result=EvaluationResult.UNGROUNDED,
            ungrounded_claims=["claim"]
        )
        
        result = chain.regenerate(
            context="Some context",
            question="Question",
            original_answer="Ungrounded answer",
            evaluation=evaluation
        )
        
        assert result.attempts >= 1

class TestRAGChain:
    
    
    def test_query_empty_question(self):
        
        from chains.rag_chain import RAGChain
        
        chain = RAGChain()
        result = chain.query("")
        
        assert result.answer != ""
        assert "empty" in result.answer.lower() or "provide" in result.answer.lower()
    
    def test_query_no_results(self):
        
        from chains.rag_chain import RAGChain
        from vectorstore.retriever import Retriever
        
        mock_retriever = MagicMock()
        mock_retriever.retrieve_with_context.return_value = ("", [])
        
        chain = RAGChain(retriever=mock_retriever)
        result = chain.query("What is Python?")
        
        assert "do not contain" in result.answer.lower()
        assert result.has_context == False

class TestGroundingHelpers:
    
    
    def test_format_issues(self):
        
        from chains.regeneration_chain import RegenerationChain
        from chains.evaluation_chain import EvaluationOutput, EvaluationResult
        
        chain = RegenerationChain()
        
        evaluation = EvaluationOutput(
            is_grounded=False,
            confidence=0.8,
            result=EvaluationResult.UNGROUNDED,
            ungrounded_claims=["Claim 1", "Claim 2"],
            explanation="These claims are not in context"
        )
        
        issues = chain._format_issues(evaluation)
        
        assert "Claim 1" in issues
        assert "Claim 2" in issues
        assert "not in context" in issues

@pytest.fixture
def mock_llm():
    
    llm = MagicMock()
    llm.generate.return_value = MagicMock(
        content="Generated response"
    )
    return llm

@pytest.fixture
def grounded_context():
    
    return 

@pytest.fixture
def sample_question():
    
    return "Who created Python and when?"

@pytest.fixture
def grounded_answer():
    
    return "Python was created by Guido van Rossum in 1991."

@pytest.fixture
def ungrounded_answer():
    
    return "Python was created in 2000 and is the most popular language ever."