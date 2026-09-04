

import pytest

class TestPromptTemplates:
    
    
    def test_rag_prompt_format(self):
        
        from config.prompts import get_rag_prompt
        
        context = "Python is a programming language."
        question = "What is Python?"
        
        prompt = get_rag_prompt(context, question)
        
        assert context in prompt
        assert question in prompt
        assert "strictly" in prompt.lower()
        assert "context" in prompt.lower()
    
    def test_rag_prompt_contains_instructions(self):
        
        from config.prompts import get_rag_prompt
        
        prompt = get_rag_prompt("context", "question")
        
        assert "do not contain" in prompt.lower()
        assert "answer" in prompt.lower()
    
    def test_evaluation_prompt_format(self):
        
        from config.prompts import get_evaluation_prompt
        
        context = "Test context"
        question = "Test question"
        answer = "Test answer"
        
        prompt = get_evaluation_prompt(context, question, answer)
        
        assert context in prompt
        assert question in prompt
        assert answer in prompt
        assert "grounded" in prompt.lower()
    
    def test_evaluation_prompt_json_structure(self):
        
        from config.prompts import get_evaluation_prompt
        
        prompt = get_evaluation_prompt("ctx", "q", "a")
        
        assert "json" in prompt.lower()
        assert "is_grounded" in prompt
        assert "confidence" in prompt
    
    def test_regeneration_prompt_format(self):
        
        from config.prompts import get_regeneration_prompt
        
        prompt = get_regeneration_prompt(
            context="Context here",
            question="Question here",
            previous_answer="Previous answer",
            issues="Issue list"
        )
        
        assert "Context here" in prompt
        assert "Question here" in prompt
        assert "Previous answer" in prompt
        assert "Issue list" in prompt
    
    def test_regeneration_prompt_instructions(self):
        
        from config.prompts import get_regeneration_prompt
        
        prompt = get_regeneration_prompt("c", "q", "a", "i")
        
        assert "context" in prompt.lower()
    
    def test_no_context_response(self):
        
        from config.prompts import NO_CONTEXT_RESPONSE
        
        assert "do not contain" in NO_CONTEXT_RESPONSE.lower()
    
    def test_default_responses(self):
        
        from config.prompts import DEFAULT_RESPONSES
        
        assert "no_documents" in DEFAULT_RESPONSES
        assert "empty_query" in DEFAULT_RESPONSES
        assert "retrieval_failed" in DEFAULT_RESPONSES

class TestPromptTemplate:
    
    
    def test_template_creation(self):
        
        from config.prompts import PromptTemplate
        
        template = PromptTemplate(template="Hello, {name}!")
        
        assert template.template == "Hello, {name}!"
    
    def test_template_format(self):
        
        from config.prompts import PromptTemplate
        
        template = PromptTemplate(template="Hello, {name}! You are {role}.")
        result = template.format(name="Alice", role="developer")
        
        assert result == "Hello, Alice! You are developer."
    
    def test_template_format_missing_var(self):
        
        from config.prompts import PromptTemplate
        
        template = PromptTemplate(template="Hello, {name}!")
        
        with pytest.raises(KeyError):
            template.format(wrong_var="value")

class TestConstants:
    
    
    def test_supported_extensions(self):
        
        from config.constants import SUPPORTED_EXTENSIONS, is_supported_extension
        
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS
        
        assert is_supported_extension(".pdf") == True
        assert is_supported_extension(".exe") == False
    
    def test_document_types(self):
        
        from config.constants import DocumentType
        
        assert DocumentType.PDF.value == "pdf"
        assert DocumentType.TXT.value == "txt"
    
    def test_evaluation_results(self):
        
        from config.constants import EvaluationResult
        
        assert EvaluationResult.GROUNDED.value == "grounded"
        assert EvaluationResult.UNGROUNDED.value == "ungrounded"
    
    def test_chunking_strategy(self):
        
        from config.constants import ChunkingStrategy
        
        assert ChunkingStrategy.RECURSIVE.value == "recursive"
        assert ChunkingStrategy.SENTENCE.value == "sentence"
    
    def test_error_messages(self):
        
        from config.constants import get_error_message
        
        msg = get_error_message("file_not_found", path="/test/path")
        
        assert "/test/path" in msg
        assert "not found" in msg.lower()
    
    def test_success_messages(self):
        
        from config.constants import get_success_message
        
        msg = get_success_message("ingestion_complete", count=10)
        
        assert "10" in msg
        assert "success" in msg.lower()

class TestSettings:
    
    
    def test_default_settings(self):
        
        from config.settings import Settings
        
        settings = Settings()
        
        assert settings.chunk_size == 500
        assert settings.chunk_overlap == 50
        assert settings.top_k_results == 5
    
    def test_settings_env_override(self, monkeypatch):
        
        monkeypatch.setenv("CHUNK_SIZE", "1000")
        
        from config.settings import Settings
        settings = Settings()
        
    
    def test_get_paths(self):
        
        from config.settings import Settings
        from pathlib import Path
        
        settings = Settings()
        
        faiss_path = settings.get_faiss_index_path()
        raw_path = settings.get_raw_data_path()
        
        assert isinstance(faiss_path, Path)
        assert isinstance(raw_path, Path)

@pytest.fixture
def sample_context():
    
    return "This is relevant information about the topic."

@pytest.fixture
def sample_question():
    
    return "What is the relevant information?"