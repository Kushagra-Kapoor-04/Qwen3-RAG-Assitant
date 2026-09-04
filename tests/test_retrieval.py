

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

class TestDocumentLoader:
    
    
    def test_load_txt_file(self, tmp_path):
        
        from ingestion.loader import DocumentLoader, Document
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is test content.")
        
        loader = DocumentLoader()
        doc = loader.load_file(str(test_file))
        
        assert doc is not None
        assert doc.content == "This is test content."
        assert doc.document_type == "txt"
    
    def test_load_nonexistent_file(self):
        
        from ingestion.loader import DocumentLoader
        
        loader = DocumentLoader()
        doc = loader.load_file("/nonexistent/file.txt")
        
        assert doc is None
    
    def test_load_unsupported_format(self, tmp_path):
        
        from ingestion.loader import DocumentLoader
        
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content")
        
        loader = DocumentLoader()
        doc = loader.load_file(str(test_file))
        
        assert doc is None
    
    def test_load_directory(self, tmp_path):
        
        from ingestion.loader import DocumentLoader
        
        (tmp_path / "doc1.txt").write_text("Content 1")
        (tmp_path / "doc2.txt").write_text("Content 2")
        (tmp_path / "ignore.xyz").write_text("Ignored")
        
        loader = DocumentLoader()
        docs = loader.load_directory(str(tmp_path))
        
        assert len(docs) == 2
        assert {doc.content for doc in docs} == {"Content 1", "Content 2"}

class TestTextSplitter:
    
    
    def test_split_document(self, tmp_path):
        
        from ingestion.loader import Document
        from ingestion.splitter import TextSplitter
        
        doc = Document(
            content="This is sentence one. This is sentence two. " * 20,
            metadata={"source": "test.txt"}
        )
        
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_document(doc)
        
        assert len(chunks) > 1
        assert all(len(chunk.content) <= 200 for chunk in chunks)  # Allow some flexibility
    
    def test_split_empty_document(self):
        
        from ingestion.loader import Document
        from ingestion.splitter import TextSplitter
        
        doc = Document(content="", metadata={})
        
        splitter = TextSplitter()
        chunks = splitter.split_document(doc)
        
        assert chunks == []
    
    def test_chunk_metadata_preserved(self):
        
        from ingestion.loader import Document
        from ingestion.splitter import TextSplitter
        
        doc = Document(
            content="Some content that will be chunked." * 10,
            metadata={"source": "test.txt", "author": "Test"}
        )
        
        splitter = TextSplitter(chunk_size=50)
        chunks = splitter.split_document(doc)
        
        for chunk in chunks:
            assert chunk.metadata.get("source") == "test.txt"
            assert chunk.metadata.get("author") == "Test"

class TestEmbedder:
    
    
    @patch('ingestion.embedder.SentenceTransformer')
    def test_embed_text(self, mock_transformer):
        
        from ingestion.embedder import Embedder
        
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embedding = embedder.embed_text("Test text")
        
        assert embedding.shape == (384,)
    
    @patch('ingestion.embedder.SentenceTransformer')
    def test_embed_empty_text(self, mock_transformer):
        
        from ingestion.embedder import Embedder
        
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder()
        embedding = embedder.embed_text("")
        
        assert embedding.shape == (384,)
        assert np.allclose(embedding, np.zeros(384))

class TestFAISSStore:
    
    
    @patch('vectorstore.faiss_store.faiss')
    def test_create_store(self, mock_faiss):
        
        from vectorstore.faiss_store import FAISSStore
        
        mock_index = MagicMock()
        mock_faiss.IndexFlatIP.return_value = mock_index
        
        store = FAISSStore(dimension=384)
        
        assert store.num_documents == 0
    
    @patch('vectorstore.faiss_store.faiss')
    def test_add_and_search(self, mock_faiss):
        
        from vectorstore.faiss_store import FAISSStore
        from ingestion.embedder import EmbeddedChunk
        from ingestion.splitter import TextChunk
        
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9, 0.8]]),
            np.array([[0, 1]])
        )
        mock_faiss.IndexFlatIP.return_value = mock_index
        mock_faiss.normalize_L2 = MagicMock()
        
        store = FAISSStore(dimension=384)
        
        chunks = [
            EmbeddedChunk(
                chunk=TextChunk(content="Content 1", metadata={"source": "doc1.txt"}),
                embedding=np.random.randn(384).astype(np.float32)
            ),
            EmbeddedChunk(
                chunk=TextChunk(content="Content 2", metadata={"source": "doc2.txt"}),
                embedding=np.random.randn(384).astype(np.float32)
            )
        ]
        
        store.add_embeddings(chunks)
        
        assert store.num_documents == 2

class TestRetriever:
    
    
    def test_retrieve_no_store(self):
        
        from vectorstore.retriever import Retriever
        
        retriever = Retriever()
        
        with pytest.raises(ValueError):
            retriever.retrieve("test query")

@pytest.fixture
def sample_documents():
    
    from ingestion.loader import Document
    
    return [
        Document(
            content="Python is a programming language.",
            metadata={"source": "python.txt"}
        ),
        Document(
            content="Machine learning is a subset of AI.",
            metadata={"source": "ml.txt"}
        )
    ]

@pytest.fixture
def sample_chunks(sample_documents):
    
    from ingestion.splitter import TextSplitter
    
    splitter = TextSplitter(chunk_size=100)
    chunks = []
    for doc in sample_documents:
        chunks.extend(splitter.split_document(doc))
    return chunks