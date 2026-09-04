

from typing import List, Optional, Tuple
import re

from vectorstore.retriever import RetrievalResult
from config.constants import MAX_CONTEXT_TOKENS, METADATA_KEYS
from services.logging_service import get_logger

logger = get_logger(__name__)

def format_context(
    results: List[RetrievalResult],
    include_sources: bool = True,
    include_scores: bool = False,
    separator: str = "\n\n---\n\n"
) -> str:
    
    if not results:
        return ""
    
    formatted_parts = []
    
    for i, result in enumerate(results, 1):
        parts = []
        
        if include_sources:
            source_name = result.source.split('/')[-1].split('\\')[-1]
            header = f"[Source {i}: {source_name}"
            if include_scores:
                header += f" (Score: {result.score:.2f})"
            header += "]"
            parts.append(header)
        elif include_scores:
            parts.append(f"[Score: {result.score:.2f}]")
        
        parts.append(result.content)
        
        formatted_parts.append("\n".join(parts))
    
    return separator.join(formatted_parts)

def truncate_context(
    context: str,
    max_length: int = MAX_CONTEXT_TOKENS * 4,  # Approximate chars per token
    preserve_start: bool = True
) -> str:
    
    if len(context) <= max_length:
        return context
    
    if preserve_start:
        truncated = context[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:
            truncated = truncated[:last_period + 1]
        return truncated + "\n[Context truncated...]"
    else:
        truncated = context[-max_length:]
        first_period = truncated.find('.')
        if first_period < max_length * 0.2 and first_period > 0:
            truncated = truncated[first_period + 1:]
        return "[...Context truncated]\n" + truncated

def extract_key_sentences(
    text: str,
    query: str,
    num_sentences: int = 5
) -> str:
    
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    if len(sentences) <= num_sentences:
        return text
    
    query_words = set(query.lower().split())
    
    scored = []
    for i, sentence in enumerate(sentences):
        sentence_words = set(sentence.lower().split())
        overlap = len(query_words & sentence_words)
        scored.append((overlap, i, sentence))
    
    scored.sort(key=lambda x: (-x[0], x[1]))
    
    top_sentences = sorted(scored[:num_sentences], key=lambda x: x[1])
    
    return ' '.join(s[2] for s in top_sentences)

def merge_overlapping_chunks(
    chunks: List[str],
    similarity_threshold: float = 0.8
) -> List[str]:
    
    if len(chunks) <= 1:
        return chunks
    
    merged = [chunks[0]]
    
    for chunk in chunks[1:]:
        similarity = _compute_overlap(merged[-1], chunk)
        
        if similarity >= similarity_threshold:
            merged[-1] = _merge_texts(merged[-1], chunk)
        else:
            merged.append(chunk)
    
    return merged

def _compute_overlap(text1: str, text2: str) -> float:
    
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    smaller = min(len(words1), len(words2))
    
    return intersection / smaller if smaller > 0 else 0.0

def _merge_texts(text1: str, text2: str) -> str:
    
    for i in range(min(len(text1), len(text2)), 0, -1):
        if text1[-i:] == text2[:i]:
            return text1 + text2[i:]
    
    return text1 + " " + text2

def deduplicate_results(
    results: List[RetrievalResult],
    similarity_threshold: float = 0.9
) -> List[RetrievalResult]:
    
    if len(results) <= 1:
        return results
    
    unique = [results[0]]
    
    for result in results[1:]:
        is_duplicate = False
        
        for existing in unique:
            similarity = _compute_overlap(result.content, existing.content)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(result)
    
    return unique

def get_source_summary(results: List[RetrievalResult]) -> str:
    
    if not results:
        return "No sources used."
    
    sources = {}
    for result in results:
        source = result.source.split('/')[-1].split('\\')[-1]
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    parts = [f"- {name}: {count} chunk(s)" for name, count in sources.items()]
    
    return "Sources used:\n" + "\n".join(parts)

def estimate_tokens(text: str) -> int:
    
    return len(text) // 4

def fit_context_to_limit(
    results: List[RetrievalResult],
    max_tokens: int = MAX_CONTEXT_TOKENS
) -> Tuple[str, List[RetrievalResult]]:
    
    if not results:
        return "", []
    
    max_chars = max_tokens * 4  # Approximate
    used_results = []
    total_chars = 0
    
    for result in results:
        result_chars = len(result.content) + 50  # Buffer for formatting
        if total_chars + result_chars <= max_chars:
            used_results.append(result)
            total_chars += result_chars
        else:
            break
    
    context = format_context(used_results)
    return context, used_results