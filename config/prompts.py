
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class PromptTemplate:
    template: str
    
    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)

RAG_SYSTEM_PROMPT = PromptTemplate(
    template="""You are Qwen3 RAG Assistant, a professional Retrieval-Augmented Generation system.

Your task is to answer the user's question using strictly and ONLY the retrieved context.

You may rephrase, summarize, or logically infer definitions ONLY if the information is clearly implied by the context.

You must NOT use any external knowledge.

If the context contains partial or indirect information, you must construct the most accurate answer possible using that information.

If the context does NOT contain enough information to answer, respond:

"The provided documents do not contain sufficient information to answer this question."

Rules:
- No hallucinations.
- No external knowledge.
- Use only retrieved context.
- Prefer explanation over refusal when possible.
- Be precise, technical, and professional.
- Do not copy sentences blindly — synthesize from context.
- If multiple interpretations exist, mention them.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:
Analyze the retrieved context carefully.
If the answer is implicitly present, infer it from the context.
If the answer is explicitly present, quote or summarize it.
If the answer is not present, clearly refuse.

Provide a complete and accurate answer now."""
)

HALLUCINATION_EVALUATION_PROMPT = PromptTemplate(
    template="""You are an expert evaluator. Your task is to check if the generated answer is fully grounded in the provided context.

Context:
{context}

User Question:
{question}

Generated Answer:
{answer}

Evaluate the answer. Check if every claim in the answer is supported by the context.
Return your response in JSON format with the following keys:
- "is_grounded": boolean (true if fully supported, false otherwise)
- "missing_info": list of strings (claims not found in context)
- "confidence": float (0.0 to 1.0)

JSON Response:"""
)

REGENERATION_PROMPT = PromptTemplate(
    template="""The previous answer you generated was not fully grounded in the context. Please rewrite it.

Context:
{context}

User Question:
{question}

Previous Answer:
{previous_answer}

Evaluation Issues:
{issues}

Instructions:
1. Remove any information that is not supported by the context.
2. Keep the answer accurate to the context.
3. strictly follow the context.

corrected Answer:"""
)

CONTEXT_SUMMARY_PROMPT = PromptTemplate(
    template="""Summarize the following text, keeping key information relevant to: {query}

Text:
{text}

Summary:"""
)

QUERY_REFORMULATION_PROMPT = PromptTemplate(
    template="""Reformulate the following user query to be more specific and suitable for retrieval.

Original Query: {query}

Reformulated Query:"""
)

NO_CONTEXT_RESPONSE = (
    "The provided documents do not contain this information. "
    "Please try rephrasing your question or providing additional documents."
)

DEFAULT_RESPONSES = {
    "no_documents": "No documents have been ingested yet. Please run the ingestion pipeline first.",
    "empty_query": "Please provide a question to search for.",
    "retrieval_failed": "Failed to retrieve relevant documents. Please try again.",
    "llm_error": "An error occurred while generating the response. Please try again.",
    "invalid_input": "Invalid input provided. Please check your query and try again."
}

def get_rag_prompt(context: str, question: str) -> str:
    return RAG_SYSTEM_PROMPT.format(context=context, question=question)

def get_evaluation_prompt(context: str, question: str, answer: str) -> str:
    return HALLUCINATION_EVALUATION_PROMPT.format(
        context=context,
        question=question,
        answer=answer
    )

def get_regeneration_prompt(
    context: str,
    question: str,
    previous_answer: str,
    issues: str
) -> str:
    return REGENERATION_PROMPT.format(
        context=context,
        question=question,
        previous_answer=previous_answer,
        issues=issues
    )