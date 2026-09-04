

import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from services.query_service import create_query_service, QueryResult
from chains.evaluation_chain import create_evaluation_chain
from vectorstore.retriever import get_retriever
from services.logging_service import get_logger

logger = get_logger(__name__)

@dataclass
class EvaluationMetrics:
    
    total_queries: int = 0
    grounded_answers: int = 0
    ungrounded_answers: int = 0
    regenerated_answers: int = 0
    avg_processing_time_ms: float = 0.0
    avg_sources_per_query: float = 0.0
    
    @property
    def grounding_rate(self) -> float:
        
        if self.total_queries == 0:
            return 0.0
        return self.grounded_answers / self.total_queries
    
    @property
    def regeneration_rate(self) -> float:
        
        if self.total_queries == 0:
            return 0.0
        return self.regenerated_answers / self.total_queries
    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "total_queries": self.total_queries,
            "grounded_answers": self.grounded_answers,
            "ungrounded_answers": self.ungrounded_answers,
            "regenerated_answers": self.regenerated_answers,
            "grounding_rate": round(self.grounding_rate, 4),
            "regeneration_rate": round(self.regeneration_rate, 4),
            "avg_processing_time_ms": round(self.avg_processing_time_ms, 2),
            "avg_sources_per_query": round(self.avg_sources_per_query, 2)
        }

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Evaluate RAG system performance"
    )
    
    parser.add_argument(
        "--queries", "-q",
        type=str,
        help="Path to JSON file with test queries"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output path for evaluation report"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_test_queries(path: str) -> List[Dict[str, str]]:
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_queries(
    queries: List[Dict[str, str]],
    verbose: bool = False
) -> tuple:
    
    query_service = create_query_service()
    
    if not query_service.is_ready():
        logger.error("Query service not ready. Run ingestion first.")
        return None, []
    
    metrics = EvaluationMetrics()
    results = []
    total_time = 0.0
    total_sources = 0
    
    for i, query_data in enumerate(queries, 1):
        question = query_data.get("question", "")
        expected = query_data.get("expected_answer", "")
        
        if not question:
            continue
        
        logger.info(f"Processing query {i}/{len(queries)}: {question[:50]}...")
        
        try:
            result = query_service.query(question)
            
            metrics.total_queries += 1
            total_time += result.processing_time_ms
            total_sources += len(result.sources)
            
            if result.is_grounded:
                metrics.grounded_answers += 1
            else:
                metrics.ungrounded_answers += 1
            
            if result.was_regenerated:
                metrics.regenerated_answers += 1
            
            result_data = {
                "question": question,
                "answer": result.answer,
                "expected": expected,
                "is_grounded": result.is_grounded,
                "was_regenerated": result.was_regenerated,
                "sources": result.sources,
                "processing_time_ms": result.processing_time_ms
            }
            results.append(result_data)
            
            if verbose:
                print(f"\nQ: {question}")
                print(f"A: {result.answer[:100]}...")
                print(f"Grounded: {result.is_grounded}, Regenerated: {result.was_regenerated}")
                
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
    
    if metrics.total_queries > 0:
        metrics.avg_processing_time_ms = total_time / metrics.total_queries
        metrics.avg_sources_per_query = total_sources / metrics.total_queries
    
    return metrics, results

def run_interactive_evaluation(verbose: bool = False):
    
    query_service = create_query_service()
    
    if not query_service.is_ready():
        print("ERROR: Query service not ready. Run ingestion first.")
        return
    
    print("\n" + "=" * 50)
    print("INTERACTIVE EVALUATION MODE")
    print("=" * 50)
    print("Enter queries to test. Type 'quit' to exit.")
    print("Type 'stats' to see statistics.")
    print("=" * 50 + "\n")
    
    queries_run = 0
    grounded = 0
    regenerated = 0
    
    while True:
        try:
            question = input("\nQuery: ").strip()
            
            if not question:
                continue
            
            if question.lower() == 'quit':
                break
            
            if question.lower() == 'stats':
                print(f"\nQueries: {queries_run}")
                print(f"Grounded: {grounded}")
                print(f"Regenerated: {regenerated}")
                if queries_run > 0:
                    print(f"Grounding rate: {grounded/queries_run:.1%}")
                continue
            
            result = query_service.query(question)
            queries_run += 1
            
            if result.is_grounded:
                grounded += 1
            if result.was_regenerated:
                regenerated += 1
            
            print(f"\nAnswer: {result.answer}")
            print(f"\n[Grounded: {result.is_grounded}]", end=" ")
            print(f"[Regenerated: {result.was_regenerated}]", end=" ")
            print(f"[Sources: {len(result.sources)}]", end=" ")
            print(f"[Time: {result.processing_time_ms:.0f}ms]")
            
            if verbose and result.sources:
                print("\nSources:")
                for src in result.sources[:3]:
                    print(f"  - {src}")
                    
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
    
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total queries:    {queries_run}")
    print(f"Grounded:         {grounded}")
    print(f"Regenerated:      {regenerated}")
    if queries_run > 0:
        print(f"Grounding rate:   {grounded/queries_run:.1%}")
    print("=" * 50)

def save_report(
    metrics: EvaluationMetrics,
    results: List[Dict],
    output_path: str
):
    
    report = {
        "metrics": metrics.to_dict(),
        "results": results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Report saved to: {output_path}")

def main():
    
    args = parse_args()
    
    settings.ensure_directories()
    
    if args.interactive:
        run_interactive_evaluation(args.verbose)
    elif args.queries:
        try:
            queries = load_test_queries(args.queries)
            metrics, results = evaluate_queries(queries, args.verbose)
            
            if metrics:
                print("\n" + "=" * 50)
                print("EVALUATION RESULTS")
                print("=" * 50)
                for key, value in metrics.to_dict().items():
                    print(f"{key}: {value}")
                print("=" * 50)
                
                if args.output:
                    save_report(metrics, results, args.output)
            
        except FileNotFoundError:
            print(f"Queries file not found: {args.queries}")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Invalid JSON in queries file: {args.queries}")
            sys.exit(1)
    else:
        print("Running simple evaluation check...")
        
        query_service = create_query_service()
        
        if not query_service.is_ready():
            print("ERROR: Query service not ready. Run ingestion first.")
            print("Usage: python scripts/ingest.py --source ./data/raw")
            sys.exit(1)
        
        test_result = query_service.query("What information is in the documents?")
        
        print("\nTest query result:")
        print(f"  Grounded: {test_result.is_grounded}")
        print(f"  Sources: {len(test_result.sources)}")
        print(f"  Time: {test_result.processing_time_ms:.0f}ms")
        
        print("\nSystem is ready for evaluation.")
        print("\nOptions:")
        print("  --interactive    Run interactive mode")
        print("  --queries FILE   Evaluate queries from JSON file")
        print("  --help           Show all options")

if __name__ == "__main__":
    main()