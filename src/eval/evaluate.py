# src/eval/evaluate.py

import sys
from pathlib import Path
# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parents[2]))

import yaml
from src.rag.rag_pipeline import RAGPipeline

def load_tests(path: Path):
    return yaml.safe_load(path.read_text())

def evaluate(pipeline, tests, top_k=5):
    total = len(tests)
    correct = 0

    for t in tests:
        query = t["query"]
        expected = t["expected"]
        result = pipeline(query, top_k)
        answer = result["answer"]

        hit = expected.lower() in answer.lower()
        status = "✓" if hit else "✗"
        print(f"Q: {query}\nExpected: {expected}\nGot: {answer}\nResult: {status}\n")

        if hit:
            correct += 1

    accuracy = correct / total * 100
    print(f"Overall Accuracy: {correct}/{total} = {accuracy:.2f}%")

if __name__ == "__main__":
    pipeline = RAGPipeline()
    tests = load_tests(Path(__file__).parents[2] / "tests" / "test_queries.yaml")
    evaluate(pipeline, tests, top_k=5)
