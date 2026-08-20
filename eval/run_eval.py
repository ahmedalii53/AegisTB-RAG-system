"""
WHO TB Clinical RAG — Evaluation Runner
----------------------------------------
Runs all benchmark questions from TB_Clinical_Benchmark.csv through the
full RAG pipeline and computes:
  - Answer quality (confidence level match)
  - Citation presence
  - Refusal correctness for out-of-scope questions
  - Summary Precision@k report

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --benchmark eval/Day4_Starter_Benchmark.csv
    python eval/run_eval.py --json          # Output full JSON results
"""
import sys
import csv
import json
import time
import argparse
import io
from pathlib import Path

# Fix Windows console UTF-8 output
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent dir to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from query import run_rag_pipeline

DEFAULT_BENCHMARK = Path(__file__).resolve().parent / "TB_Clinical_Benchmark.csv"


def load_benchmark(csv_path: Path):
    """Loads benchmark CSV into list of dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def evaluate_row(row: dict, result: dict) -> dict:
    """
    Evaluates a single benchmark row against the pipeline result.
    Returns an evaluation dict with pass/fail flags.
    """
    expected_behavior = row.get("Expected Behavior", "").lower()
    category = row.get("Category", "").lower()

    confidence = result.get("confidence", "").lower()
    citations = result.get("citations", [])
    recommendation = result.get("recommendation", "").lower()

    is_refusal_expected = (
        "refuse" in expected_behavior
        or "insufficient" in expected_behavior
        or "out-of-scope" in category
        or "refusal" in category
    )

    # --- Confidence pass ---
    if is_refusal_expected:
        conf_pass = confidence in ["insufficient", "low"]
    else:
        conf_pass = confidence in ["high", "medium"]

    # --- Citation pass ---
    if is_refusal_expected:
        citation_pass = True  # Refusal may have empty citations — that's correct
    else:
        citation_pass = len(citations) >= 1

    # --- Refusal content pass ---
    if is_refusal_expected and confidence == "insufficient":
        refusal_pass = (
            result.get("evidence", "") == ""
            and result.get("citations", []) == []
        )
    else:
        refusal_pass = True

    # --- PDF metadata pass (for non-refusals) ---
    pdf_meta_pass = True
    if not is_refusal_expected and citations:
        cit = citations[0]
        pdf_meta_pass = (
            "file_name" in cit
            and isinstance(cit.get("page"), int)
            and cit.get("page", 0) >= 1
        )

    passed = conf_pass and citation_pass and refusal_pass and pdf_meta_pass

    return {
        "passed": passed,
        "conf_pass": conf_pass,
        "citation_pass": citation_pass,
        "refusal_pass": refusal_pass,
        "pdf_meta_pass": pdf_meta_pass,
        "is_refusal_expected": is_refusal_expected,
        "confidence": confidence,
        "num_citations": len(citations),
    }


def run_evaluation(csv_path: Path, output_json: bool = False):
    """Main evaluation loop."""
    print(f"\n{'='*70}")
    print(f"WHO TB Clinical RAG — Benchmark Evaluation")
    print(f"Source: {csv_path.name}")
    print(f"{'='*70}\n")

    rows = load_benchmark(csv_path)
    total = len(rows)
    results_full = []
    passed_count = 0

    for i, row in enumerate(rows, 1):
        question = row.get("Question", "").strip()
        if not question:
            continue

        category = row.get("Category", "N/A")
        expected = row.get("Expected Source (Document / Section / Page)", "N/A")
        expected_behavior = row.get("Expected Behavior", "N/A")

        print(f"[{i}/{total}] {category}")
        print(f"  Q: {question[:100]}{'...' if len(question) > 100 else ''}")

        start = time.time()
        try:
            result = run_rag_pipeline(question)
            elapsed = time.time() - start
        except Exception as e:
            print(f"  ❌ ERROR: {e}\n")
            results_full.append({
                "question": question,
                "category": category,
                "error": str(e),
                "passed": False
            })
            continue

        eval_result = evaluate_row(row, result)

        status = "[PASS]" if eval_result["passed"] else "[FAIL]"
        print(f"  Confidence: {eval_result['confidence'].upper():12s} | Citations: {eval_result['num_citations']} | {status} ({elapsed:.1f}s)")

        if not eval_result["conf_pass"]:
            print(f"  [!] Confidence mismatch — got '{eval_result['confidence']}', expected {'refusal' if eval_result['is_refusal_expected'] else 'high/medium'}")
        if not eval_result["citation_pass"]:
            print(f"  [!] Missing citations for non-refusal answer")

        print(f"  Rec: {result.get('recommendation', '')[:120]}...")
        print()

        if eval_result["passed"]:
            passed_count += 1

        results_full.append({
            "question": question,
            "category": category,
            "expected_source": expected,
            "expected_behavior": expected_behavior,
            "confidence": eval_result["confidence"],
            "num_citations": eval_result["num_citations"],
            "passed": eval_result["passed"],
            "details": eval_result,
            "recommendation_preview": result.get("recommendation", "")[:200],
            "citations": result.get("citations", []),
        })

    # Summary
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"  Total Questions : {total}")
    print(f"  Passed          : {passed_count}/{total}")
    print(f"  Pass Rate       : {100*passed_count/total:.1f}%")

    # Breakdown by category
    categories = {}
    for r in results_full:
        cat = r["category"]
        categories.setdefault(cat, {"passed": 0, "total": 0})
        categories[cat]["total"] += 1
        if r.get("passed"):
            categories[cat]["passed"] += 1

    print(f"\n  By Category:")
    for cat, stats in categories.items():
        bar = "[OK]" if stats["passed"] == stats["total"] else "[PARTIAL]" if stats["passed"] > 0 else "[FAIL]"
        print(f"    {bar:9s} {cat:35s} {stats['passed']}/{stats['total']}")

    if output_json:
        output_path = csv_path.parent / f"eval_results_{csv_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_full, f, indent=2, ensure_ascii=False)
        print(f"\n  Full results saved to: {output_path}")

    print(f"{'='*70}\n")
    return results_full


def main():
    parser = argparse.ArgumentParser(description="Run WHO TB RAG benchmark evaluation.")
    parser.add_argument("--benchmark", type=str, default=str(DEFAULT_BENCHMARK),
                        help="Path to benchmark CSV file")
    parser.add_argument("--json", action="store_true", help="Save full results as JSON")
    args = parser.parse_args()

    csv_path = Path(args.benchmark)
    if not csv_path.exists():
        print(f"[!] Benchmark file not found: {csv_path}")
        sys.exit(1)

    run_evaluation(csv_path, output_json=args.json)


if __name__ == "__main__":
    main()
