"""
main.py - Entry point of "CV Sorting using LLMs" (capstone project CS02).

Pipeline (every model runs locally through Ollama, so no API key is required):
  1. Read the job description and all CVs (PDF / DOCX / TXT).
  2. LLM-1 (granite4.2:3b): job description -> requirement checklist; CV -> structured profile.
  3. Anonymise each CV (blind screening) and compute two retrieval signals:
     BM25 keyword score (classical baseline) and embedding similarity (nomic-embed-text).
  4. LLM-2 (qwen3.5:9b): score every requirement 0-5 with quoted evidence.
  5. Transparent weighted aggregation -> ranking (terminal table + CSV + JSON), optional
     evaluation against relevance labels and an interactive refinement console.

Example:  python main.py --jd ./job_description.txt --cvs ./cvs --labels ./labels.csv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from cache import JsonCache
from document_parser import list_documents, read_document, redact_pii
from evaluation import evaluate_methods, format_metrics, load_labels, save_evaluation
from extractor import Extractor
from interactive import InteractiveSession
from ollama_client import OllamaClient, OllamaError
from ranking import (Candidate, Judge, ScoringConfig, bm25_scores, export_ranking_csv, format_candidate,
                     format_ranking, format_requirements, min_max, rank_candidates, semantic_scores)


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    p = argparse.ArgumentParser(description="Rank CVs against a job description with two local LLMs (Ollama).")
    p.add_argument("--jd", required=True, help="job description file (.txt, .pdf or .docx)")
    p.add_argument("--cvs", required=True, help="folder containing the CVs (.pdf, .docx or .txt)")
    p.add_argument("--labels", help="optional CSV 'file,relevance' (0-3) used to evaluate the ranking")
    p.add_argument("--out", default="./output", help="output folder (default: ./output)")
    p.add_argument("--top-k", type=int, default=0, help="send only the K most similar CVs to LLM-2 (0 = all)")
    p.add_argument("--require", default="", help="comma-separated skills every candidate must have (hard filter)")
    p.add_argument("--interactive", action="store_true", help="open the refinement console after ranking")
    p.add_argument("--extractor-model", default="granite4.2:3b", help="LLM-1 used for structured extraction")
    p.add_argument("--judge-model", default="qwen3.5:9b", help="LLM-2 used for requirement-level judging")
    p.add_argument("--embed-model", default="nomic-embed-text", help="embedding model for semantic similarity")
    p.add_argument("--ollama-url", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                   help="Ollama server URL (default: $OLLAMA_HOST or http://127.0.0.1:11434)")
    p.add_argument("--think", action="store_true", help="enable LLM-2 'thinking' mode (about 9x slower)")
    p.add_argument("--no-redact", action="store_true", help="do not anonymise CVs before judging")
    p.add_argument("--semantic-weight", type=float, default=0.2, help="share of embedding similarity in the final score")
    p.add_argument("--must-weight", type=float, default=2.0, help="weight of must-have requirements")
    p.add_argument("--nice-weight", type=float, default=1.0, help="weight of nice-to-have requirements")
    p.add_argument("--must-threshold", type=int, default=3, help="minimum 0-5 score for a must-have to count as met")
    p.add_argument("--must-penalty", type=float, default=0.15, help="fractional score penalty per unmet must-have")
    p.add_argument("--cache-dir", default="./.cache", help="folder used to cache model results")
    p.add_argument("--no-cache", action="store_true", help="neither read nor write cached model results")
    p.add_argument("--no-pull", action="store_true", help="fail instead of downloading missing models")
    return p.parse_args(argv)


def log(message: str = "") -> None:
    """Print a progress message immediately."""
    print(message, flush=True)


def load_candidates(cv_folder: str) -> list:
    """Read every supported CV in the folder; unreadable or (almost) empty files are reported and skipped."""
    candidates = []
    for path in list_documents(cv_folder):
        name = os.path.basename(path)
        try:
            text = read_document(path)
        except Exception as exc:  # a corrupt or encrypted file must not stop the whole batch
            log(f"  [skip] {name}: cannot be read ({exc})")
            continue
        if len(text) < 80:
            log(f"  [skip] {name}: almost no extractable text (scanned image PDF?)")
            continue
        candidates.append(Candidate(file=name, text=text))
    return candidates


def describe_calls(label: str, calls: list) -> str:
    """One-line latency/throughput summary of the (non-cached) model calls of one LLM."""
    if not calls:
        return f"  {label}: all results served from cache"
    seconds = sum(c["seconds"] for c in calls)
    tokens = sum(c["output_tokens"] for c in calls)
    return (f"  {label}: {len(calls)} call(s), {seconds / len(calls):.1f} s per call, "
            f"{tokens / len(calls):.0f} output tokens per call, {tokens / max(seconds, 1e-9):.1f} tokens/s overall")


def save_outputs(args, cfg, job, ranked, extractor, judge, timings) -> None:
    """Write ranking.csv and ranking_details.json (evidence, settings, timings) to the output folder."""
    export_ranking_csv(os.path.join(args.out, "ranking.csv"), ranked, job["requirements"])
    details = {
        "job": job,
        "models": {"llm1_extractor": args.extractor_model, "llm2_judge": args.judge_model,
                   "embeddings": args.embed_model, "llm2_thinking": args.think},
        "scoring": {"semantic_weight": cfg.semantic_weight, "must_have_threshold": cfg.must_have_threshold,
                    "must_have_penalty": cfg.must_have_penalty, "must_weight": args.must_weight,
                    "nice_weight": args.nice_weight, "anonymised": not args.no_redact},
        "timings_seconds": {k: round(v, 1) for k, v in timings.items()},
        "llm_calls": {"llm1": extractor.calls, "llm2": judge.calls},
        "candidates": [{"rank": i, "file": c.file, "name": c.name, "final_score": round(c.final_score, 2),
                        "requirement_score": round(c.requirement_score, 2), "semantic_similarity": round(c.semantic, 4),
                        "bm25": round(c.bm25, 3), "unmet_must_haves": c.missing_must_haves,
                        "filtered_out": c.filtered_out, "assessments": c.assessments, "strengths": c.strengths,
                        "gaps": c.gaps, "summary": c.summary, "profile": c.profile}
                       for i, c in enumerate(ranked, 1)],
    }
    with open(os.path.join(args.out, "ranking_details.json"), "w", encoding="utf-8") as fh:
        json.dump(details, fh, indent=2, ensure_ascii=False)


def run_evaluation(args, candidates: list) -> None:
    """Compare the proposed ranking with baselines/ablations against relevance labels and save the results."""
    labels = load_labels(args.labels)
    missing = sorted(set(labels) - {c.file for c in candidates})
    if missing:
        log(f"  [warn] labelled file(s) not found among the CVs and ignored: {', '.join(missing)}")
    methods = {
        "BM25 keyword matching (baseline)": {c.file: c.bm25 for c in candidates},
        "Embedding similarity only": {c.file: c.semantic for c in candidates},
        "LLM-2 requirement score only": {c.file: c.requirement_score for c in candidates},
        "LLM-2 score + must-have penalty": {c.file: c.requirement_score * c.penalty_factor for c in candidates},
        "Hybrid final score (proposed)": {c.file: c.final_score for c in candidates},
    }
    rows, ranks = evaluate_methods(methods, labels)
    log(f"\nEVALUATION against {args.labels} ({len(labels)} labelled CVs)")
    log(format_metrics(rows))
    short = ["BM25", "Embed", "LLM", "LLM+pen", "Hybrid"]
    log(f"\nRank of every CV under each method (label 3 = strong fit ... 0 = not relevant)\n"
        f"{'File':<28}{'Label':>6}" + "".join(f"{s:>9}" for s in short))
    for f in sorted(labels, key=lambda name: (-labels[name], name)):
        if f in ranks["BM25 keyword matching (baseline)"]:
            log(f"{f:<28}{labels[f]:>6g}" + "".join(f"{ranks[m][f]:>9}" for m in methods))
    save_evaluation(os.path.join(args.out, "evaluation_metrics.csv"),
                    os.path.join(args.out, "evaluation_ranks.csv"), rows, ranks, labels)


def run(args) -> int:
    """Execute the full pipeline and return the process exit code."""
    started, timings = time.time(), {}
    cfg = ScoringConfig(args.semantic_weight, args.must_threshold, args.must_penalty)
    client = OllamaClient(args.ollama_url)
    cache = JsonCache(args.cache_dir, enabled=not args.no_cache)
    os.makedirs(args.out, exist_ok=True)

    log("=" * 80 + "\nCV SORTING USING LLMs  -  all models run locally via Ollama\n" + "=" * 80)
    log(f"LLM-1 extractor: {args.extractor_model} | LLM-2 judge: {args.judge_model} | embeddings: {args.embed_model}")
    client.ensure_models([args.extractor_model, args.judge_model, args.embed_model], auto_pull=not args.no_pull)

    stage = time.time()
    log(f"\n[1/5] Reading the job description ({args.jd}) and CVs ({args.cvs})")
    jd_text = read_document(args.jd)
    candidates = load_candidates(args.cvs)
    if not candidates:
        log("No readable CVs were found - nothing to rank.")
        return 1
    log(f"  {len(candidates)} CV(s) loaded")
    timings["1_document_parsing"] = time.time() - stage

    stage = time.time()
    log(f"\n[2/5] LLM-1 ({args.extractor_model}): requirement checklist and structured CV profiles")
    extractor = Extractor(client, args.extractor_model, cache)
    job = extractor.parse_job_description(jd_text, args.must_weight, args.nice_weight)
    log(f"  Role: {job['job_title']} (minimum experience: {job['min_years_experience']:g} years)")
    log(format_requirements(job["requirements"]))
    for i, c in enumerate(candidates, 1):
        c.profile = extractor.extract_profile(c.text)
        c.redacted_text = c.text if args.no_redact else redact_pii(c.text, c.profile.get("candidate_name", ""))
        log(f"  ({i:>2}/{len(candidates)}) {c.file:<28} -> {c.name}, {c.profile['years_experience']:g} yrs, "
            f"{len(c.profile.get('skills', []))} skills")
    client.unload(args.extractor_model)
    timings["2_llm1_extraction"] = time.time() - stage

    stage = time.time()
    log(f"\n[3/5] Retrieval signals: BM25 keyword score and {args.embed_model} cosine similarity")
    for c, score in zip(candidates, bm25_scores(jd_text, [c.text for c in candidates])):
        c.bm25 = score
    similarities = semantic_scores(client, args.embed_model, jd_text, [c.redacted_text for c in candidates], cache)
    for c, sim, norm in zip(candidates, similarities, min_max(similarities)):
        c.semantic, c.semantic_norm = sim, norm
    if 0 < args.top_k < len(candidates):
        keep = {c.file for c in sorted(candidates, key=lambda c: -c.semantic)[:args.top_k]}
        for c in candidates:
            c.shortlisted = c.file in keep
        log(f"  Shortlisted the {args.top_k} most similar CVs for LLM-2 ({len(candidates) - args.top_k} skipped)")
    timings["3_retrieval"] = time.time() - stage

    stage = time.time()
    shortlisted = [c for c in candidates if c.shortlisted]
    log(f"\n[4/5] LLM-2 ({args.judge_model}, thinking {'on' if args.think else 'off'}): "
        f"requirement-level assessment of {len(shortlisted)} CV(s)")
    judge = Judge(client, args.judge_model, cache, think=args.think)
    for i, c in enumerate(shortlisted, 1):
        t0 = time.time()
        result = judge.assess(c, job["requirements"], job["job_title"])
        c.assessments, c.strengths, c.gaps, c.summary = (result["assessments"], result["strengths"],
                                                         result["gaps"], result["summary"])
        scores = " ".join(str(c.assessments[r["id"]]["score"]) for r in job["requirements"])
        log(f"  ({i:>2}/{len(shortlisted)}) {c.file:<28} scores [{scores}]  {time.time() - t0:5.1f}s")
    client.unload(args.judge_model)
    timings["4_llm2_judging"] = time.time() - stage

    required = [s.strip() for s in args.require.split(",") if s.strip()]
    ranked = rank_candidates(candidates, job["requirements"], cfg, required)
    log(f"\n[5/5] Ranking for '{job['job_title']}'" + (f" - required skills: {', '.join(required)}" if required else ""))
    log(format_ranking(ranked, job["requirements"]))
    log("\nTop candidates - requirement-level evidence:")
    for c in [c for c in ranked if not c.filtered_out and c.shortlisted][:3]:
        log(format_candidate(c, job["requirements"]) + "\n")
    timings["total"] = time.time() - started
    save_outputs(args, cfg, job, ranked, extractor, judge, timings)
    log("Performance:")
    log(describe_calls(f"LLM-1 {args.extractor_model}", extractor.calls))
    log(describe_calls(f"LLM-2 {args.judge_model}", judge.calls))
    log("  Stage times (s): " + ", ".join(f"{k}={v:.1f}" for k, v in timings.items()))
    log(f"Saved {os.path.join(args.out, 'ranking.csv')} and {os.path.join(args.out, 'ranking_details.json')}")

    if args.labels:
        run_evaluation(args, candidates)
    if args.interactive:
        InteractiveSession(candidates, job, cfg, judge, args.out, args.must_weight, args.nice_weight).run()
    return 0


def main(argv=None) -> int:
    """Parse arguments, run the pipeline and turn expected errors into readable messages."""
    args = parse_args(argv)
    try:
        return run(args)
    except (OllamaError, FileNotFoundError, ValueError) as exc:
        print(f"\n[error] {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted - finished model results are cached and will be reused on the next run.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
