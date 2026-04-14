"""
ARIA Eval Layer — LLM-as-judge scoring for response quality.

Scores each ARIA response on:
- Relevance: did the retrieved sources match the question?
- Faithfulness: does the response stick to what was retrieved?
"""

import os
from anthropic import Anthropic

client = Anthropic()

EVAL_SYSTEM_PROMPT = """You are a rigorous evaluator of AI-generated clinical research responses.

You will be given:
1. A user question
2. Retrieved source material (PubMed abstracts and/or clinical trial data)
3. An AI-generated response

Score the response on TWO dimensions, each from 1-5:

RELEVANCE (1-5): Did the retrieved sources actually match what the user was asking?
- 5: Sources are directly and highly relevant to the question
- 4: Sources are mostly relevant with minor gaps
- 3: Sources are partially relevant
- 2: Sources are loosely related but miss the core question
- 1: Sources are not relevant to the question

FAITHFULNESS (1-5): Does the AI response accurately reflect the retrieved sources without hallucinating?
- 5: Every claim in the response is grounded in the retrieved sources
- 4: Almost all claims are grounded, minor extrapolation
- 3: Most claims grounded, some unsupported statements
- 2: Several claims not supported by sources
- 1: Response contains significant hallucination or fabrication

Respond in this exact format:
RELEVANCE: [score]
RELEVANCE_REASON: [one sentence explanation]
FAITHFULNESS: [score]
FAITHFULNESS_REASON: [one sentence explanation]
"""


def _build_source_summary(papers: list[dict], trials: list[dict]) -> str:
    """Format retrieved sources for the evaluator."""
    parts = []

    if papers:
        parts.append("=== PUBMED PAPERS RETRIEVED ===")
        for i, p in enumerate(papers[:5], 1):
            title = p.get("title") or "Untitled"
            abstract = p.get("abstract") or "No abstract available"
            parts.append(f"{i}. {title}\n   Abstract: {abstract[:500]}...")

    if trials:
        parts.append("=== CLINICAL TRIALS RETRIEVED ===")
        for i, t in enumerate(trials[:5], 1):
            title = t.get("title") or "Untitled"
            status = t.get("status") or "N/A"
            phase = t.get("phase") or "N/A"
            conditions = t.get("conditions") or "N/A"
            parts.append(f"{i}. {title}\n   Status: {status} | Phase: {phase} | Condition: {conditions}")

    return "\n\n".join(parts) if parts else "No sources retrieved."


def _parse_scores(eval_text: str) -> dict:
    """Parse the evaluator response into structured scores."""
    scores = {
        "relevance": None,
        "relevance_reason": "",
        "faithfulness": None,
        "faithfulness_reason": "",
    }

    for line in eval_text.strip().split("\n"):
        line = line.strip()
        if line.startswith("RELEVANCE:"):
            try:
                scores["relevance"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("RELEVANCE_REASON:"):
            scores["relevance_reason"] = line.split(":", 1)[1].strip()
        elif line.startswith("FAITHFULNESS:"):
            try:
                scores["faithfulness"] = int(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith("FAITHFULNESS_REASON:"):
            scores["faithfulness_reason"] = line.split(":", 1)[1].strip()

    return scores


def evaluate_response(
    question: str,
    response: str,
    papers: list[dict],
    trials: list[dict],
) -> dict:
    """
    Run LLM-as-judge evaluation on an ARIA response.
    Returns scores and reasoning.
    """
    if not papers and not trials:
        return {"skipped": True, "reason": "No sources retrieved — eval not applicable."}

    source_summary = _build_source_summary(papers, trials)

    eval_prompt = f"""USER QUESTION:
{question}

RETRIEVED SOURCES:
{source_summary}

AI RESPONSE:
{response}

Please evaluate this response."""

    try:
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=EVAL_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": eval_prompt}],
        )
        eval_text = message.content[0].text
        scores = _parse_scores(eval_text)
        scores["skipped"] = False
        return scores

    except Exception as e:
        return {"skipped": True, "reason": f"Eval error: {e}"}