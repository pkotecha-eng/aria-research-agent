from __future__ import annotations
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from anthropic import Anthropic
from dotenv import load_dotenv
from pubmed_tool import format_results_for_claude, search_pubmed
from clinicaltrials_tool import format_trials_for_claude, search_clinical_trials

load_dotenv()

TOOL_SEARCH_PUBMED: dict[str, Any] = {
    "name": "search_pubmed",
    "description": (
        "Search PubMed for scientific literature on a given topic. "
        "Use this when the user asks about research, studies, drugs, diseases, "
        "clinical trials, or any scientific topic."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query for PubMed"},
            "max_results": {
                "type": "integer",
                "description": "Number of papers to return",
                "default": 3,
            },
        },
        "required": ["query"],
    },
}

TOOL_SEARCH_TRIALS: dict[str, Any] = {
    "name": "search_clinical_trials",
    "description": (
        "Search ClinicalTrials.gov for active clinical trials matching a condition. "
        "Use this when the user asks about clinical trials, ongoing studies, "
        "recruiting trials, or wants to find trials for a specific disease or condition."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "condition": {
                "type": "string",
                "description": "The disease or condition to search for",
            },
            "status": {
                "type": "string",
                "description": "Trial status: RECRUITING, COMPLETED, or ALL",
                "default": "RECRUITING",
            },
            "intervention": {
                "type": "string",
                "description": "Optional drug or intervention name to filter by",
            },
            "max_results": {
                "type": "integer",
                "description": "Number of trials to return",
                "default": 3,
            },
        },
        "required": ["condition"],
    },
}

ALL_TOOLS = [TOOL_SEARCH_PUBMED, TOOL_SEARCH_TRIALS]


def _block_to_dict(b) -> dict:
    """Convert an SDK content block to a plain dict the API accepts."""
    btype = getattr(b, "type", None)
    if btype == "text":
        return {"type": "text", "text": getattr(b, "text", "")}
    elif btype == "tool_use":
        return {
            "type": "tool_use",
            "id": getattr(b, "id", None),
            "name": getattr(b, "name", None),
            "input": getattr(b, "input", {}),
        }
    return {"type": btype}


def run_agent(user_message: str, conversation_history: list, system_prompt: str = "") -> dict:
    client = Anthropic()

    messages: list[dict[str, Any]] = list(conversation_history or [])
    messages.append({"role": "user", "content": user_message})

    # --- First call: let Claude decide which tools to use ---
    resp1 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        tools=ALL_TOOLS,
        messages=messages,
        system=system_prompt,
    )

    blocks1 = list(resp1.content or [])

    # Find all tool_use blocks
    tool_uses = [
        b for b in blocks1
        if getattr(b, "type", None) == "tool_use"
        and getattr(b, "name", None) in ("search_pubmed", "search_clinical_trials")
    ]

    # No tool call — return direct response
    if not tool_uses:
        text = "".join(
            getattr(b, "text", "") for b in blocks1
            if getattr(b, "type", None) == "text"
        ).strip()
        return {
            "response": text,
            "papers": [],
            "trials": [],
            "used_tool": False,
            "steps": [],
        }

    # --- Run tool calls ---
    all_papers = []
    all_trials = []
    tool_results = []
    steps = []  # transparency log

    for tool_use in tool_uses:
        tool_input = getattr(tool_use, "input", {})
        tool_name = getattr(tool_use, "name", None)

        try:
            if tool_name == "search_pubmed":
                query = (tool_input.get("query") or "").strip()
                max_results = int(tool_input.get("max_results", 5))
                papers = search_pubmed(query=query, max_results=max_results)
                tool_text = format_results_for_claude(papers)
                all_papers.extend(papers)
                steps.append(f"🔍 Searched PubMed for **{query}** → {len(papers)} paper(s) found")

            elif tool_name == "search_clinical_trials":
                condition = (tool_input.get("condition") or "").strip()
                status = tool_input.get("status", "RECRUITING")
                intervention = tool_input.get("intervention", "")
                max_results = int(tool_input.get("max_results", 5))
                trials = search_clinical_trials(
                    condition=condition,
                    status=status,
                    intervention=intervention,
                    max_results=max_results,
                )
                tool_text = format_trials_for_claude(trials)
                all_trials.extend(trials)
                steps.append(f"🏥 Searched ClinicalTrials.gov for **{condition}** (status: {status}) → {len(trials)} trial(s) found")

            else:
                tool_text = f"Unknown tool: {tool_name}"
                steps.append(f"⚠️ Unknown tool called: {tool_name}")

        except Exception as e:
            tool_text = f"Tool call failed: {e}"
            steps.append(f"❌ Tool call failed: {e}")

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": getattr(tool_use, "id", None),
            "content": tool_text,
        })

    steps.append("✍️ Synthesizing findings...")

    # Build messages for second call
    messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in blocks1]})
    messages.append({"role": "user", "content": tool_results})

    # Force synthesis
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": "I have the search results. Let me synthesize the findings now."}]
    })
    messages.append({
        "role": "user",
        "content": "Yes, please synthesize the research findings from those results now.",
    })

    # --- Second call: synthesize ---
    resp2 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        tools=ALL_TOOLS,
        tool_choice={"type": "none"},
        messages=messages,
        system=system_prompt,
    )

    blocks2 = list(resp2.content or [])
    text = "".join(
        getattr(b, "text", "") for b in blocks2
        if getattr(b, "type", None) == "text"
    ).strip()

    return {
        "response": text,
        "papers": all_papers,
        "trials": all_trials,
        "used_tool": True,
        "steps": steps,
    }


__all__ = ["run_agent"]
