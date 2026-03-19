from __future__ import annotations
from typing import Any
from anthropic import Anthropic
from dotenv import load_dotenv
from pubmed_tool import format_results_for_claude, search_pubmed

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
                "default": 5,
            },
        },
        "required": ["query"],
    },
}


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

    # --- First call: let Claude decide to search ---
    resp1 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1500,
        tools=[TOOL_SEARCH_PUBMED],
        messages=messages,
        system=system_prompt,
    )

    blocks1 = list(resp1.content or [])

    # Find all tool_use blocks
    tool_uses = [b for b in blocks1 if getattr(b, "type", None) == "tool_use"
                 and getattr(b, "name", None) == "search_pubmed"]

    # No tool call — return direct response
    if not tool_uses:
        text = "".join(getattr(b, "text", "") for b in blocks1 
                       if getattr(b, "type", None) == "text").strip()
        return {"response": text, "papers": [], "used_tool": False}

    # --- Run PubMed searches ---
    all_papers = []
    tool_results = []
    for tool_use in tool_uses:
        tool_input = getattr(tool_use, "input", {})
        query = (tool_input.get("query") or "").strip()
        try:
            max_results = int(tool_input.get("max_results", 5))
        except (TypeError, ValueError):
            max_results = 5

        try:
            papers = search_pubmed(query=query, max_results=max_results)
            tool_text = format_results_for_claude(papers)
            all_papers.extend(papers)
        except Exception as e:
            tool_text = f"PubMed search failed: {e}"

        tool_results.append({
            "type": "tool_result",
            "tool_use_id": getattr(tool_use, "id", None),
            "content": tool_text,
        })

    # Build messages for second call using plain dicts
    messages.append({"role": "assistant", "content": [_block_to_dict(b) for b in blocks1]})
    messages.append({"role": "user", "content": tool_results})

    # Force synthesis
    messages.append({
      "role": "assistant", 
      "content": [{"type": "text", "text": "I have the search results. Let me synthesize the findings now."}]
    })
    messages.append({
      "role": "user",
      "content": "Yes, please synthesize the research findings from those papers now."
    })

    # --- Second call: synthesize results ---
    resp2 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=2000,
        tools=[TOOL_SEARCH_PUBMED],
        tool_choice={"type": "none"},
        messages=messages,
        system=system_prompt,
    )

    blocks2 = list(resp2.content or [])
    text = "".join(getattr(b, "text", "") for b in blocks2
                   if getattr(b, "type", None) == "text").strip()

    return {"response": text, "papers": all_papers, "used_tool": True}


__all__ = ["run_agent"]
