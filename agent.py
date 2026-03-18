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


def _blocks_to_text(blocks) -> str:
    if not blocks:
        return ""
    parts = []
    for b in blocks:
        block_type = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        if block_type == "text":
            text = b.get("text") if isinstance(b, dict) else getattr(b, "text", "")
            parts.append(text or "")
    return "".join(parts).strip()


def _first_tool_use(blocks):
    if not blocks:
        return None
    for b in blocks:
        block_type = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        block_name = b.get("name") if isinstance(b, dict) else getattr(b, "name", None)
        if block_type == "tool_use" and block_name == "search_pubmed":
            return b
    return None


def run_agent(user_message: str, conversation_history: list, system_prompt: str = "") -> dict:
    """
    Send a message to Claude with tool use enabled. If Claude calls the PubMed tool,
    fetch + format results, return tool_result, and get the final synthesized answer.

    Args:
        user_message: The new user message.
        conversation_history: Prior messages in Anthropic format (list of dicts).

    Returns:
        dict with keys: response (str), papers (list), used_tool (bool)
    """
    client = Anthropic()

    messages: list[dict[str, Any]] = []
    if conversation_history:
        # Expecting Anthropic message format items like:
        # {"role": "user"|"assistant", "content": "..."} OR content blocks list.
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": user_message})

    resp1 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=900,
        tools=[TOOL_SEARCH_PUBMED],
        messages=messages,
        system=system_prompt,
    )

    blocks1 = list(getattr(resp1, "content", []) or [])

    # Collect ALL tool_use blocks
    tool_uses = []
    for b in blocks1:
        block_type = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
        block_name = b.get("name") if isinstance(b, dict) else getattr(b, "name", None)
        if block_type == "tool_use" and block_name == "search_pubmed":
            tool_uses.append(b)

    if not tool_uses:
        return {"response": _blocks_to_text(blocks1), "papers": [], "used_tool": False}

    # Run PubMed search for each tool call and collect results
    all_papers = []
    tool_results = []
    for tool_use in tool_uses:
        tool_input = tool_use.get("input") if isinstance(tool_use, dict) else getattr(tool_use, "input", {})
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

        tool_use_id = tool_use.get("id") if isinstance(tool_use, dict) else getattr(tool_use, "id", None)
        tool_results.append(
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": tool_text,
            }
        )

    messages.append({"role": "assistant", "content": blocks1})
    messages.append({"role": "user", "content": tool_results})

    resp2 = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=900,
        tools=[TOOL_SEARCH_PUBMED],
        messages=messages,
        system=system_prompt,
    )

    blocks2: list[dict[str, Any]] = list(getattr(resp2, "content", []) or [])
    return {"response": _blocks_to_text(blocks2), "papers": all_papers, "used_tool": True}


__all__ = ["run_agent"]
