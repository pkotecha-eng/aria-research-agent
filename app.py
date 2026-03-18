import streamlit as st

from agent import run_agent
from prompts import ARIA_SYSTEM_PROMPT


st.set_page_config(
    page_title="ARIA — AI Research Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])  # [{role: "user"|"assistant", content: str}]
    st.session_state.setdefault("pending_user_message", None)


def _reset_session() -> None:
    st.session_state["messages"] = []
    st.session_state["pending_user_message"] = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🧬 ARIA")
        st.caption("AI Research Intelligence Assistant")
        st.markdown("**Powered by Claude + PubMed**")
        st.divider()

        st.markdown("**Suggested research topics**")
        suggestions = [
            "Velarixin pediatric neurological trials",
            "eCOA patient compliance in clinical trials",
            "Adverse event reporting in Phase II trials",
            "Rare disease drug development challenges",
        ]
        for s in suggestions:
            if st.button(s, use_container_width=True):
                st.session_state["pending_user_message"] = s

        st.divider()

        if st.button("New Research Session", type="secondary", use_container_width=True):
            _reset_session()
            st.rerun()

        st.caption("For research support only. Verify findings with your clinical team.")


def _render_empty_state_suggestions() -> None:
    st.markdown("**Try one of these:**")
    chips = [
        "Velarixin pediatric neurological trials",
        "eCOA patient compliance in clinical trials",
        "Adverse event reporting in Phase II trials",
        "Rare disease drug development challenges",
    ]
    cols = st.columns(2)
    for idx, text in enumerate(chips):
        with cols[idx % 2]:
            if st.button(text, key=f"chip_{idx}", use_container_width=True):
                st.session_state["pending_user_message"] = text
                st.rerun()


def _display_sources(papers: list[dict]) -> None:
    n = len(papers)
    with st.expander(f"📄 Sources ({n} papers found)"):
        if not papers:
            st.write("No papers returned.")
            return
        for p in papers:
            title = p.get("title") or "Untitled"
            authors = "; ".join(p.get("authors") or []) or "N/A"
            journal = (p.get("journal") or "").strip()
            year = (p.get("year") or "").strip()
            journal_year = " / ".join([x for x in [journal, year] if x]) or "N/A"
            url = p.get("url") or ""

            st.markdown(f"**{title}**")
            st.caption(f"{authors}\n\n{journal_year}")
            if url:
                st.markdown(f"[PubMed]({url})")
            st.divider()


def _anthropic_history_from_ui(ui_messages: list[dict]) -> list[dict]:
    history: list[dict] = []
    for m in ui_messages:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            history.append({"role": role, "content": content})
    return history



_init_state()
_render_sidebar()

st.markdown("## 🔬 ARIA Research Assistant")
st.caption("Search and synthesize scientific literature for clinical trial intelligence")

# Render chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if not st.session_state["messages"]:
    _render_empty_state_suggestions()

user_input = st.chat_input("Ask about a drug, disease, trial design, or research topic...")
pending = st.session_state.get("pending_user_message")
if pending and not user_input:
    user_input = pending
    st.session_state["pending_user_message"] = None

if user_input:
    # 1) Show user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2) Run agent
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching PubMed and synthesizing research..."):
                result = run_agent(
                    user_message=user_input,
                    conversation_history=_anthropic_history_from_ui(st.session_state["messages"][:-1]),
                    system_prompt=ARIA_SYSTEM_PROMPT,
                )

            response_text = (result or {}).get("response") or ""
            used_tool = bool((result or {}).get("used_tool"))
            papers = (result or {}).get("papers") or []

            # 4) Display response
            st.markdown(response_text if response_text else "I couldn't generate a response.")

            # 5) Sources expander
            if used_tool:
                _display_sources(papers)

            # Add assistant response to history
            st.session_state["messages"].append({"role": "assistant", "content": response_text})
        except Exception:
            st.error(
                "Sorry—something went wrong while searching PubMed or generating the response. "
                "Please try again in a moment (or start a new research session)."
            )
