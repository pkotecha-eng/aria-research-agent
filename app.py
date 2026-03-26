import streamlit as st

from agent import run_agent
from prompts import ARIA_SYSTEM_PROMPT


st.set_page_config(
    page_title="ARIA — AI Research Intelligence Assistant",
    page_icon="🔬",
    layout="wide",
)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_user_message", None)


def _reset_session() -> None:
    st.session_state["messages"] = []
    st.session_state["pending_user_message"] = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🧬 ARIA")
        st.caption("AI Research Intelligence Assistant")
        st.markdown("**Powered by Claude + PubMed + ClinicalTrials.gov**")
        st.divider()

        st.markdown("**Suggested research topics**")
        suggestions = [
            "Find recruiting trials for pediatric epilepsy",
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
        "Find recruiting trials for pediatric epilepsy",
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


def _display_steps(steps: list[str]) -> None:
    """Show tool call transparency — what ARIA did step by step."""
    if not steps:
        return
    with st.expander("🔎 How ARIA researched this"):
        for step in steps:
            st.markdown(step)


def _display_papers(papers: list[dict]) -> None:
    if not papers:
        return
    with st.expander(f"📄 PubMed Sources ({len(papers)} paper(s) found)"):
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
                st.markdown(f"[PubMed ↗]({url})")
            st.divider()


def _display_trials(trials: list[dict]) -> None:
    if not trials:
        return
    with st.expander(f"🏥 Clinical Trials ({len(trials)} trial(s) found)"):
        for t in trials:
            title = t.get("title") or "Untitled"
            nct_id = t.get("nct_id") or "N/A"
            status = t.get("status") or "N/A"
            phase = t.get("phase") or "N/A"
            sponsor = t.get("sponsor") or "N/A"
            conditions = t.get("conditions") or "N/A"
            min_age = t.get("min_age") or "N/A"
            max_age = t.get("max_age") or "N/A"
            locations = "; ".join(t.get("locations") or []) or "N/A"
            url = t.get("url") or ""

            st.markdown(f"**{title}**")
            st.caption(
                f"NCT: {nct_id} | Status: {status} | Phase: {phase}\n\n"
                f"Sponsor: {sponsor} | Condition: {conditions}\n\n"
                f"Age: {min_age} – {max_age} | Locations: {locations}"
            )
            if url:
                st.markdown(f"[ClinicalTrials.gov ↗]({url})")
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
st.caption("Search and synthesize scientific literature and clinical trials")

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
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching PubMed and ClinicalTrials.gov..."):
                result = run_agent(
                    user_message=user_input,
                    conversation_history=_anthropic_history_from_ui(
                        st.session_state["messages"][:-1]
                    ),
                    system_prompt=ARIA_SYSTEM_PROMPT,
                )

            response_text = (result or {}).get("response") or ""
            used_tool = bool((result or {}).get("used_tool"))
            papers = (result or {}).get("papers") or []
            trials = (result or {}).get("trials") or []
            steps = (result or {}).get("steps") or []

            st.markdown(response_text if response_text else "I couldn't generate a response.")

            if used_tool:
                _display_steps(steps)
                _display_papers(papers)
                _display_trials(trials)

            st.session_state["messages"].append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(
                f"Sorry — something went wrong: {e}. "
                "Please try again or start a new research session."
            )
            