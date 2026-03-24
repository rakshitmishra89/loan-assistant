# frontend/ui_components.py
import streamlit as st

def render_decision_card(decision: dict):
    if not decision or decision.get("status") == "NEED_MORE_INFO":
        return

    status = decision.get("status", "UNKNOWN")
    confidence = decision.get("confidence", 0.0)
    reasons = decision.get("reasoning", [])

    st.markdown("### 📋 Decision Summary")
    
    if status == "APPROVE":
        st.success(f"**Status: {status}** (Confidence: {confidence*100:.0f}%)")
    elif status == "REJECT":
        st.error(f"**Status: {status}** (Confidence: {confidence*100:.0f}%)")
    else:
        st.warning(f"**Status: {status}** (Confidence: {confidence*100:.0f}%)")

    if reasons:
        for r in reasons:
            st.markdown(f"- {r}")

def render_tool_results(tools: dict):
    if not tools or tools.get("emi") == 0.0:
        return
        
    st.markdown("### 🧮 Tool Results")
    
    # NEW: Display Eligibility Check
    if tools.get("is_eligible"):
        st.success("✅ **Eligibility:** Passed basic checks.")
    else:
        st.error(f"❌ **Eligibility Failed:** {', '.join(tools.get('eligibility_reasons', []))}")
        
    # Display Financial Math
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated EMI", f"${tools.get('emi', 0):.2f}")
    col2.metric("EMI Burden", f"{tools.get('emi_burden_pct', 0):.1f}%")
    col3.metric("Risk Band", tools.get("risk_band", "UNKNOWN"))

def render_evidence_panel(rag: dict):
    if not rag or not rag.get("used"):
        return

    with st.expander(f"🔍 RAG Evidence Panel (Top {rag.get('top_k', 0)} Chunks)", expanded=False):
        for idx, chunk in enumerate(rag.get("chunks", [])):
            st.markdown(f"**Source:** `{chunk.get('source')}` | **Section:** `{chunk.get('section')}` | **Score:** `{chunk.get('score'):.2f}`")
            st.info(f"\"{chunk.get('text')}\"")
            st.divider()

def render_guardrails_status(guardrails: dict):
    if not guardrails:
        return
        
    in_action = guardrails.get("input_action", "ALLOW")
    out_action = guardrails.get("output_action", "ALLOW")
    
    # Only show if something was flagged to keep UI clean, or show a subtle green check if safe
    if in_action != "ALLOW" or out_action != "ALLOW":
        st.error(f"🛡️ Guardrails Triggered: Input [{in_action}] | Output [{out_action}]")
        if guardrails.get("categories"):
            st.caption(f"Categories flagged: {', '.join(guardrails['categories'])}")
    else:
        st.caption("🛡️ Guardrails: Clean")