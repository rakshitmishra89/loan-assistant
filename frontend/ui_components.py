# frontend/ui_components.py
import streamlit as st
import pandas as pd

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
        
    st.markdown("### 🧮 Tool Results & Financial Breakdown")
    
    # Eligibility Check
    if tools.get("is_eligible"):
        st.success("✅ **Eligibility:** Passed basic checks.")
    else:
        st.error(f"❌ **Eligibility Failed:** {', '.join(tools.get('eligibility_reasons', []))}")
        
    # Financial Metrics (Rupee Localized)
    col1, col2, col3 = st.columns(3)
    col1.metric("Calculated EMI", f"₹{tools.get('emi', 0):.2f}")
    col2.metric("EMI Burden", f"{tools.get('emi_burden_pct', 0):.1f}%")
    col3.metric("Risk Band", tools.get("risk_band", "UNKNOWN"))

    # PHASE 3: Interactive Principal vs Interest Breakdown
    principal = tools.get("principal", 0)
    emi = tools.get("emi", 0)
    tenure = tools.get("tenure_used", 36)
    
    if principal > 0 and emi > 0:
        total_paid = emi * tenure
        total_interest = total_paid - principal
        
        st.markdown("#### 📊 Loan Repayment Structure")
        chart_data = pd.DataFrame({
            "Amount (₹)": [principal, total_interest]
        }, index=["Principal Amount", "Interest Payable"])
        
        st.bar_chart(chart_data, color="#2E86C1")

def render_evidence_panel(rag: dict):
    if not rag or not rag.get("used"):
        return

    with st.expander(f"📚 RAG Evidence Panel (Top {rag.get('top_k', 0)} Chunks)", expanded=False):
        for idx, chunk in enumerate(rag.get("chunks", [])):
            score = chunk.get('score', 0)
            confidence_color = "🟢 High Match" if score < 1.0 else "🟠 Moderate Match"
            
            st.markdown(f"**Source:** `{chunk.get('source')}` | **Section:** `{chunk.get('section')}`")
            st.caption(f"**Vector Similarity:** `{score:.4f}` ({confidence_color})")
            st.info(f"\"{chunk.get('text')}\"")
            st.divider()

def render_guardrails_status(guardrails: dict):
    if not guardrails:
        return
        
    in_action = guardrails.get("input_action", "ALLOW")
    out_action = guardrails.get("output_action", "ALLOW")
    
    if in_action != "ALLOW" or out_action != "ALLOW":
        st.error(f"🛡️ Guardrails Triggered: Input [{in_action}] | Output [{out_action}]")
        if guardrails.get("categories"):
            st.caption(f"Categories flagged: {', '.join(guardrails['categories'])}")
    else:
        st.caption("🛡️ Guardrails: Clean")

def render_agent_trace(trace: list):
    if not trace: return
    
    with st.expander("🧠 Live Agent Trace (The Glass Brain)", expanded=False):
        st.caption("Real-time autonomous workflow execution:")
        for step in trace:
            st.markdown(f"**Step {step.get('step')}: {step.get('agent')}** ➔ *{step.get('action')}*")
            with st.container():
                st.json(step.get("data", {}))
            st.divider()