# frontend/app.py
import streamlit as st
import requests
import uuid
from ui_components import render_decision_card, render_tool_results, render_evidence_panel, render_guardrails_status

API_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(page_title="Loan Assistant", page_icon="🏦", layout="wide")
st.title("🏦 Enterprise Loan & Credit Assistant")

# --- Session State Initialization ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collected_inputs" not in st.session_state:
    st.session_state.collected_inputs = {}

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Applicant Info")
    # Display the structured data we've collected so far
    if st.session_state.collected_inputs:
        for key, value in st.session_state.collected_inputs.items():
            # Check explicitly for None so it shows "Pending..." correctly
            display_val = str(value) if value is not None else "Pending..."
            st.text_input(key.replace("_", " ").title(), value=display_val, disabled=True)
    else:
        st.info("No data collected yet. Start chatting!")
        
    st.divider()
    
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Upload Policy or Financial Docs", type=["txt"])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Ingesting into Vector Database..."):
                try:
                    # Send the file to FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
                    upload_res = requests.post(f"{API_URL}/upload", files=files)
                    
                    if upload_res.status_code == 200:
                        data = upload_res.json()
                        st.success(f"✅ Successfully learned '{data['filename']}' ({data['chunks_added']} chunks added!)")
                    else:
                        st.error("Upload failed.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")
        
    st.divider()
    
    # RAG Debugger (Instructor Favorite)
    st.header("🛠️ RAG Debugger")
    debug_query = st.text_input("Test Retrieval Query")
    if st.button("Run RAG Query"):
        try:
            res = requests.post(f"{API_URL}/rag/query", json={"query": debug_query, "k": 3})
            if res.status_code == 200:
                st.json(res.json())
            else:
                st.error("RAG endpoint error.")
        except requests.exceptions.ConnectionError:
            st.error("Backend is not running!")

    st.divider()
    if st.button("Clear Session (Reset)"):
        st.session_state.clear()
        st.rerun()

# --- Main Chat Interface ---
# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render visual components if this is an assistant message with metadata
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            render_guardrails_status(meta.get("guardrails"))
            render_decision_card(meta.get("decision"))
            render_tool_results(meta.get("tool_results"))
            render_evidence_panel(meta.get("rag"))
            
            if "latency_ms" in meta:
                st.caption(f"⏱️ Latency: LLM `{meta['latency_ms'].get('llm')}ms` | End-to-End `{meta['latency_ms'].get('end_to_end')}ms`")

# Chat Input Box
if prompt := st.chat_input("Ask about policies or apply for a loan..."):
    # 1. Add user message to state and UI
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Call FastAPI Backend
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={
                    "session_id": st.session_state.session_id,
                    "message": prompt,
                    "metadata": {"channel": "streamlit"}
                })
                
                if response.status_code == 200:
                    data = response.json()
                    reply = data.get("reply", "No reply generated.")
                    
                    # Update global collected inputs for the sidebar
                    st.session_state.collected_inputs = data.get("collected_inputs", {})
                    
                    # Display the text reply
                    st.markdown(reply)
                    
                    # Display the interactive components
                    render_guardrails_status(data.get("guardrails"))
                    render_decision_card(data.get("decision"))
                    render_tool_results(data.get("tool_results"))
                    render_evidence_panel(data.get("rag"))
                    
                    if "latency_ms" in data:
                        st.caption(f"⏱️ Latency: LLM `{data['latency_ms'].get('llm')}ms` | End-to-End `{data['latency_ms'].get('end_to_end')}ms`")
                    
                    # 3. Save assistant message and metadata to state
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": reply,
                        "metadata": {
                            "decision": data.get("decision"),
                            "tool_results": data.get("tool_results"),
                            "rag": data.get("rag"),
                            "guardrails": data.get("guardrails"),
                            "latency_ms": data.get("latency_ms")
                        }
                    })
                    
                    # Force a rerun to update the sidebar with new inputs
                    st.rerun()
                    
                else:
                    st.error(f"Backend Error: {response.status_code} - {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("🚨 Cannot connect to FastAPI backend. Did you run `uvicorn backend.main:app --reload`?")